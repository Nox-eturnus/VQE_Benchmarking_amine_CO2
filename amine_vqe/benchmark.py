"""VQE/UCCSD benchmark on every active-space Hamiltonian in data/active_space/integrals.

Protocols per system
    multistart  11 starting points (theta = 0, the HF state, and 10 draws from U(-0.1, 0.1) with seeds 1..10),
                each with COBYLA (maxiter 20000, rhobeg 0.1, tol 1e-8) and with L-BFGS-B (analytic adjoint
                gradient, gtol 1e-8, ftol 1e-15)
    two_start   a lighter COBYLA protocol (maxiter 2000, rhobeg 0.05, SciPy default tol 1e-4) from theta = 0
                and from U(-0.02, 0.02) with seed 42; the lower energy is reported
    shots       (8-qubit complexes) COBYLA from theta = 0 with Gaussian noise of standard deviation
                Lambda / sqrt(M) added to every energy, M = 1e4, 1e5, 1e6 shots per evaluation, 5 noise seeds,
                Lambda = sum_i |c_i| sqrt(1 - <P_i>^2) at the HF state; the noiseless energy of the final
                parameters is reported
Every system is also validated (engine vs Qiskit StatevectorEstimator, adjoint vs finite-difference gradient,
exact ground state vs PySCF CASCI), and the shots needed for 1 kcal/mol are estimated at the best parameters.

Outputs: data/vqe/results/<system>.json and data/vqe/traces/<system>.npz (energy after every evaluation)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np
from scipy.optimize import minimize

from . import config as cfg
from .vqe import UCCSDVQE

CHEM_ACC_HA = 1.0 / cfg.HARTREE_TO_KCAL
SHOTS = (10_000, 100_000, 1_000_000)
NOISE_SEEDS = range(5)


def load(system, data_root=None):
    root = cfg.active_space_dir("integrals") if data_root is None else data_root
    z = np.load(root / f"{system}.npz")
    return float(z["ecore"]), z["h1"], z["h2"], int(z["nelecas"]), float(z["E_HF"]), float(z["E_CASCI"])


def _engine(system, root):
    ecore, h1, h2, nel, ehf, ecas = load(system, root)
    return UCCSDVQE(ecore, h1, h2, nel), ehf, ecas


def task_run(system, root, protocol, opt, label, x0):
    eng, ehf, ecas = _engine(system, root)
    t0 = time.time()
    if protocol == "two_start":
        r = eng.run(x0, "COBYLA", maxiter=2000, rhobeg=0.05, tol=1e-4)
    elif opt == "COBYLA":
        r = eng.run(x0, "COBYLA", maxiter=20000, rhobeg=0.1, tol=1e-8)
    else:
        r = eng.run(x0, "L-BFGS-B", maxiter=5000, gtol=1e-8)
    tr = r.pop("trace")
    r["x"] = r["x"].tolist()
    below = lambda thr: int(np.argmax(tr - ecas < thr)) + 1 if (tr - ecas < thr).any() else None
    r.update(system=system, protocol=protocol, optimizer=opt, start=label, wall_s=time.time() - t0,
             err_kcal=(r["E"] - ecas) * cfg.HARTREE_TO_KCAL, nfev_to_chem_acc=below(CHEM_ACC_HA),
             nfev_to_1e3_kcal=below(1e-3 / cfg.HARTREE_TO_KCAL))
    return r, tr


def task_shots(system, root, shots, seed):
    eng, ehf, ecas = _engine(system, root)
    lam = np.sqrt(eng.shots_for_precision(np.zeros(eng.npar), 1.0)[0])
    sigma = lam / np.sqrt(shots)
    rng = np.random.default_rng(seed)
    res = minimize(lambda x: eng.energy(x) + rng.normal(0.0, sigma), np.zeros(eng.npar),
                   method="COBYLA", options=dict(maxiter=3000, rhobeg=0.1, tol=1e-6))
    e_true = eng.energy(res.x)
    return dict(system=system, protocol="shots", shots_per_eval=int(shots), seed=int(seed), sigma_Ha=float(sigma),
                Lambda_Ha=float(lam), E_noiseless=float(e_true), err_kcal=(e_true - ecas) * cfg.HARTREE_TO_KCAL,
                corr_recovered=float((ehf - e_true) / (ehf - ecas)) if ehf != ecas else None, nfev=int(res.nfev))


def task_validate(system, root):
    eng, ehf, ecas = _engine(system, root)
    v = eng.validate(ecas)
    v.update(system=system, n_qubits=eng.nq, n_params=eng.npar, n_pauli_terms=len(eng.qubit_op),
             E_HF_qubit=eng.hf_energy(), E_HF=ehf, E_CASCI=ecas)
    return v


def _dispatch(t):
    kind, system, root = t[0], t[1], t[2]
    try:
        if kind == "validate":
            return kind, system, task_validate(system, root), None
        if kind == "shots":
            return kind, system, task_shots(system, root, *t[3:]), None
        r, tr = task_run(system, root, *t[3:])
        return "run", system, r, tr
    except Exception as exc:                              # keep the pool alive and report the failure
        return "error", system, dict(task=str(t[:5]), error=repr(exc)), None


def tasks_for(system, root):
    npar = _engine(system, root)[0].npar
    T = [("validate", system, root),
         ("run", system, root, "two_start", "COBYLA", "HF", np.zeros(npar)),
         ("run", system, root, "two_start", "COBYLA", "rand42", np.random.default_rng(42).uniform(-0.02, 0.02, npar))]
    for opt in ("COBYLA", "L-BFGS-B"):
        T.append(("run", system, root, "multistart", opt, "HF", np.zeros(npar)))
        for seed in range(1, 11):
            T.append(("run", system, root, "multistart", opt, f"rand{seed}",
                      np.random.default_rng(seed).uniform(-0.1, 0.1, npar)))
    if system.endswith("_CO2_cas4_4"):
        T += [("shots", system, root, M, seed) for M in SHOTS for seed in NOISE_SEEDS]
    return T


def run(workers=8, only=None, force=False):
    root = cfg.active_space_dir("integrals")
    res_dir, trc_dir = cfg.vqe_dir("results"), cfg.vqe_dir("traces")
    res_dir.mkdir(parents=True, exist_ok=True)
    trc_dir.mkdir(parents=True, exist_ok=True)
    systems = sorted(p.stem for p in root.glob("*.npz"))
    if only:
        systems = [s for s in systems if only in s]
    if not force:
        systems = [s for s in systems if not (res_dir / f"{s}.json").exists()]
    if not systems:
        return
    tasks = [t for s in systems for t in tasks_for(s, root)]
    # the heavy 12-qubit COBYLA runs first
    tasks.sort(key=lambda t: -(t[1].endswith("cas4_6") * 2 + (t[0] == "run" and t[4] == "COBYLA")))
    out = {s: dict(system=s, runs=[], shots=[], errors=[]) for s in systems}
    traces = {s: {} for s in systems}
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (kind, s, r, tr) in enumerate(pool.imap_unordered(_dispatch, tasks), 1):
            if kind == "validate":
                out[s]["validation"] = r
            elif kind == "shots":
                out[s]["shots"].append(r)
            elif kind == "error":
                out[s]["errors"].append(r)
                print("ERROR", s, r, flush=True)
            else:
                out[s]["runs"].append(r)
                traces[s][f"{r['protocol']}|{r['optimizer']}|{r['start']}"] = tr
            if n % 25 == 0:
                print(f"{n}/{len(tasks)} VQE tasks, {(time.time() - t0) / 60:.1f} min", flush=True)
    for s in systems:
        if out[s]["runs"]:
            best = min(out[s]["runs"], key=lambda r: r["E"])
            m, nt = _engine(s, root)[0].shots_for_precision(np.array(best["x"]), CHEM_ACC_HA)
            out[s]["measurement"] = dict(shots_1kcal=m, n_pauli_nonidentity=nt,
                                         note="no grouping, optimal per-term allocation, best parameters")
        (res_dir / f"{s}.json").write_text(json.dumps(out[s], indent=1))
        np.savez_compressed(trc_dir / f"{s}.npz", **traces[s])
