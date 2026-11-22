"""ORCA calculations: input generation, a parallel job runner, and the refinement of minima.

Optimisations (data/orca/opt/opt_<species>/)
    B3LYP-D3(BJ)/def2-SVP, RIJCOSX with def2/J, TightSCF, TightOpt, DefGrid3 and analytic
    harmonic frequencies. A structure with an imaginary mode below -5 cm-1 is refined in
    further rounds opt_<species>_r<k>: rounds 1 and 2 displace the structure by 0.25 bohr
    along the imaginary mode(s) and re-optimise with VeryTightSCF/VeryTightOpt (round 2
    with exact exchange, RIJONX, so that COSX grid noise cannot produce spurious soft modes);
    a mode that survives both is the internal rotation of the amine about the N...C axis,
    and round 3 rotates the amine by 30 deg about that axis before re-optimising.
    A slow, floppy complex can be pre-converged at the same level with PySCF/geomeTRIC
    (`preconverge`), and the next round then starts from that structure.

Single points (data/orca/sp/<method>/<job>/) on the final geometries
    b3lyp_svp, b3lyp_tzvp, wb97xd3_tzvp, dlpno_tzvp (DLPNO-CCSD(T)/def2-TZVP, TightPNO) for
    all species, and dlpno_cbs34 (def2-TZVPP/def2-QZVPP extrapolation) for NH3, methylamine,
    CO2 and their complexes. Every complex also gets the four counterpoise jobs
        __A_AB  amine with CO2 ghost atoms     __A_A  amine alone   (complex geometry)
        __B_AB  CO2 with amine ghost atoms     __B_B  CO2 alone     (complex geometry)

Basis-set check of the geometry (data/orca/opt_tzvp/)
    NH3, CO2 and NH3.CO2 re-optimised with def2-TZVP from their def2-SVP minima.
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib
import re
import shutil
import subprocess
import time

import numpy as np

from . import config as cfg
from .geometry import (final_dir, final_xyz, format_xyz, is_complex, opt_rounds, read_xyz, rotate_about_axis,
                       donor_index)

OPT_HEADER = {
    "base": "! B3LYP D3BJ def2-SVP def2/J RIJCOSX TightSCF TightOpt Freq DefGrid3\n%maxcore 2000\n",
    "refine_cosx": "! B3LYP D3BJ def2-SVP def2/J RIJCOSX VeryTightSCF VeryTightOpt Freq DefGrid3\n%maxcore 2000\n",
    "refine_onx": "! B3LYP D3BJ def2-SVP def2/J RIJONX VeryTightSCF VeryTightOpt Freq DefGrid3\n%maxcore 2000\n",
    "tzvp": "! B3LYP D3BJ def2-TZVP def2/J RIJONX VeryTightSCF VeryTightOpt Freq DefGrid3\n%maxcore 3000\n",
}
OPT_BODY = "%geom\n  MaxIter 400\nend\n* xyzfile 0 1 start.xyz\n"

SP_METHODS = {
    "b3lyp_svp": ("! B3LYP D3BJ def2-SVP def2/J RIJCOSX TightSCF DefGrid3", 2000),
    "b3lyp_tzvp": ("! B3LYP D3BJ def2-TZVP def2/J RIJCOSX TightSCF DefGrid3", 2000),
    "wb97xd3_tzvp": ("! wB97X-D3 def2-TZVP def2/J RIJCOSX TightSCF DefGrid3", 2000),
    "dlpno_tzvp": ("! DLPNO-CCSD(T) def2-TZVP def2-TZVP/C def2/J RIJCOSX TightPNO TightSCF", 3500),
    "dlpno_cbs34": ("! DLPNO-CCSD(T) Extrapolate(3/4,def2) TightPNO TightSCF RIJCOSX AutoAux", 8000),
}
CBS_SUBSET = {"CO2", "ammonia", "methylamine", "ammonia_CO2", "methylamine_CO2"}
TZVP_CHECK = ["CO2", "ammonia", "ammonia_CO2"]
KEEP_SUFFIXES = (".out", ".xyz", ".hess", ".engrad")
DISPLACEMENT_BOHR = 0.25
ROTOR_DEGREES = 30.0


# ------------------------------------------------------------------------------ output parsing
def finished(out: pathlib.Path) -> bool:
    if not out.exists():
        return False
    return "ORCA TERMINATED NORMALLY" in out.read_bytes()[-4000:].decode("utf-8", "ignore")


def frequencies(out_text: str):
    blk = out_text.split("VIBRATIONAL FREQUENCIES")
    if len(blk) < 2:
        return []
    return [float(f) for f in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", blk[-1].split("NORMAL MODES")[0], re.M)]


def is_minimum(stem) -> bool:
    """True when the latest optimisation round converged and has no imaginary mode below the cut-off."""
    d = final_dir(stem)
    out = d / f"{d.name}.out"
    if not final_xyz(stem).exists() or not finished(out):
        return False
    txt = out.read_text(errors="ignore")
    fr = frequencies(txt)
    return "THE OPTIMIZATION HAS CONVERGED" in txt and bool(fr) and min(fr) > -cfg.IMAGINARY_CUTOFF_CM1


def read_hess_modes(path):
    """Harmonic frequencies (cm-1) and Cartesian normal modes (3N x 3N) from an ORCA .hess file."""
    txt = pathlib.Path(path).read_text(errors="ignore")
    fblk = txt.split("$vibrational_frequencies")[1].split("$")[0].split("\n")
    n = int(fblk[1])
    freqs = np.array([float(l.split()[1]) for l in fblk[2:2 + n]])
    blk = []
    for l in txt.split("$normal_modes")[1].split("$")[0].strip().split("\n"):
        if l.lstrip().startswith("#"):
            break
        if l.strip():
            blk.append(l)
    nr, nc = map(int, blk[0].split())
    modes = np.zeros((nr, nc))
    i = 1
    while i < len(blk):
        cols = list(map(int, blk[i].split()))
        for r in range(nr):
            for c, v in zip(cols, blk[i + 1 + r].split()[1:]):
                modes[r, c] = float(v)
        i += 1 + nr
    return freqs, modes


# ------------------------------------------------------------------------------ input writing
def _write_opt_job(root, name, atoms, header, comment):
    d = root / name
    if (d / f"{name}.inp").exists():
        return False
    d.mkdir(parents=True, exist_ok=True)
    (d / "start.xyz").write_text(format_xyz(atoms, comment))
    (d / f"{name}.inp").write_text(header + OPT_BODY, encoding="utf-8")
    return True


def write_opt_inputs():
    n = 0
    for stem in cfg.SPECIES:
        atoms = read_xyz(cfg.start_geometries() / f"{stem}.xyz")
        n += _write_opt_job(cfg.orca_dir("opt"), f"opt_{stem}", atoms, OPT_HEADER["base"], f"{stem} start geometry")
    print(f"optimisation inputs: {n} new")


def write_tzvp_check_inputs():
    """def2-TZVP re-optimisation of NH3, CO2 and NH3.CO2, starting from their def2-SVP minima."""
    for stem in TZVP_CHECK:
        if is_minimum(stem):
            _write_opt_job(cfg.orca_dir("opt_tzvp"), f"opt_tzvp_{stem}", read_xyz(final_xyz(stem)),
                           OPT_HEADER["tzvp"], f"{stem} def2-SVP minimum")


def refine_minima():
    """Write the next refinement round for every species whose latest round has an imaginary mode."""
    written = []
    for stem in cfg.SPECIES:
        rounds = opt_rounds(stem)
        if not rounds:
            continue
        d = rounds[-1]
        out, hess = d / f"{d.name}.out", d / f"{d.name}.hess"
        if not finished(out) or not hess.exists():
            continue
        freqs, modes = read_hess_modes(hess)
        imag = [k for k, f in enumerate(freqs) if f < -cfg.IMAGINARY_CUTOFF_CM1]
        if not imag:
            continue
        atoms = read_xyz(d / f"{d.name}.xyz")
        sym = [s for s, _ in atoms]
        xyz = np.array([x for _, x in atoms])
        r = len(rounds)                                   # index of the new round
        if r <= 2:
            disp = np.zeros_like(xyz)
            for k in imag:
                m = modes[:, k].reshape(len(atoms), 3)
                disp += DISPLACEMENT_BOHR * cfg.BOHR_TO_ANGSTROM * m / np.linalg.norm(m)
            new = xyz + disp
            comment = f"{stem} displaced along {len(imag)} imaginary mode(s) {[round(float(freqs[k]), 2) for k in imag]}"
            header = OPT_HEADER["refine_cosx" if r == 1 else "refine_onx"]
        else:
            if not is_complex(stem):
                raise RuntimeError(f"{stem}: imaginary mode persists after {r - 1} refinement rounds")
            dn = donor_index(atoms)
            new = xyz.copy()
            new[:-3] = rotate_about_axis(xyz[:-3], xyz[dn], xyz[-3] - xyz[dn], ROTOR_DEGREES)
            comment = f"{stem} amine rotated {ROTOR_DEGREES:.0f} deg about the N...C axis"
            header = OPT_HEADER["refine_onx"]
        name = f"opt_{stem}_r{r}"
        if _write_opt_job(cfg.orca_dir("opt"), name, list(zip(sym, map(tuple, new))), header, comment):
            written.append(name)
            print(f"{stem}: imaginary {[round(float(freqs[k]), 2) for k in imag]} cm-1 -> {name}")
    return written


def preconverge(stem, threads=None):
    """Pre-converge a floppy structure at B3LYP-D3(BJ)/def2-SVP with PySCF + geomeTRIC, starting from the
    latest geometry of its current ORCA optimisation, and write the next round from the result."""
    from pyscf import dft, gto, lib
    from pyscf.geomopt.geometric_solver import optimize

    if threads:
        lib.num_threads(threads)
    d = final_dir(stem)
    trj = d / f"{d.name}_trj.xyz"
    if trj.exists():                                      # last frame of the running optimisation
        lines = trj.read_text().splitlines()
        n = int(lines[0])
        frame = lines[-(n + 2):]
        atoms = [(l.split()[0], tuple(map(float, l.split()[1:4]))) for l in frame[2:]]
    else:
        atoms = read_xyz(d / "start.xyz")
    mol = gto.M(atom=atoms, basis="def2-svp", verbose=3, max_memory=12000)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.disp = "d3bj"
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mol_eq = optimize(mf, maxsteps=300, convergence_energy=1e-6, convergence_grms=3e-5, convergence_gmax=1e-4,
                      convergence_drms=6e-4, convergence_dmax=1e-3)
    xyz = mol_eq.atom_coords(unit="Angstrom")
    new = [(mol_eq.atom_symbol(i), tuple(xyz[i])) for i in range(mol_eq.natm)]
    name = f"opt_{stem}_r{len(opt_rounds(stem))}"
    _write_opt_job(cfg.orca_dir("opt"), name, new, OPT_HEADER["base"],
                   f"{stem} pre-converged with PySCF/geomeTRIC B3LYP-D3(BJ)/def2-SVP")
    print(f"{stem}: pre-converged structure written to {name}")


def _write_sp_job(method, name, atoms, ghost=()):
    d = cfg.orca_dir("sp", method, name)
    if (d / f"{name}.inp").exists():
        return False
    kw, mem = SP_METHODS[method]
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(f"  {a}{':' if i in ghost else ' '} {x:>16s} {y:>16s} {z:>16s}\n"
                   for i, (a, x, y, z) in enumerate(atoms))
    (d / f"{name}.inp").write_text(f"{kw}\n%maxcore {mem}\n* xyz 0 1\n{body}*\n", encoding="utf-8")
    return True


def _xyz_tokens(path):
    lines = pathlib.Path(path).read_text().splitlines()
    return [l.split()[:4] for l in lines[2:2 + int(lines[0])]]


def write_sp_inputs(methods=tuple(SP_METHODS)):
    """Single-point and counterpoise inputs for every species whose final structure is a minimum."""
    for method in methods:
        n = 0
        for stem in cfg.SPECIES:
            if method == "dlpno_cbs34" and stem not in CBS_SUBSET:
                continue
            if not is_minimum(stem):
                continue
            atoms = _xyz_tokens(final_xyz(stem))          # coordinates copied verbatim
            n += _write_sp_job(method, stem, atoms)
            if is_complex(stem):
                na = len(atoms) - 3
                n += _write_sp_job(method, f"{stem}__A_AB", atoms, ghost=set(range(na, len(atoms))))
                n += _write_sp_job(method, f"{stem}__A_A", atoms[:na])
                n += _write_sp_job(method, f"{stem}__B_AB", atoms, ghost=set(range(na)))
                n += _write_sp_job(method, f"{stem}__B_B", atoms[na:])
        print(f"single points {method}: {n} new inputs")


# ------------------------------------------------------------------------------ running
def _run_one(job_dir: pathlib.Path) -> str:
    name = job_dir.name
    if finished(job_dir / f"{name}.out"):
        return f"skip {name}"
    work = cfg.SCRATCH / job_dir.parent.name / name
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(job_dir, work)
    t0 = time.time()
    with open(work / f"{name}.out", "w", encoding="utf-8", errors="ignore") as fh:
        subprocess.run([cfg.ORCA, f"{name}.inp"], cwd=work, stdout=fh, stderr=subprocess.STDOUT)
    for f in work.iterdir():
        if f.is_file() and f.name.endswith(KEEP_SUFFIXES):
            shutil.copy2(f, job_dir / f.name)
    ok = finished(job_dir / f"{name}.out")
    if ok:
        shutil.rmtree(work, ignore_errors=True)
    return f"{'done' if ok else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)"


def run_jobs(root: pathlib.Path, workers=4, only=None):
    """Run every job folder below root that has an input and no finished output (serial ORCA per job)."""
    if not root.exists():
        return
    jobs = sorted(d for d in root.iterdir() if d.is_dir() and (d / f"{d.name}.inp").exists()
                  and not finished(d / f"{d.name}.out"))
    if only:
        jobs = [j for j in jobs if only in j.name]
    if not jobs:
        return
    # largest structures first so the long jobs start early
    jobs.sort(key=lambda d: -sum(1 for f in [d / f"{d.name}.inp", d / "start.xyz"] if f.exists()
                                 for _ in open(f, encoding="utf-8")))
    print(f"{root.name}: {len(jobs)} ORCA jobs, {workers} workers", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(_run_one, jobs):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_all(workers=4, methods=tuple(SP_METHODS), max_refinements=3):
    """Optimise, refine until every structure is a minimum, then run all single points."""
    write_opt_inputs()
    run_jobs(cfg.orca_dir("opt"), workers)
    for _ in range(max_refinements):
        if not refine_minima():
            break
        run_jobs(cfg.orca_dir("opt"), workers)
    missing = [s for s in cfg.SPECIES if not is_minimum(s)]
    if missing:
        print("not (yet) minima:", missing)
    write_tzvp_check_inputs()
    run_jobs(cfg.orca_dir("opt_tzvp"), workers)
    write_sp_inputs(methods)
    for m in methods:
        run_jobs(cfg.orca_dir("sp", m), workers)
