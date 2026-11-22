"""Tables (CSV), the summary file and the figures, built from the results of all stages.

Inputs : results/opt_summary.json, results/dft_energies.json, results/frontier_orbitals_b3lyp_tzvp.json,
         data/active_space/*.json, data/vqe/results/*.json, data/vqe/traces/*.npz
Outputs: results/tables/*.csv, results/summary.json, results/figures/fig*.png (300 dpi, 6.5 in wide)
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg
from .config import AMINES, HARTREE_TO_KCAL, LABELS

EH2EV = cfg.HARTREE_TO_EV
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
                     "savefig.dpi": 300, "legend.frameon": False})
LAB = [LABELS[a] for a in AMINES]
C3 = ["#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0"]
AUX_FILES = {"size_consistency", "energy_ordered_sto3g"}


def _tables():
    return cfg.results_dir("tables")


def _figures():
    return cfg.results_dir("figures")


def write_csv(name, header, rows):
    with open(_tables() / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def load_json(p):
    return json.loads(pathlib.Path(p).read_text())


# ----------------------------------------------------------------------------- data
def collect():
    S = {}
    S["opt"] = {r["species"]: r for r in load_json(cfg.results_dir("opt_summary.json"))}
    S["dft"] = load_json(cfg.results_dir("dft_energies.json"))
    S["fo"] = {r["species"]: r for r in load_json(cfg.results_dir("frontier_orbitals_b3lyp_tzvp.json"))}
    S["as"] = {p.stem: load_json(p) for p in cfg.active_space_dir().glob("*.json") if p.stem not in AUX_FILES}
    S["energy_ordered"] = {r["species"]: r for r in load_json(cfg.active_space_dir("energy_ordered_sto3g.json"))}
    S["size_consistency"] = load_json(cfg.active_space_dir("size_consistency.json"))
    S["vqe"] = {p.stem: load_json(p) for p in cfg.vqe_dir("results").glob("*.json")}
    return S


# --------------------------------------------------------------------------- tables
def table_association(S):
    M = S["dft"]["methods"]
    th = S["dft"].get("thermal_association_b3lyp_svp_kcal", {})
    rows, out = [], {}
    for a in AMINES:
        r = dict(amine=LABELS[a], r_NC=S["opt"][f"{a}_CO2"]["r_N_C_A"], OCO=S["opt"][f"{a}_CO2"]["angle_OCO_deg"])
        for m in ("b3lyp_svp", "b3lyp_tzvp", "wb97xd3_tzvp", "dlpno_tzvp", "dlpno_cbs34"):
            v = M.get(m, {}).get("association_kcal", {}).get(a, {})
            r[f"{m}_raw"] = v.get("dE_raw")
            r[f"{m}_CP"] = v.get("dE_CP")
            r[f"{m}_BSSE"] = v.get("BSSE")
            r[f"{m}_def"] = v.get("E_def")
            if m == "dlpno_tzvp":
                r["hf_tzvp_CP"] = v.get("dE_CP_HF")
                r["hf_tzvp_raw"] = v.get("dE_raw_HF")
        r.update({f"svp_{k}": v for k, v in th.get(a, {}).items()})
        out[a] = r
        rows.append([r["amine"], f"{r['r_NC']:.3f}"] + [
            "" if r.get(k) is None else f"{r[k]:.2f}" for k in (
                "b3lyp_svp_raw", "b3lyp_svp_BSSE", "b3lyp_svp_CP", "b3lyp_tzvp_raw", "b3lyp_tzvp_BSSE",
                "b3lyp_tzvp_CP", "wb97xd3_tzvp_CP", "hf_tzvp_CP", "dlpno_tzvp_raw", "dlpno_tzvp_BSSE",
                "dlpno_tzvp_CP", "dlpno_tzvp_def", "svp_dZPE", "svp_dH298", "svp_dG298")])
    write_csv("table_association_energies",
              ["Amine", "r(N...C) A", "B3LYP/SVP raw", "B3LYP/SVP BSSE", "B3LYP/SVP CP", "B3LYP/TZVP raw",
               "B3LYP/TZVP BSSE", "B3LYP/TZVP CP", "wB97X-D3/TZVP CP", "HF/TZVP CP", "DLPNO/TZVP raw",
               "DLPNO/TZVP BSSE", "DLPNO/TZVP CP", "DLPNO/TZVP deformation", "dZPE (SVP)", "dH298 (SVP)",
               "dG298 (SVP)"], rows)
    return out


def table_frontier(S):
    M = S["dft"]["methods"]
    rows, out = [], {}
    for stem in ["CO2"] + [x for a in AMINES for x in (a, f"{a}_CO2")]:
        r = dict(species=stem)
        for m in ("b3lyp_tzvp", "wb97xd3_tzvp", "dlpno_tzvp"):
            e = M.get(m, {}).get("energies", {}).get(stem, {})
            tag = "hf_tzvp" if m == "dlpno_tzvp" else m
            if "HOMO_Eh" in e:
                r[f"{tag}_HOMO"] = e["HOMO_Eh"] * EH2EV
                r[f"{tag}_LUMO"] = e["LUMO_Eh"] * EH2EV
                r[f"{tag}_gap"] = e["gap_eV"]
        fo = S["fo"].get(stem)
        if fo:
            r["HOMO_frac_N"] = fo["HOMO"]["frac_donorN"]
            r["HOMO_frac_CO2"] = fo["HOMO"]["frac_CO2"]
            r["LUMO_frac_CO2"] = fo["LUMO"]["frac_CO2"]
            r["pyscf_HOMO"] = fo["HOMO"]["energy_eV"]
            if fo.get("CO2_pistar"):
                r["pistar_eV"] = fo["CO2_pistar"]["energy_eV"]
                r["pistar_index"] = fo["CO2_pistar"]["index_above_LUMO"]
                r["da_gap"] = fo.get("donor_acceptor_gap_eV")
        out[stem] = r
    for a in AMINES:
        for tag in ("b3lyp_tzvp", "wb97xd3_tzvp", "hf_tzvp"):
            if f"{tag}_gap" in out[a] and f"{tag}_gap" in out[f"{a}_CO2"]:
                out[f"{a}_CO2"][f"{tag}_dgap_vs_amine"] = out[f"{a}_CO2"][f"{tag}_gap"] - out[a][f"{tag}_gap"]
                out[f"{a}_CO2"][f"{tag}_dHOMO_vs_amine"] = out[f"{a}_CO2"][f"{tag}_HOMO"] - out[a][f"{tag}_HOMO"]
                out[f"{a}_CO2"][f"{tag}_dLUMO_vs_CO2"] = out[f"{a}_CO2"][f"{tag}_LUMO"] - out["CO2"][f"{tag}_LUMO"]
    epi = out["CO2"]["pistar_eV"]
    for a in AMINES:
        sep = epi - out[a]["pyscf_HOMO"]
        out[a]["da_gap_separated"] = sep
        out[f"{a}_CO2"]["d_da_gap"] = out[f"{a}_CO2"]["da_gap"] - sep
        out[f"{a}_CO2"]["pistar_shift"] = out[f"{a}_CO2"]["pistar_eV"] - epi
    keys = ["b3lyp_tzvp_HOMO", "b3lyp_tzvp_LUMO", "b3lyp_tzvp_gap", "b3lyp_tzvp_dgap_vs_amine",
            "pistar_eV", "da_gap", "d_da_gap",
            "wb97xd3_tzvp_HOMO", "wb97xd3_tzvp_LUMO", "wb97xd3_tzvp_gap", "wb97xd3_tzvp_dgap_vs_amine",
            "hf_tzvp_gap", "HOMO_frac_N", "HOMO_frac_CO2", "LUMO_frac_CO2"]
    write_csv("table_homo_lumo", ["Species"] + keys,
              [[s] + ["" if r.get(k) is None else f"{r[k]:.3f}" for k in keys] for s, r in out.items()])
    return out


def table_active_space(S):
    A = S["as"]
    out = {}
    rows = []
    for stem in ["CO2"] + [x for a in AMINES for x in (a, f"{a}_CO2")]:
        if stem not in A:
            continue
        d = A[stem]
        r = dict(species=stem, E_HF=d["E_HF"])
        for sp, v in d["spaces"].items():
            r[f"corr_CASCI{sp}"] = (d["E_HF"] - v["E_CASCI"]) * HARTREE_TO_KCAL
            r[f"corr_CASSCF{sp}"] = (d["E_HF"] - v["E_CASSCF"]) * HARTREE_TO_KCAL
            r[f"orbitals{sp}"] = v["orbitals"]
        r["AVAS_cas"] = d["AVAS"]["cas"]
        r["corr_AVAS_CASCI"] = (d["E_HF"] - d["AVAS"]["E_CASCI"]) * HARTREE_TO_KCAL
        r["AVAS_CASSCF_converged"] = d["AVAS"]["CASSCF_converged"]
        # unconverged CASSCF energies are not used in tables and figures (reported in the SI with their flag)
        r["corr_AVAS_CASSCF"] = ((d["E_HF"] - d["AVAS"]["E_CASSCF"]) * HARTREE_TO_KCAL
                                 if d["AVAS"]["CASSCF_converged"] else None)
        if "CC" in d:
            r["corr_MP2"] = (d["E_HF"] - d["CC"]["E_MP2"]) * HARTREE_TO_KCAL
            r["corr_CCSD_T"] = (d["E_HF"] - d["CC"]["E_CCSD_T"]) * HARTREE_TO_KCAL
            r["T1"] = d["CC"]["T1_diagnostic"]
        out[stem] = r
    # association energies at def2-SVP (raw) for every level available, and the
    # correlation contribution relative to HF
    assoc = {}
    for a in AMINES:
        c = f"{a}_CO2"
        if not all(k in A for k in (a, c, "CO2")):
            continue
        E = lambda s, lvl: {
            "HF": A[s]["E_HF"],
            "CAS": A[s]["spaces"]["(4,6)" if s == c else "(2,3)"]["E_CASCI"],
            "CASSCF": A[s]["spaces"]["(4,6)" if s == c else "(2,3)"]["E_CASSCF"],
            "CAS44": A[s]["spaces"]["(4,4)"]["E_CASCI"] if s == c else (
                A[s]["E_HF"] if s == a else A[s]["spaces"]["(2,3)"]["E_CASCI"]),
            "AVAS": A[s]["AVAS"]["E_CASCI"],
            "MP2": A[s].get("CC", {}).get("E_MP2"),
            "CCSD(T)": A[s].get("CC", {}).get("E_CCSD_T")}[lvl]
        r = {}
        for lvl in ("HF", "CAS44", "CAS", "CASSCF", "AVAS", "MP2", "CCSD(T)"):
            vals = [E(c, lvl), E(a, lvl), E("CO2", lvl)]
            if None in vals:
                continue
            r[lvl] = (vals[0] - vals[1] - vals[2]) * HARTREE_TO_KCAL
        assoc[a] = r
    for stem, r in out.items():
        rows.append([stem] + [("" if r.get(k) is None else f"{r[k]:.3f}") for k in (
            "corr_CASCI(2,3)", "corr_CASCI(4,4)", "corr_CASCI(4,6)", "corr_CASSCF(4,4)", "corr_CASSCF(4,6)")]
            + [r["AVAS_cas"], f"{r['corr_AVAS_CASCI']:.2f}",
               "n.c." if r["corr_AVAS_CASSCF"] is None else f"{r['corr_AVAS_CASSCF']:.2f}",
               "" if "corr_MP2" not in r else f"{r['corr_MP2']:.1f}",
               "" if "corr_CCSD_T" not in r else f"{r['corr_CCSD_T']:.1f}"])
    write_csv("table_active_space_correlation",
              ["Species", "CASCI(2,3)", "CASCI(4,4)", "CASCI(4,6)", "CASSCF(4,4)", "CASSCF(4,6)", "AVAS space",
               "AVAS-CASCI", "AVAS-CASSCF", "MP2", "CCSD(T)"], rows)
    write_csv("table_active_space_association_svp", ["Amine", "HF", "CASCI(4,4)", "CASCI(4,6)", "CASSCF(4,6)",
                                                      "AVAS-CASCI", "MP2", "CCSD(T)"],
              [[LABELS[a]] + ["" if v.get(k) is None else f"{v[k]:.2f}"
                              for k in ("HF", "CAS44", "CAS", "CASSCF", "AVAS", "MP2", "CCSD(T)")]
               for a, v in assoc.items()])
    return out, assoc


def table_vqe(S):
    out = {}
    rows = []
    for sysname in sorted(S["vqe"]):
        d = S["vqe"][sysname]
        v = d["validation"]
        r = dict(system=sysname, qubits=v["n_qubits"], params=v["n_params"], pauli=v["n_pauli_terms"],
                 dev_qiskit=v["max_dev_vs_qiskit_estimator_Ha"], exact_minus_casci=v["exact_minus_casci_Ha"],
                 corr_kcal=(v["E_HF"] - v["E_CASCI"]) * HARTREE_TO_KCAL)
        for proto, opt in (("two_start", "COBYLA"), ("multistart", "COBYLA"), ("multistart", "L-BFGS-B")):
            runs = [x for x in d["runs"] if x["protocol"] == proto and x["optimizer"] == opt]
            if not runs:
                continue
            err = np.array([x["err_kcal"] for x in runs])
            nfev = np.array([x["nfev"] for x in runs])
            gn = np.array([x["grad_norm"] for x in runs])
            tag = f"{proto}_{opt}"
            r[f"{tag}_best"] = float(err.min())
            r[f"{tag}_mean"] = float(err.mean())
            r[f"{tag}_sd"] = float(err.std(ddof=1)) if len(err) > 1 else 0.0
            r[f"{tag}_max"] = float(err.max())
            r[f"{tag}_nfev_median"] = float(np.median(nfev))
            r[f"{tag}_gradnorm_max"] = float(gn.max())
            r[f"{tag}_n"] = len(runs)
            r[f"{tag}_success"] = int(sum(x["success"] for x in runs))
            hf = [x for x in runs if x["start"] == "HF"]
            if hf:
                r[f"{tag}_HFstart_err"] = hf[0]["err_kcal"]
                r[f"{tag}_HFstart_nfev"] = hf[0]["nfev"]
        if d.get("shots"):
            for M in sorted({x["shots_per_eval"] for x in d["shots"]}):
                e = np.array([x["err_kcal"] for x in d["shots"] if x["shots_per_eval"] == M])
                r[f"shots_{M}_mean"] = float(e.mean())
                r[f"shots_{M}_max"] = float(e.max())
            r["Lambda_Ha"] = d["shots"][0]["Lambda_Ha"]
        if "measurement" in d:
            r["shots_1kcal"] = d["measurement"]["shots_1kcal"]
        out[sysname] = r
        rows.append([sysname, r["qubits"], r["params"], r["pauli"], f"{r['corr_kcal']:.3f}",
                     f"{r['dev_qiskit']:.1e}", f"{r['exact_minus_casci']:.1e}"]
                    + [("" if r.get(k) is None else f"{r[k]:.2e}") for k in (
                        "two_start_COBYLA_best", "multistart_COBYLA_mean", "multistart_COBYLA_max",
                        "multistart_L-BFGS-B_mean", "multistart_L-BFGS-B_max")]
                    + [r.get("multistart_COBYLA_nfev_median"), r.get("multistart_L-BFGS-B_nfev_median"),
                       f"{max(r.get('multistart_COBYLA_gradnorm_max', 0), r.get('multistart_L-BFGS-B_gradnorm_max', 0)):.1e}"])
    write_csv("table_vqe_statistics",
              ["System", "Qubits", "UCCSD params", "Pauli terms", "HF-CASCI (kcal/mol)", "|E_engine-E_Qiskit| (Ha)",
               "E_exact-E_CASCI (Ha)", "Two-start protocol err", "COBYLA mean err", "COBYLA max err",
               "L-BFGS-B mean err", "L-BFGS-B max err", "COBYLA median nfev", "L-BFGS-B median nfev",
               "max |grad| (Ha/rad)"], rows)
    return out


# -------------------------------------------------------------------------- figures
def fig_workflow():
    """Four-stage workflow applied to every species."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axis("off")
    stages = [
        ("#1f4e79", "#e8eef6", "STAGE 1", "Geometry and\nenergetics",
         "B3LYP-D3(BJ)/def2-SVP\noptimisation + frequencies\n(all minima)\n\ndef2-TZVP single points:\nB3LYP-D3(BJ), ωB97X-D3,\nDLPNO-CCSD(T)\ncounterpoise ΔE$_{assoc}$"),
        ("#2e7d32", "#e9f4ea", "STAGE 2", "Orbitals and\nactive space",
         "HOMO, LUMO, gap\n(amines, CO$_2$, complexes)\n\nRHF/def2-SVP; projected\nn$_N$, π$_g$, π*(in), π*(out)\nCAS(2,3) / CAS(4,4) /\nCAS(4,6) → 6/8/12 qubits"),
        ("#a0521d", "#fbefe6", "STAGE 3", "VQE\nground states",
         "Jordan–Wigner, UCCSD\non HF reference\nnoiseless statevector\n\nCOBYLA and L-BFGS-B,\n11 starts each;\nshot-noise test"),
        ("#6a1b7a", "#f3e9f5", "STAGE 4", "Validation and\ncomparison",
         "exact diagonalisation,\nCASCI, CASSCF,\nAVAS-CASCI/CASSCF,\nMP2, CCSD(T)\n\nsize consistency;\nmeasurement cost"),
    ]
    w, gap = 0.23, 0.025
    for k, (dark, light, s1, s2, body) in enumerate(stages):
        x = 0.004 + k * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.08), w, 0.9, fc=light, ec=dark, lw=1.0))
        ax.add_patch(plt.Rectangle((x, 0.74), w, 0.24, fc=dark, ec=dark, lw=1.0))
        ax.text(x + w / 2, 0.93, s1, ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(x + w / 2, 0.815, s2, ha="center", va="center", color="white", fontsize=7.4, weight="bold")
        ax.text(x + w / 2, 0.41, body, ha="center", va="center", fontsize=5.9, linespacing=1.3)
        if k < 3:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.45), xytext=(x + w + 0.004, 0.45),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#303030"))
    ax.text(0.5, 0.01, "Applied identically to CO$_2$, the 10 amines and the 10 amine·CO$_2$ complexes",
            ha="center", va="bottom", fontsize=7.5, color="#404040")
    ax.set_xlim(0, 1.005)
    ax.set_ylim(0, 1)
    fig.savefig(_figures() / "fig1_workflow.png", bbox_inches="tight")
    plt.close(fig)


def fig_association(T):
    x = np.arange(len(AMINES))
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.6), sharex=True)
    w = 0.26
    for k, (m, lab) in enumerate((("b3lyp_tzvp", "B3LYP-D3(BJ)"), ("wb97xd3_tzvp", "ωB97X-D3"),
                                  ("dlpno_tzvp", "DLPNO-CCSD(T)"))):
        y = [T[a].get(f"{m}_CP") for a in AMINES]
        axs[0].bar(x + (k - 1) * w, [np.nan if v is None else v for v in y], w, color=C3[k], label=lab)
    yc = [T[a].get("dlpno_cbs34_CP") for a in AMINES]
    if any(v is not None for v in yc):
        axs[0].plot(x + w, [np.nan if v is None else v for v in yc], "kD", ms=4,
                    label="DLPNO-CCSD(T)/CBS(TZ/QZ)")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_ylabel("ΔE$_{assoc}$ (CP), kcal mol$^{-1}$")
    axs[0].set_title("(a) Counterpoise-corrected association energies, def2-TZVP", loc="left", fontsize=9)
    axs[0].set_ylim(-5.6, 0)
    axs[0].legend(ncol=2, fontsize=7.5, loc="lower right")
    for k, (m, lab) in enumerate((("b3lyp_svp", "B3LYP-D3(BJ)/def2-SVP"), ("b3lyp_tzvp", "B3LYP-D3(BJ)/def2-TZVP"),
                                  ("dlpno_tzvp", "DLPNO-CCSD(T)/def2-TZVP"))):
        y = [T[a].get(f"{m}_BSSE") for a in AMINES]
        axs[1].bar(x + (k - 1) * w, [np.nan if v is None else v for v in y], w, color=C3[k], label=lab)
    axs[1].set_ylabel("BSSE, kcal mol$^{-1}$")
    axs[1].set_title("(b) Counterpoise correction (basis-set superposition error)", loc="left", fontsize=9)
    axs[1].set_ylim(0, 4.3)
    axs[1].legend(ncol=2, fontsize=7.5, loc="upper right")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(LAB, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(_figures() / "fig2_association_energies.png", bbox_inches="tight")
    plt.close(fig)


def fig_frontier(F):
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.8), sharex=True)
    x = np.arange(len(AMINES))
    t = "b3lyp_tzvp"
    for a_i, a in enumerate(AMINES):
        for dx, s, col in ((-0.18, a, C3[0]), (0.18, f"{a}_CO2", C3[1])):
            r = F[s]
            axs[0].hlines([r[f"{t}_HOMO"], r[f"{t}_LUMO"]], a_i + dx - 0.13, a_i + dx + 0.13, color=col, lw=2)
        pi = F[f"{a}_CO2"].get("pistar_eV")
        if pi is not None:
            axs[0].hlines(pi, a_i + 0.05, a_i + 0.31, color=C3[2], lw=2)
    co2 = F["CO2"]
    axs[0].axhline(co2[f"{t}_HOMO"], color="grey", ls=":", lw=0.8)
    axs[0].axhline(co2[f"{t}_LUMO"], color="grey", ls=":", lw=0.8)
    axs[0].text(len(AMINES) - 0.5, co2[f"{t}_LUMO"] + 0.15, "CO$_2$ LUMO", ha="right", fontsize=7, color="grey")
    axs[0].text(len(AMINES) - 0.5, co2[f"{t}_HOMO"] + 0.15, "CO$_2$ HOMO", ha="right", fontsize=7, color="grey")
    axs[0].plot([], [], color=C3[0], lw=2, label="free amine")
    axs[0].plot([], [], color=C3[1], lw=2, label="amine·CO$_2$ complex")
    axs[0].plot([], [], color=C3[2], lw=2, label="CO$_2$ π* level in complex")
    axs[0].set_ylabel("Orbital energy (eV)")
    axs[0].set_title("(a) HOMO and LUMO levels, B3LYP-D3(BJ)/def2-TZVP", loc="left", fontsize=9)
    axs[0].legend(fontsize=8, loc="center left")
    w = 0.2
    series = [("b3lyp_tzvp_dgap_vs_amine", "HOMO–LUMO gap, B3LYP-D3(BJ)", C3[0]),
              ("wb97xd3_tzvp_dgap_vs_amine", "HOMO–LUMO gap, ωB97X-D3", C3[1]),
              ("hf_tzvp_dgap_vs_amine", "HOMO–LUMO gap, HF", "#808080"),
              ("d_da_gap", "n$_N$→π*(CO$_2$) gap, B3LYP", C3[2])]
    for k, (key, lab, col) in enumerate(series):
        y = [F[f"{a}_CO2"].get(key, np.nan) for a in AMINES]
        axs[1].bar(x + (k - 1.5) * w, y, w, color=col, label=lab)
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_ylabel("change on binding (eV)")
    axs[1].set_title("(b) Change on CO$_2$ binding of the HOMO–LUMO gap and of the n$_N$→π* gap", loc="left",
                     fontsize=9)
    axs[1].set_ylim(-0.4, 2.1)
    axs[1].legend(ncol=2, fontsize=7.5, loc="upper right")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(LAB, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(_figures() / "fig3_homo_lumo.png", bbox_inches="tight")
    plt.close(fig)


def fig_active_space(A, assoc, T):
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    x = np.arange(len(AMINES))
    series = [("corr_CASCI(4,4)", "CASCI(4,4)"), ("corr_CASCI(4,6)", "CASCI(4,6)"),
              ("corr_AVAS_CASCI", "AVAS-CASCI"), ("corr_AVAS_CASSCF", "AVAS-CASSCF"), ("corr_CCSD_T", "CCSD(T)")]
    w = 0.16
    for k, (key, lab) in enumerate(series):
        y = [A.get(f"{a}_CO2", {}).get(key) for a in AMINES]
        y = [np.nan if v is None else v for v in y]
        axs[0].bar(x + (k - 2) * w, y, w, color=(C3 + ["#808080"])[k], label=lab)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("E$_{HF}$ − E$_{method}$ (kcal mol$^{-1}$)")
    axs[0].set_title("(a) Correlation energy of the complexes recovered by each space, def2-SVP", loc="left",
                     fontsize=9)
    axs[0].set_ylim(4, 6e3)
    axs[0].legend(ncol=5, fontsize=7, loc="upper left")
    series = [("HF", "HF"), ("CAS", "CASCI(4,6)"), ("AVAS", "AVAS-CASCI"), ("MP2", "MP2"), ("CCSD(T)", "CCSD(T)")]
    for k, (key, lab) in enumerate(series):
        y = [assoc.get(a, {}).get(key, np.nan) for a in AMINES]
        axs[1].bar(x + (k - 2) * w, y, w, color=(["#808080"] + C3)[k], label=lab)
    y = [T[a].get("dlpno_tzvp_CP", np.nan) for a in AMINES]
    axs[1].plot(x, y, "k_", ms=14, mew=1.6, label="DLPNO-CCSD(T)/TZVP, CP")
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_ylabel("ΔE$_{assoc}$ (kcal mol$^{-1}$)")
    axs[1].set_title("(b) Association energies, def2-SVP (no CP) and DLPNO-CCSD(T)/def2-TZVP (CP)", loc="left",
                     fontsize=9)
    axs[1].set_ylim(-8.2, 0)
    axs[1].legend(ncol=3, fontsize=7.5, loc="lower left")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(LAB, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(_figures() / "fig4_active_space_validation.png", bbox_inches="tight")
    plt.close(fig)


def fig_vqe(S, V):
    systems = [f"{a}_cas2_3" for a in AMINES] + ["CO2_cas2_3"] + [f"{a}_CO2_cas4_4" for a in AMINES] + \
              [f"{a}_CO2_cas4_6" for a in AMINES]
    systems = [s for s in systems if s in S["vqe"]]
    fig = plt.figure(figsize=(6.5, 6.6))
    ax = fig.add_axes([0.1, 0.55, 0.88, 0.40])
    xs = np.arange(len(systems))
    for k, (proto, opt, mk, col, lab) in enumerate((
            ("two_start", "COBYLA", "s", "#808080", "two-start protocol (COBYLA)"),
            ("multistart", "COBYLA", "o", C3[0], "COBYLA, 11 starts"),
            ("multistart", "L-BFGS-B", "^", C3[1], "L-BFGS-B, 11 starts"))):
        for i, s in enumerate(systems):
            e = [abs(r["err_kcal"]) for r in S["vqe"][s]["runs"] if r["protocol"] == proto and r["optimizer"] == opt]
            e = np.maximum(e, 1e-12)
            ax.scatter(np.full(len(e), i + (k - 1) * 0.25), e, s=7, marker=mk, color=col,
                       label=lab if i == 0 else None, alpha=0.75, lw=0)
    ax.axhline(1.0, color="k", ls="--", lw=0.7)
    ax.text(-0.5, 1.4, "1 kcal mol$^{-1}$", ha="left", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 30)
    ax.set_ylabel("|E$_{VQE}$ − E$_{CASCI}$| (kcal mol$^{-1}$)")
    ticks = [LABELS.get(s.split("_cas")[0].replace("_CO2", ""), s) for s in systems]
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, rotation=90, fontsize=6.5)
    for b, t in ((10.5, ""), (11.5, "")):
        ax.axvline(b, color="grey", lw=0.5)
    ax.axvline(21.5, color="grey", lw=0.5)
    for xm, t in ((5, "amines CAS(2,3)"), (11, "CO$_2$"), (16.5, "complexes CAS(4,4), 8 qubits"),
                  (26.5, "complexes CAS(4,6), 12 qubits")):
        ax.text(xm, 8, t, ha="center", fontsize=7.5)
    ax.set_title("(a) Deviation of VQE/UCCSD from CASCI for every start and optimiser", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", ncol=3, markerscale=2)
    picks = ["ammonia_CO2_cas4_4", "MEA_CO2_cas4_4", "piperazine_CO2_cas4_4"]
    for k, s in enumerate(picks):
        axk = fig.add_axes([0.1 + k * 0.3, 0.07, 0.25, 0.33])
        tr = np.load(cfg.vqe_dir("traces", f"{s}.npz"))
        ecas = S["vqe"][s]["validation"]["E_CASCI"]
        for key, col, lab in (("multistart|COBYLA|HF", C3[0], "COBYLA"), ("multistart|L-BFGS-B|HF", C3[1], "L-BFGS-B")):
            if key in tr:
                y = np.maximum((tr[key] - ecas) * HARTREE_TO_KCAL, 1e-9)
                axk.plot(np.arange(1, len(y) + 1), y, color=col, lw=0.9, label=lab)
        axk.axhline(1.0, color="k", ls="--", lw=0.6)
        axk.set_xscale("log")
        axk.set_yscale("log")
        axk.set_ylim(1e-8, 20)
        axk.set_xlabel("energy evaluations")
        if k == 0:
            axk.set_ylabel("E − E$_{CASCI}$ (kcal mol$^{-1}$)")
            axk.legend(fontsize=7)
        else:
            axk.set_yticklabels([])
        axk.set_title(f"({'bcd'[k]}) {LABELS[s.split('_CO2')[0]]}·CO$_2$", loc="left", fontsize=8.5)
    fig.savefig(_figures() / "fig5_vqe_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def fig_resources(S, V, A):
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.6))
    cats = [("CAS(2,3)", "MEA_cas2_3"), ("CAS(4,4)", "MEA_CO2_cas4_4"), ("CAS(4,6)", "MEA_CO2_cas4_6")]
    q = [V[s]["qubits"] for _, s in cats]
    p = [V[s]["params"] for _, s in cats]
    t = [V[s]["pauli"] for _, s in cats]
    x = np.arange(len(cats))
    axs[0].bar(x - 0.27, q, 0.27, color=C3[0], label="qubits")
    axs[0].bar(x, p, 0.27, color=C3[1], label="UCCSD parameters")
    axs[0].bar(x + 0.27, t, 0.27, color=C3[2], label="Pauli terms")
    for xi, vals in zip(x, zip(q, p, t)):
        for dx, v in zip((-0.27, 0, 0.27), vals):
            axs[0].text(xi + dx, v * 1.15, str(v), ha="center", fontsize=7)
    axs[0].set_yscale("log")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([c for c, _ in cats])
    axs[0].set_ylim(1, 2e4)
    axs[0].set_title("(a) Quantum resources of the active spaces used", loc="left", fontsize=9)
    axs[0].legend(ncol=3, fontsize=8, loc="upper left")
    Ms = [10_000, 100_000, 1_000_000]
    for k, a in enumerate(AMINES):
        s = f"{a}_CO2_cas4_4"
        if s not in V or f"shots_{Ms[0]}_mean" not in V[s]:
            continue
        axs[1].plot(Ms, [V[s][f"shots_{M}_mean"] for M in Ms], "o-", ms=3, lw=0.8,
                    color=plt.cm.tab10(k), label=LABELS[a])
    axs[1].axhline(1.0, color="k", ls="--", lw=0.7)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("shots per energy evaluation")
    axs[1].set_ylabel("mean E − E$_{CASCI}$ (kcal mol$^{-1}$)")
    axs[1].set_title("(b) COBYLA-VQE with sampled (shot-noise) energies, CAS(4,4) complexes", loc="left", fontsize=9)
    axs[1].set_ylim(0.6, 15)
    axs[1].legend(ncol=5, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(_figures() / "fig6_resources_shot_noise.png", bbox_inches="tight")
    plt.close(fig)


def energy_ordered_summary(S):
    """Correlation energy of the energy-ordered STO-3G windows of each complex (kcal/mol)."""
    return {s: (r["E_HF"] - r["E_CASCI"]) * HARTREE_TO_KCAL for s, r in S["energy_ordered"].items()}


def run():
    _tables().mkdir(parents=True, exist_ok=True)
    _figures().mkdir(parents=True, exist_ok=True)
    S = collect()
    T = table_association(S)
    F = table_frontier(S)
    A, assoc = table_active_space(S)
    V = table_vqe(S)
    cfg.results_dir("summary.json").write_text(json.dumps(dict(
        association=T, frontier=F, active_space=A, active_space_association_svp=assoc, vqe=V,
        energy_ordered_sto3g_correlation_kcal=energy_ordered_summary(S), size_consistency=S["size_consistency"],
        tzvp_geometry_check_ammonia_CO2=S["dft"].get("tzvp_geometry_check_ammonia_CO2")), indent=1, default=float))
    fig_workflow()
    fig_association(T)
    fig_frontier(F)
    fig_active_space(A, assoc, T)
    fig_vqe(S, V)
    fig_resources(S, V, A)
    print(f"tables, figures and summary written to {cfg.results_dir()}")
