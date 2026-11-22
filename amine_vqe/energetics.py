"""Optimised structures, association energies and thermal corrections from the ORCA outputs.

For every single-point method and complex:
    dE_raw  = E_AB - E_A(opt) - E_B(opt)
    BSSE    = [E_A_A - E_A_AB] + [E_B_B - E_B_AB]          (counterpoise at the complex geometry)
    dE_CP   = dE_raw + BSSE
    E_def   = [E_A_A - E_A(opt)] + [E_B_B - E_B(opt)]       (fragment deformation)
    dE_int  = dE_CP - E_def                                  (interaction of the frozen fragments)
The DLPNO-CCSD(T) jobs also provide the HF/def2-TZVP reference energies, and HOMO/LUMO energies
are the last doubly occupied and first empty canonical orbitals of each job. Thermal corrections
(ZPE, H298, G298) come from the B3LYP-D3(BJ)/def2-SVP frequencies.

Outputs: results/opt_summary.json and results/dft_energies.json
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import config as cfg
from .geometry import angle, distance, final_dir, is_complex, read_xyz
from .orca import finished, frequencies


def _last_float(pattern, txt):
    m = re.findall(pattern, txt)
    return float(m[-1]) if m else None


# ------------------------------------------------------------------------------ optimisations
def summarise_opt(stem):
    d = final_dir(stem)
    out = (d / f"{d.name}.out").read_text(errors="ignore")
    res = dict(species=stem, job=d.name, terminated_normally="ORCA TERMINATED NORMALLY" in out,
               converged="THE OPTIMIZATION HAS CONVERGED" in out,
               cycles=len(re.findall(r"GEOMETRY OPTIMIZATION CYCLE", out)),
               E_final_Eh=_last_float(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out))
    crit = {}
    for key in ("Energy change", "RMS gradient", "MAX gradient", "RMS step", "MAX step"):
        m = re.findall(rf"{key}\s+(-?\d\.\d+E?[-+]?\d*)\s+(\d\.\d+E?[-+]?\d*)", out)
        if m:
            crit[key] = dict(value=float(m[-1][0]), tolerance=float(m[-1][1]))
    res["final_criteria"] = crit
    nz = [f for f in frequencies(out) if abs(f) > 1e-3]
    res["n_imaginary"] = sum(1 for f in nz if f < 0)
    res["n_imaginary_above_5cm1"] = sum(1 for f in nz if f < -cfg.IMAGINARY_CUTOFF_CM1)
    res["lowest_freqs_cm1"] = sorted(nz)[:3]
    res["ZPE_Eh"] = _last_float(r"Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh", out)
    res["G_298_Eh"] = _last_float(r"Final Gibbs free energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh", out)
    res["H_298_Eh"] = _last_float(r"Total Enthalpy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh", out)
    xyz = d / f"{d.name}.xyz"
    if xyz.exists() and is_complex(stem):
        at = read_xyz(xyz)
        c, o1, o2 = (at[i][1] for i in (-3, -2, -1))
        res["r_N_C_A"] = min(distance(x, c) for a, x in at if a == "N")
        res["angle_OCO_deg"] = angle(o1, c, o2)
    return res


def opt_summary():
    rows = []
    for stem in cfg.SPECIES:
        d = final_dir(stem)
        if not finished(d / f"{d.name}.out"):
            print(f"{stem:<18s} optimisation not finished")
            continue
        r = summarise_opt(stem)
        rows.append(r)
        extra = f" r(N..C)={r['r_N_C_A']:.3f} OCO={r['angle_OCO_deg']:.1f}" if "r_N_C_A" in r else ""
        print(f"{r['job']:<22s} converged={r['converged']} cycles={r['cycles']:3d} "
              f"n_imag={r['n_imaginary']} lowest={r['lowest_freqs_cm1']}{extra}")
    cfg.results_dir().mkdir(parents=True, exist_ok=True)
    cfg.results_dir("opt_summary.json").write_text(json.dumps(rows, indent=1))
    return rows


# ------------------------------------------------------------------------------ single points
def parse_sp(path):
    txt = path.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in txt[-3000:]:
        return None
    r = {"E": _last_float(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)}
    if re.search(r"E\(CCSD\)\s+\.\.\.", txt):
        r["E_CCSD"] = _last_float(r"E\(CCSD\)\s+\.\.\.\s+(-?\d+\.\d+)", txt)
        r["E_HF"] = _last_float(r"Total Energy\s+:\s+(-?\d+\.\d+) Eh", txt)
        r["T1"] = _last_float(r"T1 diagnostic\s+\.\.\.\s+(-?\d+\.\d+)", txt)
    e_disp = _last_float(r"Dispersion correction\s+(-?\d+\.\d+)", txt)
    if e_disp is not None:
        r["E_disp"] = e_disp
    blk = txt.split("ORBITAL ENERGIES")[-1].split("*******")[0] if "ORBITAL ENERGIES" in txt else ""
    occ = [(float(m.group(2)), float(m.group(3)))
           for m in re.finditer(r"^\s*(\d+)\s+([0-2]\.0000)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", blk, re.M)]
    if occ:
        homo = max(e for o, e in occ if o > 1.5)
        lumo = min(e for o, e in occ if o < 0.5)
        r.update(HOMO_Eh=homo, LUMO_Eh=lumo, gap_eV=(lumo - homo) * cfg.HARTREE_TO_EV)
    return r


def _association(E):
    assoc = {}
    for a in cfg.AMINES:
        c = f"{a}_CO2"
        need = [c, a, "CO2", f"{c}__A_AB", f"{c}__A_A", f"{c}__B_AB", f"{c}__B_B"]
        if not all(k in E for k in need):
            continue
        keys = ["E"] + (["E_HF"] if "E_HF" in E[c] and "E_CCSD" in E[c] else [])
        for key in keys:
            tag = "" if key == "E" else "_HF"
            v = {k: E[k][key] for k in need}
            raw = v[c] - v[a] - v["CO2"]
            bsse = (v[f"{c}__A_A"] - v[f"{c}__A_AB"]) + (v[f"{c}__B_B"] - v[f"{c}__B_AB"])
            edef = (v[f"{c}__A_A"] - v[a]) + (v[f"{c}__B_B"] - v["CO2"])
            k = cfg.HARTREE_TO_KCAL
            assoc.setdefault(a, {}).update({
                f"dE_raw{tag}": raw * k, f"BSSE{tag}": bsse * k, f"dE_CP{tag}": (raw + bsse) * k,
                f"E_def{tag}": edef * k, f"dE_int_CP{tag}": (raw + bsse - edef) * k})
    return assoc


def _thermo(stem):
    d = final_dir(stem)
    txt = (d / f"{d.name}.out").read_text(errors="ignore")
    return dict(E_el=_last_float(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt),
                ZPE=_last_float(r"Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)", txt),
                H=_last_float(r"Total Enthalpy\s+\.\.\.\s+(-?\d+\.\d+)", txt),
                G=_last_float(r"Final Gibbs free energy\s+\.\.\.\s+(-?\d+\.\d+)", txt))


def _tzvp_geometry_check():
    """N...C distance and raw B3LYP-D3(BJ)/def2-TZVP association energy of NH3.CO2 at its def2-TZVP minimum."""
    root = cfg.orca_dir("opt_tzvp")
    jobs = {s: root / f"opt_tzvp_{s}" for s in ("CO2", "ammonia", "ammonia_CO2")}
    if not all(finished(d / f"{d.name}.out") for d in jobs.values()):
        return None
    E = {s: _last_float(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", (d / f"{d.name}.out").read_text(errors="ignore"))
         for s, d in jobs.items()}
    at = read_xyz(jobs["ammonia_CO2"] / "opt_tzvp_ammonia_CO2.xyz")
    c = at[-3][1]
    return dict(r_N_C_A=min(distance(x, c) for a, x in at if a == "N"),
                dE_raw_kcal=(E["ammonia_CO2"] - E["ammonia"] - E["CO2"]) * cfg.HARTREE_TO_KCAL)


def association_energies():
    res = {"methods": {}, "thermo_b3lyp_svp": {}}
    sp = cfg.orca_dir("sp")
    for mdir in sorted(p for p in sp.iterdir() if p.is_dir()) if sp.exists() else []:
        E = {}
        for job in sorted(mdir.iterdir()):
            o = job / f"{job.name}.out"
            if o.exists():
                r = parse_sp(o)
                if r:
                    E[job.name] = r
        assoc = _association(E)
        res["methods"][mdir.name] = dict(energies=E, association_kcal=assoc)
        print(f"{mdir.name}: {len(E)} jobs parsed, {len(assoc)} complete association sets")
    for stem in cfg.SPECIES:
        d = final_dir(stem)
        if finished(d / f"{d.name}.out"):
            res["thermo_b3lyp_svp"][stem] = _thermo(stem)
    th = res["thermo_b3lyp_svp"]
    res["thermal_association_b3lyp_svp_kcal"] = {}
    for a in cfg.AMINES:
        c = f"{a}_CO2"
        if all(k in th and th[k]["G"] is not None for k in (a, c, "CO2")):
            d = lambda q: (th[c][q] - th[a][q] - th["CO2"][q]) * cfg.HARTREE_TO_KCAL
            res["thermal_association_b3lyp_svp_kcal"][a] = dict(dE=d("E_el"), dZPE=d("ZPE"), dH298=d("H"),
                                                               dG298=d("G"))
    res["tzvp_geometry_check_ammonia_CO2"] = _tzvp_geometry_check()
    cfg.results_dir().mkdir(parents=True, exist_ok=True)
    cfg.results_dir("dft_energies.json").write_text(json.dumps(res, indent=1))
    return res


def export_geometries():
    """Final structures as XYZ files (one per species and one multi-structure file)."""
    out = cfg.optimised_geometries()
    out.mkdir(parents=True, exist_ok=True)
    blocks = []
    for stem in cfg.SPECIES:
        d = final_dir(stem)
        src = d / f"{d.name}.xyz"
        if not src.exists():
            continue
        lines = src.read_text().splitlines()
        n = int(lines[0])
        at = np.array([[float(v) for v in l.split()[1:4]] for l in lines[2:2 + n]])
        sym = [l.split()[0] for l in lines[2:2 + n]]
        txt = f"{n}\n{stem}  B3LYP-D3(BJ)/def2-SVP minimum (ORCA job {d.name}), Angstrom\n" + "".join(
            f"{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n" for s, (x, y, z) in zip(sym, at))
        (out / f"{stem}.xyz").write_text(txt)
        blocks.append(txt)
    (out / "all_structures.xyz").write_text("".join(blocks))
