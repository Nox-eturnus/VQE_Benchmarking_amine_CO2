"""Frontier-orbital character at B3LYP/def2-TZVP (PySCF, density fitting) on the optimised structures.

The functional is PySCF's 'b3lyp5' (VWN5 local correlation), the B3LYP form used by ORCA, so the
orbital energies also cross-check the ORCA values. For every species the stage reports the HOMO and
LUMO energies (eV) and the meta-Lowdin fraction of each on the donor N and on CO2; for CO2 and the
complexes it also reports the CO2 pi* acceptor level, defined as the lowest of the first 12 virtual
orbitals with at least 60% of its population on CO2.

Output: results/frontier_orbitals_b3lyp_tzvp.json
"""
from __future__ import annotations

import json

from pyscf import dft, gto

from . import config as cfg
from .active_space import atom_populations
from .geometry import donor_index, optimised


def analyse(stem, donor):
    atoms = optimised(stem)
    mol = gto.M(atom=atoms, basis="def2-tzvp", verbose=0, max_memory=8000)
    mf = dft.RKS(mol).density_fit()
    mf.xc = "b3lyp5"
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = mol.nelectron // 2
    pop = atom_populations(mol, mf.mo_coeff[:, [nocc - 1, nocc]])
    n = len(atoms)
    co2 = [n - 3, n - 2, n - 1] if stem.endswith("_CO2") or stem == "CO2" else []
    acc = None
    if co2:
        fr = atom_populations(mol, mf.mo_coeff[:, nocc:nocc + 12])[:, co2].sum(axis=1)
        k = next(i for i in range(12) if fr[i] >= 0.6)
        acc = dict(index_above_LUMO=int(k), energy_eV=float(mf.mo_energy[nocc + k] * cfg.HARTREE_TO_EV),
                   frac_CO2=float(fr[k]))
    out = dict(species=stem, converged=bool(mf.converged), E=float(mf.e_tot))
    for k, lab in enumerate(("HOMO", "LUMO")):
        f_co2 = float(pop[k, co2].sum()) if co2 else 0.0
        f_n = float(pop[k, donor]) if donor is not None and stem != "CO2" else 0.0
        out[lab] = dict(energy_eV=float(mf.mo_energy[nocc - 1 + k] * cfg.HARTREE_TO_EV), frac_CO2=f_co2,
                        frac_donorN=f_n, frac_amine=1.0 - f_co2)
    out["gap_eV"] = out["LUMO"]["energy_eV"] - out["HOMO"]["energy_eV"]
    out["CO2_pistar"] = acc
    if acc:
        out["donor_acceptor_gap_eV"] = acc["energy_eV"] - out["HOMO"]["energy_eV"]
    return out


def run(force=False):
    path = cfg.results_dir("frontier_orbitals_b3lyp_tzvp.json")
    if path.exists() and not force:
        return json.loads(path.read_text())
    rows = [analyse("CO2", None)]
    for a in cfg.AMINES:
        dn = donor_index(optimised(f"{a}_CO2"))
        rows += [analyse(a, dn), analyse(f"{a}_CO2", dn)]
        for r in rows[-2:]:
            print(f"{r['species']:<16s} HOMO {r['HOMO']['energy_eV']:7.3f} LUMO {r['LUMO']['energy_eV']:7.3f} "
                  f"gap {r['gap_eV']:.3f} eV", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    return rows
