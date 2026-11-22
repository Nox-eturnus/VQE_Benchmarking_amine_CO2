"""Chemically defined active spaces, their validation, and the classical reference energies.

Reference orbitals are taken from the isolated fragments:
    n_N     intrinsic bond orbital (IBO) of the amine with the largest population on the donor N
    pi_g    occupied CO2 pi orbital in the plane that contains the donor N
    pi*     the two CO2 pi* orbitals, in the plane of the donor N, pi*(in), and perpendicular, pi*(out)
    v1, v2  the two lowest canonical virtual orbitals of the amine

    CO2      CAS(2,3) = {pi_g, pi*(in), pi*(out)}
    amine    CAS(2,3) = {n_N, v1, v2}
    complex  CAS(4,4) = {n_N, pi_g, pi*(in), pi*(out)}            (8 qubits)
    complex  CAS(4,6) = {n_N, pi_g, pi*(in), pi*(out), v1, v2}    (12 qubits)

For a complex the fragment reference orbitals are computed at the complex geometry, embedded in
the AO basis of the complex and projected onto its occupied and virtual RHF spaces (SVD of the
overlap, as in AVAS). The occupied and virtual active blocks are then semicanonicalised, so the
RHF determinant is unchanged and remains the UCCSD reference. CAS(4,6) is the union of the two
fragment CAS(2,3) spaces, so its energy separates into the fragment energies at infinite
separation (size consistency). Orbital character is measured with meta-Lowdin populations.

For every species (RHF/def2-SVP on the B3LYP-D3(BJ)/def2-SVP minima) the stage computes CASCI
and CASSCF in these spaces, a larger AVAS space (donor N 2p + CO2 C/O 2p), and full-space MP2,
CCSD and CCSD(T) with frozen 1s cores. It also writes the active-space integrals used by VQE.

Outputs: data/active_space/<species>.json, data/active_space/integrals/<species>_cas<e>_<o>.npz,
         data/active_space/size_consistency.json, data/active_space/energy_ordered_sto3g.json
"""
from __future__ import annotations

import json
import time

import numpy as np
from pyscf import ao2mo, cc, gto, lo, mcscf, mp, scf
from pyscf.mcscf import avas

from . import config as cfg
from .geometry import donor_index, optimised, read_xyz

BASIS = "def2-svp"
AVAS_CASSCF_SKIP = {"TEA_CO2"}       # AVAS-CASSCF is too costly for the largest complex


# ------------------------------------------------------------------------------ building blocks
def build_rhf(atoms, basis=BASIS, conv_tol=1e-10, max_memory=6000, verbose=0):
    mol = gto.M(atom=atoms, basis=basis, verbose=verbose, max_memory=max_memory)
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    mo1, _, stable, _ = mf.stability(return_status=True)
    if not stable:
        mf.kernel(mf.make_rdm1(mo1, mf.mo_occ))
    assert mf.converged
    return mol, mf


def atom_populations(mol, C, S=None):
    """(norb, natm) meta-Lowdin population of each orbital (columns of C) on each atom."""
    S = mol.intor_symmetric("int1e_ovlp") if S is None else S
    U = lo.orth_ao(mol, "meta_lowdin")
    w = (U.T @ S @ C) ** 2
    ao_atom = np.array([s[0] for s in mol.ao_labels(fmt=False)])
    return np.stack([w[ao_atom == a].sum(axis=0) for a in range(mol.natm)], axis=1)


def lone_pair_ibo(mol, mf, donor):
    """IBO with the largest population on the donor N among the valence occupied orbitals."""
    nocc = mol.nelectron // 2
    occ = mf.mo_coeff[:, :nocc]
    val = occ[:, ~(mf.mo_energy[:nocc] < -2.0)]          # exclude the 1s cores of C, N, O
    ibo = lo.ibo.ibo(mol, val, iaos=lo.iao.iao(mol, val), verbose=0)
    pop = atom_populations(mol, ibo)
    k = int(np.argmax(pop[:, donor]))
    return ibo[:, k], float(pop[k, donor])


def co2_directions(atoms, co2=(0, 1, 2), donor_xyz=None):
    """Unit vectors (u, w, v): u along O...O, w perpendicular to u in the plane that contains the donor N
    (in-plane pi), v = u x w (out-of-plane pi). For free CO2 any perpendicular w is equivalent by symmetry."""
    c, o1, o2 = (np.array(atoms[i][1]) for i in co2)
    u = (o2 - o1) / np.linalg.norm(o2 - o1)
    if donor_xyz is None:
        trial = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    else:
        trial = c - np.array(donor_xyz)
    w = trial - (trial @ u) * u
    w /= np.linalg.norm(w)
    return u, w, np.cross(u, w)


def co2_model_orbitals(mol, co2, w, v):
    """Model CO2 pi orbitals from minimal-basis (MINAO) 2p functions, least-squares projected onto mol's basis:
        pi_g(w) = p_w(O1) - p_w(O2),  pi*(w) = p_w(C) - (p_w(O1) + p_w(O2))/2,  pi*(v) likewise along v."""
    pmol = mol.copy()
    pmol.basis = "minao"
    pmol.build(False, False)

    def p_dir(atom, d):
        idx = pmol.search_ao_label([f"{atom} {pmol.atom_symbol(atom)} 2p"])
        assert len(idx) == 3, (atom, idx)
        vec = np.zeros(pmol.nao)
        vec[idx] = d                                      # MINAO p functions are ordered x, y, z
        return vec

    c, o1, o2 = co2
    model = np.stack([p_dir(o1, w) - p_dir(o2, w),
                      p_dir(c, w) - 0.5 * (p_dir(o1, w) + p_dir(o2, w)),
                      p_dir(c, v) - 0.5 * (p_dir(o1, v) + p_dir(o2, v))], axis=1)
    S_mm = mol.intor_symmetric("int1e_ovlp")
    C = np.linalg.solve(S_mm, gto.intor_cross("int1e_ovlp", mol, pmol) @ model)
    return C / np.sqrt(np.einsum("ik,ij,jk->k", C, S_mm, C))


def amine_references(atoms, basis, donor):
    """Lone-pair IBO of the donor N and the two lowest virtuals of the amine fragment (at the given
    geometry), embedded in the AO basis of the complex (amine AOs first, CO2 AOs last)."""
    mol_a, mf_a = build_rhf(atoms[:-3], basis)
    nc = gto.M(atom=atoms[-3:], basis=basis, verbose=0).nao
    nlp, _ = lone_pair_ibo(mol_a, mf_a, donor)
    occ_a = mol_a.nelectron // 2
    ref = np.vstack([np.hstack([nlp[:, None], mf_a.mo_coeff[:, occ_a:occ_a + 2]]), np.zeros((nc, 3))])
    return ref[:, :1], ref[:, 1:]


def _project(C_space, S, refs):
    """Rotate the orthonormal space C_space so that its first k columns span the projection of the k references."""
    U, s, _ = np.linalg.svd(C_space.T @ S @ refs, full_matrices=True)
    rot = C_space @ U
    k = refs.shape[1]
    return rot[:, :k], rot[:, k:], s


def _semicanonical(C, F):
    if C.shape[1] == 0:
        return C, np.array([])
    e, v = np.linalg.eigh(C.T @ F @ C)
    return C @ v, e


def active_orbitals(mol, mf, atoms, kind, space, basis=BASIS, donor=None, donor_xyz=None):
    """MO matrix [core | active occupied | active virtual | rest] of one species.

    kind: 'amine' | 'CO2' | 'complex'; space: '2,3' | '4,4' | '4,6'. donor_xyz (CO2 only) fixes the in-plane
    direction. Returns (mo_coeff, ncas, nelecas, info); info['orbitals'] describes each reference orbital after
    projection: overlap with the reference (1 = fully contained), fractions on CO2 and on the donor N, Fock value.
    """
    S, F = mf.get_ovlp(), mf.get_fock()
    nocc = mol.nelectron // 2
    Cocc, Cvir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    if kind == "CO2":
        co2, dn = [0, 1, 2], None
        _, w, v = co2_directions(atoms, co2, donor_xyz)
        model = co2_model_orbitals(mol, co2, w, v)
        ro, rv = model[:, :1], model[:, 1:]
        labels_o, labels_v = ["pi_g"], ["pi*(in)", "pi*(out)"]
    elif kind == "amine":
        dn = [i for i, (a, _) in enumerate(atoms) if a == "N"][0] if donor is None else donor
        nlp, _ = lone_pair_ibo(mol, mf, dn)
        ro, rv = nlp[:, None], Cvir[:, :2]
        labels_o, labels_v = ["n_N"], ["v1", "v2"]
        co2 = []
    else:
        dn = donor_index(atoms)
        n = len(atoms)
        co2 = [n - 3, n - 2, n - 1]
        _, w, v = co2_directions(atoms, co2, atoms[dn][1])
        model = co2_model_orbitals(mol, co2, w, v)
        nlp, avir = amine_references(atoms, basis, dn)
        ro = np.hstack([nlp, model[:, :1]])
        rv = model[:, 1:] if space == "4,4" else np.hstack([model[:, 1:], avir])
        labels_o = ["n_N", "pi_g"]
        labels_v = ["pi*(in)", "pi*(out)"] + (["v1", "v2"] if space == "4,6" else [])

    act_o, core, _ = _project(Cocc, S, ro)
    act_v, rest, _ = _project(Cvir, S, rv)
    info = []
    for refs, Csp, labs, occ in ((ro, Cocc, labels_o, True), (rv, Cvir, labels_v, False)):
        for k, lab in enumerate(labs):
            c = Csp.T @ S @ refs[:, k]
            nrm = float(np.linalg.norm(c))
            p = Csp @ c / nrm
            pop = atom_populations(mol, p[:, None], S)[0]
            info.append(dict(label=lab, occupied=occ, overlap_with_reference=nrm,
                             frac_CO2=float(pop[co2].sum()) if co2 else 0.0,
                             frac_donorN=float(pop[dn]) if dn is not None else 0.0, fock_Ha=float(p @ F @ p)))
    act_o, eo = _semicanonical(act_o, F)
    act_v, ev = _semicanonical(act_v, F)
    mo = np.hstack([core, act_o, act_v, rest])
    return (mo, act_o.shape[1] + act_v.shape[1], 2 * act_o.shape[1],
            dict(orbitals=info, semicanonical_energies=list(map(float, np.r_[eo, ev]))))


def casci_from_mo(mf, mo, ncas, nelecas):
    """Active-space integrals (core + nuclear repulsion constant, h1, chemists' h2) and the CASCI energy, plus the
    HF determinant energy re-evaluated inside the active space as a consistency check."""
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff(mo)
    h2 = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    e_cas = mc.kernel(mo)[0]
    no = nelecas // 2
    e_hf_act = ecore + 2 * np.trace(h1[:no, :no]) + sum(
        2 * h2[i, i, j, j] - h2[i, j, j, i] for i in range(no) for j in range(no))
    return float(ecore), np.asarray(h1), np.asarray(h2), float(e_cas), float(e_hf_act), mc


def casscf_energy(mf, mo, ncas, nelecas, mol=None, co2=None, donor=None):
    """CASSCF energy, convergence flag, active natural occupations and (with mol) their CO2 / donor-N character."""
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.conv_tol = 1e-9
    mc.max_cycle_macro = 200
    e = mc.kernel(mo)[0]
    noons, natorb = mcscf.addons.make_natural_orbitals(mc)
    act = slice(mc.ncore, mc.ncore + ncas)
    occ = [float(x) for x in noons[act]]
    char = None
    if mol is not None:
        pop = atom_populations(mol, natorb[:, act])
        char = [dict(occupation=o, frac_CO2=float(pop[k, co2].sum()) if co2 else 0.0,
                     frac_donorN=float(pop[k, donor]) if donor is not None else 0.0) for k, o in enumerate(occ)]
    return float(e), bool(mc.converged), occ, char


def coupled_cluster(mol, mf):
    nocc = mol.nelectron // 2
    frozen = sum(1 for e in mf.mo_energy[:nocc] if e < -2.0)
    pt = mp.MP2(mf, frozen=frozen).run(verbose=0)
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.verbose = 0
    mycc.conv_tol = 1e-9
    mycc.max_memory = 14000
    mycc.kernel()
    et = mycc.ccsd_t()
    return dict(frozen_core=frozen, E_MP2=float(pt.e_tot), E_CCSD=float(mycc.e_tot), E_CCSD_T=float(mycc.e_tot + et),
                CCSD_converged=bool(mycc.converged),
                T1_diagnostic=float(np.sqrt((mycc.t1 ** 2).sum() / (2 * mycc.t1.shape[0]))))


# ------------------------------------------------------------------------------ per-species stage
def species_list():
    """(species, kind, donor index). The donor N of a free amine is the atom that binds CO2 in its complex."""
    out = [("CO2", "CO2", None)]
    for a in cfg.AMINES:
        dn = donor_index(optimised(f"{a}_CO2"))
        out += [(a, "amine", dn), (f"{a}_CO2", "complex", dn)]
    return out


def avas_labels(atoms, kind, donor):
    if kind == "CO2":
        return ["0 C 2p", "1 O 2p", "2 O 2p"]
    if kind == "amine":
        return [f"{donor} N 2p"]
    n = len(atoms)
    return [f"{donor} N 2p", f"{n - 3} C 2p", f"{n - 2} O 2p", f"{n - 1} O 2p"]


def analyse_species(stem, kind, donor):
    t0 = time.time()
    atoms = optimised(stem)
    mol, mf = build_rhf(atoms, BASIS, max_memory=12000)
    nocc = mol.nelectron // 2
    res = dict(species=stem, kind=kind, basis=BASIS, natoms=len(atoms), nao=mol.nao, nelectron=mol.nelectron,
               donor_N=donor, E_HF=float(mf.e_tot), HF_HOMO_Ha=float(mf.mo_energy[nocc - 1]),
               HF_LUMO_Ha=float(mf.mo_energy[nocc]), spaces={})
    co2 = [len(atoms) - 3, len(atoms) - 2, len(atoms) - 1] if kind == "complex" else ([0, 1, 2] if kind == "CO2" else [])
    integrals = cfg.active_space_dir("integrals")
    integrals.mkdir(parents=True, exist_ok=True)
    for sp in (["2,3"] if kind != "complex" else ["4,4", "4,6"]):
        mo, ncas, nel, info = active_orbitals(mol, mf, atoms, kind, sp, BASIS, donor=donor)
        ecore, h1, h2, ecas, ehf_act, _ = casci_from_mo(mf, mo, ncas, nel)
        assert abs(ehf_act - mf.e_tot) < 1e-7
        np.savez(integrals / f"{stem}_cas{nel}_{ncas}.npz", ecore=ecore, h1=h1, h2=h2, nelecas=nel,
                 E_HF=mf.e_tot, E_CASCI=ecas)
        e_scf, conv, nat, char = casscf_energy(mf, mo, ncas, nel, mol, co2, donor)
        res["spaces"][f"({nel},{ncas})"] = dict(E_CASCI=ecas, E_CASSCF=e_scf, CASSCF_converged=conv,
                                               CASSCF_natural_occupations=nat,
                                               CASSCF_natural_orbital_character=char, **info)
    labels = avas_labels(atoms, kind, donor)
    ncas, nel, mo = avas.kernel(mf, labels, threshold=0.2, minao="minao", canonicalize=True, verbose=0)
    mc = mcscf.CASCI(mf, ncas, nel)
    mc.verbose = 0
    mc.fcisolver.max_cycle = 300
    e_avas = float(mc.kernel(mo)[0])
    if stem in AVAS_CASSCF_SKIP:
        e_avas_scf, conv, nat = None, False, []
    else:
        e_avas_scf, conv, nat, _ = casscf_energy(mf, mo, ncas, nel)
    res["AVAS"] = dict(labels=labels, threshold=0.2, cas=f"({nel},{ncas})", E_CASCI=e_avas, E_CASSCF=e_avas_scf,
                       CASSCF_converged=conv, CASSCF_natural_occupations=nat, CASSCF_run=e_avas_scf is not None)
    res["CC"] = coupled_cluster(mol, mf)
    res["wall_s"] = time.time() - t0
    cfg.active_space_dir(f"{stem}.json").write_text(json.dumps(res, indent=1))
    k = cfg.HARTREE_TO_KCAL
    print(f"{stem:<16s} " + " ".join(f"CAS{s}: {(mf.e_tot - v['E_CASCI']) * k:.3f}" for s, v in res['spaces'].items())
          + f" AVAS{res['AVAS']['cas']}: {(mf.e_tot - e_avas) * k:.2f}"
          + f" CCSD(T): {(mf.e_tot - res['CC']['E_CCSD_T']) * k:.1f} kcal/mol ({res['wall_s']:.0f} s)", flush=True)


def size_consistency(separation=50.0):
    """E_CAS(4,6)[NH3...CO2 at large separation] - E_CAS(2,3)[NH3] - E_CAS(2,3)[CO2]."""
    atoms = optimised("ammonia_CO2")
    dn = donor_index(atoms)
    v = np.array(atoms[-3][1]) - np.array(atoms[dn][1])
    v /= np.linalg.norm(v)
    far = atoms[:-3] + [(a, tuple(np.array(x) + separation * v)) for a, x in atoms[-3:]]

    def e_cas(at, kind, sp, **kw):
        mol, mf = build_rhf(at, BASIS)
        mo, ncas, nel, _ = active_orbitals(mol, mf, at, kind, sp, BASIS, **kw)
        return casci_from_mo(mf, mo, ncas, nel)[3]

    e46 = e_cas(far, "complex", "4,6")
    ea = e_cas(far[:-3], "amine", "2,3", donor=dn)
    ec = e_cas(far[-3:], "CO2", "2,3", donor_xyz=far[dn][1])
    res = dict(separation_A=separation, E_CAS46_complex=e46, E_CAS23_amine=ea, E_CAS23_CO2=ec,
               deviation_kcal=(e46 - ea - ec) * cfg.HARTREE_TO_KCAL)
    cfg.active_space_dir("size_consistency.json").write_text(json.dumps(res, indent=1))
    print(f"size consistency at {separation:.0f} A: {res['deviation_kcal']:.2e} kcal/mol")
    return res


def energy_ordered_windows():
    """Energy-ordered active spaces (the default of Qiskit Nature's ActiveSpaceTransformer): STO-3G RHF on the
    start geometries, CAS(2,3) for amines and CO2 and CAS(4,4) for complexes taken symmetrically around the
    HOMO-LUMO gap, with the CO2 and donor-N (Mulliken) character of every active orbital."""
    rows = []
    for stem in cfg.SPECIES:
        cplx = stem.endswith("_CO2") and stem != "CO2"
        ncas, nelecas = (4, 4) if cplx else (3, 2)
        atoms = read_xyz(cfg.start_geometries() / f"{stem}.xyz")
        mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.kernel()
        nocc = mol.nelectron // 2
        orbs = list(range(nocc - nelecas // 2, nocc - nelecas // 2 + ncas))
        co2 = list(range(len(atoms) - 3, len(atoms))) if cplx else ([0, 1, 2] if stem == "CO2" else [])
        ns = [i for i, (a, _) in enumerate(atoms) if a == "N"]
        donor = donor_index(atoms) if cplx else (ns[0] if ns else None)
        r_nc = float(np.linalg.norm(np.array(atoms[donor][1]) - np.array(atoms[-3][1]))) if cplx else None
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        e_cas = mc.kernel()[0]
        S, C = mf.get_ovlp(), mf.mo_coeff
        ao_atom = np.array([s[0] for s in mol.ao_labels(fmt=False)])
        orbitals = []
        for i in orbs:
            pop = C[:, i] * (S @ C[:, i])
            orbitals.append(dict(mo=int(i), energy_Ha=float(mf.mo_energy[i]), occ=float(mf.mo_occ[i]),
                                 frac_CO2=float(pop[np.isin(ao_atom, co2)].sum()),
                                 frac_donorN=float(pop[ao_atom == donor].sum()) if donor is not None else None))
        rows.append(dict(species=stem, cas=f"({nelecas},{ncas})", E_HF=float(mf.e_tot), E_CASCI=float(e_cas),
                         r_N_C=r_nc, orbitals=orbitals))
    cfg.active_space_dir("energy_ordered_sto3g.json").write_text(json.dumps(rows, indent=1))
    return rows


def run(only=None, force=False):
    cfg.active_space_dir().mkdir(parents=True, exist_ok=True)
    for stem, kind, dn in species_list():
        if only and stem != only:
            continue
        if not force and cfg.active_space_dir(f"{stem}.json").exists():
            continue
        analyse_species(stem, kind, dn)
    if force or not cfg.active_space_dir("size_consistency.json").exists():
        size_consistency()
    if force or not cfg.active_space_dir("energy_ordered_sto3g.json").exists():
        energy_ordered_windows()
