"""Consistency tests of the VQE engine on the NH3.CO2 complex (CAS(4,4), 8 qubits).

Run with  python -m pytest tests  (about one minute). The engine energy must equal Qiskit's
StatevectorEstimator energy of the Qiskit Nature UCCSD circuit, the adjoint gradient must match
finite differences, exact diagonalisation of the qubit Hamiltonian must reproduce PySCF CASCI, and
both optimisers must reach CASCI from the Hartree-Fock point.
"""
import numpy as np
import pytest

from amine_vqe import config as cfg
from amine_vqe.active_space import active_orbitals, build_rhf, casci_from_mo
from amine_vqe.geometry import read_xyz
from amine_vqe.vqe import UCCSDVQE


@pytest.fixture(scope="module")
def engine():
    atoms = read_xyz(cfg.start_geometries() / "ammonia_CO2.xyz")
    mol, mf = build_rhf(atoms, "def2-svp")
    mo, ncas, nel, _ = active_orbitals(mol, mf, atoms, "complex", "4,4")
    ecore, h1, h2, ecas, ehf, _ = casci_from_mo(mf, mo, ncas, nel)
    assert abs(ehf - mf.e_tot) < 1e-7
    return UCCSDVQE(ecore, h1, h2, nel), ecas, mf.e_tot


def test_resources(engine):
    eng, _, _ = engine
    assert (eng.nq, eng.npar) == (8, 26)


def test_against_qiskit_and_casci(engine):
    eng, ecas, ehf = engine
    v = eng.validate(ecas)
    assert v["max_dev_vs_qiskit_estimator_Ha"] < 1e-10
    assert v["grad_fd_dev"] < 1e-7
    assert abs(v["exact_minus_casci_Ha"]) < 1e-9
    assert abs(eng.hf_energy() - ehf) < 1e-9


@pytest.mark.parametrize("method", ["COBYLA", "L-BFGS-B"])
def test_optimisers_reach_casci(engine, method):
    eng, ecas, _ = engine
    r = eng.run(np.zeros(eng.npar), method, maxiter=20000, rhobeg=0.1, tol=1e-8, gtol=1e-8)
    assert (r["E"] - ecas) * cfg.HARTREE_TO_KCAL < 1e-3
