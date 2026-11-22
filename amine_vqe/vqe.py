"""UCCSD-VQE engine for an active-space Hamiltonian (Qiskit Nature operators, exact statevector).

The qubit Hamiltonian, the Hartree-Fock reference circuit and the UCCSD excitation
generators are those of Qiskit Nature 0.7.2 (Jordan-Wigner mapping). The ansatz state

    |psi(theta)> = prod_k exp(-i theta_k O_k) |HF>,   O_k = i (T_k - T_k^dagger)

is the state prepared by Qiskit's UCCSD circuit (one Trotter step, operators applied
in Qiskit's order). Because A_k = -i O_k = T_k - T_k^dagger is real with A_k^3 = -A_k,
each factor is applied exactly as  I + sin(theta) A + (1 - cos(theta)) A^2 , and the
energy gradient is obtained with one adjoint (reverse) sweep. `validate()` checks the
energy against Qiskit's StatevectorEstimator on the UCCSD circuit at random parameters
and the exact ground state against the PySCF CASCI energy.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper

from .config import HARTREE_TO_KCAL  # noqa: F401  (re-exported for convenience)


class UCCSDVQE:
    def __init__(self, ecore, h1, h2, nelecas):
        self.ecore = float(ecore)
        self.norb = h1.shape[0]
        self.nel = (nelecas // 2, nelecas // 2)
        self.mapper = JordanWignerMapper()
        ham = ElectronicEnergy.from_raw_integrals(h1, h2)       # h2 in chemists' notation
        self.qubit_op = self.mapper.map(ham.second_q_op())
        self.nq = self.qubit_op.num_qubits
        H = self.qubit_op.to_matrix(sparse=True)
        assert abs(H.imag).max() < 1e-12
        self.H = sp.csr_matrix(H.real)
        self.hf_circuit = HartreeFock(self.norb, self.nel, self.mapper)
        self.psi0 = Statevector(self.hf_circuit).data.real.copy()
        self.ansatz = UCCSD(self.norb, self.nel, self.mapper, initial_state=self.hf_circuit)
        self.A, self.A2 = [], []
        for op in self.ansatz.operators:
            M = (-1j * op.to_matrix(sparse=True)).tocsr()
            assert abs(M.imag).max() < 1e-12
            A = sp.csr_matrix(M.real)
            self.A.append(A)
            self.A2.append((A @ A).tocsr())
        self.npar = len(self.A)
        assert self.npar == self.ansatz.num_parameters
        self.nfev = 0

    # ---------------------------------------------------------------- energies
    def state(self, theta):
        psi = self.psi0.copy()
        for t, A, A2 in zip(theta, self.A, self.A2):
            psi = psi + np.sin(t) * (A @ psi) + (1.0 - np.cos(t)) * (A2 @ psi)
        return psi

    def energy(self, theta):
        """Total energy (Hartree) including core and nuclear repulsion."""
        self.nfev += 1
        psi = self.state(theta)
        return float(psi @ (self.H @ psi)) + self.ecore

    def energy_and_grad(self, theta):
        self.nfev += 1
        psi = self.state(theta)
        lam = self.H @ psi
        e = float(psi @ lam) + self.ecore
        g = np.empty(self.npar)
        for k in range(self.npar - 1, -1, -1):
            A, A2, t = self.A[k], self.A2[k], theta[k]
            g[k] = 2.0 * float(lam @ (A @ psi))
            # apply U_k^{-1} = I - sin(t) A + (1 - cos(t)) A^2 to both vectors
            psi = psi - np.sin(t) * (A @ psi) + (1.0 - np.cos(t)) * (A2 @ psi)
            lam = lam - np.sin(t) * (A @ lam) + (1.0 - np.cos(t)) * (A2 @ lam)
        return e, g

    def hf_energy(self):
        return float(self.psi0 @ (self.H @ self.psi0)) + self.ecore

    def exact_energy(self):
        """Lowest eigenvalue of the qubit Hamiltonian in the correct particle-number sector."""
        n = 2 ** self.nq
        # restrict to determinants with n_alpha, n_beta electrons (JW: alpha = low qubits)
        idx = [i for i in range(n)
               if bin(i & ((1 << self.norb) - 1)).count("1") == self.nel[0]
               and bin(i >> self.norb).count("1") == self.nel[1]]
        Hs = self.H[idx][:, idx].toarray()
        return float(np.linalg.eigvalsh(Hs)[0]) + self.ecore

    def exact_energy_full(self):
        """Lowest eigenvalue over the full Fock space (all particle numbers)."""
        if self.nq <= 10:
            return float(np.linalg.eigvalsh(self.H.toarray())[0]) + self.ecore
        return float(eigsh(self.H, k=1, which="SA")[0][0]) + self.ecore

    # -------------------------------------------------------------- validation
    def validate(self, e_casci, rng_seed=7, ntest=3):
        from qiskit.primitives import StatevectorEstimator
        est = StatevectorEstimator()
        rng = np.random.default_rng(rng_seed)
        devs = []
        for _ in range(ntest):
            th = rng.uniform(-0.3, 0.3, self.npar)
            bound = self.ansatz.assign_parameters(th)
            e_q = float(est.run([(bound, self.qubit_op)]).result()[0].data.evs) + self.ecore
            devs.append(abs(e_q - self.energy(th)))
        # finite-difference check of the adjoint gradient
        th = rng.uniform(-0.3, 0.3, self.npar)
        _, g = self.energy_and_grad(th)
        k = int(rng.integers(self.npar))
        d = np.zeros(self.npar)
        d[k] = 1e-6
        fd = (self.energy(th + d) - self.energy(th - d)) / 2e-6
        e_exact = self.exact_energy()
        self.nfev = 0
        return dict(max_dev_vs_qiskit_estimator_Ha=max(devs), grad_fd_dev=abs(fd - g[k]),
                    exact_minus_casci_Ha=e_exact - e_casci, E_exact=e_exact,
                    E_exact_fullFock=self.exact_energy_full())

    # ------------------------------------------------------------ optimisation
    def run(self, x0, method="COBYLA", maxiter=20000, rhobeg=0.05, tol=1e-10, gtol=1e-9):
        trace = []
        self.nfev = 0
        if method == "COBYLA":
            def f(x):
                e = self.energy(x)
                trace.append(e)
                return e
            res = minimize(f, x0, method="COBYLA",
                           options=dict(maxiter=maxiter, rhobeg=rhobeg, tol=tol))
        elif method == "L-BFGS-B":
            def fg(x):
                e, g = self.energy_and_grad(x)
                trace.append(e)
                return e, g
            res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-15, maxfun=maxiter * 2))
        else:
            raise ValueError(method)
        e_fin, g_fin = self.energy_and_grad(res.x)
        return dict(E=e_fin, x=res.x, nfev=len(trace), nit=int(getattr(res, "nit", len(trace)) or len(trace)),
                    success=bool(res.success), message=str(res.message),
                    grad_norm=float(np.linalg.norm(g_fin)), trace=np.array(trace))

    # -------------------------------------------------------- measurement cost
    def shots_for_precision(self, theta, eps_Ha):
        """Shots needed to estimate <H> to standard error eps when each Pauli term is
        measured separately with optimal shot allocation (no grouping):
        M = (sum_i |c_i| sqrt(1 - <P_i>^2))^2 / eps^2 ."""
        psi = self.state(theta)
        tot = 0.0
        for lab, c in zip(self.qubit_op.paulis.to_labels(), self.qubit_op.coeffs):
            if set(lab) == {"I"}:
                continue
            P = SparsePauliOp(lab).to_matrix(sparse=True)
            ev = float(np.real(np.conj(psi) @ (P @ psi)))
            tot += abs(c) * np.sqrt(max(0.0, 1.0 - ev ** 2))
        return float(tot ** 2 / eps_Ha ** 2), len(self.qubit_op) - 1
