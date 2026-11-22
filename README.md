# VQE benchmarking of amine·CO2 complexes

Code and data for the study *Variational Quantum Eigensolver Benchmarking of CO2–Amine Complexes Across Ten
Industry-Relevant Amines: Association Energies, Frontier Orbitals and Validated Active Spaces*.

Ten amines (ammonia, methylamine, ethylamine, n-propylamine, n-butylamine, MEA, DEA, TEA, ethylenediamine and
piperazine), CO2 and the ten amine·CO2 complexes are treated with one workflow:

1. **Geometries and energetics** – B3LYP-D3(BJ)/def2-SVP optimisation and frequencies (all structures are minima),
   def2-TZVP single points with B3LYP-D3(BJ), ωB97X-D3 and DLPNO-CCSD(T), counterpoise-corrected association
   energies, a TZ/QZ extrapolation for two complexes and a def2-TZVP geometry check (ORCA 6.1.1).
2. **Frontier orbitals** – HOMO, LUMO, gap and the CO2 π* level, with their character on the donor N and on CO2.
3. **Active spaces** – projected n<sub>N</sub>, π<sub>g</sub>, π\*(in), π\*(out) (and two amine virtuals) orbitals
   giving CAS(2,3), CAS(4,4) and the size-consistent CAS(4,6); CASCI, CASSCF, larger AVAS spaces and full-space
   MP2/CCSD(T) as references; a size-consistency test; and the energy-ordered STO-3G windows for comparison (PySCF).
4. **VQE** – Jordan–Wigner mapping and the Qiskit Nature UCCSD ansatz on 6, 8 and 12 qubits, exact state-vector
   energies with analytic gradients, COBYLA and L-BFGS-B from eleven starting points, a lighter two-start protocol,
   a shot-noise test and measurement-cost estimates.

## Running

```bash
conda env create -f environment.yml
conda activate amine-vqe
python -m amine_vqe all            # every stage in order
python -m pytest tests             # consistency tests of the VQE engine (about a minute)
```

`python -m amine_vqe <stage>` runs a single stage (`orca`, `energetics`, `frontier`, `active-space`, `vqe`,
`report`). Each stage skips work whose results already exist, so on this repository `all` rebuilds the tables,
figures and summary from the stored outputs in a few seconds; `--force` recomputes a stage, `--only <name>`
restricts the active-space and VQE stages to matching species, and `--workers N` sets the number of parallel
ORCA jobs or VQE processes. To repeat the whole study from the start geometries, point the workflow to an empty
folder that contains only `geometries/start/`:

```bash
mkdir -p fresh/geometries && cp -r data/geometries/start fresh/geometries/
export ORCA_EXE=/path/to/orca          # ORCA 6.1.1; the stage is skipped if ORCA is not found
python -m amine_vqe all --data fresh --results fresh_results --workers 8
```

ORCA runs in a scratch folder (`AMINE_VQE_SCRATCH`, default: the system temporary folder) and only the result
files are copied back. The ORCA stage repeats the refinement of minima automatically: a structure with an
imaginary mode below −5 cm⁻¹ is displaced along that mode and re-optimised with tighter settings, and a mode that
survives (the nearly free rotation of NH3 about the N···C axis in NH3·CO2) is removed by rotating the amine by 30°
before the last re-optimisation. `python -m amine_vqe preconverge <species>` pre-converges a floppy structure with
PySCF/geomeTRIC before its next ORCA round (used for TEA·CO2). The full active-space stage with CCSD(T) and the
VQE benchmark take several hours on a workstation.

## Contents

| Path | Contents |
|---|---|
| `amine_vqe/` | The workflow package (`__main__.py` is the entry point; one module per stage) |
| `tests/` | Consistency tests of the VQE engine against Qiskit and PySCF |
| `data/geometries/` | Start geometries and the final optimised structures (XYZ, Å) |
| `data/orca/opt/`, `opt_tzvp/` | ORCA optimisation and frequency jobs (`opt_<species>`, refinement rounds `_r<k>`): input, start structure, output, final structure, Hessian |
| `data/orca/sp/<method>/` | ORCA single points and counterpoise fragments (`__A_AB`, `__A_A`, `__B_AB`, `__B_B`) for `b3lyp_svp`, `b3lyp_tzvp`, `wb97xd3_tzvp`, `dlpno_tzvp`, `dlpno_cbs34` |
| `data/active_space/` | Per-species active-space results (orbital character, CASCI/CASSCF, AVAS, MP2/CCSD(T)), size-consistency test, energy-ordered windows; `integrals/` holds the active-space Hamiltonians used by VQE |
| `data/vqe/` | Every VQE run (`results/<system>.json`) and its energy trace (`traces/<system>.npz`) |
| `results/` | Optimisation summary, association and frontier-orbital energies, `summary.json`, tables (CSV) and figures |

System names in `data/vqe` are `<species>_cas<electrons>_<orbitals>`. Energies are in hartree unless a key ends
in `_kcal` or `_eV`.

## Software

Python 3.11, NumPy, SciPy 1.17.1, Matplotlib, PySCF 2.10.0, geomeTRIC 1.1.1, Qiskit 1.4.3, Qiskit Nature 0.7.2;
ORCA 6.1.1 for the DFT and DLPNO-CCSD(T) calculations.

## License

MIT (see `LICENSE`).
