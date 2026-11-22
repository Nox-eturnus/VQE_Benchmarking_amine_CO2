"""Species, physical constants and file locations shared by every stage of the workflow.

All inputs and intermediate results live under DATA, the tables, figures and summary files
under RESULTS. Both default to folders of this repository and can be redirected with the
command-line options --data / --results (or the AMINE_VQE_DATA / AMINE_VQE_RESULTS
environment variables), so that the workflow can be rerun from scratch in a new location.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(os.environ.get("AMINE_VQE_DATA", ROOT / "data"))
RESULTS = pathlib.Path(os.environ.get("AMINE_VQE_RESULTS", ROOT / "results"))

# ORCA executable and a scratch folder without spaces for its (large) temporary files
ORCA = os.environ.get("ORCA_EXE", "orca")
SCRATCH = pathlib.Path(os.environ.get("AMINE_VQE_SCRATCH", pathlib.Path(tempfile.gettempdir()) / "amine_vqe"))

AMINES = ["ammonia", "methylamine", "ethylamine", "propylamine", "butylamine",
          "MEA", "DEA", "TEA", "EDA", "piperazine"]
COMPLEXES = [f"{a}_CO2" for a in AMINES]
SPECIES = ["CO2"] + AMINES + COMPLEXES
LABELS = {"ammonia": "Ammonia", "methylamine": "Methylamine", "ethylamine": "Ethylamine",
          "propylamine": "n-Propylamine", "butylamine": "n-Butylamine", "MEA": "MEA",
          "DEA": "DEA", "TEA": "TEA", "EDA": "EDA", "piperazine": "Piperazine", "CO2": "CO2"}

HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
IMAGINARY_CUTOFF_CM1 = 5.0          # |nu| below this is treated as zero (near-free internal rotation)


def start_geometries() -> pathlib.Path:
    return DATA / "geometries" / "start"


def optimised_geometries() -> pathlib.Path:
    return DATA / "geometries" / "optimised"


def orca_dir(*parts) -> pathlib.Path:
    return DATA.joinpath("orca", *parts)


def active_space_dir(*parts) -> pathlib.Path:
    return DATA.joinpath("active_space", *parts)


def vqe_dir(*parts) -> pathlib.Path:
    return DATA.joinpath("vqe", *parts)


def results_dir(*parts) -> pathlib.Path:
    return RESULTS.joinpath(*parts)
