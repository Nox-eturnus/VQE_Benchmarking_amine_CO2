"""Command-line entry point:  python -m amine_vqe <stage> [options]

Stages, in the order used by `all`:
    orca          geometry optimisations, refinement of minima, def2-TZVP geometry check and all single points
    energetics    optimisation summary, optimised XYZ files, association energies and thermal corrections
    frontier      frontier-orbital energies and character (PySCF, B3LYP/def2-TZVP)
    active-space  active spaces, CASCI/CASSCF/AVAS, MP2/CCSD(T), size-consistency test, energy-ordered windows
    vqe           VQE/UCCSD benchmark (validation, multistart and two-start runs, shot-noise test)
    report        tables, summary.json and figures
    all           every stage above

Each stage skips work whose results already exist (finished ORCA outputs, result files), so `all`
on a complete data folder only rebuilds the analysis; --force recomputes the selected stage.
`preconverge <species>` pre-converges a floppy structure with PySCF/geomeTRIC before its next ORCA round.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import time

from . import config as cfg

STAGES = ["orca", "energetics", "frontier", "active-space", "vqe", "report"]


def stage_orca(a):
    from . import orca
    if shutil.which(cfg.ORCA) is None and not pathlib.Path(cfg.ORCA).exists():
        print(f"ORCA executable '{cfg.ORCA}' not found (set ORCA_EXE); existing ORCA outputs are used as they are")
        return
    orca.run_all(workers=a.workers)


def stage_energetics(a):
    from . import energetics
    energetics.opt_summary()
    energetics.export_geometries()
    energetics.association_energies()


def stage_frontier(a):
    from . import frontier
    frontier.run(force=a.force)


def stage_active_space(a):
    from . import active_space
    active_space.run(only=a.only, force=a.force)


def stage_vqe(a):
    from . import benchmark
    benchmark.run(workers=a.workers, only=a.only, force=a.force)


def stage_report(a):
    from . import report
    report.run()


RUN = {"orca": stage_orca, "energetics": stage_energetics, "frontier": stage_frontier,
       "active-space": stage_active_space, "vqe": stage_vqe, "report": stage_report}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m amine_vqe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all", "preconverge"])
    ap.add_argument("species", nargs="?", help="species for 'preconverge'")
    ap.add_argument("--data", type=pathlib.Path, help="data folder (default: ./data)")
    ap.add_argument("--results", type=pathlib.Path, help="results folder (default: ./results)")
    ap.add_argument("--workers", type=int, default=4, help="parallel ORCA jobs / VQE processes")
    ap.add_argument("--only", help="restrict the active-space or VQE stage to matching species/systems")
    ap.add_argument("--force", action="store_true", help="recompute results that already exist")
    a = ap.parse_args(argv)
    if a.data:
        cfg.DATA = a.data.resolve()
    if a.results:
        cfg.RESULTS = a.results.resolve()
    if a.stage == "preconverge":
        from . import orca
        orca.preconverge(a.species)
        return
    for stage in (STAGES if a.stage == "all" else [a.stage]):
        t0 = time.time()
        print(f"=== {stage}", flush=True)
        RUN[stage](a)
        print(f"=== {stage} finished in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
