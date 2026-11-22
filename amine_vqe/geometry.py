"""XYZ files, the location of the final optimised structure of each species, and simple geometry.

Conventions: complex XYZ files list the amine atoms first and the CO2 atoms last (C, O, O).
The donor nitrogen of a complex is the N atom closest to the CO2 carbon; the same atom index
is used for the free amine (relevant for the diamines EDA and piperazine).
"""
from __future__ import annotations

import math
import pathlib
import re

import numpy as np

from . import config as cfg


def read_xyz(path):
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0])
    return [(l.split()[0], tuple(map(float, l.split()[1:4]))) for l in lines[2:2 + n]]


def format_xyz(atoms, comment=""):
    return f"{len(atoms)}\n{comment}\n" + "".join(
        f"{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n" for s, (x, y, z) in atoms)


def write_xyz(path, atoms, comment=""):
    pathlib.Path(path).write_text(format_xyz(atoms, comment))


def is_complex(stem):
    return stem.endswith("_CO2") and stem != "CO2"


def donor_index(atoms):
    """Index of the donor N: the N atom closest to the CO2 carbon (CO2 = last three atoms)."""
    cpos = np.array(atoms[-3][1])
    ns = [i for i, (a, _) in enumerate(atoms) if a == "N"]
    return min(ns, key=lambda i: np.linalg.norm(np.array(atoms[i][1]) - cpos))


def distance(a, b):
    return math.dist(a, b)


def angle(a, b, c):
    v1 = np.subtract(a, b)
    v2 = np.subtract(c, b)
    cosv = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def rotate_about_axis(points, origin, axis, degrees):
    """Rotate points (n, 3) by the given angle about the axis through origin (Rodrigues)."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(degrees)
    p = np.asarray(points, float) - origin
    rot = p * math.cos(t) + np.cross(k, p) * math.sin(t) + np.outer(p @ k, k) * (1 - math.cos(t))
    return rot + origin


# ------------------------------------------------------------- optimisation jobs of a species
def opt_rounds(stem):
    """All optimisation jobs of a species in order: opt_<stem>, then refinement rounds opt_<stem>_r<k>."""
    root = cfg.orca_dir("opt")
    rounds = [p for p in root.glob(f"opt_{stem}_r*") if re.fullmatch(rf"opt_{re.escape(stem)}_r\d+", p.name)]
    rounds.sort(key=lambda p: int(p.name.rsplit("_r", 1)[1]))
    first = root / f"opt_{stem}"
    return ([first] if first.exists() else []) + rounds


def final_dir(stem):
    rounds = opt_rounds(stem)
    return rounds[-1] if rounds else cfg.orca_dir("opt", f"opt_{stem}")


def final_xyz(stem):
    d = final_dir(stem)
    return d / f"{d.name}.xyz"


def optimised(stem):
    """Atoms of the final optimised structure of a species."""
    return read_xyz(final_xyz(stem))
