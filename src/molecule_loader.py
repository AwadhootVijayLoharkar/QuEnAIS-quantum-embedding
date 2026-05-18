from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

Geometry = List[Tuple[str, Tuple[float, float, float]]]


def from_xyz(filepath: str | Path) -> Geometry:
    """Load geometry from a .xyz file."""
    atoms = []
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            atoms.append((parts[0], (float(parts[1]),
                                     float(parts[2]),
                                     float(parts[3]))))
    return atoms


def from_dict(data: dict) -> Geometry:
    """
    data = {"Fe": [(0,0,0)], "N": [(0,0,2), (0,0,-2)]}
    """
    atoms = []
    for sym, coords_list in data.items():
        for coords in coords_list:
            atoms.append((sym, tuple(coords)))
    return atoms