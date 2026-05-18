from __future__ import annotations
from pathlib import Path
from typing import List, Tuple


Geometry = List[Tuple[str, Tuple[float, float, float]]]


def from_xyz(filepath: str | Path) -> Geometry:
    """Load geometry from a standard .xyz file."""
    atoms = []
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines[2:]:          # skip atom-count and comment lines
        parts = line.split()
        if len(parts) == 4:
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append((sym, (x, y, z)))
    return atoms


def from_dict(data: dict) -> Geometry:
    """
    Load from a plain dict:
      {"Fe": [(0,0,0)], "N": [(0,0,2),(0,0,-2), ...]}
    """
    atoms = []
    for sym, coords_list in data.items():
        for coords in coords_list:
            atoms.append((sym, tuple(coords)))
    return atoms