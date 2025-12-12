"""Optimizers for switch-bank.

This package provides:
- `Muon`: orthogonalized momentum optimizer for 2D+ bf16 matrices.
- `NeoMuon`: geometry-aware hybrid optimizer built on Muon.
"""

from .muon import Muon
from .neomuon import NeoMuon

__all__ = ["Muon", "NeoMuon"]
