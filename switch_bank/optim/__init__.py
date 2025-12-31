"""Optimizers for switch-bank.

This package provides:
- `Muon`: hybrid optimizer with a Muon/TurboMuon spectral branch and an AdamW branch.
"""

from .muon import Muon

__all__ = ["Muon"]
