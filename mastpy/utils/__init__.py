"""Utility functions for MASTpy"""

from .utils import ebayes, getLogFC, calculate_variance, getSSg_rNg, solveMoM, getMarginalHyperLikelihood

__all__ = [
    "ebayes",
    "getLogFC",
    "calculate_variance",
    "getSSg_rNg",
    "solveMoM",
    "getMarginalHyperLikelihood"
]