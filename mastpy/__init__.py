"""MASTpy: Model-based Analysis of Single Cell Transcriptomics in Python

This package provides methods and models for handling zero-inflated single cell
assay data, ported from the R MAST package.
Uses scanpy's AnnData as the data structure.
"""

__version__ = "0.1.0"

from .zlm import zlm, ZlmFit
from .stats import lrTest, waldTest, coefficient_hypothesis
from .io import from_matrix, from_flat_df

__all__ = [
    "ZlmFit",
    "zlm",
    "lrTest",
    "waldTest",
    "coefficient_hypothesis",
    "from_matrix",
    "from_flat_df",
]
