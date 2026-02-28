"""Model-based Analysis of Single-cell Transcriptomics in Python"""

from .core.single_cell_assay import SingleCellAssay
from .core.zlm import zlm, ZlmFit
from .core.hypothesis import Hypothesis, CoefficientHypothesis
from .utils.utils import ebayes, getLogFC
from .tools.find_deg import find_deg, find_all_degs

__version__ = "0.1.0"
__all__ = [
    "SingleCellAssay",
    "zlm",
    "ZlmFit",
    "Hypothesis",
    "CoefficientHypothesis",
    "ebayes",
    "getLogFC",
    "find_deg",
    "find_all_degs"
]