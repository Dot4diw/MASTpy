"""Core functionality for MASTpy"""

from .single_cell_assay import SingleCellAssay
from .zlm import zlm, ZlmFit
from .lm_wrapper import GLMlike, BayesGLMlike, LMERlike
from .hypothesis import Hypothesis, CoefficientHypothesis
from .bootstrap import bootVcov1, pbootVcov1, CovFromBoots
from .predict import predict_ZlmFit, impute, add_predict_method

# Add predict method to ZlmFit
add_predict_method()

__all__ = [
    "SingleCellAssay",
    "zlm",
    "ZlmFit",
    "GLMlike",
    "BayesGLMlike",
    "LMERlike",
    "Hypothesis",
    "CoefficientHypothesis",
    "bootVcov1",
    "pbootVcov1",
    "CovFromBoots",
    "predict_ZlmFit",
    "impute"
]