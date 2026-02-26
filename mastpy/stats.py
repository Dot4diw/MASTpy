"""
Statistical tests for ZlmFit objects.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union, Dict, Any
from .zlm import ZlmFit


def _normal_ci(center: np.ndarray, se: np.ndarray, level: float = 0.95) -> tuple:
    """Calculate confidence intervals."""
    zstar = stats.norm.ppf(1 - (1 - level) / 2)
    ci_lo = center - se * zstar
    ci_hi = center + se * zstar
    return ci_lo, ci_hi


def coefficient_hypothesis(contrast: str, coef_names: list = None) -> np.ndarray:
    """
    Create a contrast matrix from a coefficient hypothesis string.

    Parameters
    ----------
    contrast : str
        String specifying the contrast (e.g., 'GroupB - GroupA')
    coef_names : list, optional
        List of coefficient names to match against

    Returns
    -------
    np.ndarray
        Contrast matrix
    """
    import re

    contrast = contrast.strip()

    if coef_names is not None:
        contrast_vector = np.zeros(len(coef_names))
        for i, name in enumerate(coef_names):
            if name == "(Intercept)":
                continue
            if name.startswith("group") and contrast == "group":
                contrast_vector[i] = 1
            elif name.replace("group[T.", "").rstrip("]") == contrast.replace(
                "group[T.", ""
            ).rstrip("]"):
                contrast_vector[i] = 1
            elif name == contrast:
                contrast_vector[i] = 1
        return contrast_vector.reshape(1, -1)

    parts = re.split(r"\s*[-+]\s*", contrast)
    operators = re.findall(r"\s*([-+])\s*", contrast)

    if len(parts) == 1:
        return np.array([[1]])

    n_contrasts = len(parts)
    contrast_matrix = np.zeros((n_contrasts, 1))

    for i, (part, op) in enumerate(zip(parts, operators)):
        part = part.strip()
        if op == "+":
            contrast_matrix[i, 0] = 1
        else:
            contrast_matrix[i, 0] = -1

    return contrast_matrix


class Hypothesis:
    """Hypothesis specification for contrast testing."""

    def __init__(self, contrast: Union[str, np.ndarray]):
        self._contrast = contrast
        if isinstance(contrast, str):
            self._contrast_matrix = coefficient_hypothesis(contrast)
        else:
            self._contrast_matrix = contrast

    @property
    def contrast_matrix(self) -> np.ndarray:
        return self._contrast_matrix

    def __repr__(self) -> str:
        return f"Hypothesis(contrast={self._contrast})"


class CoefficientHypothesis(Hypothesis):
    """Coefficient hypothesis - test specific coefficients."""

    def __init__(self, coefficients: Union[str, list]):
        if isinstance(coefficients, str):
            coefficients = [coefficients]
        self._coefficients = coefficients
        self._index = None
        super().__init__(coefficients)

    @property
    def index(self) -> np.ndarray:
        return self._index

    @index.setter
    def index(self, value: np.ndarray):
        self._index = value

    def __repr__(self) -> str:
        return f"CoefficientHypothesis(coefficients={self._coefficients})"


def _wald_test(
    coefC: np.ndarray,
    coefD: np.ndarray,
    vcovC: np.ndarray,
    vcovD: np.ndarray,
    contrast: np.ndarray,
    converged: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Perform Wald test.

    Parameters
    ----------
    coefC : np.ndarray
        Continuous coefficients (genes x coefficients)
    coefD : np.ndarray
        Discrete coefficients
    vcovC : np.ndarray
        Continuous vcov (coeff x coeff x genes)
    vcovD : np.ndarray
        Discrete vcov
    contrast : np.ndarray
        Contrast matrix
    converged : np.ndarray
        Convergence flags

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with chi-squared stats, df, and p-values
    """
    n_genes = coefC.shape[0]
    n_coef = coefC.shape[1]

    def _complexify_na(x):
        x = np.asarray(x, dtype=np.complex128).copy()
        nan_mask = np.isnan(x.real)
        x[nan_mask] = 0 + 1j
        return x

    def _uncomplexify(x):
        x = np.asarray(x).copy()
        if np.iscomplexobj(x):
            x[np.abs(x.imag) > 1e-10] = np.nan
            return x.real
        return x

    coefC_complex = _complexify_na(coefC)
    coefD_complex = _complexify_na(coefD)

    contrC = coefC_complex @ contrast.T
    contrD = coefD_complex @ contrast.T

    contrCovC = np.zeros(
        (contrast.shape[0], contrast.shape[0], n_genes), dtype=np.complex128
    )
    contrCovD = np.zeros(
        (contrast.shape[0], contrast.shape[0], n_genes), dtype=np.complex128
    )

    for i in range(n_genes):
        vcovC_complex = _complexify_na(vcovC[:, :, i])
        vcovD_complex = _complexify_na(vcovD[:, :, i])
        contrCovC[:, :, i] = contrast @ vcovC_complex @ contrast.T
        contrCovD[:, :, i] = contrast @ vcovD_complex @ contrast.T

    contrC = _uncomplexify(contrC).flatten()
    contrD = _uncomplexify(contrD).flatten()

    n_contrasts = contrast.shape[0]
    dof = np.outer(np.array([n_contrasts, n_contrasts]), converged.astype(int))

    lambdaC = np.zeros(n_genes)
    lambdaD = np.zeros(n_genes)

    for i in range(n_genes):
        if converged[i]:
            try:
                covC = _uncomplexify(contrCovC[:, :, i])
                covD = _uncomplexify(contrCovD[:, :, i])
                c_i = np.array([contrC[i]])
                d_i = np.array([contrD[i]])
                if np.abs(c_i[0]) > 1e-10:
                    lambdaC[i] = c_i @ np.linalg.solve(covC, c_i.T).flatten()
                if np.abs(d_i[0]) > 1e-10:
                    lambdaD[i] = d_i @ np.linalg.solve(covD, d_i.T).flatten()
            except Exception as e:
                pass

    lambda_hurdle = lambdaC + lambdaD
    dof_hurdle = dof[0] + dof[1]

    # 当df=0时，p值设为1，与R MAST保持一致
    pvalC = np.where(dof[0, :] > 0, stats.chi2.sf(lambdaC, dof[0, :]), 1.0)
    pvalD = np.where(dof[1, :] > 0, stats.chi2.sf(lambdaD, dof[1, :]), 1.0)
    pvalH = np.where(dof_hurdle > 0, stats.chi2.sf(lambda_hurdle, dof_hurdle), 1.0)

    return {
        "lambda": np.column_stack([lambdaC, lambdaD, lambda_hurdle]),
        "df": np.column_stack([dof[0], dof[1], dof_hurdle]),
        "pvalue": np.column_stack([pvalC, pvalD, pvalH]),
    }


def waldTest(
    zlmfit: ZlmFit, hypothesis: Union[str, Hypothesis, np.ndarray]
) -> pd.DataFrame:
    """
    Wald test for ZlmFit object.

    Parameters
    ----------
    zlmfit : ZlmFit
        Fitted zero-inflated model
    hypothesis : str, Hypothesis, or np.ndarray
        Hypothesis to test

    Returns
    -------
    pd.DataFrame
        DataFrame with test results (chi-squared, df, p-value)
    """
    if isinstance(hypothesis, str):
        h = Hypothesis(hypothesis)
        contrast = coefficient_hypothesis(hypothesis, zlmfit.coefC.columns.tolist())
    elif isinstance(hypothesis, Hypothesis):
        contrast = coefficient_hypothesis(
            str(hypothesis), zlmfit.coefC.columns.tolist()
        )
    else:
        contrast = hypothesis

    coefC = zlmfit.coefC.values
    coefD = zlmfit.coefD.values
    vcovC = zlmfit.vcovC
    vcovD = zlmfit.vcovD
    converged = zlmfit.converged.values.all(axis=1)

    result = _wald_test(coefC, coefD, vcovC, vcovD, contrast, converged)

    genes = zlmfit.coefC.index.tolist()

    df_result = pd.DataFrame(
        {
            "primerid": genes,
            "lambda": result["lambda"][:, 2],
            "df": result["df"][:, 2],
            "Pr(>Chisq)": result["pvalue"][:, 2],
        }
    )

    return df_result


def _lr_test(zlmfit_full: ZlmFit, zlmfit_reduced: ZlmFit) -> Dict[str, np.ndarray]:
    """
    Perform likelihood ratio test.

    Parameters
    ----------
    zlmfit_full : ZlmFit
        Full model
    zlmfit_reduced : ZlmFit
        Reduced model

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with chi-squared stats, df, and p-values
    """
    loglik_full = zlmfit_full.loglik.values
    loglik_reduced = zlmfit_reduced.loglik.values

    converged_full = zlmfit_full.converged.values
    converged_reduced = zlmfit_reduced.converged.values

    testable = converged_full & converged_reduced

    lambda_lr = -2 * (loglik_reduced - loglik_full)
    lambda_lr[~testable] = 0

    df = zlmfit_full.df_resid.values - zlmfit_reduced.df_resid.values
    df[~testable] = 0

    pvalue = stats.chi2.sf(lambda_lr, df)

    return {
        "lambda": lambda_lr,
        "df": df,
        "pvalue": pvalue,
    }


def lrTest(
    zlmfit: ZlmFit, hypothesis: Union[str, Hypothesis, np.ndarray], **kwargs
) -> pd.DataFrame:
    """
    Likelihood ratio test for ZlmFit object.

    Parameters
    ----------
    zlmfit : ZlmFit
        Fitted zero-inflated model
    hypothesis : str, Hypothesis, or np.ndarray
        Hypothesis to test (reduced model formula or contrast)

    Returns
    -------
    pd.DataFrame
        DataFrame with test results
    """
    import re

    if isinstance(hypothesis, str):
        if "-" in hypothesis or "+" in hypothesis:
            h = Hypothesis(hypothesis)
            contrast = h.contrast_matrix

            coefC = zlmfit.coefC.values
            coefD = zlmfit.coefD.values
            n_genes = coefC.shape[0]

            testable = zlmfit.converged.values.all(axis=1)

            lambda_lr = np.zeros((n_genes, 3))
            df = np.zeros((n_genes, 3))
            pvalue = np.ones((n_genes, 3))

            for i in range(n_genes):
                if testable[i, 0]:
                    lambda_lr[i, 0] = (
                        coefC[i, :]
                        @ contrast
                        @ np.linalg.solve(
                            contrast.T @ zlmfit.vcovC[:, :, i] @ contrast,
                            contrast.T @ coefC[i, :],
                        )
                    )
                if testable[i, 1]:
                    lambda_lr[i, 1] = (
                        coefD[i, :]
                        @ contrast
                        @ np.linalg.solve(
                            contrast.T @ zlmfit.vcovD[:, :, i] @ contrast,
                            contrast.T @ coefD[i, :],
                        )
                    )
                lambda_lr[i, 2] = lambda_lr[i, 0] + lambda_lr[i, 1]
                df[i, 0] = contrast.shape[0]
                df[i, 1] = contrast.shape[0]
                df[i, 2] = 2 * contrast.shape[0]

                if testable[i, 0]:
                    pvalue[i, 0] = stats.chi2.sf(lambda_lr[i, 0], df[i, 0])
                if testable[i, 1]:
                    pvalue[i, 1] = stats.chi2.sf(lambda_lr[i, 1], df[i, 1])
                if testable[i, 0] and testable[i, 1]:
                    pvalue[i, 2] = stats.chi2.sf(lambda_lr[i, 2], df[i, 2])

            genes = zlmfit.coefC.index.tolist()
            result_df = pd.DataFrame(lambda_lr[:, 2], index=genes, columns=["hurdle"])
            result_df["Pr(>Chisq)"] = pvalue[:, 2]
            result_df["df"] = df[:, 2]

            return result_df
        else:
            from .zlm import zlm

            formula_parts = re.split(r"[-+]", hypothesis)
            new_formula = "~" + hypothesis

            try:
                reduced_zlmfit = zlm(
                    new_formula,
                    zlmfit._adata,
                    method="bayesglm",
                    ebayes=False,
                    silent=True,
                )
            except:
                return None

            result = _lr_test(zlmfit, reduced_zlmfit)

            genes = zlmfit.coefC.index.tolist()
            result_df = pd.DataFrame(
                {
                    "primerid": np.repeat(genes, 3),
                    "component": np.tile(["C", "D", "H"], len(genes)),
                    "lambda": result["lambda"].flatten(),
                    "df": result["df"].flatten(),
                    "Pr(>Chisq)": result["pvalue"].flatten(),
                }
            )

            return result_df.pivot_table(
                index="primerid",
                columns="component",
                values=["lambda", "df", "Pr(>Chisq)"],
            )
    else:
        raise ValueError("hypothesis must be a string")
