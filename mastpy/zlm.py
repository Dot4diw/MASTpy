"""
Zero-inflated linear model fitting for single-cell transcriptomics data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import BayesianRidge, Ridge, LogisticRegression
from typing import Optional, Union, Callable, Dict, Any, List
import warnings


class ZlmFit:
    """
    Zero-inflated linear model fit result.

    Similar to R's ZlmFit class, holds coefficients, variance-covariance matrices,
    and other model results from fitting a hurdle model.
    """

    def __init__(
        self,
        coefC: pd.DataFrame,
        coefD: pd.DataFrame,
        vcovC: np.ndarray,
        vcovD: np.ndarray,
        design: pd.DataFrame,
        formula: str,
        loglik: pd.DataFrame,
        deviance: pd.DataFrame,
        df_null: pd.DataFrame,
        df_resid: pd.DataFrame,
        dispersion: pd.DataFrame,
        converged: pd.DataFrame,
        adata: Any = None,
    ):
        self._coefC = coefC
        self._coefD = coefD
        self._vcovC = vcovC
        self._vcovD = vcovD
        self._design = design
        self._formula = formula
        self._loglik = loglik
        self._deviance = deviance
        self._df_null = df_null
        self._df_resid = df_resid
        self._dispersion = dispersion
        self._converged = converged
        self._adata = adata

    @property
    def coefC(self) -> pd.DataFrame:
        """Continuous component coefficients."""
        return self._coefC

    @property
    def coefD(self) -> pd.DataFrame:
        """Discrete component coefficients."""
        return self._coefD

    @property
    def vcovC(self) -> np.ndarray:
        """Variance-covariance matrices for continuous component."""
        return self._vcovC

    @property
    def vcovD(self) -> np.ndarray:
        """Variance-covariance matrices for discrete component."""
        return self._vcovD

    @property
    def design(self) -> pd.DataFrame:
        """Design matrix."""
        return self._design

    @property
    def formula(self) -> str:
        """Model formula."""
        return self._formula

    @property
    def loglik(self) -> pd.DataFrame:
        """Log-likelihoods."""
        return self._loglik

    @property
    def deviance(self) -> pd.DataFrame:
        """Deviances."""
        return self._deviance

    @property
    def df_null(self) -> pd.DataFrame:
        """Null degrees of freedom."""
        return self._df_null

    @property
    def df_resid(self) -> pd.DataFrame:
        """Residual degrees of freedom."""
        return self._df_resid

    @property
    def dispersion(self) -> pd.DataFrame:
        """Dispersions."""
        return self._dispersion

    @property
    def converged(self) -> pd.DataFrame:
        """Convergence flags."""
        return self._converged

    @property
    def ngenes(self) -> int:
        """Number of genes."""
        return self._coefC.shape[0]

    @property
    def ncells(self) -> int:
        """Number of cells."""
        return self._design.shape[0]

    def coef(self, which: str = "C") -> pd.DataFrame:
        """Get coefficients.

        Parameters
        ----------
        which : str
            'C' for continuous, 'D' for discrete

        Returns
        -------
        pd.DataFrame
            Coefficient matrix
        """
        if which == "C":
            return self._coefC
        elif which == "D":
            return self._coefD
        else:
            raise ValueError("which must be 'C' or 'D'")

    def vcov(self, which: str = "C") -> np.ndarray:
        """Get variance-covariance matrices.

        Parameters
        ----------
        which : str
            'C' for continuous, 'D' for discrete

        Returns
        -------
        np.ndarray
            3D array of variance-covariance matrices
        """
        if which == "C":
            return self._vcovC
        elif which == "D":
            return self._vcovD
        else:
            raise ValueError("which must be 'C' or 'D'")

    def se_coef(self, which: str = "C") -> pd.DataFrame:
        """Get standard errors of coefficients.

        Parameters
        ----------
        which : str
            'C' for continuous, 'D' for discrete

        Returns
        -------
        pd.DataFrame
            Standard error matrix
        """
        vc = self.vcov(which)
        se = np.sqrt(np.array([np.diag(vc[:, :, i]) for i in range(vc.shape[2])]).T)
        se = pd.DataFrame(
            se, index=self.coef(which).index, columns=self.coef(which).columns
        )
        return se

    def summary(self, level: float = 0.95) -> Dict[str, pd.DataFrame]:
        """Summarize the model fit.

        Parameters
        ----------
        level : float
            Confidence level for confidence intervals

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing coefficient summaries
        """
        from .stats import _normal_ci

        result = {}
        for component, which in [("C", "C"), ("D", "D")]:
            coefs = self.coef(which)
            se = self.se_coef(which)
            z = coefs / se
            ci = _normal_ci(coefs.values, se.values, level)

            df = pd.DataFrame(
                {
                    "coef": coefs.values.flatten(),
                    "se": se.values.flatten(),
                    "z": z.values.flatten(),
                    "ci.lo": ci[0].flatten(),
                    "ci.hi": ci[1].flatten(),
                },
                index=pd.MultiIndex.from_product(
                    [coefs.index, coefs.columns], names=["primerid", "contrast"]
                ),
            )
            result[component] = df
        return result

    def __repr__(self) -> str:
        return f"ZlmFit: {self.ngenes} genes, {self.ncells} cells\nFormula: {self._formula}"


def _make_design_matrix(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """Create design matrix from formula and data."""
    import patsy

    return patsy.dmatrix(formula, data)


def _ebayes(
    y: np.ndarray,
    design: np.ndarray,
    method: str = "MOM",
) -> Dict[str, np.ndarray]:
    """
    Empirical Bayes variance shrinkage.

    Parameters
    ----------
    y : np.ndarray
        Expression matrix (cells x genes)
    design : np.ndarray
        Design matrix
    method : str
        'MOM' (method of moments) or 'MLE'

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'v' (prior variance) and 'df' (degrees of freedom)
    """
    n, p = design.shape
    residuals = y - y.mean(axis=0)
    sigma2 = np.var(residuals, axis=0, ddof=p)

    if method == "MOM":
        prior_var = np.median(sigma2)
        df = 1.0
    else:
        prior_var = np.mean(sigma2)
        df = n - p

    return {"v": np.full(y.shape[1], prior_var), "df": np.full(y.shape[1], df)}


def _fit_continuous_single_gene(args):
    """Fit continuous component for a single gene (for parallel processing)."""
    yi, design, method, p = args
    mask = ~np.isnan(yi) & (yi != 0)

    if mask.sum() < p + 2:
        return np.full(p, np.nan), np.full((p, p), np.nan), False, np.nan

    if method == "bayesglm":
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
    elif method == "ridge":
        model = Ridge(alpha=1.0)
    else:
        model = BayesianRidge()

    try:
        model.fit(design[mask, :], yi[mask])
        coefs = model.coef_
        if hasattr(model, "alpha_"):
            scale = 1.0 / model.alpha_
        else:
            scale = np.var(yi[mask] - design[mask, :] @ model.coef_) + 1e-10
        try:
            vcov = scale * np.linalg.inv(
                design[mask, :].T @ design[mask, :] + np.eye(p) * 1e-6
            )
        except:
            vcov = np.eye(p) * scale
        fitted = True
        loglik = -0.5 * np.sum(
            (yi[mask] - design[mask, :] @ model.coef_) ** 2 / scale
        ) - 0.5 * len(yi[mask]) * np.log(2 * np.pi * scale)
        return coefs, vcov, fitted, loglik
    except:
        return np.full(p, np.nan), np.full((p, p), np.nan), False, np.nan


def _fit_continuous(
    y: np.ndarray,
    design: np.ndarray,
    method: str = "bayesglm",
    prior_var: float = 1.0,
    parallel: bool = True,
    n_jobs: int = -1,
    **kwargs,
) -> tuple:
    """
    Fit continuous (linear) component.
    """
    n, p = design.shape
    genes = y.shape[1]

    coefs = np.zeros((genes, p))
    vcovs = np.zeros((p, p, genes))
    fitted = np.zeros((n, genes), dtype=bool)
    logliks = np.full(genes, np.nan)

    if parallel and genes > 10:
        from joblib import Parallel, delayed

        n_cores = n_jobs if n_jobs > 0 else -1

        results = Parallel(n_jobs=n_cores, backend="loky")(
            [
                delayed(_fit_continuous_single_gene)((y[:, i], design, method, p))
                for i in range(genes)
            ]
        )

        for i, (coef, vc, fit, ll) in enumerate(results):
            coefs[i, :] = coef
            vcovs[:, :, i] = vc
            fitted[:, i] = fit
            logliks[i] = ll
    else:
        for i in range(genes):
            yi = y[:, i]
            mask = ~np.isnan(yi) & (yi != 0)

            if mask.sum() < p + 2:
                coefs[i, :] = np.nan
                vcovs[:, :, i] = np.nan
                fitted[:, i] = False
                continue

            if method == "bayesglm":
                model = BayesianRidge(
                    alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
                )
            elif method == "ridge":
                model = Ridge(alpha=kwargs.get("alpha", 1.0))
            else:
                model = BayesianRidge()

            try:
                model.fit(design[mask, :], yi[mask])
                coefs[i, :] = model.coef_
                if hasattr(model, "alpha_"):
                    scale = 1.0 / model.alpha_
                else:
                    scale = np.var(yi[mask] - design[mask, :] @ model.coef_) + 1e-10
                try:
                    vcovs[:, :, i] = scale * np.linalg.inv(
                        design[mask, :].T @ design[mask, :] + np.eye(p) * 1e-6
                    )
                except:
                    vcovs[:, :, i] = np.eye(p) * scale
                fitted[:, i] = True
                logliks[i] = -0.5 * np.sum(
                    (yi[mask] - design[mask, :] @ model.coef_) ** 2 / scale
                ) - 0.5 * len(yi[mask]) * np.log(2 * np.pi * scale)
            except Exception as e:
                coefs[i, :] = np.nan
                vcovs[:, :, i] = np.nan
                fitted[:, i] = False

    return coefs, vcovs, fitted, logliks


def _fit_discrete_single_gene(args):
    """Fit discrete component for a single gene (for parallel processing)."""
    yi, design, p = args
    mask = ~np.isnan(yi) & (yi >= 0)

    if mask.sum() < p + 2:
        return np.full(p, np.nan), np.full((p, p), np.nan), False, np.nan

    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(design[mask, :], yi[mask])
        coefs = model.coef_.flatten()

        pred_proba = model.predict_proba(design[mask, :])
        pred_proba = np.clip(pred_proba, 1e-10, 1 - 1e-10)
        hessian = (
            design[mask, :].T
            @ np.diag(pred_proba[:, 0] * (1 - pred_proba[:, 0]))
            @ design[mask, :]
        )
        vcov = np.linalg.inv(hessian + np.eye(p) * 1e-6)
        fitted = True
        loglik = np.sum(
            yi[mask] * np.log(pred_proba[:, 1])
            + (1 - yi[mask]) * np.log(pred_proba[:, 0])
        )
        return coefs, vcov, fitted, loglik
    except:
        return np.full(p, np.nan), np.full((p, p), np.nan), False, np.nan


def _fit_discrete(
    y_binary: np.ndarray,
    design: np.ndarray,
    method: str = "bayesglm",
    parallel: bool = True,
    n_jobs: int = -1,
    **kwargs,
) -> tuple:
    """
    Fit discrete (logistic) component.
    """
    n, p = design.shape
    genes = y_binary.shape[1]

    coefs = np.zeros((genes, p))
    vcovs = np.zeros((p, p, genes))
    fitted = np.zeros((n, genes), dtype=bool)
    logliks = np.zeros(genes)

    if parallel and genes > 10:
        from joblib import Parallel, delayed

        n_cores = n_jobs if n_jobs > 0 else -1

        results = Parallel(n_jobs=n_cores, backend="loky")(
            [
                delayed(_fit_discrete_single_gene)((y_binary[:, i], design, p))
                for i in range(genes)
            ]
        )

        for i, (coef, vc, fit, ll) in enumerate(results):
            coefs[i, :] = coef
            vcovs[:, :, i] = vc
            fitted[:, i] = fit
            logliks[i] = ll
    else:
        for i in range(genes):
            yi = y_binary[:, i]
            mask = ~np.isnan(yi) & (yi >= 0)

            try:
                model = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                )
                model.fit(design[mask, :], yi[mask])
                coefs[i, :] = model.coef_.flatten()

                pred_proba = model.predict_proba(design[mask, :])
                pred_proba = np.clip(pred_proba, 1e-10, 1 - 1e-10)
                hessian = (
                    design[mask, :].T
                    @ np.diag(pred_proba[:, 0] * (1 - pred_proba[:, 0]))
                    @ design[mask, :]
                )
                vcovs[:, :, i] = np.linalg.inv(hessian + np.eye(p) * 1e-6)
                fitted[:, i] = mask

                logliks[i] = np.sum(
                    yi[mask] * np.log(pred_proba[:, 1])
                    + (1 - yi[mask]) * np.log(pred_proba[:, 0])
                )
            except Exception as e:
                coefs[i, :] = np.nan
                vcovs[:, :, i] = np.nan
                fitted[:, i] = False
                logliks[i] = np.nan

    return coefs, vcovs, fitted, logliks


def zlm(
    formula: str,
    adata: Any,
    method: str = "bayesglm",
    silent: bool = True,
    ebayes: bool = True,
    ebayes_method: str = "MOM",
    only_coef: bool = False,
    layer: Optional[str] = None,
    parallel: bool = True,
    n_jobs: int = -1,
    **kwargs,
) -> ZlmFit:
    """
    Fit zero-inflated linear model for each gene.

    This fits a hurdle model: a logistic regression for expression vs no expression,
    and a linear model for the expression level when expressed.

    Parameters
    ----------
    formula : str
        Formula for the model (e.g., '~ group + batch')
    adata : AnnData
        Single-cell data (scanpy AnnData)
    method : str
        Fitting method: 'bayesglm', 'glm', 'ridge'
    silent : bool
        Suppress messages
    ebayes : bool
        Use empirical Bayes variance shrinkage
    ebayes_method : str
        Method for ebayes: 'MOM' or 'MLE'
    only_coef : bool
        Only return coefficients
    layer : str, optional
        Layer to use (default: X or .layers[0])
    parallel : bool
        Use parallel processing
    n_jobs : int
        Number of cores for parallel processing (-1 for all cores)
    **kwargs
        Additional arguments

    Returns
    -------
    ZlmFit
        Zero-inflated model fit result
    """
    import anndata

    if not isinstance(adata, anndata.AnnData):
        raise ValueError("adata must be an AnnData object")

    if layer is not None:
        if layer in adata.layers:
            exprs = adata.layers[layer]
        else:
            raise ValueError(f"Layer '{layer}' not found")
    elif adata.X is not None:
        exprs = adata.X
    elif "log1p" in adata.layers:
        exprs = adata.layers["log1p"]
    else:
        exprs = adata.layers[list(adata.layers.keys())[0]]

    if hasattr(exprs, "toarray"):
        exprs = exprs.toarray()
    elif hasattr(exprs, "todense"):
        exprs = exprs.todense()
    else:
        exprs = np.asarray(exprs)

    genes = adata.var_names.tolist()
    n_cells = adata.n_obs

    if "wellKey" in adata.obs.columns:
        cell_data = adata.obs.copy()
    else:
        cell_data = adata.obs.copy()
        cell_data["wellKey"] = cell_data.index

    design = _make_design_matrix(formula.replace("~", ""), cell_data)
    design_array = np.array(design)
    design_df = pd.DataFrame(design_array, columns=design.design_info.column_names)

    if not silent:
        print(f"Fitting zero-inflated model for {len(genes)} genes and {n_cells} cells")

    if ebayes:
        eb_result = _ebayes(exprs, np.array(design), method=ebayes_method)
        prior_var = eb_result["v"]
        prior_df = eb_result["df"]
    else:
        prior_var = np.ones(len(genes))
        prior_df = np.zeros(len(genes))

    positive = exprs > 0

    if not silent:
        print("Fitting discrete component...")

    coefD, vcovD, fittedD, loglikD = _fit_discrete(
        positive.astype(float),
        design_array,
        method=method,
        parallel=parallel,
        n_jobs=n_jobs,
        **kwargs,
    )

    if not silent:
        print("Fitting continuous component...")

    exprs_pos = exprs.copy()
    exprs_pos[~positive] = np.nan

    coefC, vcovC, fittedC, loglikC = _fit_continuous(
        exprs_pos,
        design_array,
        method=method,
        prior_var=prior_var,
        parallel=parallel,
        n_jobs=n_jobs,
        **kwargs,
    )

    df_null = np.zeros((len(genes), 2))
    df_resid = np.zeros((len(genes), 2))
    df_null[:, 0] = n_cells - 1
    df_null[:, 1] = n_cells - 1
    df_resid[:, 0] = np.sum(fittedC, axis=0) - design.shape[1]
    df_resid[:, 1] = np.sum(fittedD, axis=0) - design.shape[1]

    deviance = np.zeros((len(genes), 2))
    dispersion = np.zeros((len(genes), 2))

    for i in range(len(genes)):
        if fittedC[:, i].sum() > design.shape[1]:
            residuals = (
                exprs_pos[fittedC[:, i], i]
                - design_array[fittedC[:, i], :] @ coefC[i, :]
            )
            dispersion[i, 0] = np.var(residuals, ddof=design.shape[1])
        else:
            dispersion[i, 0] = np.nan

        if fittedD[:, i].sum() > design.shape[1]:
            dispersion[i, 1] = 1.0
        else:
            dispersion[i, 1] = np.nan

    converged = np.array([fittedC.all(axis=0), fittedD.all(axis=0)]).T

    coefC_df = pd.DataFrame(coefC, index=genes, columns=design_df.columns)
    coefD_df = pd.DataFrame(coefD, index=genes, columns=design_df.columns)

    loglik_df = pd.DataFrame({"C": loglikC, "D": loglikD}, index=genes)
    deviance_df = pd.DataFrame(deviance, index=genes, columns=["C", "D"])
    df_null_df = pd.DataFrame(df_null, index=genes, columns=["C", "D"])
    df_resid_df = pd.DataFrame(df_resid, index=genes, columns=["C", "D"])
    dispersion_df = pd.DataFrame(dispersion, index=genes, columns=["C", "D"])
    converged_df = pd.DataFrame(converged, index=genes, columns=["C", "D"])

    zfit = ZlmFit(
        coefC=coefC_df,
        coefD=coefD_df,
        vcovC=vcovC,
        vcovD=vcovD,
        design=design,
        formula=formula,
        loglik=loglik_df,
        deviance=deviance_df,
        df_null=df_null_df,
        df_resid=df_resid_df,
        dispersion=dispersion_df,
        converged=converged_df,
        adata=adata,
    )

    if not silent:
        print(f"Done! Fitted {zfit.ngenes} genes and {zfit.ncells} cells")

    return zfit
