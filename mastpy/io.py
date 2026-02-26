"""
Data import/export functions for MASTpy.

Provides functions to create AnnData objects from various input formats,
similar to FromMatrix and FromFlatDF in R MAST.
"""

import numpy as np
import pandas as pd
import anndata
from typing import Optional, Union, List, Dict, Any


def from_matrix(
    exprs: Union[np.ndarray, pd.DataFrame],
    c_data: Optional[pd.DataFrame] = None,
    f_data: Optional[pd.DataFrame] = None,
    obs: Optional[pd.DataFrame] = None,
    var: Optional[pd.DataFrame] = None,
    X: Optional[np.ndarray] = None,
    check_sanity: bool = True,
) -> anndata.AnnData:
    """
    Create AnnData from expression matrix.

    Similar to R's FromMatrix function.

    Parameters
    ----------
    exprs : np.ndarray or pd.DataFrame
        Expression matrix (genes x cells) or (cells x genes)
    c_data : pd.DataFrame, optional
        Cell metadata (colData)
    f_data : pd.DataFrame, optional
        Feature/gene metadata (rowData)
    obs : pd.DataFrame, optional
        Alias for c_data
    var : pd.DataFrame, optional
        Alias for f_data
    X : np.ndarray, optional
        Alias for exprs
    check_sanity : bool
        Check if data appears log-transformed

    Returns
    -------
    anndata.AnnData
        AnnData object
    """
    if X is not None:
        exprs = X
    if obs is not None:
        c_data = obs
    if var is not None:
        f_data = var

    if isinstance(exprs, pd.DataFrame):
        X = exprs.values
        if f_data is None:
            f_data = pd.DataFrame(index=exprs.index)
        if c_data is None:
            c_data = pd.DataFrame(index=exprs.columns)
    else:
        X = exprs

    n_genes, n_cells = X.shape

    if f_data is None:
        f_data = pd.DataFrame(
            {"primerid": [f"gene_{i}" for i in range(n_genes)]},
            index=[f"gene_{i}" for i in range(n_genes)],
        )
    else:
        if "primerid" not in f_data.columns:
            f_data["primerid"] = f_data.index

    if c_data is None:
        c_data = pd.DataFrame(
            {"wellKey": [f"cell_{i}" for i in range(n_cells)]},
            index=[f"cell_{i}" for i in range(n_cells)],
        )
    else:
        if "wellKey" not in c_data.columns:
            c_data["wellKey"] = c_data.index

    if check_sanity:
        _sanity_check(X)

    adata = anndata.AnnData(
        X=X.T,
        obs=c_data,
        var=f_data,
    )

    return adata


def _sanity_check(X: np.ndarray) -> None:
    """Check if data appears log-transformed."""
    import warnings

    X_flat = X.flatten()
    X_nonzero = X_flat[X_flat != 0]

    if len(X_nonzero) == 0:
        warnings.warn("All expression values are zero")
        return

    max_val = np.max(X_nonzero)

    if max_val > 100:
        warnings.warn(
            "Maximum expression value > 100. Data may not be log-transformed. "
            "Set check_sanity=False to override."
        )

    noninteger_fraction = np.mean(X_nonzero != np.floor(X_nonzero))
    if noninteger_fraction > 0.1 and max_val < 50:
        warnings.warn(
            "Most values appear non-integer. Data may not be log-transformed. "
            "Set check_sanity=False to override."
        )


def from_flat_df(
    df: pd.DataFrame,
    id_vars: List[str],
    primerid: str,
    measurement: str,
    cell_vars: Optional[List[str]] = None,
    feature_vars: Optional[List[str]] = None,
    phenovars: Optional[List[str]] = None,
    check_sanity: bool = True,
) -> anndata.AnnData:
    """
    Create AnnData from flat (melted) dataframe.

    Similar to R's FromFlatDF function.

    Parameters
    ----------
    df : pd.DataFrame
        Flat dataframe with columns for cell ID, gene ID, and expression
    id_vars : List[str]
        Columns that uniquely identify a cell
    primerid : str
        Column identifying the gene/feature
    measurement : str
        Column containing the expression measurement
    cell_vars : List[str], optional
        Additional cell metadata columns
    feature_vars : List[str], optional
        Additional feature metadata columns
    phenovars : List[str], optional
        Phenotype variables (alias for cell_vars)
    check_sanity : bool
        Check if data appears log-transformed

    Returns
    -------
    anndata.AnnData
        AnnData object
    """
    if phenovars is not None:
        if cell_vars is None:
            cell_vars = []
        cell_vars = list(cell_vars) + list(phenovars)

    df = df.copy()

    df["wellKey"] = df[id_vars].astype(str).agg("_".join, axis=1)

    if "primerid" not in df.columns:
        df["primerid"] = df[primerid]

    pivot_df = df.pivot_table(
        index="primerid", columns="wellKey", values=measurement, aggfunc="first"
    )

    gene_metadata_cols = []
    if feature_vars:
        gene_metadata_cols = [c for c in feature_vars if c in df.columns]
        gene_metadata = df.groupby("primerid")[gene_metadata_cols].first()
        gene_metadata = gene_metadata.loc[pivot_df.index]
    else:
        gene_metadata = pd.DataFrame(index=pivot_df.index)

    cell_metadata_cols = []
    if cell_vars:
        cell_metadata_cols = [c for c in cell_vars if c in df.columns]
        cell_metadata = df.groupby("wellKey")[cell_metadata_cols].first()
        cell_metadata = cell_metadata.loc[pivot_df.columns]
    else:
        cell_metadata = pd.DataFrame(index=pivot_df.columns)

    if check_sanity:
        _sanity_check(pivot_df.values)

    adata = anndata.AnnData(
        X=pivot_df.values,
        obs=cell_metadata,
        var=gene_metadata,
    )

    return adata


def melt(
    adata: anndata.AnnData, layer: Optional[str] = None, value_name: str = "value"
) -> pd.DataFrame:
    """
    Melt AnnData to long format dataframe.

    Similar to R's melt.SingleCellAssay function.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
    layer : str, optional
        Layer to use (default: X)
    value_name : str
        Name for the value column

    Returns
    -------
    pd.DataFrame
        Long format dataframe
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers.get(layer, adata.X)

    if hasattr(X, "toarray"):
        X = X.toarray()

    genes = adata.var_names.tolist()
    cells = adata.obs_names.tolist()

    melted = pd.DataFrame(
        X.T,
        index=pd.MultiIndex.from_product([genes, cells], names=["primerid", "wellKey"]),
        columns=[value_name],
    ).reset_index()

    obs_df = adata.obs.reset_index()
    obs_df.columns = ["wellKey"] + list(obs_df.columns[1:])

    var_df = adata.var.reset_index()
    var_df.columns = ["primerid"] + list(var_df.columns[1:])

    merged = melted.merge(obs_df, on="wellKey", how="left")
    merged = merged.merge(var_df, on="primerid", how="left")

    return merged


def read_fluidigm(
    filename: str,
    plate: Optional[str] = None,
    metadata_cols: Optional[List[str]] = None,
) -> anndata.AnnData:
    """
    Read Fluidigm single-cell data.

    Parameters
    ----------
    filename : str
        Path to the data file
    plate : str, optional
        Plate ID column name
    metadata_cols : List[str], optional
        Columns containing metadata

    Returns
    -------
    anndata.AnnData
        AnnData object
    """
    import warnings

    warnings.warn("read_fluidigm is not fully implemented yet")

    df = pd.read_csv(filename, index_col=0)

    if metadata_cols is None:
        metadata_cols = ["well", "chip", "ncells"]

    return from_matrix(df, check_sanity=False)
