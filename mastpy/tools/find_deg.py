"""Differential expression analysis using MASTpy with AnnData support"""

import numpy as np
import pandas as pd
from anndata import AnnData
from mastpy import SingleCellAssay, zlm
from mastpy.utils.utils import getLogFC
from statsmodels.stats.multitest import multipletests


def find_deg(
    adata,
    groupby,
    ident_1,
    ident_2=None,
    layer='counts',
    logfc_threshold=0.1,
    min_pct=0.01,
    test_use='MAST',
    test_method='wald',  # 'wald' or 'lr'
    n_jobs=10,  # Number of parallel jobs to use, set to 1 for serial processing
    only_pos=False,
    verbose=True
):
    """
    Find differentially expressed genes between two groups of cells using MAST
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell expression data
    groupby : str
        Column name in adata.obs used for grouping cells
    ident_1 : str or list
        Identity class(es) to define markers for
    ident_2 : str or list, optional
        A second identity class for comparison. If None, use all other cells
    layer : str, optional
        Layer in adata to use for expression data
    logfc_threshold : float, optional
        Limit testing to genes which show, on average, at least X-fold difference
        (log-scale) between the two groups of cells
    min_pct : float, optional
        Only test genes that are detected in a minimum fraction of min_pct cells
        in either of the two populations
    test_use : str, optional
        Denotes which test to use. Currently only 'MAST' is supported
    test_method : str, optional
        Statistical test method to use: 'wald' for Wald test or 'lr' for likelihood ratio test
    n_jobs : int, optional
        Number of parallel jobs to use for model fitting. Default is 10. Set to 1 for serial processing.
    only_pos : bool, optional
        Only return positive markers
    verbose : bool, optional
        Print progress messages
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing differentially expressed genes with statistics
    """
    # Validate input
    if test_use != 'MAST':
        raise ValueError("Currently only 'MAST' test is supported")
    
    # Extract expression matrix
    if layer in adata.layers:
        expression_matrix = adata.layers[layer]
    else:
        expression_matrix = adata.X
    
    # Convert to numpy array if it's a sparse matrix
    if hasattr(expression_matrix, 'toarray'):
        expression_matrix = expression_matrix.toarray()
    
    # Ensure expression matrix is in (n_genes, n_cells) format
    if expression_matrix.shape[0] != adata.n_vars or expression_matrix.shape[1] != adata.n_obs:
        expression_matrix = expression_matrix.T
    
    # Create cell metadata
    cdata = adata.obs.copy()
    
    # Create feature metadata
    fdata = adata.var.copy()
    
    # Create SingleCellAssay
    sca = SingleCellAssay(expression_matrix, cdata, fdata)
    
    # Build formula
    formula = f'~ {groupby}'
    
    # Fit ZLM model
    if verbose:
        print(f"Fitting ZLM model with formula: {formula}")
    
    zfit = zlm(
        formula=formula,
        sca=sca,
        method='glm',
        use_ebayes=True,
        n_jobs=n_jobs,  # Use user-specified number of parallel jobs
        silent=not verbose
    )
    
    # Get log fold changes
    logfc_results = getLogFC(zfit)
    
    # Filter results based on ident_1 and ident_2
    if isinstance(ident_1, str):
        ident_1 = [ident_1]
    
    # Get contrast name
    contrast_name = None
    for col in zfit.coefC.columns:
        if col != '(Intercept)':
            contrast_name = col
            break
    
    if contrast_name is None:
        raise ValueError("Could not find contrast name in model coefficients")
    
    # Filter logFC results for the specified contrast
    logfc_results = logfc_results[logfc_results['contrast'] == contrast_name]
    
    # Calculate p-values based on test_method
    if test_method == 'wald':
        if verbose:
            print(f"Calculating p-values using Wald test for contrast: {contrast_name}")
        
        # Use waldTest method to get p-values
        test_result = zfit.waldTest(contrast_name)
    elif test_method == 'lr':
        if verbose:
            print(f"Calculating p-values using Likelihood Ratio test for contrast: {contrast_name}")
        
        # Use lrTest method to get p-values
        test_result = zfit.lrTest(contrast_name)
    else:
        raise ValueError(f"Invalid test_method: {test_method}. Must be 'wald' or 'lr'.")
    
    # Extract p-values from the result (using hurdle component)
    # The result shape is (n_genes, 3, 3) where:
    # - dimension 1: genes
    # - dimension 2: components (0: continuous, 1: discrete, 2: hurdle)
    # - dimension 3: values (0: statistic, 1: df, 2: p-value)
    p_values = test_result[:, 2, 2]  # Use hurdle component's p-value
    
    # Perform multiple testing correction
    p_values_adj = multipletests(p_values, method='fdr_bh')[1]
    
    # Add p-values to logFC results
    logfc_results['p_val'] = p_values
    logfc_results['p_val_adj'] = p_values_adj
    
    # Calculate percentage of cells expressing each gene in each group
    if ident_2 is None:
        # Compare ident_1 vs all others
        cells_1 = adata.obs[groupby].isin(ident_1)
        cells_2 = ~cells_1
    else:
        if isinstance(ident_2, str):
            ident_2 = [ident_2]
        cells_1 = adata.obs[groupby].isin(ident_1)
        cells_2 = adata.obs[groupby].isin(ident_2)
    
    # Calculate pct.1 and pct.2
    pct_1 = np.sum(expression_matrix[:, cells_1] > 0, axis=1) / np.sum(cells_1)
    pct_2 = np.sum(expression_matrix[:, cells_2] > 0, axis=1) / np.sum(cells_2)
    
    # Create pct dataframe
    pct_df = pd.DataFrame({
        'primerid': adata.var_names,
        'pct.1': pct_1,
        'pct.2': pct_2
    })
    
    # Merge with logFC results
    results = pd.merge(logfc_results, pct_df, on='primerid', how='left')
    
    # Filter based on logfc_threshold and min_pct
    results = results[abs(results['logFC']) >= logfc_threshold]
    results = results[(results['pct.1'] >= min_pct) | (results['pct.2'] >= min_pct)]
    
    # Filter for positive markers if requested
    if only_pos:
        results = results[results['logFC'] > 0]
    
    # Sort by adjusted p-value
    results = results.sort_values('p_val_adj')
    
    # Rename columns to match Seurat's output
    results = results.rename(columns={
        'primerid': 'gene',
        'logFC': 'avg_log2FC'
    })
    
    # Reorder columns
    results = results[[
        'gene', 'avg_log2FC', 'pct.1', 'pct.2', 'p_val', 'p_val_adj'
    ]]
    
    # Set gene as index
    results = results.set_index('gene')
    
    if verbose:
        print(f"Found {len(results)} differentially expressed genes")
    
    return results


def find_all_degs(
    adata,
    groupby,
    layer='counts',
    logfc_threshold=0.1,
    min_pct=0.01,
    test_use='MAST',
    test_method='wald',  # 'wald' or 'lr'
    n_jobs=10,  # Number of parallel jobs to use, set to 1 for serial processing
    only_pos=False,
    verbose=True
):
    """
    Find markers for all identity classes in a dataset
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell expression data
    groupby : str
        Column name in adata.obs used for grouping cells
    layer : str, optional
        Layer in adata to use for expression data
    logfc_threshold : float, optional
        Limit testing to genes which show, on average, at least X-fold difference
        (log-scale) between the two groups of cells
    min_pct : float, optional
        Only test genes that are detected in a minimum fraction of min_pct cells
        in either of the two populations
    test_use : str, optional
        Denotes which test to use. Currently only 'MAST' is supported
    test_method : str, optional
        Statistical test method to use: 'wald' for Wald test or 'lr' for likelihood ratio test
    n_jobs : int, optional
        Number of parallel jobs to use for model fitting. Default is 10. Set to 1 for serial processing.
    only_pos : bool, optional
        Only return positive markers
    verbose : bool, optional
        Print progress messages
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing markers for all identity classes
    """
    # Get all unique identities
    identities = adata.obs[groupby].unique()
    
    all_markers = []
    
    for ident in identities:
        if verbose:
            print(f"Calculating markers for {ident}")
        
        # Find markers for this identity vs all others
        markers = find_deg(
            adata=adata,
            groupby=groupby,
            ident_1=ident,
            ident_2=None,
            layer=layer,
            logfc_threshold=logfc_threshold,
            min_pct=min_pct,
            test_use=test_use,
            test_method=test_method,
            n_jobs=n_jobs,
            only_pos=only_pos,
            verbose=verbose
        )
        
        # Add cluster information
        markers['cluster'] = ident
        markers['gene'] = markers.index
        
        all_markers.append(markers)
    
    # Combine all markers
    if all_markers:
        all_markers = pd.concat(all_markers)
        all_markers = all_markers.reset_index(drop=True)
        all_markers = all_markers.set_index('gene')
    else:
        all_markers = pd.DataFrame()
    
    return all_markers