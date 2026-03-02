import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from .zlm import zlm
from .single_cell_assay import SingleCellAssay

def bootVcov1(zlmfit, R=99, boot_index=None):
    """
    Bootstrap a zlmfit
    
    Sample cells with replacement to find bootstrapped distribution of coefficients
    
    Parameters
    ----------
    zlmfit : ZlmFit
        Fitted ZlmFit object
    R : int, optional
        Number of bootstrap replicates
    boot_index : list, optional
        List of indices to resample
    
    Returns
    -------
    numpy.ndarray
        Array of bootstrapped coefficients with dimensions (R, n_genes, n_coefs, 2)
        where 2 corresponds to C and D components
    """
    sca = zlmfit.sca
    N = sca.ncol()
    LMlike = zlmfit.LMlike
    
    if boot_index is None:
        # Generate bootstrap indices
        boot_index = [np.random.choice(N, size=N, replace=True) for _ in range(R)]
    else:
        # Validate boot_index
        r = (min(min(idx) for idx in boot_index), max(max(idx) for idx in boot_index))
        if r[0] < 0 or r[1] >= N:
            raise ValueError(f"boot_index must be a list of integer vectors between 0 and {N-1}")
    
    # Get expression matrix and design
    expression_matrix = sca.assay()
    cdata = sca.colData()
    fdata = sca.mcols()
    
    # Fit all bootstrap samples
    bootstrap_results = []
    for s in tqdm(boot_index, desc="Bootstrap replicates"):
        # Create bootstrapped SingleCellAssay
        boot_expression = expression_matrix[:, s]
        boot_cdata = cdata.iloc[s, :].reset_index(drop=True)
        boot_sca = SingleCellAssay(boot_expression, boot_cdata, fdata)
        
        # Fit model with same parameters
        boot_fit = zlm(
            formula=LMlike.formula,
            sca=boot_sca,
            method=type(LMlike).__name__.replace('like', '').lower(),
            silent=True,
            use_ebayes=False,  # Use same prior as original fit
            n_jobs=1
        )
        
        # Extract coefficients
        coefC = boot_fit.coefC.values
        coefD = boot_fit.coefD.values
        
        # Combine into a single array
        coefs = np.stack([coefC, coefD], axis=2)
        bootstrap_results.append(coefs)
    
    # Combine all bootstrap results
    bootstrap_array = np.array(bootstrap_results)
    
    return bootstrap_array

def pbootVcov1(cl, zlmfit, R=99):
    """
    Parallel version of bootVcov1
    
    Parameters
    ----------
    cl : multiprocessing.Pool
        Pool of worker processes
    zlmfit : ZlmFit
        Fitted ZlmFit object
    R : int, optional
        Number of bootstrap replicates
    
    Returns
    -------
    numpy.ndarray
        Array of bootstrapped coefficients
    """
    sca = zlmfit.sca
    N = sca.ncol()
    
    # Generate bootstrap indices
    boot_index = [np.random.choice(N, size=N, replace=True) for _ in range(R)]
    
    # Create argument tuples
    args = [(zlmfit, idx) for idx in boot_index]
    
    # Run in parallel
    bootstrap_results = list(tqdm(cl.starmap(_bootstrap_worker, args), total=R, desc="Parallel bootstrap"))
    
    # Combine results
    bootstrap_array = np.array(bootstrap_results)
    
    return bootstrap_array

def _bootstrap_worker(zlmfit, idx):
    """
    Worker function for parallel bootstrap
    
    Parameters
    ----------
    zlmfit : ZlmFit
        Fitted ZlmFit object
    idx : numpy.ndarray
        Bootstrap indices
    
    Returns
    -------
    numpy.ndarray
        Bootstrapped coefficients
    """
    sca = zlmfit.sca
    LMlike = zlmfit.LMlike
    
    # Create bootstrapped SingleCellAssay
    expression_matrix = sca.assay()
    cdata = sca.colData()
    fdata = sca.mcols()
    
    boot_expression = expression_matrix[:, idx]
    boot_cdata = cdata.iloc[idx, :].reset_index(drop=True)
    boot_sca = SingleCellAssay(boot_expression, boot_cdata, fdata)
    
    # Fit model
    boot_fit = zlm(
        formula=LMlike.formula,
        sca=boot_sca,
        method=type(LMlike).__name__.replace('like', '').lower(),
        silent=True,
        use_ebayes=False,
        n_jobs=1
    )
    
    # Extract coefficients
    coefC = boot_fit.coefC.values
    coefD = boot_fit.coefD.values
    
    # Combine into a single array
    coefs = np.stack([coefC, coefD], axis=2)
    
    return coefs

def CovFromBoots(boots, coefficient):
    """
    Extract the inter-gene covariance matrices for continuous and discrete components
    
    Parameters
    ----------
    boots : numpy.ndarray
        Bootstrap results from bootVcov1 or pbootVcov1
    coefficient : str
        Name of the model coefficient
    
    Returns
    -------
    dict
        Dictionary with components "C" and "D" containing covariance matrices
    """
    # Get coefficient index
    # Assuming boots has shape (R, n_genes, n_coefs, 2)
    n_genes = boots.shape[1]
    
    # Find coefficient index
    # This requires knowing the coefficient names, which we don't have
    # For now, assume coefficient is the first non-intercept coefficient
    # TODO: Fix this to use actual coefficient names
    coef_idx = 1  # Assuming (Intercept) is first
    
    # Extract bootstrap samples for the coefficient
    bootstat = boots[:, :, coef_idx, :]
    
    # Calculate means
    bmean = np.mean(bootstat, axis=0)
    
    # Center the bootstrap samples
    centered_boots = bootstat - bmean
    
    # Calculate covariance matrices
    covariances = {}
    for i, comp in enumerate(['C', 'D']):
        # Extract component
        comp_boots = centered_boots[:, :, i]
        
        # Calculate covariance matrix
        cov = np.cov(comp_boots.T)
        covariances[comp] = cov
    
    return covariances