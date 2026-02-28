import numpy as np
import pandas as pd
import scipy
from scipy import stats
from numba import jit

@jit(nopython=True, parallel=True)
def calculate_variance(data):
    """
    Calculate variance for each gene
    
    Parameters
    ----------
    data : numpy.ndarray
        Gene expression matrix with shape (n_cells, n_genes)
    
    Returns
    -------
    numpy.ndarray
        Variance for each gene
    """
    n_cells = data.shape[0]
    n_genes = data.shape[1]
    variances = np.zeros(n_genes)
    
    for i in range(n_genes):
        gene_data = data[:, i]
        mean = np.mean(gene_data)
        var = np.sum((gene_data - mean) ** 2) / (n_cells - 1)
        variances[i] = var
    
    return variances

def getSSg_rNg(assay_t, mm):
    """
    Calculate SSg and rNg for each gene
    
    Parameters
    ----------
    assay_t : numpy.ndarray
        Transposed assay matrix (cells x genes)
    mm : pandas.DataFrame
        Model matrix
    
    Returns
    -------
    numpy.ndarray
        Array with SSg and rNg for each gene
    """
    n_genes = assay_t.shape[1]
    results = np.zeros((n_genes, 2))
    
    for i in range(n_genes):
        y = assay_t[:, i]
        yp = y[~np.isnan(y)]
        if len(yp) == 0:
            results[i, :] = [np.nan, np.nan]
            continue
        
        # Get model matrix for non-NA values
        mp = mm.values[~np.isnan(y), :]
        
        # QR decomposition
        QR = np.linalg.qr(mp)
        rank = np.linalg.matrix_rank(mp)
        
        # Calculate residuals
        resid = yp - mp @ np.linalg.lstsq(mp, yp, rcond=None)[0]
        SSg = np.sum(resid ** 2)
        rNg = len(yp) - rank
        
        results[i, :] = [SSg, rNg]
    
    return results

def solveMoM(rNg, SSg):
    """
    Method of moments estimation for gamma distribution parameters
    
    Parameters
    ----------
    rNg : numpy.ndarray
        Residual degrees of freedom
    SSg : numpy.ndarray
        Residual sum of squares
    
    Returns
    -------
    tuple
        (a0, b0) parameters
    """
    rbar = np.mean(SSg / rNg)
    rbarbar = np.mean(SSg**2 / (rNg * (rNg + 2)))
    
    def a0mom(a0):
        return (2 * (a0 - 1)**2 * rbar**2 - rbarbar**2 * ((a0 - 2) * (a0 - 4)))**2
    
    # Find minimum using brute force search
    a0_values = np.linspace(0.1, 10, 1000)
    a0mom_values = [a0mom(a0) for a0 in a0_values]
    a0 = a0_values[np.argmin(a0mom_values)]
    b0 = (a0 - 1) * rbar
    
    return a0, b0

def getMarginalHyperLikelihood(rNg, SSg, deriv=False):
    """
    Calculate marginal hyperlikelihood
    
    Parameters
    ----------
    rNg : numpy.ndarray
        Residual degrees of freedom
    SSg : numpy.ndarray
        Residual sum of squares
    deriv : bool
        Calculate derivative
    
    Returns
    -------
    function
        Function to calculate hyperlikelihood or its derivative
    """
    if not deriv:
        def fun(theta):
            a0, b0 = theta
            Li = -scipy.special.betaln(rNg/2, a0) - rNg/2 * np.log(b0) - np.log(1 + SSg/(2*b0)) * (rNg/2 + a0)
            return np.sum(Li)
    else:
        def fun(theta):
            a0, b0 = theta
            score_a0_i = scipy.special.digamma(rNg/2 + a0) - scipy.special.digamma(a0) - np.log(1 + SSg/(2*b0))
            score_b0_i = (a0 * SSg - rNg * b0) / (SSg * b0 + 2 * b0**2)
            return np.array([np.sum(score_a0_i), np.sum(score_b0_i)])
    return fun

def ebayes(data, ebayes_control=None, design_matrix=None, truncate=np.inf):
    """
    Empirical Bayes variance shrinkage
    
    Parameters
    ----------
    data : numpy.ndarray
        Gene expression matrix with shape (n_cells, n_genes)
    ebayes_control : dict, optional
        Control parameters for empirical Bayes
    design_matrix : pandas.DataFrame, optional
        Design matrix
    truncate : float
        Genes with sample precisions exceeding this value are discarded
    
    Returns
    -------
    dict
        Dictionary with prior variance and degrees of freedom
    """
    import scipy.optimize
    import scipy.special
    
    # Default parameters
    default_ctl = {'method': 'MLE', 'model': 'H0'}
    if ebayes_control is None:
        ebayes_control = default_ctl
    else:
        # Fill missing parameters
        for key, value in default_ctl.items():
            if key not in ebayes_control:
                ebayes_control[key] = value
    
    method = ebayes_control['method']
    model = ebayes_control['model']
    
    # Set zeros to NA
    assay_t = data.copy().astype(float)
    assay_t[assay_t == 0] = np.nan
    
    if model == 'H0':
        # Center data
        assay_t = assay_t - np.nanmean(assay_t, axis=0)
        # Calculate rNg and SSg
        rNg = np.sum(~np.isnan(assay_t), axis=0) - 1
        SSg = np.nansum(assay_t**2, axis=0)
    elif model == 'H1':
        if design_matrix is None:
            raise ValueError("design_matrix is required for model='H1'")
        # Calculate SSg and rNg using model matrix
        allfits = getSSg_rNg(assay_t, design_matrix)
        SSg = allfits[:, 0]
        rNg = allfits[:, 1]
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Filter valid values
    valid = (rNg > 0) & (rNg / SSg < truncate) & (~np.isnan(rNg)) & (~np.isnan(SSg))
    rNg = rNg[valid]
    SSg = SSg[valid]
    
    if len(rNg) == 0:
        return {'v': 1.0, 'df': 0.0}
    
    if method == 'MLE':
        # Maximum likelihood estimation
        fn = getMarginalHyperLikelihood(rNg, SSg, deriv=False)
        grad = getMarginalHyperLikelihood(rNg, SSg, deriv=True)
        
        # Optimize
        O = scipy.optimize.minimize(fn, [1, 1], jac=grad, method='L-BFGS-B', 
                                   bounds=[(0.001, np.inf), (0.001, np.inf)],
                                   options={'maxiter': 1000})
        
        if O.success:
            a0, b0 = O.x
        else:
            # Fall back to MOM if MLE fails
            a0, b0 = solveMoM(rNg, SSg)
    elif method == 'MOM':
        # Method of moments estimation
        a0, b0 = solveMoM(rNg, SSg)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate prior variance and degrees of freedom
    v = max(b0 / a0, 0)
    df = max(2 * a0, 0)
    
    return {'v': v, 'df': df}

def getLogFC(zlmfit):
    """
    Calculate log fold changes
    
    Parameters
    ----------
    zlmfit : ZlmFit
        Fitted ZlmFit object
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with log fold changes
    """
    # Extract coefficients
    coefC = zlmfit.coefC
    coefD = zlmfit.coefD
    
    # Get contrast names
    contrast_names = coefC.columns
    
    # Calculate log fold changes
    logFC = []
    for contrast in contrast_names:
        if contrast == '(Intercept)':
            continue
        
        # For each gene, calculate log fold change
        for i, gene in enumerate(coefC.index):
            lfc = coefC.loc[gene, contrast]
            var_lfc = zlmfit.vcovC[:, :, i][contrast_names.get_loc(contrast), contrast_names.get_loc(contrast)]
            
            logFC.append({
                'primerid': gene,
                'contrast': contrast,
                'logFC': lfc,
                'varLogFC': var_lfc
            })
    
    return pd.DataFrame(logFC)