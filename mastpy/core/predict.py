import numpy as np
import pandas as pd
from .zlm import ZlmFit

def predict_ZlmFit(object, newdata=None, modelmatrix=None):
    """
    Return predictions from a ZlmFit object
    
    Parameters
    ----------
    object : ZlmFit
        Fitted ZlmFit object
    newdata : pandas.DataFrame, optional
        New data to predict from
    modelmatrix : pandas.DataFrame, optional
        Model matrix specifying the linear combination of coefficients
    ... : additional arguments
        Ignored
    
    Returns
    -------
    pandas.DataFrame
        Predictions with columns: muC, etaD, seC, seD, sample, primerid
    """
    if modelmatrix is None:
        modelmatrix = object.LMlike.model_matrix
    
    if newdata is not None:
        raise NotImplementedError('Currently not implemented; supply `modelmatrix` instead')
    
    # Get coefficients
    coefC = object.coefC
    coefD = object.coefD
    
    # Check if model matrix columns match coefficient names
    coef_names = coefC.columns
    if modelmatrix.columns.tolist() != coef_names.tolist():
        raise ValueError("Model matrix columns must match coefficient names")
    
    # Get variance-covariance matrices
    vcovC = object.vcovC
    vcovD = object.vcovD
    
    # Set sample names if not provided
    if modelmatrix.index is None:
        modelmatrix.index = range(len(modelmatrix))
    
    # Calculate predictions
    X = modelmatrix.values
    
    # Fitted values
    predC = X @ coefC.values.T
    predD = X @ coefD.values.T
    
    # Calculate standard errors
    def diag_safeContrastQF(X, vc):
        """Calculate diagonal of (X * vc * X^T)"""
        return np.sum(X * (X @ vc), axis=1)
    
    # Variance of predictions
    seC = np.zeros((len(X), len(coefC)))
    seD = np.zeros((len(X), len(coefD)))
    
    for i, gene in enumerate(coefC.index):
        # Continuous component
        vcC = vcovC[:, :, i]
        seC[:, i] = np.sqrt(diag_safeContrastQF(X, vcC))
        
        # Discrete component
        vcD = vcovD[:, :, i]
        seD[:, i] = np.sqrt(diag_safeContrastQF(X, vcD))
    
    # Reshape and create DataFrame
    samples = modelmatrix.index.tolist()
    genes = coefC.index.tolist()
    
    # Create long format DataFrame
    rows = []
    for i, sample in enumerate(samples):
        for j, gene in enumerate(genes):
            rows.append({
                'muC': predC[i, j],
                'etaD': predD[i, j],
                'seC': seC[i, j],
                'seD': seD[i, j],
                'sample': sample,
                'primerid': gene
            })
    
    return pd.DataFrame(rows)

def impute(object, groupby):
    """
    Impute missing continuous expression for plotting
    
    If there are no positive observations for a contrast, it is generally not estimible.
    However, for the purposes of testing we can replace it with the least favorable value
    with respect to the contrasts that are defined.
    
    Parameters
    ----------
    object : pandas.DataFrame
        Output of predict_ZlmFit
    groupby : str or list
        Variables to group by for imputation
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed values
    """
    df = object.copy()
    
    # Check for missing values
    df['missing'] = df['muC'].isna() & df['etaD'].notna()
    
    # Group by specified variables
    if isinstance(groupby, str):
        groupby = [groupby]
    
    # Impute missing values
    for _, group in df.groupby(groupby):
        if group['missing'].any():
            # Calculate mean for non-missing values
            mean_muC = group.loc[~group['missing'], 'muC'].mean()
            
            # Impute missing muC with mean
            df.loc[group.index[group['missing']], 'muC'] = mean_muC
            
            # Impute missing seC with max absolute deviation
            if not group['seC'].isna().all():
                max_seC = group['seC'].max()
                df.loc[group.index[group['missing']], 'seC'] = max_seC
    
    # Drop missing indicator
    df.drop('missing', axis=1, inplace=True)
    
    # Remove any remaining NA values
    df = df.dropna()
    
    return df

# Add predict method to ZlmFit class
def add_predict_method():
    """
    Add predict method to ZlmFit class
    """
    ZlmFit.predict = predict_ZlmFit