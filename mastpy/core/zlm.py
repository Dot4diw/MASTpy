import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from .lm_wrapper import GLMlike, BayesGLMlike, LMERlike
from ..utils.utils import ebayes
from numba import jit

method_dict = {
    'glm': 'GLMlike',
    'glmer': 'LMERlike',
    'lmer': 'LMERlike',
    'bayesglm': 'BayesGLMlike',
    'ridge': 'RidgeBGLMlike',
    'blmer': 'bLMERlike'
}

implements_ebayes = {
    'GLMlike': True,
    'LMERlike': False,
    'BayesGLMlike': True,
    'RidgeBGLMlike': True,
    'bLMERlike': False
}

class ZlmFit:
    def __init__(self, coefC, coefD, vcovC, vcovD, LMlike, sca, deviance, loglik, df_null, df_resid, dispersion, dispersionNoshrink, priorDOF, priorVar, converged, hookOut, exprs_values):
        """
        Initialize a ZlmFit object
        
        Parameters
        ----------
        coefC : pandas.DataFrame
            Continuous coefficients
        coefD : pandas.DataFrame
            Discrete coefficients
        vcovC : numpy.ndarray
            Variance-covariance matrix for continuous coefficients
        vcovD : numpy.ndarray
            Variance-covariance matrix for discrete coefficients
        LMlike : LMlike
            LMlike object
        sca : SingleCellAssay
            SingleCellAssay object
        deviance : pandas.DataFrame
            Deviance
        loglik : pandas.DataFrame
            Log likelihood
        df_null : pandas.DataFrame
            Null degrees of freedom
        df_resid : pandas.DataFrame
            Residual degrees of freedom
        dispersion : pandas.DataFrame
            Dispersion after shrinkage
        dispersionNoshrink : pandas.DataFrame
            Dispersion before shrinkage
        priorDOF : float
            Prior degrees of freedom
        priorVar : float
            Prior variance
        converged : pandas.DataFrame
            Convergence status
        hookOut : list
            Hook output
        exprs_values : str or int
            Assay used
        """
        self.coefC = coefC
        self.coefD = coefD
        self.vcovC = vcovC
        self.vcovD = vcovD
        self.LMlike = LMlike
        self.sca = sca
        self.deviance = deviance
        self.loglik = loglik
        self.df_null = df_null
        self.df_resid = df_resid
        self.dispersion = dispersion
        self.dispersionNoshrink = dispersionNoshrink
        self.priorDOF = priorDOF
        self.priorVar = priorVar
        self.converged = converged
        self.hookOut = hookOut
        self.exprs_values = exprs_values
    
    def coef(self, which):
        """
        Get coefficients
        
        Parameters
        ----------
        which : str
            Component ('C' for continuous, 'D' for discrete)
        
        Returns
        -------
        pandas.DataFrame
            Coefficients
        """
        if which == 'C':
            return self.coefC
        elif which == 'D':
            return self.coefD
        else:
            raise ValueError("which must be 'C' or 'D'")
    
    def vcov(self, which):
        """
        Get variance-covariance matrix
        
        Parameters
        ----------
        which : str
            Component ('C' for continuous, 'D' for discrete)
        
        Returns
        -------
        numpy.ndarray
            Variance-covariance matrix
        """
        if which == 'C':
            return self.vcovC
        elif which == 'D':
            return self.vcovD
        else:
            raise ValueError("which must be 'C' or 'D'")
    
    def lrTest(self, hypothesis):
        """
        Likelihood ratio test
        
        Parameters
        ----------
        hypothesis : Hypothesis or str
            Hypothesis to test
        
        Returns
        -------
        numpy.ndarray
            Test results
        """
        # Simplified implementation
        # In practice, you would implement the full likelihood ratio test
        pass
    
    def waldTest(self, hypothesis):
        """
        Wald test
        
        Parameters
        ----------
        hypothesis : Hypothesis or str
            Hypothesis to test
        
        Returns
        -------
        numpy.ndarray
            Test results
        """
        # Simplified implementation
        # In practice, you would implement the full Wald test
        pass

def zlm(formula, sca, method='bayesglm', silent=True, use_ebayes=True, ebayesControl=None, force=False, hook=None, parallel=True, LMlike=None, onlyCoef=False, exprs_values=None):
    """
    Zero-inflated regression for SingleCellAssay
    
    Parameters
    ----------
    formula : str
        Formula for the regression
    sca : SingleCellAssay
        SingleCellAssay object
    method : str, optional
        Method to use for fitting
    silent : bool, optional
        Silence warnings
    ebayes : bool, optional
        Use empirical Bayes variance shrinkage
    ebayesControl : dict, optional
        Control parameters for empirical Bayes
    force : bool, optional
        Continue fitting even after many errors
    hook : function, optional
        Function to call after each gene
    parallel : bool, optional
        Use parallel processing
    LMlike : LMlike, optional
        LMlike object to use
    onlyCoef : bool, optional
        Only return coefficients
    exprs_values : str or int, optional
        Assay to use
    
    Returns
    -------
    ZlmFit
        Fitted ZlmFit object
    """
    # Get expression matrix
    ee = sca.assay().T  # Transpose to (n_cells, n_genes)
    genes = sca.mcols().index if not sca.mcols().empty else [f'gene_{i}' for i in range(ee.shape[1])]
    ng = ee.shape[1]
    
    # Get design matrix
    design = sca.colData()
    
    # Create LMlike object if not provided
    if LMlike is None:
        # Get method class
        if method not in method_dict:
            raise ValueError(f"Unknown method: {method}")
        method_class = method_dict[method]
        
        # Check if method implements empirical Bayes
        if use_ebayes and not implements_ebayes[method_class]:
            raise ValueError(f"Method {method} does not implement empirical Bayes variance shrinkage")
        
        # Calculate empirical Bayes parameters
        prior_var = 1
        prior_df = 0
        if use_ebayes:
            # Build model matrix for ebayes
            import patsy
            model_matrix = patsy.dmatrix(formula, design, return_type='dataframe')
            ebparm = ebayes(ee, ebayesControl, model_matrix)
            prior_var = ebparm['v']
            prior_df = ebparm['df']
        
        # Create LMlike object
        if method_class == 'GLMlike':
            obj = GLMlike(formula, design, prior_var=prior_var, prior_df=prior_df)
        elif method_class == 'BayesGLMlike':
            obj = BayesGLMlike(formula, design, prior_var=prior_var, prior_df=prior_df)
        elif method_class == 'LMERlike':
            obj = LMERlike(formula, design, prior_var=prior_var, prior_df=prior_df)
        else:
            raise ValueError(f"Unknown method class: {method_class}")
    else:
        obj = LMlike
    
    # Get model matrix columns
    coef_names = obj.model_matrix.columns
    
    # Fit all genes
    if parallel and mp.cpu_count() > 1:
        # Use multiprocessing
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Create argument tuples for each gene
            args = [(idx, obj, ee, silent, hook) for idx in range(ng)]
            results = list(tqdm(pool.starmap(fit_gene, args), total=ng, desc="Fitting genes"))
    else:
        # Use sequential processing
        results = [fit_gene(idx, obj, ee, silent, hook) for idx in tqdm(range(ng), desc="Fitting genes")]
    
    # Separate summaries and hookOut
    list_of_summaries, hookOut = zip(*results)
    
    # Collect summaries
    coefC = pd.DataFrame([s['coefC'] for s in list_of_summaries], index=genes, columns=coef_names)
    coefD = pd.DataFrame([s['coefD'] for s in list_of_summaries], index=genes, columns=coef_names)
    
    # Reshape variance-covariance matrices
    vcovC = np.array([s['vcovC'] for s in list_of_summaries])
    vcovD = np.array([s['vcovD'] for s in list_of_summaries])
    vcovC = np.transpose(vcovC, (1, 2, 0))
    vcovD = np.transpose(vcovD, (1, 2, 0))
    
    # Other summaries
    df_resid = pd.DataFrame([s['df.resid'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    df_null = pd.DataFrame([s['df.null'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    deviance = pd.DataFrame([s['deviance'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    dispersion = pd.DataFrame([s['dispersion'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    dispersionNoshrink = pd.DataFrame([s['dispersionNoshrink'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    converged = pd.DataFrame([s['converged'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    loglik = pd.DataFrame([s['loglik'] for s in list_of_summaries], index=genes, columns=['C', 'D'])
    
    # Create ZlmFit object
    zfit = ZlmFit(
        coefC=coefC,
        coefD=coefD,
        vcovC=vcovC,
        vcovD=vcovD,
        LMlike=obj,
        sca=sca,
        deviance=deviance,
        loglik=loglik,
        df_null=df_null,
        df_resid=df_resid,
        dispersion=dispersion,
        dispersionNoshrink=dispersionNoshrink,
        priorDOF=prior_df,
        priorVar=prior_var,
        converged=converged,
        hookOut=hookOut,
        exprs_values=exprs_values
    )
    
    return zfit

# Function to fit a single gene
def fit_gene(idx, obj, ee, silent, hook):
    try:
        # Fit the gene
        gene_obj = obj.fit(ee[:, idx], silent=silent, quick=True)
        
        # Check if fitted
        if not gene_obj.fitted['C'] or not gene_obj.fitted['D']:
            if not silent:
                print(f"Gene {idx} not fully fitted: C={gene_obj.fitted['C']}, D={gene_obj.fitted['D']}")
        
        # Get summaries
        summaries = {
            'coefC': gene_obj.coef('C'),
            'coefD': gene_obj.coef('D'),
            'vcovC': gene_obj.vcov('C'),
            'vcovD': gene_obj.vcov('D'),
            'df.resid': np.array([gene_obj.fitC.df_residual if gene_obj.fitted['C'] else 0, 
                                 gene_obj.fitD.df_residual if gene_obj.fitted['D'] else 0]),
            'df.null': np.array([gene_obj.fitC.df_null if gene_obj.fitted['C'] else 0, 
                                gene_obj.fitD.df_null if gene_obj.fitted['D'] else 0]),
            'deviance': np.array([gene_obj.fitC.deviance if gene_obj.fitted['C'] else 0, 
                                 gene_obj.fitD.deviance if gene_obj.fitted['D'] else 0]),
            'dispersion': np.array([gene_obj.fitC.dispersionMLE if gene_obj.fitted['C'] else 0, 
                                   gene_obj.fitD.dispersion if gene_obj.fitted['D'] else 0]),
            'dispersionNoshrink': np.array([gene_obj.fitC.dispersionMLENoShrink if gene_obj.fitted['C'] else 0, 
                                           gene_obj.fitD.dispersion if gene_obj.fitted['D'] else 0]),
            'converged': np.array([gene_obj.fitted['C'], gene_obj.fitted['D']]),
            'loglik': gene_obj.logLik()
        }
        
        # Call hook if provided
        hookOut = None
        if hook is not None:
            hookOut = hook(gene_obj)
        
        return (summaries, hookOut)
    except Exception as e:
        if not silent:
            print(f"Error fitting gene {idx}: {e}")
        # Return empty summaries
        n_coef = obj.model_matrix.shape[1]
        return ({
            'coefC': np.zeros(n_coef),
            'coefD': np.zeros(n_coef),
            'vcovC': np.zeros((n_coef, n_coef)),
            'vcovD': np.zeros((n_coef, n_coef)),
            'df.resid': np.zeros(2),
            'df.null': np.zeros(2),
            'deviance': np.zeros(2),
            'dispersion': np.zeros(2),
            'dispersionNoshrink': np.zeros(2),
            'converged': np.zeros(2, dtype=bool),
            'loglik': (0.0, 0.0)
        }, None)