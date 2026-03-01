import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from .lm_wrapper import GLMlike, BayesGLMlike, LMERlike
from ..utils.utils import ebayes
from scipy.stats import chi2
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
        hypothesis : Hypothesis, CoefficientHypothesis, str, or matrix
            Hypothesis to test
        
        Returns
        -------
        numpy.ndarray
            Test results with dimensions (n_genes, 3, 3) where:
            - First dimension: genes
            - Second dimension: test components (cont, disc, hurdle)
            - Third dimension: metrics (lambda, df, p.value)
        """
        from .hypothesis import Hypothesis, CoefficientHypothesis
        
        # Get design matrix from the original model
        design = self.sca.colData()
        
        # Create reduced model matrix
        if isinstance(hypothesis, str):
            # For string hypothesis, create a reduced model with only intercept
            # This works for any single contrast hypothesis
            n = len(design)
            intercept = np.ones(n)
            reduced_model_matrix = pd.DataFrame(intercept, columns=['(Intercept)'])
        elif isinstance(hypothesis, CoefficientHypothesis):
            # For CoefficientHypothesis, create a reduced model with only intercept
            n = len(design)
            intercept = np.ones(n)
            reduced_model_matrix = pd.DataFrame(intercept, columns=['(Intercept)'])
        elif isinstance(hypothesis, Hypothesis):
            # For Hypothesis, use its contrast matrix
            # Note: This requires implementing generateHypothesis method
            # For now, we'll raise an error
            raise NotImplementedError("Hypothesis class support not fully implemented")
        elif isinstance(hypothesis, np.ndarray):
            # For matrix hypothesis, we need to rotate the model matrix
            # This is more complex and not implemented here
            raise NotImplementedError("Matrix hypothesis support not fully implemented")
        else:
            raise TypeError("hypothesis must be a string, CoefficientHypothesis, Hypothesis, or numpy array")
        
        # Check if reduced model matrix is different from original
        original_model_matrix = self.LMlike.model_matrix
        if reduced_model_matrix.shape[1] == original_model_matrix.shape[1]:
            raise ValueError(f"Removing term {hypothesis} doesn't actually alter the model, maybe due to marginality?")
        
        # Create a copy of the original LMlike object
        from copy import deepcopy
        reduced_LMlike = deepcopy(self.LMlike)
        
        # Update the model matrix
        reduced_LMlike.model_matrix = reduced_model_matrix
        
        # Refit the model with the reduced model matrix
        print("Refitting on reduced model...")
        
        # Get expression matrix
        ee = self.sca.assay().T  # Transpose to (n_cells, n_genes)
        genes = self.coefC.index
        ng = ee.shape[1]
        
        # Calculate empirical Bayes parameters if needed
        use_ebayes = True
        prior_var = 1
        prior_df = 0
        
        if use_ebayes:
            # Build model matrix for ebayes
            ebparm = ebayes(ee, None, reduced_model_matrix)
            prior_var = ebparm['v']
            prior_df = ebparm['df']
        
        # Create LMlike object if not provided
        method_class = type(reduced_LMlike).__name__
        
        # Create new LMlike object with reduced model
        # Since we're directly setting the model matrix, we can use the same formula
        if method_class == 'GLMlike':
            reduced_LMlike = GLMlike(self.LMlike.formula, design, prior_var=prior_var, prior_df=prior_df)
        elif method_class == 'BayesGLMlike':
            reduced_LMlike = BayesGLMlike(self.LMlike.formula, design, prior_var=prior_var, prior_df=prior_df)
        elif method_class == 'LMERlike':
            reduced_LMlike = LMERlike(self.LMlike.formula, design, prior_var=prior_var, prior_df=prior_df)
        else:
            raise ValueError(f"Unknown method class: {method_class}")
        
        # Manually set the model matrix to our reduced version
        reduced_LMlike.model_matrix = reduced_model_matrix
        
        # Fit all genes with reduced model
        # fit_gene is defined in this file, no need to import
        
        results = []
        for idx in tqdm(range(ng), desc="Fitting genes with reduced model"):
            result = fit_gene(idx, reduced_LMlike, ee, silent=True, hook=None)
            results.append(result)
        
        # Separate summaries and hookOut
        list_of_summaries, hookOut = zip(*results)
        
        # Collect summaries - we don't need to create DataFrames, just extract the values we need
        reduced_loglik = np.array([s['loglik'] for s in list_of_summaries])
        reduced_converged = np.array([s['converged'] for s in list_of_summaries])
        reduced_df_resid = np.array([s['df.resid'] for s in list_of_summaries])
        
        # Calculate likelihood ratio statistics
        n_genes = self.coefC.shape[0]
        results = np.zeros((n_genes, 3, 3))  # (genes, components, metrics)
        
        for i in range(n_genes):
            # Get log likelihoods
            full_loglik_C = self.loglik.iloc[i]['C']
            full_loglik_D = self.loglik.iloc[i]['D']
            reduced_loglik_C = reduced_loglik[i][0]
            reduced_loglik_D = reduced_loglik[i][1]
            
            # Get convergence status
            full_converged_C = self.converged.iloc[i]['C']
            full_converged_D = self.converged.iloc[i]['D']
            reduced_converged_C = reduced_converged[i][0]
            reduced_converged_D = reduced_converged[i][1]
            
            # Get degrees of freedom
            full_df_resid_C = self.df_resid.iloc[i]['C']
            full_df_resid_D = self.df_resid.iloc[i]['D']
            reduced_df_resid_C = reduced_df_resid[i][0]
            reduced_df_resid_D = reduced_df_resid[i][1]
            
            # Calculate degrees of freedom difference
            dfC = reduced_df_resid_C - full_df_resid_C
            dfD = reduced_df_resid_D - full_df_resid_D
            
            # Calculate likelihood ratio statistics
            lambdaC = -2 * (reduced_loglik_C - full_loglik_C)
            lambdaD = -2 * (reduced_loglik_D - full_loglik_D)
            
            # Check if both models converged
            testable_C = full_converged_C and reduced_converged_C and full_df_resid_C > 1 and reduced_df_resid_C > 1
            testable_D = full_converged_D and reduced_converged_D and full_df_resid_D > 1 and reduced_df_resid_D > 1
            
            # Set statistics to 0 if not testable
            lambdaC = lambdaC if testable_C else 0
            lambdaD = lambdaD if testable_D else 0
            dfC = dfC if testable_C else 0
            dfD = dfD if testable_D else 0
            
            # Calculate p-values
            pC = chi2.sf(lambdaC, dfC) if lambdaC > 0 and dfC > 0 else 1.0
            pD = chi2.sf(lambdaD, dfD) if lambdaD > 0 and dfD > 0 else 1.0
            
            # Calculate combined (hurdle) statistics
            lambdaH = lambdaC + lambdaD
            dfH = dfC + dfD
            pH = chi2.sf(lambdaH, dfH) if lambdaH > 0 and dfH > 0 else 1.0
            
            # Store results
            results[i, 0, 0] = lambdaC  # cont lambda
            results[i, 0, 1] = dfC       # cont df
            results[i, 0, 2] = pC        # cont p-value
            
            results[i, 1, 0] = lambdaD  # disc lambda
            results[i, 1, 1] = dfD       # disc df
            results[i, 1, 2] = pD        # disc p-value
            
            results[i, 2, 0] = lambdaH  # hurdle lambda
            results[i, 2, 1] = dfH       # hurdle df
            results[i, 2, 2] = pH        # hurdle p-value
        
        return results
    
    def waldTest(self, hypothesis):
        """
        Wald test
        
        Parameters
        ----------
        hypothesis : Hypothesis, CoefficientHypothesis, or matrix
            Hypothesis to test
        
        Returns
        -------
        numpy.ndarray
            Test results with dimensions (n_genes, 3, 3) where:
            - First dimension: genes
            - Second dimension: test components (cont, disc, hurdle)
            - Third dimension: metrics (lambda, df, p.value)
        """
        from .hypothesis import Hypothesis, CoefficientHypothesis
        
        # Generate contrast matrix from hypothesis
        if isinstance(hypothesis, str):
            # For string hypothesis, create a contrast matrix for the coefficient
            coef_names = self.coefC.columns
            if hypothesis not in coef_names:
                raise ValueError(f"Coefficient {hypothesis} not found in model")
            contrast_matrix = np.zeros((1, len(coef_names)))
            contrast_matrix[0, coef_names.get_loc(hypothesis)] = 1
        elif isinstance(hypothesis, CoefficientHypothesis):
            # For CoefficientHypothesis, create contrast matrix
            coef_names = self.coefC.columns
            if hypothesis.coefficient not in coef_names:
                raise ValueError(f"Coefficient {hypothesis.coefficient} not found in model")
            contrast_matrix = np.zeros((1, len(coef_names)))
            contrast_matrix[0, coef_names.get_loc(hypothesis.coefficient)] = 1
        elif isinstance(hypothesis, Hypothesis):
            # For Hypothesis, use its contrast matrix
            # Note: This requires implementing generateHypothesis method
            # For now, we'll raise an error
            raise NotImplementedError("Hypothesis class support not fully implemented")
        elif isinstance(hypothesis, np.ndarray):
            # For matrix hypothesis, use it directly
            contrast_matrix = hypothesis
        else:
            raise TypeError("hypothesis must be a string, CoefficientHypothesis, Hypothesis, or numpy array")
        
        n_genes = self.coefC.shape[0]
        n_contrasts = contrast_matrix.shape[0]
        
        # Initialize results array
        results = np.zeros((n_genes, 3, 3))  # (genes, components, metrics)
        
        for i in range(n_genes):
            # Get coefficients and covariance matrices for this gene
            coefC = self.coefC.iloc[i].values
            coefD = self.coefD.iloc[i].values
            vcovC = self.vcovC[:, :, i]
            vcovD = self.vcovD[:, :, i]
            
            # Calculate contrasts for continuous component
            contrC = contrast_matrix @ coefC
            contrCovC = contrast_matrix @ vcovC @ contrast_matrix.T
            
            # Calculate contrasts for discrete component
            contrD = contrast_matrix @ coefD
            contrCovD = contrast_matrix @ vcovD @ contrast_matrix.T
            
            # Calculate Wald statistics (chi-square)
            try:
                inv_covC = np.linalg.inv(contrCovC)
                lambdaC = float(contrC @ inv_covC @ contrC.T)
            except np.linalg.LinAlgError:
                lambdaC = 0
            
            try:
                inv_covD = np.linalg.inv(contrCovD)
                lambdaD = float(contrD @ inv_covD @ contrD.T)
            except np.linalg.LinAlgError:
                lambdaD = 0
            
            # Calculate degrees of freedom
            dfC = n_contrasts
            dfD = n_contrasts
            
            # Calculate p-values
            pC = chi2.sf(lambdaC, dfC) if lambdaC > 0 else 1.0
            pD = chi2.sf(lambdaD, dfD) if lambdaD > 0 else 1.0
            
            # Calculate combined (hurdle) statistics
            lambdaH = lambdaC + lambdaD
            dfH = dfC + dfD
            pH = chi2.sf(lambdaH, dfH) if lambdaH > 0 else 1.0
            
            # Store results
            results[i, 0, 0] = lambdaC  # cont lambda
            results[i, 0, 1] = dfC       # cont df
            results[i, 0, 2] = pC        # cont p-value
            
            results[i, 1, 0] = lambdaD  # disc lambda
            results[i, 1, 1] = dfD       # disc df
            results[i, 1, 2] = pD        # disc p-value
            
            results[i, 2, 0] = lambdaH  # hurdle lambda
            results[i, 2, 1] = dfH       # hurdle df
            results[i, 2, 2] = pH        # hurdle p-value
        
        return results

def zlm(formula, sca, method='bayesglm', silent=True, use_ebayes=True, ebayesControl=None, force=False, hook=None, n_jobs=1, LMlike=None, onlyCoef=False, exprs_values=None):
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
    n_jobs : int, optional
        Number of parallel jobs to use. Default is 1. Set to 1 for serial processing.
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
    if n_jobs > 1:
        # Use multiprocessing
        import multiprocessing as mp
        from tqdm import tqdm
        # Determine number of processes to use
        n_processes = min(n_jobs, mp.cpu_count())
        with mp.Pool(processes=n_processes) as pool:
            # Create argument tuples for each gene
            args = [(idx, obj, ee, silent, hook) for idx in range(ng)]
            results = list(tqdm(pool.starmap(fit_gene, args), total=ng, desc="Fitting genes"))
    else:
        # Use sequential processing
        from tqdm import tqdm
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