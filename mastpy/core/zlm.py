import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from .lm_wrapper import GLMlike, BayesGLMlike, LMERlike
from ..utils.utils import ebayes
from scipy.stats import chi2, norm
from numba import jit

# Helper functions for logFC calculation
def expit(eta):
    """Expit function (inverse of logit)"""
    return np.exp(eta) / (1 + np.exp(eta))

def dexpit(eta):
    """Derivative of expit function"""
    exp_eta = np.exp(eta)
    return exp_eta / (1 + exp_eta) ** 2

def safe_contrast_dp(contrast, coef):
    """Dot product of contrast and coefficients"""
    return np.dot(coef, contrast.T)

def safe_contrast_qf(contrast, vc):
    """Quadratic form of contrast about vc"""
    return np.dot(np.dot(contrast, vc), contrast.T)

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
            # For string hypothesis, update the formula to remove the term
            # This matches the R MAST implementation
            from patsy import dmatrix
            
            # Get the original formula
            original_formula = self.LMlike.formula
            
            # Create new formula by removing the hypothesis term
            new_formula = original_formula.replace(hypothesis, '').replace('  ', ' ').strip()
            if new_formula == '~':
                new_formula = '~ 1'  # If only intercept remains
            
            # Build new model matrix
            reduced_model_matrix = dmatrix(new_formula, design, return_type='dataframe')
            
            # Rename columns to match R's format
            reduced_model_matrix.columns = reduced_model_matrix.columns.str.replace(' ', '')
            reduced_model_matrix.columns = reduced_model_matrix.columns.str.replace('(', '')
            reduced_model_matrix.columns = reduced_model_matrix.columns.str.replace(')', '')
            reduced_model_matrix.columns = reduced_model_matrix.columns.str.replace('[', '')
            reduced_model_matrix.columns = reduced_model_matrix.columns.str.replace(']', '')
            
            # Check if the reduced model matrix has a different column space
            original_model_matrix = self.LMlike.model_matrix
            if reduced_model_matrix.shape[1] == original_model_matrix.shape[1]:
                # Perform a more thorough check
                from numpy.linalg import qr
                
                # Calculate QR decomposition of reduced model matrix
                q, r = qr(reduced_model_matrix.values)
                
                # Calculate projection of original model matrix onto reduced space
                projection = q @ (q.T @ original_model_matrix.values)
                
                # Check if the projection is close to the original
                if np.allclose(projection, original_model_matrix.values, atol=1e-6):
                    raise ValueError(f"Removing term {hypothesis} doesn't actually alter the model, maybe due to marginality?")
        elif isinstance(hypothesis, CoefficientHypothesis):
            # For CoefficientHypothesis, generate contrast matrix
            coef_names = self.coefC.columns.tolist()
            hypothesis.generate_hypothesis(coef_names)
            
            # Create reduced model by removing the coefficient
            reduced_columns = [col for col in coef_names if col != hypothesis.coefficient]
            if len(reduced_columns) == 0:
                # If only intercept was present, keep it
                n = len(design)
                intercept = np.ones(n)
                reduced_model_matrix = pd.DataFrame(intercept, columns=['(Intercept)'])
            else:
                # Create reduced model matrix with remaining columns
                reduced_model_matrix = self.LMlike.model_matrix[reduced_columns]
        elif isinstance(hypothesis, Hypothesis):
            # For Hypothesis, generate contrast matrix
            coef_names = self.coefC.columns.tolist()
            hypothesis.generate_hypothesis(coef_names)
            
            # Create reduced model by removing the contrast
            # This is a simplified implementation
            n = len(design)
            intercept = np.ones(n)
            reduced_model_matrix = pd.DataFrame(intercept, columns=['(Intercept)'])
        elif isinstance(hypothesis, np.ndarray):
            # For matrix hypothesis, create reduced model with only intercept
            # This is a simplified implementation
            n = len(design)
            intercept = np.ones(n)
            reduced_model_matrix = pd.DataFrame(intercept, columns=['(Intercept)'])
        else:
            raise TypeError("hypothesis must be a string, CoefficientHypothesis, Hypothesis, or numpy array")
        
        # Check if reduced model matrix is different from original
        original_model_matrix = self.LMlike.model_matrix
        if reduced_model_matrix.shape[1] == original_model_matrix.shape[1]:
            # Try a different approach using contrast matrix
            print(f"Removing term {hypothesis} doesn't alter the model, using contrast-based approach instead")
            # Fall back to Wald test for this case
            return self.waldTest(hypothesis)
        
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
        
        # Generate contrast matrix from hypothesis
        if isinstance(hypothesis, str):
            # For string hypothesis, create a contrast matrix for the coefficient
            coef_names = self.coefC.columns
            if hypothesis not in coef_names:
                raise ValueError(f"Coefficient {hypothesis} not found in model")
            contrast_matrix = np.zeros((1, len(coef_names)))
            contrast_matrix[0, coef_names.get_loc(hypothesis)] = 1
        elif isinstance(hypothesis, CoefficientHypothesis):
            # For CoefficientHypothesis, generate contrast matrix
            coef_names = self.coefC.columns.tolist()
            hypothesis.generate_hypothesis(coef_names)
            contrast_matrix = hypothesis.contrast_matrix
        elif isinstance(hypothesis, Hypothesis):
            # For Hypothesis, generate contrast matrix
            coef_names = self.coefC.columns.tolist()
            hypothesis.generate_hypothesis(coef_names)
            contrast_matrix = hypothesis.contrast_matrix
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
    
    def se_coef(self, which):
        """
        Get standard errors for coefficients
        
        Parameters
        ----------
        which : str
            Component ('C' for continuous, 'D' for discrete)
        
        Returns
        -------
        pandas.DataFrame
            Standard errors for coefficients
        """
        if which == 'C':
            vcov = self.vcovC
        elif which == 'D':
            vcov = self.vcovD
        else:
            raise ValueError("which must be 'C' or 'D'")
        
        # Calculate standard errors as square root of diagonal of covariance matrix
        n_genes = vcov.shape[2]
        n_coefs = vcov.shape[0]
        se = np.zeros((n_genes, n_coefs))
        
        for i in range(n_genes):
            se[i, :] = np.sqrt(np.diag(vcov[:, :, i]))
        
        # Create DataFrame
        se_df = pd.DataFrame(se, index=self.coefC.index, columns=self.coefC.columns)
        return se_df
    
    def logFC(self, contrast0=None, contrast1=None):
        """
        Calculate log-fold changes from hurdle model components
        
        Using the delta method, estimate the log-fold change from a state given by a vector contrast0 and the state(s) given by contrast1.
        
        The log-fold change is defined as follows. For each gene, let u(x) be the expected value of the continuous component,
        given a covariate x and the estimated coefficients coefC, ie, u(x) = crossprod(x, coefC).
        Likewise, Let v(x) = 1/(1+exp(-crossprod(coefD, x))) be the expected value of the discrete component.
        The log fold change from contrast0 to contrast1 is defined as
        u(contrast1)v(contrast1) - u(contrast0)v(contrast0).
        
        Parameters
        ----------
        contrast0 : vector or matrix, optional
            Baseline contrast. If missing, the intercept is used.
        contrast1 : matrix, optional
            Comparison contrasts. If missing, all non-intercept coefficients are compared.
        
        Returns
        -------
        dict
            Dictionary with logFC and varLogFC matrices
        """
        coname = self.coefC.columns.tolist()
        genes = self.coefC.index.tolist()
        n_genes = len(genes)
        
        # Precompute non-intercept indices
        non_intercept_idx = [i for i, name in enumerate(coname) if name != '(Intercept)']
        
        # Handle contrast0
        if contrast0 is None:
            if '(Intercept)' in coname:
                # Use intercept as baseline
                contrast0 = np.zeros(len(coname))
                contrast0[coname.index('(Intercept)')] = 1
                contrast0 = contrast0.reshape(1, -1)
            else:
                # No intercept, use all zeros as baseline
                contrast0 = np.zeros(len(coname)).reshape(1, -1)
        elif isinstance(contrast0, np.ndarray):
            if contrast0.ndim == 1:
                contrast0 = contrast0.reshape(1, -1)
        else:
            raise TypeError("contrast0 must be a numpy array or None")
        
        # Handle contrast1
        if contrast1 is None:
            # Create contrast1 for all non-intercept coefficients
            if not non_intercept_idx:
                # No non-intercept coefficients, use all zeros
                contrast1 = np.zeros((1, len(coname)))
            else:
                # Create contrast1 for each non-intercept coefficient
                contrast1 = np.zeros((len(non_intercept_idx), len(coname)))
                for i, idx in enumerate(non_intercept_idx):
                    contrast1[i, idx] = 1
                # Add contrast0
                contrast1 = contrast1 + contrast0
        elif isinstance(contrast1, np.ndarray):
            if contrast1.ndim == 1:
                contrast1 = contrast1.reshape(1, -1)
        else:
            raise TypeError("contrast1 must be a numpy array or None")
        
        # Combine contrasts
        Contr = np.vstack([contrast0, contrast1])
        
        # Get coefficients and covariance matrices
        coefC = self.coefC.values
        coefD = self.coefD.values
        vcovC = self.vcovC
        vcovD = self.vcovD
        
        # Calculate expectations
        mu_cont = np.dot(coefC, Contr.T)  # genes x contrasts
        eta_disc = np.dot(coefD, Contr.T)  # genes x contrasts
        
        # Calculate variances
        n_contrasts = Contr.shape[0]
        vcont = np.zeros((n_genes, n_contrasts, n_contrasts))
        vdisc = np.zeros((n_genes, n_contrasts, n_contrasts))
        
        for i in range(n_genes):
            # Continuous component variance
            vcont[i, :, :] = safe_contrast_qf(Contr, vcovC[:, :, i])
            
            # Discrete component variance (with expit transformation)
            vcc = safe_contrast_qf(Contr, vcovD[:, :, i])
            jacobian = dexpit(eta_disc[i, :])
            vdisc[i, :, :] = vcc * np.outer(jacobian, jacobian)
        
        # Calculate mu.disc
        mu_disc = expit(eta_disc)
        
        # Calculate product of continuous and discrete components
        mu_prod = mu_cont * mu_disc
        
        # Calculate variance of product
        dvcont = np.diagonal(vcont, axis1=1, axis2=2)
        dvdisc = np.diagonal(vdisc, axis1=1, axis2=2)
        v_prod = dvcont * dvdisc + mu_cont ** 2 * dvdisc + mu_disc ** 2 * dvcont
        
        # Calculate covariance between contrast1 and contrast0
        # Ensure shapes are compatible for broadcasting
        vcont_cov = vcont[:, 1:, 0:1].squeeze(axis=2)  # (n_genes, n_contrast1)
        vdisc_cov = vdisc[:, 1:, 0:1].squeeze(axis=2)  # (n_genes, n_contrast1)
        mu_cont0 = mu_cont[:, 0:1]  # (n_genes, 1)
        mu_cont1 = mu_cont[:, 1:]    # (n_genes, n_contrast1)
        mu_disc0 = mu_disc[:, 0:1]   # (n_genes, 1)
        mu_disc1 = mu_disc[:, 1:]     # (n_genes, n_contrast1)
        
        covc1c0 = vcont_cov * vdisc_cov + \
                  mu_cont0 * mu_cont1 * vdisc_cov + \
                  mu_disc0 * mu_disc1 * vcont_cov
        cov_prod = covc1c0
        
        # Calculate log-fold changes (difference between contrast1 and contrast0)
        lfc = mu_prod[:, 1:] - mu_prod[:, 0:1]
        
        # Calculate variance of log-fold changes
        vlfc = v_prod[:, 1:] + v_prod[:, 0:1] - 2 * cov_prod
        
        # Create dataframes
        contrast_names = [f"{coname[i]}" for i in non_intercept_idx] if contrast1 is None else [f"contrast_{i}" for i in range(contrast1.shape[0])]
        logFC_df = pd.DataFrame(lfc, index=genes, columns=contrast_names)
        varLogFC_df = pd.DataFrame(vlfc, index=genes, columns=contrast_names)
        
        return {'logFC': logFC_df, 'varLogFC': varLogFC_df}

    def summary(self, logFC=True, doLRT=False, level=0.95, parallel=False):
        """
        Summarize model features from a ZlmFit object
        
        Parameters
        ----------
        logFC : bool or pandas.DataFrame
            If TRUE, calculate log-fold changes, or output from a call to getLogFC
        doLRT : bool or list
            if TRUE, calculate lrTests on each coefficient, or a list of such coefficients to consider
        level : float
            What level of confidence coefficient to return. Defaults to 95 percent
        parallel : bool
            If TRUE, use parallel processing for LRT
        
        Returns
        -------
        dict
            Summary results including datatable
        """

        print('Combining coefficients and standard errors')
        
        # Calculate coefficients and confidence intervals for each component
        components = {'C': 'continuous', 'D': 'discrete'}
        coef_and_ci = []
        
        for comp_key, comp_name in components.items():
            # Get coefficients
            coefs = self.coef(comp_key)
            # Get standard errors
            se = self.se_coef(comp_key)
            
            # Calculate confidence intervals
            z_star = -norm.ppf((1 - level) / 2)
            ci_lo = coefs - se * z_star
            ci_hi = coefs + se * z_star
            
            # Calculate z-scores
            z = coefs / se
            
            # Melt dataframes for easier manipulation
            coefs_melted = coefs.melt(ignore_index=False, var_name='contrast', value_name='coef')
            se_melted = se.melt(ignore_index=False, var_name='contrast', value_name='se')
            ci_lo_melted = ci_lo.melt(ignore_index=False, var_name='contrast', value_name='ci.lo')
            ci_hi_melted = ci_hi.melt(ignore_index=False, var_name='contrast', value_name='ci.hi')
            z_melted = z.melt(ignore_index=False, var_name='contrast', value_name='z')
            
            # Combine into a single dataframe
            comp_df = pd.concat([coefs_melted, se_melted['se'], ci_lo_melted['ci.lo'], ci_hi_melted['ci.hi'], z_melted['z']], axis=1)
            comp_df['component'] = comp_key
            
            coef_and_ci.append(comp_df)
        
        # Combine results for both components
        dt = pd.concat(coef_and_ci)
        dt = dt.reset_index().rename(columns={'index': 'primerid'})
        
        # Calculate Stouffer's method for combining z-scores
        # 使用向量化操作计算Stouffer's z-score
        stouffer = dt.groupby(['primerid', 'contrast']).agg(
            z=('z', lambda x: np.sum(x) / np.sqrt(np.sum(~np.isnan(x)))),
            component=('component', lambda x: 'S')
        ).reset_index()
        
        # Add Stouffer results to dataframe
        dt = pd.concat([dt, stouffer], ignore_index=True)
        
        # Calculate log-fold changes if requested
        if isinstance(logFC, bool) and logFC:
            print("Calculating log-fold changes")
            # Implement logFC calculation consistent with R MAST
            logFC_result = self.logFC()
            logFC_df = logFC_result['logFC']
            varLogFC_df = logFC_result['varLogFC']
            
            # Melt logFC dataframe
            logFC_melted = logFC_df.melt(ignore_index=False, var_name='contrast', value_name='coef')
            logFC_melted = logFC_melted.reset_index().rename(columns={'index': 'primerid'})
            
            # Melt varLogFC dataframe
            varLogFC_melted = varLogFC_df.melt(ignore_index=False, var_name='contrast', value_name='varLogFC')
            varLogFC_melted = varLogFC_melted.reset_index().rename(columns={'index': 'primerid'})
            
            # Merge logFC and varLogFC
            logFC_df = logFC_melted.merge(varLogFC_melted, on=['primerid', 'contrast'])
            logFC_df['component'] = 'logFC'
            
            # Calculate standard error from variance
            logFC_df['se'] = np.sqrt(logFC_df['varLogFC'])
            
            # Calculate confidence intervals
            z_star = -norm.ppf((1 - level) / 2)
            logFC_df['ci.lo'] = logFC_df['coef'] - logFC_df['se'] * z_star
            logFC_df['ci.hi'] = logFC_df['coef'] + logFC_df['se'] * z_star
            
            # Calculate z-score
            logFC_df['z'] = logFC_df['coef'] / logFC_df['se']
            
            # Add logFC results to dataframe
            dt = pd.concat([dt, logFC_df], ignore_index=True)
        elif not isinstance(logFC, bool):
            # If logFC is a dataframe, use it directly
            logFC_df = logFC.copy()
            logFC_df['component'] = 'logFC'
            dt = pd.concat([dt, logFC_df], ignore_index=True)
        
        # Calculate likelihood ratio tests if requested
        if isinstance(doLRT, bool) and doLRT:
            # Test all non-intercept coefficients
            doLRT = [col for col in self.coefD.columns if col != '(Intercept)']
        
        if not isinstance(doLRT, bool) and doLRT:
            print('Calculating likelihood ratio tests')
            lrt_results = []
            
            for coef in doLRT:
                try:
                    # Perform LRT for this coefficient
                    test_result = self.lrTest(coef)
                    
                    # 使用向量化操作提取p-values
                    p_cont = test_result[:, 0, 2]  # cont p-value
                    p_disc = test_result[:, 1, 2]  # disc p-value
                    p_hurdle = test_result[:, 2, 2]  # hurdle p-value
                    
                    # 向量化构建结果列表
                    for i, gene in enumerate(self.coefC.index):
                        lrt_results.append({
                            'primerid': gene,
                            'component': 'C',
                            'contrast': coef,
                            'Pr(>Chisq)': p_cont[i]
                        })
                        lrt_results.append({
                            'primerid': gene,
                            'component': 'D',
                            'contrast': coef,
                            'Pr(>Chisq)': p_disc[i]
                        })
                        lrt_results.append({
                            'primerid': gene,
                            'component': 'H',
                            'contrast': coef,
                            'Pr(>Chisq)': p_hurdle[i]
                        })
                except Exception as e:
                    print(f"Error performing LRT for {coef}: {e}")
            
            # Create dataframe from LRT results
            if lrt_results:
                lrt_df = pd.DataFrame(lrt_results)
                
                # Merge with main dataframe
                dt = dt.merge(lrt_df, on=['primerid', 'component', 'contrast'], how='outer')
        
        # Set component for hurdle results if missing
        dt.loc[dt['component'].isna(), 'component'] = 'H'
        
        # Create summary object
        out = {'datatable': dt}
        out['__class__'] = 'summaryZlmFit'
        
        return out

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
            # Use BayesGLMlike with continuous Bayes for better compatibility with R MAST
            obj = BayesGLMlike(formula, design, prior_var=prior_var, prior_df=prior_df, use_continuous_bayes=True)
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
        # Use joblib for better parallel processing with batching
        from joblib import Parallel, delayed
        from tqdm import tqdm
        # Determine number of processes to use
        n_processes = min(n_jobs, mp.cpu_count())
        
        # Create batches to reduce inter-process communication
        batch_size = max(1, ng // (n_processes * 10))  # Adjust batch size based on number of genes
        batches = [range(i, min(i + batch_size, ng)) for i in range(0, ng, batch_size)]
        
        # Function to process a batch of genes
        def process_batch(batch):
            batch_results = []
            for idx in batch:
                batch_results.append(fit_gene(idx, obj, ee, silent, hook))
            return batch_results
        
        # Use tqdm with joblib
        with tqdm(total=ng, desc="Fitting genes") as pbar:
            def process_batch_with_progress(batch):
                results = process_batch(batch)
                pbar.update(len(batch))
                return results
            
            # Run in parallel
            batch_results = Parallel(n_jobs=n_processes, backend='loky')(delayed(process_batch_with_progress)(batch) for batch in batches)
            
            # Flatten results
            results = []
            for batch_result in batch_results:
                results.extend(batch_result)
    else:
        # Use sequential processing
        from tqdm import tqdm
        results = [fit_gene(idx, obj, ee, silent, hook) for idx in tqdm(range(ng), desc="Fitting genes")]
    
    # Separate summaries and hookOut
    list_of_summaries, hookOut = zip(*results)
    
    # 使用numpy向量化操作收集和处理结果
    # 提取所有摘要数据
    coefC_list = [s['coefC'] for s in list_of_summaries]
    coefD_list = [s['coefD'] for s in list_of_summaries]
    vcovC_list = [s['vcovC'] for s in list_of_summaries]
    vcovD_list = [s['vcovD'] for s in list_of_summaries]
    df_resid_list = [s['df.resid'] for s in list_of_summaries]
    df_null_list = [s['df.null'] for s in list_of_summaries]
    deviance_list = [s['deviance'] for s in list_of_summaries]
    dispersion_list = [s['dispersion'] for s in list_of_summaries]
    dispersionNoshrink_list = [s['dispersionNoshrink'] for s in list_of_summaries]
    converged_list = [s['converged'] for s in list_of_summaries]
    loglik_list = [s['loglik'] for s in list_of_summaries]
    
    # 使用numpy向量化操作创建数组
    coefC_array = np.array(coefC_list)
    coefD_array = np.array(coefD_list)
    vcovC_array = np.array(vcovC_list)
    vcovD_array = np.array(vcovD_list)
    df_resid_array = np.array(df_resid_list)
    df_null_array = np.array(df_null_list)
    deviance_array = np.array(deviance_list)
    dispersion_array = np.array(dispersion_list)
    dispersionNoshrink_array = np.array(dispersionNoshrink_list)
    converged_array = np.array(converged_list)
    loglik_array = np.array(loglik_list)
    
    # 创建DataFrame和调整维度
    coefC = pd.DataFrame(coefC_array, index=genes, columns=coef_names)
    coefD = pd.DataFrame(coefD_array, index=genes, columns=coef_names)
    
    # 调整方差-协方差矩阵维度
    vcovC = np.transpose(vcovC_array, (1, 2, 0))
    vcovD = np.transpose(vcovD_array, (1, 2, 0))
    
    # 创建其他摘要DataFrame
    df_resid = pd.DataFrame(df_resid_array, index=genes, columns=['C', 'D'])
    df_null = pd.DataFrame(df_null_array, index=genes, columns=['C', 'D'])
    deviance = pd.DataFrame(deviance_array, index=genes, columns=['C', 'D'])
    dispersion = pd.DataFrame(dispersion_array, index=genes, columns=['C', 'D'])
    dispersionNoshrink = pd.DataFrame(dispersionNoshrink_array, index=genes, columns=['C', 'D'])
    converged = pd.DataFrame(converged_array, index=genes, columns=['C', 'D'])
    loglik = pd.DataFrame(loglik_array, index=genes, columns=['C', 'D'])
    
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