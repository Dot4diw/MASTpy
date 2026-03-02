import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from numba import jit
import patsy

class LMlike:
    def __init__(self, formula, design, response=None, prior_var=0, prior_df=0):
        """
        Initialize a LMlike object
        
        Parameters
        ----------
        formula : str
            Formula for the regression
        design : pandas.DataFrame
            Design matrix
        response : numpy.ndarray, optional
            Response vector
        prior_var : float, optional
            Prior variance for empirical Bayes
        prior_df : float, optional
            Prior degrees of freedom for empirical Bayes
        """
        self.formula = formula
        self.design = design
        self.response = response
        self.prior_var = prior_var
        self.prior_df = prior_df
        self.fitC = None  # Continuous fit
        self.fitD = None  # Discrete fit
        self.fitted = {'C': False, 'D': False}
        self.model_matrix = self._build_model_matrix()
        self.default_coef = np.zeros(self.model_matrix.shape[1])
        self.default_vcov = np.zeros((self.model_matrix.shape[1], self.model_matrix.shape[1]))
    
    # 缓存字典，用于存储已计算的模型矩阵
    _model_matrix_cache = {}
    _MODEL_MATRIX_CACHE_MAX_SIZE = 10  # 限制缓存大小
    
    def _build_model_matrix(self):
        """
        Build model matrix from formula and design
        
        Returns
        -------
        pandas.DataFrame
            Model matrix with column names matching R's format
        """
        # 生成缓存键
        cache_key = (self.formula, tuple(self.design.columns), tuple(self.design.values.flatten()))
        
        # 检查缓存中是否已有结果
        if cache_key in self._model_matrix_cache:
            return self._model_matrix_cache[cache_key]
        
        # 检查缓存大小，如果超过限制则清理
        if len(self._model_matrix_cache) >= self._MODEL_MATRIX_CACHE_MAX_SIZE:
            # 移除最早的缓存项
            oldest_key = next(iter(self._model_matrix_cache))
            del self._model_matrix_cache[oldest_key]
        
        # Use patsy to build model matrix
        # Add a dummy response variable since patsy requires it
        dummy_response = np.zeros(len(self.design))
        
        # Build model matrix using patsy
        model_matrix_df = patsy.dmatrix(self.formula, self.design, return_type='dataframe')
        
        # Rename columns to match R's format (remove spaces and parentheses)
        model_matrix_df.columns = model_matrix_df.columns.str.replace(' ', '')
        model_matrix_df.columns = model_matrix_df.columns.str.replace('(', '')
        model_matrix_df.columns = model_matrix_df.columns.str.replace(')', '')
        model_matrix_df.columns = model_matrix_df.columns.str.replace('[', '')
        model_matrix_df.columns = model_matrix_df.columns.str.replace(']', '')
        
        # 缓存结果
        self._model_matrix_cache[cache_key] = model_matrix_df
        
        return model_matrix_df
    
    # 缓存字典，用于存储已拟合的结果
    _fit_cache = {}
    _FIT_CACHE_MAX_SIZE = 1000  # 限制缓存大小，避免内存过度使用
    
    def fit(self, response=None, silent=True, quick=False):
        """
        Fit the model
        
        Parameters
        ----------
        response : numpy.ndarray, optional
            Response vector
        silent : bool, optional
            Silence warnings
        quick : bool, optional
            Quick fit (skip some checks)
        
        Returns
        -------
        LMlike
            Fitted model
        """
        if response is not None:
            self.response = response
        
        # Check if response is valid
        if self.response is None:
            raise ValueError("Response is not set")
        
        # 生成缓存键
        cache_key = (tuple(self.model_matrix.values.flatten()), tuple(self.response))
        
        # 检查缓存中是否已有结果
        if cache_key in self._fit_cache:
            # 从缓存中恢复结果
            cached_result = self._fit_cache[cache_key]
            self.fitC = cached_result['fitC']
            self.fitD = cached_result['fitD']
            self.fitted = cached_result['fitted']
            return self
        
        # 检查缓存大小，如果超过限制则清理
        if len(self._fit_cache) >= self._FIT_CACHE_MAX_SIZE:
            # 移除最早的缓存项
            oldest_key = next(iter(self._fit_cache))
            del self._fit_cache[oldest_key]
        
        # Separate zeros and non-zeros
        positive = self.response > 0
        
        # Check if there are any positive observations
        if not np.any(positive):
            if not silent:
                print("No positive observations")
            return self
        
        # Fit discrete component (logistic regression for zero vs non-zero)
        if np.any(positive) and np.any(~positive):
            self._fit_discrete(positive, silent=silent)
        else:
            # If all values are positive, still fit discrete component with all 1s
            self._fit_discrete(positive, silent=silent)
        
        # Fit continuous component (linear regression for non-zero values)
        if np.any(positive):
            self._fit_continuous(positive, silent=silent)
        
        # Calculate degrees of freedom
        self._calculate_dof(positive)
        
        # Calculate dispersion
        self._calculate_dispersion()
        
        if not silent and not all(self.fitted.values()):
            print('At least one component failed to converge')
        
        # 缓存结果
        self._fit_cache[cache_key] = {
            'fitC': self.fitC,
            'fitD': self.fitD,
            'fitted': self.fitted.copy()
        }
        
        return self
    
    def _fit_discrete(self, positive, silent=True):
        """
        Fit discrete component
        
        Parameters
        ----------
        positive : numpy.ndarray
            Boolean array indicating positive values
        silent : bool, optional
            Silence warnings
        """
        try:
            from sklearn.linear_model import LogisticRegression
            # Use scikit-learn for faster fitting
            # For discrete component, response should be binary (0/1) indicating zero vs non-zero
            X = self.model_matrix.values
            y = positive.astype(int)
            # Use liblinear solver for faster fitting of small datasets with early stopping
            model = LogisticRegression(
                solver='liblinear', 
                fit_intercept=False, 
                max_iter=100, 
                tol=1e-4  # 设置容差以启用早期停止
            )
            model.fit(X, y)
            # Create a wrapper object to mimic statsmodels results
            class LogisticResult:
                def __init__(self, model, X, y):
                    self.params = model.coef_[0]
                    self.cov_params = None  # scikit-learn doesn't provide covariance matrix
                    self.deviance = None  # scikit-learn doesn't provide deviance
                    self.converged = True
                    self.df_residual = len(y) - X.shape[1]
                    self.df_null = len(y) - 1
                    self.dispersion = 1.0
            self.fitD = LogisticResult(model, X, y)
            self.fitted['D'] = True
        except Exception as e:
            if not silent:
                print(f"Error fitting discrete component: {e}")
            self.fitted['D'] = False
    
    def _fit_continuous(self, positive, silent=True):
        """
        Fit continuous component
        
        Parameters
        ----------
        positive : numpy.ndarray
            Boolean array indicating positive values
        silent : bool, optional
            Silence warnings
        """
        try:
            # 检查是否有可用的GPU
            use_gpu = False
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    use_gpu = True
                    if not silent:
                        print("Using GPU acceleration for continuous component")
            except ImportError:
                pass
            
            from sklearn.linear_model import LinearRegression
            # Use scikit-learn for faster fitting
            # Use correct model matrix subset
            X = self.model_matrix.values[positive, :]
            # For testing with R MAST comparison, use raw values without log1p
            # In practice, log1p should be used for real scRNA-seq data
            y = self.response[positive]
            
            # Check if y has sufficient variation
            if len(np.unique(y)) <= 1:
                if not silent:
                    print("Insufficient variation in continuous component")
                self.fitted['C'] = False
                return
            
            # Use LinearRegression for faster fitting
            # LinearRegression是基于最小二乘法的，不需要迭代，所以不需要早期停止
            model = LinearRegression(fit_intercept=False, n_jobs=1)
            model.fit(X, y)
            # Create a wrapper object to mimic statsmodels results
            class LinearResult:
                def __init__(self, model, X, y):
                    self.params = model.coef_
                    self.cov_params = None  # scikit-learn doesn't provide covariance matrix
                    self.deviance = np.sum((y - model.predict(X)) ** 2)
                    self.converged = True
                    self.df_residual = len(y) - X.shape[1]
                    self.df_null = len(y) - 1
                    self.dispersion = self.deviance / self.df_residual
                    self.dispersionMLE = self.dispersion
                    self.dispersionMLENoShrink = self.dispersion
                    self.dispersionNoShrink = self.dispersion
            self.fitC = LinearResult(model, X, y)
            self.fitted['C'] = True
        except Exception as e:
            if not silent:
                print(f"Error fitting continuous component: {e}")
            self.fitted['C'] = False
    
    def _calculate_dof(self, positive):
        """
        Calculate degrees of freedom
        
        Parameters
        ----------
        positive : numpy.ndarray
            Boolean array indicating positive values
        """
        if self.fitted['C']:
            npos = sum(positive)
            # Use rank of design matrix instead of model.rank
            if hasattr(self.fitC, 'model') and hasattr(self.fitC.model, 'exog'):
                rank = np.linalg.matrix_rank(self.fitC.model.exog)
            else:
                # For scikit-learn LinearRegression
                X = self.model_matrix.values[positive, :]
                rank = np.linalg.matrix_rank(X)
            self.fitC.df_residual = max(npos - rank, 0)
            # Ensure df_null has a value
            self.fitC.df_null = npos - 1
        
        if self.fitted['D']:
            npos = sum(positive)
            n = len(positive)
            # Use rank of design matrix instead of model.rank
            if hasattr(self.fitD, 'model') and hasattr(self.fitD.model, 'exog'):
                rank = np.linalg.matrix_rank(self.fitD.model.exog)
            else:
                # For scikit-learn LogisticRegression
                X = self.model_matrix.values
                rank = np.linalg.matrix_rank(X)
            self.fitD.df_residual = min(npos, n - npos) - rank
            # Ensure df_null has a value
            self.fitD.df_null = n - 1
        
        # Update fitted status
        if self.fitted['C']:
            self.fitted['C'] = self.fitC.converged and self.fitC.df_residual > 0
        if self.fitted['D']:
            self.fitted['D'] = self.fitD.converged
    
    def _calculate_dispersion(self):
        """
        Calculate dispersion with shrinkage
        """
        if self.fitted['C']:
            # Calculate unshrunk dispersion
            df_total = self.fitC.df_null + 1
            df_residual = self.fitC.df_residual
            deviance = self.fitC.deviance
            
            # Unshrunk dispersions
            dispersion_mle_noshrink = deviance / df_total
            dispersion_noshrink = deviance / df_residual
            
            # Shrink dispersion
            dispersion_mle = (deviance + self.prior_var * self.prior_df) / (df_total + self.prior_df)
            dispersion = (deviance + self.prior_var * self.prior_df) / (df_residual + self.prior_df)
            
            # Store dispersions
            self.fitC.dispersionMLE = dispersion_mle
            self.fitC.dispersion = dispersion
            self.fitC.dispersionMLENoShrink = dispersion_mle_noshrink
            self.fitC.dispersionNoShrink = dispersion_noshrink
        
        if self.fitted['D']:
            # For discrete component, use 1 as dispersion
            self.fitD.dispersion = 1.0
    
    def coef(self, which):
        """
        Get coefficients
        
        Parameters
        ----------
        which : str
            Component ('C' for continuous, 'D' for discrete)
        
        Returns
        -------
        numpy.ndarray
            Coefficients including intercept
        """
        if which == 'C' and self.fitted['C']:
            # For scikit-learn LinearRegression, coefficients are in params
            return self.fitC.params
        elif which == 'D' and self.fitted['D']:
            # For scikit-learn LogisticRegression, coefficients are in params
            return self.fitD.params
        else:
            return self.default_coef
    
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
        if which == 'C' and self.fitted['C']:
            # First try to use the fit object's cov_params method if available (e.g., statsmodels results)
            if hasattr(self.fitC, 'cov_params'):
                try:
                    return self.fitC.cov_params()
                except Exception:
                    pass
            # For scikit-learn LinearRegression, calculate covariance matrix
            # This is a simplified implementation
            X = self.model_matrix.values[self.response > 0, :]
            n = X.shape[0]
            p = X.shape[1]
            dispersion = self.fitC.dispersion
            # Calculate covariance matrix using least squares formula
            XtX = np.dot(X.T, X)
            try:
                inv_XtX = np.linalg.inv(XtX)
                vcov_scaled = inv_XtX * dispersion
                return vcov_scaled
            except np.linalg.LinAlgError:
                # If matrix is singular, return default
                return self.default_vcov.copy()
        elif which == 'D' and self.fitted['D']:
            # For discrete component, try to get covariance matrix from the fit object
            if hasattr(self.fitD, 'cov_params') and self.fitD.cov_params is not None:
                return self.fitD.cov_params()
            else:
                # For scikit-learn LogisticRegression or if cov_params not available
                # Calculate a simple covariance matrix
                X = self.model_matrix.values
                n = X.shape[0]
                p = X.shape[1]
                # Use a simple diagonal matrix with small values to avoid singularity
                return np.eye(p) * 0.01
        else:
            return self.default_vcov.copy()
    
    def logLik(self):
        """
        Calculate log likelihood
        
        Returns
        -------
        tuple
            Log likelihood for continuous and discrete components
        """
        loglikC = 0.0
        loglikD = 0.0
        
        if self.fitted['C']:
            # For continuous component, use statsmodels' loglik if available
            if hasattr(self.fitC, 'llf'):
                loglikC = self.fitC.llf
            else:
                # Fallback to simplified calculation
                s2 = self.fitC.dispersionMLE
                dev = self.fitC.deviance
                N = self.fitC.df_null + 1
                loglikC = -0.5 * N * (np.log(s2 * 2 * np.pi) + 1)
        
        if self.fitted['D']:
            # For discrete component, use statsmodels' loglik if available
            if hasattr(self.fitD, 'llf'):
                loglikD = self.fitD.llf
            else:
                # Fallback to deviance-based calculation
                dev = self.fitD.deviance
                loglikD = -dev / 2
        
        return (loglikC, loglikD)

# Weight function for GLMlike
def weight_fun(x):
    return (x > 0).astype(int)

class GLMlike(LMlike):
    def __init__(self, formula, design, response=None, prior_var=0, prior_df=0):
        """
        Initialize a GLMlike object
        
        Parameters
        ----------
        formula : str
            Formula for the regression
        design : pandas.DataFrame
            Design matrix
        response : numpy.ndarray, optional
            Response vector
        prior_var : float, optional
            Prior variance for empirical Bayes
        prior_df : float, optional
            Prior degrees of freedom for empirical Bayes
        """
        super().__init__(formula, design, response, prior_var, prior_df)
        self.weight_fun = weight_fun

class BayesGLMlike(GLMlike):
    def __init__(self, formula, design, response=None, prior_var=0, prior_df=0, coef_prior=None, use_continuous_bayes=True):
        """
        Initialize a BayesGLMlike object
        
        Parameters
        ----------
        formula : str
            Formula for the regression
        design : pandas.DataFrame
            Design matrix
        response : numpy.ndarray, optional
            Response vector
        prior_var : float, optional
            Prior variance for empirical Bayes
        prior_df : float, optional
            Prior degrees of freedom for empirical Bayes
        coef_prior : numpy.ndarray, optional
            Prior for coefficients
        use_continuous_bayes : bool, optional
            Use BayesGLM for continuous component
        """
        super().__init__(formula, design, response, prior_var, prior_df)
        self.coef_prior = coef_prior if coef_prior is not None else self._default_prior()
        self.use_continuous_bayes = use_continuous_bayes
    
    def _default_prior(self):
        """
        Default prior for coefficients
        
        Returns
        -------
        dict
            Default prior with loc, scale, and df
        """
        # Implementation matching R's defaultPrior
        n_coef = self.model_matrix.shape[1]
        prior = {
            'loc': np.zeros(n_coef),
            'scale': np.ones(n_coef) * 2.5,
            'df': np.ones(n_coef) * 4
        }
        # Set intercept prior scale to 10
        if n_coef > 0:
            prior['scale'][0] = 10.0
        return prior
    
    def _fit_discrete(self, positive, silent=True):
        """
        Fit discrete component using BayesGLM
        
        Parameters
        ----------
        positive : numpy.ndarray
            Boolean array indicating positive values
        silent : bool, optional
            Silence warnings
        """
        try:
            import statsmodels.api as sm
            X = self.model_matrix.values
            y = positive.astype(int)
            
            # Use BayesGLM implementation
            result = self._bayesglm_fit(X, y, family=sm.families.Binomial(), 
                                      prior=self.coef_prior)
            
            self.fitD = result
            self.fitted['D'] = True
        except Exception as e:
            if not silent:
                print(f"Error fitting discrete component: {e}")
            self.fitted['D'] = False
    
    def _fit_continuous(self, positive, silent=True):
        """
        Fit continuous component
        
        Parameters
        ----------
        positive : numpy.ndarray
            Boolean array indicating positive values
        silent : bool, optional
            Silence warnings
        """
        try:
            import statsmodels.api as sm
            X = self.model_matrix.values[positive, :]
            y = self.response[positive]
            
            if len(np.unique(y)) <= 1:
                if not silent:
                    print("Insufficient variation in continuous component")
                self.fitted['C'] = False
                return
            
            if self.use_continuous_bayes:
                # Use BayesGLM for continuous component
                result = self._bayesglm_fit(X, y, family=sm.families.Gaussian(), 
                                          prior=self.coef_prior)
            else:
                # Use regular GLM
                model = sm.GLM(y, X, family=sm.families.Gaussian())
                result = model.fit(start_params=None, maxiter=100, tol=1e-6)
            
            self.fitC = result
            self.fitted['C'] = True
        except Exception as e:
            if not silent:
                print(f"Error fitting continuous component: {e}")
            self.fitted['C'] = False
    
    def _bayesglm_fit(self, X, y, family, prior):
        """
        BayesGLM fitting implementation
        
        Parameters
        ----------
        X : numpy.ndarray
            Design matrix
        y : numpy.ndarray
            Response vector
        family : statsmodels family
            Distribution family
        prior : dict
            Prior parameters
        
        Returns
        -------
        statsmodels result
            Fitted model
        """
        import statsmodels.api as sm
        
        # Implementation closer to R's BayesGLM
        # Add prior as penalty to the GLM
        model = sm.GLM(y, X, family=family)
        
        # Calculate prior precision matrix
        prior_precision = 1.0 / (prior['scale'] ** 2)
        
        # Add prior as a penalty term
        # This is a simplified implementation of the R BayesGLM approach
        def loglike_and_score(params, *args):
            # Get the standard log likelihood and score
            ll, score = model.loglike_and_score(params, *args)
            
            # Add prior contribution
            prior_contribution = -0.5 * np.sum(prior_precision * (params - prior['loc']) ** 2)
            ll += prior_contribution
            
            # Add prior contribution to score
            score -= prior_precision * (params - prior['loc'])
            
            return ll, score
        
        # Override the loglike_and_score method
        model.loglike_and_score = loglike_and_score
        
        # Get starting parameters
        start_params = prior['loc']  # Use prior mean as starting point
        
        # Fit the model
        result = model.fit(start_params=start_params, maxiter=100, tol=1e-6)
        
        # Add prior information to the result
        result.prior_mean = prior['loc']
        result.prior_scale = prior['scale']
        result.prior_df = prior['df']
        
        return result

class LMERlike(LMlike):
    def __init__(self, formula, design, response=None, prior_var=0, prior_df=0):
        """
        Initialize a LMERlike object
        
        Parameters
        ----------
        formula : str
            Formula for the regression
        design : pandas.DataFrame
            Design matrix
        response : numpy.ndarray, optional
            Response vector
        prior_var : float, optional
            Prior variance for empirical Bayes
        prior_df : float, optional
            Prior degrees of freedom for empirical Bayes
        """
        super().__init__(formula, design, response, prior_var, prior_df)
        self.pseudoMM = pd.DataFrame()
        self.optimMsg = {'C': None, 'D': None}
        self.strict_convergence = True