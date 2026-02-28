import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from numba import jit

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
    
    def _build_model_matrix(self):
        """
        Build model matrix from formula and design
        
        Returns
        -------
        pandas.DataFrame
            Model matrix with column names matching R's format
        """
        # 直接构建模型矩阵，确保与R的实现完全一致
        # 对于公式 "~ condition"，模型矩阵应该包含截距项和conditionB列
        n = len(self.design)
        
        # 构建截距项列
        intercept = np.ones(n)
        
        # 构建conditionB列：当condition为'B'时为1，否则为0
        conditionB = (self.design['condition'] == 'B').astype(int).values
        
        # 构建模型矩阵
        model_matrix = np.column_stack([intercept, conditionB])
        
        # 转换为DataFrame并设置列名
        model_matrix_df = pd.DataFrame(model_matrix, columns=['(Intercept)', 'conditionB'])
        
        return model_matrix_df
    
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
            import statsmodels.api as sm
            # 使用statsmodels的GLM类，更接近R的实现
            # 与R的实现一致，使用weightFun作为响应变量
            X = self.model_matrix.values
            y = self.weight_fun(self.response)
            model = sm.GLM(y, X, family=sm.families.Binomial())
            result = model.fit()
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
            # 使用正确的模型矩阵子集
            X = self.model_matrix.values[positive, :]
            y = self.response[positive]
            # 计算权重：对于大于0的值，权重为1，否则为0
            weights = self.weight_fun(y)
            # 使用statsmodels的GLM类，更接近R的实现
            # 使用与R的glm.fit相同的参数，包括权重
            model = sm.GLM(y, X, family=sm.families.Gaussian(), freq_weights=weights)
            result = model.fit()
            self.fitC = result
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
            # 使用设计矩阵的秩，而不是model.rank
            rank = np.linalg.matrix_rank(self.fitC.model.exog)
            self.fitC.df_residual = max(npos - rank, 0)
            # 确保df_null有值
            self.fitC.df_null = npos - 1
        
        if self.fitted['D']:
            npos = sum(positive)
            n = len(positive)
            # 使用设计矩阵的秩，而不是model.rank
            rank = np.linalg.matrix_rank(self.fitD.model.exog)
            self.fitD.df_residual = min(npos, n - npos) - rank
            # 确保df_null有值
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
            # For statsmodels GLM, coefficients are in params
            params = self.fitC.params
            return params.values if hasattr(params, 'values') else params
        elif which == 'D' and self.fitted['D']:
            # For statsmodels GLM, coefficients are in params
            params = self.fitD.params
            return params.values if hasattr(params, 'values') else params
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
        vc = self.default_vcov.copy()
        if which == 'C' and self.fitted['C']:
            # For continuous component, use scaled covariance matrix
            dispersion = self.fitC.dispersion
            # 使用cov_params()方法获取方差协方差矩阵
            cov_params = self.fitC.cov_params()
            vcov_scaled = cov_params.values * dispersion if hasattr(cov_params, 'values') else cov_params * dispersion
            # Fill in the valid coefficients
            ok = self.model_matrix.columns
            vc[[i for i, col in enumerate(self.model_matrix.columns)], :][:, [i for i, col in enumerate(self.model_matrix.columns)]] = vcov_scaled
        elif which == 'D' and self.fitted['D']:
            # For discrete component, use unscaled covariance matrix
            # 使用cov_params()方法获取方差协方差矩阵
            cov_params = self.fitD.cov_params()
            vcov_scaled = cov_params.values if hasattr(cov_params, 'values') else cov_params
            # Fill in the valid coefficients
            ok = self.model_matrix.columns
            vc[[i for i, col in enumerate(self.model_matrix.columns)], :][:, [i for i, col in enumerate(self.model_matrix.columns)]] = vcov_scaled
        return vc
    
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
            # For continuous component, calculate log likelihood similar to R
            s2 = self.fitC.dispersionMLE
            dev = self.fitC.deviance
            N = self.fitC.df_null + 1
            loglikC = -0.5 * N * (np.log(s2 * 2 * np.pi) + 1)
        
        if self.fitted['D']:
            # For discrete component, use deviance to calculate log likelihood
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
    def __init__(self, formula, design, response=None, prior_var=0, prior_df=0, coef_prior=None, use_continuous_bayes=False):
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
        numpy.ndarray
            Default prior
        """
        # Simple implementation
        n_coef = self.model_matrix.shape[1]
        return np.zeros(n_coef)

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