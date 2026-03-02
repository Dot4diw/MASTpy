"""Hypothesis classes for MASTpy

This module defines classes for specifying hypotheses in statistical tests.
These classes are used to define contrasts and coefficient hypotheses
for Wald tests and likelihood ratio tests in MASTpy.

Classes
-------
Hypothesis
    Base class for all hypothesis types
CoefficientHypothesis
    Class for specifying hypotheses about individual coefficients
"""
import numpy as np

class Hypothesis:
    """
    Base class for hypotheses
    """
    def __init__(self, hypothesis):
        """
        Initialize a Hypothesis object
        
        Parameters
        ----------
        hypothesis : str or matrix
            Hypothesis specification
        """
        self.hypothesis = hypothesis
        self.contrast_matrix = None
    
    def generate_hypothesis(self, terms):
        """
        Generate contrast matrix from hypothesis
        
        Parameters
        ----------
        terms : list
            List of coefficient names
        
        Returns
        -------
        Hypothesis
            Self with contrast matrix
        """
        if isinstance(self.hypothesis, str):
            # Parse contrast string
            self.contrast_matrix = self._parse_contrast(self.hypothesis, terms)
        elif isinstance(self.hypothesis, np.ndarray):
            # Use provided matrix directly
            self.contrast_matrix = self.hypothesis
        else:
            raise TypeError("hypothesis must be a string or numpy array")
        return self
    
    def _parse_contrast(self, contrast_str, terms):
        """
        Parse contrast string into matrix
        
        Parameters
        ----------
        contrast_str : str
            Contrast string
        terms : list
            List of coefficient names
        
        Returns
        -------
        numpy.ndarray
            Contrast matrix
        """
        # Simple implementation for now
        # Supports basic contrasts like "A - B"
        terms_dict = {term: i for i, term in enumerate(terms)}
        
        # Split contrast into parts
        parts = contrast_str.split('-')
        if len(parts) != 2:
            raise ValueError("Only simple contrasts (A - B) are supported")
        
        # Create contrast matrix
        contrast_matrix = np.zeros((1, len(terms)))
        
        # Parse first term
        term1 = parts[0].strip()
        if term1 not in terms_dict:
            raise ValueError(f"Term {term1} not found in model")
        contrast_matrix[0, terms_dict[term1]] = 1
        
        # Parse second term
        term2 = parts[1].strip()
        if term2 not in terms_dict:
            raise ValueError(f"Term {term2} not found in model")
        contrast_matrix[0, terms_dict[term2]] = -1
        
        return contrast_matrix

class CoefficientHypothesis(Hypothesis):
    """
    Coefficient hypothesis class
    """
    def __init__(self, coefficient):
        """
        Initialize a CoefficientHypothesis object
        
        Parameters
        ----------
        coefficient : str
            Coefficient name
        """
        super().__init__(coefficient)
        self.coefficient = coefficient
    
    def generate_hypothesis(self, terms):
        """
        Generate contrast matrix for coefficient hypothesis
        
        Parameters
        ----------
        terms : list
            List of coefficient names
        
        Returns
        -------
        CoefficientHypothesis
            Self with contrast matrix
        """
        terms_dict = {term: i for i, term in enumerate(terms)}
        
        if self.coefficient not in terms_dict:
            raise ValueError(f"Coefficient {self.coefficient} not found in model")
        
        # Create contrast matrix
        contrast_matrix = np.zeros((1, len(terms)))
        contrast_matrix[0, terms_dict[self.coefficient]] = 1
        
        self.contrast_matrix = contrast_matrix
        self.index = terms_dict[self.coefficient]
        
        return self