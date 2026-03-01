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