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