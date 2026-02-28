import numpy as np
import pandas as pd

class SingleCellAssay:
    def __init__(self, expression_matrix, cdata, fdata):
        """
        Initialize a SingleCellAssay object
        
        Parameters
        ----------
        expression_matrix : numpy.ndarray
            Gene expression matrix with shape (n_genes, n_cells)
        cdata : pandas.DataFrame
            Cell metadata with shape (n_cells, n_cell_metadata)
        fdata : pandas.DataFrame
            Feature (gene) metadata with shape (n_genes, n_feature_metadata)
        """
        self.expression_matrix = expression_matrix
        self.cdata = cdata
        self.fdata = fdata
        
    def assay(self):
        """
        Get the expression matrix
        
        Returns
        -------
        numpy.ndarray
            Gene expression matrix
        """
        return self.expression_matrix
    
    def colData(self):
        """
        Get the cell metadata
        
        Returns
        -------
        pandas.DataFrame
            Cell metadata
        """
        return self.cdata
    
    def mcols(self):
        """
        Get the feature metadata
        
        Returns
        -------
        pandas.DataFrame
            Feature metadata
        """
        return self.fdata
    
    def nrow(self):
        """
        Get the number of genes
        
        Returns
        -------
        int
            Number of genes
        """
        return self.expression_matrix.shape[0]
    
    def ncol(self):
        """
        Get the number of cells
        
        Returns
        -------
        int
            Number of cells
        """
        return self.expression_matrix.shape[1]