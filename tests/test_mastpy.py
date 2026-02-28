import numpy as np
import pandas as pd
from mastpy import SingleCellAssay, zlm

if __name__ == '__main__':
    # Create test data
    np.random.seed(42)
    
    # Generate expression matrix (100 genes, 50 cells)
    expression_matrix = np.random.poisson(lam=0.5, size=(100, 50))
    
    # Generate cell metadata
    cdata = pd.DataFrame({
        'condition': np.random.choice(['A', 'B'], size=50),
        'ncells': np.ones(50, dtype=int)
    })
    
    # Generate feature metadata
    fdata = pd.DataFrame(index=[f'gene_{i}' for i in range(100)])
    
    # Create SingleCellAssay
    sca = SingleCellAssay(expression_matrix, cdata, fdata)
    
    print("Testing MASTpy...")
    print(f"Expression matrix shape: {expression_matrix.shape}")
    print(f"Cell metadata shape: {cdata.shape}")
    print(f"Feature metadata shape: {fdata.shape}")
    
    # Test zlm function
    print("\nFitting zlm model...")
    zfit = zlm('~ condition', sca, method='glm', use_ebayes=True, parallel=True)
    
    print("\nZlmFit object created successfully!")
    print(f"Number of genes: {zfit.coefC.shape[0]}")
    print(f"Number of coefficients: {zfit.coefC.shape[1]}")
    print(f"Continuous coefficients shape: {zfit.coefC.shape}")
    print(f"Discrete coefficients shape: {zfit.coefD.shape}")
    print(f"Variance-covariance matrix shape: {zfit.vcovC.shape}")
    
    print("\nMASTpy test completed successfully!")
