"""Test script for find_deg module"""

import numpy as np
import pandas as pd
import anndata as ad
from mastpy import find_deg, find_all_degs

if __name__ == '__main__':
    # Create test data
    np.random.seed(42)

    # Generate expression matrix (100 genes, 100 cells)
    n_genes = 100
    n_cells = 100
    expression_matrix = np.random.poisson(lam=0.5, size=(n_genes, n_cells))

    # Generate cell metadata
    cdata = pd.DataFrame({
        'condition': np.random.choice(['A', 'B'], size=n_cells),
        'ncells': np.ones(n_cells, dtype=int)
    })

    # Generate feature metadata
    fdata = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])

    # Create AnnData object
    adata = ad.AnnData(
        X=expression_matrix.T,  # AnnData expects (n_cells, n_genes)
        obs=cdata,
        var=fdata
    )

    # Test find_deg function
    print("Testing find_deg function...")
    deg_results = find_deg(
        adata=adata,
        groupby='condition',
        ident_1='A',
        ident_2='B',
        layer='X',
        logfc_threshold=0,  # No filtering
        min_pct=0,  # No filtering
        test_method='lr',  # Use likelihood ratio test to match Seurat MAST
        verbose=True,
        n_jobs=1  # Set to 1 to avoid multiprocessing issues on Windows
    )

    # Sort by gene name
    deg_results = deg_results.sort_index()

    print("\nDifferential expression results:")
    print(deg_results.head())
    print(f"\nFound {len(deg_results)} differentially expressed genes")
    # Save test data for Seurat testing
    deg_results.to_csv("mastpy_results.csv", index=True)
    print("\nSaving test data for Seurat testing...")
    
    # Save expression matrix as CSV (genes x cells)
    expression_df = pd.DataFrame(expression_matrix, index=fdata.index, columns=cdata.index)
    expression_df.to_csv("test_expression_matrix.csv")
    
    # Save cell metadata as CSV
    cdata.to_csv("test_cell_metadata.csv")
    
    # Save feature metadata as CSV
    fdata.to_csv("test_feature_metadata.csv")
    
    print("Test data saved successfully!")
    print("Files saved:")
    print("- test_expression_matrix.csv (genes x cells)")
    print("- test_cell_metadata.csv (cell metadata)")
    print("- test_feature_metadata.csv (gene metadata)")
    
    print("\nTest completed successfully!")