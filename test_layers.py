"""Test script to verify find_deg works with different layers"""

import numpy as np
import pandas as pd
import anndata as ad
from mastpy import find_deg

if __name__ == '__main__':
    # Create test data
    np.random.seed(42)

    # Generate expression matrix (100 genes, 100 cells)
    n_genes = 50
    n_cells = 50
    
    # Generate raw counts (Poisson distributed)
    counts = np.random.poisson(lam=0.5, size=(n_genes, n_cells))
    
    # Generate log1p normalized data
    log1p_norm = np.log1p(counts)

    # Generate cell metadata
    cdata = pd.DataFrame({
        'condition': np.random.choice(['A', 'B'], size=n_cells),
        'ncells': np.ones(n_cells, dtype=int)
    })

    # Generate feature metadata
    fdata = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])

    # Create AnnData object with multiple layers
    adata = ad.AnnData(
        X=counts.T,  # AnnData expects (n_cells, n_genes)
        obs=cdata,
        var=fdata
    )
    
    # Add log1p_norm layer
    adata.layers['log1p_norm'] = log1p_norm.T
    
    print("Testing find_deg with different layers...")
    
    # Test with raw counts (X layer)
    print("\n1. Testing with raw counts (X layer):")
    try:
        results_counts = find_deg(
            adata=adata,
            groupby='condition',
            ident_1='A',
            ident_2='B',
            layer='X',
            logfc_threshold=0,
            min_pct=0,
            test_method='lr',
            verbose=True,
            n_jobs=1
        )
        print(f"✓ Successfully ran with raw counts")
        print(f"  Found {len(results_counts)} genes")
        print(f"  Top 5 genes:")
        print(results_counts.head())
    except Exception as e:
        print(f"✗ Failed with raw counts: {e}")
    
    # Test with log1p_norm layer
    print("\n2. Testing with log1p_norm layer:")
    try:
        results_log1p = find_deg(
            adata=adata,
            groupby='condition',
            ident_1='A',
            ident_2='B',
            layer='log1p_norm',
            logfc_threshold=0,
            min_pct=0,
            test_method='lr',
            verbose=True,
            n_jobs=1
        )
        print(f"✓ Successfully ran with log1p_norm")
        print(f"  Found {len(results_log1p)} genes")
        print(f"  Top 5 genes:")
        print(results_log1p.head())
    except Exception as e:
        print(f"✗ Failed with log1p_norm: {e}")
    
    # Compare results
    print("\n3. Comparing results:")
    if 'results_counts' in locals() and 'results_log1p' in locals():
        # Check if gene lists are the same
        common_genes = set(results_counts.index) & set(results_log1p.index)
        print(f"Number of common genes: {len(common_genes)}")
        
        # Check correlation of p-values
        if len(common_genes) > 0:
            common_genes_sorted = sorted(common_genes)
            pvals_counts = results_counts.loc[common_genes_sorted]['p_val']
            pvals_log1p = results_log1p.loc[common_genes_sorted]['p_val']
            
            from scipy.stats import spearmanr
            corr, p_val = spearmanr(pvals_counts, pvals_log1p)
            print(f"Spearman correlation of p-values: {corr:.4f} (p-value: {p_val:.4f})")
    
    print("\nTest completed!")
