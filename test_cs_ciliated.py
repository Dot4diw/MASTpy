"""Test script for cs_ciliated.h5ad dataset"""

import numpy as np
import pandas as pd
import anndata as ad
from mastpy import find_deg
import os

if __name__ == '__main__':
    # Load the dataset
    print("Loading cs_ciliated.h5ad dataset...")
    adata_path = "d:\\MyCode\\Python\\MAST-devel\\MASTpy\\datasets\\cs_ciliated.h5ad"
    
    if not os.path.exists(adata_path):
        print(f"Error: Dataset not found at {adata_path}")
        exit(1)
    
    adata = ad.read_h5ad(adata_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {adata.shape}")
    print(f"Layers: {list(adata.layers.keys())}")
    print(f"Obs columns: {list(adata.obs.columns)}")
    
    # Check if log1p_norm layer exists
    if 'log1p_norm' not in adata.layers:
        print("Error: log1p_norm layer not found in the dataset")
        exit(1)
    
    # Check group column
    if 'group' not in adata.obs:
        print("Error: group column not found in obs")
        exit(1)
    
    print(f"Group values: {adata.obs['group'].unique()}")
    
    # Run differential expression analysis using log1p_norm layer
    print("\nRunning differential expression analysis with log1p_norm layer...")
    try:
        deg_results = find_deg(
            adata=adata,
            groupby='group',
            ident_1='CS',
            ident_2='WT',
            layer='log1p_norm',
            logfc_threshold=0,
            min_pct=0,
            test_method='lr',
            verbose=True,
            n_jobs=1
        )
        print(f"✓ Differential expression analysis completed")
        print(f"Found {len(deg_results)} differentially expressed genes")
        print(f"Top 10 genes:")
        print(deg_results.head(10))
        
        # Save results
        deg_results.to_csv("mastpy_cs_results.csv", index=True)
        print("\nResults saved as 'mastpy_cs_results.csv'")
    except Exception as e:
        print(f"✗ Failed to run differential expression analysis: {e}")
        exit(1)
    
    # Extract data for Seurat
    print("\nExtracting data for Seurat...")
    
    # Get expression matrices
    counts = adata.X  # raw counts
    log1p_norm = adata.layers['log1p_norm']  # log1p normalized data
    
    # Convert to dense matrices if sparse
    if hasattr(counts, 'toarray'):
        counts = counts.toarray()
    if hasattr(log1p_norm, 'toarray'):
        log1p_norm = log1p_norm.toarray()
    
    # Create dataframes (genes x cells format for Seurat)
    expression_df = pd.DataFrame(counts.T, index=adata.var.index, columns=adata.obs.index)
    log1p_norm_df = pd.DataFrame(log1p_norm.T, index=adata.var.index, columns=adata.obs.index)
    cell_metadata = adata.obs.copy()
    feature_metadata = adata.var.copy()
    
    # Save data for Seurat
    print("Saving data for Seurat...")
    expression_df.to_csv("cs_expression_matrix.csv")
    log1p_norm_df.to_csv("cs_log1p_norm_matrix.csv")
    cell_metadata.to_csv("cs_cell_metadata.csv")
    feature_metadata.to_csv("cs_feature_metadata.csv")
    
    print("\nFiles saved:")
    print("- cs_expression_matrix.csv (raw counts)")
    print("- cs_log1p_norm_matrix.csv (log1p normalized)")
    print("- cs_cell_metadata.csv (cell metadata)")
    print("- cs_feature_metadata.csv (gene metadata)")
    
    print("\nTest completed successfully!")
