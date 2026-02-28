"""Test script for find_deg module"""

import numpy as np
import pandas as pd
import anndata as ad
from mastpy import find_deg, find_all_degs

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
    logfc_threshold=0.1,
    min_pct=0.01,
    verbose=True
)

print("\nDifferential expression results:")
print(deg_results.head())
print(f"\nFound {len(deg_results)} differentially expressed genes")

# Test find_all_degs function
print("\nTesting find_all_degs function...")
all_markers = find_all_degs(
    adata=adata,
    groupby='condition',
    layer='X',
    logfc_threshold=0.1,
    min_pct=0.01,
    verbose=True
)

print("\nAll markers results:")
print(all_markers.head())
print(f"\nFound {len(all_markers)} markers in total")

print("\nTest completed successfully!")