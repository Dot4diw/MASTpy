# Advanced Usage

## Custom Models and Formulas

### Using Complex Formulas

You can use more complex formulas in the `zlm` function to include multiple covariates:

```python
# Fit model with multiple covariates
zfit = zlm('~ condition + age + batch', sca, method='glm', use_ebayes=True, n_jobs=10)
```

### Using Custom Contrasts

For more complex hypothesis testing, you can use custom contrasts:

```python
# Fit model
zfit = zlm('~ condition + age', sca, method='glm', use_ebayes=True, n_jobs=10)

# Define custom contrast
import numpy as np
contrast = np.zeros(zfit.coefC.shape[1])
contrast[1] = 1  # Test condition effect

# Perform Wald test with custom contrast
wald_results = zfit.waldTest(contrast)
```

## Advanced find_deg Parameters

### Fine-tuning DE Analysis

```python
deg_results = find_deg(
    adata=adata,
    groupby='condition',
    ident_1='CS',
    ident_2='WT',
    layer='log1p_norm',
    logfc_threshold=0.1,
    min_pct=0.01,
    test_use='MAST',
    test_method='lr',  # Use likelihood ratio test
    n_jobs=20,  # More parallel jobs for faster computation
    only_pos=True,  # Only return upregulated genes
    verbose=True
)
```

### Handling Large Datasets

For large datasets, you can optimize performance:

```python
# For very large datasets
zfit = zlm(
    '~ condition', 
    sca, 
    method='glm', 
    use_ebayes=True, 
    n_jobs=-1,  # Use all available cores
    chunk_size=1000  # Process genes in chunks
)
```

## Comparing with R MAST

### Exporting Data for R Analysis

```python
import pandas as pd

# Extract data from AnnData
counts = adata.layers['counts']
log1p_norm = adata.layers['log1p_norm']
metadata = adata.obs

# Create dataframes (genes x cells format for Seurat)
expression_df = pd.DataFrame(counts.T, index=adata.var.index, columns=adata.obs.index)
log1p_norm_df = pd.DataFrame(log1p_norm.T, index=adata.var.index, columns=adata.obs.index)
metadata_df = metadata.copy()

# Save to CSV
expression_df.to_csv('expression_matrix.csv')
log1p_norm_df.to_csv('log1p_norm_matrix.csv')
metadata_df.to_csv('metadata.csv')
```

### Running Seurat MAST in R

```r
# In R
library(Seurat)

# Load data
expression_matrix <- read.csv('expression_matrix.csv', row.names = 1)
log1p_norm_matrix <- read.csv('log1p_norm_matrix.csv', row.names = 1)
metadata <- read.csv('metadata.csv', row.names = 1)

# Create Seurat object
seurat_obj <- CreateSeuratObject(counts = expression_matrix, meta.data = metadata)

# Add log1p_norm to data layer
seurat_obj <- SetAssayData(seurat_obj, layer = "data", new.data = as.matrix(log1p_norm_matrix))

# Set identities
Idents(seurat_obj) <- seurat_obj$condition

# Find markers using MAST
mast_markers <- FindMarkers(
    seurat_obj,
    ident.1 = "CS",
    ident.2 = "WT",
    test.use = "MAST",
    logfc.threshold = 0,
    min.pct = 0,
    verbose = TRUE
)

# Save results
write.csv(mast_markers, 'seurat_mast_results.csv')
```

### Comparing Results

```python
# Compare Python and R results
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load results
mastpy_df = pd.read_csv('mastpy_results.csv', index_col=0)
seurat_df = pd.read_csv('seurat_mast_results.csv', index_col=0)

# Align results by gene name
common_genes = mastpy_df.index.intersection(seurat_df.index)
mastpy_df = mastpy_df.loc[common_genes]
seurat_df = seurat_df.loc[common_genes]

# Calculate correlations
log2fc_corr, log2fc_p = pearsonr(mastpy_df['avg_log2FC'], seurat_df['avg_log2FC'])
print(f"Log2FC Pearson correlation: {log2fc_corr:.4f} (p-value: {log2fc_p:.4f})")

pval_corr, pval_p = spearmanr(-np.log10(mastpy_df['p_val']), -np.log10(seurat_df['p_val']))
print(f"-log10(p-value) Spearman correlation: {pval_corr:.4f} (p-value: {pval_p:.4f})")
```

## Troubleshooting

### Common Issues

1. **p-values all 1**
   - Check that your contrast is correctly defined
   - Ensure you're using the appropriate test method (wald vs lr)
   - Verify that your data has sufficient variation

2. **Memory errors with large datasets**
   - Use `chunk_size` parameter to process genes in batches
   - Reduce `n_jobs` to limit memory usage
   - Consider subsetting your data for initial testing

3. **Convergence issues**
   - Try using a different `method` parameter
   - Check for outliers in your data
   - Ensure your covariates are properly scaled

4. **Parallel processing errors**
   - On Windows, set `n_jobs=1` to avoid multiprocessing issues
   - Ensure your code is wrapped in `if __name__ == '__main__':` block
