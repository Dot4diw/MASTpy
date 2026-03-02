# Basic Usage

## Using SingleCellAssay

### Creating a SingleCellAssay

```python
import numpy as np
import pandas as pd
from mastpy import SingleCellAssay, zlm

# Create test data
expression_matrix = np.random.poisson(lam=0.5, size=(100, 50))  # (genes, cells)
cdata = pd.DataFrame({
    'condition': np.random.choice(['A', 'B'], size=50),
    'ncells': np.ones(50, dtype=int)
}, index=[f'cell_{i}' for i in range(50)])

fdata = pd.DataFrame(index=[f'gene_{i}' for i in range(100)])

# Create SingleCellAssay
sca = SingleCellAssay(expression_matrix, cdata, fdata)
```

### Fitting a Zero-Inflated Model

```python
# Fit zlm model
zfit = zlm('~ condition', sca, method='glm', use_ebayes=True, n_jobs=10)

# Access results
print(f"Number of genes: {zfit.coefC.shape[0]}")
print(f"Continuous coefficients shape: {zfit.coefC.shape}")
print(f"Discrete coefficients shape: {zfit.coefD.shape}")
```

## Using AnnData with find_deg

### Basic Differential Expression Analysis

```python
import anndata as ad
from mastpy.tools import find_deg

# Load your AnnData object
adata = ad.read_h5ad('datasets/cs_ciliated.h5ad')

# Perform differential expression analysis
deg_results = find_deg(
    adata=adata,
    groupby='condition',
    ident_1='CS',
    ident_2='WT',
    layer='log1p_norm',  # Use log1p normalized data
    logfc_threshold=0.1,
    min_pct=0.01,
    test_use='MAST',
    test_method='wald',  # 'wald' or 'lr'
    n_jobs=10,
    only_pos=False,
    verbose=True
)

# View results
print(deg_results.head())
```

### Interpreting Results

The `find_deg` function returns a DataFrame with the following columns:

- **p_val**: Raw p-values
- **padj**: Adjusted p-values (Benjamini-Hochberg correction)
- **log2FC**: Log2 fold change
- **avg_log2FC**: Average log2 fold change
- **pct.1**: Percentage of cells expressing the gene in the first group
- **pct.2**: Percentage of cells expressing the gene in the second group
- **diff_pct**: Difference in expression percentage between groups

## Example Workflow

1. **Load data**
2. **Preprocess** (normalization, filtering)
3. **Perform DE analysis**
4. **Filter significant genes**
5. **Visualize results**

```python
# Example of filtering significant genes
significant_genes = deg_results[(deg_results['padj'] < 0.05) & (abs(deg_results['log2FC']) > 0.25)]
print(f"Found {len(significant_genes)} significant genes")

# Visualize top genes
import matplotlib.pyplot as plt
import seaborn as sns

top_genes = significant_genes.nlargest(10, 'abs(log2FC)')
sns.barplot(x='log2FC', y=top_genes.index, data=top_genes)
plt.title('Top 10 Differentially Expressed Genes')
plt.tight_layout()
plt.show()
```
