# MASTpy

Model-based Analysis of Single Cell Transcriptomics in Python

A Python implementation of the R MAST package for zero-inflated model analysis of single-cell transcriptomics data.

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- anndata >= 0.8.0
- scanpy >= 1.9.0
- patsy >= 0.5.0
- joblib

## Quick Start

```python
import numpy as np
import pandas as pd
import anndata
import mastpy as mt

# Load data
adata = anndata.read_h5ad('your_data.h5ad')

# Create group variable
adata.obs['group'] = 'Other'
adata.obs.loc[adata.obs['cluster'] == 'Cluster1', 'group'] = 'Cluster1'

# Fit zero-inflated model
zlmfit = mt.zlm(
    formula='~ group',
    adata=adata,
    method='bayesglm',
    ebayes=True,
    parallel=True,
    n_jobs=-1,
)

# Wald test
result = mt.waldTest(zlmfit, 'group')

# Get significant genes
sig_genes = result[result['Pr(>Chisq)'] < 0.05]
print(f"Significant genes: {len(sig_genes)}")
```

## Main Functions

- **zlm()**: Fit zero-inflated linear model (hurdle model)
- **waldTest()**: Wald test
- **lrTest()**: Likelihood ratio test
- **from_matrix()**: Create AnnData from matrix
- **from_flat_df()**: Create AnnData from flat dataframe

## Parameters

### zlm()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| formula | str | - | Model formula (e.g., `~ group + batch`) |
| adata | AnnData | - | Single cell data |
| method | str | 'bayesglm' | Fitting method: 'bayesglm', 'glm', 'ridge' |
| ebayes | bool | True | Use empirical Bayes variance shrinkage |
| parallel | bool | True | Use multi-threading |
| n_jobs | int | -1 | Number of cores (-1 for all) |

### waldTest()

| Parameter | Type | Description |
|-----------|------|-------------|
| zlmfit | ZlmFit | Output from zlm() |
| hypothesis | str | Hypothesis to test (e.g., 'group') |

## Output Format

Wald test results contain:

| Column | Description |
|--------|-------------|
| primerid | Gene name |
| lambda | Chi-square statistic |
| df | Degrees of freedom |
| Pr(>Chisq) | P-value |

## Reference

Original R MAST package: https://github.com/RGLab/MAST
