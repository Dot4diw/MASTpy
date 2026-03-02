# API Reference

## Core Classes

### SingleCellAssay

```python
class SingleCellAssay(X, cdata, fdata)
```

#### Parameters
- **X**: `numpy.ndarray` - Expression matrix (genes × cells)
- **cdata**: `pandas.DataFrame` - Cell metadata
- **fdata**: `pandas.DataFrame` - Feature (gene) metadata

#### Attributes
- **X**: Expression matrix
- **cdata**: Cell metadata
- **fdata**: Feature metadata
- **n_genes**: Number of genes
- **n_cells**: Number of cells

### ZlmFit

```python
class ZlmFit()
```

#### Methods
- **lrTest(contrast)**: Perform likelihood ratio test
- **waldTest(contrast)**: Perform Wald test
- **vcov()**: Get variance-covariance matrix

#### Attributes
- **coefC**: Continuous component coefficients
- **coefD**: Discrete component coefficients
- **vcovC**: Continuous component variance-covariance matrix
- **vcovD**: Discrete component variance-covariance matrix

## Core Functions

### zlm

```python
def zlm(formula, sca, method='glm', use_ebayes=True, n_jobs=1, chunk_size=None, verbose=False)
```

#### Parameters
- **formula**: `str` - Formula for the model (e.g., '~ condition')
- **sca**: `SingleCellAssay` - SingleCellAssay object
- **method**: `str` - Model fitting method ('glm' or 'glmrob')
- **use_ebayes**: `bool` - Whether to use empirical Bayes variance shrinkage
- **n_jobs**: `int` - Number of parallel jobs to use
- **chunk_size**: `int` - Number of genes to process per chunk
- **verbose**: `bool` - Whether to print progress

#### Returns
- **ZlmFit** object with fitted model results

## Tools Functions

### find_deg

```python
def find_deg(adata, groupby, ident_1, ident_2=None, layer='counts', logfc_threshold=0.1, min_pct=0.01, test_use='MAST', test_method='wald', n_jobs=10, only_pos=False, verbose=True)
```

#### Parameters
- **adata**: `anndata.AnnData` - AnnData object
- **groupby**: `str` - Column name in obs for grouping
- **ident_1**: `str` - First group identifier
- **ident_2**: `str` - Second group identifier (optional, if None compare to all other groups)
- **layer**: `str` - Layer in adata to use for expression data
- **logfc_threshold**: `float` - Minimum log2 fold change threshold
- **min_pct**: `float` - Minimum percentage of cells expressing the gene
- **test_use**: `str` - Statistical test to use ('MAST' only currently)
- **test_method**: `str` - Test method ('wald' or 'lr')
- **n_jobs**: `int` - Number of parallel jobs to use
- **only_pos**: `bool` - Whether to only return upregulated genes
- **verbose**: `bool` - Whether to print progress

#### Returns
- **pandas.DataFrame** with differential expression results

## Utility Functions

### fit_zi_glm

```python
def fit_zi_glm(y, X, offset=None, method='glm', verbose=False)
```

#### Parameters
- **y**: `numpy.ndarray` - Response variable
- **X**: `numpy.ndarray` - Design matrix
- **offset**: `numpy.ndarray` - Offset term
- **method**: `str` - Fitting method ('glm' or 'glmrob')
- **verbose**: `bool` - Whether to print progress

#### Returns
- **dict** with fitted model parameters

### ebayes

```python
def ebayes(fit, proportion=0.01, trend=True, robust=False, winsorize=True)
```

#### Parameters
- **fit**: `ZlmFit` - Fitted ZlmFit object
- **proportion**: `float` - Proportion of genes to use for variance estimation
- **trend**: `bool` - Whether to include trend in variance estimation
- **robust**: `bool` - Whether to use robust variance estimation
- **winsorize**: `bool` - Whether to winsorize outliers

#### Returns
- **ZlmFit** object with empirical Bayes variance shrinkage

## Example API Usage

### Basic ZlmFit Usage

```python
from mastpy import SingleCellAssay, zlm

# Create SingleCellAssay
sca = SingleCellAssay(expression_matrix, cdata, fdata)

# Fit model
zfit = zlm('~ condition', sca, method='glm', use_ebayes=True, n_jobs=10)

# Perform tests
import numpy as np
contrast = np.zeros(zfit.coefC.shape[1])
contrast[1] = 1  # Test condition effect

# Wald test
wald_results = zfit.waldTest(contrast)

# Likelihood ratio test
lr_results = zfit.lrTest(contrast)
```

### Advanced find_deg Usage

```python
from mastpy.tools import find_deg

# Perform differential expression analysis with custom parameters
deg_results = find_deg(
    adata=adata,
    groupby='condition',
    ident_1='CS',
    ident_2='WT',
    layer='log1p_norm',
    logfc_threshold=0.25,
    min_pct=0.05,
    test_method='lr',
    n_jobs=20,
    only_pos=True,
    verbose=True
)

# Filter and analyze results
significant_genes = deg_results[deg_results['padj'] < 0.05]
print(f"Found {len(significant_genes)} significant genes")
```
