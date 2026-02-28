# MASTpy

Model-based Analysis of Single-cell Transcriptomics in Python

## Overview

MASTpy is a Python implementation of the MAST (Model-based Analysis of Single-cell Transcriptomics) package originally written in R. It provides methods for analyzing single cell assay data using hurdle models, with optimized performance using Numba and multi-threading.

## Features

- Zero-inflated regression models for single-cell data
- Empirical Bayes variance shrinkage
- Parallel processing for faster computation
- Compatible with Python 3.7+
- Optimized performance using Numba

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/MASTpy.git
cd MASTpy

# Install the package
pip install -e .
```

### Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- statsmodels
- numba
- tqdm

## Usage

### Basic usage

```python
import numpy as np
import pandas as pd
from mastpy import SingleCellAssay, zlm

# Create test data
expression_matrix = np.random.poisson(lam=0.5, size=(100, 50))
cdata = pd.DataFrame({
    'condition': np.random.choice(['A', 'B'], size=50),
    'ncells': np.ones(50, dtype=int)
})
fdata = pd.DataFrame(index=[f'gene_{i}' for i in range(100)])

# Create SingleCellAssay
sca = SingleCellAssay(expression_matrix, cdata, fdata)

# Fit zlm model
zfit = zlm('~ condition', sca, method='glm', use_ebayes=True, parallel=True)

# Access results
print(f"Number of genes: {zfit.coefC.shape[0]}")
print(f"Continuous coefficients shape: {zfit.coefC.shape}")
print(f"Discrete coefficients shape: {zfit.coefD.shape}")
```

### Advanced usage

For more advanced usage, see the examples in the `examples/` directory.

## Directory Structure


## License

MIT License

## Acknowledgments

This package is based on the original MAST package written in R: https://github.com/RGLab/MAST

## References

Finak G, McDavid A, Yajima M, Deng J, Gersuk V, Shalek AK, Slichter CH, Miller H, McElrath MJ, Prlic M, et al. MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data. Genome Biol. 2015;16:278. doi: 10.1186/s13059-015-0844-5.