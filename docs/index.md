# MASTpy Documentation

## Overview

MASTpy is a Python implementation of the MAST (Model-based Analysis of Single-cell Transcriptomics) package originally written in R. It provides methods for analyzing single cell assay data using hurdle models, with optimized performance using Numba and multi-threading.

## Key Features

- **Zero-inflated regression models** for single-cell data
- **Empirical Bayes variance shrinkage** for improved statistical power
- **Parallel processing** for faster computation
- **AnnData integration** for seamless workflow with single-cell data
- **Support for both Wald and Likelihood Ratio tests** for hypothesis testing
- **Flexible data layer selection** for analysis

## Getting Started

- [Installation](installation.md)
- [Basic Usage](basic_usage.md)
- [Advanced Usage](advanced_usage.md)
- [API Reference](api_reference.md)

## Example Workflow

1. **Load your single-cell data** as an AnnData object
2. **Perform differential expression analysis** using `find_deg` function
3. **Explore results** and visualize significant genes
4. **Compare with R MAST** results for validation

## References

Finak G, McDavid A, Yajima M, Deng J, Gersuk V, Shalek AK, Slichter CH, Miller H, McElrath MJ, Prlic M, et al. MAST: a flexible statistical framework for assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data. Genome Biol. 2015;16:278. doi: 10.1186/s13059-015-0844-5.
