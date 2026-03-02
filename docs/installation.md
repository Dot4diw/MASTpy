# Installation

## From Source

To install MASTpy from source, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MASTpy.git
   cd MASTpy
   ```

2. **Install the package**
   ```bash
   pip install -e .
   ```

## Dependencies

MASTpy requires the following dependencies:

- **numpy**: Numerical computing library
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library
- **scipy**: Scientific computing library
- **statsmodels**: Statistical models and tests
- **numba**: Just-in-time compilation for performance
- **tqdm**: Progress bar for long-running tasks
- **anndata**: AnnData object support for single-cell data

## Verify Installation

To verify that MASTpy is installed correctly, run the following command:

```python
import mastpy
print(f"MASTpy version: {mastpy.__version__}")
```

You should see the version number printed without any errors.
