"""
Example script demonstrating MAST-py usage.

This script shows how to:
1. Create an AnnData object from expression data
2. Fit zero-inflated models using zlm()
3. Perform Wald and likelihood ratio tests
"""

import numpy as np
import pandas as pd
import anndata
import mastpy as mt


def create_example_data(n_cells=200, n_genes=100):
    """Create example single-cell data for demonstration."""
    np.random.seed(42)

    group = np.random.choice(["A", "B"], n_cells)
    batch = np.random.choice(["1", "2"], n_cells)

    expr = np.random.randn(n_genes, n_cells) + 5

    zero_rate = 0.3
    zero_mask = np.random.random((n_genes, n_cells)) < zero_rate
    expr[zero_mask] = 0

    for g in range(n_genes):
        if group[g == 0] == "B":
            expr[g, group == "B"] += 0.5

    obs = pd.DataFrame(
        {
            "wellKey": [f"cell_{i}" for i in range(n_cells)],
            "group": group,
            "batch": batch,
        }
    )
    obs.index = [f"cell_{i}" for i in range(n_cells)]

    var = pd.DataFrame(
        {
            "primerid": [f"gene_{i}" for i in range(n_genes)],
        }
    )
    var.index = [f"gene_{i}" for i in range(n_genes)]

    adata = anndata.AnnData(
        X=expr,
        obs=obs,
        var=var,
    )

    return adata


def main():
    print("Creating example data...")
    adata = create_example_data()
    print(f"Created AnnData with {adata.n_obs} cells and {adata.n_vars} genes")

    print("\nFitting zero-inflated model...")
    zlmfit = mt.zlm(
        formula="~ group + batch",
        adata=adata,
        method="bayesglm",
        ebayes=True,
        silent=False,
    )

    print(f"\nZlmFit result: {zlmfit}")
    print(f"\nCoefficients (continuous):")
    print(zlmfit.coefC.head())

    print(f"\nCoefficients (discrete):")
    print(zlmfit.coefD.head())

    print("\nPerforming Wald test for group effect...")
    result = mt.waldTest(zlmfit, "groupB")
    print(result.head(10))

    print("\nPerforming likelihood ratio test...")
    lr_result = mt.lrTest(zlmfit, "group")
    print(lr_result.head(10))

    print("\nDone!")


if __name__ == "__main__":
    main()
