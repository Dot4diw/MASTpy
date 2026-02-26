"""Test script for MASTpy using pancreas_day15.h5ad data."""

import sys

import time
import numpy as np
import pandas as pd
import anndata
import mastpy as mt


def main():
    start_time = time.time()

    print("Loading pancreas_day15.h5ad data...")
    adata = anndata.read_h5ad(
        "pancreas_day15.h5ad"
    )
    print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")

    # adata = adata[:, 1000:2000].copy()
    # print(f"Using subset: {adata.n_obs} cells and {adata.n_vars} genes")

    print("\nCluster distribution:")
    print(adata.obs["clusters"].value_counts())

    first_cluster = adata.obs["clusters"].unique()[0]
    print(f"\nFirst cluster: {first_cluster}")

    adata.obs["group"] = "Other"
    adata.obs.loc[adata.obs["clusters"] == first_cluster, "group"] = first_cluster

    print(f"\nGroup distribution:")
    print(adata.obs["group"].value_counts())

    print("\n" + "=" * 50)
    print("Fitting zero-inflated model (parallel processing)...")
    print("=" * 50)

    fit_start = time.time()
    zlmfit = mt.zlm(
        formula="~ group",
        adata=adata,
        method="bayesglm",
        ebayes=True,
        silent=False,
        parallel=True,
        n_jobs=-1,
    )
    fit_time = time.time() - fit_start
    print(f"\nModel fitting time: {fit_time:.2f} seconds")

    print(f"\nZlmFit result: {zlmfit}")
    print(f"\nConvergence (C): {zlmfit.converged['C'].sum()} / {zlmfit.ngenes}")
    print(f"Convergence (D): {zlmfit.converged['D'].sum()} / {zlmfit.ngenes}")

    both_converged = zlmfit.converged["C"] & zlmfit.converged["D"]
    print(f"Convergence (both): {both_converged.sum()} / {zlmfit.ngenes}")

    non_zero_coef = zlmfit.coefC.abs().sum(axis=1) > 0.01
    print(f"Non-zero continuous coef: {non_zero_coef.sum()}")

    print("\n" + "=" * 50)
    print("Performing Wald test...")
    print("=" * 50)

    non_zero_idx = (
        (zlmfit.coefC.abs().sum(axis=1) > 0.01)
        & (zlmfit.converged["C"])
        & (zlmfit.converged["D"])
    )
    non_zero_genes = zlmfit.coefC[non_zero_idx].index.tolist()[:10]
    print(f"Genes with non-zero coefficients: {non_zero_genes[:5]}")

    test_start = time.time()
    result = mt.waldTest(zlmfit, "group")
    result.to_csv("wald_test_results.csv", index=False)
    test_time = time.time() - test_start
    print(f"\nWald test time: {test_time:.2f} seconds")

    print(f"\nWald test results (using 'group'):")
    print(result.head(20))

    sig_genes = result[result["Pr(>Chisq)"] < 0.05]
    print(f"\nSignificant genes (p < 0.05): {len(sig_genes)}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"{'=' * 50}")

    print("\nDone!")


if __name__ == "__main__":
    main()
