"""Compare MASTpy and Seurat MAST results for cs_ciliated dataset"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Read results
mastpy_df = pd.read_csv("mastpy_cs_results.csv", index_col="gene")
seurat_df = pd.read_csv("seurat_cs_results.csv", index_col=0)

# Ensure both dataframes have the same genes in the same order
common_genes = sorted(list(set(mastpy_df.index) & set(seurat_df.index)))
mastpy_df = mastpy_df.loc[common_genes]
seurat_df = seurat_df.loc[common_genes]

# Calculate correlations
print("Correlation analysis:")

# Pearson correlation for log2FC
log2fc_corr, log2fc_p = pearsonr(mastpy_df['avg_log2FC'], seurat_df['avg_log2FC'])
print(f"Log2FC Pearson correlation: {log2fc_corr:.4f} (p-value: {log2fc_p:.4f})")

# Spearman correlation for p-values (since p-values are not normally distributed)
pval_corr, pval_p = spearmanr(-np.log10(mastpy_df['p_val']), -np.log10(seurat_df['p_val']))
print(f"-log10(p-value) Spearman correlation: {pval_corr:.4f} (p-value: {pval_p:.4f})")

# Plot comparisons
plt.figure(figsize=(12, 5))

# Log2FC comparison
plt.subplot(1, 2, 1)
plt.scatter(mastpy_df['avg_log2FC'], seurat_df['avg_log2FC'], alpha=0.5)
plt.xlabel('MASTpy avg_log2FC')
plt.ylabel('Seurat MAST avg_log2FC')
plt.title(f'Log2FC comparison (r={log2fc_corr:.4f})')
plt.grid(True)

# P-value comparison
plt.subplot(1, 2, 2)
plt.scatter(-np.log10(mastpy_df['p_val']), -np.log10(seurat_df['p_val']), alpha=0.5)
plt.xlabel('-log10(MASTpy p-value)')
plt.ylabel('-log10(Seurat MAST p-value)')
plt.title(f'P-value comparison (ρ={pval_corr:.4f})')
plt.grid(True)

plt.tight_layout()
plt.savefig('cs_comparison_plots.png')
print("\nPlots saved as 'cs_comparison_plots.png'")

# Top genes comparison
print("\nTop 10 genes by MASTpy p-value:")
print(mastpy_df.sort_values('p_val').head(10))

print("\nTop 10 genes by Seurat MAST p-value:")
print(seurat_df.sort_values('p_val').head(10))

# Save combined results
combined_df = pd.DataFrame({
    'MASTpy_log2FC': mastpy_df['avg_log2FC'],
    'Seurat_log2FC': seurat_df['avg_log2FC'],
    'MASTpy_pval': mastpy_df['p_val'],
    'Seurat_pval': seurat_df['p_val'],
    'MASTpy_padj': mastpy_df['p_val_adj'],
    'Seurat_padj': seurat_df['p_val_adj']
})

combined_df.to_csv('cs_combined_results.csv')
print("\nCombined results saved as 'cs_combined_results.csv'")

# Calculate number of genes with p-value < 0.05
mastpy_sig = (mastpy_df['p_val'] < 0.05).sum()
seurat_sig = (seurat_df['p_val'] < 0.05).sum()
print(f"\nNumber of significant genes (p < 0.05):")
print(f"MASTpy: {mastpy_sig}")
print(f"Seurat: {seurat_sig}")

print("\nAnalysis completed successfully!")
