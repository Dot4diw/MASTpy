"""Compare MASTpy and Seurat MAST results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Read results
mastpy_df = pd.read_csv("mastpy_results.csv", index_col="gene")
seurat_df = pd.read_csv("seurat_mast_results.csv", index_col=0)

# Rename Seurat index to match MASTpy (replace dashes with underscores)
seurat_df.index = seurat_df.index.str.replace("-", "_")

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
plt.savefig('comparison_plots.png')
print("\nPlots saved as 'comparison_plots.png'")

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

combined_df.to_csv('combined_results.csv')
print("\nCombined results saved as 'combined_results.csv'")

print("\nAnalysis completed successfully!")
