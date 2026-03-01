# Load required libraries
library(Seurat)
library(MAST)
library(dplyr)

# Set working directory
setwd("D:/MyCode/Python/MAST-devel/MASTpy")

# Read test data
print("Reading test data...")
expression_matrix <- read.csv("test_expression_matrix.csv", row.names = 1)
cell_metadata <- read.csv("test_cell_metadata.csv", row.names = 1)

# Create Seurat object
print("Creating Seurat object...")
seurat_obj <- CreateSeuratObject(
  counts = expression_matrix,
  meta.data = cell_metadata
)

# Set active identity to condition
print("Setting active identity...")
Idents(seurat_obj) <- seurat_obj$condition

# Normalize data (using log1p as MAST expects log-transformed data)
print("Normalizing data...")
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)

# Find markers using MAST
print("Finding markers using MAST...")
mast_markers <- FindMarkers(
  seurat_obj,
  ident.1 = "A",
  ident.2 = "B",
  test.use = "MAST",
  logfc.threshold = 0,  # No filtering
  min.pct = 0,  # No filtering
  verbose = TRUE
)

# Sort by gene name
mast_markers <- mast_markers[order(rownames(mast_markers)), ]

# Save results
print("Saving MAST markers results...")
write.csv(mast_markers, "seurat_mast_results.csv")

# Show top results
print("Top differentially expressed genes:")
print(head(mast_markers, 10))

print("Analysis completed successfully!")
