# Load required libraries
library(Seurat)
library(MAST)
library(dplyr)

# Set working directory
setwd("D:/MyCode/Python/MAST-devel/MASTpy")

# Read data
print("Reading data...")
expression_matrix <- read.csv("cs_expression_matrix.csv", row.names = 1)
log1p_norm_matrix <- read.csv("cs_log1p_norm_matrix.csv", row.names = 1)
cell_metadata <- read.csv("cs_cell_metadata.csv", row.names = 1)

# Create Seurat object
print("Creating Seurat object...")
seurat_obj <- CreateSeuratObject(
  counts = expression_matrix,
  meta.data = cell_metadata
)

# Add log1p_norm to data layer
print("Adding log1p_norm to data layer...")
# Ensure cell names match
colnames(log1p_norm_matrix) <- colnames(expression_matrix)
# Ensure feature names match (Seurat replaces underscores with dashes)
rownames(log1p_norm_matrix) <- gsub("_", "-", rownames(log1p_norm_matrix))
# Use the default assay and set the data layer
seurat_obj <- SetAssayData(seurat_obj, layer = "data", new.data = log1p_norm_matrix)

# Set active identity to group
print("Setting active identity...")
# Check available columns
print("Available columns in meta.data:")
print(colnames(seurat_obj@meta.data))
# Set identity using group column
seurat_obj$group <- cell_metadata$group
Idents(seurat_obj) <- seurat_obj$group

# Find markers using MAST
print("Finding markers using MAST...")
mast_markers <- FindMarkers(
  seurat_obj,
  ident.1 = "CS",
  ident.2 = "WT",
  test.use = "MAST",
  logfc.threshold = 0,  # No filtering
  min.pct = 0,  # No filtering
  verbose = TRUE
)

# Sort by gene name
mast_markers <- mast_markers[order(rownames(mast_markers)), ]

# Save results
print("Saving MAST markers results...")
write.csv(mast_markers, "seurat_cs_results.csv")

# Show top results
print("Top differentially expressed genes:")
print(head(mast_markers, 10))

print("Analysis completed successfully!")
