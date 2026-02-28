# 加载MAST包
library(MAST)

# 设置随机种子，确保结果可重复
set.seed(42)

# 生成测试数据
n_genes <- 100
n_cells <- 50

# 生成表达矩阵
expression_matrix <- matrix(rpois(n_genes * n_cells, lambda = 0.5), nrow = n_genes, ncol = n_cells)
rownames(expression_matrix) <- paste0("gene_", 1:n_genes)
colnames(expression_matrix) <- paste0("cell_", 1:n_cells)

# 生成细胞元数据
condition <- sample(c("A", "B"), n_cells, replace = TRUE)
ncells <- rep(1, n_cells)
cdata <- data.frame(condition = condition, ncells = ncells, row.names = colnames(expression_matrix))

# 生成特征元数据
fdata <- data.frame(row.names = rownames(expression_matrix))

# 创建SingleCellAssay对象
sca <- FromMatrix(exprsArray = expression_matrix, cData = cdata, fData = fdata, check_sanity = FALSE)

# 拟合ZLM模型
print("Fitting ZLM model in R MAST...")
zlm_result <- zlm(~ condition, sca, method = "glm")

# 提取结果
coefC <- coef(zlm_result, "C")
coefD <- coef(zlm_result, "D")

# 保存结果到文件
write.csv(coefC, "r_mast_coefC.csv")
write.csv(coefD, "r_mast_coefD.csv")

# 打印结果摘要
print("R MAST results:")
print(head(coefC))
print(head(coefD))

# 保存表达矩阵和元数据，供Python使用
saveRDS(list(expression_matrix = expression_matrix, cdata = cdata, fdata = fdata), "test_data.rds")

print("Test data and R MAST results saved successfully!")
