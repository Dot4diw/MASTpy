import numpy as np
import pandas as pd
from mastpy import SingleCellAssay, zlm

# 加载R生成的测试数据
print("Loading test data from R...")

# 直接读取R生成的测试数据文件
expression_matrix = pd.read_csv('expression_matrix.csv', index_col=0).values
cdata = pd.read_csv('cdata.csv', index_col=0)
fdata = pd.read_csv('fdata.csv', index_col=0)

# 获取基因和细胞名称
gene_names = fdata.index.tolist()
cell_names = cdata.index.tolist()
n_genes = expression_matrix.shape[0]
n_cells = expression_matrix.shape[1]

# 加载R MAST的结果
def load_r_mast_results():
    print("Loading R MAST results...")
    coefC = pd.read_csv('r_mast_coefC.csv', index_col=0)
    coefD = pd.read_csv('r_mast_coefD.csv', index_col=0)
    return coefC, coefD

# 使用mastpy分析数据
def analyze_with_mastpy(expression_matrix, cdata, fdata):
    print("Analyzing data with mastpy...")
    # 创建SingleCellAssay对象
    sca = SingleCellAssay(expression_matrix, cdata, fdata)
    
    # 拟合ZLM模型
    zfit = zlm('~ condition', sca, method='glm', use_ebayes=True, parallel=True, silent=False)
    
    # 提取结果
    coefC = zfit.coef('C')
    coefD = zfit.coef('D')
    
    return coefC, coefD

# 比较结果
def compare_results(r_coefC, r_coefD, py_coefC, py_coefD):
    print("Comparing results...")
    
    # 打印结果的基本信息
    print(f"R MAST coefC shape: {r_coefC.shape}")
    print(f"R MAST coefC columns: {list(r_coefC.columns)}")
    print(f"Python mastpy coefC shape: {py_coefC.shape}")
    print(f"Python mastpy coefC columns: {list(py_coefC.columns)}")
    
    print(f"R MAST coefD shape: {r_coefD.shape}")
    print(f"R MAST coefD columns: {list(r_coefD.columns)}")
    print(f"Python mastpy coefD shape: {py_coefD.shape}")
    print(f"Python mastpy coefD columns: {list(py_coefD.columns)}")
    
    # 确保基因顺序一致
    common_genes = sorted(list(set(r_coefC.index) & set(py_coefC.index)))
    print(f"Number of common genes: {len(common_genes)}")
    
    if len(common_genes) > 0:
        r_coefC = r_coefC.loc[common_genes]
        r_coefD = r_coefD.loc[common_genes]
        py_coefC = py_coefC.loc[common_genes]
        py_coefD = py_coefD.loc[common_genes]
        
        # 确保列顺序一致
        common_cols = sorted(list(set(r_coefC.columns) & set(py_coefC.columns)))
        print(f"Number of common columns: {len(common_cols)}")
        
        if len(common_cols) > 0:
            r_coefC = r_coefC[common_cols]
            r_coefD = r_coefD[common_cols]
            py_coefC = py_coefC[common_cols]
            py_coefD = py_coefD[common_cols]
            
            # 计算差异
            c_diff = r_coefC - py_coefC
            d_diff = r_coefD - py_coefD
            
            # 计算统计信息
            c_mean_diff = c_diff.mean().mean()
            c_max_diff = c_diff.abs().max().max()
            d_mean_diff = d_diff.mean().mean()
            d_max_diff = d_diff.abs().max().max()
            
            print(f"Continuous component - Mean difference: {c_mean_diff:.6f}, Max difference: {c_max_diff:.6f}")
            print(f"Discrete component - Mean difference: {d_mean_diff:.6f}, Max difference: {d_max_diff:.6f}")
            
            # 打印前5个差异最大的基因
            print("\nTop 5 genes with largest continuous component differences:")
            c_diff_sum = c_diff.abs().sum(axis=1)
            top_c_diff = c_diff_sum.sort_values(ascending=False).head(5)
            print(top_c_diff)
            
            print("\nTop 5 genes with largest discrete component differences:")
            d_diff_sum = d_diff.abs().sum(axis=1)
            top_d_diff = d_diff_sum.sort_values(ascending=False).head(5)
            print(top_d_diff)
            
            # 打印具体的差异值
            if len(top_c_diff) > 0:
                print("\nDetailed differences for top continuous component gene:")
                gene = top_c_diff.index[0]
                print(f"Gene: {gene}")
                print(f"R coefC: {r_coefC.loc[gene].values}")
                print(f"Python coefC: {py_coefC.loc[gene].values}")
                print(f"Difference: {c_diff.loc[gene].values}")
            
            # 检查是否一致（差异小于1e-6）
            is_consistent = (c_diff.abs().max().max() < 1e-6) and (d_diff.abs().max().max() < 1e-6)
            print(f"\nResults are {'consistent' if is_consistent else 'inconsistent'}")
            
            # 保存比较结果
            if len(common_cols) >= 2:
                comparison = pd.DataFrame({
                    'gene': r_coefC.index,
                    'R_coefC_intercept': r_coefC.iloc[:, 0],
                    'Python_coefC_intercept': py_coefC.iloc[:, 0],
                    'R_coefC_conditionB': r_coefC.iloc[:, 1],
                    'Python_coefC_conditionB': py_coefC.iloc[:, 1],
                    'R_coefD_intercept': r_coefD.iloc[:, 0],
                    'Python_coefD_intercept': py_coefD.iloc[:, 0],
                    'R_coefD_conditionB': r_coefD.iloc[:, 1],
                    'Python_coefD_conditionB': py_coefD.iloc[:, 1]
                })
                comparison.to_csv('comparison_results.csv', index=False)
                print("Comparison results saved to comparison_results.csv")
            else:
                print("Not enough columns to create comparison DataFrame")
            
            return is_consistent
        else:
            print("No common columns found")
            return False
    else:
        print("No common genes found")
        return False

if __name__ == '__main__':
    # 加载R MAST结果
    r_coefC, r_coefD = load_r_mast_results()
    
    # 使用mastpy分析
    py_coefC, py_coefD = analyze_with_mastpy(expression_matrix, cdata, fdata)
    
    # 比较结果
    compare_results(r_coefC, r_coefD, py_coefC, py_coefD)
