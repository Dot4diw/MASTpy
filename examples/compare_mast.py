import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from mastpy import SingleCellAssay, zlm

# 加载R生成的测试数据
def load_test_data():
    print("Loading test data from R...")
    # 使用rpy2加载RDS文件
    readRDS = robjects.r['readRDS']
    data = readRDS('test_data.rds')
    
    # 提取数据
    expression_matrix = np.array(data.rx2('expression_matrix'))
    
    # 使用localconverter进行转换
    with localconverter(pandas2ri.converter) as cv:
        cdata = robjects.conversion.rpy2py(data.rx2('cdata'))
        fdata = robjects.conversion.rpy2py(data.rx2('fdata'))
    
    return expression_matrix, cdata, fdata

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
    zfit = zlm('~ condition', sca, method='glm', use_ebayes=True, parallel=True)
    
    # 提取结果
    coefC = zfit.coef('C')
    coefD = zfit.coef('D')
    
    return coefC, coefD

# 比较结果
def compare_results(r_coefC, r_coefD, py_coefC, py_coefD):
    print("Comparing results...")
    
    # 确保基因顺序一致
    common_genes = sorted(list(set(r_coefC.index) & set(py_coefC.index)))
    r_coefC = r_coefC.loc[common_genes]
    r_coefD = r_coefD.loc[common_genes]
    py_coefC = py_coefC.loc[common_genes]
    py_coefD = py_coefD.loc[common_genes]
    
    # 确保列顺序一致
    common_cols = sorted(list(set(r_coefC.columns) & set(py_coefC.columns)))
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
    
    # 检查是否一致（差异小于1e-6）
    is_consistent = (c_diff.abs().max().max() < 1e-6) and (d_diff.abs().max().max() < 1e-6)
    print(f"Results are {'consistent' if is_consistent else 'inconsistent'}")
    
    # 保存比较结果
    comparison = pd.DataFrame({
        'R_coefC_intercept': r_coefC.iloc[:, 0],
        'Python_coefC_intercept': py_coefC.iloc[:, 0],
        'R_coefC_conditionB': r_coefC.iloc[:, 1],
        'Python_coefC_conditionB': py_coefC.iloc[:, 1],
        'R_coefD_intercept': r_coefD.iloc[:, 0],
        'Python_coefD_intercept': py_coefD.iloc[:, 0],
        'R_coefD_conditionB': r_coefD.iloc[:, 1],
        'Python_coefD_conditionB': py_coefD.iloc[:, 1]
    })
    comparison.to_csv('comparison_results.csv')
    
    print("Comparison results saved to comparison_results.csv")
    
    return is_consistent

if __name__ == '__main__':
    # 加载测试数据
    expression_matrix, cdata, fdata = load_test_data()
    
    # 加载R MAST结果
    r_coefC, r_coefD = load_r_mast_results()
    
    # 使用mastpy分析
    py_coefC, py_coefD = analyze_with_mastpy(expression_matrix, cdata, fdata)
    
    # 比较结果
    compare_results(r_coefC, r_coefD, py_coefC, py_coefD)
