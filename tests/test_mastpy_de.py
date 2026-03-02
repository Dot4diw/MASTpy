import sys
import os

# 设置环境变量以确保中文输出正常
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'zh_CN.UTF-8'

# Add the MASTpy directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import anndata as ad
from mastpy.core.single_cell_assay import SingleCellAssay
from mastpy.core.zlm import zlm
from mastpy.core.hypothesis import CoefficientHypothesis


def run_mastpy_de(adata, groupby, ident_1, ident_2, layer=None, logfc_threshold=0, min_pct=0.05, test_method='lr', n_jobs=1, output_file='MASTpy_DE_Results_with_CDR.csv'):
    """
    使用MASTpy进行差异表达分析
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData对象
    groupby : str
        分组变量名
    ident_1 : str
        比较的第一个组
    ident_2 : str
        比较的第二个组
    layer : str, optional
        使用的表达矩阵层，默认使用X
    logfc_threshold : float, optional
        log2 fold change阈值
    min_pct : float, optional
        基因表达的最小细胞比例
    test_method : str, optional
        检验方法，'lr' (似然比检验) 或 'wald' (Wald检验)
    n_jobs : int, optional
        并行计算的核心数
    output_file : str, optional
        结果输出文件
    
    Returns
    -------
    pandas.DataFrame
        差异表达分析结果
    """
    print("=== MASTpy差异表达分析 ===")
    
    # 1. 准备数据
    print("1. 准备数据...")
    
    # 获取表达矩阵
    if layer is not None and layer in adata.layers:
        expr_matrix = adata.layers[layer].T  # 转换为 (基因, 细胞) 格式
    else:
        expr_matrix = adata.X.T  # 转换为 (基因, 细胞) 格式
    
    # 确保是numpy数组
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()
    
    # 2. 创建SingleCellAssay对象
    print("2. 创建SingleCellAssay对象...")
    
    # 准备细胞元数据
    cdata = adata.obs.copy()
    
    # 准备基因元数据
    fdata = adata.var.copy()
    fdata.index = adata.var_names
    
    # 创建SingleCellAssay
    sca = SingleCellAssay(expr_matrix, cdata=cdata, fdata=fdata)
    
    # 3. 计算并标准化CDR (Cell Detection Rate)
    print("3. 计算并标准化CDR...")
    
    # 计算每个细胞的检测基因数
    cngeneson = (expr_matrix > 0).sum(axis=0)
    
    # 标准化
    cngeneson_scaled = (cngeneson - np.mean(cngeneson)) / np.std(cngeneson)
    
    # 添加到细胞元数据
    sca.cdata['cngeneson'] = cngeneson_scaled
    
    # 4. 基因过滤
    print("4. 基因过滤...")
    
    # 计算每个基因的表达频率
    gene_expr_freq = (expr_matrix > 0).sum(axis=1) / expr_matrix.shape[1]
    
    # 过滤基因
    filtered_genes = gene_expr_freq > min_pct
    print(f"过滤前基因数: {expr_matrix.shape[0]}")
    print(f"过滤后基因数: {filtered_genes.sum()}")
    
    # 应用过滤
    filtered_expr = expr_matrix[filtered_genes, :]
    filtered_fdata = fdata[filtered_genes]
    
    # 创建过滤后的SingleCellAssay
    sca_filtered = SingleCellAssay(filtered_expr, cdata=cdata, fdata=filtered_fdata)
    sca_filtered.cdata['cngeneson'] = cngeneson_scaled
    
    # 5. 准备分组变量
    print("5. 准备分组变量...")
    
    # 确保分组变量存在
    if groupby not in sca_filtered.cdata.columns:
        raise ValueError(f"分组变量 '{groupby}' 不存在于细胞元数据中")
    
    # 创建group列，确保是因子类型
    sca_filtered.cdata['group'] = sca_filtered.cdata[groupby].astype('category')
    
    # 6. 定义模型公式
    print("6. 定义模型公式...")
    formula = '~ group + cngeneson'
    
    # 7. 拟合Hurdle模型
    print("7. 拟合Hurdle模型...")
    zlm_fit = zlm(formula, sca_filtered, method='bayesglm', n_jobs=n_jobs)
    
    # 8. 执行差异表达分析
    print("8. 执行差异表达分析...")
    
    # 打印测试方法，用于调试
    print(f"测试方法: {test_method}")
    
    # 确定要测试的系数名
    # 系数名格式: group[T.ident_1]
    coef_name = f"group[T.{ident_1}]"
    
    if test_method == 'lr':
        # 似然比检验
        print(f"执行似然比检验，比较 {ident_1} vs {ident_2}...")
        summary_result = zlm_fit.summary(logFC=True, doLRT=[coef_name])
        
        # 9. 提取并整理结果
        print("9. 提取并整理结果...")
        
        dt = summary_result['datatable']
        
        # 打印表格的前几行，查看结构
        print("\nSummary table preview:")
        print(dt.head())
        
        # 打印唯一的contrast值
        print("\nUnique contrasts:")
        print(dt['contrast'].unique())
        
        # 打印唯一的component值
        print("\nUnique components:")
        print(dt['component'].unique())
        
        # 提取Hurdle部分的P值
        pvals = dt[(dt['contrast'] == coef_name) & (dt['component'] == 'H')][['primerid', 'Pr(>Chisq)']]
        print(f"\nP-values shape: {pvals.shape}")
        
        # 提取LogFC - 只选择与coef_name相关的结果
        # 由于logFC可能没有正确的contrast，我们只选择连续部分的系数作为logFC
        # 尝试不同的contrast格式
        logfc = None
        for contrast in [coef_name, coef_name.replace('[T.', 'T').replace(']', '')]:
            # 确保选择非NaN的系数值
            logfc = dt[(dt['contrast'] == contrast) & (dt['component'] == 'C') & (~dt['coef'].isna())][['primerid', 'coef', 'ci.hi', 'ci.lo']]
            if not logfc.empty:
                break
        
        # 如果仍然找不到，使用所有连续部分的系数
        if logfc is None or logfc.empty:
            # 找到包含group的contrast
            group_contrasts = [c for c in dt['contrast'].unique() if 'group' in c]
            if group_contrasts:
                for contrast in group_contrasts:
                    logfc = dt[(dt['contrast'] == contrast) & (dt['component'] == 'C') & (~dt['coef'].isna())][['primerid', 'coef', 'ci.hi', 'ci.lo']]
                    if not logfc.empty:
                        break
        
        print(f"LogFC shape: {logfc.shape}")
        
        # 合并结果
        de_results = pd.merge(pvals, logfc, on='primerid')
        print(f"Merged results shape: {de_results.shape}")
        de_results.columns = ['Gene', 'P_val', 'Log2FC', 'CI_high', 'CI_low']
    else:
        # Wald检验
        print(f"执行Wald检验，比较 {ident_1} vs {ident_2}...")
        # 创建系数假设
        hypothesis = CoefficientHypothesis(coef_name)
        # 执行Wald检验
        wald_result = zlm_fit.waldTest(hypothesis)
        # 获取summary结果
        summary_result = zlm_fit.summary(logFC=True)
        
        # 9. 提取并整理结果
        print("9. 提取并整理结果...")
        
        dt = summary_result['datatable']
        
        # 打印表格的前几行，查看结构
        print("\nSummary table preview:")
        print(dt.head())
        
        # 打印唯一的contrast值
        print("\nUnique contrasts:")
        print(dt['contrast'].unique())
        
        # 打印唯一的component值
        print("\nUnique components:")
        print(dt['component'].unique())
        
        # 从Wald检验结果中提取P值
        # wald_result shape: (n_genes, 3, 3) where 3rd dimension is [lambda, df, p-value]
        # 我们需要Hurdle部分的p值 (index 2)
        p_values = wald_result[:, 2, 2]  # Hurdle p-values
        
        # 创建pvals DataFrame
        pvals = pd.DataFrame({
            'primerid': zlm_fit.coefC.index,
            'Pr(>Chisq)': p_values
        })
        print(f"\nP-values shape: {pvals.shape}")
        
        # 提取LogFC - 只选择与coef_name相关的结果
        logfc = dt[(dt['contrast'] == coef_name) & (dt['component'] == 'C')][['primerid', 'coef', 'ci.hi', 'ci.lo']]
        print(f"LogFC shape: {logfc.shape}")
        
        # 合并结果
        de_results = pd.merge(pvals, logfc, on='primerid')
        print(f"Merged results shape: {de_results.shape}")
        de_results.columns = ['Gene', 'P_val', 'Log2FC', 'CI_high', 'CI_low']
    
    # 10. 计算校正后的P值 (FDR)
    print("10. 计算FDR...")
    from statsmodels.stats.multitest import multipletests
    de_results['FDR'] = multipletests(de_results['P_val'], method='fdr_bh')[1]
    
    # 11. 排序并保存
    print("11. 排序并保存结果...")
    de_results = de_results.sort_values('FDR')
    de_results.to_csv(output_file, index=False)
    
    print(f"\n差异表达分析完成！")
    print(f"结果已保存到: {output_file}")
    print(f"显著差异表达基因数 (FDR < 0.05): {(de_results['FDR'] < 0.05).sum()}")
    
    return de_results


if __name__ == '__main__':
    # 示例用法
    print("=== MASTpy差异表达分析示例 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    n_genes = 1000
    n_cells = 100
    
    # 创建表达矩阵
    expr = np.zeros((n_genes, n_cells))
    
    # 创建分组
    groups = np.array(['A'] * 50 + ['B'] * 50)
    
    # 生成差异表达数据
    for i in range(n_genes):
        if i < 100:  # 前100个基因差异表达
            expr[i, :50] = np.random.poisson(1, 50)
            expr[i, 50:] = np.random.poisson(5, 50)
        else:
            expr[i, :] = np.random.poisson(2, n_cells)
    
    # 创建AnnData对象
    adata = ad.AnnData(
        X=expr.T,  # (细胞, 基因)
        obs=pd.DataFrame({'group': groups}),
        var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    )
    
    # 运行差异表达分析
    results = run_mastpy_de(
        adata=adata,
        groupby='group',
        ident_1='B',
        ident_2='A',
        layer=None,
        logfc_threshold=0,
        min_pct=0.05,
        test_method='lr',
        n_jobs=1,
        output_file='MASTpy_DE_Results_with_CDR.csv'
    )
    
    # 显示前10个结果
    print("\nTop 10差异表达基因:")
    print(results.head(10))
