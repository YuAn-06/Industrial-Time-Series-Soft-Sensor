from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import os
import numpy as np
import pandas as pd

def collect_metrics(root_path):
    """
    遍历 root_path 下所有子文件夹，寻找 metrics.npy 并汇总
    """
    results = []
    # print(root_path)
    # 遍历文件夹
    for root, dirs, files in os.walk(root_path):
        # print(files)
        if 'metrics.npy' in files:
            file_path = os.path.join(root, 'metrics.npy')
            
            try:
                # 加载 npy 文件
                # 假设 metrics.npy 存储格式为 [mae,mse,rmse,mape,mspe,corr
                metrics_data = np.load(file_path)
                
                # 获取文件夹名称作为实验标识（通常包含模型名、参数等）
                exp_name = os.path.basename(root)
                
                # 构建结果字典
                # 注意：请根据你代码中保存的实际顺序调整这里的 key
                res_dict = {
                    'Experiment': exp_name,
                    'MAE':  metrics_data[0],
                    'MSE':  metrics_data[1],
                    'RMSE': metrics_data[2],
                    'MAPE': metrics_data[3],
                    'MSPE': metrics_data[4],
                    'Corr': metrics_data[5],
                    'Path': root # 保留路径方便追溯
                }
                results.append(res_dict)
                print(f"Successfully loaded: {exp_name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # 转换为 DataFrame
    if results:
        df = pd.DataFrame(results)
        # 按 MSE 排序，方便查看最优结果
        df = df.sort_values(by='MSE').reset_index(drop=True)
        return df
    else:
        print("No metrics.npy files found!")
        return None

# --- 使用示例 ---
# 替换为你的结果根目录，比如 './results'
model_name = 'SparseTSF'
target_folder = fr'C:\Study\Code\InduTS_SS\Exp_branch\Industrial-Time-Series-Soft-Sensor\results\{model_name}' 
df_results = collect_metrics(target_folder)

if df_results is not None:
    # 打印前 10 名
    print("\nTop 10 Results (Sorted by MSE):")
    print(df_results.head(10).to_string())
    
    # 导出到 Excel 或 CSV 供论文使用
    df_results.to_csv(fr'{target_folder}\all_experiments_metrics.csv', index=False)
    print("\nResults saved to 'all_experiments_metrics.csv'")