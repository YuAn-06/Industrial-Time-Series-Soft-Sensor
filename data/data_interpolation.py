import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
def interpolate_csv_column(file_path, column_name,method = 'linear'):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 将指定列中的0值替换为NaN
    df[column_name] = df[column_name].replace(0, np.nan)
    
    df[column_name] = df[column_name].replace(0, np.nan)
    df[column_name] = df[column_name].interpolate(method=method, limit_direction='both')
    
    return df

# 使用示例
if __name__ == "__main__":
    missing_rate = 50
    file_path = f'C:\Study\Code\datasets\PPGAS\PPGAS_masked_{missing_rate}.csv'  # 替换为实际的CSV文件路径
    mask_path = f'C:\Study\Code\datasets\PPGAS\PPGAS2011_{missing_rate}_mask.csv'
    column_name = 'NOX'  # 替换为实际的列名
    interpolated_df = interpolate_csv_column(file_path, column_name)
    
    # 将插值后的结果保存到新的CSV文件
    interpolated_df.to_csv(f'C:\Study\Code\datasets\PPGAS\PPGAS_interpolated_{missing_rate}.csv', index=False)



