
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler





class ZeroMaskStandardScaler:
    def __init__(self):
        self.mean_ = None  # 各特征非零值均值
        self.std_ = None   # 各特征非零值标准差（样本标准差）

    def fit(self, X):
        """
        计算各特征非零值的均值和标准差
        :param X: 二维数组 (n_samples, n_features)
        """
        if len(X.shape) != 2:
            raise ValueError("输入数据需为二维数组（样本数×特征数）")
        
        n_features = X.shape[1]
        self.mean_ = np.zeros(n_features)
        self.std_ = np.zeros(n_features)
        
        for i in range(n_features):
            # 获取非零值掩码
            non_zero_mask = X[:, i] != 0
            non_zero_vals = X[:, i][non_zero_mask]
            
            if non_zero_vals.size == 0:
                # 全0列特殊处理（避免除以0）
                self.mean_[i] = 0.0
                self.std_[i] = 1.0
            else:
                self.mean_[i] = np.mean(non_zero_vals)
                self.std_[i] = np.std(non_zero_vals, ddof=1)  # 使用样本标准差（n-1）
                
                # 防止标准差为0导致除以0错误
                if self.std_[i] < 1e-8:
                    self.std_[i] = 1.0
        return self

    def transform(self, X):
        """
        对非零值进行标准化，零值保持不变
        :param X: 二维数组 (n_samples, n_features)
        :return: 标准化后数组
        """
        X_transformed = np.copy(X)
        for i in range(X.shape[1]):
            non_zero_mask = X[:, i] != 0
            X_transformed[:, i][non_zero_mask] = (X[:, i][non_zero_mask] - self.mean_[i]) / self.std_[i]
        return X_transformed

    def inverse_transform(self, X):
        """
        逆标准化（仅对非零变换值生效）
        :param X: 标准化后数组
        :return: 原始尺度数组
        """
        X_inv = np.copy(X)
        for i in range(X.shape[1]):
            non_zero_mask = X[:, i] != 0  # 假设标准化后零值仍保持为0
            X_inv[:, i][non_zero_mask] = X[:, i][non_zero_mask] * self.std_[i] + self.mean_[i]
        return X_inv


class StandardScalerWrapper:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        使用 sklearn 的 StandardScaler 拟合数据
        :param X: 二维数组 (n_samples, n_features)
        """
        if len(X.shape) != 2:
            raise ValueError("输入数据需为二维数组（样本数×特征数）")
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """
        使用 sklearn 的 StandardScaler 对数据进行标准化
        :param X: 二维数组 (n_samples, n_features)
        :return: 标准化后数组
        """
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        """
        使用 sklearn 的 StandardScaler 对数据进行逆标准化
        :param X: 标准化后数组
        :return: 原始尺度数组
        """
        return self.scaler.inverse_transform(X)