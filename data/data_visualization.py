import statsmodels.api as sm
import math
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.manifold import TSNE


import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
rcParams['font.family'] = 'Times New Roman'

def plot_time_series(data, columns):
    N, D = data.shape
    fig, axes = plt.subplots(math.ceil(D/3), 3, figsize=(24, 18))  # D行D//3列

    axes = axes.flatten()  
    for i in range(data.shape[1]):
        axes[i].plot(data[:,i], label=columns[i])
        axes[i].legend()

    idx = np.arange(math.ceil(D/3) *3 )  

    for i in idx[D:]:  
        axes[i].set_visible(False)
    plt.tight_layout()


def plot_acf_lag(data, columns):
    N, D = data.shape
    fig, axes = plt.subplots(D//3 +1, 3, figsize=(18, 12))
    axes = axes.flatten()  # 将二维数组展平成一维，便于索引
    for i in range(D):
        # 绘制ACF，lags=60表示显示前60个滞后
        plot_acf(data[:,i], ax=axes[i], lags=60, alpha=0.05, label=columns[i])
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel('Lag', fontsize=4)
        axes[i].set_ylabel('Autocorrelation', fontsize=4)
    
    plt.tight_layout()

    idx = np.arange(math.ceil(D/3) *3 )  
    for i in idx[D:]:  
        axes[i].set_visible(False)
    plt.tight_layout()



def plot_spearmanr(data, columns, corr):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', xticklabels=columns, yticklabels=columns)
 

def plot_data_2d(train_data, test_data):
    tsne = TSNE(n_components=2, random_state=42)
    train_data_tsne = tsne.fit_transform(train_data)
    test_data_tsne = tsne.fit_transform(test_data)
    df = pd.DataFrame(
        {
            'x': np.concatenate((train_data_tsne[:, 0], test_data_tsne[:, 0])),
            'y': np.concatenate((train_data_tsne[:, 1], test_data_tsne[:, 1])),
            'label': ['training']*len(train_data_tsne) + ['testing']*len(test_data_tsne)
        }
    )

    g= sns.jointplot(x='x', y='y', data=df, kind='scatter', hue='label', palette={'training': '#3E4F94', 'testing': '#B02425'}, 
    joint_kws={'alpha': 0.5, 's': 80, 'edgecolors': 'black', 'linewidths': 0.3})
    g.ax_joint.tick_params(labelsize=12)
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.grid(True)
    plt.savefig('pics/'+data_name+'_tsne.pdf', dpi=300)
    

data_name = 'DC'

if __name__ == '__main__':


    if data_name == 'DC':
        data_df = pd.read_csv('data/DC/debutanizer_column.csv')
    elif data_name == 'SRU':
        data_df = pd.read_csv('data/SRU/SRU_data.csv')
    elif data_name == 'Ironmaking':
        data_df = pd.read_csv('data/Ironmaking/Ironmaking.csv')
        data_df = data_df.drop(columns=['date'])
    elif data_name == 'PPGAS':
        data_df = pd.read_csv('data/PPGAS/gt_2012.csv')
        data_df = data_df.drop(columns=['date'])
    data = data_df.values
    columns = data_df.columns

    # Time Series Visualization
    # plot_time_series(data, columns)
    # # # ACF Lag Plot
    # plot_acf_lag(data, columns)

    # Spearmanr Plot
    plot_spearmanr(data, columns, corr=data_df.corr(method='spearman'))

    # 2D Visualization
    # data_train = data[:int(0.7*len(data))]
    # data_test = data[int(0.8*len(data)):]
    # plot_data_2d(data_train, data_test)
    plt.legend()
    plt.show()