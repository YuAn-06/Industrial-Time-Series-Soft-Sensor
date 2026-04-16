from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data\DC\debutanizer_column.csv')

X = ['NOX', 'CO']
X = X.pop('NOX')
print(X)

# X = data.values

# X_train = X[:2000,]

# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=42)

# # 训练模型
# kmeans.fit(X_train)
# kmeans.fit_predict(X)
# labels = kmeans.labels_

# np.savetxt('data/DC/mode_labels.txt', labels)
