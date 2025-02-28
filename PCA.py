import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd

with open("DSN+GCG.json", "r", encoding="utf-8") as file:
    difference_tensor=torch.tensor(json.load(file), dtype=torch.float32)

print(difference_tensor.shape)

num_samples=difference_tensor.shape[0]
num_features=difference_tensor.shape[-1]

numpy_data=difference_tensor.cpu().numpy().reshape(num_samples,num_features)

print(numpy_data.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(numpy_data)

# 4.1 常规PCA（作为基准）
pca_regular = PCA(n_components=10)
X_pca_regular = pca_regular.fit_transform(X_scaled)

# 4.2 随机化PCA（对高维数据更快更高效）
pca_random = PCA(n_components=10, svd_solver='randomized', random_state=42)
X_pca_random = pca_random.fit_transform(X_scaled)

print("常规PCA降维后形状:", X_pca_regular.shape)
print("随机化PCA降维后形状:", X_pca_random.shape)
print("常规PCA累计方差解释率:", np.sum(pca_regular.explained_variance_ratio_))
print("随机化PCA累计方差解释率:", np.sum(pca_random.explained_variance_ratio_))

# 5.1 方差解释率图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca_random.explained_variance_ratio_)
plt.title('Explained Variance Ratio by Principal Component')  # 各主成分解释方差比
plt.xlabel('Principal Component')  # 主成分
plt.ylabel('Explained Variance Ratio')  # 解释方差比

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca_random.explained_variance_ratio_))
plt.title('Cumulative Explained Variance Ratio')  # 累计解释方差比
plt.xlabel('Number of Principal Components')  # 主成分数量
plt.ylabel('Cumulative Explained Variance Ratio')  # 累计解释方差比
plt.grid(True)
plt.tight_layout()
plt.show()

# 5.2 二维散点图
plt.figure(figsize=(10, 8))
plt.scatter(X_pca_random[:, 0], X_pca_random[:, 1], alpha=0.8)
plt.title('Randomized PCA - First Two Principal Components')  # 随机化PCA - 前两个主成分
plt.xlabel('Principal Component 1')  # 主成分1
plt.ylabel('Principal Component 2')  # 主成分2
plt.grid(True)
plt.show()

# 5.3 主成分热图
plt.figure(figsize=(12, 8))
sns.heatmap(X_pca_random[:, :5], cmap='viridis')
plt.title('Sample Representation in the First 5 Principal Components')  # 样本在前5个主成分空间中的表示
plt.xlabel('Principal Component')  # 主成分
plt.ylabel('Sample')  # 样本
plt.show()

# 5.4 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_pca_random[:, 0], 
    X_pca_random[:, 1], 
    X_pca_random[:, 2], 
    c=X_pca_random[:, 0],
    cmap='viridis', 
    alpha=0.8
)
ax.set_title('Randomized PCA - First Three Principal Components')  # 随机化PCA - 前三个主成分
ax.set_xlabel('Principal Component 1')  # 主成分1
ax.set_ylabel('Principal Component 2')  # 主成分2
ax.set_zlabel('Principal Component 3')  # 主成分3
plt.show()