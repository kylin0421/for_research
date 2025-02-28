import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(X, n_components=20):
    """先PCA降维，再应用t-SNE"""
    
    # 1. PCA降维
    print("应用PCA降维...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA保留信息量: {sum(pca.explained_variance_ratio_):.2f}")
    
    # 2. 在降维后数据上应用t-SNE
    return X_pca


# 基本t-SNE实现
def visualize_tsne(X, labels=None, perplexity=25, learning_rate=1000):
    # 记录开始时间
    t0 = time()
    
    # 创建并拟合t-SNE模型
    tsne = TSNE(
        n_components=2,           # 降到2维用于可视化
        perplexity=perplexity,    # 邻居数量参数
        learning_rate=learning_rate,
        n_iter=3000,              # 迭代次数
        random_state=42,           # 固定随机种子以便复现
        early_exaggeration=20,
    )
    X_tsne = tsne.fit_transform(X)
    
    # 记录用时
    t1 = time()
    print(f"t-SNE完成，耗时{t1-t0:.2f}秒")
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=labels,              # 按标签着色
            cmap='viridis',
            alpha=0.8,
            s=50                   # 点大小
        )
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.8, s=50)
    
    plt.title(f't-SNE Visualization (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return X_tsne

# 比较不同perplexity值的效果
def compare_perplexities(X, labels=None, perplexities=[5, 30, 50, 100]):
    plt.figure(figsize=(16, 4))
    
    for i, perp in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.subplot(1, len(perplexities), i+1)
        if labels is not None:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
        
        plt.title(f'Perplexity: {perp}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()

# 3D t-SNE可视化
def tsne_3d(X, labels=None):
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        scatter = ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
            c=labels, cmap='viridis', alpha=0.8, s=40
        )
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], alpha=0.8, s=40)
    
    ax.set_title('3D t-SNE Visualization')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.tight_layout()
    plt.show()
    
    return X_tsne



def main():
    with open("DSN+GCG.json", "r", encoding="utf-8") as file:
        difference_tensor=torch.tensor(json.load(file), dtype=torch.float32)

    print(difference_tensor.shape)

    num_samples=difference_tensor.shape[0]
    num_features=difference_tensor.shape[-1]

    X=difference_tensor.cpu().numpy().reshape(num_samples,num_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_PCA=pca(X_scaled,n_components=20)

    labels = np.array([0]*61+[1]*49)#([0] * 61 + [1] * 61+ [2] * 61+ [3] * 49+ [4] * 49+ [5] * 49)

    # 基本可视化
    X_tsne = visualize_tsne(X_PCA, labels)

    # 比较不同perplexity值
    compare_perplexities(X_PCA, labels, perplexities=[5, 30, 50])

    # 3D可视化
    X_tsne_3d = tsne_3d(X_PCA, labels)


if __name__=="__main__":
    main()