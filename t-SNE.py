import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from time import time
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def visualize_tsne(X, labels=None, perplexity=30, learning_rate=200, remove_outliers=True, contamination=0.1):
    """使用t-SNE进行降维可视化，可选使用IsolationForest移除离群值"""
    # 记录开始时间
    t0 = time()
    
    # 如果需要，使用IsolationForest移除离群值
    if remove_outliers:
        print("使用IsolationForest检测并移除离群值...")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_preds = iso_forest.fit_predict(X)
        
        # IsolationForest返回1表示正常点，-1表示异常点
        normal_mask = outlier_preds == 1
        outlier_count = np.sum(outlier_preds == -1)
        outlier_percent = outlier_count / len(X) * 100
        print(f"检测到 {outlier_count} 个异常点 ({outlier_percent:.2f}%)")
        
        X_clean = X[normal_mask]
        
        # 如果提供了标签，也相应过滤
        if labels is not None:
            labels_clean = labels[normal_mask]
        else:
            labels_clean = None
    else:
        X_clean = X
        labels_clean = labels
        normal_mask = np.ones(len(X), dtype=bool)
    
    # 创建并拟合t-SNE模型
    tsne = TSNE(
        n_components=2,           # 降到2维用于可视化
        perplexity=min(perplexity, len(X_clean) - 1),  # 确保perplexity不大于样本数-1
        learning_rate=learning_rate,
        n_iter=3000,              # 迭代次数
        metric="euclidean",       # 明确指定度量方式
        init="pca",               # 用PCA初始化，提高稳定性
        random_state=42,          # 固定随机种子以便复现
        early_exaggeration=20,
    )
    X_tsne = tsne.fit_transform(X_clean)
    
    # 记录用时
    t1 = time()
    print(f"t-SNE完成，耗时{t1-t0:.2f}秒")
    
    # 创建可视化
    plt.figure(figsize=(10, 8))
    if labels_clean is not None:
        scatter = plt.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=labels_clean,        # 按标签着色
            cmap='viridis',
            alpha=0.8,
            s=50                   # 点大小
        )
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.8, s=50)
    
    plt.title(f't-SNE Visualization (perplexity={perplexity}, outliers removed={remove_outliers})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return X_tsne, normal_mask

def compare_perplexities(X, labels=None, perplexities=[5, 30, 50, 100], remove_outliers=True, contamination=0.1):
    """比较不同perplexity值的t-SNE可视化效果"""
    plt.figure(figsize=(16, 4))
    
    # 如果需要，使用IsolationForest移除离群值
    if remove_outliers:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_preds = iso_forest.fit_predict(X)
        normal_mask = outlier_preds == 1
        X_clean = X[normal_mask]
        
        # 如果提供了标签，也相应过滤
        if labels is not None:
            labels_clean = labels[normal_mask]
        else:
            labels_clean = None
    else:
        X_clean = X
        labels_clean = labels
    
    for i, perp in enumerate(perplexities):
        # 确保perplexity不超过样本数-1
        actual_perp = min(perp, len(X_clean) - 1)
        if actual_perp < 5:  # 如果样本太少，则跳过
            plt.subplot(1, len(perplexities), i+1)
            plt.text(0.5, 0.5, "样本太少，无法使用\nperplexity=" + str(perp),
                     horizontalalignment='center', verticalalignment='center')
            continue
            
        tsne = TSNE(
            n_components=2, 
            perplexity=actual_perp,
            random_state=42,
            init="pca"
        )
        X_tsne = tsne.fit_transform(X_clean)
        
        plt.subplot(1, len(perplexities), i+1)
        if labels_clean is not None:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_clean, cmap='viridis', alpha=0.7)
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
        
        plt.title(f'Perplexity: {perp}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.show()

def tsne_3d(X, labels=None, remove_outliers=True, contamination=0.1):
    """3D t-SNE可视化"""
    # 如果需要，使用IsolationForest移除离群值
    if remove_outliers:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_preds = iso_forest.fit_predict(X)
        normal_mask = outlier_preds == 1
        X_clean = X[normal_mask]
        
        # 如果提供了标签，也相应过滤
        if labels is not None:
            labels_clean = labels[normal_mask]
        else:
            labels_clean = None
    else:
        X_clean = X
        labels_clean = labels
    
    tsne = TSNE(
        n_components=3, 
        perplexity=min(30, len(X_clean) - 1), 
        random_state=42,
        init="pca"
    )
    X_tsne = tsne.fit_transform(X_clean)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels_clean is not None:
        scatter = ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
            c=labels_clean, cmap='viridis', alpha=0.8, s=40
        )
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], alpha=0.8, s=40)
    
    ax.set_title('3D t-SNE Visualization (Outliers Removed)' if remove_outliers else '3D t-SNE Visualization')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.tight_layout()
    plt.show()
    
    return X_tsne

def analyze_contamination_effect(X, labels, contamination_values=[0.05, 0.1, 0.15, 0.2]):
    """分析不同contamination值对结果的影响"""
    plt.figure(figsize=(16, 4))
    
    for i, cont in enumerate(contamination_values):
        iso_forest = IsolationForest(contamination=cont, random_state=42)
        outlier_preds = iso_forest.fit_predict(X)
        normal_mask = outlier_preds == 1
        
        # 统计每个类别中被标记为异常的点的比例
        labels_unique = np.unique(labels)
        outlier_ratios = []
        
        for label in labels_unique:
            class_mask = labels == label
            class_outliers = np.sum((~normal_mask) & class_mask)
            class_total = np.sum(class_mask)
            outlier_ratio = class_outliers / class_total * 100
            outlier_ratios.append(outlier_ratio)
            print(f"Contamination {cont}: 类别 {label} 中有 {outlier_ratio:.2f}% 的点被标记为异常")
        
        plt.subplot(1, len(contamination_values), i+1)
        plt.bar(labels_unique, outlier_ratios)
        plt.title(f'Contamination: {cont}')
        plt.xlabel('Class')
        plt.ylabel('Outlier ratio (%)')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    with open("sum_of_4.json", "r", encoding="utf-8") as file:
        difference_tensor = torch.tensor(json.load(file), dtype=torch.float32)

    print(difference_tensor.shape)

    num_samples = difference_tensor.shape[0]
    num_features = difference_tensor.shape[-1]

    X = difference_tensor.cpu().numpy().reshape(num_samples, num_features)

    # 使用RobustScaler，对离群值更不敏感
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用PCA降维，降低噪声影响
    pca = PCA(n_components=150, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA保留信息量: {sum(pca.explained_variance_ratio_):.2f}")

    labels = np.array([0]*61+[1]*49+[2]*91+[3]*38)   #DSN+GCG+JBC+PAIR

    # 未移除离群值的可视化
    print("未移除噪声点的t-SNE可视化:")
    X_tsne_original, _ = visualize_tsne(X_pca, labels, remove_outliers=False)

    # 使用IsolationForest移除离群值后的可视化
    contamination = 0.1  # 可调整的参数，表示预期的异常点比例
    print(f"\n使用IsolationForest移除离群值 (contamination={contamination}):")
    X_tsne_iso, normal_mask = visualize_tsne(X_pca, labels, remove_outliers=True, contamination=contamination)

    # 分析不同contamination值对结果的影响
    print("\n分析不同contamination值对各类别的影响:")
    analyze_contamination_effect(X_pca, labels, contamination_values=[0.05, 0.1, 0.15, 0.2])

    # 使用筛选后的数据比较不同perplexity值的效果
    print("\n比较不同perplexity值 (使用IsolationForest过滤后的数据):")
    compare_perplexities(X_pca, labels, perplexities=[5, 30, 50, 100], remove_outliers=True, contamination=contamination)

    # 3D可视化 (可选)
    print("\n3D t-SNE可视化:")
    X_tsne_3d = tsne_3d(X_pca, labels, remove_outliers=True, contamination=contamination)

if __name__ == "__main__":
    main()