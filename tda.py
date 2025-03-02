import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, HeatKernel
from gtda.plotting import plot_diagram
from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 1. 生成示例高维数据
# 这里使用Swiss roll数据集作为示例，你可以替换为自己的高维数据
n_samples = 1000
noise = 0.05
X, color = make_swiss_roll(n_samples=n_samples, noise=noise)
print(f"生成的数据形状: {X.shape}")

# 2. 使用持续同调分析数据的拓扑特征
# 初始化Vietoris-Rips复形
persistence = VietorisRipsPersistence(
    homology_dimensions=[0, 1, 2],  # 计算0, 1, 2维同调群
    n_jobs=-1,                      # 使用所有可用的CPU核心
    metric="euclidean"
)

# 计算持续同调图
diagrams = persistence.fit_transform(X.reshape(1, *X.shape))

# 可视化持续同调图
plt.figure(figsize=(12, 4))
plt.subplot(121)
plot_diagram(diagrams[0])
plt.title("持续同调图")

# 3. 提取拓扑特征
# 使用持续熵作为拓扑特征
persistence_entropy = PersistenceEntropy()
entropy = persistence_entropy.fit_transform(diagrams)
print(f"持续熵: {entropy}")

# 使用热核特征
heat_kernel = HeatKernel()
heat_features = heat_kernel.fit_transform(diagrams)
print(f"热核特征形状: {heat_features.shape}")

# 4. 使用Mapper算法进行可视化和分析
# 定义过滤函数 (这里使用PCA的前两个分量)
filter_func = Projection(columns=[0, 1])

# 创建覆盖
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)

# 创建聚类算法
clusterer = DBSCAN(eps=0.5)

# 构建Mapper管道
mapper = make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False
)

# 应用Mapper
# 首先标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA减少维度（可选）
pca = PCA(n_components=min(10, X.shape[1]))
X_pca = pca.fit_transform(X_scaled)

# 运行Mapper
graph = mapper.fit_transform(X_pca)

# 5. 可视化Mapper图
plt.subplot(122)
fig = plot_static_mapper_graph(
    graph,
    color_by_node_values=color,
    node_size=80,
    title="Mapper图: 数据的拓扑结构"
)

plt.tight_layout()
plt.show()

# 6. 分析Mapper图的连通分量
print(f"Mapper图中节点数量: {graph.shape[0]}")
print(f"Mapper图中边的数量: {graph.shape[1]}")

# 7. 结合拓扑特征进行进一步分析
# 这里我们可以结合持续同调的特征和Mapper图的信息
# 例如，我们可以查看持续同调图中最显著的特征

# 定义一个函数来提取最持久的特征
def get_persistent_features(diagram, dim=1, top_n=3):
    """提取指定维度中最持久的特征"""
    # 过滤指定维度的特征
    dim_diag = diagram[diagram[:, 2] == dim]
    
    # 计算持久性（死亡时间 - 出生时间）
    persistence = dim_diag[:, 1] - dim_diag[:, 0]
    
    # 找到最持久的特征的索引
    top_idx = np.argsort(persistence)[-top_n:][::-1]
    
    return dim_diag[top_idx]

# 提取1维同调群中最持久的3个特征
top_features = get_persistent_features(diagrams[0], dim=1, top_n=3)
print("\n1维同调群中最持久的特征:")
for i, feature in enumerate(top_features):
    print(f"特征 {i+1}: 出生={feature[0]:.3f}, 死亡={feature[1]:.3f}, 持久性={feature[1]-feature[0]:.3f}")

# 8. 使用拓扑特征进行数据分类或聚类（示例代码）
# 假设我们要基于持续同调特征进行聚类
from sklearn.cluster import KMeans

# 使用热核特征进行聚类
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(heat_features.reshape(1, -1))
print(f"\n基于拓扑特征的聚类结果: {clusters}")

# 在真实应用中，你可能会将这些拓扑特征与其他特征结合使用，
# 或者将它们输入到机器学习模型中进行更复杂的分析