import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 模拟预训练后的高维特征和伪标签（实际数据请替换）
# 假设 features 形状为 (N, D)

def visualize_clusters(features, pseudo_labels):
    """
    使用 t-SNE 可视化聚类结果
    :param features: 高维特征，形状为 (N, D)
    :param pseudo_labels: 伪标签，形状为 (N,)
    """
    # features = np.random.rand(1000, 512)  # 例如1000个样本，每个样本512维
    # pseudo_labels = np.random.randint(0, 10, size=1000)  # 模拟10个类别的伪标签

    # 第一步：使用 PCA 降维到 50 维（作为 t-SNE 的预处理步骤）
    pca = PCA(n_components=50)
    features_reduced = pca.fit_transform(features)

    # 第二步：使用 t-SNE 将特征进一步降到 2 维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_reduced)

    # 第三步：可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=pseudo_labels, cmap='viridis', alpha=0.7)
    plt.title("基于预训练特征与伪标签的聚类可视化")
    plt.xlabel("降维维度1")
    plt.ylabel("降维维度2")
    plt.colorbar(scatter, label='伪标签')
    plt.show()

