import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# -- Load Data -- 
X = np.load('./data/p1/X.npy')
y = np.load('./data/p1/y.npy')
'''print(f"Matrix Shape: {X.shape}")'''
'''print(f"Maximum Value in Column 1: {max(X[:, 0])}")'''

# -- Transform data to log-scale --
X_transform = np.log2(X+1)
'''print(f"Maximum Value in Column 1: {max(X_transform[:, 0])}") '''

# -- PCA - original and transformed data --
pca_x = PCA()
pca_x.fit_transform(X)
'''print(f'% Variance explained (PC1) for X: {pca_x.explained_variance_ratio_[:1]}')'''

pca_trans = PCA()
pca_trans.fit_transform(X_transform)
'''print(f'% Variance explained (PC1) for X-log-transformed: {pca_trans.explained_variance_ratio_[:1]}')'''

# How many PC should we include to explain a given threshold of variance?
X_cumulative_variance = np.cumsum(pca_x.explained_variance_ratio_)
X_transform_cumulative_variance = np.cumsum(pca_trans.explained_variance_ratio_)

threshold = 0.85
X_num_components = np.searchsorted(X_cumulative_variance, threshold) + 1 # searchsored finds the idx position, +1 to get the right PC number
'''print(f'Required #PCs to explain {threshold * 100}% of the variance in X: {X_num_components}')'''
X_transform_num_components = np.searchsorted(X_transform_cumulative_variance, threshold) + 1 
'''print(f'Required #PCs to explain {threshold * 100}% of the variance in X: {X_transform_num_components}')'''

# Visualization - PC
plt.scatter(X_transform[:, 0], X_transform[:, 1])
plt.title("First and second components of X_transform", size = 18)
plt.axis("equal")
z = pca_trans.fit_transform(X_transform) # Create numpy.ndarray type for visualization
plt.scatter(z[:, 0], z[:, 1]) # First and second PCs
plt.title("First and second PCs of X_transform", size = 18)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis("equal")

# -- MDS -- 
mds = MDS()
mds.fit(X_transform)
plt.scatter(mds.embedding_[:,0], mds.embedding_[:,1],c=y)
plt.title("MDS Plot", size = 18)
plt.axis("equal")

# -- tSNE -- 
X_projet_to_50 = np.matmul(X_transform, pca_trans.components_[:50].T) # Project data to top 50 PC
tsne = TSNE(n_components = 2, perplexity = 40)
z_tsne = tsne.fit_transform(X_projet_to_50)
plt.scatter(z_tsne[:,0], z_tsne[:,1],c=y)
plt.title("TSNE on first 50 PCs, perplexity 40",size=18)
plt.axis("equal")

# -- Clustering -- [log-transformed data projected onto the top 50 PCs]
n_clusters = 5
clustering = KMeans(n_clusters=n_clusters, n_init=50).fit(X_projet_to_50)
pca_50 = PCA()
pca_X_top50 = pca_50.fit_transform(X_projet_to_50)
plt.scatter(pca_X_top50[:,0], pca_X_top50[:,1])
plt.title('KMeans single-cell RNA-seq transformed, PCA', size=15)
plt.xlabel('PC1')
plt.ylabel('PC2')

# Clustering via MDS
mds_X_top50 = MDS(verbose=1, eps=1e-5).fit_transform(X_projet_to_50)
plt.scatter(mds_X_top50[:,0], mds_X_top50[:,1])
plt.title('KMeans single-cell RNA-seq transformed, MDS', size=15)

# Clustering via tSNE
tsne_X_top50 = TSNE(n_components=2, perplexity=40).fit_transform(X_projet_to_50)
plt.scatter(tsne_X_top50[:,0], tsne_X_top50[:,1])
plt.title('KMeans single-cell RNA-seq transformed, PCA', size=15)

# Optimal Clusters - Elbow Plot
plt.plot(np.arange(1, 10), [KMeans(i, n_init=50).fit(X_projet_to_50).inertia_ for i in range(1, 10)])
plt.xticks(np.arange(1, 10, step=1))
plt.title('KMeans Sum of Squares Criterion', size=15)
plt.xlabel('#Clusters')
plt.ylabel('Within Group Sum of Squares (WGSS)')
plt.show()

n_clusters = 4
kmeans_50 = KMeans(n_clusters, n_init=50).fit(X_projet_to_50)
print(f'WGSS for {n_clusters} clusters: {kmeans_50.inertia_:.3g}')

# Clustering via medoids
# PCA
kmeans_X_centroid = KMeans(n_clusters, n_init=50).fit(X_transform).cluster_centers_
pca_X_centroid = PCA()
pca_kmeans_X_centroid = pca_X_centroid.fit_transform(kmeans_X_centroid)
plt.scatter(pca_kmeans_X_centroid[:,0], pca_kmeans_X_centroid[:,1])

# MDS
mds_kmeans_X_centroid = MDS(n_components=2, verbose=1, eps=1e-5).fit_transform(kmeans_X_centroid)
plt.scatter(mds_kmeans_X_centroid[:,0], mds_kmeans_X_centroid[:,1])
plt.show()
pca_X = PCA().fit_transform(X)
plt.scatter(pca_X[:,0], pca_X[:,1])
plt.title('single-cell RNA-seq, PCA', size=15)
plt.show()
mds_X = MDS(verbose=1, eps=1e-5).fit_transform(X)
plt.scatter(mds_X[:,0], mds_X[:,1])
plt.title('single-cell RNA-seq, MDS', size=15)

#TSNE
tsne_X = TSNE(n_components=2, perplexity=40).fit_transform(X)
plt.scatter(tsne_X[:,0], tsne_X[:,1])
plt.title('KMeans single-cell RNA-seq, t-SNE', size=15)
plt.show()