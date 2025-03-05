# Akash Das
# MIT IDS.147[J] Statistical Machine Learning and Data Science
# Module 2: Gene Engineering

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.style import rcmod
from yellowbrick.style.colors import resolve_colors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# -- Part 1: Visualization --

# -- Load Data -- 
X = np.load('/Users/akashdas/Genomics_Module2/data/p2_unsupervised/X.npy')
X_transformed = np.log2(X + 1)
print(f"X-log transformed shape {X_transformed.shape}")
print(f"Largest value in column 1 {max(X_transformed[:, 0])}")

# -- Visualization - PCA -- 
pca = PCA()
trans_pca = pca.fit_transform(X_transformed)
plt.scatter(trans_pca[:, 0], trans_pca[:, 1])
plt.title("Log-Transformed PCA Plot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# PCA - Explained Variance Plot
plt.plot(np.arange(1, trans_pca.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_[:]))
plt.title("Explained PCA Plot")
plt.xlabel("# Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()

threshold = 0.85
num_pca_components = np.searchsorted(np.cumsum(pca.explained_variance_ratio_[:]), threshold) + 1
print(f"Number of PCs to explain {threshold * 100}% of variance: {num_pca_components}")

# -- Visualization - MDS -- 
mds = MDS().fit_transform(trans_pca[:, 0:]) # Fit MDS on all PCs
plt.scatter(mds[:, 0], mds[:, 1])
plt.title("MSD Plot")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# -- KMeans Clustering  --
# Find # clusters via elbow plot
plt.figure(figsize = (10, 6))
plt.plot(np.arange(1, 11), [KMeans(n_clusters = i, n_init = 50).fit(X_transformed).inertia_ for i in range(1, 11)]) # Looks like 3 clusters is ideal
plt.title("KMeans Sum of Squares Criterion for Clustering")
plt.xlabel("# Components")
plt.ylabel("Within-Cluster Sum of Squares")
plt.show()

# Find # clusters via silhouette scores
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters = k, n_init = 50)
    labels = kmeans.fit_predict(X_transformed)
    silhouette_avg = silhouette_score(X_transformed, labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores for different k values
plt.figure(figsize=(8, 5))
plt.plot(np.arange(2, 11), silhouette_scores, marker='o', linestyle='-', color='b') # Looks like 3 clusters is ideal
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different k values")
plt.show()

# Plot k = 3 clusters for KMeans
n_clusters = 3
clustering = KMeans(n_clusters = n_clusters, n_init = 50).fit(trans_pca) # Fit KMeans to PCA
labels = clustering.labels_
centroids = clustering.cluster_centers_
colors = np.array(resolve_colors(n_clusters, 'yellowbrick'))

plt.figure(figsize=(8, 6))
plt.scatter(trans_pca[:, 0], trans_pca[:, 1], c=colors[labels], alpha=0.6, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200, label='Centroids')
plt.title("K-Means Clustering (k = 3)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.show()

# Sub-cells in the clustered cells
for i in range(n_clusters):
    sub_data = X_transformed[clustering.labels_ == i]  

    kmeans = KMeans(n_clusters=3, n_init=50).fit(sub_data)
    sub_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    colors = np.array(resolve_colors(3, 'yellowbrick'))

    pca_sub_data = pca.fit_transform(sub_data)
    plt.figure(figsize = (10, 6))
    plt.scatter(pca_sub_data[:, 0], pca_sub_data[:, 1], c=colors[sub_labels], alpha=0.6, edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200, label='Centroids')
    
    plt.title(f"Sub-structures in k = {i + 1} cluster")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()

# -- Part 2: Unsupervised Feature Selection -- 

# -- Cluster Data --

kmeans_model_path = "kmeans_model.pkl"
if os.path.exists(kmeans_model_path):
    print("Loading existing KMeans model...")
    clusters = joblib.load(kmeans_model_path)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_

else:
    print("Training KMeans model...")
    n_clusters = 3  # From elbow / silhouette plot
    clusters = KMeans(n_clusters=n_clusters, n_init=50).fit(X_transformed)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    joblib.dump(clusters, kmeans_model_path)  # Save the model

colors = np.array(resolve_colors(n_clusters, 'yellowbrick'))
plt.figure(figsize=(8, 6))
plt.scatter(trans_pca[:, 0], trans_pca[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=200, label='Centroids')
plt.title(f'Clustering KMeans Method for k={n_clusters}')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.show()

# -- Logisitic Regression --
y = labels
X_train, X_test, y_train, y_test = train_test_split(X_transformed, labels, test_size=0.3, shuffle=True) # Retain 30% of samples for testing

log_reg_model_path = "log_reg_model.pkl"
if os.path.exists(log_reg_model_path):
    print("Loading existing Logistic Regression model...")
    log_reg = joblib.load(log_reg_model_path)
    log_reg_model_score = log_reg.score(X_train, y_train)
    log_reg_model_C_score = log_reg.C_
    log_reg_model_score_test = log_reg.score(X_test, y_test)

    print(f"L2 Model Score: {log_reg_model_score}")
    print(f"L2 Model Score (Test): {log_reg_model_score_test}")

else:
    print("Training Logistic Regression model...")
    log_reg = LogisticRegressionCV(cv=5, Cs=[0.001, 0.01, 0.1, 1, 10], penalty='l2', multi_class='ovr', solver='liblinear', max_iter=5000).fit(X_train, y_train)
    joblib.dump(log_reg, log_reg_model_path)  # Save the trained model

    log_reg_model_score = log_reg.score(X_train, y_train)
    log_reg_model_C_score = log_reg.C_
    log_reg_model_score_test = log_reg.score(X_test, y_test)

    print(f"L2 Model Score: {log_reg_model_score}")
    print(f"L2 Model Score (Test): {log_reg_model_score_test}")

n_feature = 100
top_coef = np.sum(np.abs(log_reg.coef_),axis=0) # Find features with largest values
top_features = np.argsort(top_coef, axis=0)[-n_feature:]
rand_features = np.random.choice(X_transformed.shape[1], n_feature, replace=False)
max_var_features = np.var(X_transformed, axis = 0).argsort()[::-1][:n_feature]

all_features = [top_features, rand_features, max_var_features]

test_scores = []
for feature in all_features:
    X = X_transformed[:, np.array(feature)]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, shuffle=True)
    log_reg_mod = LogisticRegressionCV(cv = 5, Cs=[0.001, 0.01, 0.1, 1, 10], penalty='l2', multi_class='ovr', solver='liblinear', max_iter=5000).fit(X_train, y_train)
    
    model_score = log_reg_mod.score(X_test, y_test)
    test_scores.append(model_score)

print(test_scores)

# Histogram comparing feature variances
variance_max_features = np.var(X_transformed[:, top_features], axis = 0)
variance_max_var_features = np.var(X_transformed[:, max_var_features], axis = 0)
plt.hist(variance_max_features, color='b', alpha=0.3,bins=20)
plt.hist(variance_max_var_features, color='r', alpha=0.3,bins=20)
plt.title('Variance comparisons between max_features and max_variance_features')
plt.show()

# -- Part 3: Influence of Hyper-parameters -- 

# T-SNE using 10, 50, 100, 250, and 500 PCs
tsne = TSNE(n_components=2, perplexity=40)

for i in [10, 50, 100, 250, 500]:
    tsne_fit = tsne.fit_transform(trans_pca[:, 0:i])
    plt.title(f"TSNE Fit with {i} PCs")
    plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1])
    plt.show()

# Hyperparameter Testing: TSNE Perplexity
for i in [10, 20, 30, 40, 50, 100, 200]:
    tsne_perp_test = TSNE(n_components=2, perplexity=i).fit_transform(trans_pca[:, 0:50]) # First 50 PCs
    plt.title(f"TSNE with {i} Perplexity")
    plt.scatter(tsne_perp_test[:, 0], tsne_perp_test[:, 1])
    plt.show()

# Hyperparameter Testing: TSNE Learning Rate
for rate in [10, 50, 100, 150, 200, 400, 500, 1000]:
    tsne_LR_test = TSNE(n_components=2, perplexity=40, learning_rate=rate).fit_transform(trans_pca[:, 0:50]) # First 50 PCs
    plt.title(f"TSNE with {rate} Learning Rate")
    plt.scatter(tsne_LR_test[:, 0], tsne_LR_test[:, 1])
    plt.show()

# Hyperparameter Testing: Regularization Impact - Log Regression
l1_log_name = 'log_reg_model_l1.pkl'
if os.path.exists(l1_log_name):
    print("Loading existing l1 Log Regression model...")
    l1_log_reg = joblib.load(l1_log_name)
    l1_model_score = l1_log_reg.score(X_train, y_train)
    l1_model_score_test = l1_log_reg.score(X_test, y_test)
    print(f"L1 Model Score: {l1_model_score}")
    print(f"L1 Model Score (Test): {l1_model_score_test}")

else:
    l1_log_reg = LogisticRegressionCV(cv=5, Cs = [0.001, 0.01, 0.1, 1, 10], penalty='l1', solver='liblinear', multi_class='ovr', max_iter=5000).fit(X_train, y_train)
    l1_model_score = l1_log_reg.score(X_train, y_train)
    l1_model_score_test = l1_log_reg.score(X_test, y_test)
    print(f"L1 Model Score: {l1_model_score}")
    print(f"L1 Model Score (Test): {l1_model_score_test}")

net_log_name = 'log_reg_model_net.pkl'
if os.path.exists(net_log_name):
    print("Loading existing elastic net Log Regression model...")
    net_log_reg = joblib.load(net_log_name)
    net_model_score = net_log_reg.score(X_train, y_train)
    net_model_score_test = net_log_reg.score(X_test, y_test)
    print(f"Elastic Net Model Score: {net_model_score}")
    print(f"Elastic Net Model Score (Test): {net_model_score_test}")

else:
    net_log_reg = LogisticRegressionCV(cv=5, Cs = [0.001, 0.01, 0.1, 1, 10], penalty='elasticnet', solver='saga', multi_class='ovr', max_iter=5000, l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9]).fit(X_train, y_train)
    net_model_score = net_log_reg.score(X_train, y_train)
    net_model_score_test = net_log_reg.score(X_test, y_test)
    print(f"Elastic Net Model Score: {net_model_score}")
    print(f"Elastic Net Model Score (Test): {net_model_score_test}")