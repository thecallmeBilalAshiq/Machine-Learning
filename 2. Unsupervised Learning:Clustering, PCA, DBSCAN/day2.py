import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate synthetic customer data (replace with customer_data.csv for large dataset)
np.random.seed(42)
n_customers = 300
data = pd.DataFrame({
    'total_amount': np.random.gamma(2, 300, n_customers),
    'avg_amount': np.random.gamma(2, 50, n_customers),
    'frequency': np.random.randint(1, 20, n_customers),
    'electronics_spend': np.random.gamma(2, 100, n_customers),
    'clothing_spend': np.random.gamma(2, 80, n_customers),
    'grocery_spend': np.random.gamma(2, 60, n_customers)
})

# 2. Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 3. Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.2%}")

# 4. K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)

# 5. DBSCAN clustering
dbscan = DBSCAN(eps=0.7, min_samples=5)  # Increased eps for better clustering
dbscan_labels = dbscan.fit_predict(pca_data)

# 6. Plot PCA-reduced data with KMeans and DBSCAN labels
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='Set1', s=40)
axs[0].set_title("K-Means Clusters")
axs[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
axs[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

axs[1].scatter(pca_data[:, 0], pca_data[:, 1], c=dbscan_labels, cmap='Set2', s=40)
axs[1].set_title("DBSCAN Clusters")
axs[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
axs[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

plt.suptitle("Customer Segmentation using PCA + Clustering")
plt.tight_layout()
plt.show()

# 7. Evaluation with Silhouette Score
kmeans_silhouette = silhouette_score(pca_data, kmeans_labels)
dbscan_unique_labels = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_silhouette = silhouette_score(pca_data, dbscan_labels) if dbscan_unique_labels > 1 else -1
print(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
print(f"DBSCAN: {dbscan_unique_labels} clusters, {list(dbscan_labels).count(-1)} outliers")

# 8. Cluster Profiles
data['kmeans_cluster'] = kmeans_labels
print("\nK-Means Cluster Profiles (Mean Values):")
print(data.groupby('kmeans_cluster').mean())

data['dbscan_cluster'] = dbscan_labels
print("\nDBSCAN Cluster Profiles (Excluding Outliers):")
print(data[data['dbscan_cluster'] != -1].groupby('dbscan_cluster').mean())

# 9. Marketing Recommendation
print("\nMarketing Recommendation:")
if kmeans_silhouette > dbscan_silhouette and kmeans_silhouette > 0.5:
    print("Use K-Means for broad customer segments (e.g., high vs. low spenders).")
elif dbscan_silhouette > kmeans_silhouette and dbscan_silhouette > 0.5 and dbscan_unique_labels > 1:
    print("Use DBSCAN for niche segments and outlier targeting (e.g., VIP customers).")
else:
    print("Clustering is weak (low silhouette scores). Try tuning parameters or using a larger dataset.")