
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt



data = fetch_california_housing()
X = data.data
y = data.target



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



pca = PCA(n_components=2)  # Let's reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)



print(f"Explained variance ratio for each principal component: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_)}")



plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
plt.colorbar(label='Target (Median House Value)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on California Housing Dataset')
plt.show()





