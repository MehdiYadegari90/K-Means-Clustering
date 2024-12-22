import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df = pd.read_csv('Clustering.csv')

# Drop the 'Gender' column as it's not needed for clustering
df = df.drop("Gender", axis=1)
# Uncomment to view the first few rows of the DataFrame
# print(df.head())

# Extract feature values and handle any NaN values
x = df.values[:, 1:]  # Get all rows and columns except the first one
x = np.nan_to_num(x)  # Replace NaN with 0

# Standardize the feature values
X = StandardScaler().fit_transform(x)
# Uncomment to view the standardized features
# print(X)

# Define the number of clusters for KMeans
Cluster_number = 3

# Initialize KMeans with specified parameters
k_means = KMeans(init="k-means++", n_clusters=Cluster_number, n_init=12)

# Fit the KMeans model to the standardized data
k_means.fit(X)

# Get the cluster labels for each point
labels = k_means.labels_

# Add the cluster labels to the original DataFrame
df["Clus_km"] = labels
# Uncomment to view the first 20 rows including cluster labels
# print(df.head(20))
# Uncomment to see the mean values of each cluster
# print(df.groupby("Clus_km").mean())

# Plotting the clusters in a 2D scatter plot
area = np.pi * (X[:, 1])**2  # Area of the points based on the second feature
plt.scatter(X[:, 0], X[:, 2], s=area, c=labels.astype(float), alpha=1)
plt.xlabel("Age", fontsize=18)
plt.ylabel("Income", fontsize=16)
plt.title("2D Scatter Plot of Clusters")
plt.show()

# Plotting the clusters in a 3D scatter plot
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)
ax.set_xlabel("Education")
ax.set_ylabel("Age")
ax.set_zlabel("Income")
ax.scatter(X[:, 1], X[:, 0], X[:, 2], c=labels)
plt.title("3D Scatter Plot of Clusters")
plt.show()
