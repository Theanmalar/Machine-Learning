import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Dataset
x = [90, 85, 40, 35, 70, 65]
y = [12, 10, 4, 3, 8, 7]
students = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
data = np.array(list(zip(x, y)))

# Step 2: First KMeans clustering
kmeans1 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels1 = kmeans1.fit_predict(data)
centroids1 = kmeans1.cluster_centers_

# --- Manual centroid computation using KMeans formula ---
print("0 Centroids old:")
for k in range(2):
    points = data[labels1 == k]
    centroid_manual = points.mean(axis=0)  # μ_k = mean of points in cluster
    print(f"Cluster {k}: ({centroid_manual[0]:.2f}, {centroid_manual[1]:.2f})")

# Step 3: Second KMeans clustering
kmeans2 = KMeans(n_clusters=2, random_state=1, n_init=10)
labels2 = kmeans2.fit_predict(data)
centroids2 = kmeans2.cluster_centers_

# Align second clustering labels with first
acc = np.sum(labels1 == labels2)
inv_acc = np.sum(labels1 == (1 - labels2))
if inv_acc > acc:
    labels2 = 1 - labels2

# Step 4: Build DataFrame
df = pd.DataFrame(data, columns=["Score", "Hours"])
df["Student"] = students
df["Old_Cluster"] = labels1
df["New_Cluster"] = labels2
df = df[["Student", "Score", "Hours", "Old_Cluster", "New_Cluster"]]

print("\n📊 Final Cluster Comparison Table:")
print(df.to_string(index=False))

# Step 5: Plot clusters with centroid coordinates
plt.figure(figsize=(8, 6))
colors = df["Old_Cluster"].map({0: 'blue', 1: 'green'})
plt.scatter(df["Score"], df["Hours"], c=colors, s=100, edgecolor='black')

for i in range(len(df)):
    plt.text(df["Score"][i]+1, df["Hours"][i], df["Student"][i], fontsize=9)

# Plot old centroids
for x, y in centroids1:
    plt.scatter(x, y, c='red', marker='X', s=200)
    plt.text(x+1, y+0.5, f"({x:.1f},{y:.1f})", color='red', fontsize=9)

# Plot new centroids
for x, y in centroids2:
    plt.scatter(x, y, c='orange', marker='X', s=150)
    plt.text(x+1, y-1, f"({x:.1f},{y:.1f})", color='orange', fontsize=9)

plt.title("K-Means Clustering Comparison")
plt.xlabel("Score")
plt.ylabel("Hours")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Show students in each cluster
print("\nCluster 0 Students:", set(df[df['Old_Cluster']==0]['Student']))
print("Cluster 1 Students:", set(df[df['Old_Cluster']==1]['Student']))
