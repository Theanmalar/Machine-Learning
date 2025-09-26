import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
points_dict = {
    "P1": (3, 7), "P2": (4, 6), "P3": (5, 5), "P4": (6, 4),
    "P5": (7, 3), "P6": (6, 2), "P7": (7, 2), "P8": (8, 4),
    "P9": (3, 3), "P10": (2, 6), "P11": (3, 5), "P12": (2, 4)
}
labels_points = list(points_dict.keys())
points = np.array(list(points_dict.values()))
dist_matrix = cdist(points, points, metric="euclidean")
df_dist = pd.DataFrame(dist_matrix, index=labels_points, columns=labels_points)
print("\n===== Euclidean Distance Table =====\n")
print(df_dist.round(2))
eps = 1.9
neighbors_dict = {}
for i, p in enumerate(points):
    neighbors = [labels_points[j] for j in range(len(points)) if dist_matrix[i, j] <= eps]
    neighbors_dict[labels_points[i]] = neighbors

min_samples = 4
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(points)
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
core_status = ["Core" if core_mask[i] else "Noise" if labels[i]==-1 else "Noise" for i in range(len(points))]
border_status = ["Border" if (not core_mask[i] and labels[i]!=-1) else "-" for i in range(len(points))]
table_rows = []
for i, name in enumerate(labels_points):
    table_rows.append({
        "Datapoints": f"{name} → {', '.join(neighbors_dict[name])}",
        "Core": core_status[i],
        "Border": border_status[i]
    })

df_table = pd.DataFrame(table_rows)
print("\n===== DBSCAN DataPoints with Neighbors, Core & Border =====")
print(df_table)
plt.figure(figsize=(8,7))
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = "k"
    xy = points[labels==k]
    plt.scatter(xy[:,0], xy[:,1], c=[col], s=200, edgecolors='k', linewidths=1, label=f"Cluster {k}" if k!=-1 else "Noise")

for name, (x,y) in points_dict.items():
    plt.text(x+0.1, y+0.1, name, fontsize=9)

plt.title("DBSCAN Grouped Points (eps=1.9, minPts=4)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.show()

