import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5],
    [5, 6], [6, 7], [7, 8], [8, 9]
])


pca = PCA(n_components=1)
transformed_data = pca.fit_transform(data)


plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data (2 Features)')
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.scatter(transformed_data, np.zeros_like(transformed_data))
plt.title('Transformed Data (1 Principal Component)')
plt.xlabel('Principal Component 1')
plt.yticks([])
plt.grid(True)

plt.tight_layout()
plt.show()

print("Original Data Shape:", data.shape)
print("Transformed Data Shape:", transformed_data.shape)
print("Transformed Data (First 5 rows):\n", transformed_data[:5])
