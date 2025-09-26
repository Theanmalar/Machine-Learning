import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Create a sample dataset with two features
# Let's assume two features, 'feature_A' and 'feature_B'
data = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5 , 1.6],
    [1.1 , 0.9]
])
# 2. Initialize PCA and specify the number of components
# To reduce two features to one, set n_components=1
pca = PCA(n_components=1)

# 3. Fit PCA to the data and transform it
# This will project the 2D data onto a 1D principal component
transformed_data = pca.fit_transform(data)

# 4. (Optional) Visualize the original and transformed data/size of the entire figure using width and height in inches.
plt.figure(figsize=(8, 4))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1])#values from feature A and values from feature B
plt.title('Original Data (2 Features)')
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.grid(True)

# Transformed data (1 Principal Component)
plt.subplot(1, 2, 2)
plt.scatter(transformed_data, np.zeros_like(transformed_data)) # Plotting on a single axis for 1D
plt.title('Transformed Data (1 Principal Component)')
plt.xlabel('Principal Component 1')
plt.yticks([]) # Remove y-axis ticks as it's 1D
plt.grid(True)

plt.tight_layout()#left and right plot dont overlap
plt.show()

print("Original Data Shape:", data.shape)
print("Transformed Data Shape:", transformed_data.shape)
print("Transformed Data (First 5 rows):\n", transformed_data[:5])
