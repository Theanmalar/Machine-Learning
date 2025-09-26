#k-means clustering
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  Load the dataset
df = pd.read_csv("wdbc.data", header=None)

#  Assign column names
df.columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# Drop ID column
df.drop('ID', axis=1, inplace=True)

#  Encode Diagnosis (M=1, B=0)
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])  # M → 1, B → 0

#  Separate features and labels
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Apply KMeans clustering (on training data)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

# Predict on test data
test_preds = kmeans.predict(X_test_scaled)

#  Align predicted labels
raw_acc = accuracy_score(y_test, test_preds)
inv_acc = accuracy_score(y_test, 1 - test_preds)
aligned_test_preds = test_preds if raw_acc >= inv_acc else 1 - test_preds
final_test_accuracy = max(raw_acc, inv_acc)

# ✅ Step 11: Accuracy Summary
print("\n📊 Accuracy Comparison on Test Data:")
print(f"Raw accuracy:      {raw_acc:.4f}")
print(f"Inverted accuracy: {inv_acc:.4f}")
print(f"✅ Final aligned test accuracy: {final_test_accuracy:.4f}")

# ✅ Step 12: Confusion Matrix
cm = confusion_matrix(y_test, aligned_test_preds)
conf_matrix_df = pd.DataFrame(
    cm,
    index=['Actual Benign', 'Actual Malignant'],
    columns=['Predicted Benign', 'Predicted Malignant']
)
print("\n📋 Confusion Matrix (Test Set):")
print(conf_matrix_df)

# ✅ Step 13: Classification Report
report_dict = classification_report(
    y_test,
    aligned_test_preds,
    target_names=['Benign', 'Malignant'],
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
print("\n📋 Classification Report Table (Test Set):")
print(report_df.round(2))

# ✅ Step 14: Prediction vs Actual Table
comparison_df = pd.DataFrame({
    'Actual Label': y_test.values,
    'Predicted Label': aligned_test_preds
})
comparison_df['Match'] = np.where(comparison_df['Actual Label'] == comparison_df['Predicted Label'], '✅ Correct', '❌ Wrong')

print("\n🧾 Prediction vs Actual Table (First 15 Test Samples):")
print(comparison_df.head(15))

# ✅ Step 15: PCA for visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

# Project cluster centroids to PCA space
centroids_2d = pca.transform(kmeans.cluster_centers_)

# Print centroid coordinates
centroid_df = pd.DataFrame(centroids_2d, columns=['PCA 1', 'PCA 2'])
print("\n📌 PCA Coordinates of Cluster Centroids:")
print(centroid_df.round(2))

# ✅ Step 16: Visualize clusters + centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=aligned_test_preds, palette='viridis', s=60)
plt.scatter(
    centroids_2d[:, 0], centroids_2d[:, 1],
    c='red', s=200, marker='X', edgecolors='black', label='Centroids'
)
plt.title("K-Means Clustering on Test Set (PCA Reduced) with Centroids")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
