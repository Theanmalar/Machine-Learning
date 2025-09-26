from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Iris dataset as a pandas DataFrame
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['species'] = iris.target_names[iris.target]  # map species labels :contentReference[oaicite:1]{index=1}

# 2. Inspect basic stats and structure
print(df.head())
print(df.describe())

# 3. Visualize feature relationships
sns.set(style="ticks")
sns.pairplot(df, hue="species")
plt.show()  # shows scatter + distributions per species :contentReference[oaicite:2]{index=2}

# 4. Preprocessing: train/test split + feature scaling
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # :contentReference[oaicite:3]{index=3}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train a simple KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# 6. Evaluate the model
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. (Optional) Use model to predict some new samples
new_samples = [
    [5.1, 3.5, 1.4, 0.2],
    [6.3, 3.3, 6.0, 2.5],
    [5.9, 3.0, 4.2, 1.5]
]
y_new = knn.predict(scaler.transform(new_samples))
print("New sample predictions:", iris.target_names[y_new])
