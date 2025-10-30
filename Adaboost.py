# Adaboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. Create a weak learner (a small decision tree)
weak_learner = DecisionTreeClassifier(max_depth=1)

# 4. Create AdaBoost model (use base_estimator for older sklearn)
model = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=50, learning_rate=1.0)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# 8. Print results
print("Predicted labels:", y_pred)
print("Actual labels   :", y_test)
print("Accuracy:", accuracy)
