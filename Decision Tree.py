import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
data = {
'Maths': [30, 45, 55, 70, 90, 20, 35, 80],
'Science': [35, 40, 60, 75, 85, 25, 30, 90],
'Passed': [0, 0, 1, 1, 1, 0, 0, 1] # Target variable (label)
}
df = pd.DataFrame(data)
X = df[['Maths','Science']]
y = df['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Predictions:", y_pred.tolist())
print("Actual: ", y_test.tolist())
