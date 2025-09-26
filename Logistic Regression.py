import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Maths': [30, 60, 55, 70, 90, 30, 45, 80],
    'Science': [35, 90, 60, 75, 85, 60, 30, 90],
    'Passed': [0, 0, 1, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)
X = df[['Maths', 'Science']]
y = df['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Predictions:", y_pred.tolist())
print("Actual:", y_test.tolist())
