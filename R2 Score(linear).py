import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
'Maths': [30, 45, 55, 70, 90, 20, 35, 80],
'Science': [35, 40, 60, 75, 85, 25, 30, 90],
'Passed': [0, 0, 1, 1, 1, 0, 0, 1] # Target variable (label)
}
df = pd.DataFrame(data)
X = df[['Maths', 'Science']]
y = df['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Predictions:", y_pred.tolist())
print("Actual:", y_test.tolist())
