from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)
