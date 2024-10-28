from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
X = ...  # feature matrix
y = ...  # target vector

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()
mse_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mse_scores.append(mse)

print(f'Average MSE: {sum(mse_scores) / len(mse_scores)}')
