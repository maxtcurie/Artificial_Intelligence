import numpy as np
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.echo_state_network import ESNRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.
X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

# Take only a subset to speed up the process
X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

# Create the ESN model
input_to_node = InputToNode(hidden_layer_size=500)
node_to_node = NodeToNode(hidden_layer_size=500, spectral_radius=0.9, leakage=0.2, bias_scaling=0.0)
base_esn = ESNRegressor(input_to_node=input_to_node, node_to_node=node_to_node, regressor=None)

# Train the ESN
base_esn.fit(X_train, y_train)

# Predict
y_pred = base_esn.predict(X_test)

# Round to the nearest integer for classification
y_pred = np.round(y_pred).astype(int)
y_pred = np.clip(y_pred, 0, 9)  # Ensuring predictions are within 0-9

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

