import numpy as np
from pyrcn.echo_state_network import ESNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Generate synthetic multivariate time series data
time_steps = 1000
features = 5
X = np.random.randn(time_steps, features)  # Random data with 5 features

# Introduce synthetic anomalies (for example, random spikes in data)
anomalies = np.zeros(time_steps)
anomaly_indices = np.random.choice(np.arange(100, time_steps), size=50, replace=False)
anomalies[anomaly_indices] = 1  # Label as anomaly

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, anomalies, test_size=0.3, random_state=42)

# Create an Echo State Network Classifier for anomaly detection
esn = ESNClassifier(random_state=42)

# Train the ESN
esn.fit(X_train, y_train)

# Predict using the ESN
y_pred = esn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot the true and predicted anomalies
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Anomalies', color='blue')
plt.plot(y_pred, label='Predicted Anomalies', color='red', linestyle='dashed')
plt.title(f'Anomaly Detection (Accuracy: {accuracy:.4f})')
plt.xlabel('Time Steps')
plt.ylabel('Anomalies')
plt.legend()
plt.show()
