# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, accuracy_score

# Generating mock data
np.random.seed(42)
data_size = 100

# Features
X = pd.DataFrame({
    'Feature1': np.random.rand(data_size),
    'Feature2': np.random.rand(data_size),
    'Feature3': np.random.rand(data_size)
})

# Target variable (binary classification)
y = np.random.choice([0, 1], size=data_size)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Training the classifier
clf.fit(X_train, y_train)

# Making predictions
y_pred = clf.predict(X_test)

# Evaluating the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Displaying the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Displaying the tree structure
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n", tree_rules)

# Plotting the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=list(X.columns), class_names=['0', '1'], filled=True)
plt.show()
