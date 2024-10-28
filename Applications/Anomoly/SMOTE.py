from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

# Example dataset (same as before)
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'feature2': [1.2, 1.4, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.3, 2.4],
    'label':    [0,   0,   0,   0,   0,   0,   1,   1,   1,   1]  # 6 normal, 4 anomalies (imbalanced)
}

df = pd.DataFrame(data)

# Splitting into features and labels
X = df[['feature1', 'feature2']]
y = df['label']

# Show the original class distribution
print(f"Original dataset shape: {Counter(y)}")

# Apply SMOTE with k_neighbors adjusted to avoid the error
smote = SMOTE(k_neighbors=3, random_state=42)  # Adjust k_neighbors to 3

# Resampling the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

# Show the resampled class distribution
print(f"Resampled dataset shape: {Counter(y_resampled)}")
