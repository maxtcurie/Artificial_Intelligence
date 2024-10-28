import pandas as pd
from collections import Counter

# Example dataset
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'feature2': [1.2, 1.4, 1.5, 1.7, 1.8, 1.9, 2.0, 2.2, 2.3, 2.4],
    'label':    [0,   0,   0,   0,   0,   0,   1,   1,   1,   1]  # 6 normal, 4 anomalies (imbalanced)
}

df = pd.DataFrame(data)

# Separate normal and anomaly data
normal_data = df[df['label'] == 0]
anomaly_data = df[df['label'] == 1]

# Show original class distribution
print(f"Original dataset shape: {Counter(df['label'])}")

# Oversample anomaly data to match the number of normal data points
anomaly_oversampled = anomaly_data.sample(len(normal_data), replace=True, random_state=42)

# Combine the normal and oversampled anomaly data
df_balanced = pd.concat([normal_data, anomaly_oversampled])

# Show the new class distribution
print(f"Balanced dataset shape: {Counter(df_balanced['label'])}")

# Display the resampled dataset
print(df_balanced)
