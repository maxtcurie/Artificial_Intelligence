import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# Custom dataset for loading pairs of images or data points
class SiameseDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Randomly sample positive and negative pairs
        x1, x2 = self.data[idx]
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x1, x2

# Siamese network model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),  # Adjust based on feature size
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Training function
def train_siamese(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, label in dataloader:
            optimizer.zero_grad()
            output1, output2 = model(x1.float(), x2.float())
            loss = criterion(output1, output2, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}")

# Inference function to detect anomalies
def detect_anomalies(model, normal_data, new_data, threshold):
    model.eval()
    with torch.no_grad():
        normal_embeddings = model.forward_one(torch.FloatTensor(normal_data))
        new_embeddings = model.forward_one(torch.FloatTensor(new_data))
        distances = pairwise_distances(new_embeddings, normal_embeddings, metric='euclidean').min(axis=1)
        anomaly_scores = distances > threshold
        return anomaly_scores, distances

# Sample data (replace with actual data)
# Assuming normal_data is your training set of "normal" instances
normal_data = np.random.randn(100, 128)  # 100 normal samples, 128 features
anomalous_data = np.random.randn(10, 128) + 5  # 10 anomalous samples

# Preparing pairs for contrastive learning
train_pairs = [(normal_data[i], normal_data[j], 1) for i in range(len(normal_data)) for j in range(i+1, len(normal_data))]

# DataLoader
dataloader = DataLoader(SiameseDataset(train_pairs), batch_size=32, shuffle=True)

# Initialize model, optimizer, and loss function
model = SiameseNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss()

# Train the model
train_siamese(model, dataloader, optimizer, criterion, epochs=10)

# Set threshold for anomaly detection (e.g., 95th percentile of distances in normal data)
with torch.no_grad():
    normal_embeddings = model.forward_one(torch.FloatTensor(normal_data))
    normal_distances = pairwise_distances(normal_embeddings, normal_embeddings, metric='euclidean')
    threshold = np.percentile(normal_distances, 95)

# Detect anomalies in new data
anomaly_scores, distances = detect_anomalies(model, normal_data, anomalous_data, threshold)
print("Anomaly Scores:", anomaly_scores)
print("Distances:", distances)
