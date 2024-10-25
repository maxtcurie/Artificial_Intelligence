import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100  # Latent dimension for the generator
data_dim = 40    # Number of channels in the data
batch_size = 64
learning_rate = 0.0002
num_epochs = 1000
mask_ratio = 0.5  # Ratio of masked channels

# Define a dataset with masked data
class MaskedDataset(Dataset):
    def __init__(self, size=1000, channels=40, mask_ratio=0.5):
        self.data = np.random.randn(size, channels)
        self.masked_data = self.data.copy()
        self.mask = np.random.rand(size, channels) < mask_ratio
        self.masked_data[self.mask] = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.masked_data[idx], dtype=torch.float32), 
                torch.tensor(self.data[idx], dtype=torch.float32))

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


# Initialize dataset and dataloader
dataset = MaskedDataset(size=1000, channels=data_dim, mask_ratio=mask_ratio)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(data_dim, data_dim)
discriminator = Discriminator(data_dim)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for masked_data, real_data in dataloader:
        batch_size = real_data.size(0)
        
        # Train Discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Compute loss with real data
        outputs = discriminator(real_data)
        d_loss_real = criterion(outputs, real_labels)
        
        # Generate fake data from masked data
        fake_data = generator(masked_data)
        
        # Compute loss with fake data
        outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        fake_data = generator(masked_data)
        outputs = discriminator(fake_data)
        
        # Compute generator loss
        g_loss = criterion(outputs, real_labels)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Visualize generated data
masked_data, real_data = next(iter(dataloader))
generated_data = generator(masked_data).detach().numpy()
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title('Masked Data')
plt.imshow(masked_data.numpy(), aspect='auto', cmap='viridis')
plt.subplot(3, 1, 2)
plt.title('Generated Data')
plt.imshow(generated_data, aspect='auto', cmap='viridis')
plt.subplot(3, 1, 3)
plt.title('Real Data')
plt.imshow(real_data.numpy(), aspect='auto', cmap='viridis')
plt.show()
