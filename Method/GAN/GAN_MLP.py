import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generate synthetic data
def generate_random_signals(num_samples, num_channels, seq_length):
    data = np.random.randn(num_samples, num_channels, seq_length)
    return data

seq_length = 101  # Length of each time series
num_samples = 1000  # Number of samples
num_channels = 40  # Number of input channels
output_channels = 71  # Number of output channels

data = generate_random_signals(num_samples, num_channels, seq_length)
data = data.reshape((num_samples, num_channels * seq_length))
dataset = TensorDataset(torch.Tensor(data), torch.Tensor(np.ones((num_samples, 1))))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize models, loss function, and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 40 * 101  # Length of input noise vector
output_dim = 71  # Length of generated time series
lr = 0.0002

generator = Generator(input_dim, output_dim).to(device)
discriminator = Discriminator(output_dim).to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Train the GAN
num_epochs = 50

for epoch in range(num_epochs):
    for i, (real_data, _) in enumerate(dataloader):
        real_data = real_data.to(device)
        real_labels = torch.ones(real_data.size(0), 1).to(device)
        fake_labels = torch.zeros(real_data.size(0), 1).to(device)

        # Train Discriminator
        z = torch.randn(real_data.size(0), input_dim).to(device)
        fake_data = generator(z)
        
        real_loss = criterion(discriminator(real_data.view(-1, input_dim)[:, :output_dim]), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        g_loss = criterion(discriminator(fake_data), real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} \
                  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

# Generate and save signals
def save_signal(signal, path):
    plt.plot(signal)
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.savefig(path)
    plt.close()

with torch.no_grad():
    z = torch.randn(64, input_dim).to(device)
    generated_data = generator(z).cpu().numpy()

    for i in range(10):
        save_signal(generated_data[i], f"generated_signal_{i}.png")
