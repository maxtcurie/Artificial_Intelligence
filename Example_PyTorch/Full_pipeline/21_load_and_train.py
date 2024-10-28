import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from joblib import load
import os

class JoblibDataset(Dataset):
    def __init__(self, file_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_list = file_list
        self.joblib_files = None

    def setup(self):
        self.joblib_files = [load(f, mmap_mode='r') for f in self.file_list]

    def __getitem__(self, index):
        if self.joblib_files is None:
            self.setup()

        file = self.joblib_files[index]
        input_dataset = file[:-1]
        output_dataset = file[1:]

        return (torch.tensor(input_dataset, device=self.device),
                torch.tensor(output_dataset, device=self.device))

    def __len__(self):
        return len(self.file_list)

    def check_dataset(self, batch_size):
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            try:
                assert inputs.shape == targets.shape, f"Mismatched shapes in batch {batch_idx}"
                print(f"Batch {batch_idx} passed the check with shape {inputs.shape}.")
            except Exception as e:
                print(f"Batch {batch_idx} failed the check: {e}")

# Custom collate function
def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Define the file paths
file_list = [f'stock_data/stock_data_{year}_processed.joblib' for year in range(2000, 2021)]

# Create an instance of the dataset
dataset = JoblibDataset(file_list)

# Check the dataset in batches
batch_size = 32
dataset.check_dataset(batch_size)

# Load dataset and create DataLoader with custom collate function
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Assume input_dim and output_dim based on your dataset
input_dim = dataset[0][0].shape[1]
output_dim = dataset[0][1].shape[1]

# Create the model
model = MLP(input_dim, output_dim)
model.to(dataset.device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping and checkpointing setup
early_stopping_patience = 5
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
best_loss = float('inf')
patience_counter = 0

# Training loop with early stopping and checkpointing
num_epochs = 50
checkpoint_saved = False

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in data_loader:
        inputs, targets = inputs.float(), targets.float()
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Early stopping and checkpointing
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        checkpoint_saved = True
        # Save the best model checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f"Checkpoint saved at epoch {epoch+1} with loss {epoch_loss}")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Check if the checkpoint was saved and load the best model
if checkpoint_saved:
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
else:
    print("No checkpoint saved during training.")

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')
print("Final model saved as final_model.pth")
