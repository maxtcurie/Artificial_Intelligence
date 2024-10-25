import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from joblib import load
import numpy as np
import matplotlib.pyplot as plt

class JoblibDataset(Dataset):
    def __init__(self, file_list, device):
        self.device = device
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

        return torch.tensor(input_dataset, device=self.device), torch.tensor(output_dataset, device=self.device)

    def __len__(self):
        return len(self.file_list)

# Custom collate function
def custom_collate_fn(batch):
    inputs, outputs = zip(*batch)
    return torch.cat(inputs, dim=0), torch.cat(outputs, dim=0)

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

# Function to load the model and make predictions
def predict(file_list, model_path, scaler_path, device, batch_size=32):
    # Create an instance of the dataset
    dataset = JoblibDataset(file_list, device)

    # Load dataset and create DataLoader with custom collate function
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Assume input_dim and output_dim based on your dataset
    input_dim = dataset[0][0].shape[1]
    output_dim = input_dim  # Assuming the output_dim is the same as input_dim for prediction

    # Create the model
    model = MLP(input_dim, output_dim)
    model.to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    predictions = []
    outputs_og = []

    with torch.no_grad():
        for inputs, outputs_og_tmp in data_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            outputs_og.append(outputs_og_tmp.cpu().numpy())

    # Concatenate predictions and original outputs
    predictions = np.concatenate(predictions, axis=0)
    outputs_og = np.concatenate(outputs_og, axis=0)

    # Load the scaler and inverse transform the predictions and original outputs
    scaler = load(scaler_path)
    predictions_denorm = scaler.inverse_transform(predictions)
    outputs_og_denorm = scaler.inverse_transform(outputs_og)

    return predictions_denorm, outputs_og_denorm

# Define the file paths for prediction data
prediction_file_list = [f'stock_data/stock_data_{year}_processed.joblib' for year in range(2019, 2021)]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model
model_path = 'final_model.pth'

# Path to the saved scaler object
scaler_path = 'stock_data/stock_data_total_scaler.joblib'


# Make predictions
predictions_denorm, outputs_og_denorm = predict(prediction_file_list, model_path, scaler_path, device)

# Plot the predictions and original values
def plot_predictions(predictions, original, title="Predictions vs Original"):
    num_features = predictions.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 2*num_features))
    
    if num_features == 1:
        axes = [axes]
    
    for i in range(num_features):
        axes[i].plot(original[:, i], label="Original")
        axes[i].plot(predictions[:, i], label="Predicted", alpha=0.7)
        axes[i].set_title(f"{title} - Feature {i+1}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# Call the plot function
plot_predictions(predictions_denorm, outputs_og_denorm)
