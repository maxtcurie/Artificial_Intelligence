import torch
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

# Define the file paths
file_list = [f'stock_data/stock_data_{year}_processed.joblib' for year in range(2000, 2021)]

# Create an instance of the dataset
dataset = JoblibDataset(file_list)

# Check the dataset in batches
batch_size = 2
dataset.check_dataset(batch_size)
