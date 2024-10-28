import torch
from torch.utils.data import IterableDataset, DataLoader
from joblib import load
import os
import glob
import numpy as np
from itertools import islice
import psutil
import time
from tqdm import tqdm

class IterableJoblibDataset_dict(IterableDataset):
    def __init__(self, file_list, input_keys, output_keys, seq_len, batch_size, stride=2):
        self.file_list = file_list
        self.input_keys = input_keys
        self.seq_len=seq_len
        self.stride=stride
        self.output_keys = output_keys
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup()
        
    def setup(self):
        print("Setting up IterableJoblibDataset")
        self.file_sizes = []
        self.total_samples = 0
        self.joblib_files = [load(f, mmap_mode='r') for f in self.file_list]
        for data in self.joblib_files:
            file_size = len(data[self.input_keys[0]])
            self.file_sizes.append(file_size)
            self.total_samples += file_size
        print(f"Total number of samples across all files: {self.total_samples}")
        
    def __len__(self):
        return self.total_samples
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            file_iter = enumerate(self.joblib_files)
        else:  # in a worker process
            per_worker = int(np.ceil(len(self.file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_iter = enumerate(self.joblib_files[worker_id * per_worker:(worker_id + 1) * per_worker])
        
        for file_idx, data in file_iter:
            input_data={}
            for i in self.input_keys:
                data_tmp=data[i].values
                data_seq_tmp=[data_tmp[j:j+self.seq_len] for j in range(0,len(data_tmp)-self.seq_len,self.stride)]
                input_data[i] = np.array(data_seq_tmp)

            target_data = data[self.output_keys].values
            
            for i in range(0, len(input_data[self.input_keys[0]]), self.batch_size):
                batch_input = {key: torch.from_numpy(input_data[key][i:i+self.batch_size]).float().to(self.device) for key in self.input_keys}
                batch_target = torch.from_numpy(target_data[i:i+self.batch_size]).float().to(self.device)
                yield batch_input, batch_target

def iter_loader_dict(file_list, input_keys, output_keys, seq_len, batch_size=1000, stride=2):
    print("Setting up data loading")
    
    # Create dataset and data loader
    dataset = IterableJoblibDataset_dict(file_list, input_keys, output_keys, seq_len, batch_size, stride)
    data_loader = DataLoader(dataset, batch_size=None)

    # Determine input_dim and output_dim from a sample batch
    for input_data, target_data in data_loader:
        input_dims={key:input_data[key].shape[2] for key in input_data}
        output_dims=target_data.shape[1]
        
        key_tmp=list(input_data.keys())[0]
        seq_len=input_data[key_tmp].shape[1]

        print(f"Sample input shape: {input_data[key_tmp].shape}")
        print(f"Sample target shape: {target_data.shape}")
        break
    
    return dataset, data_loader, input_dims, output_dims, seq_len

class IterableJoblibDataset(IterableDataset):
    def __init__(self, file_list, batch_size):
        self.file_list = file_list
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup()
        
    def setup(self):
        print("Setting up IterableJoblibDataset")
        self.file_sizes = []
        self.total_samples = 0
        self.joblib_files = [load(f, mmap_mode='r') for f in self.file_list]
        for data in self.joblib_files:
            file_size = len(data['input_data'])
            self.file_sizes.append(file_size)
            self.total_samples += file_size
        print(f"Total number of samples across all files: {self.total_samples}")
        
    def __len__(self):
        return self.total_samples
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            file_iter = enumerate(self.joblib_files)
        else:  # in a worker process
            per_worker = int(np.ceil(len(self.file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_iter = enumerate(self.joblib_files[worker_id * per_worker:(worker_id + 1) * per_worker])
        
        for file_idx, data in file_iter:
            input_data = data['input_data']
            target_data = data['target_data']
            
            for i in range(0, len(input_data), self.batch_size):
                batch_input = torch.from_numpy(input_data[i:i+self.batch_size]).float().to(self.device)
                batch_target = torch.from_numpy(target_data[i:i+self.batch_size]).float().to(self.device)
                yield batch_input, batch_target

def iter_loader(file_list, batch_size=1000):
    print("Setting up data loading")
    
    # Create dataset and data loader
    dataset = IterableJoblibDataset(file_list, batch_size)
    data_loader = DataLoader(dataset, batch_size=None)

    # Determine input_dim and output_dim from a sample batch
    for batch_input, batch_target in data_loader:
        input_dim = batch_input[0].shape[-1]
        output_dim = batch_target[0].shape[-1]
        print(f"Sample input shape: {batch_input[0].shape}")
        print(f"Sample target shape: {batch_target[0].shape}")
        break
    
    return dataset, data_loader, input_dim, output_dim




# Usage example
if __name__ == "__main__":

    # Set up file list
    file_suffix = 'min_set_training_np'
    directory_path = '/scratch/gpfs/EKOLEMEN/big_d3d_data/diag2diag/CAKE/'
    
    
    # Use glob to find all files matching the pattern nums_slow.h5
    file_pattern = os.path.join(directory_path, f'1*_{file_suffix}.joblib')
    matching_files = glob.glob(file_pattern)
    
    discharges=[int(i[len(directory_path):len(directory_path)+6]) for i in matching_files]
    discharges=list(set(discharges))
    discharges.sort()
    
    discharges=discharges[:4] #last 50 for test
    
    
    # Define the file paths
    file_list = [f'{directory_path}/{discharge}_{file_suffix}.joblib' for discharge in discharges]

    dataset = iter_loader(file_list,batch_size=5000)

    data_loader = DataLoader(dataset, batch_size=None)
    
    print(f"Setup complete. Total samples: {dataset.total_samples}")
    
    # Example: Iterate through a few batches
    for i, (batch_input, batch_target) in tqdm(enumerate(data_loader)):
        print(f"Batch {i + 1}: Input shape: {batch_input.shape}, Target shape: {batch_target.shape}")