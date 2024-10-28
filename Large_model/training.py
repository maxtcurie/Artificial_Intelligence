import torch
from torch import nn, optim
import glob
import os
import random
import numpy as np 
import matplotlib.pyplot as plt
from joblib import load
import pickle
from tqdm import tqdm

from torch_models import TransformerWithDictInput as imported_model
from data_loader import iter_loader_dict as imported_loader
from loss_function import MaskedBCELoss

#*************Start of user block*********************

num_epochs = 500
seq_len = 5
batch_size = 256
num_shot_for_test=400
stride = 500

train_portion=0.8
early_stopping_patience = 20

actu=['/ech', '/gas', '/ip','/mag_pcb_coil', '/p_inj', '/t_inj']
diag=['/bes_slow','/co2_density_slow', '/ece_slow', '/mse']

#the input key for data training
input_keys=actu+diag

#the the key of the label for data training
output_keys='ELM'

#setup for the model 

n_latent=256
nhead=8
num_encoder_layers=6
dim_feedforward=512
dropout=0.1

MLP_list=actu
CNN_list=diag

encoder_types={**{i: "MLP" for i in MLP_list}, **{i: "CNN" for i in CNN_list}}

model_save_path=f'/scratch/gpfs/EKOLEMEN/big_d3d_data/diag2diag/models/'

# Set up file list
file_suffix = 'norm_1ms_clipped'
directory_path = '/scratch/gpfs/EKOLEMEN/big_d3d_data/diag2diag/ELM_pred/'
model_suffix='ELM_transformer'


rand_seed=42

#*************End of user block*********************

# Use glob to find all files matching the pattern nums_slow.h5
file_pattern = os.path.join(directory_path, f'1*_{file_suffix}.joblib')
matching_files = glob.glob(file_pattern)

discharges=[int(i[len(directory_path):len(directory_path)+6]) for i in matching_files]
discharges=list(set(discharges))
discharges.sort()

discharges=discharges[:-num_shot_for_test] #last 50 for test

# Define the file paths
file_list = [f'{directory_path}/{discharge}_{file_suffix}.joblib' for discharge in discharges]

# Print the list of matching files
#print("Found files:", discharges)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
random.seed(rand_seed)

# Shuffle the file list
random.shuffle(file_list)

# Split the file list into training and validation sets
train_size = int(train_portion * len(file_list))
train_files = file_list[:train_size]
val_files = file_list[train_size:]

# Create data loaders for training and validation
train_dataset, train_loader, input_dims, output_dims, _ = imported_loader(train_files, input_keys, output_keys, seq_len, batch_size, stride)
val_dataset, val_loader, input_dims, output_dims, _ = imported_loader(val_files, input_keys, output_keys, seq_len, batch_size, stride)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

print(f'train_dataset.device:{train_dataset.device}')
print(f"Input dimension: {input_dims}")
print(f"Output dimension: {output_dims}")
print(f"Input seq_len: {seq_len}")

# Create the model
model = imported_model(input_dims, output_dims, seq_len, encoder_types, n_latent=n_latent, nhead=nhead, 
                 num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
model.to(train_dataset.device)

# Define loss function and optimizer
criterion = MaskedBCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping and checkpointing setup

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
best_loss = float('inf')
patience_counter = 0

# Lists to store epoch losses
train_losses = []
val_losses = []

# Training loop with early stopping and checkpointing
checkpoint_saved = False

print('start training')
for epoch in range(num_epochs):
    # Training phase
    model.train()
    
    running_train_loss = 0.0
    for inputs, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # Validation phase
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")

    # Early stopping and checkpointing
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        patience_counter = 0
        checkpoint_saved = True
        # Save the best model checkpoint

        pickle.dump(model, open(os.path.join(checkpoint_dir, f'{model_save_path}/best_model_{model_suffix}.pkl'), 'wb'))
        print(f"Checkpoint saved at epoch {epoch+1} with val loss {epoch_val_loss}")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Check if the checkpoint was saved and load the best model
if checkpoint_saved:
    model = pickle.load(open(os.path.join(checkpoint_dir, f'{model_save_path}/best_model_{model_suffix}.pkl'), 'rb'))
else:
    print("No checkpoint saved during training.")

# Save the final model
pickle.dump(model, open(os.path.join(checkpoint_dir, f'{model_save_path}/final_model_{model_suffix}.pkl'), 'wb'))
print("Final model saved as final_model.pth")


# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_plot.png')
plt.show()



# Serialize the data and save to a file
with open(f'{model_save_path}/losses_{model_suffix}.pkl', 'wb') as file:
    pickle.dump({'loss':train_losses, 'val_loss':val_losses}, file)
