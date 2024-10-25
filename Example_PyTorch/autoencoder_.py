import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannel1DCNN(nn.Module):
    def __init__(self, input_channels_dict, num_classes, cnn_output_size):
        super(MultiChannel1DCNN, self).__init__()
        
        # Create a CNN for each key in the dictionary
        self.cnns = nn.ModuleDict()
        for key, in_channels in input_channels_dict.items():
            self.cnns[key] = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, cnn_output_size, kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        # Flatten and concatenate the outputs of the CNNs
        total_cnn_output_size = len(input_channels_dict) * cnn_output_size
        
        # Dense network after concatenation
        self.fc = nn.Sequential(
            nn.Linear(total_cnn_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Decoder with CNN layers
        self.decoder = nn.Sequential(
            nn.Conv1d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x_dict):
        # Apply the CNNs to each key's data
        cnn_outputs = []
        for key, cnn in self.cnns.items():
            x = x_dict[key]
            x = cnn(x)
            x = x.view(x.size(0), -1)  # Flatten
            cnn_outputs.append(x)
        
        # Concatenate CNN outputs
        x = torch.cat(cnn_outputs, dim=1)
        
        # Pass through the dense layers
        x = self.fc(x)
        
        # Decode the output using the decoder CNN
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.decoder(x)
        
        return x

# Example usage:
input_channels_dict = {
    'key1': 10,  # 1 input channel for the time series under 'key1'
    'key2': 10,  # 1 input channel for the time series under 'key2'
    # Add more keys as necessary
}

bottle_neck = 10  # Example number of classes or output features
cnn_output_size = 128  # The number of output channels after the CNN layers

model = MultiChannel1DCNN(input_channels_dict, num_classes, cnn_output_size)

# Example input: a dictionary of tensors with shape (batch_size, channels, sequence_length)
input_data = {
    key: torch.randn(32, input_channels_dict[key], 100)
    for key in input_channels_dict
}

output = model(input_data)
print(output.shape)  # Should be (batch_size, 1, sequence_length) after decoding
