import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TimeSliceEncoder(nn.Module):
    """Encodes each time slice independently from (n_feature) to (n_latent) using specified encoder type."""
    def __init__(self, input_dim, n_latent, encoder_type="MLP"):
        super(TimeSliceEncoder, self).__init__()
        if encoder_type == "MLP":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, n_latent),
                nn.ReLU(),
                nn.Linear(n_latent, n_latent)
            )
        elif encoder_type == "CNN":
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=n_latent, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=n_latent, out_channels=n_latent, kernel_size=1)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, x):
        batch_size, sequence_len, input_dim = x.size()
        if isinstance(self.encoder[0], nn.Conv1d):
            x = x.transpose(1, 2)  # Shape: (n_sample, n_feature, sequence_len)
            x = self.encoder(x)    # Shape: (n_sample, n_latent, sequence_len)
            x = x.transpose(1, 2)  # Shape: (n_sample, sequence_len, n_latent)
        else:
            x = x.view(batch_size * sequence_len, input_dim)
            x = self.encoder(x)
            x = x.view(batch_size, sequence_len, -1)  # Reshape back
        return x

class SequenceTransformer(nn.Module):
    """Processes a sequence of encoded time slices with a transformer encoder."""
    def __init__(self, n_latent, seq_len, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
        super(SequenceTransformer, self).__init__()
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(seq_len, n_latent), requires_grad=False)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_latent,  
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def _generate_positional_encoding(self, seq_len, n_latent):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_latent, 2) * (-torch.log(torch.tensor(10000.0)) / n_latent))
        pos_encoding = torch.zeros(seq_len, n_latent)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, x):
        x += self.positional_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        return x.transpose(0, 1)

class ResidualBlock(nn.Module):
    """Residual block to process and refine the transformed output."""
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual
        return self.layer_norm(x)

class TransformerWithDictInput(nn.Module):
    """Combines multiple time slice encoders, sequence transformer, and a classifier with residual block."""
    def __init__(self, input_dims, output_dim, seq_len, encoder_types, n_latent=128, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerWithDictInput, self).__init__()

        # Time slice encoders for each input key
        self.slice_encoders = nn.ModuleDict({
            key: TimeSliceEncoder(input_dim, n_latent, encoder_type=encoder_types[key])
            for key, input_dim in input_dims.items()
        })

        # Transformer encoder for the sequence of concatenated latent representations
        self.sequence_transformer = SequenceTransformer(n_latent * len(input_dims), seq_len, nhead, num_encoder_layers, dim_feedforward, dropout)
        
        # Residual block for the final classification layer
        self.residual_block = ResidualBlock(n_latent * len(input_dims))

        # Final classifier
        self.classifier = nn.Linear(n_latent * len(input_dims), output_dim)

    def forward(self, x_dict):
        # Process each input key independently with its respective slice encoder
        encoded_inputs = []
        for key, x in x_dict.items():
            encoded_x = self.slice_encoders[key](x)
            encoded_inputs.append(encoded_x)
        
        # Concatenate all encoded inputs along the feature dimension
        x = torch.cat(encoded_inputs, dim=2)  # Shape: (n_sample, sequence_len, n_latent * num_inputs)

        # Pass through the transformer to process the sequence
        x = self.sequence_transformer(x)

        # Use the representation of the last time step for the classifier
        last_time_step = x[:, -1, :]  # Shape: (n_sample, n_latent * num_inputs)
        
        # Refine using residual block
        x = self.residual_block(last_time_step)

        # Final classification output
        return self.classifier(x)  # Shape: (n_sample, output_dim)


# Helper function to determine input and output dimensions
def get_data_dim(input_data):
    input_dims = {key: input_data[key].shape[2] for key in input_data}
    seq_len = input_data[list(input_data.keys())[0]].shape[1]
    return input_dims, seq_len



def generate_nonlinear_data(batch_size, seq_len, output_dim, input_dim=[16, 12]):
    # Initialize tensors for the two sensors based on nonlinear functions
    sensor_1 = torch.zeros(batch_size, seq_len, input_dim[0])
    sensor_2 = torch.zeros(batch_size, seq_len, input_dim[1])
    
    # Generate target labels for the batch
    target_labels = torch.randint(0, output_dim, (batch_size,))
    
    # Generate time variable for sine/cosine functions
    time = torch.linspace(0, 2 * torch.pi, seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    # Nonlinear patterns for each class
    for i in range(output_dim):
        # Select indices for the current class
        indices = (target_labels == i).nonzero(as_tuple=True)[0]

        # Generate the specific pattern for each class
        if len(indices) > 0:  # Only process if there are samples for the class
            # Sensor 1: Sine function with a phase shift and small noise
            sensor_1[indices, :, :input_dim[0]//2] = torch.sin(time[indices] + i).unsqueeze(-1).expand(-1, -1, input_dim[0]//2) \
                                                     + 0.1 * torch.randn(len(indices), seq_len, input_dim[0]//2)
            
            # Sensor 1: Exponential decay with noise
            sensor_1[indices, :, input_dim[0]//2:] = (torch.exp(-0.1 * time[indices]) * (i + 1)).unsqueeze(-1).expand(-1, -1, input_dim[0]//2) \
                                                     + 0.1 * torch.randn(len(indices), seq_len, input_dim[0]//2)
            
            # Sensor 2: Cosine function with frequency shift and noise
            sensor_2[indices, :, :input_dim[1]//2] = torch.cos(time[indices] * (i + 1)).unsqueeze(-1).expand(-1, -1, input_dim[1]//2) \
                                                     + 0.1 * torch.randn(len(indices), seq_len, input_dim[1]//2)
            
            # Sensor 2: Quadratic trend with noise
            sensor_2[indices, :, input_dim[1]//2:] = (time[indices]**2 * 0.1 * (i + 1)).unsqueeze(-1).expand(-1, -1, input_dim[1]//2) \
                                                     + 0.1 * torch.randn(len(indices), seq_len, input_dim[1]//2)
    
    # Combine into dictionary for input data
    input_data = {
        "sensor_1": sensor_1,
        "sensor_2": sensor_2
    }
    return input_data, target_labels

if __name__ == "__main__":
    # Mock data for training
    batch_size = 16
    seq_len = 20
    output_dim = 2  # Number of classes for classification

    input_data, target_labels = generate_nonlinear_data(batch_size, seq_len, output_dim)

    input_dims, seq_len = get_data_dim(input_data)


    encoder_types = {
        "sensor_1": "CNN", 
        "sensor_2": "MLP"
    }

    # Instantiate the model
    model = TransformerWithDictInput(input_dims, output_dim, seq_len, encoder_types, n_latent=128)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_data)
        
        # Compute loss
        loss = criterion(outputs, target_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item()}")


    # Inference (mock example)
    model.eval()
    with torch.no_grad():
        test_input_data = {
            "sensor_1": torch.randn(batch_size, seq_len, 16),
            "sensor_2": torch.randn(batch_size, seq_len, 12),
        }
        predictions = model(test_input_data)
        print(f"Predictions shape: {predictions.shape}")
