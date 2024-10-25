class VanillaTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, d_model=128, nhead=8,
                 num_encoder_layers=6, dim_feedforward=512, dropout=0.1,
                 activation="relu"):
        super(VanillaTransformer, self).__init__()

        # Model dimensions
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.seq_len = seq_len

        # Batch normalization
        self.batch_normalization = nn.BatchNorm1d(input_dim)

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(seq_len, d_model),
            requires_grad=False)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)

        # Non-linearity
        self.activation = nn.PReLU()

    def _generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (
                    -torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.batch_normalization(x.transpose(1, 2)).transpose(1, 2)

        # Apply input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.activation(x)

        # Add positional encoding
        x += self.positional_encoding[:, :x.size(1), :]

        # Transpose for transformer (batch_size, seq_len, d_model)
        # -> (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)

        # Take the output from the last time step
        x = x[-1, :, :]  # (batch_size, d_model)

        # Pass through the output layer
        output = self.fc_out(x)  # (batch_size, 1)

        return output