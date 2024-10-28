import torch
import torch.nn as nn

# Creating mock data
input_size = 5
hidden_size = 10
num_classes = 2
num_samples = 100
num_epochs = 5
learning_rate = 0.001

class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size).to('cuda:0')  # First half of the model on GPU 0
        self.layer2 = nn.Linear(hidden_size, num_classes).to('cuda:1')  # Second half of the model on GPU 1

    def forward(self, x):
        x = self.layer1(x.to('cuda:0')) 
        x = self.layer2(x.to('cuda:1')) 
        return x

# Instantiate the model
model = LargeModel()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Dataset & DataLoader
x = torch.randn(num_samples, input_size)
y = torch.randint(0, num_classes, (num_samples,))
dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Training Loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs.cpu(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
