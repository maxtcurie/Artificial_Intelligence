import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Define a sparse matrix fully connected network
class SparseFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sparsity=0.8):
        super(SparseFCNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sparsity = sparsity
        
        # Create dense weight matrices for two layers and initialize the sparsity mask
        self.fc1_weight = nn.Parameter(torch.rand(input_size, hidden_size))
        self.fc2_weight = nn.Parameter(torch.rand(hidden_size, output_size))
        self.fc1_mask = self.create_sparse_mask(input_size, hidden_size, sparsity)
        self.fc2_mask = self.create_sparse_mask(hidden_size, output_size, sparsity)
    
    def create_sparse_mask(self, rows, cols, sparsity):
        """Create a sparse mask with a given sparsity."""
        mask = torch.rand(rows, cols) > sparsity  # Mask with the specified sparsity
        return mask.float()  # Convert to float tensor for multiplication

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, self.input_size)
        
        # Apply the sparse mask and perform dense matrix multiplication
        fc1_weight_sparse = self.fc1_weight * self.fc1_mask
        x = F.relu(torch.mm(x, fc1_weight_sparse))
        
        fc2_weight_sparse = self.fc2_weight * self.fc2_mask
        x = torch.mm(x, fc2_weight_sparse)
        
        return x

# CIFAR-10 Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the sparse model, loss function, and optimizer
input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 RGB
hidden_size = 512
output_size = 10  # 10 classes
sparsity = 0.8  # 80% of connections are removed

model = SparseFCNN(input_size, hidden_size, output_size, sparsity=sparsity)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, trainloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(trainloader)}')

# Train the sparse model
train_model(model, trainloader, criterion, optimizer)

# Save the entire model (including structure and weights)
torch.save(model, 'sparse_fcnn_model.pth')
print("Model saved successfully.")

# Load the model (you don't need to define the model class again)
loaded_model = torch.load('sparse_fcnn_model.pth')
print("Model loaded successfully.")

# Set the loaded model to evaluation mode
loaded_model.eval()

# Test the loaded model on the first batch of the test set
inputs, labels = next(iter(testloader))
outputs = loaded_model(inputs)
_, predicted = torch.max(outputs, 1)
print(f"Predicted: {predicted[:10]}")
print(f'labels:    {labels[:10]}')
