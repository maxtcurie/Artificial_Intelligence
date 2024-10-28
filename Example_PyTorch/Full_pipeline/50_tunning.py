import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from ax.service.managed_loop import optimize

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# List to store trial data
trials_data = []

# Define the training and evaluation function
def train_evaluate(parameterization):
    # Hyperparameters
    hidden_size = parameterization.get("hidden_size", 128)
    learning_rate = parameterization.get("learning_rate", 0.001)
    batch_size = parameterization.get("batch_size", 64)
    
    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(input_size=784, hidden_size=hidden_size, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    correct = 0
    total = 0
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    
    # Store trial data
    trials_data.append({
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "accuracy": accuracy
    })
    
    return accuracy

# Perform Bayesian optimization
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "hidden_size", "type": "range", "bounds": [64, 256]},
        {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "batch_size", "type": "range", "bounds": [32, 128]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
    total_trials=20,
)

# Save trials data to a CSV file
trials_df = pd.DataFrame(trials_data)
trials_df.to_csv("bayesian_optimization_trials.csv", index=False)

print("Best Parameters: ", best_parameters)
print("Best Accuracy: ", values)
print("All trials saved to 'bayesian_optimization_trials.csv'")
