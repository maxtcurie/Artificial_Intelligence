import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

# Initialize teacher and student models
teacher_model = SimpleNN(784, [128, 64], 10)
student_model = SimpleNN(784, [64, 32], 10)

# Training function for the teacher model
def train_model(model, data_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Knowledge distillation function
def distill_knowledge(teacher, student, data_loader, student_optimizer, criterion, temperature=2.0, alpha=0.7):
    teacher.eval()
    student.train()
    for images, labels in data_loader:
        student_optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = teacher(images)
        student_outputs = student(images)
        soft_labels = F.softmax(teacher_outputs / temperature, dim=1)
        soft_loss = criterion(F.log_softmax(student_outputs / temperature, dim=1), soft_labels)
        hard_loss = criterion(student_outputs, labels)
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        loss.backward()
        student_optimizer.step()

# Setup optimizer and loss function for training
teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the teacher model
train_model(teacher_model, train_loader, criterion, teacher_optimizer)

# Perform knowledge distillation
distill_knowledge(teacher_model, student_model, train_loader, student_optimizer, criterion)

# Evaluate the student model
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

evaluate(student_model, test_loader)
