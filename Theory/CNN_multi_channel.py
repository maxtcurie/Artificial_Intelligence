import numpy as np

# Define the multi-channel CNN class
class MultiChannelCNN:
    def __init__(self, num_channels, kernel_size, num_filters):
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, num_channels, kernel_size, kernel_size)
        
    def convolve(self, X):
        num_samples, _, input_size, _ = X.shape
        _, num_channels, kernel_size, _ = self.filters.shape
        output_size = input_size - kernel_size + 1
        outputs = np.zeros((num_samples, self.num_filters, output_size, output_size))
        
        for i in range(num_samples):
            for j in range(self.num_filters):
                for k in range(num_channels):
                    for l in range(output_size):
                        for m in range(output_size):
                            outputs[i, j, l, m] += np.sum(X[i, k, l:l+kernel_size, m:m+kernel_size] * self.filters[j, k])
        
        return outputs
    
    def pool(self, X):
        num_samples, _, input_size, _ = X.shape
        pool_size = 2
        output_size = input_size // pool_size
        outputs = np.zeros((num_samples, self.num_filters, output_size, output_size))
        
        for i in range(num_samples):
            for j in range(self.num_filters):
                for k in range(output_size):
                    for l in range(output_size):
                        outputs[i, j, k, l] = np.max(X[i, j, k*pool_size:k*pool_size+pool_size, l*pool_size:l*pool_size+pool_size])
        
        return outputs

# Generate mock training and test data
num_samples_train = 100
num_samples_test = 20
num_channels = 3
input_size = 10
num_classes = 2

X_train = np.random.randn(num_samples_train, num_channels, input_size, input_size)
y_train = np.random.randint(num_classes, size=num_samples_train)
X_test = np.random.randn(num_samples_test, num_channels, input_size, input_size)
y_test = np.random.randint(num_classes, size=num_samples_test)

# Create an instance of the multi-channel CNN
kernel_size = 3
num_filters = 5
cnn = MultiChannelCNN(num_channels, kernel_size, num_filters)

# Training loop
num_epochs = 10
learning_rate = 0.001

for epoch in range(num_epochs):
    # Forward pass
    conv_output = cnn.convolve(X_train)
    pool_output = cnn.pool(conv_output)
    
    # Perform training (e.g., update weights with gradient descent)
    # Your training code goes here
    
    # Print the training progress
    print("Epoch:", epoch+1)
    print("Convolution output shape:", conv_output.shape)
    print("Pooling output shape:", pool_output.shape)

# Evaluation on test data
conv_output_test = cnn.convolve(X_test)
pool_output_test = cnn.pool(conv_output_test)

# Print the test results
print("Test data - Convolution output shape:", conv_output_test.shape)
print("Test data - Pooling output shape:", pool_output_test.shape)
