import numpy as np

# Define the RNN class
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize the weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden weights
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden weights
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output weights
        
        # Initialize the bias vectors
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias
    
    def forward(self, inputs):
        self.h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        
        self.hidden_states = []
        self.outputs = []
        
        # Perform forward propagation for each time step
        for x in inputs:
            x = np.reshape(x, (self.input_size, 1))  # Reshape input into column vector
            
            # Compute hidden state
            self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
            
            # Compute output
            y = np.dot(self.Why, self.h) + self.by
            
            # Store hidden state and output
            self.hidden_states.append(self.h)
            self.outputs.append(y)
        
        return self.outputs, self.hidden_states
    
    def backward(self, inputs, targets, learning_rate=0.1):
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(self.h)
        
        # Perform backward propagation for each time step in reverse order
        for t in reversed(range(len(inputs))):
            x = np.reshape(inputs[t], (self.input_size, 1))  # Reshape input into column vector
            dy = self.outputs[t] - np.reshape(targets[t], (self.output_size, 1))  # Compute output error
            
            # Compute gradients for output layer
            dWhy += np.dot(dy, self.hidden_states[t].T)
            dby += dy
            
            # Compute gradients for hidden layer
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - self.hidden_states[t] ** 2) * dh
            dbh += dh_raw
            
            # Compute gradients for input layer
            dWxh += np.dot(dh_raw, x.T)
            dWhh += np.dot(dh_raw, self.hidden_states[t-1].T)
            
            # Update dh_next for the next time step
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        # Clip gradients to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # Update weights and biases using gradients and learning rate
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

# Example usage
input_size = 3
hidden_size = 4
output_size = 2

# Create the RNN object
rnn = RNN(input_size, hidden_size, output_size)

# Define the input sequence and target sequence
inputs = [np.array([1, 0, 1]), np.array([0, 1, 0]), np.array([1, 1, 0])]
targets = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]

# Training parameters
epochs = 1000
learning_rate = 0.1

# Perform multiple iterations (epochs) of training
for epoch in range(epochs):
    # Perform forward and backward propagation
    outputs, hidden_states = rnn.forward(inputs)
    rnn.backward(inputs, targets, learning_rate)
    
    # Print the loss for every 100 epochs
    if epoch % 100 == 0:
        loss = np.mean([(outputs[i] - targets[i]) ** 2 for i in range(len(outputs))])
        print(f"Epoch: {epoch}, Loss: {loss}")

# Print the final outputs and hidden states
print("\nFinal Outputs:")
for output in outputs:
    print(output)
    
print("\nFinal Hidden States:")
for state in hidden_states:
    print(state)
