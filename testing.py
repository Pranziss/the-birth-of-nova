import numpy as np

# in/output 2 neurons
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2)
bias_hidden = np.random.rand(1, 2)

weights_hidden_output = np.random.rand(2, 1)
bias_output = np.random.rand(1, 1)

# Training loop
for epoch in range(10000):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * 0.1
    bias_output += np.sum(d_output, axis=0, keepdims=True) * 0.1

    weights_input_hidden += X.T.dot(d_hidden) * 0.1
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

    # Print loss every 1000 epochs (take notes)
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final
print("\nFinal predictions:")
print(final_output.round())