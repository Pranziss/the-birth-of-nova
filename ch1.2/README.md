Forward propagation is the process of passing input data through the network to generate predictions. Each layer transforms the data using weights, biases, and activation functions.
🔄 How It Works:
- Input: The XOR input vector (e.g., [0, 1])
- Layer Computation:
- For each layer:
- Compute: z = weights · input + bias
- Apply activation: a = activation(z)
- Output: The final activation from the last layer is the network’s prediction (e.g., 0.92 for XOR output 1)
⚙️ Features:
- Supports multiple activation functions (sigmoid, tanh, ReLU, etc.)
- Handles arbitrary depth and neuron count
- Outputs intermediate activations for debugging and visualization
This step is crucial for evaluating how well the network is learning. The output is compared against the expected result during backpropagation to adjust weights and improve accuracy.
