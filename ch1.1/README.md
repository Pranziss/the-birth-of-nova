In this project, we built a feedforward neural network from scratch using NumPy. The architecture is designed to solve the classic XOR problem, which requires nonlinear decision boundaries.
🔧 Architecture Overview:
- Input Layer: 2 neurons (for the two binary inputs of XOR)
- Hidden Layers: Two layers, each with 4 neurons
- Output Layer: 1 neuron (produces a binary prediction)
⚙️ Key Features:
- Activation Function: Configurable (sigmoid, tanh, ReLU, leaky_relu)
- Weight Initialization: Randomized with small values for stable learning
- Biases: Added to each layer to shift activation thresholds
- Modular Design: Easily extendable to deeper networks or other datasets
This setup allows the network to learn complex, nonlinear patterns — something a single-layer perceptron cannot do. By stacking multiple layers and applying activation functions, the model can approximate the XOR logic gate with high accuracy.
