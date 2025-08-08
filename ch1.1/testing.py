import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        np.random.seed(42)
        self.weights_input_hidden = np.random.rand(2, 2)
        self.bias_hidden = np.random.rand(1, 2)
        self.weights_hidden_output = np.random.rand(2, 1)
        self.bias_output = np.random.rand(1, 1)
        self.loss_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output, lr=0.1):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * lr
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * lr

        self.weights_input_hidden += X.T.dot(d_hidden) * lr
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr

        loss = np.mean(np.square(error))
        self.loss_history.append(loss)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss_history[-1]:.4f}")

    def predict(self, X):
        return self.forward(X).round()

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

# 🚀 Run the model
if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork()
    nn.train(X, y)
    print("\nFinal predictions:")
    print(nn.predict(X))

    nn.plot_loss()