import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_layers=[4, 4], output_size=1, activation="tanh", lr=0.1, live_plot=False):
        np.random.seed(42)
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.live_plot = live_plot
        self.lr = lr
        self.loss_history = []

        # Layer sizes: input + hidden + output
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1 for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def activate(self, x):
        if self.activation_name == "tanh":
            return np.tanh(x)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == "relu":
            return np.maximum(0, x)
        elif self.activation_name == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)

    def activate_derivative(self, x):
        if self.activation_name == "tanh":
            return 1 - np.square(np.tanh(x))
        elif self.activation_name == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation_name == "relu":
            return (x > 0).astype(float)
        elif self.activation_name == "leaky_relu":
            return np.where(x > 0, 1, 0.01)

    def forward(self, X):
        self.zs = []
        self.activations = [X]
        for W, b in zip(self.weights, self.biases):
            z = self.activations[-1] @ W + b
            self.zs.append(z)
            a = self.activate(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        m = y.shape[0]
        output = self.activations[-1]
        error = y - output
        loss = np.mean(np.square(error))
        self.loss_history.append(loss)

        delta = error * self.activate_derivative(self.zs[-1])
        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            self.weights[i] += self.lr * (a_prev.T @ delta) / m
            self.biases[i] += self.lr * np.mean(delta, axis=0, keepdims=True)
            if i != 0:
                delta = (delta @ self.weights[i].T) * self.activate_derivative(self.zs[i-1])

    def train(self, X, y, epochs=10000):
        if self.live_plot:
            plt.ion()
            fig, ax = plt.subplots()
            line, = ax.plot([], [])
            ax.set_xlim(0, epochs)
            ax.set_ylim(0, 1)
            ax.set_title("Live Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss_history[-1]:.4f}")

            if self.live_plot and epoch % 10 == 0:
                line.set_xdata(np.arange(len(self.loss_history)))
                line.set_ydata(self.loss_history)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

        if self.live_plot:
            plt.ioff()

    def predict(self, X):
        return self.forward(X).round()

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

# Run the model
if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(hidden_layers=[4, 4], activation="relu", live_plot=False)
    nn.train(X, y)
    print("\nFinal predictions:")
    print(nn.predict(X))

    nn.plot_loss()