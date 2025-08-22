import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_layers=[6, 6], output_size=3,
                 activation="relu", output_activation="softmax", lr=0.1,
                 live_plot=False, batch_size=None):
        np.random.seed(42)
        self.activation_name = activation
        self.output_activation = output_activation or activation
        self.live_plot = live_plot
        self.lr = lr
        self.batch_size = batch_size
        self.loss_history = []

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
                        for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1]))
                       for i in range(len(layer_sizes)-1)]

    def activation(self, x, layer_idx):
        func = self.output_activation if layer_idx == len(self.weights)-1 else self.activation_name
        if func == "tanh":
            return np.tanh(x)
        elif func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif func == "relu":
            return np.maximum(0, x)
        elif func == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif func == "softmax":
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def activation_derivative(self, x, layer_idx):
        func = self.output_activation if layer_idx == len(self.weights)-1 else self.activation_name
        if func == "tanh":
            return 1 - np.square(np.tanh(x))
        elif func == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif func == "relu":
            return (x > 0).astype(float)
        elif func == "leaky_relu":
            return np.where(x > 0, 1, 0.01)
        elif func == "softmax":
            return np.ones_like(x)

    def forward(self, X):
        self.zs = []
        self.activations = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.zs.append(z)
            a = self.activation(z, i)
            self.activations.append(a)
        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        if self.output_activation == "softmax":
            return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]
        else:
            return np.mean(np.square(y_true - y_pred))

    def backward(self, y):
        m = y.shape[0]
        output = self.activations[-1]
        error = output - y
        loss = self.compute_loss(y, output)
        self.loss_history.append(loss)

        delta = error * self.activation_derivative(self.zs[-1], len(self.weights)-1)
        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            self.weights[i] -= self.lr * (a_prev.T @ delta) / m
            self.biases[i] -= self.lr * np.mean(delta, axis=0, keepdims=True)
            if i != 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.zs[i-1], i-1)

    def get_batches(self, X, y):
        if self.batch_size is None:
            yield X, y
        else:
            for i in range(0, X.shape[0], self.batch_size):
                yield X[i:i+self.batch_size], y[i:i+self.batch_size]

    def train(self, X, y, epochs=5000):
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
            for X_batch, y_batch in self.get_batches(X, y):
                self.forward(X_batch)
                self.backward(y_batch)

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
        output = self.forward(X)
        if self.output_activation == "softmax":
            return np.argmax(output, axis=1)
        return (output > 0.5).astype(int)

    def evaluate(self, X, y):
        preds = self.predict(X).flatten()
        y_true = np.argmax(y, axis=1) if self.output_activation == "softmax" else y.flatten()
        acc = np.mean(preds == y_true)
        print(f"Accuracy: {acc * 100:.2f}%")
        return acc

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

# 🧪 3-class toy dataset test
if __name__ == "__main__":
    X = np.array([
        [0, 0], [0, 1], [1, 0],  # Class 0
        [1, 1], [2, 1], [1, 2],  # Class 1
        [2, 2], [3, 2], [2, 3]   # Class 2
    ])
    y_raw = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y = np.eye(3)[y_raw]  # One-hot encode

    nn = NeuralNetwork(input_size=2, hidden_layers=[6, 6], output_size=3,
                       activation="relu", output_activation="softmax", lr=0.1)

    nn.train(X, y, epochs=5000)
    print("\nFinal predictions:")
    print(nn.predict(X))
    nn.evaluate(X, y)
    nn.plot_loss()