import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, hidden_size=2, activation="sigmoid", live_plot=False):
        np.random.seed(42)
        self.hidden_size = hidden_size
        self.activation = activation
        self.live_plot = live_plot

        self.weights_input_hidden = np.random.rand(2, hidden_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, 1)
        self.bias_output = np.random.rand(1, 1)
        self.loss_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.square(x)

    def activate(self, x):
        return self.tanh(x) if self.activation == "tanh" else self.sigmoid(x)

    def activate_derivative(self, x):
        return self.tanh_derivative(x) if self.activation == "tanh" else self.sigmoid_derivative(x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activate(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.activate(self.final_input)
        return self.final_output

    def backward(self, X, y, output, lr=0.1):
        error = y - output
        d_output = error * self.activate_derivative(output)

        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self.activate_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * lr
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * lr

        self.weights_input_hidden += X.T.dot(d_hidden) * lr
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr

        loss = np.mean(np.square(error))
        self.loss_history.append(loss)

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
            output = self.forward(X)
            self.backward(X, y, output)

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

    nn = NeuralNetwork(hidden_size=4, activation="tanh", live_plot=False)
    nn.train(X, y)
    print("\nFinal predictions:")
    print(nn.predict(X))

    nn.plot_loss()