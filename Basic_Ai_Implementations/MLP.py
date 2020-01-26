import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed(100)

class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """

        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons)
        self.activation = activation
        self.bias = bias if bias is not None else np.random.randn(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """

        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        # In case no activation function was chosen
        if self.activation is None:
            return r

        # tanh
        if self.activation == 'tanh':
            return np.tanh(r)

        # sigmoid
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        return r

    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        return r


class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self._layers:
            X = layer.activate(X)

        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)
        return ff
        """

        if ff.ndim == 1:
            return np.argmax(ff)
        Multiple rows
        return np.argmax(ff, axis=1)
        """
    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.feed_forward(X)

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, X, y, learning_rate, max_epochs):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        mses = []
        msest = []
        accs = []
        spes = []
        sens = []
        for i in range(max_epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)
                mse = np.mean(np.square(y - nn.feed_forward(X)))
                mses.append(mse)
                ypred = nn.predict(X_test)
                mset = np.mean(np.square(y_test - ypred))
                msest.append(mset)
                print('Epoch: #%s, MSE Train: %f' % (i, float(mse)))
                print('Epoch: #%s, MSE Test: %f' % (i, float(mset)))
                cm = confusion_matrix(y_test, np.round(nn.predict(X_test)))
                print(cm)
                acc = ((cm[0][0]+cm[1][1])/np.sum(cm))*100
                sensitivity = (cm[0][0]/(cm[0][0]+cm[0][1]))*100
                specificity = (cm[1][1]/(cm[1][1]+cm[1][0]))*100
                spes.append(specificity)
                sens.append(sensitivity)
                accs.append(acc)
                print('Test Accuracy: %.2f' % (acc))
                print('Sensitivity : %.2f' % (sensitivity))
                print('Specificity : %.2f' % (specificity))
                if acc == 100:
                    break

        return mses, accs, msest

    @staticmethod
    def tst_mse(y_pred, y_true):
        mse = np.mean(np.square(y_true - y_pred))
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
        return mse



if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_layer(Layer(116, 3, 'tanh'))
    nn.add_layer(Layer(3, 3, 'tanh'))
    nn.add_layer(Layer(3, 3, 'tanh'))
    nn.add_layer(Layer(3, 3, 'tanh'))
    nn.add_layer(Layer(3, 1, 'sigmoid'))

    # loading dataset
    data = pd.read_csv('mushroom_csv.csv')
    le = LabelEncoder()
    # encoding the class column
    data['class'] = le.fit_transform(data['class'])
    # removing the class column from feature vector
    Y = data['class'].values.reshape(-1, 1)
    data = data.drop('class', 1)
    #print(data.head(5))
    # encoding binary features into binary ones
    encoded_data = pd.get_dummies(data)
    #print(encoded_data.head(5))
    X = np.array(encoded_data.iloc[:, :])
    print(X.shape)
    print(Y.shape)
    # splitting data into 70% and 30% for training and testing
    # stratify key word is so that the same amount of each class is used in training
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True, stratify=Y )

    # Train the neural network
    errors, accs, test_errors = nn.train(X_train, y_train, 0.1, 10)

    # Plot changes in mse
    plt.plot(errors, c = 'b', label = 'Train MSE')
    plt.title('Changes in MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(linestyle='-.', linewidth=0.5)
    plt.show()
    plt.plot(test_errors, c = 'r', label = 'Test MSE')
    plt.legend()
    plt.figure()
    plt.plot(accs, c='y', label = 'Test Acc')
    plt.show()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()