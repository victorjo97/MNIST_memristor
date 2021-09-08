""" The fully connected feedforward network to recognise the MNIST digits is handled via the class Network() which
takes as input the list of the numbers of neurons of each layer starting with the input layer and ending in the output.
For example, a shallow network (1 hidden layer) with 30 hidden neurons and images of 28X28 pixels would be initialised
doing:
network = Network((784, 30, 10))
The training is achieved through the stochastic gradient descent scheme implemented as a method of the class Network():
network.stochastic_gradient_descent(train_images, train_labels, validation_images, validation_labels, test_images,
                                    test_labels, epochs, batch_size, learning_rate, patience)
An example is:
network.stochastic_gradient_descent(train_X, train_y, validation_X, validation_y, test_X, test_y, 30, 10, 0.1, 10)
 """

import numpy as np
import random
from keras.datasets import mnist

# Readout of images and labels
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = np.divide(train_X, 255).reshape(-1, train_X[0].size)  # normalisation of the input intensities to better
# fit the operational range of the neurons
validation_X = train_X[50000:]
validation_y = train_y[50000:]
train_X = train_X[:50000]
train_y = train_y[:50000]
test_X = np.divide(test_X, 255).reshape(-1, test_X[0].size)


# Miscellaneous functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(sigmoid_result):  # it is based on the result of the sigmoid to save computations
    return sigmoid_result * (1 - sigmoid_result)


# Core of the code
class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(size) for size in sizes[1:]]  # normal initialisation following N(0,1)
        self.weights = [np.random.randn(size_2, size_1) for size_2, size_1 in zip(sizes[1:], sizes[:-1])]

    def inference(self, inference_images):  # gives the activations and affine mappings of all layers
        affine_mappings = [np.zeros(size) for size in self.sizes[1:]]
        activations = [inference_images.T] + affine_mappings
        for k in range(self.num_layers - 1):
            affine_mappings[k] = self.weights[k].dot(activations[k]) + np.array(
                [self.biases[k], ] * len(inference_images)).T
            activations[k + 1] = sigmoid(affine_mappings[k])
        return activations, affine_mappings

    def accuracy(self, accuracy_images, accuracy_labels):  # computes the accuracy in the 0-1 range on a given set
        inferred = np.argmax(self.inference(accuracy_images)[0][-1], 0)
        where_no_equal = np.count_nonzero(inferred - accuracy_labels)
        correct = 1 - where_no_equal / len(accuracy_labels)
        return correct

    def error(self, error_images, error_labels):
        """
        It computes the error value. This is not used, but I coded it to check that the backpropagation is OK by
        computing random partial derivatives using Taylor approximations.
        """
        num_images = len(error_labels)
        inferred = self.inference(error_images)[0][-1]
        expected = np.zeros((10, num_images))
        expected[error_labels, [i for i in range(num_images)]] = 1
        error_norm_2 = ((inferred - expected) ** 2).sum(0).sum() / (2 * num_images)
        return error_norm_2

    def backpropagation(self, bp_images, bp_labels):  # based on the theory provided in
        # http://neuralnetworksanddeeplearning.com/chap2.html

        num_images = len(bp_labels)
        expected = np.zeros((10, num_images))
        expected[bp_labels, [i for i in range(num_images)]] = 1

        activations, affine_mappings = self.inference(bp_images)
        derivatives = [sigmoid_der(activation) for activation in activations[1:]]

        derivative_error_wrt_output = activations[-1] - expected
        deltas = [derivative_error_wrt_output * derivatives[-1]]  # refer to the link above to understand the notation
        for i in range(self.num_layers - 2):
            deltas = [(self.weights[-1 - i].T.dot(deltas[-1 - i])) * derivatives[-2 - i]] + deltas

        weights_gradients = [delta.dot(activation.T) / num_images for delta, activation in
                             zip(deltas, activations[:-1])]
        biases_gradients = [delta.sum(1) / num_images for delta in deltas]

        return weights_gradients, biases_gradients

    def stochastic_gradient_descent(self, train_images, train_labels, validation_images, validation_labels, test_images,
                                    test_labels, epochs, batch_size, learning_rate, patience):
        idx = [i for i in range(len(train_labels))]
        train_size = len(train_labels)

        best_weights = self.weights
        best_biases = self.biases

        train_accuracies = np.zeros(epochs)
        validation_accuracies = np.zeros(epochs)
        print('Initial train accuracy: {} || Initial validation accuracy: {}'.format(
            self.accuracy(train_images, train_labels),
            self.accuracy(validation_images, validation_labels)))

        for j in range(epochs):
            random.shuffle(idx)

            if train_size % batch_size != 0:
                print('The batch_size must divide ' + str(train_size))
                break

            for k in range(int(train_size / batch_size)):
                weights_gradients, biases_gradients = self.backpropagation(
                    train_images[idx[batch_size * k:batch_size * (k + 1)]],
                    train_labels[idx[batch_size * k:batch_size * (k + 1)]])
                self.weights = [weight - learning_rate * weight_gradient for weight, weight_gradient in
                                zip(self.weights, weights_gradients)]
                self.biases = [bias - learning_rate * bias_gradient for bias, bias_gradient in
                               zip(self.biases, biases_gradients)]

            train_accuracies[j] = self.accuracy(train_images, train_labels)
            validation_accuracies[j] = self.accuracy(validation_images, validation_labels)
            if j > 0 and validation_accuracies[j] > max(validation_accuracies[:j]):
                best_biases = self.biases
                best_weights = self.weights
            if j >= patience and validation_accuracies[j] < np.mean(validation_accuracies[j - patience:j]):
                break
            print('Epoch {epoch_number} of {epoch_total} || Train accuracy: {train_accuracy} || Validation '
                  'accuracy: {validation_accuracy}'.format(epoch_number=j + 1, epoch_total=epochs,
                                                           train_accuracy=train_accuracies[j],
                                                           validation_accuracy=validation_accuracies[j]))
        self.biases = best_biases
        self.weights = best_weights
        print('Test accuracy: ' + str(self.accuracy(test_images, test_labels)))

