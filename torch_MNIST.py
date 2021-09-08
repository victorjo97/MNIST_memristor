"""MNIST problem using PyTorch. The class NeuralNetwork() defines a NN with 28x28 input neurons, one hidden layer with 50 neurons,
and a classifying output layer with 10 neurons. The hidden layer and output layers are biased and use sigmoid and
softmax activations (with temperature 1) respectively. The model is initialised by doing: model = NeuralNetwork().to(
device)

This can be trained with stochastic gradient descent using the cross-entropy loss function. The stop is performed
following the fine stop algorithm.

To train one model do:

execution(exe_model, epochs, learning_rate, batch_size, target_accuracy, division, weight_decay)
E.g.: execution(model, 30, 0.1, 10, 0.95, 10, 0.0001)

division is the number of batches after which the accuracy of the model is checked in the context of the fine stop
algorithm.

To train several models with different random seeds do:

multiple_execution(max_epochs, learning_rate, batch_size, num_executions, target_accuracy=1, division=1000,
weight_decay=0, init_factor=1)
E.g.: multiple_execution(100, 0.1, 10, 30, 0.95, 100, 0.0001, 2)
"""

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

# Remove randomness
random_seed = 1
torch.manual_seed(random_seed)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download training data from open datasets.
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),  # First, due to ToTensor() the data is in the 0-1
    # range. Then, it is normalized to a Normal(0,1) being the first value the mean and the second, the std
)

train_data, valid_data = random_split(train_data, [50000, 10000])

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
)


# Create data loaders with the desired batch size
def data_loader(batch_size=1):
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data)
    test_data_loader = DataLoader(test_data)
    return train_data_loader, valid_data_loader, test_data_loader


# Defines the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.sigmoid_stack = nn.Sequential(
            nn.Linear(28 * 28, 50, bias=True),
            nn.Sigmoid(),
            nn.Linear(50, 10, bias=True),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.sigmoid_stack(x)
        return logits


loss_fn = nn.CrossEntropyLoss()  # Combines the implementation of the softmax function on the output activations
# and the negative log-likelihood loss.


# To perform one epoch
def train(train_data_loader, train_model, train_loss_fn, learning_rate, target_accuracy, division, weight_decay, fine_stop):
    # division is the number of batches after which the accuracy of the model is checked in the context of the fine
    # stop algorithm. fine_stop is a boolean values that controls the activation (1) or deactivation (0) of the fine
    # stop.
    valid_data_loader = data_loader(10)[1]
    optimizer = torch.optim.SGD(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    size = len(train_data_loader.dataset)
    for batch, (X, y) in enumerate(train_data_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        prediction = train_model(X)
        loss = train_loss_fn(prediction, y)

        # Backpropagation
        optimizer.zero_grad()  # sets the gradients to zero
        loss.backward()  # calculate the gradients doing a backward pass
        optimizer.step()  # update the weights

        if batch % division == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            if fine_stop:
                if accuracy(valid_data_loader, train_model, loss_fn, 'Validation')[1] >= target_accuracy:
                    break


# To compute the accuracy of a model
def accuracy(acc_data_loader, acc_model, acc_loss_fn, acc_type):

    size = len(acc_data_loader.dataset)
    num_batches = len(acc_data_loader)
    acc_model.eval()  # tells the model you are testing it
    accuracy_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in acc_data_loader:
            X, y = X.to(device), y.to(device)
            prediction = acc_model(X)
            accuracy_loss += acc_loss_fn(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    accuracy_loss /= num_batches
    correct /= size
    print(f"{acc_type} Error: \n Accuracy: {(100 * correct):>0.01f}%, Avg loss: {accuracy_loss:>8f} \n")
    return accuracy_loss, correct


# Trains one model with the fine stop algorithm
def execution(exe_model, epochs, learning_rate, batch_size, target_accuracy=1, division=1000, weight_decay=0):
    train_data_loader, valid_data_loader, test_data_loader = data_loader(batch_size)
    valid_accuracy = torch.zeros(epochs)
    fine_stop = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_data_loader, exe_model, loss_fn, learning_rate, target_accuracy, division, weight_decay, fine_stop)
        valid_accuracy[t] = accuracy(valid_data_loader, exe_model, loss_fn, 'Validation')[1]
        if valid_accuracy[t] >= 0.93:  # 0.93 is the trigger validation accuracy
            fine_stop = 1
        if valid_accuracy[t] >= target_accuracy:
            print('Target accuracy surpassed.')
            break
    acc = accuracy(test_data_loader, exe_model, loss_fn, 'Test')[1]
    print("Done!")
    return acc


# To train several models with different seeds
def multiple_execution(max_epochs, learning_rate, batch_size, num_executions, target_accuracy=1, division=1000, weight_decay=0, init_factor=1):
    acc = torch.zeros(num_executions)
    for i in range(num_executions):
        print('Execution ' + str(i + 1))
        torch.manual_seed(1 + i)
        model = NeuralNetwork().to(device)
        def init_weights(m):  # To set uniform or normal initialisation
            if type(m) == nn.Linear:
                y = m.in_features
                m.weight.data.normal_(0, init_factor/np.sqrt(y))
                # Or in the uniform case: m.weight.data.uniform(-init_factor/np.sqrt(y), init_factor/np.sqrt(y))
                m.bias.data.normal_(0, init_factor / np.sqrt(y))
                # Or in the uniform case: m.bias.data.uniform(-init_factor/np.sqrt(y), init_factor/np.sqrt(y))
        model.apply(init_weights)
        acc[i] = execution(model, max_epochs, learning_rate, batch_size, target_accuracy, division, weight_decay)
        torch.save(model.state_dict(), 'new_model{}_e{}_lr{}_b{}_fta{}_wd{}_uif{}.pth'.format(i + 1, max_epochs, learning_rate,
                                                                                    batch_size, round(100*target_accuracy), weight_decay, init_factor))
    return acc
