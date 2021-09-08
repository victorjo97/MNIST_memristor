"""
Implements the extraction and update of weights to the models, and the conversion from weights to conductances and
vice versa.
"""

import torch  # I've used the tools of torch for matrix/vector computations instead of numpy as the weights are
# naturally stored in the form of torch tensors


# Obtains as tensors the weight in a model
def weight_extractor(network):
    return [network.state_dict()[label] for label in network.state_dict()]


# Updates the weight values in a model
def weight_update(network, weights):
    dict = network.state_dict()
    for weight, label in zip(weights, dict):
        dict[label] = weight
    network.load_state_dict(dict)


# Generates a list of tensors with columns that alternatively implement positive and negative weights
def weights2conductances(weights, k_g=1):
    conductances = []
    for weight in weights:
        m, n = weight.shape
        conductance = torch.zeros((m, 2 * n))
        conductance[:, 0:(2 * n + 1):2] = (weight + torch.absolute(weight)) / 2
        conductance[:, 1:(2 * n + 1):2] = - (weight - torch.absolute(weight)) / 2
        conductance *= k_g
        conductances.append(conductance)
    return conductances


# Generates the list of weight tensors from the conductance tensors where the columns alternatively implement positive
# and negative weights
def conductances2weights(conductances, k_g=1):
    weights = []
    for conductance in conductances:
        m, n = conductance.shape
        weight = conductance[:, 0:(n + 1):2] - conductance[:, 1:(n + 1):2]
        weight /= k_g
        weights.append(weight)
    return weights
