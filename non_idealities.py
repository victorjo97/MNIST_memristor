"""
All algorithms related to non-idealities are coded in this script. It is divided in three parts:
1) A pair of preliminary functions.
2) The algorithms for the clipping of the largest synaptic parameters in absolute value.
3) The algorithms for the optimisation of the approximation of the synaptic parameters to the conductance levels in
differential mapping.
"""


import torch
import copy
from data_handling import *
from scipy.optimize import golden
from torch_MNIST import NeuralNetwork, data_loader, loss_fn, accuracy
import pickle

models_number = 30  #
lrs = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]  # learning rates
uifs = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # uniform initialisation factors
nifs = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # normal initialisation factors
wds = [0.000125, 0.00025, 0.0005, 0.001, 0.002]  # weight decay factors


## Preliminary functions
# Approximates to the nearest weight state
def return_closest(weight_matrix, weight_states):
    shape = weight_matrix.shape
    weight_vector = weight_matrix.flatten()
    num_states = len(weight_states)
    num_weights = len(weight_vector)
    weight_states = weight_states.reshape(num_states, 1)
    weight_states_repeat = weight_states.repeat(1, num_weights)
    weight_vector_repeat = weight_vector.repeat(num_states, 1)
    values, indices = torch.min(torch.abs(weight_vector_repeat - weight_states_repeat), 0)
    return weight_states_repeat[indices, 0].reshape(shape)


# To generate the states in differential mapping
def differential_states_generator(w_max, num_states):
    if num_states == 1:
        print('The number of states must be greater than 1.')
    conductance_states = torch.cat((torch.linspace(- w_max, 0, num_states), torch.linspace(0, w_max, num_states)))
    return conductance_states


## Clipping Algorithms
# To perform the clipping of a clipping_proportion of the largest synaptic parameters in absolute value.
def clipping(clipping_proportion, learning_rate, init_type, init_factor, weight_decay):
    test_data_loader = data_loader()[2]
    acc = torch.zeros(models_number)
    error = torch.zeros(models_number)  # Initialisation of the normalised root-mean-squared error (NRMSE)
    model = NeuralNetwork().to("cpu")
    num_parameters = sum(p.numel() for p in model.parameters())
    for k in range(models_number):
        # loading of pretrained models
        model.load_state_dict(torch.load(
            'nnew_model' + str(k + 1) + '_e200_p100_lr' + str(learning_rate) + '_b10_fta95_wd' + str(
                weight_decay) + '_' + init_type + str(init_factor) + '.pth'))
        model_parameters = weight_extractor(model)
        num_synapses = int(len(model_parameters) / 2)
        for synapse in range(num_synapses):
            weights, biases = model_parameters[synapse * num_synapses].flatten(), model_parameters[
                synapse * num_synapses + 1].flatten()
            synapse_parameters = torch.cat((weights, biases))  # weights and biases are given the same consideration
            num_synapse_parameters = len(synapse_parameters)
            values, indices = torch.absolute(synapse_parameters).sort()
            w_max_discrete = values[round(num_synapse_parameters * (1 - clipping_proportion)) - 1]
            var = torch.var(synapse_parameters)
            for i in range(len(weights)):
                if weights[i] < - w_max_discrete:
                    error[k] += (weights[i] + w_max_discrete) ** 2 / var  # To implement the normalisation of the NRMSE
                    weights[i] = - w_max_discrete
                elif weights[i] > w_max_discrete:
                    error[k] += (weights[i] - w_max_discrete) ** 2 / var
                    weights[i] = w_max_discrete
            for i in range(len(biases)):
                if biases[i] < - w_max_discrete:
                    error[k] += (biases[i] + w_max_discrete) ** 2 / var
                    biases[i] = - w_max_discrete
                elif biases[i] > w_max_discrete:
                    error[k] += (biases[i] - w_max_discrete) ** 2 / var
                    biases[i] = w_max_discrete
        error[k] /= num_parameters
        acc[k] = accuracy(test_data_loader, model, loss_fn, 'Test')[1]
    return acc.mean(), torch.sqrt(error).mean()


# To manage the clipping for the different training parameters modified
def clipping_manager(clipping_proportion, learning_rate, init_type, init_factor, weight_decay):
    accuracies = []
    errors = []
    if learning_rate == '':
        for lr in lrs:
            accuracy, error = clipping(clipping_proportion, lr, init_type, init_factor, weight_decay)
            accuracies.append(accuracy)
            errors.append(error)
    elif (init_factor == '') & (init_type == 'uif'):
        for uif in uifs:
            accuracy, error = clipping(clipping_proportion, learning_rate, init_type, uif, weight_decay)
            accuracies.append(accuracy)
            errors.append(error)
    elif (init_factor == '') & (init_type == 'nif'):
        for nif in nifs:
            accuracy, error = clipping(clipping_proportion, learning_rate, init_type, nif, weight_decay)
            accuracies.append(accuracy)
            errors.append(error)
    else:
        for wd in wds:
            accuracy, error = clipping(clipping_proportion, learning_rate, init_type, init_factor, wd)
            accuracies.append(accuracy)
            errors.append(error)

    with open('alternative_clipping' + str(clipping_proportion) + '_lr' + str(learning_rate) + '_b10_fta95_wd' + str(
            weight_decay) + '_' + init_type + str(init_factor) + '.pkl', 'wb') as f:
        pickle.dump({'accuracies': accuracies, 'errors': errors}, f)


## Differential Mapping Functions
# To approximate the synaptic parameters in an ideal model to the discrete levels in differential mapping. The model
# parameters can be extracted by doing parameters = weight_extractor(model)
def differential_mapping(model_parameters, num_states, clipping_proportion):
    if clipping_proportion < 0:  # This is included because the golden search (see below) algorithm some times tries
        # clipping proportions out of the initially indicated range
        clipping_proportion = 0
    num_synapses = int(len(model_parameters) / 2)
    for synapse in range(num_synapses):
        weights, biases = model_parameters[synapse * num_synapses], model_parameters[synapse * num_synapses + 1]
        synapse_parameters = torch.cat((weights.flatten(), biases.flatten()))
        num_synapse_parameters = len(synapse_parameters)
        values, indices = torch.absolute(synapse_parameters).sort()
        w_max_discrete = values[round(num_synapse_parameters * (1 - clipping_proportion)) - 1]
        conductance_states = differential_states_generator(w_max_discrete, num_states)
        model_parameters[synapse * num_synapses] = return_closest(weights, conductance_states)
        model_parameters[synapse * num_synapses + 1] = return_closest(biases, conductance_states)
    return model_parameters


# Defines the accuracy target function that is optimised when trying different clipping proportions
def differential_mapping_optimisation_target(clipping_proportion, num_states, accuracy_type, i, learning_rate,
                                             init_type,
                                             init_factor, weight_decay):
    valid_data_loader = data_loader()[1]
    test_data_loader = data_loader()[2]
    model = NeuralNetwork().to("cpu")
    model.load_state_dict(
        torch.load(
            'nnew_model' + str(i + 1) + '_e200_p100_lr' + str(learning_rate) + '_b10_fta95_wd' + str(
                weight_decay) + '_' + init_type + str(
                init_factor) + '.pth'))
    model_parameters = weight_extractor(model)
    models_parameters = differential_mapping(model_parameters, num_states, clipping_proportion)
    weight_update(model, models_parameters)
    if accuracy_type == 'test':
        acc = accuracy(test_data_loader, model, loss_fn, 'Test')[1]
    elif accuracy_type == 'valid':
        acc = accuracy(valid_data_loader, model, loss_fn, 'Validation')[1]
    else:
        acc = 0
    return - acc  # the sign is changed because the golden search algorithm is for minimisation


# Defines the error (NRMSE) target function that is optimised when trying different clipping proportions
def differential_mapping_optimisation_error_target(clipping_proportion, num_states, i, learning_rate, init_type,
                                                   init_factor, weight_decay):
    model = NeuralNetwork().to("cpu")
    model.load_state_dict(
        torch.load(
            'nnew_model' + str(i + 1) + '_e200_p100_lr' + str(learning_rate) + '_b10_fta95_wd' + str(
                weight_decay) + '_' + init_type + str(
                init_factor) + '.pth'))
    model_parameters = weight_extractor(model)
    copied_parameters = copy.deepcopy(model_parameters)
    mapped_parameters = differential_mapping(model_parameters, num_states, clipping_proportion)
    error = torch.zeros(1)
    for synapse in range(2):
        copied_synapse = torch.cat((copied_parameters[2*synapse].flatten(), copied_parameters[2*synapse+1].flatten()))
        var = torch.var(copied_synapse)
        mapped_synapse = torch.cat(
            (mapped_parameters[2 * synapse].flatten(), mapped_parameters[2 * synapse + 1].flatten()))
        error += ((copied_synapse - mapped_synapse) ** 2).sum() / var
    return torch.sqrt(error / sum(p.numel() for p in model.parameters()))


# Optimisation of the clipping proportion in differential mapping when maximising the validation accuracy
def differential_mapping_optimisation(num_states, learning_rate, init_type, init_factor, weight_decay):
    optimal_clipping_proportions = torch.zeros(models_number)
    optimal_validation_accuracies = torch.zeros(models_number)
    optimal_test_accuracies = torch.zeros(models_number)
    for i in range(models_number):
        # use of the golden search algorithm to maximise the validation accuracy. brack is the initial range of clipping
        # proportions where the algorithm searches the optimal value
        optimal_clipping_proportions[i] = golden(differential_mapping_optimisation_target,
                                                 args=(num_states, 'valid', i, learning_rate, init_type, init_factor,
                                                       weight_decay),
                                                 brack=(0, 0.1), tol=0.005)
        optimal_validation_accuracies[i] = - differential_mapping_optimisation_target(
            optimal_clipping_proportions[i].item(), num_states, 'valid', i, learning_rate, init_type, init_factor,
            weight_decay)
        optimal_test_accuracies[i] = - differential_mapping_optimisation_target(
            optimal_clipping_proportions[i].item(), num_states, 'test', i, learning_rate, init_type, init_factor,
            weight_decay)
    return optimal_clipping_proportions.mean(), optimal_clipping_proportions.std(), \
           optimal_validation_accuracies.mean(), optimal_validation_accuracies.std(), \
           optimal_test_accuracies.mean(), optimal_test_accuracies.std()


# Optimisation of the clipping proportion in differential mapping when minimising the NRMSE
def differential_mapping_optimisation_error(num_states, learning_rate, init_type, init_factor, weight_decay):
    optimal_clipping_proportions = torch.zeros(models_number)
    optimal_errors = torch.zeros(models_number)
    for i in range(models_number):
        optimal_clipping_proportions[i] = golden(differential_mapping_optimisation_error_target,
                                                 args=(num_states, i, learning_rate, init_type, init_factor,
                                                       weight_decay),
                                                 brack=(0, 0.1))
        optimal_errors[i] = differential_mapping_optimisation_error_target(
            optimal_clipping_proportions[i].item(), num_states, i, learning_rate, init_type, init_factor, weight_decay)
    return optimal_clipping_proportions.mean(), optimal_clipping_proportions.std(), \
            optimal_errors.mean(), optimal_errors.std()
