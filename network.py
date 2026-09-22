import math
import random


def neuron(inputs, weights, bias):
    weighted_sum = 0
    for i, w in zip(inputs, weights):
        weighted_sum += i * w
    weighted_sum += bias
    return math.tanh(weighted_sum)


def layer(inputs, weights_list, biases):
    outputs = []
    for weights, bias in zip(weights_list, biases):
        outputs.append(neuron(inputs, weights, bias))
    return outputs


def forward(inputs, hidden_weights, hidden_biases, output_weights, output_biases):
    hidden_output = layer(inputs, hidden_weights, hidden_biases)
    return layer(hidden_output, output_weights, output_biases)


def random_weights(n_neurons, n_inputs):
    return [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_neurons)]


def random_biases(n_neurons):
    return [random.uniform(-1, 1) for _ in range(n_neurons)]