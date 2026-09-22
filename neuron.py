import math

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


result = layer(
    inputs=[0.1, 0.9, 0.6, 0.4, 0.7],
    weights_list=[
        [0.3, 0.9, 0.1, -0.2, 0.5],   # neuron A's weights
        [-0.1, 0.4, 0.6, 0.2, -0.3],  # neuron B's weights
        [0.2, -0.5, 0.3, 0.1, 0.4],   # neuron C's weights
    ],
    biases=[2, -0.3, 0.1]
)
print(result)