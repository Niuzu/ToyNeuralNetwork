import numpy as np

def sigmoid(x):
    """
    The sigmoid function produces an output in scale of [0, +1].
    x = 0 -> sigmoid(0) = 0.5
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x + (1 - x)


def hyperbolic_tangent(x):
    return np.tanh(x)


def derivative_hyperbolic_tangent(x):
    tan_h = hyperbolic_tangent(x)
    return 1 - np.power(tan_h, 2)


def relu(x):
    """
    :param x:
    :return:
    """
    return x * (x >= 0)


def derivative_relu(x):
        return 1 * (x >= 0)
