import numpy as np
import activationFunctions as ActFunc
import copy


# TODO Neural network has problems using the ReLU function

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output,
                 learning_rate=0.1,
                 activation_function=None,
                 derivative_activation_function=None):

        self.__input_nodes__ = n_input
        self.__hidden_nodes__ = n_hidden
        self.__output_nodes__ = n_output

        self.__learning_rate__ = learning_rate

        if activation_function is None:
            self.__activation_fun__ = ActFunc.sigmoid
        else:
            self.__activation_fun__ = activation_function

        if activation_function is None:
            self.__derivative_activation_func__ = ActFunc.derivative_sigmoid
        else:
            self.__derivative_activation_func__ = derivative_activation_function

        self.__weights_input_hidden__ = np.random.uniform(-1, 1, (self.__hidden_nodes__, self.__input_nodes__))
        self.__weights_hidden_output__ = np.random.uniform(-1, 1, (self.__output_nodes__, self.__hidden_nodes__))

        self.__bias_h__ = np.random.uniform(-1, 1, (self.__hidden_nodes__, 1))
        self.__bias_o__ = np.random.uniform(-1, 1, (self.__output_nodes__, 1))

    def train(self, input_array, target_array):
        # convert inputs into a 1-dim matrix
        input_matrix = np.matrix(input_array)
        input_matrix = input_matrix.reshape((-1, 1))

        # do the matrix product of the input and the weights
        hidden_matrix = np.dot(self.__weights_input_hidden__, input_matrix)

        # add the hidden bias matrix
        hidden_matrix += self.__bias_h__

        # apply the sigmoid function
        vector_act_func = np.vectorize(self.__activation_fun__)
        hidden_matrix = vector_act_func(hidden_matrix)

        # do the matrix product of the hidden output weights and the hidden layer
        output_matrix = np.dot(self.__weights_hidden_output__, hidden_matrix)

        # add the output bias matrix
        output_matrix += self.__bias_o__

        # apply the sigmoid function
        output_matrix = vector_act_func(output_matrix)

        # -------------------------------------------

        # convert targets into a 1-dim matrix
        target_matrix = np.matrix(target_array)
        target_matrix = target_matrix.reshape((-1, 1))

        # calc output error - ERROR = TARGETS - OUTPUTS
        output_error_matrix = target_matrix - output_matrix

        # calc delta weight matrix
        derivative_vector_act_func = np.vectorize(self.__derivative_activation_func__)
        output_gradient_matrix = derivative_vector_act_func(output_matrix)
        output_gradient_matrix = np.multiply(output_gradient_matrix, output_error_matrix)
        output_gradient_matrix = np.multiply(output_gradient_matrix, self.__learning_rate__)

        hidden_matrix_transposed = np.transpose(hidden_matrix)

        weight_hidden_output_deltas = np.dot(output_gradient_matrix, hidden_matrix_transposed)

        self.__weights_hidden_output__ += weight_hidden_output_deltas
        self.__bias_o__ += output_gradient_matrix

        # transpose/flip weight matrix
        weights_ho_transposed = self.__weights_hidden_output__.transpose()

        # calc hidden error
        hidden_error_matrix = np.dot(weights_ho_transposed, output_error_matrix)

        hidden_gradient_matrix = derivative_vector_act_func(hidden_matrix)
        hidden_gradient_matrix = np.multiply(hidden_gradient_matrix, hidden_error_matrix)
        hidden_gradient_matrix = np.multiply(hidden_gradient_matrix, self.__learning_rate__)

        input_matrix_transposed = np.transpose(input_matrix)

        weight_input_hidden_deltas = np.dot(hidden_gradient_matrix, input_matrix_transposed)

        self.__weights_input_hidden__ += weight_input_hidden_deltas
        self.__bias_h__ += hidden_gradient_matrix

    def feed_forward(self, input_array, return_as="M"):
        # convert inputs into a 1-dim matrix
        input_matrix = np.matrix(input_array)
        input_matrix = input_matrix.reshape((-1, 1))

        # do the matrix product of the input and the weights
        hidden = np.dot(self.__weights_input_hidden__, input_matrix)

        # add the hidden bias matrix
        hidden += self.__bias_h__

        # apply the sigmoid function
        vector_act_func = np.vectorize(self.__activation_fun__)
        hidden = vector_act_func(hidden)

        # do the matrix product of the hidden output weights and the hidden layer
        output = np.dot(self.__weights_hidden_output__, hidden)

        # add the output bias matrix
        output += self.__bias_o__

        # apply the sigmoid function
        output = vector_act_func(output)

        if return_as == "M":
            return output
        elif return_as == "L":
            return np.matrix.tolist(output)

    def copy(self):
        return copy.copy(self)

    def set_learning_rate(self, new_learning_rate):
        self.__learning_rate__ = new_learning_rate


