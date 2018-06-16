# -*- coding: utf-8 -*-

"""
File: layer.py
Description: All the layers
"""

import numpy as np

class Layer(object):
    """ An abstract class for layers """
    def __init__(self):
        super(Layer, self).__init__()
        self.name = self.__class__.__name__

    def forward(self, inputs):
        """ Forward pass """
        pass

    def backward(self, inputs, gradients):
        """ Back propagation pass """
        pass

class Linear(Layer):
    """ A fully connected layer without activation. """
    def __init__(self, input_shape, output_shape, initializer, session):
        super(Linear, self).__init__()
        self.input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.input_vals = np.prod(self.input_shape)
        self.output_vals = np.prod(self.output_shape)
        self.weight_shape = (self.input_vals, self.output_vals)
        self.weights = session.variable(self.weight_shape, initializer, 'L_w')
        self.bias = session.variable((self.output_vals,), initializer, 'L_b')

    def forward(self, inputs):
        index_shape = inputs.shape[:-len(self.input_shape)]
        inputs = inputs.reshape(index_shape + (self.input_vals,))
        ltrans = np.dot(inputs, self.weights.values)
        expanded_bias = self.bias.values.reshape(tuple(1 for _ in inputs.shape[:-1]) +
                                                 (self.output_vals,))
        return (ltrans + expanded_bias).reshape(index_shape + self.output_shape)

    def backward(self, inputs, gradients):
        index_shape = inputs.shape[:-len(self.input_shape)]
        ex_inputs = np.broadcast_to(inputs.reshape(index_shape + (self.input_vals, 1)),
                                    index_shape + (self.input_vals, self.output_vals))
        ex_gradients = np.broadcast_to(gradients.reshape(index_shape + (1, self.output_vals)),
                                       index_shape + (self.input_vals, self.output_vals))
        gradients_on_input = np.dot(gradients, self.weights.values.T)
        self.weights.values -= np.sum(ex_inputs * ex_gradients, axis=tuple(range(len(index_shape))))
        self.bias.values -= np.sum(gradients, axis=tuple(range(len(index_shape))))
        return gradients_on_input


def sigmoid(inputs):
    """ sigmoid(x) = 1 / (1 - exp(-x))"""
    return 1 / (1 + np.exp(-inputs))


class Sigmoid(Layer):
    """ Sigmoid(x) = 1 / (1 + exp(-x))"""
    def forward(self, inputs):
        return sigmoid(inputs)

    def backward(self, inputs, gradients):
        sig = sigmoid(inputs)
        return  sig * (1 - sig) * gradients


class Softmax(Layer):
    """ Softmax = [ exp(-x_i) / sum_j(exp(-x_j)) ]"""
    def __init__(self, input_shape):
        super(Softmax, self).__init__()
        self.input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape,)
        self.input_vals = np.prod(self.input_shape)

    def forward(self, inputs):
        numerators = np.exp(inputs)
        denominator = numerators.sum(axis=tuple(range(len(inputs.shape)))[-len(self.input_shape):],
                                     keepdims=True)
        return numerators / denominator

    def backward(self, inputs, gradients):
        index_shape = inputs.shape[:-len(self.input_shape)]
        numerators = np.exp(inputs)
        denominator = numerators.sum(axis=tuple(range(len(inputs.shape)))[-len(self.input_shape):],
                                     keepdims=True)

        sigma = numerators / denominator
        sigma = np.broadcast_to(np.expand_dims(sigma, -1),
                                sigma.shape + (self.input_vals,))
        eye = np.eye(self.input_vals).reshape([1 for _ in index_shape]+
                                              [self.input_vals] * 2)
        eye = np.broadcast_to(eye, index_shape + (self.input_vals, self.input_vals))
        derivative = sigma * (eye - sigma)
        gradients_on_input = np.einsum('...jk,...j->...k', derivative, gradients)
        return gradients_on_input
