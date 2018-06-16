# -*- coding: utf-8 -*-
"""
File: measure.py
Description: Cost functions
"""

import numpy as np
from .layer import sigmoid

class Measure(object):
    """ Measure the prediction to ground truth """
    def __init__(self, shape):
        super(Measure, self).__init__()
        self.shape = shape

    def measure(self, predictions, targets):
        """ Measure the difference. """
        pass

    def gradient(self, predictions, targets):
        """ Generate gradients for back propagation """
        pass


class L2Measure(Measure):
    """ L2 """

    def measure(self, predictions, targets):
        """ Measuring ||predictions - targets||_2
        """
        return np.square(predictions - targets).mean() / 2

    def gradient(self, predictions, targets):
        """ The gradient of the measure
        """
        return predictions - targets


class BinaryCrossEntropy(Measure):
    """ CrossEntropy """

    def measure(self, logits, targets):
        """ J(x) = -log(predictions) * targets - log(1-prediction) * (1-targets)
        """
        predictions = sigmoid(logits)
        return (-np.log(predictions) * targets - np.log(1-predictions) * (1-targets)).mean() / 2

    def gradient(self, logits, targets):
        """ The gradient of the measure
        """
        return sigmoid(logits) - targets


class CrossEntropy(Measure):
    """ CrossEntropy """

    def measure(self, logits, targets):
        """ J(x) = -log(predictions) * targets - log(1-prediction) * (1-targets)
        """
        exps = np.exp(logits)
        predictions = exps / exps.sum(axis=tuple(range(1, len(exps.shape))))\
            .reshape((exps.shape[0],) + (1,) * (len(exps.shape) - 1))
        return (-np.log(predictions) * targets).mean()

    def gradient(self, logits, targets):
        """ The gradient of the measure
        """
        exps = np.exp(logits)
        predictions = exps / exps.sum(axis=tuple(range(1, len(exps.shape))))\
            .reshape((exps.shape[0],) + (1,) * (len(exps.shape) - 1))
        return predictions - targets
