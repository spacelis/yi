"""
File: operator.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description: Operators that support auto derivative
"""

import numpy as np

# pylint: disable=invalid-name

class Operator(object):
    """ The base operator function """
    def __init__(self, operands=tuple()):
        super(Operator, self).__init__()
        self.operands = operands
        self.trainables = [x for x in operands if x.is_trainable()]
        self.value = None

    def compute(self):
        """ Compute the output for input """

    def perturb(self, gradients):
        """ Back Propagate the gradient to input """

    def setval(self, value):
        """ Set the value to the parameter """
        self.value = value

    def is_trainable(self):
        """ Whether this operator leads to a source that is trainable """


class Variable(Operator):
    """ A place holder for """
    def __init__(self, shape):
        super(Variable, self).__init__()
        self.value = np.zeros(shape)

    def perturb(self, gradients):
        self.value[:] += gradients

    def is_trainable(self):
        return True


class Placeholder(Operator):

    """ A place holder for inputs """

    def __init__(self, shape_spec):
        Operator.__init__(self)
        self._shape_spec = shape_spec

    def is_trainable(self):
        return False


class MatMul(Operator):
    """ Matrix Multiplication """
    def __init__(self, op1, op2):
        super(MatMul, self).__init__([op1, op2])

    def compute(self):
        self.value = np.einsum('...jk,...kl->...jl', self.operands[0], self.operands[1])

    def perturb(self, gradients):
        pass
