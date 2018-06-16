# -*- coding: utf-8 -*-

"""
File: variables.py
"""

import numpy as np

class Variable(object):
    """ An abstruct class for variables. """
    def __init__(self, shape, initializer, name):
        super(Variable, self).__init__()
        self.name = name
        self.shape = shape
        self.initializer = initializer
        self.values = initializer(shape)


class Input(object):
    """ The input """
    def __init__(self, shape):
        super(Input, self).__init__()
        self.shape = shape

