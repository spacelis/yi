# -*- coding: utf-8 -*-
"""
File: session.py
Description: Sessions are instances of network.
"""

from .variable import Variable

class Session(object):
    """ An instance of network """
    def __init__(self):
        super(Session, self).__init__()
        self.variables = {}

    def all_variables(self):
        pass

    def variable(self, shape, initializer, name):
        var = Variable(shape, initializer, name)
        self.variables[name] = var
        return var
