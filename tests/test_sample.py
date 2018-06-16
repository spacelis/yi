"""
File: test_sample.py
Author: Wen Li
Email: spacelis@gmail.com
Github: http://github.com/spacelis
Description: Testing
"""

# Sample Test passing with nose and pytest
# pylint: disable=invalid-name

import numpy as np

from yi.session import Session
from yi.initializer import naive_initializer

def test_linear_regression():
    """ A simple test"""
    from yi.measure import L2Measure
    from yi.layer import Linear
    sess = Session()
    lop = Linear((1,), (1,), naive_initializer, sess)
    msr = L2Measure((1,))
    data = np.array([
        [1, 3],
        [2, 5],
        [3, 6],
        [4, 10],
        [20, 44],
    ])
    inputs = data[:, 0].reshape(-1, 1)
    targets = data[:, 1].reshape(-1, 1)
    for i in range(100):
        preds = lop.forward(inputs)
        cost = msr.measure(preds, targets)
        gradients = msr.gradient(preds, targets) * 0.0005
        lop.backward(inputs, gradients)
        if i % 10 == 0:
            print('cost=', cost)
    print('h(x) = {0}x + {1}'.format(lop.weights.values[0, 0], lop.bias.values[0]))


def test_logistic_regression():
    """ A simple test"""
    from yi.measure import BinaryCrossEntropy
    from yi.layer import Linear, Sigmoid
    sess = Session()
    lop = Linear((2,), (1,), naive_initializer, sess)
    sig = Sigmoid()
    msr = BinaryCrossEntropy((1,))
    features = np.array([
        [-1, 3],
        [-2, 0],
        [1, -1],
        [4, 1],
    ])
    labels = np.array([
        [1],
        [1],
        [0],
        [0],
    ])
    for i in range(100):
        o1 = lop.forward(features)
        o2 = sig.forward(o1)
        cost = msr.measure(o1, labels)
        gradients = msr.gradient(o1, labels)
        lop.backward(features, gradients * 0.05)
        if i % 10 == 0:
            print('cost=', cost)
    print('pridctions=\n', np.array2string(o2, formatter={'float_kind':lambda x: "%.2f" % x}))
    print('h(x)= {0:.2f}x_1 + {1:.2f}x_2 + {2:.2f}'.format(
        lop.weights.values[0, 0],
        lop.weights.values[1, 0],
        lop.bias.values[0]
        ))


def test_logistic_regression_softmax():
    """ A simple test"""
    from yi.measure import CrossEntropy
    from yi.layer import Linear, Softmax
    sess = Session()
    lop = Linear((2,), (2,), naive_initializer, sess)
    som = Softmax((2,))
    msr = CrossEntropy((2,))
    features = np.array([
        [-1, 3],
        [-2, 0],
        [1, -1],
        [4, 1],
    ])
    labels = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    for i in range(100):
        o1 = lop.forward(features)
        o2 = som.forward(o1)
        cost = msr.measure(o1, labels)
        gradients = msr.gradient(o2, labels)
        gradients = som.backward(o1, gradients)
        lop.backward(features, gradients * 0.05)
        if i % 10 == 0:
            print('cost=', cost)
    print('pridctions=\n', np.array2string(o2, formatter={'float_kind':lambda x: "%.2f" % x}))
    print('h1(x)= σ({0:.2f}x_1 + {1:.2f}x_2 + {2:.2f})'.format(
        lop.weights.values[0, 0],
        lop.weights.values[1, 0],
        lop.bias.values[0]
        ))


if __name__ == "__main__":
    test_linear_regression()
    test_logistic_regression()
    test_logistic_regression_softmax()
