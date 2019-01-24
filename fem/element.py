import numpy as np

from fem.basis_functions import lagrange, derivative_lagrange


class Element(object):

    def __init__(self, index, nodes, indices, coefficients, previous_coefficients):
        self.index = index
        self.nodes = nodes
        self.indices = indices
        self.coefficients = coefficients
        self.previous_coefficients = previous_coefficients

        self.x_left = nodes[0]
        self.x_right = nodes[-1]
        self.h = self.x_right - self.x_left


    def u(self, x):
        x = np.asarray(x)
        u = np.zeros_like(x)

        for i, index in enumerate(self.indices):
            u += self.coefficients[index] * self.phi(x, i)

        return u


    def du(self, x):
        x = np.asarray(x)
        u = np.zeros_like(x)

        for i, index in enumerate(self.indices):
            u += self.coefficients[index] * self.dphi(x, i)

        return u


    def previous_u(self, x):
        x = np.asarray(x)
        u = np.zeros_like(x)

        for i, index in enumerate(self.indices):
            u += self.previous_coefficients[index] * self.phi(x, i)

        return u


    def phi(self, x, i):
        return lagrange(x, i, self.nodes)


    def dphi(self, x, i):
        return derivative_lagrange(x, i, self.nodes)
