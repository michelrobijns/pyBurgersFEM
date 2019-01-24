from fem.basis_functions import piecewise_linear, derivative_piecewise_linear


class LinearElement(object):

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
        return self.coefficients[self.indices[0]] * self.phi(x, 0) + \
               self.coefficients[self.indices[1]] * self.phi(x, 1)


    def du(self, x):
        return self.coefficients[self.indices[0]] * self.dphi(x, 0) + \
               self.coefficients[self.indices[1]] * self.dphi(x, 1)


    def previous_u(self, x):
        return self.previous_coefficients[self.indices[0]] * self.phi(x, 0) + \
               self.previous_coefficients[self.indices[1]] * self.phi(x, 1)


    def previous_du(self, x):
        return self.previous_coefficients[self.indices[0]] * self.dphi(x, 0) + \
               self.previous_coefficients[self.indices[1]] * self.dphi(x, 1)


    def phi(self, x, i):
        return piecewise_linear(x, i, self.nodes)


    def dphi(self, x, i):
        return derivative_piecewise_linear(x, i, self.nodes)
