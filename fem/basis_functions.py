import numpy as np


def piecewise_linear(x, i, nodes):
    if i == 1:
        return (x - nodes[0]) / (nodes[1] - nodes[0])
    else:
        return (x - nodes[1]) / (nodes[0] - nodes[1])


def derivative_piecewise_linear(x, i, nodes):
    if i == 1:
        return 1.0 / (nodes[1] - nodes[0])
    else:
        return 1.0 / (nodes[0] - nodes[1])


def lagrange(x, i, nodes):
    result = 1.0

    for j in range(len(nodes)):
        if (j != i):
            result *= ((x - nodes[j]) / (nodes[i] - nodes[j]))

    return result


def derivative_lagrange(x, i, nodes):
    result = 0.0

    for j in range(len(nodes)):
        if (j != i):
            partial_result = 1.0 / (nodes[i] - nodes[j])

            for k in range(len(nodes)):
                if (k != i and k != j):
                    partial_result *= ((x - nodes[k]) / (nodes[i] - nodes[k]))

            result += partial_result;

    return result;