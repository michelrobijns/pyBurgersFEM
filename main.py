import math
import argparse
import numpy as np

from fem.Burgers_model import BurgersModel


def forcing_function(x, t):
    return 0.0


def left_boundary_value(t):
    return 0.0


def right_boundary_value(t):
    return 0.0


def initial_condition(x):
    return 1.0 + np.sin(2.0 * np.pi * x - 1.0)


def main(args):
    # Set variables that define the problem.
    nu = 0.01
    number_of_elements = 1024 if not args.elements else args.elements
    x_left = 0.0
    x_right = 1.0
    t_begin = 0.0
    t_end = 2.0
    dt = 0.01

    # Create a vector containing the x-coordinates of the nodes
    number_of_nodes = number_of_elements + 1
    nodes = np.linspace(x_left, x_right, number_of_nodes)

    # Initialize the model
    model = BurgersModel(number_of_elements,
                         nodes,
                         nu,
                         forcing_function,
                         None, #left_boundary_value,
                         None, #right_boundary_value,
                         initial_condition)

    # Allocate storage for the solution
    t = np.arange(t_begin, t_end + dt, dt)
    x = np.copy(nodes)
    u = np.zeros((t.size, number_of_nodes))

    u[0, :] = model.coefficients

    # Time loop
    for i in range(1, len(t)):
        # Advance `dns_model' to `t[i]'
        model.advance_in_time(t[i])

        # Store coefficients
        u[i, :] = model.coefficients

    # Save solution
    np.save("t.npy", t)
    np.save("x.npy", x)
    np.save("u.npy", u)


if __name__ == '__main__':
    np.set_printoptions(
        precision=4,
        linewidth=200,
        suppress=True)

    # Parse the optional command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--elements", type=int, help="number of elements")
    args = parser.parse_args()

    main(args)
