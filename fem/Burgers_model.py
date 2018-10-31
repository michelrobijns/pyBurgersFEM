import sys
import numpy as np
from math import sqrt

from fem.linear_element import LinearElement

class BurgersModel(object):

    def __init__(self,
                 number_of_elements,
                 nodes,
                 nu,
                 forcing_function,
                 left_boundary_value,
                 left_boundary_type,
                 right_boundary_value,
                 right_boundary_type,
                 initial_condition,
                 integration_points=3,
                 newton_iterations=20,
                 newton_tolerance=1e-6):

        self.number_of_elements = number_of_elements
        self.nodes = nodes
        self.nu = nu
        self.forcing_function = forcing_function
        self.left_boundary_value = left_boundary_value
        self.left_boundary_type = left_boundary_type
        self.right_boundary_value = right_boundary_value
        self.right_boundary_type = right_boundary_type
        self.initial_condition = initial_condition
        self.integration_points = integration_points
        self.newton_iterations = newton_iterations
        self.newton_tolerance = newton_tolerance

        self.number_of_nodes = nodes.size
        self.nodes_per_element = 2

        self.coefficients = np.empty(self.number_of_nodes)
        self.previous_coefficients = np.empty(self.number_of_nodes)

        # Check boundary conditions
        if self.left_boundary_type == "Dirichlet" and self.right_boundary_type == "Dirichlet":
            print()
            print("Boundary conditions: Dirichlet")
        elif self.left_boundary_type == None and self.right_boundary_type == None:
            print()
            print("Boundary conditions: periodic")
        else:
            print()
            print("Unsupported boundary conditions, aborting.")
            sys.exit()

        # Store `Element' objects in `self.elements'
        self.elements = self.create_elements(number_of_elements)

        # Set time variable
        self.time = 0.0

        # Preallocate Jacobian matrix and residual vector
        self.J = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.F = np.zeros(self.number_of_nodes)

        # Project the `initial_condition' function onto the finite element space
        np.copyto(self.coefficients, self.project_initial_condition())


    def create_elements(self, number_of_elements):
        print()
        print("Number of elements: {}".format(number_of_elements))
        print()

        elements = []

        for i in range(self.number_of_elements):
            local_nodes = [self.nodes[i], self.nodes[i+1]]
            indices = [i, i+1]

            elements.append(LinearElement(local_nodes, indices,
                                          self.coefficients,
                                          self.previous_coefficients))

        return elements


    def advance_in_time(self, new_time):
        print('Advancing from t ={:6.3f} to t ={:6.3f}'.format(self.time, new_time))

        # Set time variables
        self.previous_time = self.time
        self.time = new_time

        # Copy previous solution
        np.copyto(self.previous_coefficients, self.coefficients)

        # Starting guess
        #self.c = np.random.rand(self.number_of_nodes)
        if self.left_boundary_type == "Dirichlet" and self.right_boundary_type == "Dirichlet":
            self.coefficients[0] = self.left_boundary_value(self.time)
            self.coefficients[-1] = self.right_boundary_value(self.time)
        # Starting guess is solution from previous timestep

        # Newton's method
        for k in range(self.newton_iterations):
            # Assemble the residual `-F' and the Jacobian `J'
            self.assemble_F()
            self.assemble_J()

            # Solve the system and update `coefficients'
            if self.left_boundary_type == "Dirichlet" and self.right_boundary_type == "Dirichlet":
                # Modify the system to enforce the boundary conditions
                self.constrain_system(self.J, self.F)

                delta_coefficients = np.linalg.solve(self.J, -self.F)
            if self.left_boundary_type == None and self.right_boundary_type == None:
                delta_coefficients = self.solve_cyclic(self.J, -self.F)

            self.coefficients += delta_coefficients

            # Check for convergence
            if np.linalg.norm(delta_coefficients) < self.newton_tolerance:
                print('    Newton\'s method converged after {} iterations'.format(k+1))
                break

        if k == self.newton_iterations - 1:
            print('    Newton\'s method FAILED to converge within {} iterations'.format(k+1))


    def assemble_F(self, integration_points=3):
        dt = self.time - self.previous_time

        self.F.fill(0.0)

        ip_ref_locs, ip_weights = self.integration_points_and_weights(integration_points)

        # Loop over elements
        for element in self.elements:
            # Loop over local nodes
            for i in range(self.nodes_per_element):
                # Loop over integration points
                for ip in range(integration_points):
                    # Get reference location of the integration point
                    ref_loc = ip_ref_locs[ip]

                    # Calculate x location of quadrature point
                    x_ip = element.x_left + 0.5 * (1 + ref_loc) * element.h

                    # Evaluate variables at `ip_x'
                    u      = element.u(x_ip)
                    du_dx  = element.du(x_ip)
                    du_dt  = (u - element.previous_u(x_ip)) / dt
                    phi_i  = element.phi(x_ip, i)
                    dphi_i = element.dphi(x_ip, i)
                    f      = self.forcing_function(x_ip, self.time)
                    nu     = self.nu

                    # The value of the integrand at `ip_x' will be stored as `result'
                    result = 0.0

                    # Term 1
                    result += du_dt * phi_i

                    # Term 2
                    result += -0.5 * u * u * dphi_i

                    # Term 3
                    result += nu * du_dx * dphi_i

                    # Term 4
                    result += -1.0 * f * phi_i

                    # Add `result' to `vector'
                    self.F[element.indices[i]] += \
                        ip_weights[ip] * 0.5 * element.h * result


    def assemble_J(self, integration_points=3):
        dt = self.time - self.previous_time

        self.J.fill(0.0)

        ip_ref_locs, ip_weights = self.integration_points_and_weights(integration_points)

        # Loop over elements
        for element in self.elements:
            # Loop over local nodes
            for i in range(self.nodes_per_element):
                # Loop over local nodes
                for j in range(self.nodes_per_element):
                    # Loop over integration points
                    for ip in range(integration_points):
                        # Get reference location of the integration point
                        ref_loc = ip_ref_locs[ip]

                        # Calculate x location of quadrature point
                        x_ip = element.x_left + 0.5 * (1 + ref_loc) * element.h

                        # The value of the integrand at `ip_x' will be stored as
                        # `result'
                        result = 0.0

                        # Evaluate variables at `ip_x'
                        u      = element.u(x_ip)
                        phi_i  = element.phi(x_ip, i)
                        phi_j  = element.phi(x_ip, j)
                        dphi_i = element.dphi(x_ip, i)
                        dphi_j = element.dphi(x_ip, j)
                        nu     = self.nu

                        # Term 1
                        result += 1.0 / dt * phi_j * phi_i

                        # Term 2
                        result += -1.0 * u * phi_j * dphi_i

                        # Term 3
                        result += nu * dphi_j * dphi_i

                        # Add contribution to result
                        self.J[element.indices[i], element.indices[j]] += \
                            ip_weights[ip] * 0.5 * element.h * result


    def project_initial_condition(self):
        A = self.assemble_A()
        b = self.assemble_b()

        if self.left_boundary_type == "Dirichlet" and self.right_boundary_type == "Dirichlet":
            self.apply_boundary_conditions(A, b, self.initial_condition(self.nodes[0]), self.initial_condition(self.nodes[-1]))

            return np.linalg.solve(A, b)
        elif self.left_boundary_type == None and self.right_boundary_type == None:
            return self.solve_cyclic(A, b)


    def assemble_b(self, integration_points=3):
        b = np.zeros(self.number_of_nodes)

        ip_ref_locs, ip_weights = self.integration_points_and_weights(integration_points)

        # Loop over elements
        for element in self.elements:
            # Loop over local nodes
            for i in range(self.nodes_per_element):
                # Loop over integration points
                for ip in range(integration_points):
                    # Get reference location of the integration point
                    ref_loc = ip_ref_locs[ip]

                    # Calculate x location of quadrature point
                    x_ip = element.x_left + 0.5 * (1 + ref_loc) * element.h

                    # Evaluate variables at `x_ip'
                    phi_i = element.phi(x_ip, i)
                    f     = self.initial_condition(x_ip)

                    # Term 1
                    result = phi_i * f

                    # Add `result' to `vector'
                    b[element.indices[i]] += \
                        ip_weights[ip] * 0.5 * element.h * result

        return b


    def assemble_A(self, integration_points=3):
        A = np.zeros((self.number_of_nodes, self.number_of_nodes))

        ip_ref_locs, ip_weights = self.integration_points_and_weights(integration_points)

        # Loop over elements
        for element in self.elements:
            # Loop over local nodes
            for i in range(self.nodes_per_element):
                # Loop over local nodes
                for j in range(self.nodes_per_element):
                    # Loop over integration points
                    for ip in range(integration_points):
                        # Get reference location of the integration point
                        ref_loc = ip_ref_locs[ip]

                        # Calculate x location of quadrature point
                        x_ip = element.x_left + 0.5 * (1 + ref_loc) * element.h

                        # Evaluate variables at `x_ip'
                        phi_i = element.phi(x_ip, i)
                        phi_j = element.phi(x_ip, j)

                        # Term 1
                        result = phi_i * phi_j

                        # Add contribution to result
                        A[element.indices[i], element.indices[j]] += \
                            ip_weights[ip] * 0.5 * element.h * result

        return A


    def integration_points_and_weights(self, integration_points):
        if integration_points == 1:
            points = [1.0]
            weights = [2.0]
        elif integration_points == 2:
            points = [-1.0 / sqrt(3.0),
                       1.0 / sqrt(3.0)]
            weights = [1.0,
                       1.0]
        elif integration_points == 3:
            points = [-1.0 * sqrt(3.0 / 5.0),
                       0.0,
                       sqrt(3.0 / 5.0)]
            weights = [5.0 / 9.0,
                       8.0 / 9.0,
                       5.0 / 9.0]
        elif integration_points == 4:
            points = [-sqrt((15.0 + 2.0 * sqrt(30.0)) / 35.0),
                      -sqrt((15.0 - 2.0 * sqrt(30.0)) / 35.0),
                       sqrt((15.0 - 2.0 * sqrt(30.0)) / 35.0),
                       sqrt((15.0 + 2.0 * sqrt(30.0)) / 35.0)]
            weights = [(18.0 - sqrt(30.0)) / 36.0,
                       (18.0 + sqrt(30.0)) / 36.0,
                       (18.0 + sqrt(30.0)) / 36.0,
                       (18.0 - sqrt(30.0)) / 36.0]
        else:
            print("Unsupported Guassian quadrature rule, aborting.")
            sys.exit()

        return points, weights


    def u(self, x):
        x = np.atleast_1d(x)
        u = np.zeros_like(x)

        element_indices = (np.searchsorted(self.nodes, x, side="right") - 1) // self.polynomial_degree

        for i, x in enumerate(x):

            if x < self.nodes[0] or x > self.nodes[-1]:
                u[i] = 0.0
            elif x == self.nodes[-1]:
                u[i] = self.elements[-1].u(x)
            else:
                u[i] = self.elements[element_indices[i]].u(x)

        return u


    @staticmethod
    def constrain_system(matrix, vector):
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0

        matrix[-1, :] = 0.0
        matrix[-1, -1] = 1.0

        vector[0] = 0.0
        vector[-1] = 0.0


    @staticmethod
    def apply_boundary_conditions(matrix, vector, left_value, right_value):
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0

        matrix[-1, :] = 0.0
        matrix[-1, -1] = 1.0

        vector[0] = left_value
        vector[-1] = right_value


    @staticmethod
    def solve_cyclic(matrix, vector):
        matrix[0, 0] += matrix[-1, -1]
        matrix[-2, 0] += matrix[-2, -1]
        matrix[0, -2] += matrix[-1, -2]

        vector[0] += vector[-1]

        ans = np.linalg.solve(matrix[:-1, :-1], vector[:-1])

        return np.concatenate((ans, np.array([ans[0]])))
