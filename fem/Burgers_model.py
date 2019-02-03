import sys
import numpy as np
from math import sqrt

from fem.linear_element import LinearElement

class BurgersModel(object):

    def __init__(self,
                 number_of_elements,
                 nodes,
                 nu,
                 initial_condition,
                 forcing_function,
                 left_boundary_value=None,
                 right_boundary_value=None,
                 use_nodal_projection=False,
                 integration_points=3,
                 newton_iterations=100,
                 newton_tolerance=1e-6,
                 stab_type=None):

        self.number_of_elements = number_of_elements
        self.nodes = nodes
        self.nu = nu
        self.forcing_function = forcing_function
        self.left_boundary_value = left_boundary_value
        self.right_boundary_value = right_boundary_value
        self.initial_condition = initial_condition
        self.use_nodal_projection = use_nodal_projection
        self.integration_points = integration_points
        self.newton_iterations = newton_iterations
        self.newton_tolerance = newton_tolerance
        self.stab_type = stab_type

        self.number_of_nodes = nodes.size
        self.nodes_per_element = 2

        self.coefficients = np.empty(self.number_of_nodes)
        self.previous_coefficients = np.empty(self.number_of_nodes)

        # Check boundary conditions
        if self.left_boundary_value and self.right_boundary_value:
            print()
            print("Boundary conditions: Dirichlet")
        else:
            print()
            print("Boundary conditions: periodic")

        # Store `Element' objects in `self.elements'
        self.elements = self.create_elements(number_of_elements)

        # Set time variable
        self.time = 0.0

        # Preallocate Jacobian matrix and residual vector
        self.J = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.F = np.zeros(self.number_of_nodes)

        # Project the `initial_condition' function onto the finite element space
        if use_nodal_projection:
            np.copyto(self.coefficients, self.interpolate_initial_condition())

            print()
            print("Projection type for initial_condition: nodal")
        else:
            np.copyto(self.coefficients, self.project_initial_condition())

        print()


    def create_elements(self, number_of_elements):
        print()
        print("Number of elements: {}".format(number_of_elements))

        elements = []

        for i in range(self.number_of_elements):
            local_nodes = [self.nodes[i], self.nodes[i+1]]
            indices = [i, i+1]

            elements.append(LinearElement(i, local_nodes, indices,
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
        if self.left_boundary_value and self.right_boundary_value:
            self.coefficients[0] = self.left_boundary_value(self.time)
            self.coefficients[-1] = self.right_boundary_value(self.time)
        # Starting guess is solution from previous timestep

        # Newton's method
        for k in range(self.newton_iterations):
            # Assemble the residual `-F' and the Jacobian `J'
            self.assemble_F()
            self.assemble_J()

            # Solve the system and update `coefficients'
            if self.left_boundary_value and self.right_boundary_value:
                # Modify the system to enforce the boundary conditions
                self.constrain_system(self.J, self.F)

                delta_coefficients = np.linalg.solve(self.J, -self.F)
            else:
                delta_coefficients = self.solve_cyclic(self.J, -self.F)

            self.coefficients += delta_coefficients

            # Check for convergence
            if np.linalg.norm(delta_coefficients) < self.newton_tolerance:
                print('    Newton\'s method converged after {} iterations'.format(k+1))
                break

        if k == self.newton_iterations - 1:
            print('    Newton\'s method FAILED to converge within {} iterations'.format(k+1))

        #self.energy_balance()


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
                    u_prev = element.previous_u(x_ip)
                    du_dx  = element.du(x_ip)
                    du_dx_prev = element.previous_du(x_ip)
                    du_dt  = (u - u_prev) / dt
                    phi_i  = element.phi(x_ip, i)
                    dphi_i = element.dphi(x_ip, i)
                    f      = self.forcing_function(x_ip, self.time)
                    f_prev = self.forcing_function(x_ip, self.previous_time)
                    nu     = self.nu

                    # The value of the integrand at `ip_x' will be stored as `result'
                    result = 0.0

                    # Second-order timestep
                    
                    # Term 1
                    #result += du_dt * phi_i

                    # Term 2
                    #result += -0.5 * u * u * dphi_i

                    # Term 3
                    #result += nu * du_dx * dphi_i

                    # Term 4
                    #result += -1.0 * f * phi_i

                    # Third-order timestep (Crank-Nicolson)

                    # Term 1
                    result += du_dt * phi_i

                    # Term 2
                    result += -0.5 * u * u * dphi_i * 0.5

                    # Term 3
                    result += nu * du_dx * dphi_i * 0.5

                    # Term 4
                    result += -1.0 * f * phi_i * 0.5

                    # Term 5
                    result += -0.5 * u_prev * u_prev * dphi_i * 0.5

                    # Term 6
                    result += nu * du_dx_prev * dphi_i * 0.5

                    # Term 7
                    result += -1.0 * f_prev * phi_i * 0.5


                    # Shakib's stabilization
                    if self.stab_type == "Shakib":
                        h2 = element.h * element.h
                        h4 = h2 * h2
                        tau = 1.0 / (sqrt(4.0 * u * u / h2 + 16.0 * self.nu * self.nu / h4))

                        residual = du_dt + u * du_dx - f;

                        uprime = -tau * residual

                        result += -1.0 * u * uprime * dphi_i

                        result += -0.5 * uprime * uprime * dphi_i

                    # Add `result' to `vector'
                    self.F[element.indices[i]] += \
                        ip_weights[ip] * 0.5 * element.h * result


    def energy_balance(self, integration_points=3):
        dt = self.time - self.previous_time

        ip_ref_locs, ip_weights = self.integration_points_and_weights(integration_points)

        kinetic_energy = 0.0
        kinetic_energy_change = 0.0
        net_flux = 0.0
        viscous_work = 0.0
        forcing_work = 0.0
        boundary_terms = 0.0

        # Loop over elements
        for element in self.elements:
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
                f      = self.forcing_function(x_ip, self.time)
                nu     = self.nu

                kinetic_energy += \
                    ip_weights[ip] * 0.5 * element.h * 0.5 * u * u

                kinetic_energy_change += \
                    ip_weights[ip] * 0.5 * element.h * u * du_dt

                net_flux -= \
                    ip_weights[ip] * 0.5 * element.h * 0.5 * du_dx * u * u

                viscous_work -= \
                    ip_weights[ip] * 0.5 * element.h * nu * du_dx * du_dx

                forcing_work += \
                    ip_weights[ip] * 0.5 * element.h * u * f

            x_l = element.x_left
            x_r = element.x_right

            boundary_terms += 0.5 * element.u(x_r) * element.u(x_r) * element.u(x_r)
            boundary_terms += -0.5 * element.u(x_l) * element.u(x_l) * element.u(x_l)

            #boundary_terms += self.nu * element.u(x_r) * element.du(x_r)
            #boundary_terms += -1.0 * self.nu * element.u(x_l) * element.du(x_l)

        balance = net_flux + viscous_work + forcing_work

        print()
        print("    Kinetic energy:        {:.4f}".format(kinetic_energy))
        print()
        print("    Kinetic energy change: {:.4f}".format(kinetic_energy_change))
        print("    Net flux:              {:.4f}".format(net_flux))
        print("    Viscous work:          {:.4f}".format(viscous_work))
        print("    Forcing work:          {:.4f}".format(forcing_work))
        print()
        print("    Balance:               {:.4f}".format(kinetic_energy_change - balance))
        print()

        #print("    {:.4f}".format(boundary_terms))

        #print()
        #print("    {:.4f}".format(0.5 * self.elements[-1].u(1.0) * self.elements[-1].u(1.0) * self.elements[-1].u(1.0)))
        #print("    {:.4f}".format(-0.5 * self.elements[0].u(0.0) * self.elements[0].u(0.0) * self.elements[0].u(0.0)))
        #print()

        #print()
        #print("    {:.4f}".format(self.nu * self.elements[-1].u(1.0) * self.elements[-1].du(1.0)))
        #print("    {:.4f}".format(-1.0 * self.nu * self.elements[0].u(0.0) * self.elements[0].du(0.0)))
        #print()


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

                        if self.stab_type == "Shakib": # Shakib's stabilization
                            du_dx = element.du(x_ip)

                            h2 = element.h * element.h
                            h4 = h2 * h2
                            tau = 1.0 / (sqrt(4.0 * u * u / h2 + 16.0 * self.nu * self.nu / h4))

                            result += 2.0 * tau * u * du_dx * dphi_i * phi_j

                            result += tau * u * u * dphi_i * dphi_j

                            result += -1.0 * tau * dphi_i * phi_j

                        # Add contribution to result
                        self.J[element.indices[i], element.indices[j]] += \
                            ip_weights[ip] * 0.5 * element.h * result


    def interpolate_initial_condition(self):
        return self.initial_condition(self.nodes)


    def project_initial_condition(self):
        A = self.assemble_A()
        b = self.assemble_b()

        if self.left_boundary_value and self.right_boundary_value:
            self.apply_boundary_conditions(A, b, self.left_boundary_value(self.time), self.right_boundary_value(self.time))

            return np.linalg.solve(A, b)
        else:
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
