"""Classes for network design and simple reach-based hydraulics.

:copyright: 2016, See AUTHORS for more details.
:license: GNU General Public License, See LICENSE for more details.

"""

import math

from scipy import optimize

from .constants import K_MANNING, G


MAX_ITERATIONS = 100
TOLERANCE = 1e-12
MAX_SOLUTION = 100


class Link:

    def __init__(self, **kwargs):
        self.node_1 = kwargs.pop('node_1', None)
        self.node_2 = kwargs.pop('node_2', None)

    @property
    def node_2(self):
        return self._node_2

    @node_2.setter
    def node_2(self, value):
        if value and value is self.node_1:
            raise ValueError('Node 2 cannot be the same as Node 1.')
        self._node_2 = value


class Weir(Link):

    def __init__(self, invert, section, **kwargs):
        super(Weir, self).__init__(**kwargs)
        self.section = section


class Reach(Link):

    def __init__(self, invert_1, invert_2, length, section, **kwargs):
        """A link between two nodes with hydraulic attributes, dimensions, and methods.

        Args:
            invert_1 (float): The bottom elevation at the upstream ("from") end, in :math:`feet`.
            invert_2 (float): The bottom elevation at the downstream ("to") end, in :math:`feet`.
            length (float): The total longitudinal distance, end-to-end, in :math:`feet`.
            section (hydra.sections.Circle): The cross sectional shape.
            **kwargs: Arbitrary keyword arguments.

        """
        super(Reach, self).__init__(**kwargs)
        self.invert_1 = invert_1
        self.invert_2 = invert_2
        self.length = length
        self.section = section

    @property
    def drop(self):
        return self.invert_1 - self.invert_2

    @property
    def long_slope(self):
        return self.drop / self.length

    def get_velocity(self, depth):
        """Get the velocity of a partial flow section, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`feet`.

        Returns:
            float: Velocity, in :math:`feet/second`.

        """
        hyd_radius = self.section.hyd_radius(depth)
        velocity = K_MANNING * pow(hyd_radius, 2.0 / 3.0) * pow(self.long_slope, 0.5) / self.section.n
        return velocity

    def get_hyd_flow(self, depth):
        """Get the flow of a partial flow section, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`feet`.

        Returns:
            float: Flow, in :math:`feet^2/second`.

        """
        flow_area = self.section.flow_area(depth)
        velocity = self.get_velocity(depth)
        hyd_flow = flow_area * velocity
        return hyd_flow

    def get_depth_critical_accuracy(self, depth):
        hyd_flow = self.get_hyd_flow(depth)
        flow_area = self.section.flow_area(depth)
        surface_width = self.section.surface_width(depth)
        a = pow(hyd_flow, 2.0) / G
        b = pow(flow_area, 3.0) / surface_width
        # return a / b
        return a - b

    def solve_depth_critical(self):
        for i in range(MAX_ITERATIONS):
            depth_trial = i * MAX_SOLUTION
            if self.get_depth_critical_accuracy(depth_trial) > 1.0:
                # depth = goal_seek(
                #     function=self.critical_depth_accuracy,
                #     bounds=(TOLERANCE, depth_trial),
                #     goal=1.0,
                #     max_iterations=MAX_ITERATIONS,
                #     tolerance=TOLERANCE
                # )
                depth = optimize.bisect(
                    f=self.get_depth_critical_accuracy,
                    a=TOLERANCE,
                    b=depth_trial
                )
                return float(depth)
        raise Exception('Maximum iterations reached while trying to find an upper bound')

    def solve_velocity_critical(self):
        depth_critical = self.solve_depth_critical()
        return math.sqrt(G * depth_critical)

    def solve_slope_critical(self):
        depth_critical = self.solve_depth_critical()
        velocity_critical = self.solve_velocity_critical()
        hyd_radius = self.section.hyd_radius(depth_critical)
        a = velocity_critical * self.section.n
        b = K_MANNING * pow(hyd_radius, 2.0 / 3.0)
        return pow(a / b, 2.0)

    def get_friction_slope(self, depth, flow):
        """Get the water surface grade, based on frictional properties.

        Args:
            depth (float): Depth, in :math:`feet`.
            flow (float): hydrology flow in :math:`feet^2/second`.

        Returns:
            float: The slope of the water surface profile, in :math:`feet/feet`.

        """
        flow_area = self.section.flow_area(depth)
        velocity = flow / flow_area
        hyd_radius = self.section.hyd_radius(depth)
        a = pow(velocity, 2.0) * pow(self.section.n, 2.0)
        b = pow(K_MANNING, 2.0) * pow(hyd_radius, 4.0 / 3.0)
        if b <= 0.0:
            return 0.0
        return a / b

    def get_friction_head(self, depth, flow):
        """Get the water surface elevation difference, based on frictional properties.

        Args:
            depth (float): Depth, in :math:`feet`.
            flow (float): hydrology flow in :math:`feet^2/second`.

        Returns:
            float: The difference in elevation, in :math:`feet`.

        """
        friction_slope = self.get_friction_slope(depth, flow)
        return friction_slope * self.length

    def get_time_section(self, depth, flow):
        """Get the travel time of water from end-to-end.

        Args:
            depth (float): Depth, in :math:`feet`.
            flow (float): hydrology flow in :math:`feet^2/second`.

        Returns:
            float: Travel time, in :math:`minutes`.

        """
        flow_area = self.section.flow_area(depth)
        velocity = flow / flow_area
        return self.length / velocity / 60.0

    def get_depth_normal_accuracy(self, depth, flow):
        """Check solution convergence in a open flow case.

        Args:
            depth (float): The guessed value of the depth, in :math:`feet`.
            flow (float): hydrology flow in :math:`feet^2/second`.

        Returns:
            float: A ratio of hydraulic to hydrology flow.

        Raises:
            ValueError: If a "flow" key is not defined.

        """
        hyd_flow = self.get_hyd_flow(depth)
        #return hyd_flow / flow
        return hyd_flow - flow

    def solve_depth_normal(self, flow):
        """Goal seek a the depth in a open flow case.

        Args:
            flow (float): In :math:`feet^2/second`.

        Returns:
            float: Depth, in :math:`feet`.

        Note:
            The goal is to find a 1:1 ratio of hydraulic to hydrology flow.

        """

        if self.section.rise:
            hyd_flow = self.get_hyd_flow(self.section.rise)
            if flow / hyd_flow > 1.0:
                return self.section.rise
        for i in range(1, MAX_ITERATIONS):
            depth_trial = i * MAX_SOLUTION
            if self.get_depth_normal_accuracy(depth_trial, flow) > 1.0:
                # depth = goal_seek(
                #     function=self.normal_depth_accuracy,
                #     bounds=(TOLERANCE, depth_trial),
                #     goal=0.0,
                #     max_iterations=MAX_ITERATIONS,
                #     tolerance=TOLERANCE,
                #     func_args=(flow,)
                # )
                depth = optimize.bisect(
                    f=self.get_depth_normal_accuracy,
                    a=TOLERANCE,
                    b=depth_trial,
                    args=(flow,)
                )
                return float(depth)
        raise Exception('Maximum iterations reached while trying to find an upper bound')

    def get_velocity_head(self, depth, flow):
        flow_area = self.section.flow_area(depth)
        velocity = flow / flow_area
        return pow(velocity, 2.0) / 2.0 / G

    def get_hgl_lower(self, tw, flow):
        hgl_lower_cond_1 = tw
        hgl_lower_cond_2 = self.invert_2 + self.solve_depth_normal(flow)
        return max(hgl_lower_cond_1, hgl_lower_cond_2)

    def get_hgl_upper(self, hgl_lower, flow):
        hgl_upper_cond_1 = self.invert_1 + self.solve_depth_normal(flow)
        hgl_upper_cond_2 = self.invert_1 + self.solve_hw_depth(hgl_lower, flow)
        return max(hgl_upper_cond_1, hgl_upper_cond_2)

    def get_hw_depth_accuracy(self, depth, hgl_lower, flow):
        y_2 = hgl_lower - self.invert_2
        y_1 = depth
        h_vel1 = self.get_velocity_head(y_1, flow)
        h_vel2 = self.get_velocity_head(y_2, flow)
        z_1 = self.invert_1
        z_2 = self.invert_2
        s_f1 = self.get_friction_slope(y_1, flow)
        s_f2 = self.get_friction_slope(y_2, flow)
        s_f = (s_f1 + s_f2) / 2.0
        h_f = self.length * s_f
        h_2 = z_2 + y_2 + h_vel2 + h_f
        h_1 = z_1 + y_1 + h_vel1
        # return h_1 / h_2
        return h_1 - h_2

    def solve_hw_depth(self, hgl_lower, flow):
        for i in range(1, MAX_ITERATIONS):
            depth_trial = i * MAX_SOLUTION
            if self.get_hw_depth_accuracy(depth_trial, hgl_lower, flow) > 1.0:
                # depth = goal_seek(
                #     function=self.get_depth_steady_state_accuracy,
                #     bounds=(TOLERANCE, depth_trial),
                #     goal=1.0,
                #     max_iterations=MAX_ITERATIONS,
                #     tolerance=TOLERANCE,
                #     func_args=(hgl_2, flow)
                # )
                depth = optimize.bisect(
                    f=self.get_hw_depth_accuracy,
                    a=TOLERANCE,
                    b=depth_trial,
                    args=(hgl_lower, flow)
                )
                return float(depth)
        raise Exception('Maximum iterations reached while trying to find an upper bound')


class Node:

    def __init__(self, network):
        """A point where a :class:`Basin` or :class:`Reach` can be assigned.

        Args:
            network (Network): The parent.

        """
        self.network = network
        self.reach = None
        self.basin = None

    def create_reach(self, node_2, invert_1, invert_2, length, section):
        """Create a new :class:`Reach` and assign the node to the upstream end.

        Args:
            node_2 (Node): The node to link at the downstream ("to") end.
            invert_1 (float): The reach bottom elevation at the upstream ("from") end, in :math:`feet`.
            invert_2 (float): The reach bottom elevation at the downstream ("to") end, in :math:`feet`.
            length (float): The total longitudinal distance, end-to-end, in :math:`feet`.
            section (hydra.sections.Circle): The cross sectional shape.

        Returns:
            Reach: The created instance.

        """
        reach = Reach(invert_1, invert_2, length, section, node_1=self, node_2=node_2)
        self.reach = reach
        return reach
    
    def add_reach(self, reach):
        """Assign the node to the upstream end of the :class:`Reach`.

        Warning:
            If a reach is currently assigned to the node, it will be overwritten.

        """
        reach.node_1 = self
        self.reach = reach
    
    def add_basin(self, basin):
        """Assign a :class:`Basin` instance as a child of the node.

        Warning:
            If a basin is currently assigned to the node, it will be overwritten.

        """
        self.basin = basin


class Network:

    def __init__(self):
        """A top level container for storing network components"""
        self.nodes = []

    @property
    def reaches(self):
        return [node.reach for node in self.nodes if node.reach]

    @property
    def basins(self):
        return [node.basin for node in self.nodes if node.basin]

    def create_node(self):
        """Create a new :class:`Node` instance and house it as a child of the network.

        Returns:
            Node: The created instance.

        """
        node = Node(self)
        self.nodes.append(node)
        return node

    def add_node(self, node):
        """Add a :class:`Node` instance to the network and house it as a child of the network."""
        node.network = self
        self.nodes.append(node)
