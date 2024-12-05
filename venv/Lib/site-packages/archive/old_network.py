"""Classes for network design and simple pipe-based hydraulics.

:copyright: 2016, See AUTHORS for more details.
:license: GNU General Public License, See LICENSE for more details.

"""

import math

from archive.maths import inches2feet, goal_seek
from pyflo import build
from pyflo.constants import FULL, OPEN, K_MANNING, MAX_ITERATIONS, TOLERANCE, MAX_SOLUTION


class Pipe:

    def __init__(self, node_1, node_2, invert_1, invert_2, diameter, length, mannings, count=1):
        """A link between two nodes with hydraulic attributes, dimensions, and methods.

        Args:
            node_1 (Node):
            node_2:
            invert_1:
            invert_2:
            diameter:
            length:
            mannings:
            count:

        """
        self.node_1 = node_1
        self.node_2 = node_2
        self.invert_1 = invert_1
        self.invert_2 = invert_2
        self.diameter = diameter
        self.length = length
        self.mannings = mannings
        self.count = count

    @property
    def long_slope(self):
        return (self.invert_1 - self.invert_2) / self.length

    @property
    def node_2(self):
        return self._node_2

    @node_2.setter
    def node_2(self, value):
        if value is self.node_1:
            raise ValueError('Node 2 cannot be the same as Node 1.')
        self._node_2 = value

    def get_flow_area(self, depth):
        """Get the cross sectional area of flow, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`inches`.

        Returns:
            float: Area, in :math:`feet^2`.

        """
        depth = inches2feet(depth)
        diameter = inches2feet(self.diameter)
        alpha = math.acos(1.0 - depth / (diameter / 2.0))
        flow_area = self.count * pow(diameter, 2.0) / 4.0 * (alpha - math.sin(2.0 * alpha) / 2.0)
        return flow_area

    def get_wet_perimeter(self, depth):
        """Get the wet perimeter of flow, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`inches`

        Returns:
            float: Wet perimeter, in :math:`feet`.

        """
        depth = inches2feet(depth)
        diameter = inches2feet(self.diameter)
        alpha = math.acos(1.0 - depth / (diameter / 2.0))
        wet_perimeter = self.count * alpha * diameter
        return wet_perimeter

    def get_hyd_radius(self, depth):
        """Get the hydraulic radius of flow, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`inches`.

        Returns:
            float: Hydraulic radius, in :math:`feet`.

        """
        area = self.get_flow_area(depth)
        perimeter = self.get_wet_perimeter(depth)
        hyd_radius = area / perimeter
        return hyd_radius

    def get_velocity_open(self, depth):
        """Get the velocity of a partial flow section, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`inches`.

        Returns:
            float: Velocity, in :math:`feet/second`.

        """
        hyd_radius = self.get_hyd_radius(depth)
        velocity = K_MANNING * pow(hyd_radius, 2.0 / 3.0) * pow(self.long_slope, 0.5) / self.mannings
        return velocity

    def get_velocity_full(self, slope):
        """Get the velocity of a full flow section, given a hydraulic slope.

        Args:
            slope (float): Slope, in :math:`feet/feet`.

        Returns:
            float: Velocity, in :math:`feet/second`.

        """
        hyd_radius = self.get_hyd_radius(self.diameter)
        velocity = K_MANNING * pow(hyd_radius, 2.0 / 3.0) * pow(slope, 0.5) / self.mannings
        return velocity

    def get_hyd_flow_open(self, depth):
        """Get the  of a partial flow section, given a depth from the invert.

        Args:
            depth (float): Depth, in :math:`inches`.

        Returns:
            float: Flow, in :math:`feet^2/second`.

        """
        flow_area = self.get_flow_area(depth)
        velocity = self.get_velocity_open(depth)
        hyd_flow = flow_area * velocity
        return hyd_flow

    def get_hyd_flow_full(self, slope):
        """Get the velocity of a full flow section, given a hydraulic slope.

        Args:
            slope (float): Slope, in :math:`feet/feet`.

        Returns:
            float: Flow, in :math:`feet^2/second`.

        """
        flow_area = self.get_flow_area(self.diameter)
        velocity = self.get_velocity_full(slope)
        hyd_flow = flow_area * velocity
        return hyd_flow

    def get_flow_type(self, flow):
        flow_full_accuracy = self.get_depth_open_accuracy(self.diameter, flow=flow)
        if flow_full_accuracy <= 1.0:
            flow_type = FULL
        else:
            flow_type = OPEN
        return flow_type

    def get_friction_slope(self, depth):
        hyd_flow = self.get_hyd_flow_open(depth)
        flow_area = self.get_flow_area(depth)
        hyd_radius = self.get_hyd_radius(depth)
        a = pow(hyd_flow, 2.0) * self.mannings ** 2.0
        b = pow(K_MANNING, 2.0) * pow(flow_area, 2.0) * pow(hyd_radius, 4.0 / 3.0)
        friction_slope = a / b
        return friction_slope

    def get_friction_head(self, depth):
        friction_slope = self.get_friction_slope(depth)
        friction_head = friction_slope * self.length
        return friction_head

    def get_time_section_open(self, depth):
        velocity = self.get_velocity_open(depth)
        time_section = self.length / velocity / 60.0
        return time_section

    def get_depth_open_accuracy(self, depth, **kwargs):
        if 'flow' in kwargs:
            flow = kwargs.pop('flow')
            hyd_flow = self.get_hyd_flow_open(depth)
            basin_flow = flow * 43560.0 / 12.0 / 60.0 / 60.0
            accuracy = hyd_flow / basin_flow
            return accuracy
        else:
            raise ValueError('Need to defined "flow" in parameters in order to compare solution.')

    def solve_depth_open(self, flow):
        depth = goal_seek(
            function=self.get_depth_open_accuracy,
            bound1=TOLERANCE,
            bound2=(self.diameter - TOLERANCE),
            goal=1.0,
            max_iterations=MAX_ITERATIONS,
            tolerance=TOLERANCE,
            flow=flow
        )
        return depth

    def get_slope_full_accuracy(self, slope, **kwargs):
        if 'flow' in kwargs:
            flow = kwargs.pop('flow')
            hyd_flow = self.get_hyd_flow_full(slope)
            basin_flow = flow * 43560.0 / 12.0 / 60.0 / 60.0
            accuracy = hyd_flow / basin_flow
            return accuracy
        else:
            raise ValueError('Need to defined "flow" in parameters in order to compare solution.')

    def solve_slope_full(self, flow):
        slope = goal_seek(
            function=self.get_slope_full_accuracy,
            bound1=TOLERANCE,
            bound2=MAX_SOLUTION,
            goal=1.0,
            max_iterations=MAX_ITERATIONS,
            tolerance=TOLERANCE,
            flow=flow
        )
        return slope


class Basin:

    def __init__(self, node, area, c, tc):
        """A watershed draining to a node with hydrology attributes, dimensions, and methods.

        Args:
            node:
            area:
            c:
            tc:

        """
        self.node = node
        self.area = area
        self.c = c
        self.tc = tc

    @property
    def runoff(self):
        return self.area * self.c


class Node:

    pipe = None
    basin = None

    def __init__(self, network):
        """

        Args:
            network:

        """
        self.network = network

    def create_pipe(self, node_2, invert_1, invert_2, diameter, length, mannings, count=1):
        """Create a new :class:`Pipe` instance as a child of the node.

        Args:
            node_2:
            invert_1:
            invert_2:
            diameter:
            length:
            mannings:
            count:

        Returns:
            Pipe: The created instance.

        """
        pipe = Pipe(self, node_2, invert_1, invert_2, diameter, length, mannings, count)
        self.pipe = pipe
        return pipe
    
    def add_pipe(self, pipe):
        """Assign a :class:`Pipe` instance as a child of the node.

        Warning:
            If a pipe is currently assigned to the node, it will be overwritten.

        """
        pipe.node_1 = self
        self.pipe = pipe

    def create_basin(self, area, c, tc):
        """Create a new :class:`Basin` instance as a child of the node.

        Args:
            area (float): The delineated region concentrating to a point, in :math:`acres`.
            c (float): The runoff_area coefficient; A ratio between 0.0 and 1.0.
            tc (float): The estimated time of concentration, in :math:`minutes`.

        Returns:
            Basin: The created instance.

        """
        basin = Basin(self, area, c, tc)
        self.basin = basin
        return basin
    
    def add_basin(self, basin):
        """Assign a :class:`Basin` instance as a child of the node.

        Warning:
            If a basin is currently assigned to the node, it will be overwritten.

        """
        basin.node = self
        self.basin = basin

    def get_cumulative_runoff_data(self):
        """Get the data of cumulative runoff_area for each pipe ordered downstream to the node.

        Returns:
            list[dict]: A list of data.
            Each line of data is a dictionary in the form::

                {
                    'pipe': Pipe,
                    'area': float,
                    'runoff_area': float
                }

        """
        pipes = self.network.pipes
        pipes_ordered_down = build.pipes_ordered_down_to_node(self, pipes)
        data = []

        # Accumulate runoff_area and area, top-down
        for pipe in pipes_ordered_down:
            if pipe.node_1.basin:
                basin = pipe.node_1.basin
                area = basin.area
                runoff = basin.runoff_area
            else:
                area = 0.0
                runoff = 0.0
            if data:
                for line in data:
                    line_pipe = line['pipe']
                    if pipe.node_1 == line_pipe.node_2:
                        area += line['area']
                        runoff += line['runoff_area']
            line = {'pipe': pipe, 'area': area, 'runoff_area': runoff}
            data.append(line)
        return data


class Network:

    def __init__(self):
        """A top level container for storing network components"""
        self.nodes = []

    @property
    def pipes(self):
        return [node.pipe for node in self.nodes if node.pipe]

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
