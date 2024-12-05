"""Classes used for rendering hydra data to the responsive canvas.

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

import tkinter as tk

from archive.maths import inches2feet, is_even


SCALED_UNIT = 0.05


class Renderer:

    def __init__(self, canvas, pipes):
        """The base class for rendering data to a canvas.

        Args:
            canvas (tk.Canvas): The canvas instance on which data will be rendered.
            pipes (list[pyflo.models.Pipe]): A series of pipes.

        """
        self.canvas = canvas
        self.pipes = pipes

    @property
    def total_length(self):
        """The total summed up lengths of the pipes series."""
        return sum(pipe.length for pipe in self.pipes)
    
    @property
    def elevations(self):
        """A list of pipe and node elevations in the pipe series."""
        nodes = [pipe.node_1 for pipe in self.pipes]
        elevations_nodes = [node.elevation for node in nodes]
        elevations_invert_1 = [pipe.invert_1 for pipe in self.pipes]
        elevations_invert_2 = [pipe.invert_2 for pipe in self.pipes]
        return elevations_nodes + elevations_invert_1 + elevations_invert_2
    
    @property
    def canvas_min(self):
        """The minimum x & y coordinates for the series rendering to utilize."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        x_left = width / 10
        y_top = height / 4
        return x_left, y_top
    
    @property
    def canvas_max(self):
        """The maximum x & y coordinates for the series rendering to utilize."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        x_right = width * 9 / 10
        y_bottom = height * 3 / 4
        return x_right, y_bottom

    def get_point_unscaled(self, length, el):
        """Get the horizontal and vertical ratios of a point within a range.

        Args:
            length (float): The horizontal location along the length.
            el (float): The vertical location between the elevation range.

        Returns:
            (float, float): The x & y coordinate ratios of canvas location, respectively.

        """
        max_el = max(self.elevations)
        min_el = min(self.elevations)
        x = length / self.total_length
        y = (el - max_el) / (min_el - max_el)
        return x, y
    
    def scale_shape(self, shape):
        """Get scaled coordinates from a list of coordinate ratios and the current canvas size.

        Args:
            shape (list[float]): Coordinate ratios alternating between x & y (e.g. [x_1, y_1, x_2, y_2, ... x_n, y_n]).

        Returns:
            list[int]: Coordinates scaled to the current canvas size.

        """
        x_min, y_min = self.canvas_min
        x_max, y_max = self.canvas_max
        shape_scaled = []
        for i, coordinate in enumerate(shape):
            if is_even(i):
                coordinate_scaled = x_min + (x_max - x_min) * coordinate
            else:
                coordinate_scaled = y_min + (y_max - y_min) * coordinate
            shape_scaled.append(int(coordinate_scaled))
        return shape_scaled
    
    def scale_point(self, point):
        """Get scaled coordinates from a tuple of coordinate ratios and the current canvas size.

        Args:
            point (float, float): The x & y coordinate ratios of canvas location, respectively.

        Returns:
            (int, int): The x & y coordinates of canvas location, respectively.

        """
        x_min, y_min = self.canvas_min
        x_max, y_max = self.canvas_max
        x, y = point
        x_scaled = x_min + (x_max - x_min) * x
        y_scaled = y_min + (y_max - y_min) * y
        return int(x_scaled), int(y_scaled)

    def draw(self):
        return None


class PipesRenderer(Renderer):

    def __init__(self, canvas, pipes):
        """A class for rendering pipe data to a canvas.

        Args:
            canvas (tk.Canvas): The canvas instance on which data will be rendered.
            pipes (list[pyflo.models.Pipe]): A series of pipes.

        """
        super(PipesRenderer, self).__init__(canvas, pipes)
        
    def get_pipe_shape_unscaled(self, pipe, start_len, end_len):
        """Get a list of horizontal and vertical ratios of a pipe shape.

        Args:
            pipe (pyflo.models.Pipe): The pipe that will determine dimensions.
            start_len (float): The horizontal location of the start of the pipe.
            end_len (float): The horizontal location of the end of the pipe.

        Returns:
            list[float]: Coordinate ratios alternating between x & y (e.g. [x_1, y_1, x_2, y_2, ... x_n, y_n]).

        """
        rise = inches2feet(pipe.section.rise)
        x_sw, y_sw = self.get_point_unscaled(start_len, pipe.invert_1)
        x_se, y_se = self.get_point_unscaled(end_len, pipe.invert_2)
        x_ne, y_ne = self.get_point_unscaled(end_len, pipe.invert_2 + rise)
        x_nw, y_nw = self.get_point_unscaled(start_len, pipe.invert_1 + rise)
        points = [
            x_sw + SCALED_UNIT / 6, y_sw,
            x_se - SCALED_UNIT / 6, y_se,
            x_ne - SCALED_UNIT / 6, y_ne,
            x_nw + SCALED_UNIT / 6, y_nw,
        ]
        return points
    
    def get_node_shape_unscaled(self, pipe, start_len):
        """Get a list of horizontal and vertical ratios of a node shape.

        Args:
            pipe (pyflo.models.Pipe): The pipe that will determine dimensions.
            start_len (float): The horizontal location of the start of the pipe

        Returns:
            list[float]: Coordinate ratios alternating between x & y (e.g. [x_1, y_1, x_2, y_2, ... x_n, y_n]).

        """
        x_s, y_s = self.get_point_unscaled(start_len, pipe.invert_1)
        x_n, y_n = self.get_point_unscaled(start_len, pipe.node_1.elevation)
        points = [
            x_s - SCALED_UNIT / 6, y_s,
            x_s + SCALED_UNIT / 6, y_s,
            x_n + SCALED_UNIT / 6, y_n,
            x_n - SCALED_UNIT / 6, y_n,
        ]
        return points

    def get_node_name_point_unscaled(self, pipe, start_len):
        """Get the horizontal and vertical ratios of a node name text point.

        Args:
            pipe (pyflo.models.Pipe): The pipe that will determine dimensions.
            start_len (float): The horizontal location of the start of the pipe.

        Returns:
            (float, float): The x & y coordinate ratios of canvas location, respectively.

        """
        x, y = self.get_point_unscaled(start_len, pipe.node_1.elevation)
        y -= 1.5 * SCALED_UNIT
        return x, y

    def get_node_elevation_point_unscaled(self, pipe, start_len):
        """Get the horizontal and vertical ratios of a node elevation text point.

        Args:
            pipe (pyflo.models.Pipe): The pipe that will determine dimensions.
            start_len (float): The horizontal location of the start of the pipe.

        Returns:
            (float, float): The x & y coordinate ratios of canvas location, respectively.

        """
        x, y = self.get_point_unscaled(start_len, pipe.node_1.elevation)
        x -= 1 * SCALED_UNIT
        return x, y

    def get_pipe_size_point_unscaled(self, pipe, start_len, end_len):
        """Get the horizontal and vertical ratios of a pipe size text point.

        Args:
            pipe (pyflo.models.Pipe): The pipe that will determine dimensions.
            start_len (float): The horizontal location of the start of the pipe.
            end_len (float): The horizontal location of the end of the pipe.

        Returns:
            (float, float): The x & y coordinate ratios of canvas location, respectively.

        """
        x_sw, y_sw = self.get_point_unscaled(start_len, pipe.invert_1)
        x_ne, y_ne = self.get_point_unscaled(end_len, pipe.invert_2 + inches2feet(pipe.section.rise))
        x = (x_sw + x_ne) / 2
        y = (y_sw + y_ne) / 2
        return x, y
    
    def draw(self):
        """Render all pipe shapes and text to the canvas."""
        start_len = 0
        for pipe in self.pipes:
            end_len = start_len + pipe.length
            pipe_shape_unscaled = self.get_pipe_shape_unscaled(pipe, start_len, end_len)
            pipe_shape = self.scale_shape(pipe_shape_unscaled)
            self.canvas.create_polygon(pipe_shape, fill='', width=2, outline='black')
            node_shape_unscaled = self.get_node_shape_unscaled(pipe, start_len)
            node_shape = self.scale_shape(node_shape_unscaled)
            self.canvas.create_polygon(node_shape, fill='', width=2, outline='black')
            node_name_point_unscaled = self.get_node_name_point_unscaled(pipe, start_len)
            x_name, y_name = self.scale_point(node_name_point_unscaled)
            self.canvas.create_text(x_name, y_name, text=pipe.node_1.name, fill='black')
            node_elevation_point_unscaled = self.get_node_elevation_point_unscaled(pipe, start_len)
            x_el, y_el = self.scale_point(node_elevation_point_unscaled)
            self.canvas.create_text(x_el, y_el, text='{:.2f}'.format(pipe.node_1.elevation), fill='black')
            pipe_size_point_unscaled = self.get_pipe_size_point_unscaled(pipe, start_len, end_len)
            x_size, y_size = self.scale_point(pipe_size_point_unscaled)
            self.canvas.create_text(x_size, y_size, text='{:.0f} "'.format(pipe.section.rise), fill='black')
            start_len = end_len


class ResultsRenderer(Renderer):
    
    def __init__(self, canvas, results):
        """A class for rendering result data to a canvas.

        Args:
            canvas (tk.Canvas): The canvas instance on which data will be rendered.
            results (list[pyflo.models.Result]): A series of results.

        """
        pipes = [result.pipe for result in results]
        super(ResultsRenderer, self).__init__(canvas, pipes)
        self.results = results
    
    def get_hgl_shape_unscaled(self, elevation, start_len):
        """Get a list of horizontal and vertical ratios of an hgl shape.

        Args:
            elevation (float): the vertical location of the hgl result.
            start_len (float): The horizontal location of the start of the pipe.

        Returns:
            list[float]: Coordinate ratios alternating between x & y (e.g. [x_1, y_1, x_2, y_2, ... x_n, y_n]).

        """
        x, y = self.get_point_unscaled(start_len, elevation)
        points = [
            x, y,
            x + SCALED_UNIT / 12, y - SCALED_UNIT / 3,
            x - SCALED_UNIT / 12, y - SCALED_UNIT / 3,
        ]
        return points
    
    def get_hgl_point_unscaled(self, elevation, start_len):
        """Get the horizontal and vertical ratios of an hgl elevation text point.

        Args:
            elevation (float): the vertical location of the hgl result.
            start_len (float): The horizontal location of the start of the pipe.

        Returns:
            (float, float): The x & y coordinate ratios of canvas location, respectively.

        """
        x, y = self.get_point_unscaled(start_len, elevation)
        x += 1 * SCALED_UNIT
        return x, y
    
    def draw(self):
        """Render result pipe shapes and text to the canvas."""
        start_len = 0
        for result in self.results:
            end_len = start_len + result.pipe.length
            hgl_upper_shape_unscaled = self.get_hgl_shape_unscaled(result.hgl_upper, start_len)
            hgl_upper_shape = self.scale_shape(hgl_upper_shape_unscaled)
            self.canvas.create_polygon(hgl_upper_shape, fill='', width=2, outline='blue')
            hgl_lower_shape_unscaled = self.get_hgl_shape_unscaled(result.hgl_lower, end_len)
            hgl_lower_shape = self.scale_shape(hgl_lower_shape_unscaled)
            self.canvas.create_polygon(hgl_lower_shape, fill='', width=2, outline='blue')
            hgl_upper_point_unscaled = self.get_hgl_point_unscaled(result.hgl_upper, start_len)
            x_upper, y_upper = self.scale_point(hgl_upper_point_unscaled)
            self.canvas.create_text(x_upper, y_upper, text='{:.2f}'.format(result.hgl_upper), fill='blue')
            hgl_lower_point_unscaled = self.get_hgl_point_unscaled(result.hgl_lower, end_len)
            x_lower, y_lower = self.scale_point(hgl_lower_point_unscaled)
            self.canvas.create_text(x_lower, y_lower, text='{:.2f}'.format(result.hgl_lower), fill='blue')
            start_len = end_len
        if self.results:
            analysis = self.results[0].analysis
            if analysis.last_run:
                analysis_name = 'Analysis: ' + analysis.display_name
                node_name = 'Node: ' + analysis.node.display_name
                intensity_polynomial_name = 'Intensity Polynomial: ' + analysis.intensity_polynomial.display_name
                last_run = 'Last Run: ' + analysis.last_run.strftime('%a %b %d %Y %H:%M:%S')
                height = self.canvas.winfo_height()
                self.canvas.create_text(10, height - 80, anchor=tk.W, text=analysis_name, fill='blue')
                self.canvas.create_text(10, height - 60, anchor=tk.W, text=node_name, fill='blue')
                self.canvas.create_text(10, height - 40, anchor=tk.W, text=last_run, fill='blue')
                self.canvas.create_text(10, height - 20, anchor=tk.W, text=intensity_polynomial_name, fill='blue')
