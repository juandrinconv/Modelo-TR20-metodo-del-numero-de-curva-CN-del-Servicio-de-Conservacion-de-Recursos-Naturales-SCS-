"""Functions used for calculating coordinates for rendering on the responsive canvas.

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

from archive.maths import inches2feet
from pyflo.constants import SCALED_UNIT


def point_unscaled(length, el, min_el, max_el, total_len):
    """Get the horizontal and vertical ratios of a point within a range.

    Args:
        length (float): The desired horizontal coordinate along the length.
        el (float): The desired vertical coordinate between the elevation range.
        min_el (float): The maximum elevation to interpolate between.
        max_el (float): The minimum elevation to interpolate between.
        total_len (float): The total length to interpolate between.

    Returns:
        (float, float): The x and y ratios, respectively.

    """
    x = length / total_len
    y = (el - max_el) / (min_el - max_el)
    return x, y


def scale_shapes(shapes_unscaled, x_min, y_min, x_max, y_max):
    shapes_scaled = []
    for shape in shapes_unscaled:
        shape_scaled = []
        for i, coordinate in enumerate(shape):
            if i % 2 == 0:  # Even
                coordinate_scaled = x_min + (x_max - x_min) * coordinate
            else:           # Odd
                coordinate_scaled = y_min + (y_max - y_min) * coordinate
            shape_scaled.append(int(coordinate_scaled))
        shapes_scaled.append(shape_scaled)
    return shapes_scaled


def scale_text_positions(positions_unscaled, x_min, y_min, x_max, y_max):
    positions_scaled = []
    for position in positions_unscaled:
        x_scaled = x_min + (x_max - x_min) * position[0]
        y_scaled = y_min + (y_max - y_min) * position[1]
        position_scaled = (int(x_scaled), int(y_scaled), position[2])
        positions_scaled.append(position_scaled)
    return positions_scaled


def get_pipe_vars_unscaled(pipes):
    nodes = [pipe.node_1 for pipe in pipes]
    elevations_nodes = [node.elevation for node in nodes]
    elevations_invert_1 = [pipe.invert_1 for pipe in pipes]
    elevations_invert_2 = [pipe.invert_2 for pipe in pipes]
    elevations = elevations_nodes + elevations_invert_1 + elevations_invert_2
    length_total = sum(pipe.length for pipe in pipes)
    max_elevation = max(elevations)
    min_elevation = min(elevations)
    return min_elevation, max_elevation, length_total


def get_result_vars_unscaled(results):
    nodes = [result.pipe.node_1 for result in results]
    elevations_nodes = [node.elevation for node in nodes]
    elevations_invert_1 = [result.pipe.invert_1 for result in results]
    elevations_invert_2 = [result.pipe.invert_2 for result in results]
    elevations = elevations_nodes + elevations_invert_1 + elevations_invert_2
    length_total = sum(result.pipe.length for result in results)
    max_elevation = max(elevations)
    min_elevation = min(elevations)
    return min_elevation, max_elevation, length_total


def pipe_shape_unscaled(pipe, min_el, max_el, start_len, end_len, total_len):
    diameter = inches2feet(pipe.diameter)
    point_sw = point_unscaled(start_len, pipe.invert_1, min_el, max_el, total_len)
    point_se = point_unscaled(end_len, pipe.invert_2, min_el, max_el, total_len)
    point_ne = point_unscaled(end_len, pipe.invert_2 + diameter, min_el, max_el, total_len)
    point_nw = point_unscaled(start_len, pipe.invert_1 + diameter, min_el, max_el, total_len)
    points = [
        point_sw[0] + SCALED_UNIT / 6, point_sw[1],
        point_se[0] - SCALED_UNIT / 6, point_se[1],
        point_ne[0] - SCALED_UNIT / 6, point_ne[1],
        point_nw[0] + SCALED_UNIT / 6, point_nw[1],
    ]
    return points


def node_shape_unscaled(pipe, min_el, max_el, start_len, end_len, total_len):
    point_s = point_unscaled(start_len, pipe.invert_1, min_el, max_el, total_len)
    point_n = point_unscaled(start_len, pipe.node_1.elevation, min_el, max_el, total_len)
    points = [
        point_s[0] - SCALED_UNIT / 6, point_s[1],
        point_s[0] + SCALED_UNIT / 6, point_s[1],
        point_n[0] + SCALED_UNIT / 6, point_n[1],
        point_n[0] - SCALED_UNIT / 6, point_n[1],
    ]
    return points


def node_name_text_unscaled(pipe, min_el, max_el, start_len, end_len, total_len):
    """Get text and unscaled coordinate data necessary to render node name text to the canvas.

    Args:
        pipe (pyflo.models.Pipe): A pipe object.

    Returns:
        list[float, float, str]: A list of tuples containing x-ratio, y-ratio, and string.

    """
    point = point_unscaled(start_len, pipe.node_1.elevation, min_el, max_el, total_len)
    x = point[0]
    y = point[1] - 1.5 * SCALED_UNIT
    text = pipe.node_1.name
    return x, y, text


def node_elevation_text_unscaled(pipe, min_el, max_el, start_len, end_len, total_len):
    """Get text and unscaled coordinate data necessary to render node elevation text to the canvas.

    Args:
        pipe (pyflo.models.Pipe): A pipe object.

    Returns:
        list[float, float, str]: A list of tuples containing x-ratio, y-ratio, and string.

    """
    point = point_unscaled(start_len, pipe.node_1.elevation, min_el, max_el, total_len)
    x = point[0] - 1 * SCALED_UNIT
    y = point[1]
    text = '{:.2f}'.format(pipe.node_1.elevation)
    return x, y, text


def pipe_size_text_unscaled(pipe, min_el, max_el, start_len, end_len, total_len):
    """Get text and unscaled coordinate data necessary to render pipe size text to the canvas.

    Args:
        pipe (pyflo.models.Pipe): A pipe object.

    Returns:
        list[float, float, str]: A list of tuples containing x-ratio, y-ratio, and string.

    """
    point_sw = point_unscaled(start_len, pipe.invert_1, min_el, max_el, total_len)
    point_ne = point_unscaled(end_len, pipe.invert_2 + inches2feet(pipe.diameter), min_el, max_el, total_len)
    x = (point_sw[0] + point_ne[0]) / 2
    y = (point_sw[1] + point_ne[1]) / 2
    text = '{:.0f} "'.format(pipe.diameter)
    return x, y, text


def pipe_shapes_unscaled(pipes):
    min_el, max_el, total_len = get_pipe_vars_unscaled(pipes)
    shapes = []
    start_len = 0
    for pipe in pipes:
        end_len = start_len + pipe.length
        pipe_shape = pipe_shape_unscaled(pipe, min_el, max_el, start_len, end_len, total_len)
        shapes.append(pipe_shape)
        node_shape = node_shape_unscaled(pipe, min_el, max_el, start_len, end_len, total_len)
        shapes.append(node_shape)
        start_len = end_len
    return shapes


def pipe_text_unscaled(pipes):
    """Get text and unscaled coordinate data necessary to render all pipe text to the canvas.

    Args:
        pipes (list[pyflo.models.Pipe]): A series of pipes.

    Returns:
        list[float, float, str]: A list of tuples containing x-ratio, y-ratio, and string.

    """
    min_el, max_el, total_len = get_pipe_vars_unscaled(pipes)
    text_positions = []
    start_len = 0
    for pipe in pipes:
        end_len = start_len + pipe.length
        node_name_text = node_name_text_unscaled(pipe, min_el, max_el, start_len, end_len, total_len)
        text_positions.append(node_name_text)
        node_elevation_text = node_elevation_text_unscaled(pipe, min_el, max_el, start_len, end_len, total_len)
        text_positions.append(node_elevation_text)
        pipe_size_text = pipe_size_text_unscaled(pipe, min_el, max_el, start_len, end_len, total_len)
        text_positions.append(pipe_size_text)
        start_len = end_len
    return text_positions


def pipe_shapes_scaled(pipes, x_min, y_min, x_max, y_max):
    shapes_unscaled = pipe_shapes_unscaled(pipes)
    shapes_scaled = scale_shapes(shapes_unscaled, x_min, y_min, x_max, y_max)
    return shapes_scaled


def pipe_text_positions_scaled(pipes, x_min, y_min, x_max, y_max):
    text_unscaled = pipe_text_unscaled(pipes)
    text_scaled = scale_text_positions(text_unscaled, x_min, y_min, x_max, y_max)
    return text_scaled


def result_upper_hgl_shape_unscaled(result, min_el, max_el, start_len, end_len, total_len):
    point = point_unscaled(start_len, result.hgl_1, min_el, max_el, total_len)
    points = [
        point[0], point[1],
        point[0] + SCALED_UNIT / 12, point[1] - SCALED_UNIT / 3,
        point[0] - SCALED_UNIT / 12, point[1] - SCALED_UNIT / 3,
    ]
    return points


def result_lower_hgl_shape_unscaled(result, min_el, max_el, start_len, end_len, total_len):
    point = point_unscaled(end_len, result.hgl_2, min_el, max_el, total_len)
    points = [
        point[0], point[1],
        point[0] + SCALED_UNIT / 12, point[1] - SCALED_UNIT / 3,
        point[0] - SCALED_UNIT / 12, point[1] - SCALED_UNIT / 3,
    ]
    return points


def result_shapes_unscaled(results):
    min_el, max_el, total_len = get_result_vars_unscaled(results)
    shapes = []
    start_len = 0
    for result in results:
        end_len = start_len + result.pipe.length
        hgl_upper_shape = result_upper_hgl_shape_unscaled(result, min_el, max_el, start_len, end_len, total_len)
        shapes.append(hgl_upper_shape)
        hgl_lower_shape = result_lower_hgl_shape_unscaled(result, min_el, max_el, start_len, end_len, total_len)
        shapes.append(hgl_lower_shape)
        start_len = end_len
    return shapes


def result_hgl_upper_text_unscaled(result, min_el, max_el, start_len, end_len, total_len):
    point = point_unscaled(start_len, result.hgl_1, min_el, max_el, total_len)
    x = point[0] + 1 * SCALED_UNIT
    y = point[1]
    text = '{:.2f}'.format(result.hgl_1)
    return x, y, text


def result_hgl_lower_text_unscaled(result, min_el, max_el, start_len, end_len, total_len):
    point = point_unscaled(end_len, result.hgl_2, min_el, max_el, total_len)
    x = point[0] + 1 * SCALED_UNIT
    y = point[1]
    text = '{:.2f}'.format(result.hgl_2)
    return x, y, text


def result_text_unscaled(results):
    min_el, max_el, total_len = get_result_vars_unscaled(results)
    text_positions = []
    start_len = 0
    for result in results:
        end_len = start_len + result.pipe.length
        hgl_upper_text = result_hgl_upper_text_unscaled(result, min_el, max_el, start_len, end_len, total_len)
        text_positions.append(hgl_upper_text)
        hgl_lower_text = result_hgl_lower_text_unscaled(result, min_el, max_el, start_len, end_len, total_len)
        text_positions.append(hgl_lower_text)
        start_len = end_len
    return text_positions


def result_shapes_scaled(results, x_min, y_min, x_max, y_max):
    shapes_unscaled = result_shapes_unscaled(results)
    shapes_scaled = scale_shapes(shapes_unscaled, x_min, y_min, x_max, y_max)
    return shapes_scaled


def result_text_positions_scaled(results, x_min, y_min, x_max, y_max):
    text_unscaled = result_text_unscaled(results)
    text_scaled = scale_text_positions(text_unscaled, x_min, y_min, x_max, y_max)
    return text_scaled
