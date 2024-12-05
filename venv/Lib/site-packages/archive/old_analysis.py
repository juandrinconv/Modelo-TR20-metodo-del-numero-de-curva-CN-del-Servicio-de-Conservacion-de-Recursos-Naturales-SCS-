"""Functions for performing hydraulic analysis.

:copyright: 2016, See AUTHORS for more details.
:license: GNU General Public License, See LICENSE for more details.

"""

from archive.maths import inches2feet
from pyflo.constants import FULL, OPEN


def solve_hgl_el_solution_data(analysis):
    """Runs an HGL analysis.

    Args:
        analysis (pyflo.models.Analysis): The object that packages the information to run an HGL analysis.

    Returns:
        list[dict]: A list of data.
        Each line is a dictionary in the form::

            {
                'pipe': pyflo.models.Pipe,
                'basin': pyflo.models.Basin,
                'area': float,
                'runoff_area': float,
                'second': float,
                'time': float,
                'flow': float,
                'velocity_actual': float,
                'velocity_physical': float,
                'hgl_1': float,
                'hgl_2': float,
            }

    """
    data = analysis.node.get_cumulative_runoff_data()
    basins = [line['basin'] for line in data]
    data[-1]['hgl_2'] = analysis.tw

    # Accumulate travel times, top-down (assuming physical slope)
    for line in data:
        time_upper = 0.0
        pipe = line['pipe']
        for basin in basins:
            if basin.node == pipe.node_1:
                time_upper = basin.tc
                break
        for r in data:
            line_pipe = r['pipe']
            if pipe.node_1 == line_pipe.node_2 and 'time' in r and type(r['time']) is float:
                time_upper = max(time_upper, r['time'])
        time_section = pipe.get_time_section_open(pipe.diameter)
        line['time'] = time_upper + time_section
        line['velocity_physical'] = pipe.get_velocity_open(pipe.diameter)

    # Trace back HGL, bottom-up
    reversed_data = reversed(data)
    for line in reversed_data:
        pipe = line['pipe']
        hgl_lower_cond_1 = analysis.tw
        for r in data:
            line_pipe = r['pipe']
            if pipe.node_2 == line_pipe.node_1 and 'hgl_1' in r and type(r['hgl_1']) is float:
                hgl_lower_cond_1 = max(hgl_lower_cond_1, r['hgl_1'])
        full_flow_friction_slope = pipe.friction_slope(pipe.diameter)
        time = line['time']
        intensity = analysis.intensity_polynomial.get_y(time)
        line['second'] = intensity
        runoff = line['runoff_area']
        flow = intensity * runoff
        if hgl_lower_cond_1 > pipe.invert_2 + inches2feet(pipe.diameter):
            hgl_lower_cond_2 = pipe.invert_2 + inches2feet(pipe.diameter)
        elif pipe.slope > full_flow_friction_slope:
            depth = pipe.solve_depth_open(flow)
            hgl_lower_cond_2 = pipe.invert_2 + inches2feet(depth)
        else:
            hgl_lower_cond_2 = pipe.invert_2 + inches2feet(pipe.diameter)
        hgl_lower = max(hgl_lower_cond_1, hgl_lower_cond_2)
        full_flow_friction_head = pipe.friction_loss(pipe.diameter)
        hgl_upper_cond_1 = hgl_lower + full_flow_friction_head
        flow_type = pipe.get_flow_type(flow)
        if flow_type == OPEN:
            depth = pipe.solve_depth_open(flow)
            velocity_actual = pipe.get_velocity_open(depth)
            flow = pipe.get_hyd_flow_open(flow)
            hgl_upper_cond_2 = pipe.invert_1 + inches2feet(depth)
        elif flow_type == FULL:
            slope = pipe.solve_slope_full(flow)
            velocity_actual = pipe.get_velocity_full(slope)
            flow = pipe.get_hyd_flow_full(slope)
            hgl_upper_cond_2 = hgl_lower + pipe.length * slope
        else:
            raise Exception('Invalid pipe detected.')
        hgl_upper = max(hgl_upper_cond_1, hgl_upper_cond_2)
        line['hgl_2'] = hgl_lower
        line['hgl_1'] = hgl_upper
        line['flow'] = flow
        line['velocity_actual'] = velocity_actual

    return data
