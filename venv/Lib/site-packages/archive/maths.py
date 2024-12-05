"""Some useful Hydra math-based functions.

:copyright: 2016, See AUTHORS.md for more details.
:license: GNU General Public License, See LICENSE for more details.

"""


def is_even(value):
    """Lets you know if a number is even

    Args:
        value (int | float):

    Returns:
        bool: True if number even, otherwise False

    """
    if value % 2 == 0:  # Even
        return True
    return False        # Odd


def goal_seek(function, bounds, goal, max_iterations, tolerance, func_args=None):
    """Find a function's input that gives a desired output.

    Args:
        function: A function with at least one argument.
        bounds ((float, float)): A pair of initial inputs. One causes the output to solve too low, the other too high.
        goal (float): The desired number for the function's output.
        max_iterations (int): The maximum number of attempted function calls before failure.
        tolerance (float): The maximum difference from the output and goal for a sufficient solution.
        func_args: Additional arguments to pass to the function.

    Note:
        Depreciated; Use :func:`scipy.optimize.bisect` or equivalent solver instead.

    Returns:
        float: The estimated function input.

    Raises:
        Exception: If the defined maximum iterations are reached before finding a sufficient solution.
        ValueError: If one defined bound doesn't cause a solution to be too low, the other too high.

    """
    a, b = bounds
    if func_args:
        solve_a = function(a, *func_args)
        solve_b = function(b, *func_args)
    else:
        solve_a = function(a)
        solve_b = function(b)
    if min(solve_a, solve_b) < goal < max(solve_a, solve_b):
        for i in range(max_iterations):
            c = (a + b) / 2.0
            if func_args:
                solve_a = function(a, *func_args)
                solve_c = function(c, *func_args)
            else:
                solve_a = function(a)
                solve_c = function(c)
            y_a = (solve_a / goal) - 1.0
            y_c = (solve_c / goal) - 1.0
            if abs(b - a) / 2.0 < tolerance:  # Solution Found
                return c
            elif y_a * y_c < 0.0:
                b = c
            else:
                a = c  # New Interval
        raise Exception('Maximum iterations reached while trying to find a solution')
    raise ValueError('One bound has to cause a solution to be too low, the other too high.')


def interpolate(g1, g2, g, d1, d2):
    d = d1 + (d2 - d1) * ((g - g1) / (g2 - g1))
    return d


def interpolate_from_table(value, table, g_col, d_col):
    min_tuple = table[0]
    max_tuple = table[-1]
    if min_tuple[g_col] < value < max_tuple[g_col]:
        index = next(i for i, v in enumerate(table) if (v[g_col] > value))
        g1 = table[index - 1][g_col]
        g2 = table[index][g_col]
        g = value
        d1 = table[index - 1][d_col]
        d2 = table[index][d_col]
        interpolated_value = interpolate(g1, g2, g, d1, d2)
    elif value <= min_tuple[g_col]:
        interpolated_value = min_tuple[d_col]
    else:
        interpolated_value = max_tuple[d_col]
    return interpolated_value
