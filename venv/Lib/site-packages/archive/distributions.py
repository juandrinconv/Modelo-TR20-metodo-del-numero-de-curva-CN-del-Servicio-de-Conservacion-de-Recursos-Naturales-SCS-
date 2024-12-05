
import math

import simpleeval
from scipy import interpolate



EVAL_FUNCS = {
    'acos': math.acos,
    'acosh': math.acosh,
    'asin': math.asin,
    'asinh': math.asinh,
    'atan': math.atan,
    'atan2': math.atan2,
    'atanh': math.atanh,
    'ceil': math.ceil,
    'copysign': math.copysign,
    'cos': math.cos,
    'cosh': math.cosh,
    'degrees': math.degrees,
    'e': math.e,
    'exp': math.exp,
    'fabs': math.fabs,
    'factorial': math.factorial,
    'floor': math.floor,
    'fmod': math.fmod,
    'frexp': math.frexp,
    'fsum': math.fsum,
    'hypot': math.hypot,
    'isinf': math.isinf,
    'isnan': math.isnan,
    'ldexp': math.ldexp,
    'log': math.log,
    'log10': math.log10,
    'log1p': math.log1p,
    'modf': math.modf,
    'pi': math.pi,
    'pow': math.pow,
    'radians': math.radians,
    'sin': math.sin,
    'sinh': math.sinh,
    'sqrt': math.sqrt,
    'tan': math.tan,
    'tanh': math.tanh,
    'trunc': math.trunc,
}


class Generator:

    def __init__(self, equation, first_key, eq_kwargs=None):
        self.equation = equation
        self.first_key = first_key
        self.eq_kwargs = eq_kwargs
        self.second(first=1.0)

    def second(self, first):
        equation = self.equation
        for key, val in self.eq_kwargs.items():
            equation = equation.replace(key, str(val))
        equation = equation.replace(self.first_key, str(first))
        try:
            evaluation = simpleeval.simple_eval(equation, functions=EVAL_FUNCS)
            return evaluation
        except:
            raise ValueError('Error in the produced equation: {0}'.format(equation))

    def hydrograph(self, duration, interval):
        hydrograph = []
        time_steps = math.ceil(duration / interval)
        for time_step in range(time_steps + 1):
            time = time_step * interval     # hr
            intensity = self.second(time)   # in/hr
            rainfall = time * intensity     # in
            hydrograph.append((time, rainfall))
        return hydrograph


class UnitDistribution:
    def __init__(self, ratio_pairs):
        """Represents the fraction of a secondary ratio that has been accumulated over each fraction of duration.

        Args:
            ratio_pairs (list[tuple[float, float]]): Pairs of time a secondary ratio, all between 0 and 1.

        Raises:
            ValueError: If any of the values within the tuples of pairs are negative or greater than 1.

        Warning:
            - If time ratio of 0.0 and/or 1.0 not defined, (0.0, 0.0) and/or (1.0, 1.0) added to the pairs list.
            - If time ratio of 0.0 and/or 1.0 defined, the pairs will be updated to (0.0, 0.0) and/or (1.0, 1.0).

        """
        ratio_pairs.sort(key=lambda pair: pair[0])
        time_ratios = [ratio[0] for ratio in ratio_pairs]
        if time_ratios[0] < 0.0:
            raise ValueError('Time ratios in pairs must all be positive numbers.')
        self.ratio_pairs = ratio_pairs

    def second(self, first):
        """Get the secondary fraction that has been accumulated over the specified fraction of time.

        Args:
            first (float): Fraction of time, between 0 and 1.

        Returns:
            float: Secondary fraction, between 0 and 1.

        Raises:
            ValueError: If the time_ratio is negative or greater than 1.

        """
        if first < 0.0:
            raise ValueError('Defined time ratio must be positive number.')
        # x = [row[0] for row in self.pairs]
        # y = [row[1] for row in self.pairs]
        x, y = zip(*self.ratio_pairs)
        y_interp = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        return float(y_interp(first))

    def hydrograph(self, duration, peak_value, **kwargs):
        """Generate a hydrograph from scaling time and secondary ratios against the defined totals.

        Args:
            duration (float): The time when peak value occurs, in :math:`hours`.
            peak_value (float): The secondary total to scale ratios from.

        Returns:
            list[tuple[float, float]]: Pairs of time (in :math:`hours`) and rainfall (in :math:`inches`).

        Note:
            - If interval is undefined, hydrograph is generated from each pair in the ratio pairs.

        """
        step = kwargs.pop('step', None)
        interval = kwargs.pop('interval', None)
        hydrograph = []
        if step or interval:
            if not step:
                step = float(interval) / duration
            last_time_step = self.ratio_pairs[-1][0]
            time_steps = math.ceil(last_time_step / step)
            for time_step in range(time_steps + 1):
                time_ratio = time_step * step
                time = time_ratio * duration
                secondary_ratio = self.second(time_ratio)
                value = peak_value * secondary_ratio
                hydrograph.append((time, value))
        else:
            for time_ratio, secondary_ratio in self.ratio_pairs:
                hydrograph.append((time_ratio * duration, secondary_ratio * peak_value))
        return hydrograph
