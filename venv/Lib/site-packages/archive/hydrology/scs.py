"""Classes for performing rational method basin hydrology analysis.

:copyright: 2016, See AUTHORS for more details.
:license: GNU General Public License, See LICENSE for more details.

"""

import math

from scipy import interpolate


class Basin:

    def __init__(self, area, cn, tc, runoff_dist, peak_factor):
        """

        Args:
            area (float): In :math:`miles^2`.
            cn (float):
            tc (float): In :math:`hours`.
            runoff_dist (UnitDistribution):
            peak_factor (float):

        """
        self.area = area
        self.cn = cn
        self.tc = tc
        self.runoff_dist = runoff_dist
        self.peak_factor = peak_factor

    @property
    def potential_retention(self):
        return 1000.0 / self.cn - 10.0

    @property
    def initial_abstraction(self):
        return 0.2 * self.potential_retention

    def runoff_depth(self, rainfall_depth):
        """Get the depth of runoff_area generated from a defined rainfall and properties of the basin.

        Args:
            rainfall_depth (float): In :math:`inches`.

        Returns:
            float: Runoff, in :math:`inches`.

        """
        if rainfall_depth > self.initial_abstraction:
            a = pow(rainfall_depth - self.initial_abstraction, 2.0)
            b = rainfall_depth - self.initial_abstraction + self.potential_retention
            return a / b
        return 0.0

    def runoff_volume(self, rainfall_depth):
        """Get the volume of runoff_area generated from a defined rainfall and properties of the basin.
        
        Args:
            rainfall_depth (float): In :math:`inches`.

        Returns:
            float: Runoff, in :math:`feet^3`.

        """
        runoff = self.runoff_depth(rainfall_depth)
        return runoff * self.area * 43560.0 / 12.0

    def runoff_depth_incremental(self, rainfall_hydrograph, interval):
        duration = rainfall_hydrograph[-1][0]
        time_steps = math.ceil(duration / interval)
        x = [row[0] for row in rainfall_hydrograph]
        y = [row[1] for row in rainfall_hydrograph]
        y_interp = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        x_new = [time_step * interval for time_step in range(time_steps + 1)]
        y_new = y_interp(x_new)
        hydrograph = list(zip(x_new, y_new))
        runoff_depth_incremental = []
        for first, second in zip(hydrograph, hydrograph[1:]):
            time_1, rainfall_1 = first
            time_2, rainfall_2 = second
            runoff_depth_1 = self.runoff_depth(rainfall_1)
            runoff_depth_2 = self.runoff_depth(rainfall_2)
            runoff_incremental = runoff_depth_2 - runoff_depth_1
            runoff_depth_incremental.append(runoff_incremental)
        return runoff_depth_incremental

    def flood_hydrograph(self, rainfall_hydrograph, interval):
        rh = self.runoff_hydrograph(interval=interval)
        ri = self.runoff_depth_incremental(rainfall_hydrograph, interval)
        ri_reversed = list(reversed(ri))
        composite_length = len(rh) + len(ri)
        hydrograph = [(i * interval, 0.0) for i in range(composite_length - 1)]
        for i, row in enumerate(hydrograph):
            upper = i + 1
            lower = max(upper - len(ri), 0)
            total = sum(ri_reversed[j - upper] * rh[j][1] for j in range(lower, upper) if j < len(rh))
            hydrograph[i] = hydrograph[i][0], total
        return hydrograph

    @property
    def peak_time(self):
        delta = 0.133 * self.tc
        lag = 0.6 * self.tc
        return delta / 2.0 + lag

    @property
    def peak_runoff(self):
        return self.peak_factor * self.area / self.peak_time  # cfs

    def runoff_hydrograph(self, **kwargs):
        return self.runoff_dist.hydrograph(self.peak_time, self.peak_runoff, **kwargs)


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
    
    def secondary_ratio(self, time_ratio):
        """Get the secondary fraction that has been accumulated over the specified fraction of time.

        Args:
            time_ratio (float): Fraction of time, between 0 and 1.

        Returns:
            float: Secondary fraction, between 0 and 1.

        Raises:
            ValueError: If the time_ratio is negative or greater than 1.

        """
        if time_ratio < 0.0:
            raise ValueError('Defined time ratio must be positive number.')
        x = [row[0] for row in self.ratio_pairs]
        y = [row[1] for row in self.ratio_pairs]
        y_interp = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
        return float(y_interp(time_ratio))

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
                secondary_ratio = self.secondary_ratio(time_ratio)
                value = peak_value * secondary_ratio
                hydrograph.append((time, value))
        else:
            for time_ratio, secondary_ratio in self.ratio_pairs:
                hydrograph.append((time_ratio * duration, secondary_ratio * peak_value))
        return hydrograph
