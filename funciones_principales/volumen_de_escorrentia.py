import numpy as np


def scs_runoff_volume(rain_depth, CN, catchment_area, lambda_val):
    S = (25400 - 254 * CN) / CN
    Ia = lambda_val * S

    # Precipitación acumulada (Pe)
    cumulative_rain_depth = np.cumsum(rain_depth)

    # Calcular Pe sólo cuando la lluvia acumulada exceda Ia
    Pe_cumulative = np.where(cumulative_rain_depth > Ia,
                             ((cumulative_rain_depth - Ia) ** 2) /
                             (cumulative_rain_depth - Ia + S),
                             0)

    # Exceso incremental de precipitación
    incremental_Pe = np.diff(Pe_cumulative, prepend=0)

    # Exceso incremental en metros cúbicos por paso de tiempo
    incremental_excess = incremental_Pe * catchment_area * 1e6 / 1000

    return incremental_excess
