from scipy.signal import convolve
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from datos_del_hidrograma_unitario_adimensional.datos import SCS_DIMENSIONLESS_UH


def scs_unit_hydrograph(runoff_volume, lag_time, dt):
    # Tiempo sin dimensiones
    num_points = int(5 * lag_time / dt)
    if num_points < 1:
        num_points = 1  # Asegurar al menos un punto
    t_d = np.linspace(0, 5, num_points)
    scs_q_curve = interp1d(SCS_DIMENSIONLESS_UH[:, 0],
                           SCS_DIMENSIONLESS_UH[:, 1],
                           kind='linear',
                           fill_value="extrapolate",
                           bounds_error=False)
    scs2 = scs_q_curve(t_d)

    # Normalizar el hidrograma unitario
    area = trapezoid(scs2, t_d)
    scs2 /= area

    # Escale scs2 para tener unidades consistentes
    scs2 /= lag_time

    # Convolución
    discharge_rate = convolve(runoff_volume, scs2, mode="full")[
        :len(runoff_volume)]

    return discharge_rate
