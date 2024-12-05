import numpy as np


def calculate_nse(observed, simulated):
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)
