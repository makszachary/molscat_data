import numpy as np
from typing import Any

def chi_squared(theory: np.ndarray[Any, float], experiment: float | np.ndarray[Any, float], std: float | np.ndarray[Any, float] = None) -> float | np.ndarray[Any, float]:
    """Evaluate chi-squared from a set of models.
    
    :param theory: values obtained from the set of models;
    the last axes of theory should match the shape of experiment
    :param experiment: measured values
    :param std: standard devations of the measuered values
    
    :return: array of the chi-squared values corresponding to each model;
    the shape of this array is determined by the first axes of theory,
    without the part matching the shape of experiment
    """

    theory = np.array(theory)
    experiment = np.array(experiment)
    if std is None:
        std = np.full_like(experiment, 1, dtype=float)
    std = np.array(std)

    if not (len(experiment.shape) == 0 or theory.shape[-len(experiment.shape):] == experiment.shape) or std.shape != experiment.shape:
        raise ValueError(f"The final part of the shape of theory ({theory.shape[-len(experiment.shape):]}) should match the shape of experiment ({experiment.shape}).")    

    chi_sq = np.sum(((theory-experiment)/std)**2, axis=tuple(range(-len(experiment.shape), 0)))

    return chi_sq