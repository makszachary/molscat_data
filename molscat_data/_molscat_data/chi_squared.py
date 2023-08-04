import numpy as np
from typing import Any

def chi_squared(theory: np.ndarray[Any, float], experiment: float | np.ndarray[Any, float], std: float | np.ndarray[Any, float] = None) -> float | np.ndarray[Any, float]:

    theory = np.array(theory)
    experiment = np.array(experiment)
    if std is None:
        std = np.full_like(experiment, 1, dtype=float)
    std = np.array(std)

    if not (len(experiment.shape) == 0 or theory.shape[-len(experiment.shape):] == experiment.shape) or std.shape != experiment.shape:
        raise ValueError(f"The final part of the shape of theory ({theory.shape[-len(experiment.shape):]}) should match the shape of experiment ({experiment.shape}).")    

    chi_sq = np.sum(((theory-experiment)/std)**2, axis=tuple(range(-len(experiment.shape), 0)))

    return chi_sq