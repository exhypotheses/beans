"""
Exploring a named tuple class for training data, and complementary objects.
"""
import typing

import numpy as np
import pandas as pd
import sklearn.decomposition as skd
import sklearn.preprocessing as skp


class Training(typing.NamedTuple):
    """
    
    :return: Named tuple class
    """

    data: pd.DataFrame = None
    scaler: skp.StandardScaler = None
    scaled: pd.DataFrame = None
    projector: skd.KernelPCA = None
    projected: pd.DataFrame = None
    encoded: pd.DataFrame = None
    labels: list = None
    x_points: np.ndarray = None
    y_points: np.ndarray = None
