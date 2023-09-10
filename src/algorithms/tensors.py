import pandas as pd
import numpy as np
import pymc


class Tensors:

    def __init__(self) -> None:
        pass

    def structure(self, blob: pd.DataFrame, labels: list):
        """
        Structures ... according to what is expected by the neural network model

        :param blob:
        :param labels:
        :return:
        """

        x_matrix = blob.drop(columns=labels).to_numpy()
        y_matrix = blob[labels].to_numpy()

        unity = np.ones((x_matrix.shape[0], 1))
        x_points = np.concatenate((unity, x_matrix), axis=1).astype(pymc.smartfloatX)
        y_points = y_matrix.astype(pymc.smartfloatX)

        return x_points, y_points