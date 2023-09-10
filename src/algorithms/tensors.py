"""Builds tensors for the neural network model"""
import pandas as pd
import numpy as np
import pymc
import pytensor


class Tensors:
    """
    Tensors
        This class builds tensors the neural network model's tensors
    """

    def __init__(self) -> None:
        pass


    def __single(self, blob: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        
        :param blob: The data frame of the neural network's input vectors
        :return:
            x_points: The model's input tensor
            y_points: None
        """

        x_matrix: np.ndarray = blob.copy().to_numpy()
        unity = np.ones((x_matrix.shape[0], 1))
        x_points = np.concatenate((unity, x_matrix), axis=1).astype(pytensor.config.floatX)

        return x_points, None

    def __doublet(self, blob: pd.DataFrame, labels: list) -> (np.ndarray, np.ndarray):
        """
        
        :param blob: The data frame of the neural network's input & output vectors
        :param labels: The names of the output vectors fields
        :return:
            x_points: The model's input tensor
            y_points: The corresponding output tensor
        """

        x_matrix: np.ndarray = blob.copy().drop(columns=labels).to_numpy()
        unity: np.ndarray[np.float64] = np.ones((x_matrix.shape[0], 1))
        x_points: np.ndarray = np.concatenate((unity, x_matrix), axis=1).astype(pytensor.config.floatX)

        y_matrix: np.ndarray = blob[labels].to_numpy()
        y_points: np.ndarray = y_matrix.astype(pytensor.config.floatX)

        return x_points, y_points


    def exc(self, blob: pd.DataFrame, labels: list = None) -> (np.ndarray, np.ndarray):
        """
        

        :param blob:
        :param labels:
        :return:
        """

        if labels is None:
            return self.__single(blob=blob)

        return self.__doublet(blob=blob, labels=labels)
