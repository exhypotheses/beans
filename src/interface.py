"""This module will run through the modelling steps"""
import pandas as pd

import src.algorithms.split


class Interface:
    """
    Interface
    """

    def __init__(self, train_size: float = 0.735) -> None:
        """
        
        :param train_size: The decimal fraction for training
        """

        self.__train_size = train_size

    def __split(self, blob: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        return src.algorithms.split.Split().exc(
            data=blob, train_size=self.__train_size)

    def exc(self, blob: pd.DataFrame):
        """
        
        :param blob: The beans data
        """

        # Step ...
        training, testing = self.__split(blob=blob.copy())

        return training, testing
