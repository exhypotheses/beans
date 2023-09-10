"""This module will run through the modelling steps"""
import logging

import pandas as pd

import src.algorithms.tensors
import src.modelling.preprocessing
import src.types.training


class Interface:
    """
    Interface
    """

    Training = src.types.training.Training

    def __init__(self) -> None:
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger: logging.Logger = logging.getLogger(__name__)

    def exc(self, train: pd.DataFrame):
        """
        
        :param train: The training data
        """

        training = src.modelling.preprocessing.Preprocessing().exc(train=train)
        self.__logger.info('%s', training.encoded.info())

        x_points, y_points = src.algorithms.tensors.Tensors().exc(
            blob=training.encoded, labels=training.labels)
        self.__logger.info(x_points)
        self.__logger.info(y_points)
