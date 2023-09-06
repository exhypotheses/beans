"""This module will run through the modelling steps"""
import logging
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

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self, blob: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        return src.algorithms.split.Split().exc(
            data=blob, train_size=self.__train_size)

    def exc(self, blob: pd.DataFrame):
        """
        
        :param blob: The beans data
        """

        # Step ...
        training, testing = self.__split(blob=blob.copy())

        self.__logger.info('%s', training.info())
        self.__logger.info('%s', testing.info())

        self.__logger.info(training['class'].value_counts() / training.shape[0])
        self.__logger.info(msg=testing['class'].value_counts() / testing.shape[0])
