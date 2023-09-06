"""This module will run through the modelling steps"""
import logging
import pandas as pd
import sklearn.preprocessing

import src.algorithms.split
import src.algorithms.scale

import config


class Interface:
    """
    Interface
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.__configurations = config.Config()

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self, blob: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        return src.algorithms.split.Split().exc(
            data=blob, train_size=self.__configurations.train_size)

    def __scale(self, blob: pd.DataFrame) -> (sklearn.preprocessing.StandardScaler, pd.DataFrame):

        objects = src.algorithms.scale.Scale(blob=blob, numeric=self.__configurations.numeric)

        return objects.scaler, objects.scaled

    def exc(self, blob: pd.DataFrame):
        """
        
        :param blob: The beans data
        """

        # Step ...
        training, testing = self.__split(blob=blob.copy())
        self.__scale(blob=training)

        self.__logger.info('%s', training.info())
        self.__logger.info('%s', testing.info())

        self.__logger.info(training['class'].value_counts() / training.shape[0])
        self.__logger.info(msg=testing['class'].value_counts() / testing.shape[0])
