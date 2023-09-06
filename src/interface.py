"""This module will run through the modelling steps"""
import logging
import pandas as pd
import sklearn.preprocessing

import src.algorithms.split
import src.algorithms.scale

import config
import src.structures


class Interface:
    """
    Interface
    """

    # Structures
    Training = src.structures.Structures().Training
    Testing = src.structures.Structures().Testing

    def __init__(self) -> None:
        """
        Constructor
        """

        self.__configurations = config.Config()
        self.__meta = self.__configurations.meta

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def __split(self, blob: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        return src.algorithms.split.Split().exc(
            data=blob, train_size=self.__configurations.train_size)

    def __scale(self, blob: pd.DataFrame) -> (sklearn.preprocessing.StandardScaler, pd.DataFrame):

        objects = src.algorithms.scale.Scale(blob=blob, numeric=self.__meta.numeric)

        return objects.scaler, objects.scaled

    def exc(self, blob: pd.DataFrame):
        """
        
        :param blob: The beans data
        """

        # Step ...
        train, test = self.__split(blob=blob.copy())

        # Training
        scaler, scaled = self.__scale(blob=train)
        training = self.Training(data=train, scaler=scaler, scaled=scaled)

        # Testing
        testing = self.Testing(data=test, scaled=None)

        self.__logger.info('%s', training.data.info())
        self.__logger.info('%s', testing.data.info())

        self.__logger.info(training.data['class'].value_counts() / training.data.shape[0])
        self.__logger.info(msg=testing.data['class'].value_counts() / testing.data.shape[0])
