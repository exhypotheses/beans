"""This module will run through the modelling steps"""
import logging

import pandas as pd
import sklearn.preprocessing as skp

import config
import src.algorithms.scale
import src.algorithms.split
import src.algorithms.encode
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

    def __scale(self, blob: pd.DataFrame, scaler: skp.StandardScaler = None) -> (pd.DataFrame, skp.StandardScaler):

        scaled: pd.DataFrame
        scaler: skp.StandardScaler
        scaled, scaler  = src.algorithms.scale.Scale(numeric=self.__meta.numeric).exc(blob=blob, scaler=None)

        return scaled, scaler

    def exc(self, blob: pd.DataFrame):
        """
        
        :param blob: The beans data
        """

        # Step ...
        train, test = self.__split(blob=blob.copy())

        # Training
        scaled, scaler = self.__scale(blob=train)        
        training = self.Training(data=train, scaler=scaler, scaled=scaled, encoded=None)
        encoded = src.algorithms.encode.Encode().exc(blob=training.data)
        training._replace(encoded=encoded)

        # Testing
        testing = self.Testing(data=test, scaled=self.__scale(blob=test, scaler=scaler))

        self.__logger.info(training.encoded.head())
        self.__logger.info(testing._fields)
