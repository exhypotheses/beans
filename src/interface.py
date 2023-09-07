"""This module will run through the modelling steps"""
import logging

import pandas as pd
import sklearn.preprocessing as skp

import config
import src.algorithms.scale
import src.algorithms.encode
import src.algorithms.project
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

    def __scale(self, blob: pd.DataFrame, scaler: skp.StandardScaler = None) -> (pd.DataFrame, skp.StandardScaler):

        scaled: pd.DataFrame
        scaler: skp.StandardScaler
        scaled, scaler  = src.algorithms.scale.Scale(numeric=self.__meta.numeric).exc(blob=blob, scaler=None)

        return scaled, scaler

    def exc(self, train: pd.DataFrame):
        """
        
        :param train: The beans data
        """

        # Scaling
        scaled, scaler = self.__scale(blob=train)
        training = self.Training(data=train, scaler=scaler, scaled=scaled, projector=None, projected=None, encoded=None)

        # Number of components

        # Projecting independent variables
        projected, projector = src.algorithms.project.Project().exc(blob=training.scaled, exclude=[self.__meta.dependent])
        training = training._replace(projected=projected, projector=projector)

        # Encoding the dependent variable
        encoded = src.algorithms.encode.Encode().exc(blob=training.projected, field=self.__meta.dependent)
        training = training._replace(encoded=encoded)

        self.__logger.info('%s', training.encoded.info())
