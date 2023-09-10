"""This module will run through the modelling steps"""
import logging

import pandas as pd
import sklearn.preprocessing as skp

import config
import src.algorithms.scale
import src.algorithms.encode
import src.algorithms.project
import src.algorithms.knee
import src.types.cases


class Interface:
    """
    Interface
    """

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
        self.__logger: logging.Logger = logging.getLogger(__name__)

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
        scaled: pd.DataFrame
        scaler: skp.StandardScaler
        scaled, scaler = self.__scale(blob=train)
        cases = src.types.cases.Cases(data=train, scaler=scaler, scaled=scaled)

        # Determining the best # of projection components
        n_components: int = src.algorithms.knee.Knee().exc(blob=cases.scaled.drop(columns=self.__meta.dependent))
        self.__logger.info('Plausible # of components: %s', n_components)

        # Projecting the independent variables, i.e., dimensionality reduction via
        # kernel principal component analysis
        projected, projector = src.algorithms.project.Project().exc(
            blob=cases.scaled, exclude=[self.__meta.dependent], n_components=n_components)
        cases = cases._replace(projected=projected, projector=projector)

        # Encoding the dependent variable
        encoded, labels = src.algorithms.encode.Encode().exc(blob=cases.projected, field=self.__meta.dependent)
        cases = cases._replace(encoded=encoded, labels=labels)

        self.__logger.info('%s', cases.encoded.info())
