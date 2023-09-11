"""This will run through ..."""
import logging

import pandas as pd
import sklearn.preprocessing as skp

import config
import src.algorithms.scale
import src.algorithms.encode
import src.algorithms.project
import src.algorithms.knee
import src.types.training
import src.algorithms.tensors


class Preprocessing:
    """
    Preprocessing
        Sequentially prepares the raw training data for modelling
    """
    Training = src.types.training.Training

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

    def __scale(self, training: Training) -> Training:
        """
        
        :param training:
        :return: training: Training
        """

        scaled: pd.DataFrame
        scaler: skp.StandardScaler
        scaled, scaler  = src.algorithms.scale.Scale(numeric=self.__meta.numeric).exc(blob=training.data)

        return training._replace(scaler=scaler, scaled=scaled)

    def __project(self, training: Training) -> Training:
        """
        For dimension reduction purposes.  It projects the independent variables 
        via kernel principal component analysis
        
        :param training:
        :return: training: Training
        """

        # Determining the best # of projection components
        n_components: int = src.algorithms.knee.Knee().exc(blob=training.scaled.drop(columns=self.__meta.dependent))
        self.__logger.info('Plausible # of components: %s', n_components)

        # Projecting
        projected, projector = src.algorithms.project.Project().exc(
            blob=training.scaled, exclude=[self.__meta.dependent], n_components=n_components)

        return training._replace(projected=projected, projector=projector)

    def __encode(self, training: Training) -> Training:
        """
        
        :param training:
        :return: training: Training
        """

        encoded, labels = src.algorithms.encode.Encode().exc(
            blob=training.projected, field=self.__meta.dependent)

        return training._replace(encoded=encoded, labels=labels)

    def __tensors(self, training: Training) -> Training:

        x_points, y_points = src.algorithms.tensors.Tensors().exc(
            blob=training.encoded, labels=training.labels)

        return training._replace(x_points=x_points, y_points=y_points)

    def exc(self, train: pd.DataFrame) -> Training:
        """
        
        :param train: The training data
        """

        # Setting-up
        training = src.types.training.Training(data=train)

        # Scaling
        training = self.__scale(training=training)

        # Projecting
        training = self.__project(training=training)

        # Encoding
        training = self.__encode(training=training)

        # Tensors
        training = self.__tensors(training=training)

        return training
