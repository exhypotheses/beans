"""This module will run through the modelling steps"""
import logging

import pandas as pd
import pymc

import src.algorithms.tensors
import src.inference.sampling.blackjax
import src.inference.variational.differentiation
import src.modelling.neural.snapshot
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

        # Preprocessing
        training = src.modelling.preprocessing.Preprocessing().exc(train=train)
        self.__logger.info('Algebraic objects data:\n%s', training.encoded.head())
        self.__logger.info('\nX: %s\n%s', training.x_points.shape, training.x_points)
        self.__logger.info('\nY: %s\n%s', training.y_points.shape, training.y_points)

        # Neural Network Model Architecture
        snapshot = src.modelling.neural.snapshot.Snapshot()
        model: pymc.Model = snapshot.model_(features=training.x_points, output=training.y_points)

        # Inference
        details = src.inference.variational.differentiation.Differentiation().full_rank_advi(
            model=model, n_iterations=500)
        self.__logger.info(details)
