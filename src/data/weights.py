"""Evaluates the target field's weights per label"""
import sklearn
import pandas as pd

import config


class Weights:
    """
    Weights
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        # The dependent variable field
        self.__dependent = config.Config().dependent

    def __weights(self, blob: pd.DataFrame) -> dict:
        """

        :param blob:
        :return:
        """

        labels = blob.copy()[self.__dependent].unique()

        values = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=labels, y=blob.copy()[self.__dependent])

        return dict(labels, values)

    def exc(self, blob: pd.DataFrame) -> dict:
        """
        
        :param blob: The data set
        :return:
        """

        weights = self.__weights(blob=blob)

        return weights
