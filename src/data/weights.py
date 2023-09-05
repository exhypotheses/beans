"""Evaluates the target field's weights per label"""
import sklearn
import pandas as pd


class Weights:
    """
    Weights
    """

    def __init__(self) -> None:
        pass

    def __weights(self, blob: pd.DataFrame, target: str) -> dict:
        """

        :param blob:
        :return:
        """

        labels = blob.copy()[target].unique()

        values = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=labels, y=blob.copy()[target])

        return dict(labels, values)

    def exc(self, blob: pd.DataFrame, target: str) -> dict:
        """
        
        :param blob: The data set
        :param target: The dependent variable field
        :return:
        """

        weights = self.__weights(blob=blob, target=target)

        return weights
