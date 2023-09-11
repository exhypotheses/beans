"""Splits the beans data set into training & testing parts"""
import pandas as pd

import sklearn.model_selection

import config


class Split:
    """
    Split
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = config.Config()
        self.__meta = self.__configurations.meta

    def exc(self, data: pd.DataFrame, train_size: float) -> (pd.DataFrame, pd.DataFrame):
        """

        :param data: The beans data
        :param train_size: The decimal fraction of the data for training
        :return:
        """

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            data.drop(columns=self.__meta.dependent),
            data[self.__meta.dependent],
            train_size=train_size,
            random_state=self.__configurations.seed,
            stratify=data[self.__meta.dependent])

        training = pd.concat((x_train.reset_index(drop=True), y_train.reset_index(drop=True)), axis=1,
                             ignore_index=False)
        testing = pd.concat((x_test.reset_index(drop=True), y_test.reset_index(drop=True)), axis=1,
                            ignore_index=False)

        return training, testing
