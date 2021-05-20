import pandas as pd
import collections

import sklearn.model_selection


class Split:
    
    def __init__(self, splitting: collections.namedtuple):
        """

        :param splitting: A collection of named parameters, and their values, for
                         the sklearn.model_selection.train_test_split() function
        """

        self.splitting = splitting

    def exc(self, data: pd.DataFrame, labels: list, strata: list) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """

        :param data:
        :param labels:
        :param strata:
        :return:
        """

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            data.drop(columns=labels),
            data[labels],
            test_size=self.splitting.test_size,
            random_state=self.splitting.random_state,
            stratify=data[strata])

        return x_train, x_test, y_train, y_test
