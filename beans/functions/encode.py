import numpy as np
import pandas as pd
import collections
import sklearn.preprocessing


class Encode:
    """
    For OneHot & Ordinal Encoding
    """

    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def bits(frame: pd.DataFrame, categories: collections.namedtuple) -> pd.DataFrame:
        """
        For One Hot Encoding

        :param frame: A DataFrame of data
        :param categories: A collection consisting of 2 items.  Item 1 -> categorical fields, Item 2 -> The unique
                                      values of each field
        :return:
        """

        enc = sklearn.preprocessing.OneHotEncoder(categories=categories.arrays, sparse=False, dtype=np.int)
        bits_ = enc.fit_transform(X=frame[[categories.fields]])

        columns = [column[(column.rindex('_') + 1):] for column in enc.get_feature_names()]
        return pd.DataFrame(data=bits_, columns=columns)
