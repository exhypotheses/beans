import numpy as np
import pandas as pd
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
    def bits(frame: pd.DataFrame, field_of_labels: str, unique_labels: np.ndarray) -> pd.DataFrame:
        """
        For One Hot Encoding

        :param frame: A DataFrame of data
        :param field_of_labels: The field that would be transformed into a one-hot format
        :param unique_labels: The unique values of the 'field_of_labels'
        :return:
        """

        enc = sklearn.preprocessing.OneHotEncoder(categories=unique_labels, sparse=False, dtype=np.int)
        bits_ = enc.fit_transform(X=frame[list(field_of_labels)])

        columns = [column[(column.rindex('_') + 1):] for column in enc.get_feature_names()]
        return pd.DataFrame(data=bits_, columns=columns)