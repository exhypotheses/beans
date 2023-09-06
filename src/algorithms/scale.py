"""This module scales the numeric, ratio, independent variables of beans; all the 
independent variables are numeric/ratio  """
import numpy as np
import pandas as pd
import sklearn.preprocessing


class Scale:

    def __init__(self):
        """
        Constructor
        
        """

    @staticmethod
    def apply(blob: pd.DataFrame, scaler: sklearn.preprocessing.StandardScaler) -> pd.DataFrame:
        """
        Use scaler to scale the numerical data, subsequently reconstruct the data

        :param blob:
        :param scaler:
        :return:
        """        

        # Scaling numerical fields
        scaled_: np.ndarray = scaler.transform(X=blob)

        return pd.DataFrame(data=scaled_, columns=blob.columns)

    @staticmethod
    def exc(blob: pd.DataFrame) -> sklearn.preprocessing.StandardScaler:
        """
        
        :param blob: A frame of numeric fields
        :return:
        """

        # Scaling
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True)
        scaler.fit(X=blob.values)

        return scaler
