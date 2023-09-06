"""This module scales the numeric, ratio, independent variables of beans; all the 
independent variables are numeric/ratio  """
import numpy as np
import pandas as pd
import sklearn.preprocessing


class Scale:
    """
    Scale: Scales field values.
    """

    def __init__(self, blob: pd.DataFrame):
        """
        
        :param blob: A frame of numeric fields     
        """

        self.__blob = blob

        # The outputs
        self.scaler = self.__scaler()
        self.scaled = self.__scale()

    def __scale(self) -> pd.DataFrame:
        """
        Use scaler to scale the numerical data, subsequently reconstruct the data

        :return:
        """        

        # Scaling numerical fields
        scaled_: np.ndarray = self.scaler.transform(X=self.__blob)

        return pd.DataFrame(data=scaled_, columns=self.__blob.columns)

    def __scaler(self) -> sklearn.preprocessing.StandardScaler:
        """
        
        :return:
        """

        # Scaling
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True)
        scaler.fit(X=self.__blob.values)

        return scaler
