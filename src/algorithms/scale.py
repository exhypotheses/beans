"""This module scales the numeric, ratio, independent variables of beans; all the 
independent variables are numeric/ratio  """
import numpy as np
import pandas as pd
import sklearn.preprocessing


class Scale:
    """
    Scale: Scales field values.
    """

    def __init__(self, blob: pd.DataFrame, numeric: list):
        """
        
        :param blob: A frame of ...
        :param numeric: The numeric fields ...
        :return:
            scaler: The numeric fields scaler
            scaled: The transformed <blob> data frame, wherein the numeric fields have been rescaled
        """

        self.__data = blob[numeric]

        # The numeric fields scaler
        self.scaler = self.__scaler()

        # Hence, the frame of scaled numeric fields
        __frame = self.__scale()

        # Finally, the transformed <blob> data frame, wherein the numeric fields have been rescaled
        self.scaled = pd.concat((blob.copy().drop(columns=numeric), __frame),
                                axis=1, ignore_index=False)

    def __scale(self) -> pd.DataFrame:
        """
        Use scaler to scale the numerical data, subsequently reconstruct the data

        :return:
        """

        # Scaling numerical fields
        scaled_: np.ndarray = self.scaler.transform(X=self.__data)

        return pd.DataFrame(data=scaled_, columns=self.__data.columns)

    def __scaler(self) -> sklearn.preprocessing.StandardScaler:
        """
        
        :return:
        """

        # Scaling
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True)
        scaler.fit(X=self.__data.values)

        return scaler

    def exc(self):
        """
        Free
        """
