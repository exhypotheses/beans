"""This module scales the numeric, ratio, independent variables of beans; all the 
independent variables are numeric/ratio  """
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp


class Scale:
    """
    Scale: Scales field values.
    """

    def __init__(self, numeric: list):
        """
        
        
        :param numeric: The numeric fields ...        
        """

        self.__numeric = numeric

    def __scale(self, data: pd.DataFrame, scaler: skp.StandardScaler) -> pd.DataFrame:
        """
        Uses scaler to scale the numerical data

        :param data: The frame of numeric fields
        :param scaler: The transformer
        :return:
        """

        # Scaling numerical fields
        scaled_: np.ndarray = scaler.transform(X=data)

        return pd.DataFrame(data=scaled_, columns=data.columns)

    def __scaler(self, data: pd.DataFrame) -> skp.StandardScaler:
        """
        
        :return:
        """

        # Scaling
        scaler = skp.StandardScaler(with_mean=True)
        scaler.fit(X=data)

        return scaler

    def exc(self, blob: pd.DataFrame, scaler: skp.StandardScaler = None) -> (
            pd.DataFrame, skp.StandardScaler):
        """
        :param blob: A frame of ...
        :return:
            scaled: The transformed <blob> data frame, wherein the numeric 
                    fields have been rescaled
            scaler: The numeric fields scaler
        """

        # The numeric fields
        data = blob[self.__numeric]

        # Scaler
        if scaler is None:
            scaler = self.__scaler(data=data.copy())

        # Hence, the frame of scaled numeric fields
        __frame = self.__scale(data=data.copy(), scaler=scaler)

        # Finally, the transformed <blob> data frame, wherein the each numeric field
        # is the rescaled numeric field.
        scaled = pd.concat((blob.copy().drop(columns=self.__numeric), __frame), 
                           axis=1, ignore_index=False)

        return scaled, scaler
