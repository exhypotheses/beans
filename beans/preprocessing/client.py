import pandas as pd
import sklearn.preprocessing

import beans.functions.scale

class Client:
    
    def __init__(self, target: str):
        """
        
        """
        self.target = target

    def scale_(self, blob: pd.DataFrame, scaler: sklearn.preprocessing.StandardScaler):
        """
        Scales the sampled data

        :param blob:  The sampled data
        :param scaler:  The scale transform object
        :return:
        """

        scale = beans.functions.scale.Scale()
        x_scaled = scale.apply(blob=blob.drop(columns=self.target), scaler=scaler)
        scaled = pd.concat((x_scaled, blob[self.target]), axis=1, ignore_index=False)

        return scaled

    def encode_(self, blob: pd.DataFrame):
        """
        One-hot-encodes the dependent variable

        :param blob: In general, this will be the DataFrame w.r.t. the final preprocessing step before
        structuring
        :return:
        """