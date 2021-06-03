import pandas as pd
import sklearn.preprocessing

import beans.functions.scale

class Client:
    
    def __init__(self, target: str):
        """
        
        """
        self.target = target

    def scale_(self, testing_split: pd.DataFrame, scaler: sklearn.preprocessing.StandardScaler):
        """
        Scales the sampled data

        :param testing_split:  The sampled data
        :return:
        """

        scale = beans.functions.scale.Scale()
        x_testing_scaled = scale.apply(blob=testing_split.drop(columns=self.target), scaler=scaler)
        testing_scaled = pd.concat((x_testing_scaled, testing_split[self.target]), axis=1, ignore_index=False)

        return testing_scaled, scaler

    def encode_(self, testing_scaled: pd.DataFrame):
        """
        One-hot-encodes the depedent variable

        :param testing_scaled: In general, this will be the DataFrame w.r.t. the final preprocessing step before
        structuring
        :return:
        """