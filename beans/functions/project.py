import numpy as np
import pandas as pd

import sklearn.decomposition

import config


class Project:

    def __init__(self):
        """
        Constructor
        """

        configurations = config.Config()
        self.SEED = configurations.SEED

    @staticmethod
    def apply(matrix: np.ndarray, vector: pd.Series, projector: sklearn.decomposition.KernelPCA) -> pd.DataFrame:
        """

        :param matrix: The independent variables matrix
        :param vector:  The dependent variable vector
        :param projector: A transformer
        :return:
        """

        principals = projector.transform(X=matrix)
        names = ['kpc_' + str(i).zfill(2) for i in range(1, 1 + principals.shape[1])]

        return pd.concat((pd.DataFrame(data=principals, columns=names), vector), axis=1)

    def exc(self, matrix: np.ndarray,  n_components: int = None) -> sklearn.decomposition.KernelPCA:
        """

        :param matrix: The matrix that will be decomposed
        :param n_components: The number of components required
        :return:
        """

        projector = sklearn.decomposition.KernelPCA(kernel='sigmoid', random_state=self.SEED,  n_components=n_components)
        projector.fit(X=matrix)

        return projector
