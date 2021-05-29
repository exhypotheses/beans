import numpy as np

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
    def apply(matrix: np.ndarray, projector: sklearn.decomposition.KernelPCA):
        """

        :param matrix:
        :param projector:
        :return:
        """

        return projector.transform(X=matrix)

    def exc(self, matrix: np.ndarray,  n_components: int = None) -> sklearn.decomposition.KernelPCA:
        """

        :param matrix: The matrix that will be decomposed
        :param n_components: The number of components required
        :return:
        """

        projector = sklearn.decomposition.KernelPCA(kernel='cosine', random_state=self.SEED,  n_components=n_components)
        projector.fit(X=matrix)

        return projector
