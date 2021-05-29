import numpy as np

import sklearn.decomposition


class Project:

    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def apply(matrix: np.ndarray, projector: sklearn.decomposition.KernelPCA):
        """

        :param matrix:
        :param projector:
        :return:
        """

        return projector.transform(X=matrix)

    @staticmethod
    def exc(matrix: np.ndarray) -> sklearn.decomposition.KernelPCA:
        """

        :param matrix:
        :return:
        """

        projector = sklearn.decomposition.KernelPCA(kernel='cosine', random_state=5)
        projector.fit(X=matrix)

        return projector
