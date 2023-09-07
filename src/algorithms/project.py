"""Projecting"""
import numpy as np
import pandas as pd

import sklearn.decomposition as skd

import config


class Project:
    """
    Project
    """

    def __init__(self):
        """
        Constructor
        """

        configurations = config.Config()
        self.__seed = configurations.seed
        self.__meta = configurations.meta

    @staticmethod
    def __project(matrix: np.ndarray, vector: pd.Series, projector: skd.KernelPCA) -> pd.DataFrame:
        """

        :param matrix: The independent variables matrix
        :param vector: The dependent variable vector
        :param projector: A transformer
        :return:
        """

        principals = projector.transform(X=matrix)
        names = ['kpc_' + str(i).zfill(2) for i in range(1, 1 + principals.shape[1])]

        return pd.concat((pd.DataFrame(data=principals, columns=names), vector), axis=1)

    def __projector(self, matrix: np.ndarray,  n_components: int = None) -> skd.KernelPCA:
        """
        
        :param matrix: The independent variables matrix
        :param n_components: The number of components of interest
        """

        projector = skd.KernelPCA(kernel='cosine', random_state=self.__seed,  n_components=n_components)
        projector.fit(X=matrix)

        return projector

    def exc(self, blob: pd.DataFrame,  n_components: int = None, projector: skd.KernelPCA = None) -> (
            pd.DataFrame, skd.KernelPCA):
        """

        :param blob: 
        :param n_components: The number of components of interest
        :return:
        """

        # The matrix that will be decomposed
        matrix: np.ndarray = blob.copy().drop(columns=self.__meta.dependent).array

        # The parallel dependent variable vector, as a series
        vector: pd.Series = blob.copy()[self.__meta.dependent]

        # The projector
        if projector is None:
            projector = self.__projector(matrix=matrix, n_components=n_components)

        # Projecting
        projected: pd.DataFrame = self.__project(matrix=matrix, vector=vector, projector=projector)

        return projected, projector
