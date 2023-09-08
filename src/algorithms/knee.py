"""Principal vectors knee"""
import pandas as pd
import sklearn.cluster
import yellowbrick

import config
import src.graphing.relational


class Knee:
    """
    Knee
        Determines the kneepoint 
    """

    def __init__(self):
        """
        Constructor
        """

        self.__seed = config.Config().seed

        self.relational = src.graphing.relational.Relational()

    def exc(self, blob: pd.DataFrame) -> int:
        """

        :param blob: The data matrix that will undergo dimension reduction
        :return:
        """

        axes = self.relational.figure(width=3.9, height=3.3)

        # A rough estimate of the number of effective clusters that <blob> can be divided into
        kmc = sklearn.cluster.KMeans(random_state=self.__seed, max_iter=1000, algorithm='full')
        ybc = yellowbrick.cluster.KElbowVisualizer(estimator=kmc, k=(3, blob.shape[1]), metric='distortion', timings=False, ax=axes)
        ybc.fit(X=blob)
        axes.figure.clf()

        return ybc.elbow_value_
