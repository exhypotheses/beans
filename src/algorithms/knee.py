import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

import beans.graphing.relational

import config


class Knee:

    def __init__(self):
        """
        Constructor
        """

        configurations = config.Config()
        self.SEED = configurations.SEED

        self.relational = beans.graphing.relational.Relational()

    def exc(self, blob: pd.DataFrame, target: str) -> int:
        """

        :param blob: The data set that will be sampled
        :param target: The labels field
        :return:
        """

        ax = self.relational.figure(width=3.9, height=3.3)

        # A rough estimate of the number of effective clusters that <blob> can be divided into
        kmc = sklearn.cluster.KMeans(random_state=self.SEED, max_iter=1000, algorithm='full')
        ybc = yellowbrick.cluster.KElbowVisualizer(estimator=kmc, k=(3, 16), metric='distortion', timings=False, ax=ax)
        ybc.fit(X=blob.drop(columns=target))
        ax.figure.clf()

        return ybc.elbow_value_
