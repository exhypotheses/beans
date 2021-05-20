import pandas as pd
import seaborn as sns
import collections

import beans.graphics.relational


# noinspection PyUnresolvedReferences,PyProtectedMember
class Derivations:

    def __init__(self, derivations: pd.DataFrame):
        """

        :param derivations:
        """

        self.derivations = derivations

        self.relational = beans.graphics.relational.Relational()
        self.RelationalGraphLabels = collections.namedtuple(
            typename='RelationalGraphLabels', field_names=['title', 'xlabel', 'ylabel'])

    def optimal(self):
        """

        :return: The threshold at which error matrix frequencies, and derivations, are used for
                 model analysis & predictions
        """

        return self.derivations.iloc[self.derivations.matthews.idxmax(), :].threshold

    def excerpt(self):
        """

        :return:
        """

        data = self.derivations.set_index(keys='threshold')

        return data[['precision', 'sensitivity', 'specificity', 'fscore', 'youden', 'matthews']]

    def exc(self):
        """

        :return:
        """

        optimal = self.optimal()
        excerpt = self.excerpt()

        ax = self.relational.figure(width=4.0, height=3.1)
        sns.lineplot(data=excerpt)
        ax.axvline(x=optimal, alpha=0.25)

        self.relational.annotation(
            handle=ax,
            labels=self.RelationalGraphLabels._make(['\nMeasures\n', '\nthreshold', 'measure\n']))

        ax.set_xlim(left=0.1, right=1.4)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc='center right', fontsize='small')
