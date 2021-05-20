import pandas as pd


class Binary:

    def __init__(self, contingencies: pd.DataFrame, labels: str):
        """

        :param contingencies:
            This frame must include the fields
                segment -> which has values tp, tn, fp, or fn only
                threshold -> the threshold at which the error matrix frequency was calculated
            and the field denoted by variable 'labels'

        :param labels: The name of the field of labels
        """

        self.labels = labels
        self.contingencies = contingencies[['segment', self.labels, 'threshold']]

    def structure(self):
        """

        :return
        """

        pivoted = pd.pivot_table(data=self.contingencies, columns='segment', values=self.labels, index='threshold')
        pivoted.reset_index(drop=False, inplace=True)

        return pd.DataFrame(data={'threshold': pivoted.threshold, 'tp': pivoted.tp, 'fn': pivoted.fn,
                                  'tn': pivoted.tn, 'fp': pivoted.fp})

    def exc(self):
        """

        :return
        """

        return self.structure()
