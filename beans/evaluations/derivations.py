import pandas as pd


class Derivations:

    def __init__(self, frq: pd.DataFrame):
        """

        :param frq: For a specific class, and a series of thresholds, the tp, tn, fp, fn
                    frequencies.  All the frequency fields must exist, and must be
                    named tp, tn, fp, fn.
        """

        self.frq = frq

    def precision(self) -> pd.Series:
        """
        Positive Predictive Value
        """

        return self.frq.tp.truediv(self.frq.tp + self.frq.fp)

    def sensitivity(self) -> pd.Series:
        """"
        True Positive Rate, Recall, Hit rate
        """

        return self.frq.tp.truediv(self.frq.tp + self.frq.fn)

    def specificity(self) -> pd.Series:
        """
        True Negative Rate, Selectivity
        """

        return self.frq.tn.truediv(self.frq.tn + self.frq.fp)

    def fscore(self) -> pd.Series:
        """
        F1 Score, F Score, F Measure
        """

        ppv = self.precision()
        tpr = self.sensitivity()

        return 2 * (ppv.multiply(tpr)).truediv(ppv + tpr)

    def youden(self) -> pd.Series:
        """
        Youden's J Statistic, Youden's Index
        """

        tpr = self.sensitivity()
        tnr = self.specificity()

        return tpr + tnr - 1

    def matthews(self) -> pd.Series:
        """
        Matthews Correlation Coefficient

        :return:
        """

        numerator = self.frq.tp.multiply(self.frq.tn) - self.frq.fp.multiply(self.frq.fn)

        pcp = (self.frq.tp + self.frq.fp)
        tcp = (self.frq.tp + self.frq.fn)
        tcn = (self.frq.fp + self.frq.tn)
        pcn = (self.frq.fn + self.frq.tn)
        denominator = pcp.multiply(tcp).multiply(tcn).multiply(pcn).pow(0.5)

        return numerator.truediv(denominator)

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        data = self.frq
        data.loc[:, 'precision'] = self.precision()
        data.loc[:, 'sensitivity'] = self.sensitivity()
        data.loc[:, 'specificity'] = self.specificity()
        data.loc[:, 'fscore'] = self.fscore()
        data.loc[:, 'youden'] = self.youden()
        data.loc[:, 'matthews'] = self.matthews()

        return data
