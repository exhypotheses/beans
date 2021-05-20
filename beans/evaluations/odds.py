import numpy as np
import pandas as pd


class Odds:

    def __init__(self, probabilities_: np.ndarray):
        """

        :param probabilities_: The probability points at which odds ratio ranges should be calculated
        """

        # Probabilities
        probabilities = pd.DataFrame(
            data={'probability': probabilities_})

        # Percentiles
        lower = 100 * 0.5 * (1 - probabilities['probability'])
        upper = 100 - lower

        self.points = pd.concat(
            (probabilities, lower.rename('lower_percentile'), upper.rename('upper_percentile')), axis=1)

    @staticmethod
    def exponentials(trace_: np.ndarray, quantiles: np.ndarray):
        """

        :param trace_: The set of measures whose set of exp(percentile(...)) values will be calculated
        :param quantiles: The tile points to be calculated
        :return:
        """

        return np.exp(np.percentile(a=trace_, q=quantiles))

    def exc(self, trace_):
        """
        A credible interval -> an odds ratio range: The calculations herein include the median
        of each odds ratio range

        :param trace_: A set of trace measures
        :return:
        """

        points = self.points
        points.loc[:, 'lower_odds_ratio'] = self.exponentials(trace_=trace_, quantiles=points['lower_percentile'])
        points.loc[:, 'upper_odds_ratio'] = self.exponentials(trace_=trace_, quantiles=points['upper_percentile'])
        points.loc[:, 'median_odds_ratio'] = points[['lower_odds_ratio', 'upper_odds_ratio']].median(axis=1)

        return points
