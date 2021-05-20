import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import beans.graphics.relational


class Odds:

    def __init__(self):
        """
        Constructor
        """

        self.relational = beans.graphics.relational.Relational()

    def exc(self, odds: pd.DataFrame, estimates: np.ndarray, name: str):
        """
        odds = beans.evaluation.odds.Odds(probabilities_=...).exc(trace_=trace[variable] .)
        estimates = trace[variable]

        :param odds:
        :param estimates:
        :param name: A comprehensible, concise, text that describes the 'variable' in focus
        :return:
        """

        plt.rcParams.update({'ytick.labelsize': 11, 'xtick.labelsize': 11, 'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                             'figure.constrained_layout.wspace': 0.35})

        self.relational.figure(width=6.1, height=2.6)

        left = plt.subplot(1, 2, 1)
        sns.lineplot(data=odds, x='probability', y='lower_odds_ratio', ax=left, color='#FF7F00')
        sns.lineplot(data=odds, x='probability', y='upper_odds_ratio', ax=left, color='#FF7F00')
        left.set_ylabel(ylabel='odds ratio')
        left.set_title('\n\n{}\n'.format(name))

        right = plt.subplot(1, 2, 2)
        sns.histplot(x=np.exp(estimates), stat='density', ax=right)
        right.set_xlabel(xlabel='odds ratio')
        right.set_title('\n\n{}\n'.format(' '))
