import collections

import pandas as pd
import numpy as np


class Set:

    def __init__(self):
        """
        Constructor
        """

        self.target = 'class'
        self.labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

    def attributes(self):
        """

        :return:
        """

        InstancesAttributes = collections.namedtuple(
            typename='InstancesAttributes', field_names=['url_training', 'url_testing', 'usecols', 'dtype', 'target'])

        url_training = 'https://raw.githubusercontent.com/exhypotheses/beans/develop/warehouse/splits/training.csv'
        url_testing = 'https://raw.githubusercontent.com/exhypotheses/beans/develop/warehouse/splits/testing.csv'

        usecols = ['area', 'perimeter', 'majoraxislength', 'minoraxislength', 'aspectratio', 'eccentricity',
                   'convexarea', 'equivdiameter', 'extent', 'solidity', 'roundness', 'compactness', 'shapefactor1',
                   'shapefactor2', 'shapefactor3', 'shapefactor4', 'class']

        dtype = {'area': np.int, 'perimeter': np.float, 'majoraxislength': np.float, 'minoraxislength': np.float,
                 'aspectratio': np.float, 'eccentricity': np.float, 'convexarea': np.int, 'equivdiameter': np.float,
                 'extent': np.float, 'solidity': np.float, 'roundness': np.float, 'compactness': np.float,
                 'shapefactor1': np.float, 'shapefactor2': np.float, 'shapefactor3': np.float, 'shapefactor4': np.float,
                 'class': str}

        return InstancesAttributes._make((url_training, url_testing, usecols, dtype, self.target))

    def training(self):
        """

        :return: The training data set
        """

        attributes = self.attributes()

        try:
            data_ = pd.read_csv(filepath_or_buffer=attributes.url_training, header=0, usecols=attributes.usecols,
                                dtype=attributes.dtype, encoding='utf-8')
        except OSError as err:
            raise Exception(err.strerror) in err

        return data_

    def testing(self):
        """

        :return: The testing data set
        """

        attributes = self.attributes()

        try:
            data_ = pd.read_csv(filepath_or_buffer=attributes.url_testing, header=0, usecols=attributes.usecols,
                                dtype=attributes.dtype, encoding='utf-8')
        except OSError as err:
            raise Exception(err.strerror) in err

        return data_
