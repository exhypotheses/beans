import pandas as pd
import numpy as np
import collections


# noinspection PyUnresolvedReferences,PyProtectedMember
class Modelling:

    def __init__(self):
        """
        Constructor
        """

        self.target = 'class'
        self.labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

    def attributes(self):

        InstancesAttributes = collections.namedtuple(
            typename='InstancesAttributes', field_names=['url', 'usecols', 'dtype', 'target'])

        url = 'https://raw.githubusercontent.com/exhypotheses/beans/develop/warehouse/data/modelling.csv'

        usecols = ['area', 'perimeter', 'majoraxislength', 'minoraxislength', 'aspectratio', 'eccentricity', 'convexarea',
                   'equivdiameter', 'extent', 'solidity', 'roundness', 'compactness', 'shapefactor1', 'shapefactor2', 'shapefactor3',
                   'shapefactor4', 'class']

        dtype = {'area': np.int, 'perimeter': np.float, 'majoraxislength': np.float, 'minoraxislength': np.float, 'aspectratio': np.float,
                 'eccentricity': np.float, 'convexarea': np.int, 'equivdiameter': np.float, 'extent': np.float, 'solidity': np.float,
                 'roundness': np.float, 'compactness': np.float, 'shapefactor1': np.float, 'shapefactor2': np.float, 'shapefactor3': np.float,
                 'shapefactor4': np.float, 'class': str}

        return InstancesAttributes._make((url, usecols, dtype, self.target))

    def __weights(self, blob: pd.DataFrame):
        """

        :param blob:
        :return:
        """

        weight_values = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=self.labels, y=blob[self.target])

        return dict(zip(labels, weight_values))

    def data(self):

        attributes = self.attributes()

        try:
            data_ = pd.read_csv(filepath_or_buffer=attributes.url, header=0, usecols=attributes.usecols, dtype=attributes.dtype, encoding='utf-8')
        except OSError as err:
            raise Exception(err.strerror) in err

        weights_ = self.__weights(blob=data_)

        return data_, weights_