import pandas as pd
import numpy as np
import collections

import sklearn.utils


# noinspection PyUnresolvedReferences,PyProtectedMember
class Modelling:

    def __init__(self, basename: str):
        """
        Constructor
        """

        self.basename = basename
        self.target = 'class'

    def attributes(self) -> collections.namedtuple:

        InstancesAttributes = collections.namedtuple(
            typename='InstancesAttributes', field_names=['url', 'usecols', 'dtype', 'target'])

        url = 'https://raw.githubusercontent.com/exhypotheses/beans/develop/warehouse/data/{}'.format(self.basename)

        usecols = ['area', 'perimeter', 'majoraxislength', 'minoraxislength', 'aspectratio', 'eccentricity',
                   'convexarea', 'equivdiameter', 'extent', 'solidity', 'roundness', 'compactness', 'shapefactor1',
                   'shapefactor2', 'shapefactor3', 'shapefactor4', 'class']

        dtype = {'area': np.int, 'perimeter': np.float, 'majoraxislength': np.float, 'minoraxislength': np.float,
                 'aspectratio': np.float, 'eccentricity': np.float, 'convexarea': np.int, 'equivdiameter': np.float,
                 'extent': np.float, 'solidity': np.float, 'roundness': np.float, 'compactness': np.float,
                 'shapefactor1': np.float, 'shapefactor2': np.float, 'shapefactor3': np.float, 'shapefactor4': np.float,
                 'class': str}

        return InstancesAttributes._make((url, usecols, dtype, self.target))

    def __weights(self, blob: pd.DataFrame, labels: list) -> dict:
        """

        :param blob:
        :param labels:
        :return:
        """

        weight_values = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=labels, y=blob[self.target])

        return dict(zip(labels, weight_values))

    def data(self) -> (pd.DataFrame, dict, list):
        """

        :return: The data set, data weights w.r.t. instances per label
        """

        attributes = self.attributes()

        try:
            data_ = pd.read_csv(filepath_or_buffer=attributes.url, header=0, usecols=attributes.usecols,
                                dtype=attributes.dtype, encoding='utf-8')
        except OSError as err:
            raise Exception(err.strerror) in err

        labels = data_[self.target].unique()

        weights_ = self.__weights(blob=data_, labels=labels)

        return data_, weights_, labels
