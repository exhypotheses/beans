import pandas as pd
import numpy as np
import collections


# noinspection PyUnresolvedReferences,PyProtectedMember
class Modelling:

    def __init__(self):
        """
        Constructor
        """

        self.field_of_labels = 'class'
        self.labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

    @staticmethod
    def attributes():

        InstancesAttributes = collections.namedtuple(
            typename='InstancesAttributes', field_names=['url', 'usecols', 'dtype', 'field_of_labels'])

        url = 'https://raw.githubusercontent.com/exhypotheses/beans/develop/data/beans.csv'

        usecols = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                   'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']

        dtype = {'Area': np.int, 'Perimeter': np.float, 'MajorAxisLength': np.float, 'MinorAxisLength': np.float, 'AspectRation': np.float,
                 'Eccentricity': np.float, 'ConvexArea': np.int, 'EquivDiameter': np.float, 'Extent': np.float, 'Solidity': np.float,
                 'roundness': np.float, 'Compactness': np.float, 'ShapeFactor1': np.float, 'ShapeFactor2': np.float, 'ShapeFactor3': np.float,
                 'ShapeFactor4': np.float, 'Class': str}

        field_of_labels = 'Class'

        return InstancesAttributes._make((url, usecols, dtype, field_of_labels))

    def data(self):

        attributes = self.attributes()

        try:
            data_ = pd.read_csv(filepath_or_buffer=attributes.url, header=0, usecols=attributes.usecols, dtype=attributes.dtype, encoding='utf-8')
        except OSError as err:
            raise Exception(err.strerror) in err

        data_.rename(str.lower, axis=1, inplace=True)

        return data_