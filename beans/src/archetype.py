import collections


class Archetype:

    def __init__(self):
        """
        The details of the original dataset
            http://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

        The  original zip file
            http://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip

        The develop's copy of this original zip file
            https://github.com/exhypotheses/beans/raw/develop/data/beans.zip
        """

    @staticmethod
    def attributes():
        """
        :return:
        """

        InstancesAttributes = collections.namedtuple(
            typename='InstancesAttributes', field_names=['url', 'labels'])

        # A copy of this original zip file
        url = 'https://raw.githubusercontent.com/exhypotheses/beans/develop/data/beans.csv'

        # The field that hosts the labels/classes
        labels = 'Class'

        return InstancesAttributes._make((url, labels))
