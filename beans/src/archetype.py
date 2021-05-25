import collections


class Archetype:

    def __init__(self):
        """
        The details of the original dataset
            http://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

        The  original zip file
            http://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip

        The developer's copy of this original zip file
            https://github.com/miscellane/hub/raw/develop/data/beans/beans.zip
        """

    @staticmethod
    def attributes():
        """
        :return:
        """

        InstancesAttributes = collections.namedtuple(
            typename='InstancesAttributes', field_names=['url', 'labels'])

        # The CSV version of the beans data
        url = 'https://raw.githubusercontent.com/miscellane/hub/develop/data/beans/beans.csv'

        # The field that hosts the labels/classes
        labels = 'Class'

        return InstancesAttributes._make((url, labels))
