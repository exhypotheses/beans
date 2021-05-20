
class Archetype:

    def __init__(self):
        """
        The details of the original dataset
            http://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
        """

        # A copy of this original zip file is available at https://github.com/exhypotheses/beans/tree/develop/data
        self.url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip'

        # The field the hosts the labels/classes
        self.field_of_classes = 'Class'
