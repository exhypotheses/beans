"""Structures for storing data"""
import collections


class Structures:
    """
    Sructures
    """

    Training = collections.namedtuple(
        typename='Training',
        field_names=['data', 'scaler', 'scaled', 'projector', 'projected', 'encoded', 'labels'])

    Testing = collections.namedtuple(
        typename='Testing', field_names=['data', 'scaled'])

    def __init__(self) -> None:
        pass
