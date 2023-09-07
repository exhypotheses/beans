"""Structures for storing data"""
import collections


class Structures:
    """
    Sructures
    """

    Training = collections.namedtuple(
        typename='Training',
        field_names=['data', 'scaler', 'scaled', 'encoded', 'projected'],
        defaults=[None, None, None, None, None])

    Testing = collections.namedtuple(
        typename='Testing', field_names=['data', 'scaled'])

    def __init__(self) -> None:
        pass
