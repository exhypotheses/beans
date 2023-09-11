"""
Exploring a named tuple class for testing data, and complementary objects.
"""
import typing

import pandas as pd


class Testing(typing.NamedTuple):
    """
    
    :return: Named tuple class
    """

    data: pd.DataFrame = None
    scaled: pd.DataFrame = None
