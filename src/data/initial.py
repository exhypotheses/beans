"""For beans data reading ..."""
import os

import numpy as np
import pandas as pd

import src.functions.streams


class Read:
    """
    Read
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.__uri = os.path.join(os.getcwd(), 'data', 'beans.csv')

        self.__rename = {
            'Area': 'area', 'Perimeter': 'perimeter', 'MajorAxisLength': 'major_axis_length', 
            'MinorAxisLength': 'minor_axis_length', 'AspectRation': 'aspect_ratio', 'Eccentricity': 'eccentricity', 
            'ConvexArea': 'convex_area', 'EquivDiameter': 'equiv_diameter', 'Extent': 'extent', 'Solidity': 'solidity',
            'roundness': 'roundness', 'Compactness': 'compactness', 'ShapeFactor1': 'shape_factor_1', 
            'ShapeFactor2': 'shape_factor_2', 'ShapeFactor3': 'shape_factor_3', 'ShapeFactor4': 'shape_factor_4', 
            'Class': 'class'}

        self.__dtype = {
            'Area': int, 'Perimeter': np.float64, 'MajorAxisLength': np.float64, 
            'MinorAxisLength': np.float64, 'AspectRation': np.float64, 'Eccentricity': np.float64, 
            'ConvexArea': int, 'EquivDiameter': np.float64, 'Extent': np.float64, 'Solidity': np.float64,
            'roundness': np.float64, 'Compactness': np.float64, 'ShapeFactor1': np.float64, 
            'ShapeFactor2': np.float64, 'ShapeFactor3': np.float64, 'ShapeFactor4': np.float64, 'Class': str}

    def __read(self) -> pd.DataFrame:
        """
        
        :return:
        """

        return src.functions.streams.Streams().read(
            uri=self.__uri, header=0, usecols=self.__dtype.keys(), dtype=self.__dtype)

    def exc(self) -> pd.DataFrame:
        """
        
        :return:
        """

        data = self.__read()
        data.rename(columns=self.__rename, inplace=True)

        return data
