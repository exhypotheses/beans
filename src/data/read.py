import numpy as np
import pandas as pd

import os

import src.functions.streams

class Read:

    def __init__(self) -> None:

        self.__uri = os.path.join(os.getcwd(), 'data', 'beans.csv')

        self.__rename = {'Area': 'area', 'Perimeter': 'perimeter', 'MajorAxisLength': 'major_axis_length', 
                         'MinorAxisLength': 'minor_axis_length', 'AspectRation': 'aspect_ratio', 'Eccentricity': 'eccentricity', 
                         'ConvexArea': 'convex_area', 'EquivDiameter': 'equiv_diameter', 'Extent': 'extent', 'Solidity': 'solidity',
                         'roundness': 'roundness', 'Compactness': 'compactness', 'ShapeFactor1': 'shape_factor_1', 
                         'ShapeFactor2': 'shape_factor_2', 'ShapeFactor3': 'shape_factor_3', 'ShapeFactor4': 'shape_factor_4', 
                         'Class': 'class'}
        
        self.__dtype = {'Area': np.int, 'Perimeter': np.float, 'MajorAxisLength': np.float, 
                        'MinorAxisLength': np.float, 'AspectRation': np.float, 'Eccentricity': np.float, 
                        'ConvexArea': np.int, 'EquivDiameter': np.float, 'Extent': np.float, 'Solidity': np.float,
                        'roundness': np.float, 'Compactness': np.float, 'ShapeFactor1': np.float, 
                        'ShapeFactor2': np.float, 'ShapeFactor3': np.float, 'ShapeFactor4': np.float, 'Class': str}

    def __read(self) -> pd.DataFrame:

        return src.functions.streams.Streams().read(
            uri=self.__uri, header=0, usecols=self.__dtype.keys(), dtype=self.__dtype)

    def exc(self):

        data = self.__read()
        data.rename(columns=self.__rename, inplace=True)



