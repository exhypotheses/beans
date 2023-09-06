"""The global definitions"""
import os
import numpy as np


class Config:
    """
    Config
    """

    def __init__(self):
        """
        Constructor
        """

        # The storage directory for all outputs
        self.warehouse: str = os.path.join(os.getcwd(), 'warehouse')

        # The seed number for all algorithms
        self.seed: int = 5

        # The beans data
        self.uri: str = os.path.join(os.getcwd(), 'data', 'beans.csv')

        # The original field names, and their data types
        self.dtype = {
            'Area': int, 'Perimeter': np.float64, 'MajorAxisLength': np.float64, 
            'MinorAxisLength': np.float64, 'AspectRation': np.float64, 'Eccentricity': np.float64, 
            'ConvexArea': int, 'EquivDiameter': np.float64, 'Extent': np.float64, 'Solidity': np.float64,
            'roundness': np.float64, 'Compactness': np.float64, 'ShapeFactor1': np.float64, 
            'ShapeFactor2': np.float64, 'ShapeFactor3': np.float64, 'ShapeFactor4': np.float64, 'Class': str}

        # Renaming ...
        self.rename: dict[str, str] = {
            'Area': 'area', 'Perimeter': 'perimeter', 'MajorAxisLength': 'major_axis_length', 
            'MinorAxisLength': 'minor_axis_length', 'AspectRation': 'aspect_ratio', 'Eccentricity': 'eccentricity', 
            'ConvexArea': 'convex_area', 'EquivDiameter': 'equiv_diameter', 'Extent': 'extent', 'Solidity': 'solidity',
            'roundness': 'roundness', 'Compactness': 'compactness', 'ShapeFactor1': 'shape_factor_1', 
            'ShapeFactor2': 'shape_factor_2', 'ShapeFactor3': 'shape_factor_3', 'ShapeFactor4': 'shape_factor_4', 
            'Class': 'class'}

        # The name of the dependent variable field
        self.dependent: str = 'class'

        # The numeric fields
        self.numeric = ['area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'aspect_ratio', 'eccentricity',
                         'convex_area', 'equiv_diameter', 'extent', 'solidity', 'roundness', 'compactness', 'shape_factor_1', 
                         'shape_factor_2', 'shape_factor_3', 'shape_factor_4']
