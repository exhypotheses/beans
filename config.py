"""The global definitions"""
import os


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

        # The name of the dependent variable field
        self.dependent: str = 'class'
