import os


# noinspection PyUnresolvedReferences,PyProtectedMember
class Config:

    def __init__(self):
        """
        self.root = os.path.abspath(__package__)


        """

        self.numeric = ['area', 'perimeter', 'majoraxislength', 'minoraxislength',  'aspectratio', 'eccentricity', 'convexarea', 'equivdiameter', 'extent',  'solidity', 'roundness',
                        'compactness', 'shapefactor1', 'shapefactor2', 'shapefactor3', 'shapefactor4', 'class']

        self.warehouse = os.path.join(os.getcwd(), 'warehouse')

        self.SEED = 5
