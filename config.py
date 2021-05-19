import os


# noinspection PyUnresolvedReferences,PyProtectedMember
class Config:

    def __init__(self):
        """
        self.root = os.path.abspath(__package__)


        """

        self.warehouse = os.path.join(os.getcwd(), 'warehouse')
