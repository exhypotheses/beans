"""For one-hot and ordinal encoding"""
import logging

import pandas as pd

import config


class Encode:
    """
    For OneHot & Ordinal Encoding
    """

    def __init__(self):
        """
        Constructor
        """

        self.__meta = config.Config().meta

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, blob: pd.DataFrame) -> pd.DataFrame:
        """
        For One Hot Encoding

        :param blob: A DataFrame of data
        :return:
        """

        # Get the one-hot-encodes of the dependent variable field
        conditions = pd.get_dummies(blob.copy()[self.__meta.dependent])

        # Values {0, 1} instead of {False, True}
        conditions = conditions.copy().astype(dtype=int)

        # Reconstruct the data
        data = pd.concat((blob.copy().drop(columns=self.__meta.dependent), conditions), 
                         axis=1, ignore_index=False)

        self.__logger.info('%s', data.info())

        return data
