"""For one-hot and ordinal encoding"""
import logging

import pandas as pd


class Encode:
    """
    For OneHot & Ordinal Encoding
    """

    def __init__(self):
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.WARNING,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, blob: pd.DataFrame, field: str) -> pd.DataFrame:
        """
        For One Hot Encoding

        :param blob: A DataFrame of data
        :param field: The field that will be encoded
        :return:
        """

        # Get the one-hot-encodes of the dependent variable field
        conditions = pd.get_dummies(blob.copy()[field])

        # Values {0, 1} instead of {False, True}
        conditions = conditions.copy().astype(dtype=int)

        # Reconstruct the data
        try:
            data = pd.concat((blob.copy().drop(columns=field), conditions), axis=1, ignore_index=False)
        except RuntimeError as err:
            raise (self.__logger.warning(err)) from err

        return data
