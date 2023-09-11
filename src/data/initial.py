"""For beans data reading ..."""
import pandas as pd

import config
import src.functions.streams


class Read:
    """
    Read
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        self.__meta = config.Config().meta

    def __read(self) -> pd.DataFrame:
        """

        :return:
        """

        return src.functions.streams.Streams().read(
            uri=self.__meta.uri, header=0, usecols=self.__meta.dtype.keys(), dtype=self.__meta.dtype)

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        data: pd.DataFrame = self.__read()
        data.rename(columns=self.__meta.rename, inplace=True)

        return data
