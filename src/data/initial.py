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

        configurations = config.Config()
        self.__uri: str = configurations.uri
        self.__rename: dict[str, str] = configurations.rename
        self.__dtype = configurations.dtype

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

        data: pd.DataFrame = self.__read()
        data.rename(columns=self.__rename, inplace=True)

        return data
