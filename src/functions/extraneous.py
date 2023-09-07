"""Delete __pycache__"""
import logging
import pathlib
import shutil


class Extraneous:
    """
    Extraneous
    """

    def __init__(self) -> None:
        """
        Constructor
        """

        # Logging
        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def extraneous(self):
        """
        
        :return:
        """

        for path in pathlib.Path.cwd().rglob('__pycache__'):
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                except PermissionError as err:
                    raise (err) from err

                self.__logger.info('Deleted: %s', path)
