import json
import hmac
import hashlib
import pickle


class Inspect:

    def __init__(self):
        """

        """

    @staticmethod
    def kappa(path_of_kappa: str) -> bytes:
        """

        :param path_of_kappa: The path to the kappa file (JSON)
        :return:
        """

        f = open(path_of_kappa)
        value = json.load(f)
        f.close()

        return bytes.fromhex(value['kappa'])

    @staticmethod
    def read(path_of_pickle: str):
        """

        :param path_of_pickle: The path to the pickled model+
        :return:
        """

        with open(path_of_pickle, 'rb') as f:
            digest = f.readline()
            pickled = f.read()

        digest = digest.rstrip(b'\n')

        return digest, pickled

    def exc(self, path_of_kappa: str, path_of_pickle: str):
        """

        :param path_of_kappa: The path to the kappa file (JSON)
        :param path_of_pickle: The path to the pickled model+
        :return:
        """

        kappa = self.kappa(path_of_kappa=path_of_kappa)
        digest, pickled = self.read(path_of_pickle=path_of_pickle)

        # Re-design: Different, and absent, digest option.
        recomputed = hmac.digest(kappa, pickled, digest=hashlib.sha384)

        if not hmac.compare_digest(digest, recomputed):
            raise Exception('unverifiable byte stream')
        else:
            return pickle.loads(pickled)            
