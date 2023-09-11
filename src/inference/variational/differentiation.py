"""Variational Inference"""
import pymc


class Differentiation:
    """
    Differentiation
        This class if for the differentiation options Automatic Differentiation Variational
        Inference (ADVI) & Full Rank Automatic Differentiation Variational Inference (ADVI)
    """

    def __init__(self) -> None:
        pass

    def full_rank_advi(self, model: pymc.Model, n_iterations: int):
        """
        The pymc.fit( ) **kwargs options are of format -> **{'obj_n_mc': 1000, ...}

        :param model:
        :param n_iterations:
        :return:
        """

        with model:
            inference = pymc.FullRankADVI()
            trace = pymc.fit(n=n_iterations, method=inference)

        return trace