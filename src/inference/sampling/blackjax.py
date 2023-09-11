"""
blackjax.py
"""
import os

import arviz
import jax
import pymc
import pymc.sampling_jax

import config


class BlackJAX:
    """
    The BlakJAX option
    """

    def __init__(self):
        """
        Constructor
        """

        # Use a GPU (Graphics Processing Unit); the NVIDIA unit.
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # Configurations
        configurations = config.Config()
        self.random_seed: int = configurations.seed

    def exc(self, model: pymc.Model, method: str) -> arviz.InferenceData:
        """

        :param model: A pymc.Model architecture
        :param method: The sampling method - parallel or vectorized
        :return:
        """

        if method == 'parallel':
            chains: int = jax.device_count(backend='gpu')
        else:
            chains = 4

        with model:
            # Inference
            trace: arviz.InferenceData = pymc.sampling_jax.sample_blackjax_nuts(
                draws=2000, tune=1000, chains=chains, target_accept=0.9,
                random_seed=self.random_seed, chain_method=method)

        return trace
