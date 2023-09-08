"""A neural network model"""
import pymc
import numpy as np
import pytensor

import config


class Neural:
    """
    Neural
        A Bayesian neural network model for beans classification
    """

    def __init__(self):
        """
        Constructor
        """

        configurations = config.Config()
        self.rng = np.random.default_rng(seed=configurations.seed)

    @staticmethod
    def inference_(model: pymc.Model, n_iterations: int):
        """
        The pymc.fit( ) **kwargs options are of format -> **{'obj_n_mc': 1000, ...}

        :param model:
        :param n_iterations:
        :return:
        """

        with model:
            inference = pymc.FullRankADVI()
            approximation = pymc.fit(n=n_iterations, method=inference)

            return approximation, inference

    def model_(self, features: np.ndarray, output: np.ndarray) -> pymc.Model:
        """
        A neural network model consisting of an input layer, two hidden layers, and an output layer

        :param features:  The feature's tensor, including the bias cells
        :param output:  During a training step, this is the tensor of expected outputs
        :return:
        """

        # Architecture
        n_hidden_1 = 6
        n_hidden_2 = 5

        init_1 = self.rng.standard_normal(size=(features.shape[1], n_hidden_1)).astype(pytensor.config.floatX)
        init_2 = self.rng.standard_normal(size=(n_hidden_1, n_hidden_2)).astype(pytensor.config.floatX)
        init_out = self.rng.standard_normal(size=(n_hidden_2, output.shape[1])).astype(pytensor.config.floatX)

        # Coordinates


        # Model
        with pymc.Model() as network:          

            # features & Output
            ann_input = pymc.Data('ann_input', features)
            ann_output = pymc.Data('ann_output', output)

            # Weights from input to hidden layer
            weights_in_1 = pymc.StudentT(name='w_in_1', nu=5, mu=0, sigma=2.5, shape=(features.shape[1], n_hidden_1),
                                          testval=init_1)

            # Weights from 1st to 2nd layer
            weights_1_2 = pymc.StudentT(name='w_1_2', nu=5, mu=0, sigma=2.5, shape=(n_hidden_1, n_hidden_2), testval=init_2)

            # Weights from hidden layer to output
            weights_2_out = pymc.Normal(name='w_2_out', mu=0, sigma=1.5, shape=(n_hidden_2, output.shape[1]),
                                           testval=init_out)

            # Build neural-network using tanh activation function
            act_1 = pymc.math.tanh(pymc.math.dot(ann_input, weights_in_1))
            act_2 = pymc.math.tanh(pymc.math.dot(act_1, weights_1_2))
            act_out = pymc.math.sigmoid(pymc.math.dot(act_2, weights_2_out))

            # Inference
            pymc.Bernoulli(name='out', p=act_out, observed=ann_output, total_size=output.shape[0])

        return network
