import pymc3
import numpy as np
import theano

import pymc3.theanof

import config


class Neural:

    def __init__(self):
        """
        Constructor

        Options:
            * pymc3.theanof.set_tt_rng(pymc3.theanof.MRG_RandomStreams(seed=configurations.SEED))
        """

        configurations = config.Config()
        self.rng = np.random.default_rng(seed=configurations.SEED)

    @staticmethod
    def inference_(model: pymc3.Model, n_iterations: int):
        """
        The pymc3.fit( ) **kwargs options are of format -> **{'obj_n_mc': 1000, ...}

        :param model:
        :param n_iterations:
        :return:
        """

        with model:
            inference = pymc3.FullRankADVI()
            approximation = pymc3.fit(n=n_iterations, method=inference)

            return approximation, inference

    def model_(self, features: np.ndarray, output: np.ndarray) -> pymc3.Model:
        """

        :param features:  The features tensor, including the bias cells
        :param output:  During a training step, this is the tensor of expected outputs
        :return:
        """

        # Model
        with pymc3.Model() as network:
            """
            A neural network model consisting of an input layer, two hidden layers, and an output layer
            """

            # Architecture
            hidden_1 = 6
            hidden_2 = 5

            init_1 = self.rng.standard_normal(size=(features.shape[1], hidden_1)).astype(theano.config.floatX)
            init_2 = self.rng.standard_normal(size=(hidden_1, hidden_2)).astype(theano.config.floatX)
            init_out = self.rng.standard_normal(size=(hidden_2, output.shape[1])).astype(theano.config.floatX)

            # features & Output
            ann_input = pymc3.Data('ann_input', features)
            ann_output = pymc3.Data('ann_output', output)

            # Weights from input to hidden layer
            weights_in_1 = pymc3.StudentT(name='w_in_1', nu=5, mu=0, sigma=2.5, shape=(features.shape[1], hidden_1),
                                          testval=init_1)

            # Weights from 1st to 2nd layer
            weights_1_2 = pymc3.StudentT(name='w_1_2', nu=5, mu=0, sigma=2.5, shape=(hidden_1, hidden_2), testval=init_2)

            # Weights from hidden layer to output
            weights_2_out = pymc3.Normal(name='w_2_out', mu=0, sigma=1.5, shape=(hidden_2, output.shape[1]),
                                           testval=init_out)

            # Build neural-network using tanh activation function
            act_1 = pymc3.math.tanh(pymc3.math.dot(ann_input, weights_in_1))
            act_2 = pymc3.math.tanh(pymc3.math.dot(act_1, weights_1_2))
            act_out = pymc3.math.sigmoid(pymc3.math.dot(act_2, weights_2_out))

            # Inference
            pymc3.Bernoulli(name='out', p=act_out, observed=ann_output, total_size=output.shape[0])

        return network
