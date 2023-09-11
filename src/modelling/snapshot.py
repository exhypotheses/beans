"""A neural network model"""
import collections
import pymc
import numpy as np
import pytensor

import config


class Snapshot:
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

    def __arc(self, d_features: np.ndarray, d_output: np.ndarray):

        Arc = collections.namedtuple(
            typename='Arc',
            field_names=['n_hidden_layer_1', 'n_hidden_layer_2', 'init_1', 'init_2', 'init_out', 'coordinates'])

        # Architecture
        n_hidden_layer_1 = 6
        n_hidden_layer_2 = 5

        return Arc(n_hidden_layer_1=n_hidden_layer_1, n_hidden_layer_2=n_hidden_layer_2,
                 init_1 = self.rng.standard_normal(size=(d_features[1], n_hidden_layer_1)).astype(pytensor.config.floatX),
                 init_2 = self.rng.standard_normal(size=(n_hidden_layer_1, n_hidden_layer_2)).astype(pytensor.config.floatX),
                 init_out = self.rng.standard_normal(size=(n_hidden_layer_2, d_output[1])).astype(pytensor.config.floatX),
                 coordinates = {
                     'i_hidden_layer_1': np.arange(n_hidden_layer_1),
                     'i_hidden_layer_2': np.arange(n_hidden_layer_2),
                     'i_features': np.arange(d_features[1]),
                     'i_observations': np.arange(d_features[0]),
                     'i_labels': np.arange(d_output[1])}
        )

    def model_(self, features: np.ndarray, output: np.ndarray) -> pymc.Model:
        """
        A neural network model consisting of an input layer, two hidden layers, and an output layer

        :param features:  The feature's tensor, including the bias cells
        :param output:  During a training step, this is the tensor of expected outputs
        :return:
        """

        arc = self.__arc(d_features=features.shape, d_output=output.shape)

        # Model
        with pymc.Model(coords=arc.coordinates) as network:

            # features & Output
            ann_input = pymc.Data('ann_input', features, mutable=True, dims=('i_observations', 'i_features'))
            ann_output = pymc.Data('ann_output', output, mutable=True, dims=('i_observations', 'i_labels'))

            # Weights from input to first hidden layer
            weights_in_1 = pymc.StudentT(name='w_in_1', nu=5, mu=0, sigma=2.5, 
                                         dims=('i_features', 'i_hidden_layer_1'), initval=arc.init_1)

            # Weights from first hidden layer -> second hidden layer
            weights_1_2 = pymc.StudentT(name='w_1_2', nu=5, mu=0, sigma=2.5, 
                                        dims=('i_hidden_layer_1', 'i_hidden_layer_2'), initval=arc.init_2)

            # Weights from second hidden layer to output
            weights_2_out = pymc.Normal(name='w_2_out', mu=0, sigma=1.5, 
                                        dims=('i_hidden_layer_2', 'i_labels'), initval=arc.init_out)

            # Build neural-network using tanh activation function
            act_1 = pymc.math.tanh(pymc.math.dot(ann_input, weights_in_1))
            act_2 = pymc.math.tanh(pymc.math.dot(act_1, weights_1_2))
            act_out = pymc.math.sigmoid(pymc.math.dot(act_2, weights_2_out))

            # Inference
            pymc.Bernoulli(name='out', p=act_out, observed=ann_output, total_size=output.shape[0])

        return network
