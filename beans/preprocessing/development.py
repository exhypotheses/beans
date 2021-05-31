import pandas as pd
import numpy as np
import logging
import collections
import theano

import beans.functions.sample
import beans.functions.scale
import beans.functions.knee
import beans.functions.encode


class Development:

    def __init__(self, training_split: pd.DataFrame, labels: list, target: str):
        """

        :param training_split: The training data
        :param labels: The list of distinct labels in the target field
        :param target: The target field
        """

        self.training_split = training_split
        self.labels = labels
        self.target = target

        logging.basicConfig(level=logging.WARNING, format='%(message)s\n%(asctime)s.%(msecs)03d', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def sample_(self):
        """
        Over-samples, and re-samples, the training data via SVNSMOTE; ref. beans.functions.sample

        :return:
        """

        sample = beans.functions.sample.Sample()

        return sample.exc(blob=self.training_split, target=self.target)

    def scale_(self, training_sampled: pd.DataFrame):
        """
        Scales the sampled data

        :param training_sampled:  The sampled data
        :return:
        """

        scale = beans.functions.scale.Scale()
        scaler = scale.exc(blob=training_sampled.drop(columns=self.target))
        x_training_scaled = scale.apply(blob=training_sampled.drop(columns=self.target), scaler=scaler)
        training_scaled = pd.concat((x_training_scaled, training_sampled[self.target]), axis=1, ignore_index=False)

        return training_scaled, scaler

    def encode_(self, training_scaled: pd.DataFrame):
        """
        One-hot-encodes the depedent variable

        :param training_scaled: In general, this will be the DataFrame w.r.t. the final preprocessing step before structuring
        :return:
        """

        CategoricalData = collections.namedtuple(
            typename='CategoricalData', field_names=['fields', 'arrays'])

        fields = [self.target]
        arrays = [np.array(self.labels)]
        categories = CategoricalData._make((fields, arrays))

        encode = beans.functions.encode.Encode()
        bits = encode.bits(frame=training_scaled, categories=categories)
        training_encoded = pd.concat((training_scaled.drop(columns=fields), bits), axis=1, ignore_index=False)

        return training_encoded

    def structure(self, training_encoded: pd.DataFrame):
        """
        Structures the training data according to what is expected by the neural network model

        :param training_encoded:
        :return:
        """

        x_training_encoded = training_encoded.drop(columns=self.labels).to_numpy()
        y_training_encoded = training_encoded[self.labels].to_numpy()

        unity = np.ones((x_training_encoded.shape[0], 1))
        x_training_points = np.concatenate((unity, x_training_encoded), axis=1).astype(theano.config.floatX)
        y_training_points = y_training_encoded.astype(theano.config.floatX)

        return x_training_points, y_training_points

    def exc(self):
        """

        :return:
        """

        training_sampled = self.sample_()
        self.logger.warning('\n1. The training data shape after SVNSMOTE sampling: {}'.format(training_sampled.shape))
        self.logger.warning('\nFrequencies:\n{]'.format(training_sampled['class'].value_counts()))

        training_scaled, scaler = self.scale_(training_sampled=training_sampled)
        self.logger.warning('\n2. The training data shape after scaling: {}'.format(training_scaled.shape))

        training_encoded = self.encode_(training_scaled=training_scaled)
        self.logger.warning('\n3. The training data shape after one-hot-encoding the dependent variable: {}'.format(training_encoded.shape))

        x_training_points, y_training_points = self.structure(training_encoded=training_encoded)
        self.logger.warning('\n4. The training matrices after structuring: x -> {}, y -> {}'.format(x_training_points.shape, y_training_points.shape))

        return x_training_points, y_training_points, scaler