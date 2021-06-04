import pandas as pd
import numpy as np
import logging
import collections
import theano

import beans.functions.sample
import beans.functions.scale
import beans.functions.knee
import beans.functions.project
import beans.functions.encode


class Development:

    def __init__(self, data: pd.DataFrame, labels: list, target: str):
        """

        :param data: The training data
        :param labels: The list of distinct labels in the target field
        :param target: The target field
        """

        self.data = data
        self.labels = labels
        self.target = target

        logging.basicConfig(level=logging.WARNING, format='%(message)s\n%(asctime)s.%(msecs)03d', 
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def sample_(self):
        """
        Over-samples, and re-samples, the training data via SVNSMOTE; ref. beans.functions.sample

        :return:
        """

        sample = beans.functions.sample.Sample()

        return sample.exc(blob=self.data, target=self.target)

    def scale_(self, blob: pd.DataFrame):
        """
        Scales the independent variables of blob

        :param blob:  The data
        :return:
        """
        
        focus = blob.copy().drop(columns=self.target)

        scale = beans.functions.scale.Scale()
        scaler = scale.exc(blob=focus)
        x_scaled = scale.apply(blob=focus, scaler=scaler)
        scaled = pd.concat((x_scaled, blob[self.target]), axis=1, ignore_index=False)

        return scaled, scaler

    def project_(self, blob: pd.DataFrame):
        """

        :param blob:  The data that will be projected
        :return:
        """

        knee = beans.functions.knee.Knee()
        n_components = knee.exc(blob=blob, target=self.target)
        self.logger.warning('The # of Kernel Principal Components will be set to {}'.format(n_components))

        project = beans.functions.project.Project()
        matrix = blob.drop(columns=self.target).to_numpy()
        projector = project.exc(matrix=matrix, n_components=n_components)
        projected = project.apply(matrix=matrix, vector=blob[self.target], projector=projector)

        return projected, projector

    def encode_(self, blob: pd.DataFrame):
        """
        One-hot-encodes the dependent variable

        :param blob: In general, this will be the DataFrame w.r.t. the final preprocessing step 
                                              before structuring
        :return:
        """

        CategoricalData = collections.namedtuple(
            typename='CategoricalData', field_names=['fields', 'arrays'])

        fields = [self.target]
        arrays = [np.array(self.labels)]
        categories = CategoricalData._make((fields, arrays))

        encode = beans.functions.encode.Encode()
        bits = encode.bits(frame=blob, categories=categories)
        encoded = pd.concat((blob.drop(columns=fields), bits), axis=1, ignore_index=False)

        return encoded

    def structure(self, blob: pd.DataFrame):
        """
        Structures the training data according to what is expected by the neural network model

        :param blob:
        :return:
        """

        x_matrix = blob.drop(columns=self.labels).to_numpy()
        y_matrix = blob[self.labels].to_numpy()

        unity = np.ones((x_matrix.shape[0], 1))
        x_points = np.concatenate((unity, x_matrix), axis=1).astype(theano.config.floatX)
        y_points = y_matrix.astype(theano.config.floatX)

        return x_points, y_points
