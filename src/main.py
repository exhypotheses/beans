"""
The main module for running other classes
"""
import logging
import os
import sys

import jax


def main():
    """
    Entry point
    :return: None
    """

    # Notes: The NVIDIA graphics processing unit (GPU) is successfully identified
    logger.info('JAX')
    logger.info(jax.devices(backend='gpu'))
    logger.info('The number of GPU devices: %s\n\n', jax.device_count(backend='gpu'))

    # The data
    initial = src.data.initial.Read().exc()
    initial.info()

    # The training & testing splits: persist later
    train, _ = src.algorithms.split.Split().exc(data=initial, train_size=config.Config().train_size)

    # Modelling: In progress
    src.modelling.interface.Interface().exc(train=train)

    # Clean-up
    src.functions.extraneous.Extraneous().extraneous()


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # Classes ...
    import src.algorithms.split
    import src.data.initial
    import src.functions.extraneous
    import src.modelling.interface

    import config

    main()
