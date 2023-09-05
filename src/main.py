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
    logger.info('The number of GPU devices: %s', jax.device_count(backend='gpu'))


    # The data
    data = src.data.read.Read().exc()
    data.info()
    logger.info(data.head())


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

    # Classes
    import src.data.read

    main()
