from src.predictor import vTrain
from src.config import vTrainConfig

import logging

import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(args):
    '''
    Usage:
        python example.py -c /path/to/config/file
        e.g., python example.py -c config/validation/single/config_val_single_0818.json
    '''

    # Load configuration file
    config = vTrainConfig.load_from_file(args.config)

    # Initialize vTrain with the configuration file
    sim = vTrain(config)

    # Run simulation and get the results
    result, breakdown = sim()

    # Show the predicted single-iteration training time
    pred_iter_time = max(result.values())/1000/1000
    logger.info(f"predicted iteration time: {pred_iter_time:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c, --config", type=str, dest="config")
    args = parser.parse_args()

    main(args)
