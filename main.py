import os
from argparse import ArgumentParser

import finrl.autotrain.training
from finrl.config import config
from finrl.marketdata.utils import fetch_and_store


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    if options.mode == "train":
        finrl.autotrain.training.train_one()

    elif options.mode == "download":
        fetch_and_store()


if __name__ == "__main__":
    main()
