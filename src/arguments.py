"""
This file implements a LSTM regressor controll class configurable from CLI or to be used as a object

Author: Michal Glos (xglosm01)
ZPJa 2023 - FIT VUT
"""

import argparse
from sys import argv
from pathlib import Path

from datasets import DATASETS
from LSTM.LSTM import LSTM


class CLIParser:
    """
    Custom argument parser to controll the LSTM and dataflow

    The main parser is used for CLI and program runtime with custom
    argument group to configure training runtime
    Then a custom argument group is created for LSTM configuration
    And a choice from custom, mutually exclusive arg parsers for each dataset
    """

    @staticmethod
    def _remove_options(mock_argv, option):
        """Removes all occurences of option(flag) in mock_argv list, does not work for options with parameters"""
        while option in mock_argv:
            mock_argv.remove(option)
        return mock_argv

    def _define_lstm_argparse(self):
        """Define lstm and regressor parameters"""
        lstm = self.parser.add_argument_group("LSTM model", "Configuration of LSTM model parameters.")
        lstm.add_argument(
            "-m",
            "--mode",
            type=str,
            default="last",
            choices=LSTM.modes,
            help="Select mode of LSTM hidden state aggregation into LSTM output.",
        )
        lstm.add_argument(
            "-hs1", "--hidden-size", default=12, type=int, help="Set the size of hidden state of the LSTM cell."
        )
        lstm.add_argument(
            "-hs2", "--reg-hidden-size", type=int, default=8, help="Specify the size of hidden layer of the regressor."
        )
        lstm.add_argument("-bi", "--bidirectional", action="store_true", help="Use bidirectional LSTM cells.")

    def _define_runtime_argparse(self):
        """Define runtime parameters"""
        # save path is defined to be constant (default) in order not to mess up file organization
        default_model_folder = str(Path(__file__).parent.joinpath("../models"))
        runtime = self.parser.add_argument_group("App runtime", "Configurationthe runtime parameters.")
        runtime.add_argument("--load", type=str, default=None, help="Path to load the pickled model.")
        runtime.add_argument(
            "--save",
            type=str,
            default=default_model_folder,
            help="Provide path to folder where will the model be dumped.",
        )
        runtime.add_argument(
            "-d", "--device", type=str, choices=["cpu", "cuda"], help="Choose the device to compute on.", default="cuda"
        )
        runtime.add_argument(
            "--type",
            type=str,
            choices=["float32", "float64", "float16"],
            default="float32",
            help="Choose the datatype to work with.",
        )

    def _define_training_argparse(self):
        """Define training parameters
        Has to be defined first because of those subparsers"""
        subparsers = self.parser.add_subparsers(dest="training")
        training = subparsers.add_parser("train", help="Execute a training sequence.")
        training.add_argument(
            "--lr",
            type=float,
            default=1,
            help="Configure learning rate, is normalized by batch_size to standardize per-step learning rate.",
        )
        training.add_argument("-b", "--batch", type=int, default=64, help="Set the batch size for training.")
        training.add_argument(
            "-s", "--steps", type=int, default=10000, help="Train the model for this number of entries."
        )
        training.add_argument(
            "--eval-skips",
            type=int,
            default=8,
            help="Perform this number of batch training steps per one evaluation on testing dataset.",
        )

    def _define_dataset_argparse(self):
        """Define dataset configuration"""
        # First define the dataset option, partially load it and based on the chosed class - create custom argparser
        self.parser.add_argument(
            "-ds",
            "--dataset",
            type=str,
            default="sum_int",
            choices=DATASETS.keys(),
            help="Choose the dataset used.",
        )
        # In order to parse args partially here and not fail when --help/-h encountered,
        # we copy the argv and remove possible --help or -h flags
        mock_args = argv.copy()[1:]
        mock_args = self._remove_options(mock_argv=mock_args, option="-h")
        mock_args = self._remove_options(mock_argv=mock_args, option="--help")
        self.dataset_cls = DATASETS[self.parser.parse_known_args(args=mock_args)[0].dataset]
        self.dataset_cls.define_argument_group(self.parser)

    def __init__(self):
        """Initialize the argparser for all objects and runtime"""

        self.parser = argparse.ArgumentParser()
        self._define_training_argparse()
        self._define_lstm_argparse()
        self._define_runtime_argparse()
        self._define_dataset_argparse()

    def parse_args(self):
        """Parse the arguments"""
        return self.parser.parse_args(argv[1:])
