"""This module implements tripadvisor dataset manager"""
import os
import logging
import random

import torch
from tqdm import tqdm

from datasets.datasetBase import DatasetBase


MAX_DS_LEN = 50264531


class TripAdvisorDataset(DatasetBase):
    """Dataset manager for trip advisor reviews"""

    name = "trip"

    DOWNLOAD_URL = "https://liapc24.epfl.ch/Datasets/Full_HotelRec.zip"
    # We care only about one of those, but we will use already implemented
    # methods and just filter it later
    ATTRIBUTES_OF_INTEREST = ["text", "rating"]

    def __init__(
        self,
        data_folder,
        device,
        torch_type,
        embedding_size=64,
        data_ratios=(0.8, 0.1, 0.1),
        entries=10000,
        balanced=False,
        max_seq_len=176,
    ):
        """
        Initialize, load and download tripadvisor data

        Args:
            data_folder (str): A path to store the downloaded dataset fragments
            device (str): Specify torch device
            torch_type (torch.dtype): Specify the dtype of data (float16, float32, float64)
            embedding_size (int): Length of embed vector (token)
            specific_ranking (bool or str): If false - overall ranking is used, otherwise specific sub-ranking is chosen
            data_ratios (Tuple(float, float, float)): A tuple of data division (train, test, eval)
            entries (int): How many entries will be loaded
            balanced (bool): Balance the dataset with oversampling
            max_seq_len (int): Maximal length of sequence until it is cut off
        """
        assert sum(data_ratios) == 1, "Data ratios should sum to 1!"
        super().__init__(torch_type, device)
        self.max_seq_len = max_seq_len
        self.rating_entries = entries
        self.data_ratios = data_ratios
        self.data_folder = data_folder
        self.balanced = balanced

        if not self.file_exists(self.abs_filename):
            logging.info("TripAdvisor dataset not found, therefore will be downloaded.")
            os.makedirs(self.data_folder, exist_ok=True)
            self.download_data(self.DOWNLOAD_URL, self.abs_filename)

        logging.info(
            f"The dataset will be loaded, tokenized, embedded and converted into tensors. Since the zipped file has more then 50GB, the time counting of the dataset size might take several minutes."
        )

        self.raw_data = self.load_zip(
            self.abs_filename, self.ATTRIBUTES_OF_INTEREST, "text", "HotelRec.txt", ds_len=entries
        )
        logging.info(f"Loaded {len(self.raw_data)} entries of data.")
        self.embedd_dataset("text", self.raw_data, embedding_size)
        self.finish_dataset()

    def finish_dataset(self):
        """Train Word2Vec model and encode each token from data into a torch tensor"""
        for dato in tqdm(self.raw_data, desc=f"Preparing dataset ..."):
            # Obtain the y (y = model(x))
            dato["y"] = self.tensor_factory([dato["rating"]]).squeeze(0)

        self.X = [dato["X"] for dato in self.raw_data]
        self.y = [dato["y"] for dato in self.raw_data]

        # Shuffle those
        data = list(zip(self.X, self.y))
        random.shuffle(data)
        self.X, self.y = zip(*data)
        self.X, self.y = list(self.X), list(self.y)

        # Obtain the border indices
        self.train_border_idx = int(self.data_ratios[0] * len(self.y))
        self.test_border_idx = int((self.data_ratios[0] + self.data_ratios[1]) * len(self.y))

        # Select the particular data
        self.X_train = self.X[: self.train_border_idx]
        self.y_train = self.y[: self.train_border_idx]
        self.X_test = self.X[self.train_border_idx : self.test_border_idx]
        self.y_test = self.y[self.train_border_idx : self.test_border_idx]
        self.X_eval = self.X[self.test_border_idx :]
        self.y_eval = self.y[self.test_border_idx :]
        # Log it to the user (as print to be seen in jupyter also)
        logging.info(
            f"A ration of {':'.join(map(str, self.data_ratios))} was requested on dataset of len {len(self.X)}."
        )
        logging.info(
            f"The X component was loaded - train: {len(self.X_train)}; test: {len(self.X_test)}; eval: {len(self.X_eval)}."
        )
        logging.info(
            f"The y component was loaded - train: {len(self.y_train)}; test: {len(self.y_test)}; eval: {len(self.y_eval)}."
        )

    @property
    def _filename(self):
        """Obtain the file name from the DOWNLOAD_URL"""
        return os.path.split(self.DOWNLOAD_URL)[-1]

    @property
    def abs_filename(self):
        """Return the absolute path to the (potentially) downloaded dataset"""
        return os.path.join(self.data_folder, self._filename)

    @classmethod
    def instantiate_from_args(cls, args):
        """Instantiate this dataset and get it into data-loaded state from argparse Namespace object"""
        return cls(
            data_folder=args.data_path,
            embedding_size=args.embedding_len,
            data_ratios=(args.division[0], args.division[1], 1 - (args.division[0] + args.division[1])),
            device=args.device,
            torch_type=getattr(torch, args.type),
            entries=args.entries,
        )

    @staticmethod
    def define_argument_group(parser):
        """
        Define custom argument group for each dataset
        Weird arg. names are a must, because we have shared
        argument among the several dataset classes parsers

        Args:
            parser (argparse.ArgumentParser): Argument parser object
        """
        group = parser.add_argument_group(title="Dataset: Amazon Dataset")
        group.add_argument("-e", "--entries", type=int, default=10000, help="Number of entries to load.")
        group.add_argument(
            "--data-path", type=str, default="./data", help="The path to the folder with data to be downloaded."
        )
        group.add_argument("--balanced", action="store_true", help="Balance the dataset with oversampling.")
        group.add_argument("-emb", "--embedding-len", type=int, default=64, help="Size of the embedded word vector.")
        group.add_argument(
            "-div",
            "--division",
            nargs=2,
            type=float,
            default=[0.8, 0.1],
            help="Specify the percentage of data to be used for training, testing and eval. Takes 2 parameters - train, test. Eval is computed afterwards",
        )
        return group
