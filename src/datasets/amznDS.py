import os
import logging

from tqdm import tqdm

import torch

from datasets.datasetBase import DatasetBase


# Datasets with category as key and url as values
BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles"
DATASETS = {
    "Fashion": "AMAZON_FASHION.json.gz",
    "Beauty": "All_Beauty.json.gz",
    "Appliances": "Appliances.json.gz",
    "Arts": "Arts_Crafts_and_Sewing.json.gz",
    "Automotive": "Automotive.json.gz",
    "Books": "Books.json.gz",
    "CDs": "CDs_and_Vinyl.json.gz",
    "Phones": "Cell_Phones_and_Accessories.json.gz",
    "Clothing": "Clothing_Shoes_and_Jewelry.json.gz",
    "Music": "Digital_Music.json.gz",
    "Electronics": "Electronics.json.gz",
    "GiftCards": "Gift_Cards.json.gz",
    "Groceries": "Grocery_and_Gourmet_Food.json.gz",
    "HomeKitchen": "Home_and_Kitchen.json.gz",
    "IndustrialScientific": "Industrial_and_Scientific.json.gz",
    "Kindle": "Kindle_Store.json.gz",
    "LuxuryBeauty": "Luxury_Beauty.json.gz",
    "Magazines": "Magazine_Subscriptions.json.gz",
    "Movies": "Movies_and_TV.json.gz",
    "MusicalInstruments": "Musical_Instruments.json.gz",
    "Office": "Office_Products.json.gz",
    "PatioGarden": "Patio_Lawn_and_Garden.json.gz",
    "Pets": "Pet_Supplies.json.gz",
    "PrimePantry": "Prime_Pantry.json.gz",
    "Software": "Software.json.gz",
    "Sports": "Sports_and_Outdoors.json.gz",
    "Tools": "Tools_and_Home_Improvement.json.gz",
    "Toys": "Toys_and_Games.json.gz",
    "VideoGames": "Video_Games.json.gz",
}


DATASET_OF_INTEREST = ["Magazines", "GiftCards"]


class AmazonDataset(DatasetBase):
    """
    Dataset class for Amazon product reviews

    This class implements:
        Selecting different subsets of dataset based on product category
        Downloads data directly from internet
        Loads data into desired torch datasets

    All data are obtained during object initialization, download included.
    Filesystem is checked if datasets are downloaede, if some or all are missing,
    download sequence is initiated and the data are then loaded from gzip files.
    After the embedding model is loaded and applied on all data. Finally, data
    are cut into maximum len. and padded with zeros to homogene lenghts.
    """

    name = "amazon"

    def __init__(
        self,
        data_folder,
        device,
        torch_type,
        fragments=DATASET_OF_INTEREST,
        verified=False,
        balanced=False,
        embedding_size=64,
        data_ratios=(0.8, 0.1, 0.1),
        entries=10000,
    ):
        """
        Instantiate self, include dataset fragments included in the global var DATASET_OF_INTEREST

        Args:
            data_folder (str): A path to store the downloaded dataset fragments
            device (str): Specify torch device
            torch_type (torch.dtype): Specify the dtype of data (float16, float32, float64)
            data_ratios (Tuple(float, float, float)): A tuple of data division (train, test, eval)
            entries (int): How many entries will be loaded
            embedding_size (int): Length of embed vector (token)
            fragments  (List[str]): List of fragments of interest (Keys from DATASETS list)
            verified (bool): Keep verified attribute of data
            balanced (bool): Balance the dataset with oversampling
        """
        super().__init__(torch_type, device)
        assert sum(data_ratios) == 1, "Data ratios should sum to 1!"
        self.data_ratios = data_ratios
        self.rating_entries = entries
        self.balanced = balanced

        # Just obtain neccessary classes
        self.get_tensor_factory(torch_type, device)

        self.embedding_size = embedding_size
        self.raw_data, self.formatted_data = [], []
        self.fragments = fragments
        self.data_folder = data_folder

        # Get the attributes we care about from the entries
        self.attributes_of_interest = ["overall", "reviewText"]
        if verified:
            self.attributes_of_interest.append("verified")

        os.makedirs(self.data_folder, exist_ok=True)

        to_download = []
        for fragment in DATASET_OF_INTEREST:
            if not self.file_exists(self._get_fragment_path(fragment)):
                to_download.append(fragment)

        if to_download:
            logging.info(f"Following dataset fragments will be downloaded: {', '.join(to_download)}")
            for fragment in to_download:
                fragment_url = self._get_fragment_url(fragment)
                fragment_path = self._get_fragment_path(fragment)
                self.download_data(fragment_url, fragment_path)

        logging.info(
            f"Following dataset fragments will be loaded, tokenized, embedde and converted into tensors: {', '.join(DATASET_OF_INTEREST)}"
        )

        for fragment in DATASET_OF_INTEREST:
            line_limit = entries - len(self.raw_data)
            self.raw_data.extend(
                self.load_gzip_json(self._get_fragment_path(fragment), self.attributes_of_interest, "reviewText", line_limit)
            )

        self.embedd_dataset("reviewText", self.raw_data, embedding_size)
        self.finish_dataset()

    def _get_fragment_path(self, fragment):
        """Obtain the path to the potentially existing dataset fragment"""
        return os.path.join(self.data_folder, DATASETS[fragment])

    def _get_fragment_url(self, fragment):
        """Obtain URL to a dataset fragment"""
        return os.path.join(BASE_URL, DATASETS[fragment])

    def finish_dataset(self):
        """Train Word2Vec model and encode each token from data into a torch tensor"""
        for dato in tqdm(self.raw_data, desc=f"Preparing dataset ..."):
            # Obtain the y (y = model(x))
            dato["y"] = self.tensor_factory([dato["overall"]]).squeeze(0)
            # If required, attributed from data entries (verified bool) is appended to the vectors
            if "verified" in self.attributes_of_interest:
                verified_element = self.tensor_factory([float(dato["verified"])])
                repeated_verified_element = verified_element.expand(dato["X"].shape[0], 1)
                dato["X"] = torch.cat((dato["X"], repeated_verified_element), dim=1)

        # Load the data into X and y components
        self.X = [dato["X"] for dato in self.raw_data]
        self.y = [dato["y"] for dato in self.raw_data]

        # Obtain the border indices
        self.train_border_idx = int(self.data_ratios[0] * len(self.y))
        self.test_border_idx = int((self.data_ratios[0] + self.data_ratios[1]) * len(self.y))

        # Select the particular data
        self.X_train = self.X[: self.train_border_idx]
        self.y_train = self.y[: self.train_border_idx]
        self.X_test = self.X[self.train_border_idx : self.test_border_idx]
        self.y_test = self.y[self.train_border_idx : self.test_border_idx]
        self.X_eval = self.X[self.test_border_idx : ]
        self.y_eval = self.y[self.test_border_idx : ]
        # Log it to the user (as print to be seen in jupyter also)
        logging.info(f"A ration of {':'.join(map(str, self.data_ratios))} was requested on dataset of len {len(self.X)}.")
        logging.info(f"The X component was loaded - train: {len(self.X_train)}; test: {len(self.X_test)}; eval: {len(self.X_eval)}.")
        logging.info(f"The y component was loaded - train: {len(self.y_train)}; test: {len(self.y_test)}; eval: {len(self.y_eval)}.")
        

    @classmethod
    def instantiate_from_args(cls, args):
        """Instantiate this dataset and get it into data-loaded state from argparse Namespace object"""
        return cls(
            data_folder=args.data_path,
            verified=args.verified,
            embedding_size=args.embedding_len,
            data_ratios=(args.division[0], args.division[1], 1 - (args.division[0] + args.division[1])),
            device=args.device,
            torch_type=getattr(torch, args.type),
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
        group.add_argument(
            "-e", "--entries", type=int, default=10000, help="Number of entries to load."
        )
        group.add_argument(
            "--data-path", type=str, default="./data", help="The path to the folder with data to be downloaded."
        )
        group.add_argument("-emb", "--embedding-len", type=int, default=64, help="Size of the embedded word vector.")
        group.add_argument(
            "--verified", action="store_true", help="Use verified bool value as part of embedding vectors."
        )
        group.add_argument(
            "--balanced", action="store_true", help="Balance the dataset with oversampling."
        )
        group.add_argument(
            "-div",
            "--division",
            nargs=2,
            type=float,
            default=[0.8, 0.1],
            help="Specify the percentage of data to be used for training, testing and eval. Takes 2 parameters - train, test. Eval is computed afterwards",
        )
        return group
