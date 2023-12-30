"""
This file implements base class for other datasets and custom pytorch dataset class compatibile with torch standards

Author: Michal Glos (xglosm01)
ZPJa 2023 - FIT VUT
"""
import os
import sys
import gzip
import json
import random
import logging
import requests
from typing import List
from zipfile import ZipFile

import time
import multiprocessing
from collections import defaultdict

import numpy as np
import nltk
from nltk import word_tokenize
from gensim.models import Word2Vec

import torch
from tqdm import tqdm


class EasyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device, torch_type, balance=False):
        """
        Args:
            balance (bool): whether to balance the dataset on y property
        """
        # Select the correct tensor datatype
        torch_factory = torch.cuda if device == "cuda" else torch
        if torch.float16 == torch_type:
            self.torch_factory = torch_factory.HalfTensor
        elif torch.float32 == torch_type:
            self.torch_factory = torch_factory.FloatTensor
        elif torch.float64 == torch_type:
            self.torch_factory = torch_factory.DoubleTensor
        else:
            raise TypeError(f"Unknown type {torch_type}, only torch.float16/32/64 is supported.")

        if balance:
            old_len = len(y)
            data_entries = defaultdict(list)
            # Now obtain dataset for each rating value
            for _x, _y in tqdm(list(zip(x, y)), ncols=120, desc="Creating data balancind object ..."):
                data_entries[_y.item()].append(_x)

            # Now oversample the less prevalent rating values
            max_len = max([len(sub_ds) for sub_ds in data_entries.values()])

            # Start the oversampling
            for sub_ds in tqdm(data_entries.values(), desc="Balancing the dataset ...", ncols=120):
                # To not have random indices, just have an iterator
                i = 0
                # Load util the oversampled data exist
                while len(sub_ds) < max_len:
                    sub_ds.append(sub_ds[i])
                    i += 1
            # Put the new data into the dataset
            self._X = []
            self._y = []
            for rating, data in data_entries.items():
                self._X += data
                self._y += [rating] * len(data)

            assert (max_len * len(data_entries)) == len(self._y), "Something went wrong when balancing the dataset!"

            logging.info(f"Balaced dataset with oversampling, from {old_len} entries to {len(self._y)} entries.")

        else:
            self._X = x
            self._y = y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        # It turned out there is a bug in Amazon dataset implementation and it allows for luckily very small chance (~ one per million)
        # so in order not to return an empty tensors, check if tensor is not empty. If is, return some random non-empty review rating pair
        x, y = (self._X[idx], self._y[idx])
        if x.shape[0] > 0:
            return (x, y)
        else:
            while x.shape[0] == 0:
                random_idx = random.randint(0, len(self._X) - 1)
                x, y = (self._X[random_idx], self._y[random_idx])
            return (x, y)

    @property
    def X(self):
        """Get the whola X data vector (padded)"""
        return torch.nn.utils.rnn.pad_sequence(self._X, batch_first=True)

    @property
    def y(self):
        """Obtain all labels as a tensor"""
        return self.torch_factory(self._y)

    def _padded_batch_loader(self, batch):
        """
        Return tuple of tensors (data, labels)
        Data will be padded with zeros
        """
        data, labels = [], []
        for dato, label in batch:
            data.append(dato)
            labels.append(label)
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True), self.torch_factory(labels)


class DatasetBase:
    """Base class for datasets to handle polymorphism"""

    def __init__(self, torch_type, device):
        """
        Initialize the base class

        Args:
            torch_type: Torch datatype (torch.float32 or else ...)
            device (str): Torch device cuda or cpu
        """
        self.torch_type, self.device = torch_type, device
        self.X_train, self.X_test, self.X_eval = None, None, None
        self.y_train, self.y_test, self.y_eval = None, None, None
        self.balanced = False  # By default, we do not care about balancing dummy datasets

    @property
    def tensor_factory(self):
        if self.torch_type == torch.float16:
            return self.torch_factory.HalfTensor
        elif self.torch_type == torch.float32:
            return self.torch_factory.FloatTensor
        if self.torch_type == torch.float64:
            return self.torch_factory.DoubleTensor

    @property
    def torch_factory(self):
        return torch.cuda if self.device == "cuda" else torch

    def embedd_dataset(self, attr_to_embedd, raw_data, embedding_size):
        """Embedd the attribute attr_to_embed"""
        corpus = []
        for entry in tqdm(raw_data, desc="Creating vocabulary ...", ncols=112):
            corpus.append(entry[attr_to_embedd])

        pbar = tqdm(raw_data, desc=f"Embedding dataset ...", ncols=120)
        model = Word2Vec(sentences=corpus, min_count=1, vector_size=embedding_size, window=5)

        del corpus

        for dato in pbar:
            dato["X"] = self.tensor_factory(
                np.array([model.wv[token] for token in dato[attr_to_embedd][: self.max_seq_len]])
            )
            del dato[attr_to_embedd]

    @property
    def entry_size(self):
        """Obtain the size of a single data entry vector"""
        return self.X_train.shape[2]

    @property
    def train_data(self):
        """Obtain the training data"""
        return EasyDataset(self.X_train, self.y_train, self.device, self.torch_type, balance=self.balanced)

    @property
    def test_data(self):
        """Obtain the testing data"""
        return EasyDataset(self.X_test, self.y_test, self.device, self.torch_type, balance=self.balanced)

    @property
    def eval_data(self):
        """Obtain the testing data"""
        return EasyDataset(self.X_eval, self.y_eval, self.device, self.torch_type, balance=self.balanced)

    @staticmethod
    def read_dato(dato, attributes, tokenize):
        """Load the provided dato string of single JSON entry and parse it into our data object"""
        dato = json.loads(dato)
        if DatasetBase._is_data_complete(dato, attributes):
            new_entry = {attr: dato[attr] for attr in attributes}
            new_entry[tokenize] = word_tokenize(new_entry[tokenize])
            return new_entry

    @staticmethod
    def _init_reader(file_d, read_queue, counters, per_rating_entries, queue_max_len=128):
        """
        Read a file in the read_queue, do not exceed the given limit on Queue size

        Args:
            file_d (file descriptor): File to be read - already opened
            read_queue (multiprocessing.Queue): Queue to load the file into
            counters (multiprocessing.Array): Counter of reviews per rating
            per_rating_entries (int): How many reviews to load for each rating value
            queue_max_len (int): Max len of Queue until reading is suspemded for a while
        """
        pbar = tqdm([None], total=per_rating_entries * 5, desc="Loading data", ncols=112)
        total = 0
        # Read line-by-line
        while line := file_d.readline():
            if all([counter >= per_rating_entries for counter in counters]):
                break

            # Wait if Queue is full
            while True:
                if read_queue.qsize() >= queue_max_len:
                    time.sleep(0.01)
                else:
                    break

            # If there is a free space, let's push it in
            _total = sum(counters)
            pbar.update(_total - total)
            total = _total

            read_queue.put(line)

    @staticmethod
    def _init_parser(read_queue, write_queue, attributes, tokenize, counters, per_rating_entries):
        """
        Init parser subprocess which takes entries from read_queue and parses it into the write_queue

        Args:
            read_queue (multiprocessing.Queue): Queue to load the file into
            write_queue (multiprocessing.Queue): Queue to load the data dicts into
            attributes (List[str]): A list of attributes a dato must have to be loaded, to be loaded
                        means only attributes from the list will be loaded into a dict
            tokenize (str): the attribute of loaded dict to be tokenized (required)
            counters (multiprocessing.Array): Counter of reviews per rating
            per_rating_entries (int): How many reviews to load for each rating value
        """
        while True:
            # Try to fetch a line, exit if queue empty
            try:
                line = read_queue.get(timeout=1)  # A second for parsers to know it's done
            except multiprocessing.queues.Empty:
                break

            # Load the data entry onto the write queue
            if dato := DatasetBase.read_dato(line, attributes, tokenize):
                idx = int(dato["rating"]) - 1
                if counters[idx] >= per_rating_entries:
                    del dato
                    continue
                counters[idx] += 1
                write_queue.put(dato)

    @staticmethod
    def load_zip(
        file_dst, attributes: List[str], tokenize: str, zipfile_src: str, ds_len: int = None, num_workers: int = 16
    ):
        """
        Load an array of JSON objects compressed into gzip format into a array of dicts

        This would take approx. 10 hours, hence multiprocessing is used

        Args:
            file_dst (str): file to load
            attributes (List[str]): A list of attributes a dato must have to be loaded, to be loaded
                        means only attributes from the list will be loaded into a dict
            tokenize (str): the attribute of loaded dict to be tokenized (required)
            ds_len (str): Len of the dataset in order not to iterate 2 times through large files
            zipfile_src (str): Source - path to the file to be extracted from the zpifile
            num_workers (int): Number of wrokers to load the read file
        """
        # Download module if not already downloaded
        nltk.download("punkt")

        read_queue, write_queue = multiprocessing.Queue(), multiprocessing.Queue()
        counters = multiprocessing.Array("i", 5)

        with ZipFile(file_dst, "r") as zipfile:
            with zipfile.open(zipfile_src) as extracted_file:
                # If we want to compute the number of lines ...
                if ds_len is None:
                    ds_len = 100000
                ds_len = int(ds_len / 5)

                # The main read process
                reader = multiprocessing.Process(
                    target=DatasetBase._init_reader, args=(extracted_file, read_queue, counters, ds_len)
                )
                reader.start()

                # Initialize the workers
                workers = []
                for _ in range(num_workers):
                    p = multiprocessing.Process(
                        target=DatasetBase._init_parser,
                        args=(read_queue, write_queue, attributes, tokenize, counters, ds_len),
                    )
                    p.start()
                    workers.append(p)

                # Finish subprocesses
                reader.join()

                # Collect results so workers wont be blocked
                raw_data = []
                while True:
                    try:
                        item = write_queue.get(timeout=1)
                    except multiprocessing.queues.Empty:
                        break

                    raw_data.append(item)
                # Finish the workers
                for worker in workers:
                    worker.join()

        return raw_data

    @staticmethod
    def load_gzip_json(file_dst, attributes: List[str], tokenize: str):
        """
        Load an array of JSON objects compressed into gzip format into a array of dicts

        Args:
            file_dst (str): file to load
            tokenize (str): the attribute of loaded dict to be tokenized (required)
            num_lines (int): Number of lines to be loaded
        """
        # Download module if not already downloaded
        nltk.download("punkt")

        raw_data = []
        with gzip.open(file_dst, "r") as file:
            all_lines = file.readlines()
            for entry in tqdm(all_lines, total=len(all_lines), desc=f"Reading {file_dst} ...", ncols=120):
                if dato := DatasetBase.read_dato(entry, attributes, tokenize):
                    raw_data.append(dato)

        return raw_data

    @staticmethod
    def file_exists(file):
        return os.path.isfile(file)

    @staticmethod
    def _is_data_complete(dato, attributes):
        """Return bool whether the data contains all required attributes"""
        return all(map(lambda x: x in dato, attributes))

    @staticmethod
    def download_data(url, file_dst):
        """Download data from url and save it into file_dst"""
        # Mock the human user
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)"
                "AppleWebKit/537.36 (KHTML, like Gecko)"
                "Chrome/39.0.2171.95 Safari/537.36"
            )
        }

        data_request = requests.get(url, headers=headers, stream=True)
        total_download_size = int(data_request.headers.get("content-length", 0))
        tqdm_bar = tqdm(
            total=total_download_size, desc=f"Downloading {file_dst} ...", unit="iB", unit_scale=True, ncols=120
        )
        try:
            if data_request.status_code == 200:
                with open(file_dst, "wb") as f:
                    for chunk in data_request.iter_content(chunk_size=4192):
                        tqdm_bar.update(len(chunk))
                        f.write(chunk)
            else:
                logging.error(
                    "Download failed, please try it once again or download data"
                    + "manually from: "
                    + url
                    + " and save the json.gz file here."
                )
                sys.exit(1)
        except KeyboardInterrupt:
            # Rather delete it then to delete it manually when download fails
            os.remove(file_dst)
            raise KeyboardInterrupt()

    @staticmethod
    def define_argument_group(parser):
        """
        Define custom argument group for each dataset

        Args:
            parser (argparse.ArgumentParser): Argument parser object
        """
        raise NotImplementedError("Base class could not define it's argument group!")
