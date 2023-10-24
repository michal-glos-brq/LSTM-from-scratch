"""
This file implements base class for other datasets and  also a dummy datasets,
artificially generated for model behaviour verification.
"""
import os
import sys
import gzip
import json
import logging
import requests
from typing import List
from zipfile import ZipFile

import time
import multiprocessing

import numpy as np
import nltk
from nltk import word_tokenize
from gensim.models import Word2Vec

import torch
from tqdm import tqdm


class EasyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device, torch_type):
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

        self._X = x
        self._y = y

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        return (self._X[idx], self._y[idx])

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

    def get_tensor_factory(self, torch_type, device):
        self.torch_factory = torch.cuda if device == "cuda" else torch
        if torch_type == torch.float16:
            self.tensor_factory = self.torch_factory.HalfTensor
        elif torch_type == torch.float32:
            self.tensor_factory = self.torch_factory.FloatTensor
        if torch_type == torch.float64:
            self.tensor_factory = self.torch_factory.DoubleTensor

    def embedd_dataset(self, attr_to_embedd, raw_data, embedding_size):
        """Embedd the attribute attr_to_embed"""
        pbar = tqdm(raw_data, desc=f"Embedding dataset ...", ncols=120)
        corpus = [entry[attr_to_embedd] for entry in raw_data]
        model = Word2Vec(sentences=corpus, min_count=1, vector_size=embedding_size, window=5)
        for dato in pbar:
            dato["X"] = self.tensor_factory(np.array([model.wv[token] for token in dato[attr_to_embedd]]))

    @property
    def entry_size(self):
        """Obtain the size of a single data entry vector"""
        return self.X_train.shape[2]

    @property
    def train_data(self):
        """Obtain the training data"""
        return EasyDataset(self.X_train, self.y_train, self.device, self.torch_type)

    @property
    def test_data(self):
        """Obtain the testing data"""
        return EasyDataset(self.X_test, self.y_test, self.device, self.torch_type)

    @property
    def eval_data(self):
        """Obtain the testing data"""
        return EasyDataset(self.X_eval, self.y_eval, self.device, self.torch_type)

    @staticmethod
    def read_dato(dato, attributes, tokenize):
        """Load the provided dato string of single JSON entry and parse it into our data object"""
        dato = json.loads(dato)
        if DatasetBase._is_data_complete(dato, attributes):
            new_entry = {attr: dato[attr] for attr in attributes}
            new_entry[tokenize] = word_tokenize(new_entry[tokenize])
            return new_entry

    @staticmethod
    def _init_reader(file_d, read_queue, num_lines: int, queue_max_len=128):
        """
        Read a file in the read_queue, do not exceed the given limit on Queue size

        Args:
            file_d (file descriptor): File to be read - already opened
            read_queue (multiprocessing.Queue): Queue to load the file into
            num_lines (int): Number of lines to be loaded
            queue_max_len (int): Max len of Queue until reading is suspemded for a while
        """
        pbar = tqdm([None], total=num_lines, desc="Reading ...", ncols=120)
        i = 0
        # Read line-by-line
        while line := file_d.readline():
            # If max lines exceeded, finish
            if i >= num_lines:
                break
            
            # Wait if Queue is full
            while True:
                if read_queue.qsize() >= queue_max_len:
                    time.sleep(0.01)
                else:
                    break

            # If there is a free space, let's push it in
            read_queue.put(line)
            pbar.update(1)
            i += 1

    @staticmethod
    def _init_parser(read_queue, write_queue, attributes, tokenize):
        """
        Init parser subprocess which takes entries from read_queue and parses it into the write_queue

        Args:
            read_queue (multiprocessing.Queue): Queue to load the file into
            write_queue (multiprocessing.Queue): Queue to load the data dicts into
            attributes (List[str]): A list of attributes a dato must have to be loaded, to be loaded
                        means only attributes from the list will be loaded into a dict
            tokenize (str): the attribute of loaded dict to be tokenized (required)
        """
        while True:
            # Try to fetch a line, exit if queue empty
            try:
                line = read_queue.get(timeout=1) # A second for parsers to know it's done
            except multiprocessing.queues.Empty:
                break

            # Load the data entry onto the write queue
            if dato := DatasetBase.read_dato(line, attributes, tokenize):
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

        with ZipFile(file_dst, "r") as zipfile:
            with zipfile.open(zipfile_src) as extracted_file:
                # If we want to compute the number of lines ...
                if ds_len is None:
                    ds_len = sum((1 for _ in extracted_file))
                
                # The main read process
                reader = multiprocessing.Process(target=DatasetBase._init_reader, args=(extracted_file, read_queue, ds_len))
                reader.start()

                # Initialize the workers
                workers = []
                for _ in range(num_workers):
                    p = multiprocessing.Process(
                        target=DatasetBase._init_parser, args=(read_queue, write_queue, attributes, tokenize)
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
    def load_gzip_json(file_dst, attributes: List[str], tokenize: str, num_lines: int):
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
        i = 0
        with gzip.open(file_dst, "r") as file:
            all_lines = file.readlines()
            for entry in tqdm(all_lines, total=min(len(all_lines), num_lines) ,desc=f"Reading {file_dst} ...", ncols=120):
                if i >= num_lines:
                    break
         
                if dato := DatasetBase.read_dato(entry, attributes, tokenize):
                    raw_data.append(dato)
         
                # If limit of read entries exceeded, return
                i += 1

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