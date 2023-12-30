"""
This module implements Dummy datasets for testing and validating purposes

Author: Michal Glos (xglosm01)
ZPJa 2023 - FIT VUT
"""

import torch

from datasets.datasetBase import DatasetBase


class SummingIntDataset(DatasetBase):
    """A dataset with random integer sequences with y_gt only their respective sums"""

    name = "sum_int"

    def __init__(
        self,
        device: str = "cpu",
        torch_type: torch.dtype = torch.float32,
        min_val: int = 0,
        max_val: int = 16,
        seq_len: int = 64,
        vector_size: int = 32,
        train_data: int = 5000,
        test_data: int = 1000,
        eval_data: int = 1000,
    ):
        super().__init__(torch_type, device)
        # Create the training and eval Xs
        X_train = torch.randint(
            min_val,
            max_val,
            (train_data, seq_len, vector_size),
            device=self.device,
            dtype=self.torch_type,
            requires_grad=False,
        )
        X_test = torch.randint(
            min_val,
            max_val,
            (test_data, seq_len, vector_size),
            device=self.device,
            dtype=self.torch_type,
            requires_grad=False,
        )
        X_eval = torch.randint(
            min_val,
            max_val,
            (eval_data, seq_len, vector_size),
            device=self.device,
            dtype=self.torch_type,
            requires_grad=False,
        )
        # Compute adequate ground truths
        y_train = X_train.sum(dim=(1, 2))
        y_test = X_test.sum(dim=(1, 2))
        y_eval = X_eval.sum(dim=(1, 2))
        # Save the values
        self.X_train, self.X_test, self.X_eval = X_train, X_test, X_eval
        self.y_train, self.y_test, self.y_eval = y_train, y_test, y_eval

    @classmethod
    def instantiate_from_args(cls, args):
        """Instantiate self dataset object from CLI argparse parameters"""
        return cls(
            device=args.device,
            torch_type=getattr(torch, args.type),
            min_val=args.min_val,
            max_val=args.max_val,
            seq_len=args.sequence_len,
            vector_size=args.vector_len,
            train_data=args.train,
            test_data=args.test,
            eval_data=args.eval,
        )

    @staticmethod
    def define_argument_group(parser):
        """
        Define custom argument group for each dataset

        Args:
            parser (argparse.ArgumentParser): Argument parser object
        """
        group = parser.add_argument_group(title="Dataset: SummingIntDataset")
        group.add_argument(
            "-min", "--min-val", type=int, default=0, help="Set the minimal value for seqence/vector element."
        )
        group.add_argument(
            "-max", "--max-val", type=int, default=10, help="Set the maximal value for seqence/vector element."
        )
        group.add_argument(
            "-sl", "--sequence-len", type=int, default=8, help="Set the length of the artificial sequences."
        )
        group.add_argument(
            "-vl", "--vector-len", type=int, default=4, help="Size of the vector as a single element of sequence."
        )
        group.add_argument(
            "-tr",
            "--train",
            type=int,
            default=10000,
            help="Specify how many training sequences shall be generated.",
        )
        group.add_argument(
            "-te", "--test", type=int, default=1000, help="Specify how many testing sequences shall be generated."
        )
        group.add_argument(
            "-ev", "--eval", type=int, default=1000, help="Specify how many eval sequences shall be generated."
        )
        return group


class SummingFloatDataset(DatasetBase):
    """A dataset with random integer sequences with y_gt only their respective sums"""

    name = "sum_float"

    def __init__(
        self,
        device: str = "cpu",
        torch_type: torch.dtype = torch.float32,
        seq_len: int = 64,
        vector_size: int = 32,
        train_data: int = 5000,
        test_data: int = 1000,
        eval_data: int = 1000,
    ):
        super().__init__(torch_type, device)
        # Create the training and eval Xs
        X_train = torch.rand(
            (train_data, seq_len, vector_size),
            device=self.device,
            dtype=self.torch_type,
            requires_grad=False,
        )
        X_test = torch.rand(
            (test_data, seq_len, vector_size),
            device=self.device,
            dtype=self.torch_type,
            requires_grad=False,
        )
        X_eval = torch.rand(
            (eval_data, seq_len, vector_size),
            device=self.device,
            dtype=self.torch_type,
            requires_grad=False,
        )
        # Compute adequate ground truths
        y_train = torch.round(X_train.sum(dim=(1, 2)))
        y_test = torch.round(X_test.sum(dim=(1, 2)))
        y_eval = torch.round(X_eval.sum(dim=(1, 2)))
        # Save the values
        self.X_train, self.X_test, self.X_eval = X_train, X_test, X_eval
        self.y_train, self.y_test, self.y_eval = y_train, y_test, y_eval

    @classmethod
    def instantiate_from_args(cls, args):
        """Instantiate this dataset and get it into data-loaded state from argparse Namespace object"""
        return cls(
            device=args.device,
            torch_type=getattr(torch, args.type),
            seq_len=args.sequence_len,
            vector_size=args.vector_len,
            train_data=args.train,
            test_data=args.test,
            eval_data=args.eval,
        )

    @staticmethod
    def define_argument_group(parser):
        """
        Define custom argument group for each dataset
        Args:
            parser (argparse.ArgumentParser): Argument parser object
        """
        group = parser.add_argument_group(title="Dataset: SummingFloatDataset")
        group.add_argument(
            "-sl", "--sequence-len", type=int, default=8, help="Set the length of the artificial sequences."
        )
        group.add_argument(
            "-vl", "--vector-len", type=int, default=4, help="Size of the vector as a single element of sequence."
        )
        group.add_argument(
            "-tr",
            "--train",
            type=int,
            default=10000,
            help="Specify how many training sequences shall be generated.",
        )
        group.add_argument(
            "-te", "--test", type=int, default=1000, help="Specify how many testing sequences shall be generated."
        )
        group.add_argument(
            "-ev", "--eval", type=int, default=1000, help="Specify how many eval sequences shall be generated."
        )
