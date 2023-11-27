#! /usr/bin/env python3

"""
This simple script provides opportunity to controll the LSTM model and data flow

Author: Michal Glos (xglosm01)
"""
import pickle
import os

import torch
from torch import float16, float32, float64

TORCH_FLOATS = {
    'float16': float16,
    'float32': float32,
    'float64': float64,
}

from arguments import CLIParser
from LSTM.LSTM import LSTM


if __name__ == "__main__":
    ## Parse arguments
    parser = CLIParser()

    args = parser.parse_args()

    with torch.no_grad():

        ## Load dataset
        dataset = parser.dataset_cls.instantiate_from_args(args)

        ## Load model
        if args.load:
            with open(args.load, "rb") as file:
                model = LSTM.from_pickle(file)
            assert model.lstm.input_size == dataset.entry_size, "Dataset entry size and model input size do not check up!"
        else:
            model = LSTM(
                input_size=dataset.entry_size,
                hidden_lstm=args.hidden_size,
                hidden_dense=args.reg_hidden_size,
                mode=args.mode,
                device=args.device,
                torch_type=TORCH_FLOATS[args.type],
                bidirectional=args.bidirectional,
            )

        ## Train model
        if args.training is not None:
            model.train(
                dataset.train_data, dataset.test_data, args.lr, args.batch, args.steps, args.eval_skips
            )

            # Load the best behaving model for evaluation purposes
            if hasattr(model, "last_checkpoint"):
                with open(model.last_checkpoint, "rb") as model_file:
                    model = pickle.load(model_file)

        ## Evaluate model
        correct, mse_loss = model.eval(dataset.eval_data)
        print("Evaluation finished with {:.2f} % correct predictions with mean square error of {:.4f}.".format(correct, mse_loss))
