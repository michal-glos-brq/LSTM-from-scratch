#! /usr/bin/env python3

"""
This simple script provides opportunity to controll the LSTM model and data flow

Author: Michal Glos (xglosm01)
"""

import os
import json
from datetime import datetime

from torch import float16, float32, float64

TORCH_FLOATS = {
    'float16': float16,
    'float32': float32,
    'float64': float64,
}

from arguments import CLIParser
from LSTM.LSTM import LSTM


def save_model(model, args):
    """
    Save the model and it's configuration from args

    Args:
        model (LSTM.LSTM): Model to be saved
        args (argparse.Namespace or whatever with __dict__ dunder method): Configuration to be kept
    """
    # Make sure destination folder exists
    os.makedirs(args.save, exist_ok=True)
    # OK: {Bi}LSTM_{hidden_size}_{regressor_hidden_size}_{timestamp}
    filename = f"{'Bi' if args.bidirectional else ''}LSTM_{args.hidden_size}_{args.reg_hidden_size}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    filename_path = os.path.join(args.save, filename)

    with open(f"{filename_path}.p", "wb") as file:
        model.pickle(file)

    with open(f"{filename_path}.json", "w") as file:
        # Wtf is this dict rodeo arpagrse?
        json.dump(dict(args.__dict__), file, indent=4)


if __name__ == "__main__":
    ## Parse arguments
    parser = CLIParser()

    args = parser.parse_args()

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

    ## Evaluate model
    model.eval(dataset.eval_data)

    ## Save model (Model will be always saved) and the CLI parameters into a JSON to vorify later
    save_model(model=model, args=args)
