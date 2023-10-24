import pickle
from typing import List
from dataclasses import dataclass, field

from tqdm import tqdm
import torch

from LSTM.model_parts import LSTMCell, BiLSTMCell, Regressor
from LSTM.utils import mse_loss, mse_loss_d, num_correct


class ChartData:
    """This class stores, aggregates and annotates training data"""

    class PlotData:
        """A set of data to be plotted from either training or testing data"""

        def __init__(self, steps_per_entry):
            """Instantiate the single plot line data"""
            self.loss_steps = []
            self.loss = []
            self.correct_steps = []
            self.correct = []
            self.step = 0
            self.steps_per_entry = steps_per_entry

        def add_entry(self, correct, loss):
            self.loss.append(loss)
            self.loss_steps.append(self.step)
            self.correct.append(correct)
            self.correct_steps.append(self.step)
            self.step += self.steps_per_entry

    def __init__(self, eval_step_skips):
        # Training data holds the Y and X values of a loss chart
        self.training_data = self.PlotData(1)
        self.testing_data = self.PlotData(eval_step_skips)


class LSTM:
    """
    This class implements a LSTM regressor model for regression tasks
    on variable-sized vactor sequences
    """

    modes = ["bag", "mean", "last"]

    def __init__(
        self, input_size, hidden_lstm, hidden_dense, mode, device="cpu", torch_type=torch.float32, bidirectional=False
    ):
        """
        Initialize the LSTM regressor

        Args:
            input_size (int): Lenth of one element of data sequence
            hidden_lstm (int): Size of hidden state in the LSTM cell, also input size of the regressor
            hidden_dense (int): Size of the only hidden layer on the regressor
            device (str/torch.device): Device to work on (GPU/CPU) or rather (cpu/cuda)
            mode (str): How to process LSTM states from each step into the regressing model?
                        choices: bag - sum all, last - take the last, mean - average the states
            bidirectional (bool): Use bidirectional LSTM cell (with concat. as aggregating method)
        """
        if mode not in self.modes:
            raise ValueError(f"LSTM could not be initiated with mode={mode}")

        self.regressor = Regressor(hidden_lstm, hidden_dense, device=device, torch_type=torch_type)
        lstm_cell_cls = BiLSTMCell if bidirectional else LSTMCell
        self.lstm = lstm_cell_cls(input_size, hidden_lstm, device=device, torch_type=torch_type)
        self.device = device
        self.torch_type = torch_type
        self.mode = mode

    def divide_grads(self, grads, timesteps):
        """
        Divide gradients based on the method used to process LSTM hidden states

        Args:
            grads (torch.Tensor): Chained gradients from the regressor, will be divided based
                    on hidden state merging strategy (self.mode) into according timesteps in LSTM
                    shape: (batch_size, hidden_size)
            timesteps (int): Number of steps in time taken when LSTM was forwarded, also the len of last sequence
        """
        if self.mode.lower() == "bag" or self.mode.lower() == "mean":
            return grads.tile(timesteps, 1, 1)  # grads should always have 2 dims
        elif self.mode.lower() == "last":
            chained_grads = torch.zeros(
                (timesteps, *grads.shape), dtype=self.torch_type, device=self.device, requires_grad=False
            )
            # Store it as first, we would not need to revert it when backwarding through the sequence
            chained_grads[0] = grads
            return chained_grads.transpose(1, 0)

    def process_states(self, states):
        """
        Process LSTM hidden states to fit into regressor according to selected method

        Args:
            states (torch.Tensor): hidden states taken from LSTSM cell in each timestep (batch, sequence, hidden_size)
        Returns: torch.Tensor of shape (batch, hidden_size), the sequence is collapsed into homogenous vectors
        """
        if self.mode.lower() == "bag":
            return states.sum(dim=1)
        elif self.mode.lower() == "mean":
            return states.mean(dim=1)
        elif self.mode.lower() == "last":
            return states[:, -1, :]

    def forward(self, x, requires_grad=True):
        """
        Perform forward method on a data batch

        Args:
            x (torch.Tensor): A batch of input data of shape (batch_size, sequence_len, input_size)
        Returns: (torch.Tensor): a regressed values (estimation) of shape (batch_size, 1)
        """
        # Obtain the LSTM output
        lstm_hidden_states = self.lstm.forward(x, requires_grad=requires_grad)
        lstm_output = self.process_states(lstm_hidden_states)
        # Return the regressor output
        return self.regressor.forward(lstm_output, requires_grad=requires_grad)

    def backward(self, loss_d, lr):
        """
        Execute a single backward step (always after a single forward step was executed)

        Args:
            loss_d (torch.Tensor): Chained gradient from the objective function
                (chained directly to regressor output)
            lr (float): Learning rate
        Returns: None
        """
        # Train the regressor
        chained_d = self.regressor.backward(loss_d, lr)
        # Obtain gradients for LSTM and apply them
        last_sequence_len = self.lstm.stack_size  # Obtain the last sequence len
        chained_divided_d = self.divide_grads(chained_d, last_sequence_len)
        self.lstm.backward(chained_divided_d, lr)

    def train(self, train_dataset, test_dataset, lr, batch_size, steps, steps_per_eval):
        """
        Train the model on given dataset

        Args:
            train_dataset (torch.utils.data.Dataset) instance of dataset for training the model
            test_dataset (torch.utils.data.Dataset) instance of dataset for testing the model
            lr (float): Learning rate
            batch_size (int): The size of batch for model to work with
            steps (int): How many data entries to train the model on (entries/batch_size) == num model updates
            steps_per_eval (int): How many trainig steps to perform eval
            tqdm (bool): Display the TQDM progress bar
        """
        # Here we keep data of traninig progress for plotting
        self.plot_data = ChartData(steps_per_eval)

        # Counting training steps to perform test eval in every steps_per_eval steps
        steps_to_eval = steps_per_eval - 1

        # Create the dataset iterator
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset._padded_batch_loader
        )

        steps_taken = 0
        pbar = tqdm(range(steps), desc="Training ...", ncols=100)

        while steps_taken < steps:
            for X, y in train_dataloader:
                # If the step limit achieved, finish the training
                if steps_taken >= steps:
                    break
                y_gt = y.unsqueeze(1)
                # Infere!
                y_pred = self.forward(X)

                # Make the GT a batch of one-element vectors and calculate the losses and gradients
                loss = mse_loss(y_gt, y_pred)
                correct = num_correct(y_gt, y_pred)
                self.plot_data.training_data.add_entry(correct, loss)

                loss_d = mse_loss_d(y_gt, y_pred)

                # Do the backward step
                self.backward(loss_d, lr=lr)

                # Testing part
                if steps_to_eval == 0:
                    steps_to_eval = steps_per_eval - 1
                    self.plot_data.testing_data.add_entry(*self.eval(test_dataset))
                    # Calculate and display the losses
                    loss_avg = sum(self.plot_data.training_data.loss[-8:]) / len(self.plot_data.training_data.loss[-8:])
                    last_eval_correct_rate = self.plot_data.testing_data.correct[-1]
                    pbar.set_description(
                        (
                            f"Training ... (l: {loss_avg:.4f}; test: {(last_eval_correct_rate) * 100:.0f} % "
                            f"{self.plot_data.testing_data.loss[-1]:.4f} l)"
                        )
                    )
                else:
                    steps_to_eval -= 1

                steps_taken += 1
                pbar.update(1)

    def eval(self, dataset):
        """
        Evaluate the model success rate on given dataset

        Args:
            dataset (torch.utils.data.Dataset): An instance of torch dataset for evaluation
        """
        X, y_gt = dataset.X, dataset.y
        y_pred = self.forward(X, requires_grad=False).squeeze(1)
        return num_correct(y_gt, y_pred), mse_loss(y_gt, y_pred)

    def pickle(self, file_descriptor):
        """Pickle self into opened file descriptor"""
        pickle.dump(self, file_descriptor)

    @staticmethod
    def from_pickle(file_descriptor):
        """Open the file descriptor and load the pickled binary self"""
        return pickle.load(file_descriptor)
