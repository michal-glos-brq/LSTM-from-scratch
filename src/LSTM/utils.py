from dataclasses import dataclass, field
from typing import List, Tuple
import torch


# Derivative of the sigmoid function
def sigmoid_d(x):
    """Sigmoid derivative
    Parameters:
        x (torch.Tensor) : Input tensor
    """
    return x.mul(x.sub(1).mul(-1))


def tanh_d(x):
    """Tangens derivative
    Parameters:
        x (torch.Tensor) : Input tensor
    """
    return torch.tanh(x).pow(2).sub(1).mul(-1)


# Mean Squared Error loss function
def mse_loss(y_gt, y_pred):
    """Mean square error loss fucntion
    Parameters:
        y_gt (torch.Tensor) : ground truth tensor, (batch_size, 1)
        y_pred (torch.Tensor) : tensor predicted by the model, (batch_size, 1)
    """
    return (((y_gt.sub(y_pred)).pow(2)).mean())


def mse_loss_d(y_gt, y_pred):
    """Derivative of mean square error loss fucntion
    Parameters:
        y_gt (torch.Tensor) : ground truth tensor, (batch_size, 1)
        y_pred (torch.Tensor) : tensor predicted by the model, (batch_size, 1)
    """
    return y_pred.sub(y_gt)


def num_correct(y_gt, y_pred):
    """Obtain the ration of cerrect predicitons"""
    # Rounding the GTs so our regression model with floats could learn from floats with enough gradient
    # while the class is it's rounded value - it has to learn how to count properly
    return ((torch.round(y_pred) == torch.round(y_gt)).sum().div(y_pred.shape[0]))


@dataclass
class LSTMGradParams:
    """This dataclass serves as a stack for LSTM data-flow for gradient computation"""
    memory: List[Tuple] = field(default_factory=list)
    stack_size: int = field(default_factory=lambda: 0)

    def add_entry(self, cat_input, cell_state_in, cell_state, input_gate, output_gate, forget_gate, candidate_gate):
        """Add a single timestep inference data on the stack"""
        self.memory.append((cat_input, cell_state_in, cell_state, input_gate, output_gate, forget_gate, candidate_gate))
        self.stack_size += 1

    def clear(self):
        """Clear the whole stack"""
        assert self.stack_size == 0 and self.memory == [], "Something went wrong in keeping inference data."
        self.stack_size = 0
        self.memory.clear()


class LSTMGrads:
    """Class for cumulating weight and bias gradients for a LSTM cell"""

    def __init__(self, input_size, hidden_size, device="cpu", torch_type=torch.float32):
        """
        Args:
            input_size (int): Input vector len
            hidden_size (int): Hidden LSTM state (vector) size
        """
        self.zero = torch.Tensor()
        # Initialize weight grads
        self.dWf = torch.zeros(
            (input_size + hidden_size, hidden_size), requires_grad=False, device=device, dtype=torch_type
        )
        self.dWi = torch.zeros(
            (input_size + hidden_size, hidden_size), requires_grad=False, device=device, dtype=torch_type
        )
        self.dWo = torch.zeros(
            (input_size + hidden_size, hidden_size), requires_grad=False, device=device, dtype=torch_type
        )
        self.dWc = torch.zeros(
            (input_size + hidden_size, hidden_size), requires_grad=False, device=device, dtype=torch_type
        )
        # Initialize bias grads
        self.dbf = torch.zeros((hidden_size,), requires_grad=False, device=device, dtype=torch_type)
        self.dbi = torch.zeros((hidden_size,), requires_grad=False, device=device, dtype=torch_type)
        self.dbo = torch.zeros((hidden_size,), requires_grad=False, device=device, dtype=torch_type)
        self.dbc = torch.zeros((hidden_size,), requires_grad=False, device=device, dtype=torch_type)

    def zero_grad(self):
        """Reset all gradients to zeros"""
        self.dWf[:] = 0
        self.dWi[:] = 0
        self.dWo[:] = 0
        self.dWc[:] = 0
        self.dbf[:] = 0
        self.dbi[:] = 0
        self.dbo[:] = 0
        self.dbc[:] = 0
