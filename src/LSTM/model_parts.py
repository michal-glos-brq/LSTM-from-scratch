"""
This file implements simple LSTM cell implementation with BTT and also Bidirectional LSTM cell.
The cell is capable of processing a batch of heterogenous input vectors, one at a time.

This file also implements a neural network model with 2 linear layers - first with activation sigmoid,
the second is None. This model works like rough regressor/estimator, final values have to be rounded
for integer ranking eventually when evaluating (decimals provide good gradients when learning)
"""
import torch

from LSTM.utils import LSTMGradParams, LSTMGrads, sigmoid_d, tanh_d


class LSTMCell:
    """
    A full LSTM cell implementation, handling heterogenous sequences in a batch with zero padding.
    """

    def __init__(self, input_size, hidden_size, device="cpu", torch_type=torch.float32):
        """
        Initialize the class

        Args:
            input_size (int): Cell input length
            hidden_size (int): Size of the hidden state vector
            device (str): Device to work on (GPU/CPU) or rather (cpu/cuda)
            torch_type (torch.dtype): Type of LSTM matrix values
        """
        self.hidden_size, self.input_size = hidden_size, input_size
        self.device, self.torch_type = device, torch_type
        # Initialize zero with zeros of shape (batch_size, hidden_size) as initial states
        self.init_cell_state = torch.zeros((1, hidden_size), requires_grad=False, device=device, dtype=torch_type)
        self.init_hidden_state = torch.zeros((1, hidden_size), requires_grad=False, device=device, dtype=torch_type)

        # Init forget gate
        self.Wf = torch.nn.init.xavier_uniform_(
            torch.empty(input_size + hidden_size, hidden_size, requires_grad=False, device=device, dtype=torch_type)
        )
        self.bf = torch.zeros((1, hidden_size), requires_grad=False, device=device, dtype=torch_type)
        # Init input gate
        self.Wi = torch.nn.init.xavier_uniform_(
            torch.empty(input_size + hidden_size, hidden_size, requires_grad=False, device=device, dtype=torch_type)
        )
        self.bi = torch.zeros((1, hidden_size), requires_grad=False, device=device, dtype=torch_type)
        # Init output gate
        self.Wo = torch.nn.init.xavier_uniform_(
            torch.empty(input_size + hidden_size, hidden_size, requires_grad=False, device=device, dtype=torch_type)
        )
        self.bo = torch.zeros((1, hidden_size), requires_grad=False, device=device, dtype=torch_type)
        # Init candidate state gate
        self.Wc = torch.nn.init.xavier_uniform_(
            torch.empty(input_size + hidden_size, hidden_size, requires_grad=False, device=device, dtype=torch_type)
        )
        self.bc = torch.zeros((1, hidden_size), requires_grad=False, device=device, dtype=torch_type)

        # Initialize memory stack for gradient computation (BPTT) and gradient cumulation object
        self.memory = LSTMGradParams()
        self.grads = LSTMGrads(input_size, hidden_size, device=self.device)

    def apply_grads(self, lr):
        """
        Apply cumulated gradients through sequence to weights and biases.

        Args:
            lr (float): Learning rate
        """
        # We use lr as limits because it's a faster way on how to apply the clamping after multiplying with learning rate
        self.Wf.sub(torch.clamp(self.grads.dWf, min=-1, max=1).mul(lr))
        self.bf.sub(torch.clamp(self.grads.dbf, min=-1, max=1).mul(lr))
        self.Wi.sub(torch.clamp(self.grads.dWi, min=-1, max=1).mul(lr))
        self.bi.sub(torch.clamp(self.grads.dbi, min=-1, max=1).mul(lr))
        self.Wo.sub(torch.clamp(self.grads.dWo, min=-1, max=1).mul(lr))
        self.bo.sub(torch.clamp(self.grads.dbo, min=-1, max=1).mul(lr))
        self.Wc.sub(torch.clamp(self.grads.dWc, min=-1, max=1).mul(lr))
        self.bc.sub(torch.clamp(self.grads.dbc, min=-1, max=1).mul(lr))
        self.grads.zero_grad()

    def apply_init_state_grads(self, h_state_chained_d, c_state_chained_d, lr):
        """
        Apply gradients on initial hidden and cell state tensors,
        both gradients have the same size as initial states themselves - (hidden_state, )

        Args:
            h_state_chained_d (torch.Tensor): Tensor with gradients for the init. hidden state
            c_state_chained_d (torch.Tensor): Tensor with gradients for the init. cell state
        """
        # Sum over batch and apply gradient descent to the intial state tensors
        init_h_state_grad = torch.sum(h_state_chained_d, dim=0)
        init_c_state_grad = torch.sum(c_state_chained_d, dim=0)
        self.init_hidden_state.sub(torch.clamp(init_h_state_grad, min=-1, max=1).mul(lr))
        self.init_cell_state.sub(torch.clamp(init_c_state_grad, min=-1, max=1).mul(lr))

    def forward_step(self, x, cell_state_in, hidden_state_in, requires_grad=True):
        """
        Perform single step forward

        Args:
            x (torch.Tensor): Input fot the one step, shape: (batch_size, input_size)
            cell_state_in (torch.Tensor): cell_state from previous timestamp of shape (batch_size, hidden_state_size)
            hidden_state_in (torch.Tensor): hidden_state from previous timestamp of shape (batch_size, hidden_state_size)
            requires_grad (bool): Keep track of tensor flow in order to compute gradients and apply gradient descent

        Returns: torch.Tensor of shape (batch_size, vector_size)
        """
        cat_input = torch.column_stack((hidden_state_in, x))
        forget_gate = torch.sigmoid(torch.matmul(cat_input, self.Wf).add(self.bf))
        input_gate = torch.sigmoid(torch.matmul(cat_input, self.Wi).add(self.bi))
        output_gate = torch.sigmoid(torch.matmul(cat_input, self.Wo).add(self.bo))
        candidate_gate = torch.tanh(torch.matmul(cat_input, self.Wc).add(self.bc))

        cell_state = forget_gate.mul(cell_state_in).add(input_gate.mul(candidate_gate))
        hidden_state = output_gate.mul(torch.tanh(cell_state))

        if requires_grad:
            self.memory.add_entry(
                cat_input, cell_state_in, cell_state, input_gate, output_gate, forget_gate, candidate_gate
            )

        return hidden_state, cell_state

    def forward(self, X, requires_grad=True):
        """
        Inference of batched input sequences

        Args:
            X (torch.Tensor) : (batch_size, sequence_len, sequence_element_size)
            requires_grad (bool): Keep track of data flow for BTT
        Returns: (torch.Tensor) : (batch_size, sequence_len, hidden_state_size)
        """
        # Clear gradient computation metadata
        self.memory.clear()
        # Initialize initial hidden and cell state values. Only vectors are allowed, hence
        # repeat is parametrized only for batch and single input dimenstions
        batch_size = X.shape[0]
        hidden_state = self.init_hidden_state.expand(batch_size, -1)
        cell_state = self.init_cell_state.expand(batch_size, -1)

        # Collect hidden states for output
        hidden_states = []
        # Transpose tensor so we iterate over sequences, not the batch
        for x in range(X.shape[1]):
            hidden_state, cell_state = self.forward_step(
                X[:, x, :].type(self.tensor_factory), cell_state, hidden_state, requires_grad=requires_grad
            )
            hidden_states.append(hidden_state)
        # Let's transpose it back to the original shape
        return torch.stack(hidden_states).transpose(1, 0)

    def backward_step(self, cell_state_d_in, hidden_state_d_in, cell_output_d_in):
        """
        Perform one backward step of gradient descent, cumulate gradients

        Args:
            cell_state_d_in (torch.Tensor): Chained derivative of cell state from previous timestep (bacth_size, hidden_state)
            hidden_state_d_in (torch.Tensor): Chained derivative of hidden state from previous timestep (bacth_size, hidden_state)
            cell_output_d_in (torch.Tensor): Chained derivative of hidden state obtained from model
                    output - regressor chained gradients (bacth_size, hidden_state)
        """
        # Invoke the tensors from memory stack
        (
            cat_input,
            cell_state_in,
            cell_state_out,
            input_gate,
            output_gate,
            forget_gate,
            candidate_gate,
        ) = self.memory.memory.pop()
        self.memory.stack_size -= 1

        # Compute the state gradients
        hidden_state_d = hidden_state_d_in.add(cell_output_d_in)
        cell_state_d = tanh_d(cell_state_out).mul(output_gate).mul(hidden_state_d).add(cell_state_d_in)
        # Computa candidate state gate gradients
        candidate_gate_d = cell_state_d.mul(input_gate).mul(tanh_d(candidate_gate))
        # Compute other state gradients
        forget_gate_d = candidate_gate_d.mul(cell_state_in).mul(sigmoid_d(forget_gate))
        output_gate_d = torch.tanh(cell_state_out).mul(hidden_state_d).mul(sigmoid_d(output_gate))
        input_gate_d = cell_state_d.mul(candidate_gate.mul(sigmoid_d(input_gate)))

        # Cumulate gradients (scaled by learning rate when applying gradients)
        self.grads.dWf.add(torch.matmul(cat_input.T, forget_gate_d))
        self.grads.dbf.add(torch.sum(forget_gate_d, dim=0))
        self.grads.dWi.add(torch.matmul(cat_input.T, input_gate_d))
        self.grads.dbi.add(torch.sum(input_gate_d, dim=0))
        self.grads.dWo.add(torch.matmul(cat_input.T, output_gate_d))
        self.grads.dbo.add(torch.sum(output_gate_d, dim=0))
        self.grads.dWc.add(torch.matmul(cat_input.T, candidate_gate_d))
        self.grads.dbc.add(torch.sum(candidate_gate_d, dim=0))

        # Derivative of concatenated vectors of LSTM input and hidden_state_in (will be split further)
        combined_gate_gradients = (
            torch.matmul(forget_gate_d, self.Wf.T)
            .add(torch.matmul(candidate_gate_d, self.Wc.T))
            .add(torch.matmul(input_gate_d, self.Wi.T))
            .add(torch.matmul(output_gate_d, self.Wo.T))
        )

        # Now calculate the chained derivative for LSTM input (X)
        # and it's hidden and cell state derivatives for previous timestep
        cell_state_chain_d = forget_gate.mul(cell_state_d)
        input_chain_d = combined_gate_gradients[:, self.hidden_size :]
        hidden_state_chain_d = combined_gate_gradients[:, : self.hidden_size]

        # Return chained derivatives for backpropagation through time and layesr
        # Tuple:
        ## - chained derivative for cell state from t-1
        ## - chained derivative for hidden state from t-1
        ## - chained derivative for cell input from t
        return cell_state_chain_d, hidden_state_chain_d, input_chain_d

    def backward(self, hidden_states_d_in, lr):
        """
        Perform backpropagation through time
        Args:
            hidden_states_d_in (torch.Tensor) : chained derivatives from the output of shape:
                                                    (batch_size, sequence, hidden_size)
            lr (float) : learning rate
        """
        assert self.memory.stack_size == hidden_states_d_in.shape[1]
        self.grads.zero_grad()

        cell_state_d = torch.zeros_like(
            hidden_states_d_in[:, 0, :], requires_grad=False, device=self.device, dtype=self.torch_type
        )
        hidden_state_d = torch.zeros_like(
            hidden_states_d_in[:, 0, :], requires_grad=False, device=self.device, dtype=self.torch_type
        )

        # Collect derivatives
        input_derivatives = []
        h_state_derivatives = []
        c_state_derivatives = []

        # Do the backward steps, transpose, cause we iterate through timesteps, not the batch
        for i in range(hidden_states_d_in.shape[1]):  # hidden_states_d_in.transpose(1, 0):
            cell_state_d, hidden_state_d, input_chained_d = self.backward_step(
                cell_state_d, hidden_state_d, hidden_states_d_in[:, i, :]
            )
            input_derivatives.append(input_chained_d)
            c_state_derivatives.append(cell_state_d)
            h_state_derivatives.append(hidden_state_d)

        # Pass there the last chained derivatives
        self.apply_init_state_grads(h_state_derivatives[-1], c_state_derivatives[-1], lr)
        self.apply_grads(lr)

        return (
            torch.stack(c_state_derivatives),
            torch.stack(h_state_derivatives),
            torch.stack(input_derivatives),
        )

    @property
    def torch_factory(self):
        return torch.cuda if self.device == "cuda" else torch

    @property
    def tensor_factory(self):
        if self.torch_type == torch.float16:
            return self.torch_factory.HalfTensor
        elif self.torch_type == torch.float32:
            return self.torch_factory.FloatTensor
        if self.torch_type == torch.float64:
            return self.torch_factory.DoubleTensor

    @property
    def stack_size(self):
        """Access stack size (of inference history) from LSTM cell class"""
        return self.memory.stack_size


class BiLSTMCell:
    """A specific BiDirectional LSTM cell, composed of 2 vanilla LSTM cells"""

    def __init__(self, input_size, hidden_size, device="cpu", torch_type=torch.float32):
        assert hidden_size % 2 == 0, "Hidden state size of the bidirectional LSTM has to be even!"
        self.input_size = input_size
        self.new_hidden_size = int(hidden_size / 2)
        self.forwardLSTM = LSTMCell(input_size, self.new_hidden_size, device=device, torch_type=torch_type)
        self.backwardLSTM = LSTMCell(input_size, self.new_hidden_size, device=device, torch_type=torch_type)

    def forward(self, X, requires_grad=True):
        """
        Forward input to both LSTM cells, reverse the sequence for the backwards cell and
        concatenate their hidden states as outputs
        """
        forwardLSTM_h = self.forwardLSTM.forward(X, requires_grad=requires_grad)
        # Just flip the sequence dimension
        backwardLSTM_h = self.backwardLSTM.forward(torch.flip(X, dims=[1]), requires_grad=requires_grad)
        return torch.cat((forwardLSTM_h, backwardLSTM_h), dim=2)

    def backward(self, hidden_states_d_in, lr):
        """Perform backpropagation in both cells accordingly"""
        # We have to split the vectors in half, take all batches, all sequence steps and just half the element vector
        self.forwardLSTM.backward(hidden_states_d_in[:, :, : self.new_hidden_size], lr)
        self.backwardLSTM.backward(hidden_states_d_in[:, :, self.new_hidden_size :], lr)

    @property
    def stack_size(self):
        """
        Access stack size (of inference history) from LSTM cell class
        Both stacks should have equal stack size, return one
        """
        assert self.forwardLSTM.stack_size == self.backwardLSTM.stack_size, "Bidirectional stack size do not compare!"
        return self.forwardLSTM.stack_size


class Regressor:
    """
    Simple dense neural network to process LSTM features into a predicted value

    Manually programmed backpropagation results in possibl only one last batch being
    used for training, as previous batch process data would be overwritten by the new one
    """

    def __init__(self, input_size, hidden_size, device="cpu", torch_type=torch.float32):
        """Initialize the regression model"""
        self.hidden_size = hidden_size
        self.torch_type = torch_type
        self.device = device
        self.W1 = torch.nn.init.xavier_uniform_(
            torch.empty((input_size, hidden_size), requires_grad=False, device=device, dtype=self.torch_type)
        )
        self.W2 = torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, 1), requires_grad=False, device=device, dtype=self.torch_type)
        )
        self.b1 = torch.zeros((hidden_size,), requires_grad=False, device=device, dtype=self.torch_type)
        self.b2 = torch.zeros((1,), requires_grad=False, device=device, dtype=self.torch_type)

    @property
    def torch_factory(self):
        return torch.cuda if self.device == "cuda" else torch

    @property
    def tensor_factory(self):
        if self.torch_type == torch.float16:
            return self.torch_factory.HalfTensor
        elif self.torch_type == torch.float32:
            return self.torch_factory.FloatTensor
        if self.torch_type == torch.float64:
            return self.torch_factory.DoubleTensor

    def forward(self, X, requires_grad=True):
        """
        Forward pass with 2 layered regressor

        X (torch.Tensor) : Input tensor of shape (batch_size, input_size)
        """
        hidden_preactiv_output = torch.matmul(X, self.W1).add(self.b1)
        hidden_output = torch.sigmoid(hidden_preactiv_output)
        output = torch.matmul(hidden_output, self.W2).add(self.b2)

        # Keep data for gradient computation
        if requires_grad:
            self.X = X
            self.hidden_preactiv_output = hidden_preactiv_output
            self.hidden_output = hidden_output

        return output

    def backward(self, loss_grad, lr):
        """
        Perform backward pass and apply gradients

        Args:
            loss_grad (torch.Tensor): Chained derivative of objective function of shape (batch_size, 1)
            lr (float): Learning rate
        """
        # loss_grad is actually chained gradient, derivative of MSE and activation function, which is None
        loss_grad = loss_grad.type(self.tensor_factory)
        hidden_layer_loss = torch.matmul(loss_grad, self.W2.T)
        hidden_layer_chain_d = hidden_layer_loss.mul(sigmoid_d(self.hidden_preactiv_output))

        # Update weights and biases
        dW2 = torch.matmul(self.hidden_output.T, loss_grad)
        db2 = torch.sum(loss_grad, dim=0)  # Sum over batches

        # We use lr as limits because it's a faster way on how to apply the clamping after multiplying with learning rate
        self.W2 -= torch.clip(dW2, min=-1, max=1).mul(lr)
        self.b2 -= torch.clip(db2, min=-1, max=1).mul(lr)

        dW1 = torch.matmul(self.X.T, hidden_layer_chain_d)
        db1 = torch.sum(hidden_layer_chain_d, dim=0)  # Sum over batches
        self.W1 -= torch.clip(dW1, min=-1, max=1).mul(lr)
        self.b1 -= torch.clip(db1, min=-1, max=1).mul(lr)

        # Return chained derivatives for layers before the classifier
        out_chain_d = torch.matmul(hidden_layer_chain_d, self.W1.T)
        return out_chain_d
