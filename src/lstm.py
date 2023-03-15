                                    #####################################
                                    #@$%&                           &%$@#
                                    #!      SFC - Projekt LSTM         !#
                                    #!          Michal Glos            !#
                                    #!           xglosm01              !#
                                    #!              __                 !#
                                    #!            <(o )___             !#
                                    #!             ( ._> /             !#
                                    #!              `---'              !#
                                    #@$%&                           &%$@#
                                    #####################################

# This module contains implementation of LSTM neural network

from tqdm import tqdm
import os
import numpy as np
import amznDS
import sys
import pickle
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    '''Apply sigmoid function to input numpy vector'''
    return 1/(1+np.exp(-x))

def d_sigmoid(y):
    '''Sigmoid derivative of input numpy vector'''
    return y*(1-y)

def d_tanh(y):
    '''Tanh derivative of input numpy vector'''
    return 1 - y**2

def softmax(x):
    '''Softmax vector from input numpy vector'''
    return np.exp(x) / (np.sum(np.exp(x)) + sys.float_info.min)

def cross_entropy(output, gt):
    '''Calculate cross entropy loss function'''
    # This should be a sum, but as long as ground truth is vector of zeros with
    # one 1, 0*ln(output) = 0 -> whatever the output is
    return -np.log(output[0, gt-1] + sys.float_info.min)

def print_success_rate(data):
    '''
    Print success rate of LSTM evaluation
    
    Input arguments:
        data:   list of tuples - ([class predictions], gt value)
    '''
    gt = np.array([y[1] for y in data])
    guess = [np.argmax(y[0]) + 1 for y in data]
    print("Success rate: " + str(((gt - guess) == 0).sum() / len(gt) * 100) + "%")

def naive_evaluate(XX, Y):
    '''
    Evaluate input class naively by probabilities from embedded vectors
    
    Input arguments:
        XX: list of lists of words
        Y:  class ground truth
    '''
    # Load embedding dictionary with probabilities
    embed_dict = emb_dict()
    results = []
    # For each review - sum the probabilities of occurence in each class for each word
    for X, y in tqdm(list(zip(XX, Y)), desc="Naive evaluation:", ncols=100):
        mid_results = np.zeros(5)
        for x in X:
            vector = embed_dict[amznDS.word_trim_lower(x)]
            mid_results += vector
        results.append((mid_results, y))
    print_success_rate(results)
        

def emb_dict():
    '''Load embedding dictionary from pickle file or create one'''
    # Load the embedding dictionary, if not found, create it
    if not os.path.exists(amznDS.EMBED):
        amznDS.create_embedding_dict()
    with open(amznDS.EMBED, 'rb') as f:
        return pickle.load(f)


class LSTM:
    '''
    Model of LSTM neural network
    
    Properties:
        self.lr:        learning rate
        self.model:     model of LSTM containing weights and biases
        self.lstm_w:    number of LSTM cells in network
        self.emb_dict:  embedding dictionary

    '''


    def __init__(self, n_classes, lstm_cells, learning_rate):
        '''
        Initialize LSTM model.

        Input arguments:
            n_classes:      number of classes to be predicted
            lstm_cells:     number of LSTM cells in network
            learning_rate:  learning rate requested for this network
        '''
        self.lr = learning_rate
        self.lstm_w = lstm_cells
        self.model = {
            # initiate model weights (random) and biases (zeros)
            # forget gate weights and biases
            "Wf": np.random.randn(n_classes + lstm_cells, lstm_cells) / (n_classes + lstm_cells)**(1/2),
            "Bf": np.zeros((1, lstm_cells)),
            # input gate weights and biases
            "Wi": np.random.randn(n_classes + lstm_cells, lstm_cells) / (n_classes + lstm_cells)**(1/2),
            "Bi": np.zeros((1, lstm_cells)),
            # correct input weights and biases
            "Wc": np.random.randn(n_classes + lstm_cells, lstm_cells) / (n_classes + lstm_cells)**(1/2),
            "Bc": np.zeros((1, lstm_cells)),
            # output gate wieghts and biases
            "Wo": np.random.randn(n_classes + lstm_cells, lstm_cells) / (n_classes + lstm_cells)**(1/2),
            "Bo": np.zeros((1, lstm_cells)),
            # dense output network gates and biases
            "Wy": np.random.randn(lstm_cells, n_classes) / (n_classes + lstm_cells)**(1/2),
            "By": np.zeros((1, n_classes))
        }
        # Get and save embedding dictionary
        self.emb_dict = emb_dict()


    def _forward_step(self, in_X, old_h, old_c):
        '''
        Make one forward step iteration (for single word).

        Input arguments:
            in_X:   input word
            old_h:  hidden state from previous iteration
            old_c:  cell state from previous iteration
        
        Returns:
            cache: dictionary with all parameters needed for backpropagation
        '''
        # Embed the input and concatenate it with previous state
        cache = {}
        in_X = self.emb_dict[amznDS.word_trim_lower(in_X)]
        cache['X'] = np.column_stack((old_h, in_X.reshape((1, -1))))
        # Remember old states
        cache['h_old'], cache['c_old'] = old_h, old_c
       
        # Calculate the network values
        cache['Hf'] = sigmoid(cache['X'] @ self.model['Wf'] + self.model['Bf'])
        cache['Hi'] = sigmoid(cache['X'] @ self.model['Wi'] + self.model['Bi'])
        cache['Ho'] = sigmoid(cache['X'] @ self.model['Wo'] + self.model['Bo'])
        cache['Hc'] = np.tanh(cache['X'] @ self.model['Wc'] + self.model['Bc'])
        
        # Calculate network state and output
        cache['c'] = cache['Hf'] * old_c + cache['Hi'] * cache['Hc']
        cache['h'] = cache['Ho'] * np.tanh(cache['c'])
        
        # Calculate fully connected layer and feed it into softmax activation function
        cache['output'] = softmax(self.model['By'] + cache['h'] @ self.model['Wy'])
        
        return cache


    def _backward_step(self, y_gt, cache, dh_chain, dc_chain):
        '''
        Compute gradients for backpropagation
        
        Input arguments:
            y_gt:       class ground truth
            cache:      cache from forward step
            dh_chain:   output of previous backward step (chain rule)
            dc_chain:   output of previous backward step (chain rule)
        
        Returns:
            gradients:  weights and biases gradients
            dh_chain:   hidden state derivative for next backpropagation iteration
            dc_chain:   cell state derivative for next backpropagation iteration
        '''
        # declare gradient dictionary
        gradients = {}
        # Output gradient
        doutput = cache['output'].copy()
        doutput[0, y_gt-1] -= 1

        # Fully connected -> output layer gradient
        gradients['Wy'] = cache['h'].T @ doutput
        gradients['By'] = doutput

        # Hidden state gradient
        dh = doutput @ self.model['Wy'].T + dh_chain
        # Carry state gradient
        dc = cache['Ho'] * dh * d_tanh(cache['c']) + dc_chain

        # Forget gate gradient
        dhf = d_sigmoid(cache['Hf']) * (cache['c_old'] * dc)
        # Input gate gradient
        dhi = d_sigmoid(cache['Hi']) * (cache['Hc'] * dc)
        dhc = d_tanh(cache['Hc']) * (cache['Hi'] * dc)
        # Output gate gradient
        dho = d_sigmoid(cache['Ho']) * (np.tanh(cache['c']) * dh)
        
        # Weights and biases cells gradients
        # Weights
        gradients['Wf'] = cache['X'].T @ dhf
        gradients['Wi'] = cache['X'].T @ dhi
        gradients['Wc'] = cache['X'].T @ dhc
        gradients['Wo'] = cache['X'].T @ dho
        # Biases
        gradients['Bf'] = dhf
        gradients['Bi'] = dhi
        gradients['Bc'] = dhc
        gradients['Bo'] = dho
        # Input gradient
        dX = (dhf @ self.model['Wf'].T) + (dhi @ self.model['Wi'].T) + \
             (dhc @ self.model['Wc'].T) + (dho @ self.model['Wo'].T)
        # Calculate derivatives for chain rule (dh - do not care about input X)
        dh_chain, dc_chain = dX[:, :self.lstm_w], cache['Hf'] * dc
        return gradients, dh_chain, dc_chain


    def _train_step(self, X, y):
        '''
        Train network on whole data entry

        Input arguments:
            X:  list of words to be fed into LSTM
            y:  class ground truth
        
        Returns:
            gradients:  gradients from backpropagation
            loss:       loss function value
        '''
        # First forward step will be calculated with state h=0, c=0
        cache = [self._forward_step(X[0], np.zeros((1, self.lstm_w)), np.zeros((1, self.lstm_w)))]
        # Forward calculation with calculated states
        for x in X[1:]:
            cache.append(self._forward_step(x, cache[-1]['c'], cache[-1]['h']))
        loss = cross_entropy(cache[-1]['output'], y)
        
        # Declare placeholders for gradients
        gradients = {key: np.zeros_like(self.model[key]) for key in self.model}
        # For last step, previous gradients of states are all zeros
        dh, dc = np.zeros(self.lstm_w), np.zeros(self.lstm_w)
        
        # Cycle through backpropagation
        for x, _cache in reversed(list(zip(X, cache))):
            _gradients, dh, dc = self._backward_step(y, _cache, dh, dc)
            gradients = {key: gradients[key] + _gradients[key] for key in gradients.keys()}
        
        return gradients, loss


    def _apply_gradients(self, gradients):
        '''Apply gradients computed during backpropagation steps'''
        # Apply gradients multiplied by learning rate
        gradients = {key: gradients[key] * self.lr for key in gradients.keys()}
        self.model = {key: self.model[key] - gradients[key] for key in self.model}


    def _plot_losses(self, losses, single_epoch_losses, epoch, max_points=10000):
        '''
        Plot losses and save the plot.
        
        This method will generate two plots. One for the whole training process
        and second only with losses from the last epoch. Because of too large 
        lists of losses, convolution is used in order for the plot to be comprehensive.
        If the len of losses exceeds max_points, every nth loss will be plotted,
        otherwise the generated svg would be too large to view.

        Input arguments:
            losses:                 list of losses from training
            single_epoch_losses:    number of losses from single epoch
            epoch:                  number of epochs undertaken
            max_points:             maximal number of plotted entries
        '''
        # Setup the figure parameters
        plt.rcParams.update({'font.size': 45})
        fig, ax = plt.subplots(1, 1, figsize=(100,25))
        # Get losses for single epoch
        epoch_losses = losses[(epoch-1)*single_epoch_losses:]
        for l, name in tqdm(list(zip([losses, epoch_losses], ['all', ''])), desc="Plotting charts", ncols=100):
            # Determine convolution sizes and apply the convolution
            conv3 = min(20000, int(len(l) / 5))
            conv2 = int(conv3 / 5)
            conv1 = int(conv2 / 5)
            _conv1 = np.convolve(l, np.ones(conv1), 'valid') / conv1
            _conv2 = np.convolve(l, np.ones(conv2), 'valid') / conv2
            _conv3 = np.convolve(l, np.ones(conv3), 'valid') / conv3
            # First, reverse the loss list, second, slice it from last to nth element where
            # n is from <0, len(l)-len(_convX)>. for each array, sum it and divide by it's length 
            _conv1 = np.concatenate((np.array([sum(n)/len(n) for n in [l[:len(l)-len(_conv1)-i+1] for i in range(conv1)]][::-1]), _conv1))
            _conv2 = np.concatenate((np.array([sum(n)/len(n) for n in [l[:len(l)-len(_conv2)-i+1] for i in range(conv2)]][::-1]), _conv2))
            _conv3 = np.concatenate((np.array([sum(n)/len(n) for n in [l[:len(l)-len(_conv3)-i+1] for i in range(conv3)]][::-1]), _conv3))
            # If more then max points loss values -> create sparse plots
            step = 1
            if len(l) > max_points:
                # Determine the needed step
                step = int(len(l)/max_points) + 1
                _conv1 = _conv1[::step]
                _conv2 = _conv2[::step]
                _conv3 = _conv3[::step]
            # plot losses with step size
            x_ticks = list(map(lambda x: x*step, list(range(len(_conv1)))))
            ax.plot(x_ticks, _conv1, label=f"Rolling avg. {conv1}")
            ax.plot(x_ticks, _conv2, label=f"Rolling avg. {conv2}")
            ax.plot(x_ticks, _conv3, label=f"Rolling avg. {conv3}")
            # set other plot features
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel("Training steps")
            plt.ylabel("Loss")
            # Save the plot and clear it for next iteration
            if not os.path.exists('plots'):
                os.mkdir('plots')
            fig.savefig(f"plots/{self.lstm_w}_{self.lr}_epoch{epoch}{name}plot.svg")
            ax.cla()


    def _save_model(self, model_name):
        '''Pickle the model without embedding dictionary'''
        tmp_embed_dict = self.emb_dict
        self.emb_dict = {}
        with open(model_name, 'wb') as f:
            pickle.dump(self, f)
        self.emb_dict = tmp_embed_dict


    def train(self, XX, Y, testX, testY, model_name, epochs=1):
        '''
        Train the LSTM on given data

        Input arguments:
            XX:     training input data - list of lists of words
            Y:      training list of ground truth classes - list of ints
            testX:  evaluate input
            testY:  evaluate ground truth
            model_name: name of the model to be saved (further processed in code)
        '''
        # If emb dict not loaded, load it
        if not self.emb_dict:
            self.emb_dict = emb_dict()
        # Check if destination folder exists, if not, create it
        if not os.path.exists(os.path.dirname(model_name)):
            os.makedirs(os.path.dirname(model_name))

        # Actual training
        losses = np.array([])
        for epoch in range(1, epochs + 1):
            # Shuffle the values and classes the same way
            shuffle = np.random.permutation(len(XX))
            XX, Y = XX[shuffle], Y[shuffle]
            for X, y in tqdm(list(zip(XX, Y)), desc=f"Epoch {epoch}:", ncols=100):
                gradients, loss = self._train_step(X, y)
                # Do not crash when the learning explodes/implodes
                if math.isnan(loss) or math.isinf(loss):
                    loss = 0
                losses = np.append(losses, loss)
                self._apply_gradients(gradients)
            print_success_rate(self.evaluate(testX, testY))
            # Spare some space with 'caching' embed dict
            self._save_model(model_name + f"_epoch_{epoch}.p")
            self._plot_losses(losses, len(XX), epoch)

    def _evaluate_entry(self, X):
        '''
        Evaluate single data entry entry

        Input arguments:
            X: list of words

        Returns:
            chache of last forward step
        '''
        # Initialize 0th state with zero hidden and cell state
        cache = {'c': np.zeros((1, self.lstm_w)), 'h': np.zeros((1, self.lstm_w))}
        for x in X:
            cache = self._forward_step(x, cache['h'], cache['c'])
        return cache['output']


    def evaluate(self, XX, Y):
        '''
        Evaluate network on given dataset
        
        Input arguments:
            XX: list of lists of words
            Y:  list of class ground truths
        
        Returns:
            list of tuples - ([class predictions], ground truth)
        '''
        if not self.emb_dict:
            self.emb_dict = emb_dict()
        guesses = []
        for X, y in tqdm(list(zip(XX, Y)), ncols=100, desc="Evaluating"):
            guesses.append((self._evaluate_entry(X), y))
        return guesses

