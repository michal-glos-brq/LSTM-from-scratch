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

##### This is the main script controlling LSTM network and its' actions #####

import argparse
import lstm
import amznDS
import pickle
import numpy as np

# Maximal cardinality of Video Games dataset wuth same amount of data for each class
TOTAL_REVIEWS = 124393 * 5

class LSTMctl:
    '''Class controlling LSTM actions'''

    def __init__(self, args):
        '''Initialize this class based on input arguments'''
        # Recognize one of requested actions - if none present, default action is training
        self.naive_evaluate = args.naive_evaluate
        if args.evaluate and not args.load_model:
            print("ERROR! You have to specify model path to be evaluated.")
            exit(1)
        self.evaluate = args.evaluate or self.naive_evaluate

        # If ratio required, load the whole dataset and decide,
        # how much data are in training and evaluating dataset
        self.data_ratio = None
        if args.data_ratio or args.data_ratio == 0:
            self.data_ratio = args.data_ratio
            self.data_ratio = min(1, max(0, args.data_ratio))
            self.test_data_len = int(TOTAL_REVIEWS * (1 - args.data_ratio))
            self.train_data_len = int(TOTAL_REVIEWS * args.data_ratio) if not self.evaluate else 0
        else:
            self.test_data_len = args.eval_data_entries
            # Load train data length (if evaluation only, set it to 0)
            self.train_data_len = args.train_data_entries if not self.evaluate else 0

        # Check if requested data size is not bigger, the all entries in amznDS
        # It's in loop in case of numerical error, hopefully will converge
        total_data_size = self.train_data_len + self.test_data_len
        while (total_data_size) > TOTAL_REVIEWS:
            self.train_data_len = int((self.train_data_len / total_data_size) * TOTAL_REVIEWS)
            self.test_data_len = int((self.test_data_len / total_data_size) * TOTAL_REVIEWS)
            total_data_size = self.test_data_len + self.train_data_len
            print(f"New training data size: {self.train_data_len}\nNew testing data size: {self.test_data_len}")

        # Store other important values - learning rate, number of lstm cells and requested epochs for training
        self.lr = args.learning_rate
        self.lstm_cells = args.lstm_cells
        self.epochs = args.epochs
    
        # Path to save the model default: LSTM_lr_lstmcells.p
        self.save_path = args.save_model if args.save_model else f"models/LSTM_{self.lr}_{self.lstm_cells}"
        if self.save_path.endswith('.p'):
            self.save_path = self.save_path[:-2]

        # Load or initiate LSTM neural network model
        if args.load_model:
            with open(args.load_model, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = lstm.LSTM(5, self.lstm_cells, self.lr)

        # Load the dataset
        self.load_data()
    
    def load_data(self):
        '''Load Amazon reviews based on cli arguments'''
        # If data ration provided, loaa all data. Otherwise request cropped dataset
        if self.data_ratio:
            data = amznDS.get_data()
        else:
            single_class_entries = int((self.train_data_len + self.test_data_len) / 5)
            data = amznDS.get_data(one_class_size=single_class_entries)
        # Get train data and split according to command line arguments
        self.train_data = data[:self.train_data_len]
        self.trainX = np.array([d['reviewText'] for d in self.train_data], dtype=object)
        self.trainY = np.array([d['overall'] for d in self.train_data])
        # Get evaluate data and split according to command line arguments
        self.test_data = data[self.train_data_len:]
        self.testX = np.array([d['reviewText'] for d in self.test_data], dtype=object)
        self.testY = np.array([d['overall'] for d in self.test_data])

    def execute_action(self):
        '''Exacute one of requested actions (evaluate, train, naive evaluate)'''
        if self.evaluate:
            if self.naive_evaluate:
                lstm.naive_evaluate(self.testX, self.testY)
            else:
                lstm.print_success_rate(self.model.evaluate(self.testX, self.testY))
        else:
            print("First, evaluate the network before trainig.")
            lstm.print_success_rate(self.model.evaluate(self.testX, self.testY))
            self.model.train(self.trainX, self.trainY, self.testX, self.testY, self.save_path, epochs=self.epochs)
        

# Define argument parser
parser = argparse.ArgumentParser()
# First decide if the model would train or just evaluate (bool: evaluate)
parser.add_argument('--naive-evaluate', action='store_true',
    help="Run naive (no neural network) evaluation only.")
parser.add_argument('--evaluate', action='store_true',
    help="Run evaluation only. You probably want to load a model first (--load-model $MODEL_PATH).")
# Define how much data we want to use for trainig end evaluation
parser.add_argument('--train-data-entries', action='store', type=int, default=100000,
    help="How much data entries would be used to train the network? Default: 100000")
parser.add_argument('--eval-data-entries', action='store', type=int, default=10000,
    help="How much data entries would be used to evaluate the network? Default: 10000")
# If defined ratio between training and testing data, load all data
parser.add_argument('--data-ratio', action='store', type=float, default=None,
    help="If you want all the data, just enter ration test_data/all_data")
# Set the learning rate (Best results were reached with value of 0.00005)
parser.add_argument('--learning-rate', action='store', type=float, default=0.00005,
    help="Set the learning rate. Recommended values are below 0.01. Default: 0.00005")
# Set number of LSTM cells
parser.add_argument('--lstm-cells', action='store', type=int, default=96,
    help="Set the number of LSTM cells in the neural network. Default: 96")
# Set number of epochs for training
parser.add_argument('--epochs', action='store', type=int, default=1,
    help="Set the number of epochs for training. Default: 1")
# Load model
parser.add_argument('--load-model', action='store', type=str, default=None,
    help="If set, will load model from a file, instead of initializing a new one.")
# Save model
parser.add_argument('--save-model', action='store', type=str, default=None,
    help="Set path to save the model after training. If not set, model will be " + \
    "stored as data/LSTM_{$LR}_{$LSTM_CELLS}.p, formatted with learning rate and LSTM cells.")

if __name__ == "__main__":
    # Parse arguments, construct the LSTMctl class and execute required action
    args = parser.parse_args()
    ctl = LSTMctl(args)
    ctl.execute_action()
