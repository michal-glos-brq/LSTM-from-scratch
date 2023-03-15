### Project LSTM:
This is implementation of LSTM neural network from scratch (with numpy only). It's purpose is predicting the Amazon review star rating from the actual text of the review. For further imformation, read the `technicka_zprava.pdf` (which is availible in Czech language only, but Google translater handles that kind of job quite well now)

### How to run the app
List of lstm-cli.py script launch options:
 - `--naive-evaluate` statistical (no ml) approach to predictions, has priority over `--evaluate` option
 - `--evaluate` perform the dataset evaluation with LSTM
 - `--train-data-entries n` reviews to be trained on 
 - `--eval-data-entries n` reviews to be used for evaluation
 - `--data-ratio r` ratio of training data, has priority over `--train-data-entries n` and `--eval-data-entries n`
 - `--learning-rate` specify learning rate
 - `--lstm-cells n` specify LSTM cell count
 - `--epochs n` specify the number of epochs
 - `--load-model path` load model from path
 - `--save-movel path` save model to path