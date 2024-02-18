# LSTM from scratch

This repository implements and documents a custom LSTM neural network used for sentiment analysis, more generally - single value regression from a written text. The Amazon and TripAdvisor datasets include particular reviews with star ranking given along them. Those rankings are the subject of prediction.
This project was made as a part of my MsC. studies in FIT - BUT: As a demonstration of understanding the inner working of recurrent (LSTM in particular) neural networks. The examples, even though were from the real world, be considered toy examples. The trained models could not be capable of meaningful operation given the combination of hardware limitations in computational power and memory (RTX 2060 eGPU + i7 11th gen), model architecture and the complexity of datasets 


# Key features
 - Custom backpropagation implementation in PyTorch (no autograd)
 - Bidirectional LSTM implemented
 - MLP regressor to work with the LSTM sequence extracted features with several types of aggregations
 - Trainable initial LSTM states
 - Dataset management, tokenizing, Word2Vec embedding and pickling the once done datasets
 - Real world and custom generated datasets

### How to setup venv for this project

This software works ideally in python3.8 virtual environment with requirements.txt installed.
Preffered way of installing and setting up the venv is:

```
conda create --name ZPJa python==3.8
conda activate ZPJa
pip install -r requirements.txt
```

### CLI arguments
Because of custom CLI arguements for each dataset, argparse gymnastics had to be done. This means all CLI runs have to be written in this format:
`src/lstm-cli.py -ds=sum_float -te=10 -tr=10 -ev=10 -d=cpu -bi -m=mean train`
In other words - all parameters need to be entered with equation sign (short or long versions of flags could be used) if they require a parameter. Otherwise, the code would fail.
This was kind of a hack and argparse does not really like it, so this order has to be admitted, otherwise it protests ...
