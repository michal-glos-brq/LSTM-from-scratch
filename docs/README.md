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