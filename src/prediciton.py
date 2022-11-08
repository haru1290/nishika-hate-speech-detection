import os
import pandas as pd
import torch
import torch.nn as nn
import argparse

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from load_data import *


def prediction(args):
    pass


def main(args):
    pass


if __name__ == "__main__":
    main()