import string
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SEED = 42
DATA_DIR = Path("./data")

def main():
    train_valid_df = pd.read_csv(DATA_DIR/"train.csv")
    test_df = pd.read_csv(DATA_DIR/"test.csv")

    train_valid_df['text'].str.replace('[{}]'.format(string.punctuation), '')



if __name__ == "__main__":
    main()