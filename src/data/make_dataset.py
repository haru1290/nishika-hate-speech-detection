# データのダウンロード

import pandas as pd
from torch.utils.data import Dataset


class HateSpeechDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        input = {
            "input_ids": self.X[index]["input_ids"],
            "attention_mask": self.X[index]["attention_mask"],
        }

        if self.y is not None:
            input["labels"] = self.y[index]

        return input
