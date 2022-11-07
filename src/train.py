import os
import random
from pathlib import Path

import string
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer, EvalPrediction, Trainer, TrainingArguments, AutoModelForSequenceClassification,
)

SEED = 42
DATA_DIR = Path("./data")


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
            input["label"] = self.y[index]

        return input


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = f1_score(p.label_ids, preds)
    return {"f1_score":result}


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    train_valid_df = pd.read_csv(DATA_DIR/"train.csv")
    test_df = pd.read_csv(DATA_DIR/"test.csv")
    sub_df = pd.read_csv(DATA_DIR/"sample_submission.csv")

    train_df, valid_df = train_test_split(train_valid_df, test_size=0.1, stratify=train_valid_df["label"], random_state=SEED)

    train_valid_df['text'].str.replace('[{}]'.format(string.punctuation), '')
    test_df['text'].str.replace('[{}]'.format(string.punctuation), '')

    config = {
        "model_name":"cl-tohoku/bert-base-japanese-whole-word-masking",
        "max_length":-1,
        "train_epoch":3,
        "lr":3e-5,
    }
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"])

    train_X = [tokenizer(text, padding="max_length", max_length=config["max_length"], truncation=True) for text in tqdm(train_df["text"])]
    train_dataset = HateSpeechDataset(train_X, train_df["label"].tolist())
    valid_X = [tokenizer(text, padding="max_length", max_length=config["max_length"], truncation=True) for text in tqdm(valid_df["text"])]
    valid_dataset = HateSpeechDataset(valid_X, valid_df["label"].tolist())
    test_X = [tokenizer(text, padding="max_length", max_length=config["max_length"], truncation=True) for text in tqdm(test_df["text"])]
    test_dataset = HateSpeechDataset(test_X)

    trainer_args = TrainingArguments(
        seed=SEED,
        output_dir=".",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_steps=1e6,
        log_level="critical",
        num_train_epochs=config["train_epoch"],
        learning_rate=config["lr"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=12,
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_preds = trainer.predict(test_dataset)
    sub_df["label"] = np.argmax(test_preds.predictions, axis=1)
    sub_df.to_csv(DATA_DIR/"output/sub.csv", index=False)


if __name__ == "__main__":
    seed_everything(SEED)
    main()