import os
import random
import numpy as np
import argparse
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, EvalPrediction, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback,
)
from tqdm import tqdm
from load_data import *


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(p.label_ids, preds)
    return {
        "f1_score": f1,
    }


def train(args):
    seed_everything(args.seed)

    train_valid_df = load_data("./data/train.csv")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for index, (train_index, valid_index) in enumerate(skf.split(train_valid_df["text"], train_valid_df["label"])):
        X_train = train_valid_df["text"].iloc[train_index]
        X_valid = train_valid_df["text"].iloc[valid_index]
        y_train = train_valid_df["label"].iloc[train_index]
        y_valid = train_valid_df["label"].iloc[valid_index]

        X_train = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in tqdm(X_train)]
        X_valid = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in tqdm(X_valid)]

        train_dataset = HateSpeechDataset(X_train, y_train.tolist())
        valid_dataset = HateSpeechDataset(X_valid, y_valid.tolist())

        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        model.to(device)

        trainer_args = TrainingArguments(
            output_dir=f"./data/models/kfold_{str(index)}_{args.model_name}",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            log_level="critical",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            fp16=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="f1_score",
            greater_is_better=True,
            report_to="none",
            seed=args.seed,
        )
        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=args.patience,
            )],
        )

        trainer.train()


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=float, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--save_steps", type=int, default=1e6)
    parser.add_argument("--save_total_limit", type=int, default=1)
    args = parser.parse_args()
    
    main(args)