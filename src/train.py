import os
import pandas as pd
import torch
import torch.nn as nn
import argparse

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from tqdm import tqdm
from load_data import *


def seed_everything(seed: int):    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

        training_args = TrainingArguments(
            output_dir=f"./data/models/kfold_{str(index)}/bert-base-japanese",
            overwrite_output_dir=args.overwrite_output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            evaluation_strategy=args.evaluation_strategy,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            log_level=args.log_level,
            logging_strategy=args.logging_strategy,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            fp16=args.fp16,
            remove_unused_columns=args.remove_unused_columns,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=args.metric_for_best_model,
            report_to=args.report_to,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
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
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--log_level", type=str, default="critical")
    parser.add_argument("--logging_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--remove_unused_columns", type=bool, default=False)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="f1_score")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--model_name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--max_length", type=float, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)