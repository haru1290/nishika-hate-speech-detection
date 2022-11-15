import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from load_data import *


class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma):
        super(Focal_MultiLabel_Loss, self).__init__()
        self.gamma = gamma
        self.bceloss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        bce = self.bceloss(outputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = (1-bce_exp)**self.gamma * bce
        return focal_loss.mean()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([4950/4950, 4950/306]).cuda())
        loss_fct = Focal_MultiLabel_Loss(gamma=2)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(p.label_ids, preds)
    return {
        "f1_score": f1,
    }


def train(args, X_train, y_train, X_test, sub_df):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    X_train = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_train]
    X_test = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_test]
    test_dataset = HateSpeechDataset(X_test)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for kfold_idx, (tra_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tra = [X_train[i] for i in tra_idx]
        X_val = [X_train[i] for i in val_idx]
        y_tra = [y_train[i] for i in tra_idx]
        y_val = [y_train[i] for i in val_idx]
        
        training_dataset = HateSpeechDataset(X_tra, y_tra)
        validation_dataset = HateSpeechDataset(X_val, y_val)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        
        training_args = TrainingArguments(
            output_dir=f"./models/kfold_{str(kfold_idx)}/",
            overwrite_output_dir=args.overwrite_output_dir,
            evaluation_strategy=args.evaluation_strategy,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            log_level=args.log_level,
            logging_strategy=args.logging_strategy,
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            data_seed=args.seed,
            fp16=args.fp16,
            remove_unused_columns=args.remove_unused_columns,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=args.metric_for_best_model,
            label_smoothing_factor=args.label_smoothing_factor,
            report_to=args.report_to,
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
            )],
        )
        trainer.train()

        test_preds = trainer.predict(test_dataset)
        sub_df["label"] = np.argmax(test_preds.predictions, axis=1)
        sub_df.to_csv(f"./data/submission/sub_{str(kfold_idx)}.csv", index=False)


def main(args):
    seed_everything(args.seed)

    train_df = pd.read_csv("./data/input/train.csv")
    test_df = pd.read_csv("./data/input/test.csv")
    sub_df = pd.read_csv("./data/input/sample_submission.csv")

    train(
        args, train_df["text"].tolist(), train_df["label"].tolist(), test_df["text"].tolist(), sub_df
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--log_level", type=str, default="critical")
    parser.add_argument("--logging_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--remove_unused_columns", type=bool, default=False)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="f1_score")
    parser.add_argument("--label_smoothing_factor", type=float, default=0.2)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--kfold", type=int, default=5)

    parser.add_argument("--model_name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--max_length", type=float, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    main(args)