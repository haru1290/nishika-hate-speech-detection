import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import hydra

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from ..data.rmake_dataset import *


class CustomLoss(nn.Module):
    def __init__(self, alpha=None):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
    
    def forward(self, outputs, targets):
        h_label = targets.T[0].to(torch.int64)
        s_label = targets.T[1].float()
        s_label = torch.stack([s_label,1 - s_label], dim=1)
        s_loss = self.mse_loss(outputs,s_label).mean()
        s_loss.mul_(self.alpha)
        s_loss.div_(2)
        h_loss = self.ce_loss(outputs, h_label).mean()
        loss = (s_loss + h_loss).float()
        loss /= 2*(1+self.alpha)**2
        return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CustomLoss(alpha=1.0)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
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
    f1 = f1_score(p.label_ids.T[0], preds)
    return {
        "f1_score": f1,
    }


# X_tra_val = [tokenizer(text, padding="max_length", max_length=args.max_length, truncation=True) for text in X_train_valid]
def train(train_df, soft_lable, cfg):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    for fold_index, (train_index, valid_index) in enumerate(skf.split(train_df["text"].values, train_df["label"].values)):

        train_data = train_df.iloc[train_index]
        valid_data = train_df.iloc[valid_index]
        train_dataset = HateSpeechDataset(train_data, tokenizer)
        valid_dataset = HateSpeechDataset(valid_data, tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

        training_args = TrainingArguments(
            output_dir=f"./models/{args.run_name}/kfold_{fold_index}/",
            evaluation_strategy="epoch",
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            num_train_epochs=cfg.training.epoch,
            save_strategy="epoch",
            save_steps=args.steps,
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
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
            ],
        )

        trainer.train()
        trainer.save_state()
        trainer.save_model()


# soft_labelをtrain_dfと結合
# make_dataset内で[s_label, h_label]に直してreturn
@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    seed_everything(args.seed)

    # training data
    train_df = pd.read_csv(cfg.path.train)

    # soft label data
    soft_label = np.load(cfg.path.soft_label)

    train(
        train_df,
        soft_label[:, 0],
        cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--log_level", type=str, default="critical")
    parser.add_argument("--logging_strategy", type=str, default="epoch")
    parser.add_argument("--steps", type=int, default=25)
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

    parser.add_argument("--model_name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--run_name", type=str, default="bert-base-japanese")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    main(args)
