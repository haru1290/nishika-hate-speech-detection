import argparse
import os
import random
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


# class CustomLoss(nn.Module):
#     def __init__(self, alpha=None):
#         super(CustomLoss, self).__init__()
#         self.alpha = alpha
#         self.ce_loss = nn.CrossEntropyLoss(reduction="none")
#         self.mse_loss = nn.MSELoss(reduction="none")
    
#     def forward(self, outputs, targets):
#         h_label = targets.T[0].to(torch.int64)
#         s_label = targets.T[1].float()
#         s_label = torch.stack([s_label,1 - s_label], dim=1)
#         s_loss = self.mse_loss(outputs,s_label).mean()
#         s_loss.mul_(self.alpha)
#         s_loss.div_(2)
#         h_loss = self.ce_loss(outputs, h_label).mean()
#         loss = (s_loss + h_loss).float()
#         loss /= 2*(1+self.alpha)**2
#         return loss


# ここでソフトラベルを受け取る
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        # outputs = model(**inputs)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.get("logits")
        # loss_fct = CustomLoss(alpha=1.0)
        loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    # f1 = f1_score(p.label_ids.T[0], preds)
    f1 = f1_score(p.label_ids, preds)
    return {
        "f1_score": f1,
    }


def train(train_df):
    tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-japanese-large")
    tokenizer_function = lambda dataset: tokenizer(
        text=dataset["text"],
        padding="max_length",
        max_length=256,
        truncation=True,
    )

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold_index, (train_index, valid_index) in enumerate(skf.split(train_df["text"].values, train_df["label"].values)):
        
        train_data = train_df.iloc[train_index]
        valid_data = train_df.iloc[valid_index]
        train_dataset = Dataset.from_pandas(train_data).map(tokenizer_function, batched=True)
        valid_dataset = Dataset.from_pandas(valid_data).map(tokenizer_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained("studio-ousia/luke-japanese-large")

        training_args = TrainingArguments(
            output_dir=f"./models/kfold_{fold_index}/",
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            num_train_epochs=100,
            save_strategy="epoch",
            save_steps=1e6,
            save_total_limit=1,
            seed=42,
            data_seed=42,
            fp16=True,
            label_names=["labels", "soft_label"],
            load_best_model_at_end=True,
            metric_for_best_model="f1_score",
            label_smoothing_factor=0.2,
            report_to="none",
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ],
        )
        trainer.train()
        trainer.save_state()
        trainer.save_model()


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


# soft_labelをtrain_dfと結合
# make_dataset内で[s_label, h_label]に直してreturn
@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    seed_everything(args.seed)

    # training data
    train_df = pd.read_csv(cfg.path.train)

    # soft label data
    soft_label = np.load(cfg.path.soft_label)
    train_df["soft_label"] = soft_label[:, 0]
    # print(soft_label[:, 0])

    # print(train_df)
    # return

    train(train_df)


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

    parser.add_argument("--model_name", type=str, default="studio-ousia/luke-japanese-large")
    parser.add_argument("--run_name", type=str, default="luke-large-japanese")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    main()
