import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME     = "distilbert-base-uncased"
MAX_LENGTH     = 512
BATCH_TRAIN    = 16
BATCH_EVAL     = 32
EPOCHS         = 3
WEIGHTS_STAGE2 = "./weights_isot"
WEIGHTS_STAGE3 = "./weights_final"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

def prepare_dataset(df):
    df = df[["text", "label"]].dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds.train_test_split(test_size=0.1, seed=42)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

# Load PolitiFact
pf_raw = []
with open("politifact_factcheck_data.json", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                pf_raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue

pf_records = []
for item in pf_raw:
    statement = item.get("statement", "") or ""
    verdict   = item.get("verdict", "")   or ""
    label = 1 if verdict.lower() in ["true", "mostly-true", "half-true"] else 0
    if statement.strip():
        pf_records.append({"text": statement, "label": label})

pf_df = pd.DataFrame(pf_records)
print(f"PolitiFact samples: {len(pf_df)} (real={pf_df['label'].sum()}, fake={(pf_df['label']==0).sum()})")

pf_splits = prepare_dataset(pf_df)

model = DistilBertForSequenceClassification.from_pretrained(WEIGHTS_STAGE2, num_labels=2)

training_args = TrainingArguments(
    output_dir="./stage3_checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pf_splits["train"],
    eval_dataset=pf_splits["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()
print(f"\nStage 3 results — Accuracy: {results['eval_accuracy']:.4f}  F1: {results['eval_f1']:.4f}")

model.save_pretrained(WEIGHTS_STAGE3)
tokenizer.save_pretrained(WEIGHTS_STAGE3)
print(f"Final weights saved → {WEIGHTS_STAGE3}")