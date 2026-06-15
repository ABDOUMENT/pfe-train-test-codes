"""
DistilBERT Sequential Fine-tuning Pipeline
Datasets: WELFake → ISOT → PolitiFact
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ══════════════════════════════════════════════════
# GPU CHECK
# ══════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"  Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*55}\n")

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════

MODEL_NAME      = "distilbert-base-uncased"
MAX_LENGTH      = 512
BATCH_TRAIN     = 16
BATCH_EVAL      = 32
EPOCHS          = 3

WEIGHTS_STAGE1  = "./weights_welfake"
WEIGHTS_STAGE2  = "./weights_isot"
WEIGHTS_STAGE3  = "./weights_final"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

def prepare_dataset(df):
    """Convert a DataFrame with 'text' and 'label' columns into a tokenized HF Dataset."""
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
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

def get_training_args(output_dir, learning_rate):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=learning_rate,
        warmup_steps=500,  
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        #logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),   # mixed precision on GPU
        dataloader_num_workers=2,
        report_to="none",
    )

def run_stage(stage_num, dataset_name, ds_splits, base_weights, save_path, lr):
    print(f"\n{'='*55}")
    print(f"  STAGE {stage_num} — {dataset_name}")
    print(f"  Loading weights from : {base_weights}")
    print(f"  Learning rate        : {lr}")
    print(f"  Train samples        : {len(ds_splits['train'])}")
    print(f"  Val   samples        : {len(ds_splits['test'])}")
    print(f"{'='*55}\n")

    model = DistilBertForSequenceClassification.from_pretrained(
        base_weights, num_labels=2
    )

    trainer = Trainer(
        model=model,
        args=get_training_args(f"./stage{stage_num}_checkpoints", lr),
        train_dataset=ds_splits["train"],
        eval_dataset=ds_splits["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    print(f"\n  ✅ Stage {stage_num} results:")
    print(f"     Accuracy : {results['eval_accuracy']:.4f}")
    print(f"     F1       : {results['eval_f1']:.4f}")

    # Save
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"     Weights saved → {save_path}\n")

    return model

# ══════════════════════════════════════════════════
# STAGE 1 — WELFake  (train from pretrained)
# ══════════════════════════════════════════════════

print("\n>>> Loading WELFake dataset...")
welfake_df = pd.read_csv("WELFake_Dataset.csv")

# WELFake columns: Unnamed:0, title, text, label  (0=fake, 1=real)
if "text" not in welfake_df.columns:
    welfake_df["text"] = welfake_df["title"].fillna("") + " " + welfake_df["text"].fillna("")

welfake_splits = prepare_dataset(welfake_df)

model_stage1 = run_stage(
    stage_num     = 1,
    dataset_name  = "WELFake",
    ds_splits     = welfake_splits,
    base_weights  = MODEL_NAME,          # start from HuggingFace pretrained
    save_path     = WEIGHTS_STAGE1,
    lr            = 5e-5,
)

# ══════════════════════════════════════════════════
# STAGE 2 — ISOT  (fine-tune from WELFake weights)
# ══════════════════════════════════════════════════

print("\n>>> Loading ISOT dataset...")
true_df = pd.read_csv("True.csv").assign(label=1)
fake_df = pd.read_csv("Fake.csv").assign(label=0)
isot_df = pd.concat([true_df, fake_df], ignore_index=True)

# ISOT columns: title, text, subject, date
isot_df["text"] = isot_df["title"].fillna("") + " " + isot_df["text"].fillna("")

isot_splits = prepare_dataset(isot_df)

model_stage2 = run_stage(
    stage_num     = 2,
    dataset_name  = "ISOT",
    ds_splits     = isot_splits,
    base_weights  = WEIGHTS_STAGE1,      # continue from stage 1
    save_path     = WEIGHTS_STAGE2,
    lr            = 2e-5,
)

# ══════════════════════════════════════════════════
# STAGE 3 — PolitiFact  (fine-tune from ISOT weights)
# ══════════════════════════════════════════════════

print("\n>>> Loading PolitiFact dataset...")
with open("politifact_factcheck_data.json", "r") as f:
    pf_raw = json.load(f)

pf_records = []
for item in pf_raw:
    statement = item.get("statement", "") or ""
    verdict   = item.get("verdict", "")   or ""
    # Map multi-class verdicts → binary (0=fake, 1=real)
    if verdict.lower() in ["true", "mostly-true", "half-true"]:
        label = 1
    else:
        label = 0
    if statement.strip():
        pf_records.append({"text": statement, "label": label})

pf_df = pd.DataFrame(pf_records)
print(f"  PolitiFact samples : {len(pf_df)}  "
      f"(real={pf_df['label'].sum()}, fake={(pf_df['label']==0).sum()})")

pf_splits = prepare_dataset(pf_df)

model_stage3 = run_stage(
    stage_num     = 3,
    dataset_name  = "PolitiFact",
    ds_splits     = pf_splits,
    base_weights  = WEIGHTS_STAGE2,      # continue from stage 2
    save_path     = WEIGHTS_STAGE3,
    lr            = 1e-5,
)

# ══════════════════════════════════════════════════
# MANUAL TEST
# ══════════════════════════════════════════════════

print(f"\n{'='*55}")
print("  MANUAL TEST — Final Model")
print(f"{'='*55}\n")

# Load final model
final_model = DistilBertForSequenceClassification.from_pretrained(WEIGHTS_STAGE3)
final_model.to(device)
final_model.eval()

test_samples = [
    {
        "text":  "Scientists confirm that the Earth is approximately 4.5 billion years old, "
                 "based on radiometric dating of meteorites and Earth rocks.",
        "truth": "REAL",
    },
    {
        "text":  "The government is secretly adding mind-control chemicals to the water supply "
                 "to keep citizens from questioning authority.",
        "truth": "FAKE",
    },
    {
        "text":  "NASA successfully launched the Artemis I rocket, sending an uncrewed Orion "
                 "capsule around the Moon in November 2022.",
        "truth": "REAL",
    },
    {
        "text":  "Doctors are being paid to suppress a natural cure for cancer that big pharma "
                 "doesn't want the public to know about.",
        "truth": "FAKE",
    },
    {
        "text":  "The United Nations reported that global hunger affects over 700 million "
                 "people worldwide, worsened by climate change and conflict.",
        "truth": "REAL",
    },
    {
        "text":  "A leaked document proves that the moon landing was filmed in a Hollywood "
                 "studio and never actually happened.",
        "truth": "FAKE",
    },
]

label_map = {0: "FAKE", 1: "REAL"}
correct = 0

print(f"  {'#':<3} {'Predicted':<10} {'Actual':<10} {'Confidence':<12} {'Match'}")
print(f"  {'-'*55}")

for i, sample in enumerate(test_samples, 1):
    inputs = tokenizer(
        sample["text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = final_model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id    = torch.argmax(probs).item()
    confidence = probs[pred_id].item() * 100
    predicted  = label_map[pred_id]
    match      = "✅" if predicted == sample["truth"] else "❌"

    if predicted == sample["truth"]:
        correct += 1

    print(f"  {i:<3} {predicted:<10} {sample['truth']:<10} {confidence:>6.1f}%      {match}")
    print(f"      \"{sample['text'][:80]}...\"")
    print()

accuracy = correct / len(test_samples) * 100
print(f"  {'='*55}")
print(f"  Manual test accuracy : {correct}/{len(test_samples)}  ({accuracy:.0f}%)")
print(f"  {'='*55}\n")

print("🎉 All done! Final weights are in:", WEIGHTS_STAGE3)