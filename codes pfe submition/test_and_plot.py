"""
Stage 4 — Testing & Visualization
Loads ./weights_final and runs evaluation + graphs
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════

WEIGHTS_STAGE3 = "./weights_welfake"
MAX_LENGTH      = 512
OUTPUT_DIR      = "./test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"  Device : {device}")
print(f"{'='*55}\n")

tokenizer = DistilBertTokenizer.from_pretrained(WEIGHTS_STAGE3)
model = DistilBertForSequenceClassification.from_pretrained(WEIGHTS_STAGE3)
model.to(device)
model.eval()

# Flip the label map
label_map = {0: "REAL", 1: "FAKE"}  # was {0: "FAKE", 1: "REAL"}

# ══════════════════════════════════════════════════
# TEST SAMPLES
# ══════════════════════════════════════════════════

test_samples = [
    {"text": "Scientists confirm that the Earth is approximately 4.5 billion years old, based on radiometric dating of meteorites and Earth rocks.", "truth": "REAL"},
    {"text": "The government is secretly adding mind-control chemicals to the water supply to keep citizens from questioning authority.", "truth": "FAKE"},
    {"text": "NASA successfully launched the Artemis I rocket, sending an uncrewed Orion capsule around the Moon in November 2022.", "truth": "REAL"},
    {"text": "Doctors are being paid to suppress a natural cure for cancer that big pharma doesn't want the public to know about.", "truth": "FAKE"},
    {"text": "The United Nations reported that global hunger affects over 700 million people worldwide, worsened by climate change and conflict.", "truth": "REAL"},
    {"text": "A leaked document proves that the moon landing was filmed in a Hollywood studio and never actually happened.", "truth": "FAKE"},
    {"text": "The World Health Organization declared COVID-19 a global pandemic in March 2020.", "truth": "REAL"},
    {"text": "5G towers are being used to spread coronavirus by activating it in people's bodies.", "truth": "FAKE"},
    {"text": "Research published in Nature confirmed that climate change is causing polar ice caps to melt at an accelerating rate.", "truth": "REAL"},
    {"text": "Bill Gates is using COVID vaccines to implant microchips for population tracking.", "truth": "FAKE"},
    {"text": "The European Space Agency confirmed the James Webb telescope captured its first full-color images of the universe in 2022.", "truth": "REAL"},
    {"text": "Eating a tablespoon of bleach daily boosts the immune system and kills all viruses.", "truth": "FAKE"},
]

# ══════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════

results = []
all_probs = []

print(f"  {'#':<3} {'Predicted':<10} {'Actual':<10} {'Confidence':<12} {'Match'}")
print(f"  {'-'*58}")

for i, sample in enumerate(test_samples, 1):
    inputs = tokenizer(
        sample["text"],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    pred_id    = int(np.argmax(probs))
    confidence = float(probs[pred_id]) * 100
    predicted  = label_map[pred_id]
    match      = "✅" if predicted == sample["truth"] else "❌"

    results.append({
        "text":       sample["text"][:60] + "...",
        "truth":      sample["truth"],
        "predicted":  predicted,
        "confidence": confidence,
        "correct":    predicted == sample["truth"],
        "prob_fake":  float(probs[0]),
        "prob_real":  float(probs[1]),
    })
    all_probs.append(probs)

    print(f"  {i:<3} {predicted:<10} {sample['truth']:<10} {confidence:>6.1f}%      {match}")
    print(f"      \"{sample['text'][:72]}\"")
    print()

correct = sum(r["correct"] for r in results)
accuracy = correct / len(results) * 100
print(f"  Manual test accuracy : {correct}/{len(results)}  ({accuracy:.0f}%)\n")

# ══════════════════════════════════════════════════
# TRAINING HISTORY (hardcoded from your runs)
# ══════════════════════════════════════════════════

history = {
    "stage1": {
        "name": "WELFake",
        "epochs": [1, 2, 3],
        "eval_loss":     [0.02391, 0.02486, 0.03773],
        "eval_accuracy": [0.9926,  0.9947,  0.9945],
        "eval_f1":       [0.9926,  0.9947,  0.9945],
    },
    "stage2": {
        "name": "ISOT",
        "epochs": [1, 2, 3],
        "eval_loss":     [0.0001043, 0.0001242, 0.000006056],
        "eval_accuracy": [1.0,       1.0,        1.0],
        "eval_f1":       [1.0,       1.0,        1.0],
    },
    "stage3": {
        "name": "PolitiFact",
        "epochs": [1, 2, 3],
        "eval_loss":     [0.5862, 0.5815, 0.5899],
        "eval_accuracy": [0.6749, 0.6848, 0.6829],
        "eval_f1":       [0.6747, 0.6853, 0.6834],
    },
}

# ══════════════════════════════════════════════════
# PLOT 1 — Training Dashboard
# ══════════════════════════════════════════════════

DARK  = "#0d0d0f"
CARD  = "#16161a"
ACC   = "#00e5ff"
F1C   = "#ff6b6b"
LOSSC = "#ffd166"
GRID  = "#2a2a32"
WHITE = "#e8e8f0"

colors = {
    "stage1": "#00e5ff",
    "stage2": "#7fff6e",
    "stage3": "#ff6b6b",
}

fig = plt.figure(figsize=(18, 11), facecolor=DARK)
fig.suptitle("DistilBERT Sequential Fine-tuning — Training Dashboard",
             color=WHITE, fontsize=17, fontweight="bold", y=0.97)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
              left=0.06, right=0.97, top=0.91, bottom=0.08)

# --- Accuracy per stage ---
ax_acc = fig.add_subplot(gs[0, 0])
ax_acc.set_facecolor(CARD)
for sid, sd in history.items():
    ax_acc.plot(sd["epochs"], [v*100 for v in sd["eval_accuracy"]],
                marker="o", linewidth=2.5, markersize=7,
                color=colors[sid], label=sd["name"])
ax_acc.set_title("Validation Accuracy", color=WHITE, fontsize=12, pad=8)
ax_acc.set_xlabel("Epoch", color=WHITE, fontsize=9)
ax_acc.set_ylabel("Accuracy (%)", color=WHITE, fontsize=9)
ax_acc.tick_params(colors=WHITE)
ax_acc.grid(color=GRID, linewidth=0.6)
ax_acc.legend(fontsize=8, facecolor=DARK, labelcolor=WHITE, edgecolor=GRID)
for spine in ax_acc.spines.values(): spine.set_edgecolor(GRID)

# --- F1 per stage ---
ax_f1 = fig.add_subplot(gs[0, 1])
ax_f1.set_facecolor(CARD)
for sid, sd in history.items():
    ax_f1.plot(sd["epochs"], [v*100 for v in sd["eval_f1"]],
               marker="s", linewidth=2.5, markersize=7,
               color=colors[sid], label=sd["name"])
ax_f1.set_title("Validation F1 Score", color=WHITE, fontsize=12, pad=8)
ax_f1.set_xlabel("Epoch", color=WHITE, fontsize=9)
ax_f1.set_ylabel("F1 (%)", color=WHITE, fontsize=9)
ax_f1.tick_params(colors=WHITE)
ax_f1.grid(color=GRID, linewidth=0.6)
ax_f1.legend(fontsize=8, facecolor=DARK, labelcolor=WHITE, edgecolor=GRID)
for spine in ax_f1.spines.values(): spine.set_edgecolor(GRID)

# --- Loss per stage ---
ax_loss = fig.add_subplot(gs[0, 2])
ax_loss.set_facecolor(CARD)
for sid, sd in history.items():
    ax_loss.plot(sd["epochs"], sd["eval_loss"],
                 marker="^", linewidth=2.5, markersize=7,
                 color=colors[sid], label=sd["name"])
ax_loss.set_title("Validation Loss", color=WHITE, fontsize=12, pad=8)
ax_loss.set_xlabel("Epoch", color=WHITE, fontsize=9)
ax_loss.set_ylabel("Loss", color=WHITE, fontsize=9)
ax_loss.set_yscale("log")
ax_loss.tick_params(colors=WHITE)
ax_loss.grid(color=GRID, linewidth=0.6, which="both")
ax_loss.legend(fontsize=8, facecolor=DARK, labelcolor=WHITE, edgecolor=GRID)
for spine in ax_loss.spines.values(): spine.set_edgecolor(GRID)

# --- Final accuracy bar chart ---
ax_bar = fig.add_subplot(gs[1, 0])
ax_bar.set_facecolor(CARD)
stage_names = ["WELFake\n(Stage 1)", "ISOT\n(Stage 2)", "PolitiFact\n(Stage 3)"]
final_accs  = [
    history["stage1"]["eval_accuracy"][-1] * 100,
    history["stage2"]["eval_accuracy"][-1] * 100,
    history["stage3"]["eval_accuracy"][-1] * 100,
]
bar_colors = [colors["stage1"], colors["stage2"], colors["stage3"]]
bars = ax_bar.bar(stage_names, final_accs, color=bar_colors, width=0.5, edgecolor=DARK, linewidth=1.5)
for bar, val in zip(bars, final_accs):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")
ax_bar.set_title("Final Accuracy per Stage", color=WHITE, fontsize=12, pad=8)
ax_bar.set_ylabel("Accuracy (%)", color=WHITE, fontsize=9)
ax_bar.set_ylim(0, 110)
ax_bar.tick_params(colors=WHITE)
ax_bar.grid(axis="y", color=GRID, linewidth=0.6)
for spine in ax_bar.spines.values(): spine.set_edgecolor(GRID)

# --- Manual test confidence chart ---
ax_conf = fig.add_subplot(gs[1, 1])
ax_conf.set_facecolor(CARD)
labels_short = [f"#{i+1}" for i in range(len(results))]
conf_vals    = [r["confidence"] for r in results]
bar_c        = [colors["stage2"] if r["correct"] else colors["stage3"] for r in results]
ax_conf.bar(labels_short, conf_vals, color=bar_c, edgecolor=DARK, linewidth=1)
ax_conf.axhline(y=50, color=LOSSC, linestyle="--", linewidth=1, alpha=0.7)
ax_conf.set_title("Manual Test — Confidence per Sample", color=WHITE, fontsize=12, pad=8)
ax_conf.set_xlabel("Sample #", color=WHITE, fontsize=9)
ax_conf.set_ylabel("Confidence (%)", color=WHITE, fontsize=9)
ax_conf.set_ylim(0, 110)
ax_conf.tick_params(colors=WHITE)
ax_conf.grid(axis="y", color=GRID, linewidth=0.6)
for spine in ax_conf.spines.values(): spine.set_edgecolor(GRID)
patch_correct = mpatches.Patch(color=colors["stage2"], label="Correct")
patch_wrong   = mpatches.Patch(color=colors["stage3"], label="Wrong")
ax_conf.legend(handles=[patch_correct, patch_wrong], fontsize=8,
               facecolor=DARK, labelcolor=WHITE, edgecolor=GRID)

# --- Confusion matrix ---
ax_cm = fig.add_subplot(gs[1, 2])
ax_cm.set_facecolor(CARD)
y_true = [0 if r["truth"] == "FAKE" else 1 for r in results]
y_pred = [0 if r["predicted"] == "FAKE" else 1 for r in results]
cm = confusion_matrix(y_true, y_pred)
im = ax_cm.imshow(cm, cmap="Blues", aspect="auto")
ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["FAKE", "REAL"], color=WHITE, fontsize=10)
ax_cm.set_yticklabels(["FAKE", "REAL"], color=WHITE, fontsize=10, rotation=90, va="center")
ax_cm.set_title("Confusion Matrix\n(Manual Test)", color=WHITE, fontsize=12, pad=8)
ax_cm.set_xlabel("Predicted", color=WHITE, fontsize=9)
ax_cm.set_ylabel("Actual", color=WHITE, fontsize=9)
for (row, col), val in np.ndenumerate(cm):
    ax_cm.text(col, row, str(val), ha="center", va="center",
               color=WHITE if val < cm.max()/2 else DARK, fontsize=16, fontweight="bold")
for spine in ax_cm.spines.values(): spine.set_edgecolor(GRID)

path1 = os.path.join(OUTPUT_DIR, "training_dashboard.png")
plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  ✅ Saved: {path1}")

# ══════════════════════════════════════════════════
# PLOT 2 — ROC Curve (manual test)
# ══════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(7, 6), facecolor=DARK)
ax.set_facecolor(CARD)

prob_real = np.array([r["prob_real"] for r in results])
fpr, tpr, _ = roc_curve(y_true, prob_real)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color=ACC, linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1.5, label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.15, color=ACC)
ax.set_title("ROC Curve — Final Model (Manual Test)", color=WHITE, fontsize=13, pad=10)
ax.set_xlabel("False Positive Rate", color=WHITE, fontsize=10)
ax.set_ylabel("True Positive Rate", color=WHITE, fontsize=10)
ax.tick_params(colors=WHITE)
ax.legend(fontsize=9, facecolor=DARK, labelcolor=WHITE, edgecolor=GRID)
for spine in ax.spines.values(): spine.set_edgecolor(GRID)
ax.grid(color=GRID, linewidth=0.6)
fig2.tight_layout()

path2 = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  ✅ Saved: {path2}")

# ══════════════════════════════════════════════════
# PLOT 3 — Probability distribution
# ══════════════════════════════════════════════════

fig3, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
fig3.suptitle("Prediction Probability Distribution", color=WHITE, fontsize=14, fontweight="bold")

fake_samples = [r for r in results if r["truth"] == "FAKE"]
real_samples = [r for r in results if r["truth"] == "REAL"]

for ax, group, title, col in zip(
    axes,
    [fake_samples, real_samples],
    ["Ground Truth: FAKE", "Ground Truth: REAL"],
    [F1C, ACC]
):
    ax.set_facecolor(CARD)
    indices  = range(len(group))
    p_fake   = [s["prob_fake"]*100 for s in group]
    p_real   = [s["prob_real"]*100 for s in group]
    x        = np.arange(len(group))
    width    = 0.35
    ax.bar(x - width/2, p_fake, width, label="P(FAKE)", color=F1C, alpha=0.85, edgecolor=DARK)
    ax.bar(x + width/2, p_real, width, label="P(REAL)", color=ACC, alpha=0.85, edgecolor=DARK)
    ax.axhline(50, color=LOSSC, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(title, color=WHITE, fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in range(len(group))], color=WHITE, fontsize=9)
    ax.set_ylabel("Probability (%)", color=WHITE, fontsize=9)
    ax.set_ylim(0, 110)
    ax.tick_params(colors=WHITE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(fontsize=9, facecolor=DARK, labelcolor=WHITE, edgecolor=GRID)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID)

fig3.tight_layout(rect=[0, 0, 1, 0.95])
path3 = os.path.join(OUTPUT_DIR, "probability_distribution.png")
plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  ✅ Saved: {path3}")

print(f"\n{'='*55}")
print(f"  All plots saved to: {OUTPUT_DIR}/")
print(f"  Manual test accuracy: {correct}/{len(results)} ({accuracy:.0f}%)")
print(f"{'='*55}\n")