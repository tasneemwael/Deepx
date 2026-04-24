"""
=============================================================
  Arabic ABSA — Joint Aspect + Sentiment Classification
  Backbone: AraBERT v2  |  Labels: 25 joint (aspect, sentiment)
  Author: DeepX Team
=============================================================
"""

import os, json, ast, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG — edit paths before running
# ─────────────────────────────────────────────
CFG = {
    # Model — try in order; first one that downloads wins
    "model_name": "aubmindlab/bert-base-arabertv02",
    # Fallback (uncomment if above is slow):
    # "model_name": "CAMeL-Lab/bert-base-arabic-camelbert-mix",

    # Paths
    "train_path":      "DeepX_train.xlsx",
    "val_path":        "DeepX_validation.xlsx",
    "unlabeled_path":  "DeepX_unlabeled.xlsx",
    "output_dir":      "checkpoints",
    "submission_path": "submission.json",

    # Training
    "max_len":        128,
    "batch_size":     16,      # lower to 8 if OOM
    "epochs":         5,
    "lr":             2e-5,
    "warmup_ratio":   0.1,
    "dropout":        0.2,
    "weight_decay":   0.01,

    # Label threshold (tuned automatically on val set)
    "default_threshold": 0.35,

    "seed": 42,
}

# ─────────────────────────────────────────────
#  LABEL SCHEMA
# ─────────────────────────────────────────────
ASPECTS    = ["food", "service", "price", "cleanliness",
              "delivery", "ambiance", "app_experience", "general"]
SENTIMENTS = ["positive", "negative", "neutral"]

# 24 real (aspect, sentiment) pairs + 1 "none_neutral"
JOINT_LABELS = [f"{a}_{s}" for a in ASPECTS for s in SENTIMENTS] + ["none_neutral"]
NUM_LABELS   = len(JOINT_LABELS)          # 25
LABEL2IDX    = {l: i for i, l in enumerate(JOINT_LABELS)}
IDX2LABEL    = {i: l for l, i in LABEL2IDX.items()}


# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_arabic(text: str) -> str:
    """
    Light cleaning that preserves Arabic meaning:
    - Remove URLs, HTML, excessive punctuation
    - Normalize Arabic chars (alef variants → ا, teh marbuta → ه, etc.)
    - Keep emojis — they carry strong sentiment signal
    - Keep English words (appear in app reviews)
    """
    if not isinstance(text, str):
        return ""
    # URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize Arabic letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى",     "ي", text)
    text = re.sub("ة",     "ه", text)
    text = re.sub("گ",     "ك", text)
    # Remove diacritics (tashkeel)
    text = re.sub("[\u064b-\u065f\u0670]", "", text)
    # Remove tatweel
    text = re.sub("\u0640", "", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_text(row: pd.Series) -> str:
    """
    Enrich the review text with structured metadata.
    star_rating is a POWERFUL signal for sentiment.
    """
    star  = int(row.get("star_rating", 3))
    cat   = str(row.get("business_category", "")).strip()
    text  = clean_arabic(str(row["review_text"]))
    return f"[نجوم:{star}] [فئة:{cat}] {text}"


# ─────────────────────────────────────────────
#  LABEL ENCODING / DECODING
# ─────────────────────────────────────────────
def encode_labels(aspects_list: list, sentiment_dict: dict) -> np.ndarray:
    """Convert (aspects, sentiments) → binary vector of length 25."""
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    if "none" in aspects_list:
        vec[LABEL2IDX["none_neutral"]] = 1.0
        return vec
    for asp in aspects_list:
        sent = sentiment_dict.get(asp)
        if sent is None:
            continue
        key = f"{asp}_{sent}"
        if key in LABEL2IDX:
            vec[LABEL2IDX[key]] = 1.0
    return vec


def decode_predictions(probs: np.ndarray, threshold: float = 0.35) -> dict:
    """
    Convert probability vector → {"aspects": [...], "aspect_sentiments": {...}}
    Handles the 'none' special case automatically.
    """
    active = np.where(probs >= threshold)[0]

    # If nothing passes threshold → take the single highest-confidence label
    if len(active) == 0:
        active = [int(np.argmax(probs))]

    aspects_set = {}
    for idx in active:
        label = IDX2LABEL[idx]           # e.g. "food_positive" or "none_neutral"
        if label == "none_neutral":
            return {"aspects": ["none"], "aspect_sentiments": {"none": "neutral"}}
        asp, sent = label.rsplit("_", 1)
        # Keep the highest-confidence sentiment per aspect
        if asp not in aspects_set or probs[idx] > aspects_set[asp][1]:
            aspects_set[asp] = (sent, probs[idx])

    if not aspects_set:
        return {"aspects": ["none"], "aspect_sentiments": {"none": "neutral"}}

    aspects      = sorted(aspects_set.keys())
    aspect_sents = {a: aspects_set[a][0] for a in aspects}
    return {"aspects": aspects, "aspect_sentiments": aspect_sents}


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, labeled: bool = True):
        self.tokenizer = tokenizer
        self.labeled   = labeled
        self.texts     = [build_input_text(row) for _, row in df.iterrows()]
        self.ids       = df["review_id"].tolist()

        if labeled:
            self.labels = []
            for _, row in df.iterrows():
                aspects  = ast.literal_eval(row["aspects"])
                sentiments = ast.literal_eval(row["aspect_sentiments"])
                self.labels.append(encode_labels(aspects, sentiments))
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=CFG["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "review_id":      self.ids[idx],
        }
        if self.labeled:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class ABSAModel(nn.Module):
    """
    AraBERT backbone + multi-label classification head.
    Uses [CLS] token representation.
    """
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]   # [CLS] token
        logits = self.classifier(cls)
        return logits


# ─────────────────────────────────────────────
#  FOCAL LOSS  (handles class imbalance)
# ─────────────────────────────────────────────
class FocalBCELoss(nn.Module):
    """
    Focal loss for multi-label: down-weights easy negatives,
    focuses on hard/rare positives (especially 'neutral').
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce   = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, loader, threshold=0.35):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs  = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    preds  = (probs >= threshold).astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return micro_f1, macro_f1, probs, labels


def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find the best global threshold on the validation set."""
    best_f1, best_t = 0, 0.35
    for t in np.arange(0.20, 0.65, 0.025):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"   Best threshold: {best_t:.3f}  →  val micro-F1: {best_f1:.4f}")
    return best_t


# ─────────────────────────────────────────────
#  INFERENCE → JSON
# ─────────────────────────────────────────────
def predict_to_json(model, df_unlabeled: pd.DataFrame,
                    tokenizer, threshold: float) -> list:
    dataset    = ABSADataset(df_unlabeled, tokenizer, labeled=False)
    loader     = DataLoader(dataset, batch_size=CFG["batch_size"],
                            shuffle=False, num_workers=0)
    model.eval()
    results    = []
    review_ids = df_unlabeled["review_id"].tolist()
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)

    for i, rid in enumerate(review_ids):
        pred = decode_predictions(all_probs[i], threshold)
        results.append({
            "review_id":         int(rid),
            "aspects":           pred["aspects"],
            "aspect_sentiments": pred["aspect_sentiments"],
        })
    return results


# ─────────────────────────────────────────────
#  COMPUTE CLASS WEIGHTS (for focal loss)
# ─────────────────────────────────────────────
def compute_pos_weight(df_train: pd.DataFrame) -> torch.Tensor:
    all_vecs = []
    for _, row in df_train.iterrows():
        aspects   = ast.literal_eval(row["aspects"])
        sentiments = ast.literal_eval(row["aspect_sentiments"])
        all_vecs.append(encode_labels(aspects, sentiments))
    matrix = np.vstack(all_vecs)
    # pos_weight = (N - n_pos) / n_pos  (clamped between 1 and 20)
    n_pos = matrix.sum(axis=0) + 1e-6
    w     = (len(matrix) - n_pos) / n_pos
    w     = np.clip(w, 1.0, 20.0)
    return torch.tensor(w, dtype=torch.float).to(DEVICE)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(CFG["output_dir"], exist_ok=True)

    # ── Load data
    print("📂 Loading data...")
    df_train    = pd.read_excel(CFG["train_path"])
    df_val      = pd.read_excel(CFG["val_path"])
    df_unlabeled = pd.read_excel(CFG["unlabeled_path"])

    # Combine train + val for final training (after tuning)
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    print(f"   Train: {len(df_train)} | Val: {len(df_val)} | Unlabeled: {len(df_unlabeled)}")

    # ── Tokenizer
    print(f"🤖 Loading tokenizer: {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # ── Datasets
    train_ds = ABSADataset(df_train, tokenizer, labeled=True)
    val_ds   = ABSADataset(df_val,   tokenizer, labeled=True)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"]*2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model
    print("🏗  Building model...")
    model = ABSAModel(CFG["model_name"], NUM_LABELS, CFG["dropout"]).to(DEVICE)

    # ── Loss
    pos_weight = compute_pos_weight(df_train)
    loss_fn    = FocalBCELoss(gamma=2.0, pos_weight=pos_weight)

    # ── Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    params   = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": CFG["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],     "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=CFG["lr"])

    total_steps   = len(train_loader) * CFG["epochs"]
    warmup_steps  = int(total_steps * CFG["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop
    print(f"\n🚀 Training for {CFG['epochs']} epochs...")
    best_f1   = 0
    best_path = Path(CFG["output_dir"]) / "best_model.pt"

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        micro_f1, macro_f1, val_probs, val_labels = evaluate(
            model, val_loader, CFG["default_threshold"])

        print(f"  Epoch {epoch}/{CFG['epochs']}  "
              f"loss={train_loss:.4f}  "
              f"val_micro_F1={micro_f1:.4f}  "
              f"val_macro_F1={macro_f1:.4f}")

        if micro_f1 > best_f1:
            best_f1 = micro_f1
            torch.save(model.state_dict(), best_path)
            print(f"   ✅ Saved best model (micro-F1={best_f1:.4f})")

    # ── Load best & tune threshold
    print(f"\n🎯 Tuning decision threshold on validation set...")
    model.load_state_dict(torch.load(best_path))
    _, _, val_probs, val_labels = evaluate(model, val_loader, CFG["default_threshold"])
    best_threshold = tune_threshold(val_probs, val_labels)

    # ── Save threshold
    with open(Path(CFG["output_dir"]) / "threshold.json", "w") as f:
        json.dump({"threshold": best_threshold}, f)

    # ── (Optional) Retrain on ALL labeled data
    print("\n🔄 Fine-tuning on train+val combined for final model...")
    all_ds     = ABSADataset(df_all, tokenizer, labeled=True)
    all_loader = DataLoader(all_ds, batch_size=CFG["batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)

    # 2 extra epochs on full data at lower LR
    for pg in optimizer.param_groups:
        pg["lr"] = CFG["lr"] * 0.5

    total_steps2  = len(all_loader) * 2
    warmup_steps2 = int(total_steps2 * 0.1)
    scheduler2    = get_linear_schedule_with_warmup(optimizer, warmup_steps2, total_steps2)
    pos_weight2   = compute_pos_weight(df_all)
    loss_fn2      = FocalBCELoss(gamma=2.0, pos_weight=pos_weight2)

    for epoch in range(1, 3):
        loss = train_epoch(model, all_loader, optimizer, scheduler2, loss_fn2)
        print(f"   Final epoch {epoch}/2  loss={loss:.4f}")

    final_path = Path(CFG["output_dir"]) / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"   💾 Final model saved → {final_path}")

    # ── Predict unlabeled → JSON
    print("\n🔮 Generating predictions for unlabeled set...")
    predictions = predict_to_json(model, df_unlabeled, tokenizer, best_threshold)

    with open(CFG["submission_path"], "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Submission saved → {CFG['submission_path']}")
    print(f"   Total predictions: {len(predictions)}")

    # Quick sanity check
    sample = predictions[:3]
    print("\nSample predictions:")
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
