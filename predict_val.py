"""
predict_val.py — Generate predictions for the VALIDATION set
so you can evaluate your model with evaluate.py.

Usage (from your project folder):
    python predict_val.py

Output:
    val_predictions.json   ← pass this to evaluate.py

Then run:
    python evaluate.py val_predictions.json DeepX_validation.xlsx
"""

import os, json, ast, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Copy the same CFG from train.py ──────────────────────────────
CFG = {
    "model_name":   "aubmindlab/bert-base-arabertv02",
    "val_path":     "DeepX_validation.xlsx",
    "output_dir":   "checkpoints",
    "max_len":      128,
    "batch_size":   32,    # can go higher here (inference only, less VRAM)
    "dropout":      0.2,
    "default_threshold": 0.35,
}

ASPECTS    = ["food", "service", "price", "cleanliness",
              "delivery", "ambiance", "app_experience", "general"]
SENTIMENTS = ["positive", "negative", "neutral"]
JOINT_LABELS = [f"{a}_{s}" for a in ASPECTS for s in SENTIMENTS] + ["none_neutral"]
NUM_LABELS   = len(JOINT_LABELS)   # 25
LABEL2IDX    = {l: i for i, l in enumerate(JOINT_LABELS)}
IDX2LABEL    = {i: l for l, i in LABEL2IDX.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥  Device: {DEVICE}")


# ── Same preprocessing as train.py ───────────────────────────────
def clean_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"[ىي]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF\U0001F300-\U0001FAFF!?.,،]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_input_text(row) -> str:
    text = clean_arabic(str(row.get("review_text", "")))
    stars = int(row.get("star_rating", 3))
    cat   = str(row.get("business_category", ""))
    return f"[CAT:{cat}] [STARS:{stars}] {text}"


def decode_predictions(probs: np.ndarray, threshold: float) -> dict:
    aspects, sents = [], {}
    for i, p in enumerate(probs):
        if p >= threshold:
            label = IDX2LABEL[i]
            if label == "none_neutral":
                continue
            asp, sent = label.rsplit("_", 1)
            if asp not in aspects:
                aspects.append(asp)
            sents[asp] = sent
    if not aspects:
        aspects = ["general"]
        sents   = {"general": "neutral"}
    return {"aspects": aspects, "aspect_sentiments": sents}


# ── Dataset (same as train.py) ────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, labeled=True):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.labeled   = labeled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        input_text = build_input_text(row)
        enc = self.tokenizer(
            input_text,
            max_length=CFG["max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        return item


# ── Same model architecture as train.py ──────────────────────────
class ABSAModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.2):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden_size   = self.encoder.config.hidden_size
        self.dropout  = nn.Dropout(dropout)
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
        return self.classifier(cls)


# ── Main ──────────────────────────────────────────────────────────
def main():
    # 1. Load validation data
    print("📂 Loading validation data...")
    df_val = pd.read_excel(CFG["val_path"])
    print(f"   Validation rows: {len(df_val)}")

    # 2. Load tokenizer
    print(f"🤖 Loading tokenizer: {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # 3. Load model checkpoint
    # Priority: final_model.pt > best_model.pt
    ckpt_dir  = Path(CFG["output_dir"])
    ckpt_path = ckpt_dir / "final_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        print("❌  No checkpoint found!")
        print(f"   Looked in: {ckpt_dir.resolve()}")
        print("   Run python train.py first.")
        return

    print(f"📦 Loading model from: {ckpt_path}")
    model = ABSAModel(CFG["model_name"], NUM_LABELS, CFG["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # 4. Load tuned threshold
    thr_path  = ckpt_dir / "threshold.json"
    threshold = CFG["default_threshold"]
    if thr_path.exists():
        with open(thr_path) as f:
            threshold = json.load(f)["threshold"]
        print(f"✅  Loaded tuned threshold: {threshold:.3f}")
    else:
        print(f"⚠   threshold.json not found, using default: {threshold}")

    # 5. Run inference on validation set
    dataset = ABSADataset(df_val, tokenizer, labeled=False)
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"],
                         shuffle=False, num_workers=0)

    all_probs  = []
    review_ids = df_val["review_id"].tolist()

    print("🔮 Running inference on validation set...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting val"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)

    # 6. Decode and build JSON
    predictions = []
    for i, rid in enumerate(review_ids):
        pred = decode_predictions(all_probs[i], threshold)
        predictions.append({
            "review_id":         int(rid),
            "aspects":           pred["aspects"],
            "aspect_sentiments": pred["aspect_sentiments"],
        })

    # 7. Save
    out_path = "val_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(predictions)} validation predictions → {out_path}")
    print("\nSample:")
    print(json.dumps(predictions[:2], ensure_ascii=False, indent=2))
    print("\n" + "="*50)
    print("Now run:")
    print(f"  python evaluate.py {out_path} {CFG['val_path']}")
    print("="*50)


if __name__ == "__main__":
    main()