import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW



# =========================
# Config
# =========================

DATA_PATH = "clinical_notes.csv"   # assumes CSV is in same folder
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

MAX_LEN = 512
BATCH_SIZE = 8          # keep small for 512 tokens
NUM_EPOCHS = 3          # you can change later if you want
LR = 2e-5
VAL_SPLIT_FALLBACK = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

CAT_COLS = ["gender", "race", "ethnicity", "language", "maritalstatus"]
CAT_EMBED_DIM = 8
TAB_HIDDEN_DIM = 32


# =========================
# Dataset
# =========================

class GlaucomaBERTDataset(Dataset):
    def __init__(self, df, tokenizer, cat2idx, age_mean, age_std, max_len):
        self.tokenizer = tokenizer
        self.cat2idx = cat2idx
        self.age_mean = age_mean
        self.age_std = age_std
        self.max_len = max_len

        self.input_ids = []
        self.attention_masks = []
        self.age_norm = []
        self.cat_indices = []
        self.labels = []
        self.races = []

        df = df.reset_index(drop=True)

        for _, row in df.iterrows():
            note = str(row.get("note", ""))
            summ = str(row.get("gpt4_summary", ""))
            if summ.lower() == "nan":
                summ = ""
            text = note + " " + summ

            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            )

            self.input_ids.append(enc["input_ids"])
            self.attention_masks.append(enc["attention_mask"])

            age_raw = float(row["age"])
            age_n = (age_raw - age_mean) / (age_std + 1e-8)
            self.age_norm.append(age_n)

            cat_vec = []
            for c in CAT_COLS:
                val = str(row[c])
                idx = self.cat2idx[c].get(val, 0)
                cat_vec.append(idx)
            self.cat_indices.append(cat_vec)

            self.labels.append(int(row["label"]))
            self.races.append(str(row["race"]).lower())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        age = torch.tensor(self.age_norm[idx], dtype=torch.float32)
        cat_idx = torch.tensor(self.cat_indices[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        race = self.races[idx]
        return input_ids, attention_mask, age, cat_idx, label, race


# =========================
# Model
# =========================

class ClinicalBERTWithTabular(nn.Module):
    def __init__(self, bert_model_name, cat_cardinalities, cat_embed_dim, tab_hidden_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # typically 768

        # categorical embeddings
        self.cat_embs = nn.ModuleList()
        for num_cat in cat_cardinalities:
            self.cat_embs.append(nn.Embedding(num_cat, cat_embed_dim))

        cat_total_dim = len(cat_cardinalities) * cat_embed_dim
        tab_in_dim = cat_total_dim + 1  # +1 for age

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_in_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = bert_hidden_size + tab_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, age_norm, cat_idx):
        # BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS]

        # tabular
        cat_feats = []
        for i, emb_layer in enumerate(self.cat_embs):
            cat_feats.append(emb_layer(cat_idx[:, i]))
        cat_feats = torch.cat(cat_feats, dim=-1)
        age_norm = age_norm.view(-1, 1)
        tab_input = torch.cat([cat_feats, age_norm], dim=-1)
        tab_repr = self.tab_mlp(tab_input)

        fusion = torch.cat([cls_repr, tab_repr], dim=-1)
        logits = self.classifier(fusion).squeeze(-1)
        return logits


# =========================
# Metrics
# =========================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return auc, sensitivity, specificity


def compute_group_auc(y_true, y_prob, races, group_name):
    idx = [i for i, r in enumerate(races) if r == group_name]
    if len(idx) == 0:
        return float("nan")
    y_g = np.array(y_true)[idx]
    p_g = np.array(y_prob)[idx]
    try:
        auc = roc_auc_score(y_g, p_g)
    except ValueError:
        auc = float("nan")
    return auc


# =========================
# Main training pipeline
# =========================

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # basic cleaning
    for c in CAT_COLS:
        df[c] = df[c].fillna("unknown").astype(str).str.lower()
    df["note"] = df["note"].fillna("").astype(str)
    df["gpt4_summary"] = df["gpt4_summary"].fillna("").astype(str)
    df["race"] = df["race"].fillna("unknown").astype(str).str.lower()

    df["label"] = (df["glaucoma"].astype(str).str.lower() == "yes").astype(int)

    # split by 'use'
    if "use" in df.columns:
        train_df = df[df["use"].str.lower() == "training"].copy()
        val_df = df[df["use"].str.lower() == "validation"].copy()
        test_df = df[df["use"].str.lower() == "test"].copy()

        if len(val_df) == 0:
            val_size = max(1, int(len(train_df) * VAL_SPLIT_FALLBACK))
            val_df = train_df.sample(n=val_size, random_state=42)
            train_df = train_df.drop(val_df.index)
    else:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=True)

    # categorical mappings from train
    cat2idx = {}
    cat_cardinalities = []
    for c in CAT_COLS:
        uniques = sorted(train_df[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        cat2idx[c] = mapping
        cat_cardinalities.append(len(uniques))
        print(f"{c}: {len(uniques)} categories")

    # age stats
    age_mean = train_df["age"].mean()
    age_std = train_df["age"].std()
    print(f"Age mean: {age_mean:.3f}, std: {age_std:.3f}")

    # datasets
    train_dataset = GlaucomaBERTDataset(train_df, tokenizer, cat2idx, age_mean, age_std, MAX_LEN)
    val_dataset = GlaucomaBERTDataset(val_df, tokenizer, cat2idx, age_mean, age_std, MAX_LEN)
    test_dataset = GlaucomaBERTDataset(test_df, tokenizer, cat2idx, age_mean, age_std, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = ClinicalBERTWithTabular(
        bert_model_name=BERT_MODEL_NAME,
        cat_cardinalities=cat_cardinalities,
        cat_embed_dim=CAT_EMBED_DIM,
        tab_hidden_dim=TAB_HIDDEN_DIM,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_auc = -float("inf")
    best_state = None

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            input_ids, attention_mask, age, cat_idx, labels, _ = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            age = age.to(DEVICE)
            cat_idx = cat_idx.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, age, cat_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # validation
        model.eval()
        val_losses = []
        val_true = []
        val_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                input_ids, attention_mask, age, cat_idx, labels, _ = batch
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                age = age.to(DEVICE)
                cat_idx = cat_idx.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(input_ids, attention_mask, age, cat_idx)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)

                val_losses.append(loss.item())
                val_true.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())

        avg_val_loss = np.mean(val_losses)
        val_auc, val_sens, val_spec = compute_metrics(val_true, val_probs)

        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {avg_val_loss:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")
        print(f"  Val Sens:   {val_sens:.4f}")
        print(f"  Val Spec:   {val_spec:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            print("  -> New best model saved.")

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, "transformer_all_features_best.pt")
        print("Best model saved to transformer_all_features_best.pt")

    # test evaluation
    model.eval()
    test_true = []
    test_probs = []
    test_races = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids, attention_mask, age, cat_idx, labels, races = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            age = age.to(DEVICE)
            cat_idx = cat_idx.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask, age, cat_idx)
            probs = torch.sigmoid(logits)

            test_true.extend(labels.cpu().numpy().tolist())
            test_probs.extend(probs.cpu().numpy().tolist())
            test_races.extend(list(races))

    overall_auc, overall_sens, overall_spec = compute_metrics(test_true, test_probs)
    print("\n=== Test Metrics (Overall) ===")
    print(f"AUC:         {overall_auc:.4f}")
    print(f"Sensitivity: {overall_sens:.4f}")
    print(f"Specificity: {overall_spec:.4f}")

    for group in ["asian", "black", "white"]:
        auc_g = compute_group_auc(test_true, test_probs, test_races, group)
        print(f"AUC ({group}): {auc_g:.4f}")


if __name__ == "__main__":
    main()
