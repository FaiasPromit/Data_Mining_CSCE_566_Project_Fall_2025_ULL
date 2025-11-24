import os
import json
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix


# =========================
# Config
# =========================

DATA_PATH = "clinical_notes.csv"  # assumes CSV is in same folder as this script
MAX_LEN = 512
VOCAB_SIZE = 30000
TEXT_EMBED_DIM = 128
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 1
CAT_EMBED_DIM = 8
TAB_HIDDEN_DIM = 32
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-3
VAL_SPLIT_FALLBACK = 0.1  # if no 'validation' rows exist


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# =========================
# Simple tokenizer & vocab
# =========================

def simple_tokenize(text: str):
    return text.lower().split()


def build_vocab(texts, max_size=VOCAB_SIZE, min_freq=1):
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(str(t))
        counter.update(tokens)

    # reserve:
    # 0: <pad>, 1: <unk>
    most_common = [w for w, c in counter.items() if c >= min_freq]
    most_common = sorted(most_common, key=lambda w: -counter[w])
    most_common = most_common[: max_size - 2]

    word2idx = {"<pad>": 0, "<unk>": 1}
    for i, w in enumerate(most_common, start=2):
        word2idx[w] = i

    return word2idx


def text_to_ids(text, word2idx, max_len=MAX_LEN):
    tokens = simple_tokenize(str(text))
    ids = [word2idx.get(tok, 1) for tok in tokens]  # 1 = <unk>
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))  # 0 = <pad>
    return ids


# =========================
# Dataset
# =========================

CAT_COLS = ["gender", "race", "ethnicity", "language", "maritalstatus"]


class GlaucomaDataset(Dataset):
    def __init__(self, df, word2idx, cat2idx, age_mean, age_std):
        self.df = df.reset_index(drop=True)
        self.word2idx = word2idx
        self.cat2idx = cat2idx
        self.age_mean = age_mean
        self.age_std = age_std

        self.texts = []
        self.age_norm = []
        self.cat_indices = []
        self.labels = []
        self.races = []

        for i, row in self.df.iterrows():
            # Combine note + gpt4_summary
            note = str(row.get("note", ""))
            summ = str(row.get("gpt4_summary", ""))
            if summ.lower() == "nan":
                summ = ""
            text = note + " " + summ

            text_ids = text_to_ids(text, self.word2idx, MAX_LEN)

            # age normalization (train stats passed in)
            age_raw = float(row["age"])
            age_n = (age_raw - age_mean) / (age_std + 1e-8)

            # categorical indices
            cat_vec = []
            for c in CAT_COLS:
                val = str(row[c])
                idx = self.cat2idx[c].get(val, 0)  # 0 will be some category, fine
                cat_vec.append(idx)

            label = int(row["label"])
            race = str(row["race"]).lower()

            self.texts.append(text_ids)
            self.age_norm.append(age_n)
            self.cat_indices.append(cat_vec)
            self.labels.append(label)
            self.races.append(race)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_ids = torch.tensor(self.texts[idx], dtype=torch.long)
        age = torch.tensor(self.age_norm[idx], dtype=torch.float32)
        cat_idx = torch.tensor(self.cat_indices[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        race = self.races[idx]
        return text_ids, age, cat_idx, label, race


# =========================
# Model
# =========================

class BiLSTMWithTabular(nn.Module):
    def __init__(self, vocab_size, text_embed_dim, lstm_hidden_dim, lstm_num_layers,
                 cat_cardinalities, cat_embed_dim, tab_hidden_dim):
        super().__init__()

        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        text_out_dim = lstm_hidden_dim * 2  # bidirectional

        # Tabular encoder: embeddings for categoricals
        self.cat_embs = nn.ModuleList()
        for num_cat in cat_cardinalities:
            emb = nn.Embedding(num_cat, cat_embed_dim)
            self.cat_embs.append(emb)

        cat_total_dim = len(cat_cardinalities) * cat_embed_dim
        tab_in_dim = cat_total_dim + 1  # +1 for age

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_in_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.ReLU(),
        )

        # Fusion + classifier
        fusion_dim = text_out_dim + tab_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, text_ids, age_norm, cat_idx):
        """
        text_ids: (B, L)
        age_norm: (B,)
        cat_idx: (B, num_cat_cols)
        """
        # Text path
        emb = self.text_embedding(text_ids)  # (B, L, E)
        lstm_out, (h_n, c_n) = self.lstm(emb)  # h_n: (num_layers*2, B, H)

        # Take last layer's forward and backward hidden states
        # shape: (2, B, H) -> forward = -2, backward = -1
        forward_h = h_n[-2, :, :]  # (B, H)
        backward_h = h_n[-1, :, :]  # (B, H)
        text_repr = torch.cat([forward_h, backward_h], dim=-1)  # (B, 2H)

        # Tabular path
        cat_feats = []
        for i, emb_layer in enumerate(self.cat_embs):
            # cat_idx[:, i] -> (B,)
            cat_feats.append(emb_layer(cat_idx[:, i]))  # (B, cat_embed_dim)
        cat_feats = torch.cat(cat_feats, dim=-1)  # (B, cat_total_dim)

        age_norm = age_norm.view(-1, 1)  # (B,1)
        tab_input = torch.cat([cat_feats, age_norm], dim=-1)  # (B, cat_total_dim+1)
        tab_repr = self.tab_mlp(tab_input)  # (B, tab_hidden_dim)

        # Fusion
        fusion = torch.cat([text_repr, tab_repr], dim=-1)  # (B, fusion_dim)
        logits = self.classifier(fusion).squeeze(-1)  # (B,)
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
    # ---------- Load data ----------
    df = pd.read_csv(DATA_PATH)

    # Clean basic things
    # fillna for categoricals
    for c in CAT_COLS:
        df[c] = df[c].fillna("unknown").astype(str).str.lower()
    df["note"] = df["note"].fillna("").astype(str)
    df["gpt4_summary"] = df["gpt4_summary"].fillna("").astype(str)
    df["race"] = df["race"].fillna("unknown").astype(str).str.lower()

    # label: glaucoma yes/no
    df["label"] = (df["glaucoma"].astype(str).str.lower() == "yes").astype(int)

    # ---------- Train/val/test split ----------
    if "use" in df.columns:
        train_df = df[df["use"].str.lower() == "training"].copy()
        val_df = df[df["use"].str.lower() == "validation"].copy()
        test_df = df[df["use"].str.lower() == "test"].copy()

        # if no explicit validation set, split from train
        if len(val_df) == 0:
            train_df = df[df["use"].str.lower() == "training"].copy()
            val_size = max(1, int(len(train_df) * VAL_SPLIT_FALLBACK))
            val_df = train_df.sample(n=val_size, random_state=42)
            train_df = train_df.drop(val_df.index)
    else:
        # if no 'use' column, do simple split
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # ---------- Build vocab from TRAIN text ----------
    train_texts = []
    for _, row in train_df.iterrows():
        txt = str(row["note"]) + " " + str(row["gpt4_summary"])
        train_texts.append(txt)

    word2idx = build_vocab(train_texts, max_size=VOCAB_SIZE, min_freq=1)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")

    # ---------- Build categorical mappings from TRAIN ----------
    cat2idx = {}
    cat_cardinalities = []
    for c in CAT_COLS:
        uniques = sorted(train_df[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        cat2idx[c] = mapping
        cat_cardinalities.append(len(uniques))
        print(f"{c}: {len(uniques)} categories")

    # ---------- Age normalization from TRAIN ----------
    age_mean = train_df["age"].mean()
    age_std = train_df["age"].std()
    print(f"Age mean: {age_mean:.3f}, std: {age_std:.3f}")

    # ---------- Create datasets ----------
    train_dataset = GlaucomaDataset(train_df, word2idx, cat2idx, age_mean, age_std)
    val_dataset = GlaucomaDataset(val_df, word2idx, cat2idx, age_mean, age_std)
    test_dataset = GlaucomaDataset(test_df, word2idx, cat2idx, age_mean, age_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------- Init model ----------
    model = BiLSTMWithTabular(
        vocab_size=vocab_size,
        text_embed_dim=TEXT_EMBED_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        cat_cardinalities=cat_cardinalities,
        cat_embed_dim=CAT_EMBED_DIM,
        tab_hidden_dim=TAB_HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -float("inf")
    best_state = None

    # =========================
    # Training loop
    # =========================
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            text_ids, age, cat_idx, labels, _ = batch
            text_ids = text_ids.to(DEVICE)
            age = age.to(DEVICE)
            cat_idx = cat_idx.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(text_ids, age, cat_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---------- Validation ----------
        model.eval()
        val_losses = []
        val_true = []
        val_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                text_ids, age, cat_idx, labels, _ = batch
                text_ids = text_ids.to(DEVICE)
                age = age.to(DEVICE)
                cat_idx = cat_idx.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(text_ids, age, cat_idx)
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

    # ---------- Load best model ----------
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, "bilstm_best.pt")
        print("Best model saved to bilstm_best.pt")

    # =========================
    # Test evaluation
    # =========================
    model.eval()
    test_true = []
    test_probs = []
    test_races = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            text_ids, age, cat_idx, labels, races = batch
            text_ids = text_ids.to(DEVICE)
            age = age.to(DEVICE)
            cat_idx = cat_idx.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(text_ids, age, cat_idx)
            probs = torch.sigmoid(logits)

            test_true.extend(labels.cpu().numpy().tolist())
            test_probs.extend(probs.cpu().numpy().tolist())
            test_races.extend(list(races))

    overall_auc, overall_sens, overall_spec = compute_metrics(test_true, test_probs)
    print("\n=== Test Metrics (Overall) ===")
    print(f"AUC:         {overall_auc:.4f}")
    print(f"Sensitivity: {overall_sens:.4f}")
    print(f"Specificity: {overall_spec:.4f}")

    # Race-wise AUC: asian, black, white
    for group in ["asian", "black", "white"]:
        auc_g = compute_group_auc(test_true, test_probs, test_races, group)
        print(f"AUC ({group}): {auc_g:.4f}")


if __name__ == "__main__":
    main()
