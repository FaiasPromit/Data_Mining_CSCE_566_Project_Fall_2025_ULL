import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix


# =========================
# Config
# =========================

DATA_PATH = "clinical_notes.csv"

MAX_LEN = 512
VOCAB_SIZE = 30000

TEXT_EMBED_DIM = 128
GRU_HIDDEN_DIM = 128
GRU_NUM_LAYERS = 1

CAT_COLS = ["gender", "race", "ethnicity", "language", "maritalstatus"]
CAT_EMBED_DIM = 8
TAB_HIDDEN_DIM = 32

BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# =========================
# Tokenizer + vocab
# =========================

def simple_tokenize(text):
    return str(text).lower().split()

def build_vocab(texts, max_size=VOCAB_SIZE):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    most_common = [w for w, _ in counter.most_common(max_size - 2)]
    word2idx = {"<pad>": 0, "<unk>": 1}
    for i, w in enumerate(most_common, start=2):
        word2idx[w] = i
    return word2idx

def text_to_ids(text, word2idx):
    tokens = simple_tokenize(text)
    ids = [word2idx.get(tok, 1) for tok in tokens]
    if len(ids) > MAX_LEN:
        return ids[:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))


# =========================
# Dataset
# =========================

class GRUDataset(Dataset):
    def __init__(self, df, word2idx, cat2idx, age_mean, age_std):
        self.df = df.reset_index(drop=True)
        self.word2idx = word2idx
        self.cat2idx = cat2idx
        self.age_mean = age_mean
        self.age_std = age_std

        self.texts = []
        self.ages = []
        self.cat_idx = []
        self.labels = []
        self.races = []

        for _, row in df.iterrows():
            note = str(row["note"])
            summ = str(row["gpt4_summary"])
            summ = "" if summ.lower() == "nan" else summ
            text = note + " " + summ

            text_ids = text_to_ids(text, word2idx)
            self.texts.append(text_ids)

            age_raw = float(row["age"])
            age_n = (age_raw - age_mean) / (age_std + 1e-8)
            self.ages.append(age_n)

            cat_list = []
            for c in CAT_COLS:
                val = str(row[c])
                cat_list.append(cat2idx[c].get(val, 0))
            self.cat_idx.append(cat_list)

            self.labels.append(int(row["label"]))
            self.races.append(str(row["race"]).lower())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        cats = torch.tensor(self.cat_idx[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        race = self.races[idx]
        return text, age, cats, label, race


# =========================
# GRU Model + Tabular
# =========================

class GRUWithTabular(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 cat_cardinalities, cat_embed_dim, tab_hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        gru_output_dim = hidden_dim * 2  # because bidirectional

        self.cat_embs = nn.ModuleList([
            nn.Embedding(num_cat, cat_embed_dim)
            for num_cat in cat_cardinalities
        ])

        cat_total_dim = len(cat_cardinalities) * cat_embed_dim
        tab_input_dim = cat_total_dim + 1  # +1 for age

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_input_dim, tab_hidden_dim),
            nn.ReLU(),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = gru_output_dim + tab_hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, text_ids, age_norm, cat_idx):
        emb = self.embedding(text_ids)  # (B, L, E)

        _, h_n = self.gru(emb)          # h_n: (num_layers*2, B, H)
        forward_h = h_n[-2, :, :]
        backward_h = h_n[-1, :, :]
        text_repr = torch.cat([forward_h, backward_h], dim=-1)

        cat_feats = []
        for i, emb_layer in enumerate(self.cat_embs):
            cat_feats.append(emb_layer(cat_idx[:, i]))
        cat_feats = torch.cat(cat_feats, dim=-1)

        age_norm = age_norm.view(-1, 1)
        tab_input = torch.cat([cat_feats, age_norm], dim=-1)
        tab_repr = self.tab_mlp(tab_input)

        fusion = torch.cat([text_repr, tab_repr], dim=-1)
        logits = self.classifier(fusion).squeeze(-1)

        return logits


# =========================
# Metrics
# =========================

def compute_metrics(y_true, y_prob):
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    return auc, sens, spec


def group_auc(y_true, y_prob, races, group):
    idx = [i for i, r in enumerate(races) if r == group]
    if len(idx) == 0:
        return float("nan")
    try:
        return roc_auc_score(np.array(y_true)[idx], np.array(y_prob)[idx])
    except:
        return float("nan")


# =========================
# Main Training Pipeline
# =========================

def main():
    df = pd.read_csv(DATA_PATH)

    # Clean categorical columns
    for c in CAT_COLS:
        df[c] = df[c].fillna("unknown").astype(str).str.lower()

    df["note"] = df["note"].fillna("")
    df["gpt4_summary"] = df["gpt4_summary"].fillna("")
    df["race"] = df["race"].fillna("unknown").astype(str).str.lower()
    df["label"] = (df["glaucoma"].astype(str).str.lower() == "yes").astype(int)

    train_df = df[df["use"].str.lower() == "training"]
    val_df   = df[df["use"].str.lower() == "validation"]
    test_df  = df[df["use"].str.lower() == "test"]

    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    # Build vocab from training text
    train_texts = [
        str(r["note"]) + " " + str(r["gpt4_summary"])
        for _, r in train_df.iterrows()
    ]
    word2idx = build_vocab(train_texts)
    vocab_size = len(word2idx)
    print("Vocab size:", vocab_size)

    # Build categorical mappings
    cat2idx = {}
    cat_cardinalities = []
    for c in CAT_COLS:
        uniq = sorted(train_df[c].unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        cat2idx[c] = mapping
        cat_cardinalities.append(len(uniq))

    # Age normalization
    age_mean = train_df["age"].mean()
    age_std = train_df["age"].std()

    # Dataset + loaders
    train_ds = GRUDataset(train_df, word2idx, cat2idx, age_mean, age_std)
    val_ds   = GRUDataset(val_df, word2idx, cat2idx, age_mean, age_std)
    test_ds  = GRUDataset(test_df, word2idx, cat2idx, age_mean, age_std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = GRUWithTabular(
        vocab_size=vocab_size,
        embed_dim=TEXT_EMBED_DIM,
        hidden_dim=GRU_HIDDEN_DIM,
        num_layers=GRU_NUM_LAYERS,
        cat_cardinalities=cat_cardinalities,
        cat_embed_dim=CAT_EMBED_DIM,
        tab_hidden_dim=TAB_HIDDEN_DIM
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = -1
    best_state = None

    # Training
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        for text, age, cats, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            text = text.to(DEVICE)
            age = age.to(DEVICE)
            cats = cats.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(text, age, cats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_true = []
        val_prob = []

        with torch.no_grad():
            for text, age, cats, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                text = text.to(DEVICE)
                age = age.to(DEVICE)
                cats = cats.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(text, age, cats)
                probs = torch.sigmoid(logits)

                val_true.extend(labels.cpu().numpy())
                val_prob.extend(probs.cpu().numpy())

        auc, sens, spec = compute_metrics(val_true, val_prob)
        print(f"\nEpoch {epoch}: AUC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    torch.save(best_state, "gru_all_features_best.pt")
    print("Saved best → gru_all_features_best.pt")

    # Test
    model.load_state_dict(best_state)
    model.eval()

    test_true = []

    test_prob = []
    test_races = []

    with torch.no_grad():
        for text, age, cats, labels, races in tqdm(test_loader, desc="Test"):
            text = text.to(DEVICE)
            age = age.to(DEVICE)
            cats = cats.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(text, age, cats)
            probs = torch.sigmoid(logits)

            test_true.extend(labels.cpu().numpy())
            test_prob.extend(probs.cpu().numpy())
            test_races.extend(list(races))

    auc, sens, spec = compute_metrics(test_true, test_prob)
    print("\n=== Test (GRU + All Features) ===")
    print(f"AUC:         {auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")

    for g in ["asian", "black", "white"]:
        print(f"AUC ({g}): {group_auc(test_true, test_prob, test_races, g):.4f}")


if __name__ == "__main__":
    main()
