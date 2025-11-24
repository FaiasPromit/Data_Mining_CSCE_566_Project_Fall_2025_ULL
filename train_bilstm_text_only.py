import os
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

DATA_PATH = "clinical_notes.csv"     # must be in same folder
MAX_LEN = 512
VOCAB_SIZE = 30000
TEXT_EMBED_DIM = 128
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 1

BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


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

def text_to_ids(text, word2idx, max_len=MAX_LEN):
    tokens = simple_tokenize(text)
    ids = [word2idx.get(tok, 1) for tok in tokens]   # 1 = <unk>
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))


# =========================
# Dataset
# =========================

class NotesDataset(Dataset):
    def __init__(self, df, word2idx):
        self.df = df.reset_index(drop=True)
        self.word2idx = word2idx

        self.texts = []
        self.labels = []

        for _, row in self.df.iterrows():
            txt = str(row["note"])
            self.texts.append(text_to_ids(txt, word2idx))
            self.labels.append(int(row["label"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# =========================
# BiLSTM Model (text only)
# =========================

class BiLSTM_TextOnly(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, text_ids):
        emb = self.embed(text_ids)
        _, (h_n, _) = self.lstm(emb)
        # forward hidden = h_n[-2], backward hidden = h_n[-1]
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        text_repr = torch.cat([forward_h, backward_h], dim=-1)
        return self.classifier(text_repr).squeeze(-1)


# =========================
# Metrics
# =========================

def compute_metrics(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return auc, sensitivity, specificity


# =========================
# Main
# =========================

def main():
    df = pd.read_csv(DATA_PATH)

    # label
    df["label"] = (df["glaucoma"].astype(str).str.lower() == "yes").astype(int)
    df["note"] = df["note"].fillna("")

    # train/val/test split using 'use' column
    train_df = df[df["use"].str.lower() == "training"]
    val_df   = df[df["use"].str.lower() == "validation"]
    test_df  = df[df["use"].str.lower() == "test"]

    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    # Build vocab on training notes
    vocab = build_vocab(train_df["note"].tolist(), max_size=VOCAB_SIZE)
    print("Vocab size:", len(vocab))

    # datasets
    train_ds = NotesDataset(train_df, vocab)
    val_ds   = NotesDataset(val_df, vocab)
    test_ds  = NotesDataset(test_df, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # model
    model = BiLSTM_TextOnly(
        vocab_size=len(vocab),
        embed_dim=TEXT_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = -1
    best_state = None

    # training
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        losses = []
        for text_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            text_ids = text_ids.to(DEVICE)
            labels   = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(text_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # validation
        model.eval()
        val_true = []
        val_prob = []
        with torch.no_grad():
            for text_ids, labels in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                text_ids = text_ids.to(DEVICE)
                labels   = labels.to(DEVICE)

                logits = model(text_ids)
                prob = torch.sigmoid(logits)

                val_true.extend(labels.cpu().numpy().tolist())
                val_prob.extend(prob.cpu().numpy().tolist())

        auc, sens, spec = compute_metrics(val_true, val_prob)
        print(f"Epoch {epoch}: AUC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    # save best
    torch.save(best_state, "bilstm_text_only.pt")
    print("Saved best model → bilstm_text_only.pt")

    # test
    model.load_state_dict(best_state)
    model.eval()

    test_true = []
    test_prob = []
    with torch.no_grad():
        for text_ids, labels in tqdm(test_loader, desc="Test"):
            text_ids = text_ids.to(DEVICE)
            labels   = labels.to(DEVICE)

            logits = model(text_ids)
            prob = torch.sigmoid(logits)

            test_true.extend(labels.cpu().numpy().tolist())
            test_prob.extend(prob.cpu().numpy().tolist())

    auc, sens, spec = compute_metrics(test_true, test_prob)
    print("=== Test Results (Text Only) ===")
    print(f"AUC:         {auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")


if __name__ == "__main__":
    main()
