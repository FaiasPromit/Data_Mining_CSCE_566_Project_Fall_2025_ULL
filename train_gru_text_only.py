import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VOCAB_SIZE = 30000
EMB_DIM = 128
HIDDEN_DIM = 128

print("Using device:", DEVICE)

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("clinical_notes.csv")

# Only use "note"
df["note"] = df["note"].fillna("").astype(str)

# Train/val/test split like before
train_df = df[df["use"] == "training"]
val_df   = df[df["use"] == "validation"]
test_df  = df[df["use"] == "test"]

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

# -----------------------------
# 2. Tokenizer (word-level)
# -----------------------------
from collections import Counter

def build_vocab(texts, max_vocab=VOCAB_SIZE):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<pad>":0, "<unk>":1}
    for word,_ in counter.most_common(max_vocab-2):
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(train_df["note"])
print("Vocab size:", len(vocab))

def encode(text):
    tokens = text.lower().split()
    ids = [vocab.get(t, 1) for t in tokens]
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
    else:
        ids += [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# -----------------------------
# 3. Dataset
# -----------------------------
class NoteDataset(Dataset):
    def __init__(self, df):
        self.notes = df["note"].tolist()
        self.labels = df["glaucoma"].apply(lambda x: 1 if x=="yes" else 0).tolist()
        self.race = df["race"].tolist()

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        return {
            "input_ids": encode(self.notes[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "race": self.race[idx]
        }

train_ds = NoteDataset(train_df)
val_ds   = NoteDataset(val_df)
test_ds  = NoteDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# -----------------------------
# 4. GRU Text-Only Model
# -----------------------------
class GRU_TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), EMB_DIM)
        self.gru = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM*2, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]   # last hidden state
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))

model = GRU_TextOnly().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -----------------------------
# 5. Evaluation Helper
# -----------------------------
def evaluate(loader):
    model.eval()
    preds, labels, races = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            out = model(x).squeeze()
            preds.extend(out.cpu().numpy())
            labels.extend(y.cpu().numpy())
            races.extend(batch["race"])
    auc = roc_auc_score(labels, preds)
    return auc, np.array(labels), np.array(preds), races

def subgroup_auc(labels, preds, races, group):
    mask = (np.array(races) == group)
    if mask.sum() < 5:
        return np.nan
    return roc_auc_score(labels[mask], preds[mask])

# -----------------------------
# 6. Training Loop
# -----------------------------
best_auc = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        x = batch["input_ids"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        out = model(x).squeeze()
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_auc, val_labels, val_preds, val_races = evaluate(val_loader)

    print(f"\nEpoch {epoch}: Val AUC = {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "gru_text_only.pt")
        print("→ New best model saved.")

# -----------------------------
# 7. Test Evaluation
# -----------------------------
model.load_state_dict(torch.load("gru_text_only.pt"))
test_auc, labels, preds, races = evaluate(test_loader)

print("\n=== Test Results (GRU Text Only) ===")
print("AUC:", round(test_auc, 4))
print("AUC (Asian):", subgroup_auc(labels, preds, races, "asian"))
print("AUC (Black):", subgroup_auc(labels, preds, races, "black"))
print("AUC (White):", subgroup_auc(labels, preds, races, "white"))
