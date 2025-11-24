import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

print("Using device:", DEVICE)

# --------------------------------
# 1. Load Dataset
# --------------------------------
df = pd.read_csv("clinical_notes.csv")
df["note"] = df["note"].fillna("").astype(str)

train_df = df[df["use"] == "training"]
val_df   = df[df["use"] == "validation"]
test_df  = df[df["use"] == "test"]

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

# --------------------------------
# 2. Tokenizer
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

# --------------------------------
# 3. Dataset
# --------------------------------
class TextOnlyDataset(Dataset):
    def __init__(self, df):
        self.notes = df["note"].tolist()
        self.labels = df["glaucoma"].apply(lambda x: 1 if x=="yes" else 0).tolist()
        self.race = df["race"].tolist()

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        input_ids, mask = encode(self.notes[idx])
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "race": self.race[idx]
        }

train_ds = TextOnlyDataset(train_df)
val_ds   = TextOnlyDataset(val_df)
test_ds  = TextOnlyDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --------------------------------
# 4. ClinicalBERT Model (no MLP for structured data)
# --------------------------------
class ClinicalBERT_TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask).last_hidden_state
        pooled = out[:, 0, :]     # CLS token
        pooled = self.dropout(pooled)
        return torch.sigmoid(self.fc(pooled)).squeeze()

model = ClinicalBERT_TextOnly().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --------------------------------
# 5. Evaluation Helpers
# --------------------------------
def evaluate(loader):
    model.eval()
    preds, labels, races = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            out = model(ids, mask)
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

# --------------------------------
# 6. Training
# --------------------------------
best_auc = 0

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        out = model(ids, mask)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_auc, _, _, _ = evaluate(val_loader)

    print(f"\nEpoch {epoch}: Val AUC = {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "transformer_text_only.pt")
        print("→ New best model saved.")

# --------------------------------
# 7. Final Test Evaluation
# --------------------------------
model.load_state_dict(torch.load("transformer_text_only.pt"))

test_auc, labels, preds, races = evaluate(test_loader)

print("\n=== Test Results (ClinicalBERT Text Only) ===")
print("AUC:", round(test_auc, 4))
print("AUC (Asian):", subgroup_auc(labels, preds, races, "asian"))
print("AUC (Black):", subgroup_auc(labels, preds, races, "black"))
print("AUC (White):", subgroup_auc(labels, preds, races, "white"))
