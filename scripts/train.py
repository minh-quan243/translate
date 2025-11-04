import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from models.model import EncoderLNRes, DecoderLNRes, Seq2SeqLNRes
import time

# ===================== HYPERPARAMS =====================
BATCH_SIZE = 32
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1
DROPOUT = 0.3
N_EPOCHS = 30
LEARNING_RATE = 1e-3
CLIP = 1.0
TEACHER_FORCING = 0.2
PATIENCE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PICKLE_FILE = r"D:\Quân\project\translate\data\tokenized.pkl"
VOCAB_FILE = r"D:\Quân\project\translate\data\vocab_transform.pkl"

# ===================== DATASET + COLLATE_FN =====================
class TranslationDataset(Dataset):
    def __init__(self, df, src_field='en_tok', tgt_field='vi_tok', stoi_src=None, stoi_tgt=None):
        self.src = df[src_field].tolist()
        self.tgt = df[tgt_field].tolist()
        self.stoi_src = stoi_src
        self.stoi_tgt = stoi_tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_seq = [self.stoi_src.get(tok, self.stoi_src['<unk>']) for tok in self.src[idx]]
        tgt_seq = [self.stoi_tgt.get(tok, self.stoi_tgt['<unk>']) for tok in self.tgt[idx]]
        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # --- Sort batch by src length (descending) để bucketing ---
    src_batch, tgt_batch = zip(*sorted(zip(src_batch, tgt_batch), key=lambda x: len(x[0]), reverse=True))
    src_padded = pad_sequence(src_batch, padding_value=SRC_PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, padding_value=TGT_PAD_IDX)
    return src_padded, tgt_padded

# ===================== LOAD DATA =====================
df = pd.read_pickle(PICKLE_FILE)
with open(VOCAB_FILE, "rb") as f:
    vocab_transform = pickle.load(f)

SRC_VOCAB_SIZE = len(vocab_transform['en']['stoi'])
TGT_VOCAB_SIZE = len(vocab_transform['vi']['stoi'])
SRC_PAD_IDX = vocab_transform['en']['stoi']['<pad>']
TGT_PAD_IDX = vocab_transform['vi']['stoi']['<pad>']

# --- Split train/val/test (ví dụ 80/10/10) ---
n_total = len(df)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)

df_train = df[:n_train].reset_index(drop=True)
df_val = df[n_train:n_train+n_val].reset_index(drop=True)
df_test = df[n_train+n_val:].reset_index(drop=True)

train_dataset = TranslationDataset(df_train, stoi_src=vocab_transform['en']['stoi'], stoi_tgt=vocab_transform['vi']['stoi'])
val_dataset = TranslationDataset(df_val, stoi_src=vocab_transform['en']['stoi'], stoi_tgt=vocab_transform['vi']['stoi'])
test_dataset = TranslationDataset(df_test, stoi_src=vocab_transform['en']['stoi'], stoi_tgt=vocab_transform['vi']['stoi'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ===================== INIT MODEL =====================
enc = EncoderLNRes(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = DecoderLNRes(TGT_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2SeqLNRes(enc, dec, device).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

# ===================== TRAIN & EVAL =====================
def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(iterator, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt, teacher_forcing_ratio)
        output_dim = output.shape[-1]

        # --- Flatten, ignore first token (assume <bos>) ---
        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        # --- Gradient clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in iterator:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].reshape(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# ===================== TRAIN LOOP =====================
train_losses, val_losses = [], []
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, TEACHER_FORCING)
    val_loss = evaluate(model, val_loader, criterion)
    end_time = time.time()
    mins, secs = divmod(int(end_time - start_time), 60)
    print(f"\nEpoch {epoch:02} | Time: {mins}m {secs}s | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # --- Early stopping & checkpoint ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), r"D:\Quân\project\translate\checkpoint\checkpoint_best.pth")
        print("💾 Saved best checkpoint")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("🛑 Early stopping triggered!")
            break

# ===================== PLOT LOSS =====================
plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label="Train Loss")
plt.plot(range(1, len(val_losses)+1), val_losses, marker='x', label="Val Loss")
plt.title("Training & Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
