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
import os

# ===================== HYPERPARAMS =====================
BATCH_SIZE = 32
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2  # Tăng lên 2 layers để capture context tốt hơn
DROPOUT = 0.3
N_EPOCHS = 30
LEARNING_RATE = 1e-3
CLIP = 1.0
TEACHER_FORCING = 0.5  # Bắt đầu cao, sẽ giảm dần
PATIENCE = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PICKLE_FILE = r"D:\Quân\project\translate\data\tokenized.pkl"
VOCAB_FILE = r"D:\Quân\project\translate\data\vocab_transform.pkl"
CHECKPOINT_DIR = r"D:\Quân\project\translate\checkpoint"

# Tạo thư mục checkpoint nếu chưa tồn tại
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ===================== DATASET FIXED =====================
class TranslationDataset(Dataset):
    def __init__(self, df, src_field='en_tok', tgt_field='vi_tok', stoi_src=None, stoi_tgt=None, max_len=50):
        self.src = df[src_field].tolist()
        self.tgt = df[tgt_field].tolist()
        self.stoi_src = stoi_src
        self.stoi_tgt = stoi_tgt
        self.max_len = max_len
        self.bos_idx = stoi_tgt['<bos>']
        self.eos_idx = stoi_tgt['<eos>']
        self.unk_idx = stoi_tgt['<unk>']

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # Source: không thêm special tokens
        src_seq = [self.stoi_src.get(tok, self.stoi_src['<unk>']) for tok in self.src[idx]]
        if len(src_seq) > self.max_len:
            src_seq = src_seq[:self.max_len]

        # Target: CẦN cả input và output cho teacher forcing
        tgt_tokens = self.tgt[idx]
        tgt_seq = [self.stoi_tgt.get(tok, self.unk_idx) for tok in tgt_tokens]
        if len(tgt_seq) > self.max_len - 2:  # Trừ chỗ cho <bos> và <eos>
            tgt_seq = tgt_seq[:self.max_len - 2]

        # Target INPUT: thêm <bos> nhưng không thêm <eos>
        tgt_input = [self.bos_idx] + tgt_seq

        # Target OUTPUT: thêm <eos> nhưng không thêm <bos>
        tgt_output = tgt_seq + [self.eos_idx]

        return (torch.tensor(src_seq, dtype=torch.long),
                torch.tensor(tgt_input, dtype=torch.long),
                torch.tensor(tgt_output, dtype=torch.long))


def collate_fn(batch):
    src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)

    # Sort by source length (descending) để tận dụng packed sequences
    sorted_indices = sorted(range(len(src_batch)), key=lambda i: len(src_batch[i]), reverse=True)
    src_batch = [src_batch[i] for i in sorted_indices]
    tgt_input_batch = [tgt_input_batch[i] for i in sorted_indices]
    tgt_output_batch = [tgt_output_batch[i] for i in sorted_indices]

    src_padded = pad_sequence(src_batch, padding_value=SRC_PAD_IDX)
    tgt_input_padded = pad_sequence(tgt_input_batch, padding_value=TGT_PAD_IDX)
    tgt_output_padded = pad_sequence(tgt_output_batch, padding_value=TGT_PAD_IDX)

    # Tính lengths cho packed sequences (nếu muốn tối ưu)
    src_lengths = torch.tensor([len(seq) for seq in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_input_batch], dtype=torch.long)

    return src_padded, tgt_input_padded, tgt_output_padded, src_lengths, tgt_lengths


# ===================== LOAD DATA FIXED =====================
df = pd.read_pickle(PICKLE_FILE)
with open(VOCAB_FILE, "rb") as f:
    vocab_transform = pickle.load(f)

SRC_VOCAB_SIZE = len(vocab_transform['en']['stoi'])
TGT_VOCAB_SIZE = len(vocab_transform['vi']['stoi'])
SRC_PAD_IDX = vocab_transform['en']['stoi']['<pad>']
TGT_PAD_IDX = vocab_transform['vi']['stoi']['<pad>']

print(f"📊 Vocabulary Sizes - EN: {SRC_VOCAB_SIZE}, VI: {TGT_VOCAB_SIZE}")

# Split train/val/test
n_total = len(df)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)

df_train = df[:n_train].reset_index(drop=True)
df_val = df[n_train:n_train + n_val].reset_index(drop=True)
df_test = df[n_train + n_val:].reset_index(drop=True)

print(f"📈 Dataset Split - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

train_dataset = TranslationDataset(df_train, stoi_src=vocab_transform['en']['stoi'],
                                   stoi_tgt=vocab_transform['vi']['stoi'])
val_dataset = TranslationDataset(df_val, stoi_src=vocab_transform['en']['stoi'],
                                 stoi_tgt=vocab_transform['vi']['stoi'])
test_dataset = TranslationDataset(df_test, stoi_src=vocab_transform['en']['stoi'],
                                  stoi_tgt=vocab_transform['vi']['stoi'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ===================== INIT MODEL =====================
enc = EncoderLNRes(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = DecoderLNRes(TGT_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2SeqLNRes(enc, dec, device).to(device)

print(f"🚀 Model initialized - Encoder: {SRC_VOCAB_SIZE}→{EMB_DIM}→{HID_DIM}")
print(f"                   Decoder: {TGT_VOCAB_SIZE}←{EMB_DIM}←{HID_DIM}")


# Đếm số parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"📐 Trainable parameters: {count_parameters(model):,}")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                 patience=2, verbose=True)


# ===================== TRAIN & EVAL FIXED =====================
def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=0.5, epoch=0):
    model.train()
    epoch_loss = 0

    # Giảm teacher forcing theo thời gian
    current_tf = max(0.1, teacher_forcing_ratio * (0.95 ** epoch))

    for src, tgt_input, tgt_output, src_lengths, tgt_lengths in tqdm(iterator, desc=f"Training Epoch {epoch}",
                                                                     leave=False):
        src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

        optimizer.zero_grad()

        # FIXED: Truyền tgt_input (có <bos>) thay vì tgt_output
        output = model(src, tgt_input, current_tf)  # output: [tgt_len, batch, vocab_size]
        output_dim = output.shape[-1]

        # Reshape output và target để tính loss
        # Bỏ qua timestep đầu tiên (thường là <bos>)
        output = output[1:].view(-1, output_dim)  # [ (tgt_len-1)*batch, vocab_size ]
        tgt_output = tgt_output[1:].view(-1)  # [ (tgt_len-1)*batch ]

        loss = criterion(output, tgt_output)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), current_tf


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt_input, tgt_output, src_lengths, tgt_lengths in iterator:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            # Không dùng teacher forcing khi evaluation
            output = model(src, tgt_input, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            tgt_output = tgt_output[1:].view(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# ===================== TRAIN LOOP IMPROVED =====================
train_losses, val_losses = [], []
best_val_loss = float("inf")
epochs_no_improve = 0

print("🎯 Starting Training...")

for epoch in range(1, N_EPOCHS + 1):
    start_time = time.time()

    train_loss, current_tf = train_epoch(model, train_loader, optimizer, criterion, CLIP, TEACHER_FORCING, epoch)
    val_loss = evaluate(model, val_loader, criterion)

    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    end_time = time.time()
    mins, secs = divmod(int(end_time - start_time), 60)

    print(f"\nEpoch {epoch:02} | Time: {mins}m {secs}s")
    print(f"   Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
    print(f"   LR: {current_lr:.2e} | TF: {current_tf:.2f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping & checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab_transform': vocab_transform
        }, os.path.join(CHECKPOINT_DIR, "checkpoint_best.pth"))
        print("   💾 Saved BEST checkpoint")
    else:
        epochs_no_improve += 1
        print(f"   ⏳ No improvement: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print("   🛑 Early stopping triggered!")
            break

    # Save periodic checkpoint
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))

# ===================== PLOT LOSS =====================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Train Loss", linewidth=2)
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', label="Val Loss", linewidth=2)
plt.title("Training & Validation Loss Curve", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

print("✅ Training completed!")
print(f"📊 Final - Best Val Loss: {best_val_loss:.3f}, Final Train Loss: {train_losses[-1]:.3f}")