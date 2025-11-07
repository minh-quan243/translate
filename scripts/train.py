import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm  # <-- thêm tqdm
from processing import train_ds, val_ds, collate_fn, vocab_transform
from models.model import Seq2SeqTransformer, create_mask

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX = 1

BATCH_SIZE = 64
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DROP_OUT = 0.1

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, len(vocab_transform['en']), len(vocab_transform['vi']), FFN_HID_DIM, DROP_OUT).to(DEVICE)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
accumulation_steps = 5

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    optimizer.zero_grad()
    for i, (src, tgt) in enumerate(tqdm(train_dataloader, desc="Training")):  # <-- tqdm
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        losses += loss.item()
    return losses / len(train_dataloader)

def evaluate(model):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc="Validation"):  # <-- tqdm
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)) / accumulation_steps
            losses += loss.item()
    return losses / len(val_dataloader)

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:
                self.early_stop = True

# Training loop
from timeit import default_timer as timer

early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)
NUM_EPOCHS = 50
history = {"loss": [], "val_los": []}

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    val_loss = evaluate(transformer)
    history['loss'].append(train_loss)
    history['val_los'].append(val_loss)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Time={(timer()-start_time):.3f}s")
    early_stopping(train_loss, val_loss)
    if early_stopping.early_stop:
        print("Early stopping at epoch:", epoch)
        break

# Save model
torch.save(transformer.state_dict(), "transformer_en_vi.pth")
