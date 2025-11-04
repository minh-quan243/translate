import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
import pickle

# ===================== ENCODE SENTENCE =====================
def encode_sentence(tokens, vocab_stoi):
    """
    Chuyển list token thành list index theo vocab.
    Nếu token không có trong vocab, dùng <unk>
    """
    return torch.tensor([vocab_stoi.get(tok, vocab_stoi["<unk>"]) for tok in tokens], dtype=torch.long)

# ===================== DATASET =====================
class TranslationDataset(Dataset):
    def __init__(self, df, vocab_transform):
        """
        df: pandas DataFrame với các cột 'en_tok' và 'vi_tok' (dạng list token)
        vocab_transform: dict với structure {'en': {'stoi':...}, 'vi': {'stoi':...}}
        """
        self.df = df.copy()
        self.vocab_en = vocab_transform["en"]["stoi"]
        self.vocab_vi = vocab_transform["vi"]["stoi"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = encode_sentence(self.df.iloc[idx]["en_tok"], self.vocab_en)
        tgt = encode_sentence(self.df.iloc[idx]["vi_tok"], self.vocab_vi)
        return src, tgt

# ===================== COLLATE FN (PADDING) =====================
def collate_fn(batch, vocab_transform):
    """
    batch: list of tuples (src_tensor, tgt_tensor)
    Trả về 2 tensor đã padding: src_padded, tgt_padded
    """
    src_batch, tgt_batch = zip(*batch)
    src_pad_idx = vocab_transform["en"]["stoi"]["<pad>"]
    tgt_pad_idx = vocab_transform["vi"]["stoi"]["<pad>"]

    # pad_sequence trả về tensor shape (max_len, batch_size)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=src_pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded

# ===================== CREATE DATALOADERS =====================
def create_dataloaders_from_pickle(pickle_file, vocab_file, batch_size, num_workers=0, val_frac=0.1, test_frac=0.1, shuffle_train=True):
    """
    Trả về 3 DataLoader: train, val, test
    """
    # Load pickle tokenized
    df = pd.read_pickle(pickle_file)

    # Load vocab
    with open(vocab_file, "rb") as f:
        vocab_transform = pickle.load(f)

    # Shuffle trước khi chia
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_total = len(df)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val - n_test

    train_df = df[:n_train]
    val_df = df[n_train:n_train+n_val]
    test_df = df[n_train+n_val:]

    # Dataset
    train_dataset = TranslationDataset(train_df, vocab_transform)
    val_dataset = TranslationDataset(val_df, vocab_transform)
    test_dataset = TranslationDataset(test_df, vocab_transform)

    collate = partial(collate_fn, vocab_transform=vocab_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate, drop_last=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

# ===================== USAGE =====================
if __name__ == "__main__":
    PICKLE_FILE = r"D:\Quân\project\translate\data\tokenized.pkl"
    VOCAB_FILE = r"D:\Quân\project\translate\data\vocab_transform.pkl"
    BATCH_SIZE = 32

    train_loader, val_loader, test_loader = create_dataloaders_from_pickle(
        PICKLE_FILE, VOCAB_FILE, batch_size=BATCH_SIZE
    )

    # Quick test
    for src, tgt in train_loader:
        print(src.shape, tgt.shape)
        break
