import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
import pickle


# ===================== ENCODE SENTENCE IMPROVED =====================
def encode_sentence(tokens, vocab_stoi, add_bos=False, add_eos=False, max_len=None):
    """
    Chuyển list token thành list index theo vocab.
    Nếu token không có trong vocab, dùng <unk>
    """
    indices = [vocab_stoi.get(tok, vocab_stoi["<unk>"]) for tok in tokens]

    # Thêm <bos> và <eos>
    if add_bos:
        indices = [vocab_stoi["<bos>"]] + indices
    if add_eos:
        indices = indices + [vocab_stoi["<eos>"]]

    # Giới hạn độ dài tối đa (nếu có)
    if max_len and len(indices) > max_len:
        if add_eos:
            indices = indices[:max_len - 1] + [vocab_stoi["<eos>"]]
        else:
            indices = indices[:max_len]

    return torch.tensor(indices, dtype=torch.long)


# ===================== DATASET IMPROVED =====================
class TranslationDataset(Dataset):
    def __init__(self, df, vocab_transform, max_seq_len=50):
        """
        df: pandas DataFrame với các cột 'en_tok' và 'vi_tok' (dạng list token)
        vocab_transform: dict với structure {'en': {'stoi':...}, 'vi': {'stoi':...}}
        max_seq_len: giới hạn độ dài tối đa (tính cả special tokens)
        """
        self.df = df.copy()
        self.vocab_en = vocab_transform["en"]["stoi"]
        self.vocab_vi = vocab_transform["vi"]["stoi"]
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Source (encoder input): không cần <bos>/<eos>
        src = encode_sentence(
            self.df.iloc[idx]["en_tok"],
            self.vocab_en,
            add_bos=False,
            add_eos=False,
            max_len=self.max_seq_len
        )

        # Target (decoder input và output)
        tgt_tokens = self.df.iloc[idx]["vi_tok"]

        # Decoder INPUT: thêm <bos> nhưng không thêm <eos>
        tgt_input = encode_sentence(
            tgt_tokens,
            self.vocab_vi,
            add_bos=True,
            add_eos=False,
            max_len=self.max_seq_len
        )

        # Decoder OUTPUT (target): thêm <eos> nhưng không thêm <bos>
        tgt_output = encode_sentence(
            tgt_tokens,
            self.vocab_vi,
            add_bos=False,
            add_eos=True,
            max_len=self.max_seq_len
        )

        return src, tgt_input, tgt_output


# ===================== COLLATE FN IMPROVED =====================
def collate_fn(batch, vocab_transform, return_lengths=True):
    """
    batch: list of tuples (src_tensor, tgt_input_tensor, tgt_output_tensor)
    Trả về: src_padded, tgt_input_padded, tgt_output_padded, [src_lengths, tgt_lengths]
    """
    src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)

    src_pad_idx = vocab_transform["en"]["stoi"]["<pad>"]
    tgt_pad_idx = vocab_transform["vi"]["stoi"]["<pad>"]

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=src_pad_idx)
    tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input_batch, padding_value=tgt_pad_idx)
    tgt_output_padded = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, padding_value=tgt_pad_idx)

    if return_lengths:
        # Tính độ dài thực của mỗi sequence (trước khi padding)
        src_lengths = torch.tensor([len(seq) for seq in src_batch], dtype=torch.long)
        tgt_lengths = torch.tensor([len(seq) for seq in tgt_input_batch], dtype=torch.long)

        return src_padded, tgt_input_padded, tgt_output_padded, src_lengths, tgt_lengths
    else:
        return src_padded, tgt_input_padded, tgt_output_padded


# ===================== CREATE DATALOADERS IMPROVED =====================
def create_dataloaders_from_pickle(pickle_file, vocab_file, batch_size, num_workers=0,
                                   val_frac=0.1, test_frac=0.1, shuffle_train=True, max_seq_len=50):
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
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]

    print(f"Dataset sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Dataset với max_seq_len
    train_dataset = TranslationDataset(train_df, vocab_transform, max_seq_len=max_seq_len)
    val_dataset = TranslationDataset(val_df, vocab_transform, max_seq_len=max_seq_len)
    test_dataset = TranslationDataset(test_df, vocab_transform, max_seq_len=max_seq_len)

    collate = partial(collate_fn, vocab_transform=vocab_transform, return_lengths=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate, drop_last=False)

    print(f"DataLoader sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Print vocab sizes
    en_vocab_size = len(vocab_transform["en"]["vocab"])
    vi_vocab_size = len(vocab_transform["vi"]["vocab"])
    print(f"Vocabulary sizes - EN: {en_vocab_size}, VI: {vi_vocab_size}")

    return train_loader, val_loader, test_loader, vocab_transform


# ===================== USAGE =====================
if __name__ == "__main__":
    PICKLE_FILE = r"D:\Quân\project\translate\data\tokenized.pkl"
    VOCAB_FILE = r"D:\Quân\project\translate\data\vocab_transform.pkl"
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 50

    train_loader, val_loader, test_loader, vocab_transform = create_dataloaders_from_pickle(
        PICKLE_FILE, VOCAB_FILE, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN
    )

    # Quick test - Kiểm tra format
    for src, tgt_input, tgt_output, src_lens, tgt_lens in train_loader:
        print(f"src: {src.shape}")  # (src_seq_len, batch_size)
        print(f"tgt_input: {tgt_input.shape}")  # (tgt_seq_len, batch_size)
        print(f"tgt_output: {tgt_output.shape}")  # (tgt_seq_len, batch_size)
        print(f"src_lens: {src_lens}")  # Độ dài thực của source sequences
        print(f"tgt_lens: {tgt_lens}")  # Độ dài thực của target sequences

        # Kiểm tra special tokens
        print(f"First src sequence: {src[:, 0]}")
        print(f"First tgt_input: {tgt_input[:, 0]}")  # Nên bắt đầu bằng <bos>
        print(f"First tgt_output: {tgt_output[:, 0]}")  # Nên kết thúc bằng <eos>
        break