import pandas as pd
import re, string
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List

# File path
DATA_DIR = r"D:\Quân\project\translate\data\file_song_ngu.csv"
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Read CSV
df = pd.read_csv(DATA_DIR)

def preprocessing(df):
    df["en"] = df["en"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)).lower().strip())
    df["vi"] = df["vi"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)).lower().strip())
    df["en"] = df["en"].apply(lambda ele: re.sub("\s+", " ", ele))
    df["vi"] = df["vi"].apply(lambda ele: re.sub("\s+", " ", ele))
    return df

df = preprocessing(df)

# Tokenizers
def vi_tokenizer(sentence):
    return word_tokenize(sentence)

token_transform = {
    SRC_LANGUAGE: get_tokenizer('basic_english'),
    TGT_LANGUAGE: vi_tokenizer
}

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    for _, data_sample in data_iter:
        yield token_transform[language](data_sample[language])

# Build vocab
vocab_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = df.iterrows()
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_iter, ln),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    vocab_transform[ln].set_default_index(UNK_IDX)

# Sequential transforms
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform)

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Split train/val
split_ratio = 0.9
split = round(len(df) * split_ratio)
train_ds = list(zip(df['en'][:split], df['vi'][:split]))
val_ds = list(zip(df['en'][split:], df['vi'][split:]))

print("Data processing complete.")
