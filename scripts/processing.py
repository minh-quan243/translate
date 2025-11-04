import pandas as pd
import re
import string
import warnings
import pickle
from underthesea import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# ===================== CLEAN, TOKENIZE & FILTER =====================
def load_tokenize_and_filter(file_path: str, max_token_diff=5):
    """
    Load CSV, clean text, tokenize, và lọc các câu có số token giữa EN và VI gần bằng nhau.
    """
    df = pd.read_csv(file_path).dropna(subset=["en", "vi"])

    # --- Clean text ---
    def clean_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\s+", " ", text.strip().lower())
        return text

    df["en"] = df["en"].apply(clean_text)
    df["vi"] = df["vi"].apply(clean_text)

    # --- Tokenizer ---
    def en_tokenizer(text: str):
        return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())

    def vi_tokenizer(sentence: str):
        return word_tokenize(sentence)

    # --- Tokenize ---
    df["en_tok"] = df["en"].apply(en_tokenizer)
    df["vi_tok"] = df["vi"].apply(vi_tokenizer)

    # --- Count token lengths ---
    df["en_len"] = df["en_tok"].apply(len)
    df["vi_len"] = df["vi_tok"].apply(len)
    df["diff_len"] = abs(df["en_len"] - df["vi_len"])

    # --- Visualize before filtering ---
    plt.figure(figsize=(12,5))
    sns.histplot(df["en_len"], color='blue', label='EN', kde=True, bins=30)
    sns.histplot(df["vi_len"], color='green', label='VI', kde=True, bins=30)
    plt.title("Sentence length distribution BEFORE filtering (tokenized)")
    plt.xlabel("Number of tokens")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    sns.histplot(df["diff_len"], bins=range(0, df["diff_len"].max()+1), color='purple', kde=False)
    plt.title("Token length differences (EN vs VI) BEFORE filtering")
    plt.xlabel("Token difference")
    plt.ylabel("Count")
    plt.show()

    # --- Filter câu có diff <= max_token_diff ---
    df_filtered = df[df["diff_len"] <= max_token_diff].reset_index(drop=True)
    print(f"✅ Filtered {len(df_filtered)} rows with token difference <= {max_token_diff}")

    # --- Visualize after filtering ---
    plt.figure(figsize=(12,5))
    sns.histplot(df_filtered["en_len"], color='blue', label='EN', kde=True, bins=30)
    sns.histplot(df_filtered["vi_len"], color='green', label='VI', kde=True, bins=30)
    plt.title(f"Sentence length distribution AFTER filtering (diff <= {max_token_diff})")
    plt.xlabel("Number of tokens")
    plt.legend()
    plt.show()

    return df_filtered, en_tokenizer, vi_tokenizer

# ===================== BUILD VOCAB =====================
def build_vocab_from_tokens(df, en_tokenizer, vi_tokenizer):
    SRC, TGT = 'en', 'vi'
    token_transform, vocab_transform = {}, {}

    token_transform[SRC] = en_tokenizer
    token_transform[TGT] = vi_tokenizer

    def build_vocab(token_lists):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        vocab = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(counter.keys())
        stoi = {word: i for i, word in enumerate(vocab)}
        itos = {i: word for word, i in stoi.items()}

        # --- Visualize top 20 tokens ---
        most_common = counter.most_common(20)
        words, counts = zip(*most_common)
        plt.figure(figsize=(12,5))
        sns.barplot(x=list(words), y=list(counts))
        plt.title("Top 20 most frequent tokens")
        plt.xticks(rotation=45)
        plt.show()

        return {"vocab": vocab, "stoi": stoi, "itos": itos}

    vocab_transform[SRC] = build_vocab(df["en_tok"])
    vocab_transform[TGT] = build_vocab(df["vi_tok"])

    return token_transform, vocab_transform

# ===================== SAVE PICKLE =====================
def save_pickle(df, token_pickle_path, vocab_pickle_path, vocab_transform):
    df.to_pickle(token_pickle_path)
    print(f"💾 Tokenized pickle saved → {token_pickle_path}")

    with open(vocab_pickle_path, "wb") as f:
        pickle.dump(vocab_transform, f)
    print(f"💾 Saved vocab → {vocab_pickle_path}")

# ===================== MAIN =====================
if __name__ == "__main__":
    FILE_PATH = r"D:\Quân\project\translate\data\file_song_ngu.csv"
    TOKEN_PICKLE = r"D:\Quân\project\translate\data\tokenized.pkl"
    VOCAB_PICKLE = r"D:\Quân\project\translate\data\vocab_transform.pkl"
    MAX_DIFF = 5

    # --- Load, clean, tokenize và filter ---
    df_filtered, en_tokenizer, vi_tokenizer = load_tokenize_and_filter(FILE_PATH, max_token_diff=MAX_DIFF)

    # --- Build vocab từ token đã tokenize ---
    token_transform, vocab_transform = build_vocab_from_tokens(df_filtered, en_tokenizer, vi_tokenizer)

    # --- Save pickle ---
    save_pickle(df_filtered, TOKEN_PICKLE, VOCAB_PICKLE, vocab_transform)
