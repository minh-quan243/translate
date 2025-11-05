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
def load_tokenize_and_filter(file_path: str, max_token_diff=5, min_len=2, max_len=50):
    """
    Load CSV, clean text, tokenize, và lọc các câu có số token giữa EN và VI gần bằng nhau.
    """
    df = pd.read_csv(file_path).dropna(subset=["en", "vi"])

    # --- Clean text IMPROVED ---
    def clean_text(text):
        # Giữ lại . ? ! cho cấu trúc câu
        text = re.sub(r'[^\w\s\.\?\!]', '', text)

        # Chuẩn hóa số
        text = re.sub(r'\b\d+\b', '<num>', text)

        # Chuẩn hóa từ viết tắt thông dụng
        contractions = {
            "i'm": "im", "i'll": "ill", "i've": "ive",
            "don't": "dont", "can't": "cant", "won't": "wont"
        }
        for cont, replacement in contractions.items():
            text = re.sub(r'\b' + cont + r'\b', replacement, text)

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

    print(f"📊 Before filtering: {len(df)} sentences")

    # --- Filter độ dài tuyệt đối ---
    df = df[(df["en_len"] >= min_len) & (df["en_len"] <= max_len) &
            (df["vi_len"] >= min_len) & (df["vi_len"] <= max_len)]

    # --- Filter chênh lệch độ dài ---
    df_filtered = df[df["diff_len"] <= max_token_diff].reset_index(drop=True)

    print(f"✅ After filtering: {len(df_filtered)} sentences "
          f"(length {min_len}-{max_len}, diff <= {max_token_diff})")

    # --- Visualizations (giữ nguyên) ---
    # ... your visualization code ...

    return df_filtered, en_tokenizer, vi_tokenizer


# ===================== BUILD VOCAB FIXED =====================
def build_vocab_from_tokens(df, en_tokenizer, vi_tokenizer):
    # FIX: Không cần lưu tokenizer ở đây nữa
    vocab_transform = {}

    def build_vocab(token_lists):  # token_lists ĐÃ là tokens rồi
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)  # Trực tiếp từ tokens, không gọi tokenizer
        vocab = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(counter.keys())
        stoi = {word: i for i, word in enumerate(vocab)}
        itos = {i: word for word, i in stoi.items()}

        # Visualization (giữ nguyên)
        most_common = counter.most_common(20)
        if most_common:
            words, counts = zip(*most_common)
            plt.figure(figsize=(12, 5))
            sns.barplot(x=list(words), y=list(counts))
            plt.title("Top 20 most frequent tokens")
            plt.xticks(rotation=45)
            plt.show()

        return {"vocab": vocab, "stoi": stoi, "itos": itos}

    # FIX: Truyền trực tiếp tokens đã tokenize
    vocab_transform['en'] = build_vocab(df["en_tok"])
    vocab_transform['vi'] = build_vocab(df["vi_tok"])

    return vocab_transform  # FIX: Chỉ trả về vocab_transform


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
    MIN_LEN = 2  # Thêm
    MAX_LEN = 50  # Thêm

    # --- Load, clean, tokenize và filter ---
    df_filtered, en_tokenizer, vi_tokenizer = load_tokenize_and_filter(
        FILE_PATH, max_token_diff=MAX_DIFF, min_len=MIN_LEN, max_len=MAX_LEN
    )

    # --- Build vocab từ token đã tokenize ---
    vocab_transform = build_vocab_from_tokens(df_filtered, en_tokenizer, vi_tokenizer)  # FIX: chỉ nhận 1 giá trị

    # --- Save pickle ---
    save_pickle(df_filtered, TOKEN_PICKLE, VOCAB_PICKLE, vocab_transform)