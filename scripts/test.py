import pickle
import pandas as pd
if __name__ == "__main__":
    with open(r"D:\Quân\project\translate\data\vocab_transform.pkl", "rb") as f:
        vocab_transform = pickle.load(f)
    with open(r"D:\Quân\project\translate\data\tokenized.pkl", "rb") as f:
        df = pickle.load(f)
    print(df.head(10))

    # Kiểm tra kiểu dữ liệu
    print(list(vocab_transform['en']['stoi'].keys())[:10])

    print(type(df.loc[0, "en_tok"]), df.loc[0, "en_tok"])
