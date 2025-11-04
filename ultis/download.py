from datasets import load_dataset
import pandas as pd

# Dùng dataset opus100 (có eng-vie)
dataset = load_dataset("opus100", "en-vi")

df = pd.DataFrame(dataset["train"]["translation"])
df.to_csv("tatoeba_en_vi.csv", index=False, encoding="utf-8")
print(df.head())
