import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from rouge import Rouge
from model import Encoder, Decoder, Seq2Seq
from dataloader import create_dataloaders

# ===============================================================
# ⚙️ CONFIG
# ===============================================================
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ids_to_text(ids, vocab_itos):
    return " ".join([
        vocab_itos[i] for i in ids
        if i in vocab_itos and vocab_itos[i] not in ["<pad>", "<sos>", "<eos>"]
    ])

def evaluate_loss(model, iterator, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(iterator, desc="Evaluating Loss"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            loss = criterion(output[1:].view(-1, output_dim), tgt[1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(iterator)

def evaluate_metrics(model, iterator, vocab_transform):
    from sklearn.metrics import precision_score, recall_score, f1_score
    rouge = Rouge()
    vi_itos = {v: k for k, v in vocab_transform["vi"]["stoi"].items()}
    rouge_scores, precisions, recalls, f1s = [], [], [], []

    model.eval()
    with torch.no_grad():
        for src, tgt in tqdm(iterator, desc="Evaluating Metrics"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            preds = output.argmax(2).detach().cpu().numpy()
            tgts = tgt.detach().cpu().numpy()

            for i in range(preds.shape[1]):
                pred_text = ids_to_text(preds[:, i], vi_itos)
                tgt_text = ids_to_text(tgts[:, i], vi_itos)
                try:
                    rouge_scores.append(rouge.get_scores(pred_text, tgt_text)[0]["rouge-l"]["f"])
                    p, r, f = rouge.get_scores(pred_text, tgt_text)[0]["rouge-l"].values()
                    precisions.append(p)
                    recalls.append(r)
                    f1s.append(f)
                except:
                    continue

    return {
        "ROUGE-L": sum(rouge_scores)/len(rouge_scores),
        "Precision": sum(precisions)/len(precisions),
        "Recall": sum(recalls)/len(recalls),
        "F1": sum(f1s)/len(f1s)
    }

if __name__ == "__main__":
    # ===============================================================
    # 🧠 LOAD VOCAB + DATALOADER
    # ===============================================================
    with open("vocab_transform.pkl", "rb") as f:
        vocab_transform = pickle.load(f)

    SRC_VOCAB_SIZE = len(vocab_transform["en"]["stoi"])
    TGT_VOCAB_SIZE = len(vocab_transform["vi"]["stoi"])
    SRC_PAD_IDX = vocab_transform["en"]["stoi"]["<pad>"]
    TGT_PAD_IDX = vocab_transform["vi"]["stoi"]["<pad>"]

    _, _, test_loader = create_dataloaders(BATCH_SIZE, num_workers=NUM_WORKERS)
    print("✅ Test DataLoader loaded!")

    # ===============================================================
    # 🧩 LOAD MODEL
    # ===============================================================
    enc = Encoder(SRC_VOCAB_SIZE, 256, 512, 1, 0.5)
    dec = Decoder(TGT_VOCAB_SIZE, 256, 512, 1, 0.5)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    checkpoint = torch.load("checkpoint_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Model checkpoint loaded!")

    # ===============================================================
    # 🧮 TEST LOSS
    # ===============================================================
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    test_loss = evaluate_loss(model, test_loader, criterion)
    print(f"📉 Test Loss: {test_loss:.4f}")

    # ===============================================================
    # 📊 METRICS
    # ===============================================================
    metrics = evaluate_metrics(model, test_loader, vocab_transform)
    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n✅ Evaluation complete!")
