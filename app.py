import streamlit as st
import torch
import pickle
import re
from underthesea import word_tokenize
from models.model import EncoderLNRes, DecoderLNRes, Seq2SeqLNRes

# =================== CONFIG ===================
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1
DROPOUT = 0.3
MAX_EXTRA_TOKENS = 5  # số token "dư" cho phép
BEAM_WIDTH = 3
LENGTH_PENALTY = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"✅ Device: {device}")

# =================== TOKENIZER ===================
def en_tokenizer(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def vi_tokenizer(text: str):
    return word_tokenize(text)

# =================== TRANSLATE ===================
def translate_sentence(model, sentence, vocab_transform, device,
                       beam_width=BEAM_WIDTH, max_extra=MAX_EXTRA_TOKENS, length_penalty=LENGTH_PENALTY):
    model.eval()

    tokens = en_tokenizer(sentence)
    vocab_en = vocab_transform['en']['stoi']
    vocab_vi = vocab_transform['vi']['stoi']
    itos_vi = vocab_transform['vi']['itos']

    bos_en = vocab_en.get("<bos>", vocab_en.get("<sos>"))
    eos_en = vocab_en.get("<eos>", vocab_en.get("</s>"))
    bos_vi = vocab_vi.get("<bos>", vocab_vi.get("<sos>"))
    eos_vi = vocab_vi.get("<eos>", vocab_vi.get("</s>"))
    pad_vi = vocab_vi.get("<pad>")

    src_indices = [bos_en] + [vocab_en.get(tok, vocab_en["<unk>"]) for tok in tokens] + [eos_en]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)  # [seq_len, 1]

    # Beam search với length penalty và ràng buộc mềm
    with torch.no_grad():
        seq_len = src_tensor.shape[0]
        max_len = seq_len + max_extra

        hidden, cell, _ = model.encoder(src_tensor)
        sequences = [[[], 0.0, hidden, cell]]  # [tokens, score, hidden, cell]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden, cell in sequences:
                input_idx = seq[-1] if seq else bos_vi
                input_tensor = torch.tensor([input_idx]).to(device)
                output, hidden_new, cell_new = model.decoder(input_tensor, hidden, cell)
                log_probs = torch.log_softmax(output, dim=1)
                topk_log_probs, topk_idx = log_probs.topk(beam_width)

                for i in range(beam_width):
                    candidate_seq = seq + [topk_idx[0, i].item()]
                    norm_score = (score + topk_log_probs[0, i].item()) / (len(candidate_seq) ** length_penalty)
                    all_candidates.append([candidate_seq, norm_score, hidden_new, cell_new])

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop nếu tất cả sequences đều kết thúc eos
            if all(seq[0][-1] == eos_vi for seq in sequences):
                break

        best_seq = sequences[0][0]
        translation = [itos_vi[i] for i in best_seq if i not in [bos_vi, eos_vi, pad_vi]]
        return " ".join(translation)

# =================== STREAMLIT APP ===================
st.title("🌍 English → Vietnamese Translation")
st.write("Nhập câu tiếng Anh để dịch sang tiếng Việt bằng mô hình Seq2SeqLNRes với Beam Search + Length Penalty.")

# Load vocab
try:
    with open(r"D:\Quân\project\translate\data\vocab_transform.pkl", "rb") as f:
        vocab_transform = pickle.load(f)
    st.success("✅ Đã tải vocab_transform.pkl")
except Exception as e:
    st.error(f"❌ Lỗi khi tải vocab_transform.pkl: {e}")
    st.stop()

SRC_VOCAB_SIZE = len(vocab_transform['en']['stoi'])
TGT_VOCAB_SIZE = len(vocab_transform['vi']['stoi'])
st.write(f"Vocab sizes → EN: {SRC_VOCAB_SIZE} | VI: {TGT_VOCAB_SIZE}")

# Khởi tạo mô hình
try:
    enc = EncoderLNRes(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = DecoderLNRes(TGT_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2SeqLNRes(enc, dec, device).to(device)
except Exception as e:
    st.error(f"❌ Lỗi khi khởi tạo mô hình: {e}")
    st.stop()

# Load checkpoint
try:
    checkpoint = torch.load(r"D:\Quân\project\translate\checkpoint\checkpoint_best.pth", map_location=device)
    # Nếu checkpoint chỉ lưu state_dict trực tiếp:
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    st.success("✅ Mô hình đã được load thành công!")
except Exception as e:
    st.error(f"❌ Lỗi khi tải checkpoint: {e}")
    st.stop()

# Input text
input_text = st.text_area("✏️ Nhập câu tiếng Anh:", placeholder="Ví dụ: Hello, how are you?")

if st.button("Dịch"):
    if input_text.strip():
        with st.spinner("🔁 Đang dịch..."):
            translation = translate_sentence(model, input_text, vocab_transform, device)
            if translation:
                st.success("**Bản dịch (Tiếng Việt):**")
                st.write(translation)
            else:
                st.warning("⚠️ Không tạo được bản dịch.")
    else:
        st.warning("❗ Vui lòng nhập câu để dịch.")
