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
MAX_EXTRA_TOKENS = 10  # Tăng lên để xử lý câu dài
BEAM_WIDTH = 5  # Tăng beam width
LENGTH_PENALTY = 0.7
REPETITION_PENALTY = 1.5  # Thêm penalty chống lặp
NO_REPEAT_NGRAM_SIZE = 3  # Chặn n-gram lặp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"✅ Device: {device}")


# =================== TOKENIZER FIXED ===================
def clean_text(text):
    """Giống hệt với training!"""
    # Giữ lại . ? ! cho cấu trúc câu
    text = re.sub(r'[^\w\s\.\?\!]', '', text)

    # Chuẩn hóa số - GIỐNG TRAINING
    text = re.sub(r'\b\d+\b', '<num>', text)

    # Chuẩn hóa từ viết tắt thông dụng - GIỐNG TRAINING
    contractions = {
        "i'm": "im", "i'll": "ill", "i've": "ive",
        "don't": "dont", "can't": "cant", "won't": "wont"
    }
    for cont, replacement in contractions.items():
        text = re.sub(r'\b' + cont + r'\b', replacement, text)

    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def en_tokenizer(text: str):
    """GIỐNG HỆT TRAINING!"""
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def vi_tokenizer(text: str):
    return word_tokenize(text)


def has_repeated_ngram(sequence, ngram, ngram_size):
    """Kiểm tra n-gram lặp trong sequence"""
    if len(sequence) < ngram_size:
        return False
    seq_tuple = tuple(sequence)
    for i in range(len(seq_tuple) - ngram_size + 1):
        if seq_tuple[i:i + ngram_size] == ngram:
            return True
    return False


# =================== TRANSLATE IMPROVED ===================
def translate_sentence_beam_search(model, sentence, vocab_transform, device,
                                   beam_width=BEAM_WIDTH, max_extra=MAX_EXTRA_TOKENS,
                                   length_penalty=LENGTH_PENALTY, repetition_penalty=REPETITION_PENALTY,
                                   no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE):
    """Beam search với chống lặp từ"""
    model.eval()

    # CLEAN TEXT GIỐNG TRAINING
    cleaned_sentence = clean_text(sentence)
    tokens = en_tokenizer(cleaned_sentence)

    vocab_en = vocab_transform['en']['stoi']
    vocab_vi = vocab_transform['vi']['stoi']
    itos_vi = vocab_transform['vi']['itos']

    bos_en = vocab_en.get("<bos>", vocab_en.get("<sos>"))
    eos_en = vocab_en.get("<eos>", vocab_en.get("</s>"))
    bos_vi = vocab_vi.get("<bos>", vocab_vi.get("<sos>"))
    eos_vi = vocab_vi.get("<eos>", vocab_vi.get("</s>"))
    pad_vi = vocab_vi.get("<pad>")
    unk_vi = vocab_vi.get("<unk>")

    # Tạo source sequence GIỐNG TRAINING
    src_indices = [vocab_en.get(tok, vocab_en["<unk>"]) for tok in tokens]
    src_indices = [bos_en] + src_indices + [eos_en]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)  # [seq_len, 1]

    with torch.no_grad():
        seq_len = src_tensor.shape[0]
        max_len = min(seq_len + max_extra, 50)  # Giới hạn tối đa

        hidden, cell, _ = model.encoder(src_tensor)
        sequences = [[[], 0.0, hidden, cell]]  # [tokens, score, hidden, cell]

        for step in range(max_len):
            all_candidates = []

            for seq, score, hidden, cell in sequences:
                # Nếu sequence đã kết thúc, giữ nguyên
                if seq and seq[-1] == eos_vi:
                    all_candidates.append([seq, score, hidden, cell])
                    continue

                input_idx = seq[-1] if seq else bos_vi
                input_tensor = torch.tensor([input_idx]).to(device)
                output, hidden_new, cell_new = model.decoder(input_tensor, hidden, cell)
                log_probs = torch.log_softmax(output, dim=1)

                # 🛑 ÁP DỤNG REPETITION PENALTY
                if repetition_penalty > 1.0 and seq:
                    for token_id in set(seq[-no_repeat_ngram_size:]):
                        log_probs[0, token_id] /= repetition_penalty

                topk_log_probs, topk_idx = log_probs.topk(beam_width * 2)  # Lấy nhiều hơn để filter

                for i in range(beam_width * 2):
                    token = topk_idx[0, i].item()

                    # 🛑 CHẶN N-GRAM LẶP
                    if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                        ngram = tuple(seq[-(no_repeat_ngram_size - 1):] + [token])
                        if has_repeated_ngram(seq, ngram, no_repeat_ngram_size):
                            continue

                    candidate_seq = seq + [token]
                    new_score = score + topk_log_probs[0, i].item()

                    # Length normalization
                    lp = len(candidate_seq) ** length_penalty
                    length_norm_score = new_score / lp if lp > 0 else new_score

                    all_candidates.append([candidate_seq, length_norm_score, hidden_new, cell_new])

            # Chọn top k, ưu tiên sequence không lặp
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)

            # Lọc sequences trùng lặp (theo tokens)
            unique_sequences = []
            seen_tokens = set()
            for seq in ordered:
                seq_tokens = tuple(seq[0])
                if seq_tokens not in seen_tokens:
                    unique_sequences.append(seq)
                    seen_tokens.add(seq_tokens)
                if len(unique_sequences) >= beam_width:
                    break

            sequences = unique_sequences[:beam_width]

            # Stop nếu tất cả sequences đều kết thúc
            if all(seq[0][-1] == eos_vi for seq in sequences):
                break

        # Chọn sequence tốt nhất
        best_seq = sequences[0][0]

        # Convert tokens to words, filter special tokens
        translation_tokens = []
        for token_id in best_seq:
            if token_id == eos_vi:
                break
            if token_id not in [bos_vi, eos_vi, pad_vi, unk_vi]:
                translation_tokens.append(itos_vi[token_id])

        return " ".join(translation_tokens) if translation_tokens else "Không thể dịch"


def translate_sentence_greedy(model, sentence, vocab_transform, device):
    """Fallback: Greedy decoding đơn giản"""
    model.eval()

    cleaned_sentence = clean_text(sentence)
    tokens = en_tokenizer(cleaned_sentence)

    vocab_en = vocab_transform['en']['stoi']
    vocab_vi = vocab_transform['vi']['stoi']
    itos_vi = vocab_transform['vi']['itos']

    bos_en = vocab_en.get("<bos>")
    eos_en = vocab_en.get("<eos>")
    bos_vi = vocab_vi.get("<bos>")
    eos_vi = vocab_vi.get("<eos>")

    src_indices = [vocab_en.get(tok, vocab_en["<unk>"]) for tok in tokens]
    src_indices = [bos_en] + src_indices + [eos_en]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell, _ = model.encoder(src_tensor)

        input_idx = bos_vi
        max_len = len(src_indices) + 10
        translated_tokens = []

        for _ in range(max_len):
            input_tensor = torch.tensor([input_idx]).to(device)
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)

            top1 = output.argmax(1).item()
            if top1 == eos_vi:
                break

            if top1 not in [bos_vi, eos_vi]:
                translated_tokens.append(itos_vi[top1])

            input_idx = top1

        return " ".join(translated_tokens) if translated_tokens else "Không thể dịch"


# =================== STREAMLIT APP IMPROVED ===================
st.title("🌍 English → Vietnamese Translation")
st.write("Nhập câu tiếng Anh để dịch sang tiếng Việt bằng mô hình Seq2Seq với Beam Search + Chống lặp từ.")

# Sidebar for settings
st.sidebar.header("⚙️ Cài đặt dịch")
beam_width = st.sidebar.slider("Beam Width", min_value=1, max_value=10, value=BEAM_WIDTH)
use_beam_search = st.sidebar.checkbox("Sử dụng Beam Search", value=True)
show_debug = st.sidebar.checkbox("Hiển thị thông tin debug", value=False)

# Load vocab
try:
    with open(r"D:\Quân\project\translate\data\vocab_transform.pkl", "rb") as f:
        vocab_transform = pickle.load(f)
    st.success("✅ Đã tải vocab_transform.pkl")

    if show_debug:
        st.write(f"📊 Vocab sizes → EN: {len(vocab_transform['en']['stoi'])} | VI: {len(vocab_transform['vi']['stoi'])}")

except Exception as e:
    st.error(f"❌ Lỗi khi tải vocab_transform.pkl: {e}")
    st.stop()

# Khởi tạo mô hình
try:
    enc = EncoderLNRes(len(vocab_transform['en']['stoi']), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = DecoderLNRes(len(vocab_transform['vi']['stoi']), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2SeqLNRes(enc, dec, device).to(device)
except Exception as e:
    st.error(f"❌ Lỗi khi khởi tạo mô hình: {e}")
    st.stop()

# Load checkpoint
try:
    checkpoint_path = r"D:\Quân\project\translate\checkpoint\checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Xử lý cả hai định dạng checkpoint
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    st.success("✅ Mô hình đã được load thành công!")

    if show_debug and "epoch" in checkpoint:
        st.write(f"📈 Checkpoint từ epoch: {checkpoint['epoch']}")

except Exception as e:
    st.error(f"❌ Lỗi khi tải checkpoint: {e}")
    st.stop()

# Input text
input_text = st.text_area("✏️ Nhập câu tiếng Anh:",
                          placeholder="Ví dụ: Hello, how are you? I'm fine, thank you.",
                          height=100)

if st.button("🚀 Dịch Ngay"):
    if input_text.strip():
        with st.spinner("🔁 Đang dịch..."):
            try:
                if use_beam_search:
                    translation = translate_sentence_beam_search(
                        model, input_text, vocab_transform, device, beam_width=beam_width
                    )
                    method = "Beam Search"
                else:
                    translation = translate_sentence_greedy(model, input_text, vocab_transform, device)
                    method = "Greedy"

                if translation and translation != "Không thể dịch":
                    st.success(f"**✅ Bản dịch ({method}):**")
                    st.info(f"**{translation}**")

                    if show_debug:
                        st.write("---")
                        st.write("**🐛 Debug Info:**")
                        st.write(f"- Phương pháp: {method}")
                        st.write(f"- Beam width: {beam_width if use_beam_search else 'N/A'}")
                        st.write(f"- Input length: {len(input_text.split())} từ")
                        st.write(f"- Output length: {len(translation.split())} từ")
                else:
                    st.warning("⚠️ Không tạo được bản dịch. Thử câu khác hoặc điều chỉnh cài đặt.")

            except Exception as e:
                st.error(f"❌ Lỗi khi dịch: {e}")
                # Fallback to greedy
                st.info("🔄 Thử dùng Greedy decoding...")
                try:
                    translation = translate_sentence_greedy(model, input_text, vocab_transform, device)
                    if translation:
                        st.success("**✅ Bản dịch (Greedy Fallback):**")
                        st.info(f"**{translation}**")
                except:
                    st.error("❌ Không thể dịch với cả hai phương pháp.")

    else:
        st.warning("❗ Vui lòng nhập câu để dịch.")

# Example sentences
st.sidebar.header("📝 Ví dụ")
examples = [
    "Hello, how are you?",
    "I love programming and artificial intelligence.",
    "What is your name?",
    "The weather is beautiful today.",
    "Can you help me with this problem?"
]

for example in examples:
    if st.sidebar.button(example, key=example):
        st.experimental_set_query_params(text=example)
        st.experimental_rerun()

# Check if there's text in URL parameters
query_params = st.experimental_get_query_params()
if "text" in query_params:
    default_text = query_params["text"][0]
else:
    default_text = ""

if default_text:
    st.text_area("✏️ Nhập câu tiếng Anh:", value=default_text, height=100)