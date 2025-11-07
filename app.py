import streamlit as st
import torch
import os
from models.model import Seq2SeqTransformer, translate, translate_beam

# -----------------------------
# ⚙️ Cấu hình
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"D:\Quân\project\translate\checkpoint\viEn_transformer.pth"
VOCAB_PATH = r"D:\Quân\project\translate\checkpoint\vocab_transform.pth"

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


# -----------------------------
# 🧠 Load vocab và tạo text_transform
# -----------------------------
@st.cache_resource
def load_vocab_and_transform():
    import pickle
    from torchtext.data.utils import get_tokenizer

    st.sidebar.info("🔄 Đang load vocab...")

    # Load vocab_transform.pth
    vocab_transform = torch.load(VOCAB_PATH)

    SRC_LANGUAGE = "en"
    TGT_LANGUAGE = "vi"

    # Tokenizers
    def vi_tokenizer(sentence):
        from underthesea import word_tokenize
        return word_tokenize(sentence)

    token_transform = {
        SRC_LANGUAGE: get_tokenizer("basic_english"),
        TGT_LANGUAGE: vi_tokenizer
    }

    # Sequential transform helper
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    def tensor_transform(token_ids):
        return torch.cat((
            torch.tensor([BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([EOS_IDX])
        ))

    # Build text_transform
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],
            vocab_transform[ln],
            tensor_transform
        )

    st.sidebar.success("✅ Vocab loaded!")
    return vocab_transform, text_transform


# -----------------------------
# 🧠 Load model
# -----------------------------
@st.cache_resource
def load_model_and_vocab():
    vocab_transform, text_transform = load_vocab_and_transform()

    SRC_VOCAB_SIZE = len(vocab_transform["en"])
    TGT_VOCAB_SIZE = len(vocab_transform["vi"])

    EMB_SIZE = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    FFN_HID_DIM = 512
    DROPOUT = 0.1

    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(checkpoint, strict=True)
            st.sidebar.success("✅ Model loaded successfully!")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Warning when loading weights: {e}")
            model.load_state_dict(checkpoint, strict=False)
    else:
        st.sidebar.error(f"❌ Không tìm thấy checkpoint: {MODEL_PATH}")

    model.eval()
    return model, vocab_transform, text_transform


model, vocab_transform, text_transform = load_model_and_vocab()

# -----------------------------
# ⚙️ Hàm dịch chính
# -----------------------------
def translate_text(text: str, model, text_transform, vocab_transform, decode_mode="greedy", beam_size=4):
    if decode_mode == "beam":
        return translate_beam(model, text, text_transform, vocab_transform, beam_size=beam_size)
    else:
        return translate(model, text, text_transform, vocab_transform)


# -----------------------------
# 🌐 Streamlit UI
# -----------------------------
st.set_page_config(page_title="🌍 Neural Machine Translation", layout="centered")
st.title("🌍 Neural Machine Translation App")
st.markdown("Dịch **Anh → Việt** bằng mô hình **Seq2Seq Transformer (PyTorch)**.")

# Ví dụ nhanh
st.sidebar.header("🧩 Ví dụ nhanh")
examples = [
    "Hello, how are you?",
    "I love programming and artificial intelligence.",
    "What is your name?",
    "The weather is beautiful today.",
    "Can you help me with this problem?"
]
for example in examples:
    if st.sidebar.button(example, key=example):
        st.session_state["text"] = example
        st.experimental_rerun()

# Chọn giải mã
st.sidebar.header("⚙️ Tuỳ chọn giải mã")
decode_mode = st.sidebar.radio(
    "Phương pháp giải mã:",
    ["Greedy decoding", "Beam search (4)", "Beam search (5)"],
    index=0
)

if "4" in decode_mode:
    mode = "beam"
    beam_size = 4
elif "5" in decode_mode:
    mode = "beam"
    beam_size = 5
else:
    mode = "greedy"
    beam_size = None

# Input text
text = st.text_area(
    "✏️ Nhập câu tiếng Anh:",
    value=st.session_state.get("text", ""),
    height=120
)

if st.button("🚀 Dịch ngay"):
    if text.strip():
        with st.spinner(f"⏳ Đang dịch bằng {decode_mode}..."):
            try:
                translated_text = translate_text(
                    text, model, text_transform, vocab_transform,
                    decode_mode=mode, beam_size=beam_size or 4
                )
                st.markdown("### 📘 Bản dịch:")
                st.success(translated_text)
            except Exception as e:
                st.error(f"⚠️ Lỗi khi dịch: {e}")
    else:
        st.warning("⚠️ Vui lòng nhập câu cần dịch!")

st.markdown("---")
st.caption("⚙️ Mô hình: Seq2Seq Transformer | Giải mã: Greedy / Beam Search | Framework: PyTorch + Streamlit")
