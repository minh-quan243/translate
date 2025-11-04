
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import tempfile
import os
from io import BytesIO
from typing import List, Tuple
import zipfile
# Optional parsers
try:
    import docx
except Exception:
    docx = None

try:
    import pypdf
except Exception:
    pypdf = None

st.set_page_config(page_title="VN↔EN Translator (Local)", layout="wide")

# ---------- Utilities ----------
def chunk_text(text: str, max_chars: int = 3500) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 <= max_chars:
            cur += (("\n\n" if cur else "") + para)
        else:
            if cur:
                chunks.append(cur); cur = ""
            for sent in para.split(". "):
                if len(cur) + len(sent) + 2 <= max_chars:
                    cur += (("" if not cur else ". ") + sent)
                else:
                    if cur: chunks.append(cur)
                    cur = sent
            if cur:
                chunks.append(cur); cur = ""
    if cur:
        chunks.append(cur)
    return chunks

@st.cache_resource(show_spinner=False)
def load_models():
    en_vi_name = "Helsinki-NLP/opus-mt-en-vi"
    vi_en_name = "Helsinki-NLP/opus-mt-vi-en"
    en_vi_tok = AutoTokenizer.from_pretrained(en_vi_name)
    en_vi_model = AutoModelForSeq2SeqLM.from_pretrained(en_vi_name)
    vi_en_tok = AutoTokenizer.from_pretrained(vi_en_name)
    vi_en_model = AutoModelForSeq2SeqLM.from_pretrained(vi_en_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    en_vi_model.to(device)
    vi_en_model.to(device)
    return (en_vi_tok, en_vi_model, vi_en_tok, vi_en_model, device)

def translate_text(text: str, direction: str, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
    en_vi_tok, en_vi_model, vi_en_tok, vi_en_model, device = load_models()
    if direction == "auto":
        lang = detect(text) or "vi"
        direction = "vi->en" if lang.startswith("vi") else "en->vi"
    tok = vi_en_tok if direction == "vi->en" else en_vi_tok
    model = vi_en_model if direction == "vi->en" else en_vi_model

    outputs = []
    for chunk in chunk_text(text):
        inputs = tok(chunk, return_tensors="pt", truncation=True).to(device)
        gen = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0.0,
            temperature=float(temperature) if temperature > 0 else None,
            num_beams=4 if temperature == 0 else 1,
        )
        out = tok.batch_decode(gen, skip_special_tokens=True)[0]
        outputs.append(out)
    return "\n\n".join(outputs)

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def read_docx(file_bytes: bytes) -> str:
    if docx is None:
        raise RuntimeError("Chưa cài python-docx")
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        d = docx.Document(tmp.name)
        os.unlink(tmp.name)
    return "\n".join(p.text for p in d.paragraphs)

def read_pdf(file_bytes: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("Chưa cài pypdf")
    reader = pypdf.PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def load_file_to_text(upload) -> Tuple[str, str]:
    name = upload.name
    data = upload.read()
    ext = os.path.splitext(name.lower())[1]
    if ext == ".txt":
        return read_txt(data), name
    elif ext == ".docx":
        return read_docx(data), name
    elif ext == ".pdf":
        return read_pdf(data), name
    else:
        raise RuntimeError(f"Định dạng chưa hỗ trợ: {ext} (hỗ trợ: .txt, .docx, .pdf)")

def make_zip(named_texts: List[Tuple[str, str]]) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for text, out_name in named_texts:
            zf.writestr(out_name, text)
    return bio.getvalue()

# ---------- UI ----------
st.title("VN–EN Translator — chạy local (VSCode/Streamlit)")
st.caption("Model: Helsinki-NLP opus-mt (offline). Hỗ trợ dán văn bản hoặc kéo–thả tệp .txt/.docx/.pdf.")

with st.sidebar:
    st.subheader("Thiết lập")
    direction = st.radio("Hướng dịch", ["auto", "vi->en", "en->vi"], index=0)
    temperature = st.slider("Độ sáng tạo (temperature)", 0.0, 1.0, 0.0, 0.1)
    max_new = st.slider("max_new_tokens", 64, 1024, 512, 32)
    st.markdown("---")
    st.markdown("💡 Để dịch sát nghĩa, để temperature = 0 và bật beam search (mặc định).")

tab1, tab2 = st.tabs(["Dán văn bản", "Dịch tệp (1 hoặc nhiều)"])

with tab1:
    text = st.text_area("Dán văn bản vào đây", height=220, placeholder="Thả chữ vào đây...")
    if st.button("Dịch văn bản"):
        if text.strip():
            with st.spinner("Đang dịch..."):
                result = translate_text(text, direction=direction, temperature=temperature, max_new_tokens=max_new)
            st.success("Xong!")
            st.text_area("Bản dịch", result, height=220)
            st.download_button("Tải bản dịch (.txt)", data=result, file_name="translation.txt")
        else:
            st.warning("Vui lòng nhập văn bản.")

with tab2:
    uploads = st.file_uploader("Kéo–thả 1 hoặc nhiều tệp (.txt, .docx, .pdf)", type=["txt","docx","pdf"], accept_multiple_files=True)
    if st.button("Dịch tệp đã chọn"):
        if not uploads:
            st.warning("Hãy chọn ít nhất 1 tệp.")
        else:
            results = []
            with st.spinner("Đang dịch..."):
                for up in uploads:
                    try:
                        content, in_name = load_file_to_text(up)
                        translated = translate_text(content, direction=direction, temperature=temperature, max_new_tokens=max_new)
                        base, ext = os.path.splitext(in_name)
                        out_name = f"{base}_translated.txt"
                        results.append((translated, out_name))
                    except Exception as e:
                        st.error(f"Lỗi với {up.name}: {e}")
            if results:
                if len(results) == 1:
                    st.success("Xong! Tải file bên dưới.")
                    st.download_button("Tải bản dịch (.txt)", data=results[0][0], file_name=results[0][1])
                else:
                    # zip many
                    import zipfile, io
                    bio = io.BytesIO()
                    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
                        for text, out_name in results:
                            zf.writestr(out_name, text)
                    st.success("Xong! Tải file .zip bên dưới.")
                    st.download_button("Tải tất cả (.zip)", data=bio.getvalue(), file_name="translations.zip")