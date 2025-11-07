import math
import torch
from torch import nn, Tensor
from torch.nn import Transformer
import torch.nn.functional as F

PAD_IDX = 1

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Greedy decode for inference
def greedy_decode(model, src, src_mask, max_len, start_symbol, EOS_IDX=3):
    src = src
    src_mask = src_mask
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
    for i in range(max_len-1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

@torch.no_grad()
def beam_search_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    EOS_IDX=3,
    beam_size=5,
    length_penalty=0.7,
):
    """
    Beam search decode ổn định, tránh None output.
    """

    device = src.device
    memory = model.encode(src, src_mask)

    # Beam khởi tạo
    beam = [(torch.tensor([[start_symbol]], dtype=torch.long, device=device), 0.0)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beam:
            # Nếu kết thúc câu rồi thì giữ nguyên
            if seq[-1].item() == EOS_IDX:
                completed.append((seq, score))
                continue

            tgt_mask = generate_square_subsequent_mask(seq.size(0)).to(device)
            out = model.decode(seq, memory, tgt_mask)
            out = out.transpose(0, 1)
            logits = model.generator(out[:, -1])
            log_probs = F.log_softmax(logits, dim=-1)

            # Lấy top beam_size token tiếp theo
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)
            for k in range(beam_size):
                next_token = topk_indices[0, k].item()
                new_seq = torch.cat(
                    [seq, torch.tensor([[next_token]], dtype=torch.long, device=device)],
                    dim=0,
                )
                new_score = score + topk_log_probs[0, k].item()
                candidates.append((new_seq, new_score))

        if not candidates:
            break

        # Giữ lại top beam_size ứng viên tốt nhất
        ordered = sorted(
            candidates,
            key=lambda t: t[1] / ((len(t[0]) ** length_penalty) + 1e-6),
            reverse=True,
        )
        beam = ordered[:beam_size]

        # Dừng khi tất cả beam đều kết thúc
        if all(seq[-1].item() == EOS_IDX for seq, _ in beam):
            completed.extend(beam)
            break

    # Nếu không có câu hoàn chỉnh → lấy beam hiện tại
    if not completed:
        completed = beam

    # Chọn câu có score cao nhất
    best_seq, best_score = sorted(
        completed,
        key=lambda t: t[1] / ((len(t[0]) ** length_penalty) + 1e-6),
        reverse=True,
    )[0]

    return best_seq

def translate(model, src_sentence, text_transform, vocab_transform, BOS_IDX=2, EOS_IDX=3):
    model.eval()
    src = text_transform['en'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens+5, start_symbol=BOS_IDX, EOS_IDX=EOS_IDX).flatten()
    return " ".join(vocab_transform['vi'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def translate_beam(model, src_sentence, text_transform, vocab_transform, BOS_IDX=2, EOS_IDX=3, beam_size=5):
    model.eval()
    device = next(model.parameters()).device
    src = text_transform['en'](src_sentence).view(-1, 1).to(device)
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool, device=device)
    tgt_tokens = beam_search_decode(
        model, src, src_mask,
        max_len=num_tokens + 10,
        start_symbol=BOS_IDX,
        EOS_IDX=EOS_IDX,
        beam_size=beam_size
    ).flatten()

    tokens = vocab_transform['vi'].lookup_tokens(list(tgt_tokens.cpu().numpy()))
    print("🔍 Tokens:", tokens)  # debug
    translated = " ".join(tokens).replace("<bos>", "").replace("<eos>", "").strip()
    if not translated:
        return "(⚠️ Beam output empty — model may output only <eos>)"
    return translated
