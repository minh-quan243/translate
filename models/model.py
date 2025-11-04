import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Encoder với LayerNorm + Residual + Linear Projection
# -----------------------------
class EncoderLNRes(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding2hidden = nn.Linear(emb_dim, hid_dim)  # projection
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim)

    def forward(self, src):
        # src: [seq_len, batch_size]
        embedded = self.dropout(self.embedding(src))          # [seq_len, batch, emb_dim]
        embedded_proj = self.embedding2hidden(embedded)      # [seq_len, batch, hid_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # LayerNorm + Residual
        outputs = self.layernorm(outputs + embedded_proj)
        return hidden, cell, outputs  # outputs dùng nếu muốn attention

# -----------------------------
# Decoder với LayerNorm + Residual + Linear Projection
# -----------------------------
class DecoderLNRes(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding2hidden = nn.Linear(emb_dim, hid_dim)  # projection
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))          # [1, batch, emb_dim]
        embedded_proj = self.embedding2hidden(embedded)         # [1, batch, hid_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # LayerNorm + Residual
        output = self.layernorm(output + embedded_proj)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# -----------------------------
# Seq2Seq
# -----------------------------
class Seq2SeqLNRes(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        tgt_len = tgt.shape[0]
        batch_size = src.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        hidden, cell, _ = self.encoder(src)
        input = tgt[0, :]

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

    def beam_search_decode(self, src, sos_idx=2, eos_idx=3, max_extra=5, beam_width=3, length_penalty=0.7):
        """
        src: [seq_len, batch_size] tensor
        sos_idx, eos_idx: chỉ số token đặc biệt
        max_extra: cho phép dài hơn input bao nhiêu token
        length_penalty: alpha trong length normalization
        """
        self.eval()
        with torch.no_grad():
            seq_len = src.shape[0]
            max_len = seq_len + max_extra

            hidden, cell, _ = self.encoder(src)
            batch_size = src.shape[1]

            sequences = [[list(), 0.0, hidden, cell]]  # [tokens, score, hidden, cell]

            for _ in range(max_len):
                all_candidates = []
                for seq, score, hidden, cell in sequences:
                    input_idx = seq[-1] if seq else sos_idx
                    input_tensor = torch.tensor([input_idx]).to(self.device)
                    output, hidden_new, cell_new = self.decoder(input_tensor, hidden, cell)
                    log_probs = torch.log_softmax(output, dim=1)
                    topk_log_probs, topk_idx = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        candidate_seq = seq + [topk_idx[0, i].item()]
                        # Áp dụng length penalty
                        length_norm_score = (score + topk_log_probs[0, i].item()) / (
                                    len(candidate_seq) ** length_penalty)
                        all_candidates.append([candidate_seq, length_norm_score, hidden_new, cell_new])

                # Chọn top k sequences
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

                # Stop nếu tất cả top sequences đều kết thúc eos
                if all(seq[0][-1] == eos_idx for seq in sequences):
                    break

            return sequences[0][0]  # best sequence