import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Encoder với LayerNorm + Residual + Linear Projection FIXED
# -----------------------------
class EncoderLNRes(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding2hidden = nn.Linear(emb_dim, hid_dim)  # projection
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim)

    def forward(self, src):
        # src: [seq_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [seq_len, batch, emb_dim]

        # Project embedding to hidden dimension
        embedded_proj = self.embedding2hidden(embedded)  # [seq_len, batch, hid_dim]

        # LSTM forward
        outputs, (hidden, cell) = self.rnn(embedded)  # [seq_len, batch, hid_dim]

        # LayerNorm + Residual - FIXED: cộng outputs với embedded_proj
        # outputs: LSTM outputs, embedded_proj: linear projection của input
        outputs = self.layernorm(outputs + embedded_proj)  # [seq_len, batch, hid_dim]

        return hidden, cell, outputs


# -----------------------------
# Decoder với LayerNorm + Residual + Linear Projection FIXED
# -----------------------------
class DecoderLNRes(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding2hidden = nn.Linear(emb_dim, hid_dim)  # projection
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=False)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [n_layers, batch, hid_dim], cell: [n_layers, batch, hid_dim]

        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]
        embedded_proj = self.embedding2hidden(embedded)  # [1, batch, hid_dim]

        # LSTM forward - FIXED: sử dụng hidden và cell từ encoder
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: [1, batch, hid_dim]

        # LayerNorm + Residual
        output = self.layernorm(output + embedded_proj)  # [1, batch, hid_dim]

        # Prediction
        prediction = self.fc_out(output.squeeze(0))  # [batch, output_dim]

        return prediction, hidden, cell


# -----------------------------
# Seq2Seq FIXED
# -----------------------------
class Seq2SeqLNRes(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Kiểm tra compatibility
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have the same number of layers!"

    def forward(self, src, tgt_input, teacher_forcing_ratio=0.5):
        """
        src: [src_seq_len, batch_size] - source sequence
        tgt_input: [tgt_seq_len, batch_size] - target input (with <bos> but no <eos>)
        teacher_forcing_ratio: probability to use teacher forcing
        """
        tgt_len = tgt_input.shape[0]
        batch_size = src.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)

        # Encode source sequence
        hidden, cell, encoder_outputs = self.encoder(src)

        # First input to decoder is <bos> token
        # tgt_input[0] should be <bos> tokens for all batches
        input = tgt_input[0, :]  # [batch_size]

        for t in range(1, tgt_len):
            # Decoder forward
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Store prediction
            outputs[t] = output

            # Get top1 prediction
            top1 = output.argmax(1)

            # Teacher forcing: use true next token, else use predicted token
            if teacher_forcing_ratio > 0:
                input = tgt_input[t] if torch.rand(1).item() < teacher_forcing_ratio else top1
            else:
                input = top1

        return outputs  # [tgt_seq_len, batch_size, tgt_vocab_size]

    def beam_search_decode(self, src, sos_idx=2, eos_idx=3, max_extra=10, beam_width=5, length_penalty=0.7):
        """
        Beam search decoding for SINGLE sequence
        src: [seq_len, 1] tensor (batch_size=1)
        """
        self.eval()
        with torch.no_grad():
            seq_len = src.shape[0]
            max_len = seq_len + max_extra

            # Encode source - src should have batch_size=1
            hidden, cell, _ = self.encoder(src)

            # Initialize beam: [tokens, score, hidden, cell]
            sequences = [[[sos_idx], 0.0, hidden, cell]]

            for step in range(max_len):
                all_candidates = []

                for seq, score, hidden, cell in sequences:
                    # If sequence already ended with <eos>, keep as is
                    if seq[-1] == eos_idx:
                        all_candidates.append([seq, score, hidden, cell])
                        continue

                    # Last token as input
                    input_idx = seq[-1]
                    input_tensor = torch.tensor([input_idx]).to(self.device)

                    # Decoder forward
                    output, hidden_new, cell_new = self.decoder(input_tensor, hidden, cell)
                    log_probs = F.log_softmax(output, dim=1)  # [1, vocab_size]

                    # Get top k candidates
                    topk_log_probs, topk_idx = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        token = topk_idx[0, i].item()
                        log_prob = topk_log_probs[0, i].item()

                        candidate_seq = seq + [token]
                        new_score = score + log_prob

                        # Apply length penalty
                        lp = len(candidate_seq) ** length_penalty
                        length_norm_score = new_score / lp if lp > 0 else new_score

                        all_candidates.append([candidate_seq, length_norm_score, hidden_new, cell_new])

                # Sort and select top beam_width sequences
                ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                sequences = ordered[:beam_width]

                # Check if all top sequences ended
                if all(seq[0][-1] == eos_idx for seq in sequences):
                    break

            # Return best sequence (without <sos> token)
            best_sequence = sequences[0][0]
            return best_sequence[1:]  # Remove <sos>

    def translate_batch(self, src_batch, sos_idx=2, eos_idx=3, beam_width=3):
        """
        Translate a batch of sequences using greedy decoding (faster than beam search)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_batch.shape[1]
            max_len = src_batch.shape[0] + 20  # Allow extra length

            # Encode entire batch
            hidden, cell, _ = self.encoder(src_batch)

            # Start with <sos> for all sequences
            current_input = torch.full((batch_size,), sos_idx, dtype=torch.long, device=self.device)
            translations = [[] for _ in range(batch_size)]
            completed = [False] * batch_size

            for step in range(max_len):
                if all(completed):
                    break

                output, hidden, cell = self.decoder(current_input, hidden, cell)
                next_tokens = output.argmax(dim=1)  # Greedy decoding

                for i in range(batch_size):
                    if not completed[i]:
                        token = next_tokens[i].item()
                        translations[i].append(token)
                        if token == eos_idx:
                            completed[i] = True

                # Prepare next input
                current_input = next_tokens

            return translations


# -----------------------------
# Model Initialization Helper
# -----------------------------
def initialize_model(input_dim, output_dim, emb_dim, hid_dim, n_layers, dropout, device):
    """Helper function to initialize encoder, decoder and seq2seq model"""
    encoder = EncoderLNRes(input_dim, emb_dim, hid_dim, n_layers, dropout)
    decoder = DecoderLNRes(output_dim, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2SeqLNRes(encoder, decoder, device)

    return model.to(device)