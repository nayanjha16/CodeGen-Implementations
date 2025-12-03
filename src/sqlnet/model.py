import torch
import torch.nn as nn
from typing import Tuple


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hid_dim: int = 256, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx) #We used a standard trainable embedding layer inside PyTorch
        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)

    def forward(self, src_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        src_ids: (batch, src_len)
        returns: encoder_outputs (batch, src_len, 2*hid_dim), (h, c)
        """
        emb = self.embedding(src_ids)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        decoder_hidden: (batch, dec_hid_dim)
        encoder_outputs: (batch, src_len, 2*enc_hid_dim)
        returns: attention weights (batch, src_len)
        """
        src_len = encoder_outputs.size(1)
        # repeat decoder hidden for each position
        hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)  # (batch, src_len)
        return torch.softmax(scores, dim=1)


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attn = Attention(enc_hid_dim, dec_hid_dim)
        self.lstm = nn.LSTM(emb_dim + enc_hid_dim * 2, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim * 2 + emb_dim, vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input_token: (batch,) one token index
        hidden, cell: (1, batch, dec_hid_dim)
        encoder_outputs: (batch, src_len, 2*enc_hid_dim)
        """
        input_token = input_token.unsqueeze(1)  # (batch, 1)
        emb = self.embedding(input_token)       # (batch, 1, emb_dim)
        dec_hidden = hidden[-1]                 # (batch, dec_hid_dim)
        attn_weights = self.attn(dec_hidden, encoder_outputs)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, 2*enc_hid_dim)

        rnn_input = torch.cat((emb, context), dim=2)  # (batch, 1, emb_dim + 2*enc_hid_dim)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = output.squeeze(1)       # (batch, dec_hid_dim)
        context = context.squeeze(1)     # (batch, 2*enc_hid_dim)
        emb = emb.squeeze(1)             # (batch, emb_dim)

        prediction = self.fc_out(torch.cat((output, context, emb), dim=1))  # (batch, vocab_size)
        return prediction, hidden, cell


class SQLNet(nn.Module):
    """
    Simplified SQLNet-style seq2seq with attention.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, emb_dim: int = 256, hid_dim: int = 256):
        super().__init__()
        self.encoder = EncoderLSTM(vocab_size, emb_dim, hid_dim, pad_idx)
        self.decoder = DecoderLSTM(vocab_size, emb_dim, hid_dim, hid_dim, pad_idx)
        self.pad_idx = pad_idx

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor, teacher_forcing: float = 0.5) -> torch.Tensor:
        """
        src_ids: (batch, src_len)
        tgt_ids: (batch, tgt_len)
        returns: logits (batch, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt_ids.size()
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src_ids.device)

        encoder_outputs, (h, c) = self.encoder(src_ids)
        # initialize decoder hidden and cell as zeros (could also map from encoder)
        dec_hidden = torch.zeros(1, batch_size, self.decoder.lstm.hidden_size, device=src_ids.device)
        dec_cell = torch.zeros_like(dec_hidden)

        # first input token = first token in target (usually <pad> or <bos>)
        input_token = tgt_ids[:, 0]

        for t in range(1, tgt_len):
            logits, dec_hidden, dec_cell = self.decoder(input_token, dec_hidden, dec_cell, encoder_outputs)
            outputs[:, t, :] = logits
            teacher = torch.rand(1).item() < teacher_forcing
            top1 = logits.argmax(dim=1)
            input_token = tgt_ids[:, t] if teacher else top1

        return outputs

    def generate(
        self,
        src_ids: torch.Tensor,
        max_len: int,
        start_token_id: int,
        end_token_id: int,
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.
        src_ids: (1, src_len)
        returns: generated_ids (1, <= max_len)
        """
        self.eval()
        device = src_ids.device
        encoder_outputs, (h, c) = self.encoder(src_ids)
        batch_size = src_ids.size(0)
        dec_hidden = torch.zeros(1, batch_size, self.decoder.lstm.hidden_size, device=device)
        dec_cell = torch.zeros_like(dec_hidden)

        input_token = torch.tensor([start_token_id], device=device)
        generated = [start_token_id]

        for _ in range(max_len - 1):
            logits, dec_hidden, dec_cell = self.decoder(input_token, dec_hidden, dec_cell, encoder_outputs)
            next_token = logits.argmax(dim=1)
            generated.append(next_token.item())
            if next_token.item() == end_token_id:
                break
            input_token = next_token

        return torch.tensor([generated], device=device)
