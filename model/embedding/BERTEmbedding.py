import torch.nn as nn
from .token import TokenEmbedding
from .segment import SegmentEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(BERTEmbedding, self).__init__()

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, seq, seq_label):
        x = self.token(seq) + self.position(seq) + self.segment(seq_label)
        return self.dropout(x)
