import torch.nn as nn

from .attention.multi_head import MultiHeadAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads,feed_forward_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(head=attn_heads,hidden_size=hidden)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden, ff_hidden=feed_forward_hidden,dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x,_x,_x,mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)