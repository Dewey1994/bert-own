import torch.nn as nn
from .single import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self,head,hidden_size,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % head == 0
        self.d_k = hidden_size // head
        self.head = head

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(3)])
        self.output_layers = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        self.attention = Attention()

    def forward(self,query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2) for l,x in zip(self.linear_layers,(query,key,value))]
        x, attn = self.attention(query,key,value,mask=mask,dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        return self.output_layers(x)