import torch.nn as nn

class BERT(nn.Module):
    def __init__(self,vocab_size,hidden=768,n_layers=12,attn_heads=12,dropout=0.1):
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.tranformer_blocks = nn.ModuleList([
            TransformerBlock(hidden,attn_heads,hidden*4,dropout) for _ in range(n_layers)])

