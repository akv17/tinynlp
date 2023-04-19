import torch


class Model(torch.nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = 256
        self.src_emb = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size, padding_idx=0)
        self.dst_emb = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size, padding_idx=0)
        self.encoder = torch.nn.LSTM(
            input_size=self.emb_size,
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )
        self.decoder = torch.nn.LSTM(
            input_size=self.emb_size,
            hidden_size=512,
            bidirectional=False,
            batch_first=True
        )
        self.attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=1, batch_first=True)
        self.norm = torch.nn.LayerNorm(512)
        self.head = torch.nn.Linear(512, self.vocab_size)

    def forward(self, src, dst, attn_masks):
        x_src = self.src_emb(src)
        x_dst = self.dst_emb(dst)
        x_src, _ = self.encoder(x_src)
        x_dst, _ = self.decoder(x_dst)
        x_attn, x_attn_w = self.attn(x_dst, x_src, x_src, attn_mask=attn_masks)
        x_attn = x_dst + x_attn
        x_attn = self.norm(x_attn)
        x = self.head(x_attn)
        return x

    def encode(self, ids):
        x_src = self.src_emb(ids)
        x_src, _ = self.encoder(x_src)
        return x_src
    
    def decode(self, enc, ids, attn_mask, state):
        x_dst = self.dst_emb(ids)
        x_dst, state = self.decoder(x_dst, state)
        x_attn, x_attn_w = self.attn(x_dst, enc, enc, attn_mask=attn_mask)
        x_attn = x_dst + x_attn
        x_attn = self.norm(x_attn)
        x = self.head(x_attn)
        return x, state, x_attn_w
