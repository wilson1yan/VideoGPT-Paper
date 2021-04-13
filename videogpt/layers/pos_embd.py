import torch
import torch.nn as nn

class BroadcastPosEmbedND(nn.Module):
    def __init__(self, shape, embd_dim):
        super().__init__()
        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embd_dim = embd_dim

        assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
        self.emb = nn.ModuleDict(
            {f'd_{i}': nn.Embedding(num_embeddings=shape[i], embedding_dim=embd_dim // n_dim)
             for i in range(n_dim)}
        )

        for i in range(n_dim):
            self.emb[f'd_{i}'].weight.data.normal_(std=0.01)

    def forward(self, x):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}'](torch.arange(self.shape[i]).to(x.device))
            # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
            e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
            e = e.expand(x.shape[0], *self.shape, -1)
            embs.append(e)

        embs = torch.cat(embs, dim=-1)
        return embs

