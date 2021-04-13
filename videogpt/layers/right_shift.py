import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.utils import shift_dim
from videogpt.layers.utils import tensor_slice

class RightShiftSequence(nn.Module):
    """
    Prepend SOS.
    """

    def __init__(self, embd_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.sos = nn.Parameter(torch.FloatTensor(embd_dim).normal_(std=0.02), requires_grad=True)

    def forward(self, x, decode_step, decode_idx):
        """Right shift sequence

        Args:
            x: a [batch, d1, d2, .., dn, depth] tensor
            decode_step
            decode_idx

        Returns:
            a [batch, d1, d2, ..., dn, depth] tensor, right shifted

        """
        if decode_step is not None and decode_step > 0:
            return x
        
        x_shape = list(x.shape)

        x = x.flatten(start_dim=1, end_dim=-2) # (b, seq_len,embd_dim)

        # prepend SOS, all tokens shifted right by one position
        # receptive field shifted left accordingly to prevent information leak
        sos = torch.ones(x_shape[0], 1, self.embd_dim, dtype=torch.float32).to(self.sos) * self.sos
        sos = sos.type_as(x)
        x = torch.cat([sos, x[:, :-1, :]], axis=1)  # (batch, seq, embd_dim)

        x = x.view(*x_shape)
        return x
