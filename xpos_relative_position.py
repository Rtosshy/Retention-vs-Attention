import torch
import torch.nn as nn

def fixed_pos_embedding(x):
    # x: (seq_len, dim)
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum(
            "i, j -> i j", # (seq_len,) * (dim,) -> (seq_len, dim)
            torch.arange(0, seq_len, dtype=torch.float),
            inv_freq
        ).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    # x: (batch, seq_len, dim)
    x1 = x[:, :, 0::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1) # (batch, seq_len, dim // 2, 2)
    if x.shape[-1] % 2 == 1:
        x2 = torch.concat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1)
    return x.flatten(-2)

def duplicate_interleave(m):
    # m: (dim0, dim1)
    dim0 = m.shape[0]
    m = m.view(-1, 1) # (dim0 * dim1, 1)
    m = m.repeat(1, 2) # (dim0 * dim1, 2)
    m = m.view(dim0, -1) # (dim0, dim1 * 2)
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return (x * cos[:, :x.shape[-1]]) + (rotate_every_two(x) * sin)[:, :, :x.shape[-1]]

class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super(XPOS, self).__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )
    
    def forward(self, x, offset=0, downscale=False):
        # x: (batch, seq_len, dim)
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
    
    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x

if __name__ == "__main__":
    x = torch.eye(4).unsqueeze(0)
    xpos = XPOS(4)
    x_rot = xpos(x)
    # apply reverse
    x_rot_rev = xpos.forward(x)

    print(x_rot @ x_rot_rev.transpose(-1, -2))