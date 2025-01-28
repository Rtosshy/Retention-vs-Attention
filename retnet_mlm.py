import torch
import torch.nn as nn
import math
import random
from xpos_relative_position import XPOS
from icecream import ic



class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        self.xpos = XPOS(head_size)
    
    def forward(self, X, mask=None):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device) # (sequence_length, sequence_length)

        Q = (X @ self.W_Q) # (batch_size, sequence_length, head_size)
        K = (X @ self.W_K) # (batch_size, sequence_length, head_size)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V # (batch_size, sequence_length, v_dim)

        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0) # (batch_size, sequence_length, sequence_length)
        
        if mask is not None:
            ret = ret.masked_fill(mask == 0, 0) # Masking

        return ret @ V # (batch_size, sequence_length, v_dim)
    
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size) トークン一つ分
        s_n_1: (batch_size, hidden_size, v_dim) 直前の隠れ状態
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, head_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n
    
    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, chunk_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1
        ret = (Q @ K.transpose(-1, -2)) * D.unsqueeze(0)
        inner_chunk = ret @ V

        # e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e
        
        return inner_chunk + cross_chunk, r_i


    
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        D = (self.gamma ** (n - m)) * (n >= m).float()

        D[D != D] = 0

        return D

class MultiScaleRetention(nn.Module):

    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()

        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads

        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()
        
        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])
    
    def forward(self, X, mask=None):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = [] # (heads, batch_size, sequence_length, v_dim)
        for i in range(self.heads):
            Y.append(self.retentions[i](X, mask)) # (batch_size, sequence_length, v_dim)
        
        Y = torch.cat(Y, dim=2) # (batch_size, sequence_length, v_dim * heads)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        Recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)
        """

        # apply each individual retention mechanism to a slice of x
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :,:], s_n_1s[i], n
            )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        Chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, chunk_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of x
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
            )
            Y.append(y)
            r_is.append(r_i)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is

class Encoder(nn.Module):
    def __init__(self, layers, hidden_size, ff_size, heads, double_v_dim=False):
        super(Encoder, self).__init__()

        self.layers = layers
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.heads = heads
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.retention = MultiScaleRetention(hidden_size, heads, double_v_dim)

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_size, heads, double_v_dim)
            for _ in range(layers)
        ])

        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ff_size),
                nn.GELU(),
                nn.Linear(ff_size, hidden_size)
            )
            for _ in range(layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_size) 
            for _ in range(layers)
        ])

        self.norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(layers)
        ])
    
    def forward(self, X, mask=None):
        for i in range(self.layers):
            Y = self.retentions[i](self.norms1[i](X), mask) + X

            X = self.ffs[i](self.norms2[i](Y)) + Y
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        s_ns = []
        for i in range(self.layers):
            o_n, s_n = self.retentions[i].forward_recurrent(self.norms1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffs[i](self.norms2[i](y_n)) + y_n
        
        return x_n, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """

        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.norms1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffs[j](self.norms2[j](y_i)) + y_i
        
        return x_i, r_is

class RetNetMLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, heads=8, layers=6, max_seq_len=512, dropout=0.1, double_v_dim=False):
        super(RetNetMLM, self).__init__()

        self.hidden_size = hidden_size
        ff_size = hidden_size * 4

        self.layers = layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        self.encoder = Encoder(layers, hidden_size, ff_size, heads, double_v_dim)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, mask=None):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.token_embedding(input_ids)
        
        if mask is not None:
            mask = mask.unsqueeze(1)

            mask = (mask * mask.permute(0, 1, 2)).bool()
        
        encoder_output = self.encoder(X, mask)

        encoder_output = self.layer_norm(encoder_output)

        logits = self.mlm_head(encoder_output)

        return logits
    
    def forward_recurrent(self, input_ids, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        x_n = self.token_embedding(input_ids)

        encoder_output, s_ns = self.encoder.forward_recurrent(x_n, s_n_1s, n)

        encoder_output = self.layer_norm(encoder_output)

        logits = self.mlm_head(encoder_output)

        return logits, s_ns
    
    def forward_chunkwise(self, input_ids, r_i_1s, i):
        x_i = self.token_embedding(input_ids)

        encoder_output, r_is = self.encoder.forward_chunkwise(x_i, r_i_1s, i)

        encoder_output = self.layer_norm(encoder_output)

        logits = self.mlm_head(encoder_output)

        return logits, r_is

    
    def compute_loss(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = loss_fn(logits_flat, labels_flat)

        return loss

if __name__ == "__main__":
    """
        verify that the three implementations of RetNet are identical
    """
    vocab_size = 10
    batch_size = 2
    hidden_size = 36
    sequence_length = 5
    heads = 3
    layers = 4
    dropout = 0.1

    X = torch.randint(0, vocab_size, (batch_size, sequence_length))
    retnet = RetNetMLM(vocab_size, hidden_size, heads, layers, sequence_length, dropout=dropout, double_v_dim=False)

    Y_parallel = retnet(X)

    s_n_1s = [
        [
            torch.zeros(hidden_size // heads, retnet.v_dim // heads).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        for _ in range(layers)
    ]
    Y_recurrent = []
    for i in range(sequence_length):
        y_n, s_ns = retnet.forward_recurrent(X[:, i:i+1], s_n_1s, i)
        Y_recurrent.append(y_n)
        s_n_1s = s_ns

    Y_recurrent = torch.concat(Y_recurrent, dim=1)

    r_n_1s = [
        [
            torch.zeros(hidden_size // heads, retnet.v_dim // heads).unsqueeze(0).repeat(batch_size, 1, 1)
            for _ in range(heads)
        ]
        for _ in range(layers)
    ]
    Y_chunkwise = []
    for i in range(sequence_length):
        y_i, r_i = retnet.forward_chunkwise(X[:, i:i+1], r_n_1s, i)
        Y_chunkwise.append(y_i)
        r_n_1s = r_i
    
    Y_chunkwise = torch.concat(Y_chunkwise, dim=1)