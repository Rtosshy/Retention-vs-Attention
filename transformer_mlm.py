import torch
import torch.nn as nn
import math
import random

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, mask=None):
        # Q: (batch_size, num_heads, seq_len, d_k)
        # K: (batch_size, num_heads, seq_len, d_k)
        # V: (batch_size, num_heads, seq_len, d_v)
        d_k = Q.size(-1)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # Masking
        
        attention_weights = torch.softmax(attention_scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        
        context = torch.matmul(attention_weights, V) # (batch_size, num_heads, seq_len, d_v)

        return context, attention_weights # (batch_size, num_heads, seq_len, d_v), (batch_size, num_heads, seq_len, seq_len)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads" 
        self.d_model = d_model
        self.n_heads = n_heads

        # d_k: dimension of key/query vectors per head (d_model/n_heads)
        self.d_k = d_model // n_heads
        # d_v: dimension of value vectors per head (d_model/n_heads)
        self.d_v = d_model // n_heads
        
        # Linear Layers
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projection
        Q = self.W_Q(Q) # (batch_size, seq_len, d_model)
        K = self.W_K(K) # (batch_size, seq_len, d_model)
        V = self.W_V(V) # (batch_size, seq_len, d_model)

        # Split 
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k) # (batch_size, seq_len, n_heads, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k) # (batch_size, seq_len, n_heads, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_v) # (batch_size, seq_len, n_heads, d_v)

        # Reshape
        Q = Q.transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        K = K.transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        V = V.transpose(1, 2) # (batch_size, n_heads, seq_len, d_v)

        if mask is not None:
            # mask: (batch_size, 1, seq_len, seq_len)
            mask = mask.repeat(1, self.n_heads, 1, 1) # (batch_size, n_heads, seq_len, seq_len)
        
        # Apply attention
        context, attention_weights = self.attention(Q, K, V, mask) # (batch_size, n_heads, seq_len, d_v), (batch_size, n_heads, seq_len, seq_len)

        # Concatenate heads
        context = context.transpose(1, 2) # (batch_size, seq_len, n_heads, d_v)
        # Ensure contiguous memory layout after transpose for the subsequent view operation
        context = context.contiguous() # (batch_size, seq_len, n_heads, d_v)
        context = context.view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)

        # Final linear projection
        output = self.W_O(context) # (batch_size, seq_len, d_model)

        return output, attention_weights

# Position-wise Feed-Forward Networks
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x) # (batch_size, seq_len, d_ff)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) # (batch_size, seq_len, d_model)
        return x # (batch_size, seq_len, d_model)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout= 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # (max_seq_length, 1)

        # The use of exponential and logarithm in div_term calculation:
        # 1. Direct calculation of 10000^(2i/d_model) would result in extremely large numbers
        # 2. Using log(10000) enables the following transformation:
        #    exp(log(10000^(-2i/d_model))) = exp(-2i/d_model * log(10000)) = 1/10000^(2i/d_model)
        # 3. This improves numerical stability and computational efficiency
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)

        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe: (max_seq_length, d_model)
        pe = pe.unsqueeze(0) # (1, max_seq_length, d_model)
        self.register_buffer('pe', pe) # Register the buffer as a persistent buffer

    def forward(self, x):
        """
        Args:
            x: input tensor (batch_size, seq_len, d_model)
        """

        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
# Transformer Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Multi-Head Attention Layer
        self.attention = MultiHeadAttention(d_model, n_heads)

        # Feed-Forward Layer
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x:    input tensor (batch_size, seq_len, d_model)
            mask: mask tensor  (batch_size, seq_len, seq_len)
        """

        # Multi-head attention with residual connection and layer normalization
        attention_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output)) # Residual connection and layer normalization

        # Feed-forward with residual connection and layer normalization
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x:    input tensor (batch_size, seq_len, d_model)
            mask: mask tensor  (batch_size, seq_len, seq_len)
        """

        for layer in self.layers:
            x = layer(x, mask)
        
        return x

# Transformer Masked Language Model
class TransformerMLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, max_seq_len=512, dropout=0.1):
        super(TransformerMLM, self).__init__()

        self.d_model = d_model
        d_ff = d_model * 4

        # Token Embedding Layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder Layers
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, max_seq_len, dropout)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # MLM Prediction Head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights similar to BERT"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Tensor of token ids (batch_size, seq_len)
            attention_mask: Tensor of attention mask (batch_size, seq_len, seq_len)
                          1 for tokens to attend to, 0 for tokens to ignore
        Returns:
            logits: Prediction logits for each token position (batch_size, seq_len, vocab_size)
        """

        # Embedding Layer
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Create attention mask if provided
        if attention_mask is not None:
            # Convert attention mask from (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1) # (batch_size, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(2) # (batch_size, 1, 1, seq_len)

            # Convert 0s to -inf, 1s to 0
            attention_mask = (1.0 - attention_mask) * -1e9

            # During addition, attention_mask is automatically broadcast to match attention_scores
            # Therefore, we do not need to explicitly expand the attention_mask
            # (batch_size, 1, 1, seq_len) + (batch_size, num_heads, seq_len, seq_len)
        
        # Encoder Layers
        encoder_output = self.encoder(x, attention_mask)

        # Layer Normalization
        encoder_output = self.layer_norm(encoder_output)

        # MLM Prediction Head
        logits = self.mlm_head(encoder_output)

        return logits

def test_transformer_mlm():
    vocab_size = 8
    d_model = 512
    n_heads = 8
    n_layers = 6
    max_seq_len = 5
    batch_size = 1

    model = TransformerMLM(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len)) # (batch_size, max_seq_len)
    attention_mask = torch.ones(batch_size, max_seq_len) # (batch_size, max_seq_len)

    mask = torch.zeros(batch_size, max_seq_len) # (batch_size, max_seq_len)

    for i in range(batch_size):
        random_index = random.randint(0, max_seq_len-1)
        mask[i][random_index] = 1

    print(f"Input: {input_ids} \nInput shape: {input_ids.shape}",)
    print(f"Attention Mask: {attention_mask} \nAttention Mask shape: {attention_mask.shape}")
    print(f"Mask: {mask} \nMask shape: {mask.shape}")

    logits = model(input_ids, attention_mask) # (batch_size, max_seq_len, vocab_size)
    assert logits.shape == (batch_size, max_seq_len, vocab_size), "Invalid shape"
    
    probs = nn.Softmax(dim=-1)(logits) # (batch_size, max_seq_len, vocab_size)

    masked_probs = probs * mask.unsqueeze(-1) # (batch_size, max_seq_len, vocab_size)

    print(f"Logits: {logits} \nLogits shape: {logits.shape}")

    print(f"Probabilities: {probs} \nProbabilities shape: {probs.shape}")

    print(f"Masked Probabilities: {masked_probs} \nMasked Probabilities shape: {masked_probs.shape}")

    print("PASSED")

if __name__ == "__main__":
    test_transformer_mlm()

