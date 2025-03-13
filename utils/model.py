# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------
# Global Positional Encoding
# -------------------------------------------------------------------------------------------
class GlobalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, use_global_positional_encoding=True, use_rope=True):
        super(GlobalPositionalEncoding, self).__init__()
        self.use_global_positional_encoding = use_global_positional_encoding
        self.use_rope = use_rope
        self.d_model = d_model

        if use_global_positional_encoding and not use_rope:
            pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)  # Sinusoidal encoding for even dimensions
            pe[:, 1::2] = torch.cos(position * div_term)  # Cosine encoding for odd dimensions
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)

    def forward(self, x):
        if not self.use_global_positional_encoding:
            return x  # No positional encoding

        seq_len = x.size(1)
        if self.use_rope:
            # Apply RoPE-based Positional Encoding (Global)
            position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)  # (seq_len, 1)
            dim_indices = torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device)  # (d_model // 2)
            div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * dim_indices / self.d_model)

            angle = position * div_term  # (seq_len, d_model // 2)
            sin = torch.sin(angle).unsqueeze(0)  # (1, seq_len, d_model // 2)
            cos = torch.cos(angle).unsqueeze(0)  # (1, seq_len, d_model // 2)

            def rope_transform(embed):
                x1, x2 = embed[..., ::2], embed[..., 1::2]  # Split into even and odd parts
                x_rope_even = x1 * cos - x2 * sin
                x_rope_odd = x1 * sin + x2 * cos
                return torch.stack([x_rope_even, x_rope_odd], dim=-1).flatten(-2)

            x = rope_transform(x)
        else:
            x = x + self.pe[:, :seq_len]
        return x


# -------------------------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE) for Local Attention
# -------------------------------------------------------------------------------------------
def apply_rope_qk(q, k, use_local_positional_encoding=True):
    if not use_local_positional_encoding:
        return q, k  # Return unmodified q, k if RoPE is disabled

    batch_size, num_heads, seq_len, head_dim = q.size()
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    position = torch.arange(seq_len, dtype=torch.float, device=q.device).unsqueeze(1)  # (seq_len, 1)
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float, device=q.device)  # (head_dim // 2)
    div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * dim_indices / head_dim)

    angle = position * div_term  # (seq_len, head_dim // 2)
    sin = torch.sin(angle).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    cos = torch.cos(angle).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)

    def rope_transform(x):
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even and odd parts
        x_rope_even = x1 * cos - x2 * sin
        x_rope_odd = x1 * sin + x2 * cos
        return torch.stack([x_rope_even, x_rope_odd], dim=-1).flatten(-2)

    q = rope_transform(q)
    k = rope_transform(k)
    return q, k


# -------------------------------------------------------------------------------------------
# Multi-Head Attention with optional RoPE
# -------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # Reshape to (B, H, L, D)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (if enabled)
        query, key = apply_rope_qk(query, key)

        if self.flash:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
            attn_weights = None
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(attn_output)
        output = self.resid_dropout(output)

        return output, attn_weights

# -------------------------------------------------------------------------------------------
# Feed-Forward Network
# -------------------------------------------------------------------------------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward=2048, dropout=0.0):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# -------------------------------------------------------------------------------------------
# Custom Transformer Encoder/Decoder
# -------------------------------------------------------------------------------------------
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, 4 * hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2, _ = self.self_attn(src, src, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, 4 * hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# -------------------------------------------------------------------------------------------
# Encoder with Global Positional Encoding
# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, num_heads, dropout=0.0, use_norm=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.global_pos_encoder = GlobalPositionalEncoding(hidden_dim)
        self.transformer_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(hidden_dim, num_heads, dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, x):
        x = self.embedding(x)
        x = self.global_pos_encoder(x)
        for layer in self.transformer_encoder:
            x = layer(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        return x

# -------------------------------------------------------------------------------------------
# Decoder with Global Positional Encoding
# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, num_heads, dropout=0.0, use_norm=True):
        super(Decoder, self).__init__()
        self.global_pos_encoder = GlobalPositionalEncoding(hidden_dim)
        self.transformer_decoder = nn.ModuleList([
            CustomTransformerDecoderLayer(hidden_dim, num_heads, dropout) for _ in range(n_layers)
        ])
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, encoder_outputs):
        x = self.global_pos_encoder(encoder_outputs)
        for layer in self.transformer_decoder:
            x = layer(x, encoder_outputs)
        if self.layer_norm:
            x = self.layer_norm(x)
        return self.fc_output(x)

# -------------------------------------------------------------------------------------------
# Seq2Seq Model
# -------------------------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src):
        encoder_outputs = self.encoder(src)
        output = self.decoder(encoder_outputs)
        return output

class Loss(nn.Module):
    def __init__(self, delta=1.0, w1=1.0, w2=1.0, w3=1.0):

        super(Loss, self).__init__()
        self.delta = delta
        self.w1 = w1  # reconstruction loss weight
        self.w2 = w2  # temporal consistency loss weight
        self.w3 = w3  # directional consistency loss weight
        self.rec_loss_fn = nn.SmoothL1Loss(beta=self.delta)

    def forward(self, predictions, targets, current_step=None, total_steps=None):
        rec_loss = self.rec_loss_fn(predictions, targets)
        pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]  # (B, T-1, F)
        target_diff = targets[:, 1:, :] - targets[:, :-1, :]          # (B, T-1, F)
        temp_loss = F.l1_loss(pred_diff, target_diff)
        eps = 1e-8  # small constant to avoid division by zero
        pred_norm = pred_diff / (pred_diff.norm(dim=-1, keepdim=True) + eps)
        target_norm = target_diff / (target_diff.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1)  # (B, T-1)
        dir_loss = 1 - cos_sim.mean()
    
        
        total_loss = self.w1 * rec_loss + self.w2 * temp_loss + self.w3 * dir_loss
        return total_loss


'''

class Loss(nn.Module):
    def __init__(self, delta=1.0, w1=1.0, w2=1.0, w3=1.0):
        """
        Args:
            delta (float): Delta parameter for the Smooth L1 (Huber) loss.
            w1 (float): Weight for reconstruction loss.
            w2 (float): Weight for temporal consistency loss.
            w3 (float): Weight for directional consistency loss.
        """
        super(Loss, self).__init__()
        self.delta = delta
        self.w1 = w1  # Reconstruction loss weight
        self.w2 = w2  # Temporal consistency loss weight
        self.w3 = w3  # Directional consistency loss weight
        self.rec_loss_fn = nn.SmoothL1Loss(beta=self.delta)
        
        # Indices of features/columns that should always be zero.
        self.zero_indices = [0, 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
        # Extra penalty weight for nonzero values in these columns.
        self.zero_penalty_weight = 1.0

    def forward(self, predictions, targets, current_step=None, total_steps=None):
        # 1. Reconstruction Loss: Ensure predictions match the targets.
        rec_loss = self.rec_loss_fn(predictions, targets)
        
        # 2. Temporal Consistency Loss: Enforce similar frame-to-frame differences.
        pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]  # (B, T-1, F)
        target_diff = targets[:, 1:, :] - targets[:, :-1, :]          # (B, T-1, F)
        temp_loss = F.l1_loss(pred_diff, target_diff)
        
        # 3. Directional Consistency Loss: Enforce similar directions of change.
        eps = 1e-8  # small constant to avoid division by zero
        pred_norm = pred_diff / (pred_diff.norm(dim=-1, keepdim=True) + eps)
        target_norm = target_diff / (target_diff.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1)  # (B, T-1)
        dir_loss = 1 - cos_sim.mean()
        
        # 4. Zero Penalty Loss: Penalize nonzero predictions for specified features.
        # Extract the features (columns) that should always be zero.
        zero_features = predictions[:, :, self.zero_indices]
        zero_target = torch.zeros_like(zero_features)
        zero_loss = F.l1_loss(zero_features, zero_target)
        
        # Combine all the loss components.
        total_loss = (
            self.w1 * rec_loss +
            self.w2 * temp_loss +
            self.w3 * dir_loss +
            self.zero_penalty_weight * zero_loss
        )
        return total_loss


class Loss(nn.Module):
    def __init__(self, delta=1.0, w1=1.0, w2=1.0, w3=1.0):

        super(Loss, self).__init__()
        self.delta = delta
        self.w1 = w1  # reconstruction loss weight
        self.w2 = w2  # temporal consistency loss weight
        self.w3 = w3  # directional consistency loss weight
        self.rec_loss_fn = nn.SmoothL1Loss(beta=self.delta)

    def forward(self, predictions, targets, current_step=None, total_steps=None):
        rec_loss = self.rec_loss_fn(predictions, targets)
        pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]  # (B, T-1, F)
        target_diff = targets[:, 1:, :] - targets[:, :-1, :]          # (B, T-1, F)
        temp_loss = F.l1_loss(pred_diff, target_diff)
        eps = 1e-8  # small constant to avoid division by zero
        pred_norm = pred_diff / (pred_diff.norm(dim=-1, keepdim=True) + eps)
        target_norm = target_diff / (target_diff.norm(dim=-1, keepdim=True) + eps)
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1)  # (B, T-1)
        dir_loss = 1 - cos_sim.mean()
    
        
        total_loss = self.w1 * rec_loss + self.w2 * temp_loss + self.w3 * dir_loss
        return total_loss




class Loss(nn.Module):
    def __init__(self, delta=1.0, w1=1.0, w2=1.0, w3=1.0, w4=1.0, 
                 smoothness_anneal_rate=1, enable_smoothness_annealing=False):
        super(Loss, self).__init__()
        self.delta = delta  # Delta parameter for Huber loss
        self.w1 = w1        # Weight for target (Huber) loss
        self.w2 = w2        # Weight for L2 smoothness loss
        self.w3 = w3        # Weight for cosine similarity loss
        self.w4 = w4        # Weight for second-order smoothness loss
        self.smoothness_anneal_rate = smoothness_anneal_rate  # Rate for annealing smoothness loss weight
        self.enable_smoothness_annealing = enable_smoothness_annealing  # Enable/Disable smoothness annealing, not sure if it helps.

    def huber_loss(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.where(
            abs_error <= self.delta,
            0.5 * torch.square(error),
            self.delta * abs_error - 0.5 * self.delta**2
        )
        return torch.mean(quadratic)

    # ------------------------------------------------------------------------------
    # Vectorized version of the L2 smoothness loss
    # ------------------------------------------------------------------------------
    def l2_smoothness_loss(self, predictions, targets):
        """
        Equivalent to summing the L2 difference between consecutive frames in 
        predictions and targets, then averaging.
        """
        # *CHANGED: Remove the for loop, use slicing*
        pred_diff = predictions[:, :-1] - predictions[:, 1:]
        target_diff = targets[:, :-1] - targets[:, 1:]
        smoothness_loss = (pred_diff - target_diff).pow(2).mean()
        return smoothness_loss

    # ------------------------------------------------------------------------------
    # Vectorized version of the second-order smoothness loss
    # ------------------------------------------------------------------------------
    def second_order_smoothness_loss(self, predictions, targets):
        """
        Equivalent to summing the second-order difference (difference of first-order 
        differences) between consecutive frames in predictions and targets, then averaging.
        """
        pred_diff_1 = predictions[:, :-2] - predictions[:, 1:-1]
        pred_diff_2 = predictions[:, 1:-1] - predictions[:, 2:]
        target_diff_1 = targets[:, :-2] - targets[:, 1:-1]
        target_diff_2 = targets[:, 1:-1] - targets[:, 2:]

        pred_second_order = pred_diff_1 - pred_diff_2
        target_second_order = target_diff_1 - target_diff_2

        second_order_loss = (pred_second_order - target_second_order).pow(2).mean()
        return second_order_loss

    # ------------------------------------------------------------------------------
    # Vectorized version of the cosine similarity loss
    # ------------------------------------------------------------------------------
    def cosine_similarity_loss(self, predictions, targets):
        """
        Computes the average cosine similarity loss over all frames and the batch dimension.
        """
        # *CHANGED: Remove the for loop, flatten frame and batch dimensions together*
        pred_flat = predictions.reshape(-1, predictions.size(-1))
        target_flat = targets.reshape(-1, targets.size(-1))
        cosine_vals = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        return 1 - cosine_vals.mean()

    def cosine_annealing(self, current_step, total_steps):
        # Set the number of full cycles (oscillations) you want across the entire training
        num_cycles = 1000  # You can adjust this to control how many times the oscillations occur
    
        # Normalize current_step to a value between 0 and 2π * num_cycles for multiple cycles
        phase = (current_step / total_steps) * 3.14159265359 * 2 * num_cycles

        # Cosine for L2 smoothness weight (oscillates between 0 and 1)
        l2_weight = (torch.cos(torch.tensor(phase)) + 1) / 2

        # Sine for second-order smoothness weight, out of phase with cosine (oscillates between 0 and 1)
        second_order_weight = (torch.cos(torch.tensor(phase) + 3.14159265359) + 1) / 2

        return l2_weight, second_order_weight

    def forward(self, predictions, targets, current_step, total_steps):
        # Calculate the target accuracy loss using Huber loss
        target_loss = self.huber_loss(targets, predictions)
        
        # Determine smoothness weights: cosine for L2, sine for second-order, both annealed
        if self.enable_smoothness_annealing:
            l2_smoothness_weight, second_order_smoothness_weight = self.cosine_annealing(current_step, total_steps)
        else:
            l2_smoothness_weight, second_order_smoothness_weight = 1.0, 1.0

        # Calculate both first-order (L2) and second-order smoothness losses
        l2_smoothness = self.l2_smoothness_loss(predictions, targets)
        second_order_smoothness = self.second_order_smoothness_loss(predictions, targets)

        # Calculate the cosine similarity loss between the corresponding prediction and target frames
        cosine_loss = self.cosine_similarity_loss(predictions, targets)
        
        # Combine the losses with their respective weights
        total_loss = (
            self.w1 * target_loss +
            l2_smoothness_weight * self.w2 * l2_smoothness +
            second_order_smoothness_weight * self.w4 * second_order_smoothness +
            self.w3 * cosine_loss
        )

        return total_loss
'''
