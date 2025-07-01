import torch

def rotary_positional_encoding(x, rope_theta:float = 10000.0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

    positions = torch.arange(seq_len, device = x.device).unsqueeze(1)

    # Compute the frequency for each dimension (0, 2, 4, ...)
    # The frequency is computed as a sine wave with a frequency that decreases linearly
    # with the sequence length (seq_len) and the head dimension (head_dim).
    # This allows the model to learn different frequencies for different dimensions.
    # The frequencies are scaled by a factor of sqrt(head_dim) to ensure the
    # positional encoding has the correct scale.
    # TODO: User should be able to choose what rope_theta to use.
    freqs = torch.exp(torch.arange(0, head_dim, 2, device = x.device) *
                      -(torch.log(torch.tensor(rope_theta)) / head_dim))
    angles = positions * freqs

    sin_vals = torch.sin(angles).repeat(1,2)
    cos_vals = torch.cos(angles).repeat(1,2)

    x_rot = torch.stack([-x[...,1::2], x[...,::2]], dim = -1).reshape(
        batch_size, num_heads, seq_len, head_dim
    )
    x_enc = x * cos_vals + x_rot * sin_vals

    return x_enc

def mla_rotary_positional_encoding(x):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"

    freqs = 1.

