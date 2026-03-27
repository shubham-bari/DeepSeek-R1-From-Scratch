from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    d_model: int = 128
    mlp_hidden_mult: int = 4
    dropout: float = 0.1
    attention_type: str = "mha"  # mha | mqa | gqa | mla
    kv_latent_dim: int = 64
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_every: int = 2
    use_moe: bool = True
