"""
model.py — CrickLM Transformer (built from scratch)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# Hyperparameter config 
@dataclass
class CrickLMConfig:
    """
    All model hyperparameters in one place.
    Changing these values scales the model up or down.

    Default config ≈ 10M parameters:
      vocab_size=8000, d_model=256, n_heads=8, n_layers=6, d_ff=1024
    """
    vocab_size: int   = 8000    # size of token vocabulary
    d_model:    int   = 256     # embedding dimension (must be divisible by n_heads)
    n_heads:    int   = 8       # number of attention heads
    n_layers:   int   = 6       # number of stacked transformer blocks
    d_ff:       int   = 1024    # feed-forward inner dimension (usually 4 × d_model)
    seq_len:    int   = 128     # maximum sequence length (context window)
    dropout:    float = 0.1     # dropout probability (regularisation)
    pad_id:     int   = 0       # token ID used for padding

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.d_head = self.d_model // self.n_heads   # dimension per head

    def num_params(self) -> int:
        """Estimate total trainable parameters."""
        embed    = self.vocab_size * self.d_model          # token embedding
        pos_emb  = self.seq_len   * self.d_model          # positional embedding
        # Per layer: 4 weight matrices for attention (Q,K,V,O) + 2 for FFN + norms
        per_layer = (4 * self.d_model * self.d_model +    # attention projections
                     2 * self.d_model * self.d_ff    +    # FFN weights
                     4 * self.d_model)                    # layer norm params
        head_proj = self.vocab_size * self.d_model        # final projection
        return embed + pos_emb + self.n_layers * per_layer + head_proj


# 1. Token + Positional Embeddings 
class Embeddings(nn.Module):
    def __init__(self, config: CrickLMConfig):
        super().__init__()
        # Token embedding table: vocab_size rows, each d_model wide
        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,   # PAD tokens get zero gradient
        )
        # Positional embedding table: one row per position
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.dropout  = nn.Dropout(config.dropout)
        self.d_model  = config.d_model
        self.seq_len  = config.seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        assert S <= self.seq_len, f"Sequence {S} exceeds max {self.seq_len}"

        positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        tok = self.token_emb(x) * math.sqrt(self.d_model)   # (B, S, d_model)
        pos = self.pos_emb(positions)                         # (1, S, d_model)

        return self.dropout(tok + pos)   # (B, S, d_model)


# 2. Causal Self-Attention 

class CausalSelfAttention(nn.Module):
    def __init__(self, config: CrickLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_head
        self.d_model = config.d_model
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        S = config.seq_len
        causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
        
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)                    
        Q, K, V = qkv.split(self.d_model, dim=-1) 

        # Step 2: reshape into heads 
        def split_heads(t):
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(Q)   
        K = split_heads(K)   
        V = split_heads(V)   

        # Step 3: scaled dot-product attention 
        scale = math.sqrt(self.d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale 

        mask = self.causal_mask[:S, :S]             
        scores = scores.masked_fill(~mask, float("-inf"))  

        weights = F.softmax(scores, dim=-1)         
        weights = self.attn_dropout(weights)

        
        attended = torch.matmul(weights, V)  

        attended = attended.transpose(1, 2).contiguous().view(B, S, D)

        output = self.resid_dropout(self.out_proj(attended))
        return output


# 3. Feed-Forward Network 

class FeedForward(nn.Module):

    def __init__(self, config: CrickLMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),    # expand
            nn.GELU(),                                  # non-linearity
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),    # compress back
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 4. Transformer Block 

class TransformerBlock(nn.Module):

    def __init__(self, config: CrickLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)  
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)   
        self.ffn  = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        
        x = x + self.attn(self.ln1(x))
        
        x = x + self.ffn(self.ln2(x))
        return x


# 5. Full CrickLM Model 
class CrickLM(nn.Module):

    def __init__(self, config: CrickLMConfig):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)

        
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        
        self.ln_final = nn.LayerNorm(config.d_model)

        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        
        self.lm_head.weight = self.embeddings.token_emb.weight

        
        self.apply(self._init_weights)

        
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        total = sum(p.numel() for p in self.parameters())
        print(f"CrickLM initialised: {total/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialise weights using small normal distribution.
        Biases → 0, embeddings and linear weights → N(0, 0.02).
        This is the standard GPT-2 initialisation.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        h = self.embeddings(x)             

        
        for block in self.blocks:
            h = block(h)                   

        
        h = self.ln_final(h)               

        
        logits = self.lm_head(h)           

        # 5. Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,   
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            context = ids[:, -self.config.seq_len:]
            context = context.clamp(0, self.config.vocab_size - 1)

            
            logits, _ = self(context)
            next_logits = logits[:, -1, :] 

            
            next_logits = next_logits / max(temperature, 1e-6)

            
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                topk_vals = torch.topk(next_logits, k).values
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(
                    next_logits, descending=True
                )
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                next_logits = torch.zeros_like(next_logits).scatter(
                    1, sorted_idx, sorted_logits
                )

            
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1) 
            ids = torch.cat([ids, next_id], dim=1)

            
            if next_id.item() == 3: 
                break

        return ids