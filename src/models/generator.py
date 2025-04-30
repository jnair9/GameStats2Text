import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.process.stats_encoder import StatsEncoder


class GameStats2TextGenerator(nn.Module):
    """
    Generator that fuses game‐stat embeddings with GPT-2’s LM head,
    and includes a `generate()` helper for inference (beam search, top-k/p).
    """
    def __init__(
        self,
        stats_input_dim: int,
        stats_hidden_dims: list[int] = [128, 64],
        stats_output_dim: int = 32,
        gpt_model_name: str = 'gpt2',
        fusion_method: str = 'concat'
    ):
        super().__init__()
        self.stats_encoder = StatsEncoder(
            input_dim=stats_input_dim,
            hidden_dims=stats_hidden_dims,
            output_dim=stats_output_dim
        )

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        hidden_size = self.gpt2.config.n_embd

        if fusion_method == 'concat':
            self.proj = nn.Linear(hidden_size + stats_output_dim, hidden_size)
        elif fusion_method == 'add':
            self.proj = nn.Linear(stats_output_dim, hidden_size)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        self.fusion_method = fusion_method

    def forward(self, stats, input_ids, attention_mask, labels=None):
        stats_emb = self.stats_encoder(stats)
        token_emb = self.gpt2.transformer.wte(input_ids)
        L = token_emb.size(1)

        stats_exp = stats_emb.unsqueeze(1).expand(-1, L, -1)

        if self.fusion_method == 'concat':
            fused = torch.cat([token_emb, stats_exp], dim=-1)
            fused = self.proj(fused)
        else:
            stats_proj = self.proj(stats_emb).unsqueeze(1).expand(-1, L, -1)
            fused = token_emb + stats_proj

        return self.gpt2(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def generate(
        self,
        stats,
        prompt: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 128,
        num_beams: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.2
    ) -> str:

        enc = tokenizer(prompt, return_tensors='pt')
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 3) prepare stats tensor [1, D]
        stats_t = stats if isinstance(stats, torch.Tensor) else torch.tensor(stats, dtype=torch.float32, device=device)
        if stats_t.ndim == 1:
            stats_t = stats_t.unsqueeze(0)

        # 4) get token embeddings and fuse (same as forward)
        token_emb = self.gpt2.transformer.wte(input_ids)
        L = token_emb.size(1)
        stats_emb = self.stats_encoder(stats_t)
        stats_exp = stats_emb.unsqueeze(1).expand(-1, L, -1)

        if self.fusion_method == 'concat':
            fused = torch.cat([token_emb, stats_exp], dim=-1)
            fused = self.proj(fused)
        else:
            stats_proj = self.proj(stats_emb).unsqueeze(1).expand(-1, L, -1)
            fused = token_emb + stats_proj

        out_ids = self.gpt2.generate(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)