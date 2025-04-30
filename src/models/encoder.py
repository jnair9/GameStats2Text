import torch
import torch.nn as nn
from transformers import GPT2Model
from src.process.stats_encoder import StatsEncoder


class GameStats2TextModel(nn.Module):

    def __init__(self,
                 stats_input_dim: int,
                 stats_hidden_dims: list = [128, 64],
                 stats_output_dim: int = 32,
                 gpt_model_name: str = 'gpt2',
                 fusion_method: str = 'concat'):
        super().__init__()

    
        self.stats_encoder = StatsEncoder(
            input_dim=stats_input_dim,
            hidden_dims=stats_hidden_dims,
            output_dim=stats_output_dim
        )

        self.gpt2 = GPT2Model.from_pretrained(gpt_model_name)

        if fusion_method == 'concat':
            self.proj = nn.Linear(
                self.gpt2.config.hidden_size + stats_output_dim,
                self.gpt2.config.hidden_size
            )
        elif fusion_method == 'add':
            self.proj = nn.Linear(stats_output_dim, self.gpt2.config.hidden_size)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.fusion_method = fusion_method

    def forward(self, stats, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            stats (Tensor): (batch_size, stats_input_dim)
            input_ids (Tensor): (batch_size, seq_len)
            attention_mask (Tensor): (batch_size, seq_len)

        Returns:
            Tensor: last_hidden_state (batch_size, seq_len, hidden_size)
        """
        stats_emb = self.stats_encoder(stats)

        gpt_outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        token_embeddings = gpt_outputs.last_hidden_state

        stats_emb_expanded = stats_emb.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)

        if self.fusion_method == 'concat':
            combined = torch.cat([token_embeddings, stats_emb_expanded], dim=-1)
        elif self.fusion_method == 'add':
            stats_proj = self.proj(stats_emb)
            stats_proj_expanded = stats_proj.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)
            combined = token_embeddings + stats_proj_expanded
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        if self.fusion_method == 'concat':
            combined = self.proj(combined)

        return combined

