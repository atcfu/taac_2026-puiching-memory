from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
	def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.w1 = nn.Linear(dim, hidden_dim)
		self.w2 = nn.Linear(dim, hidden_dim)
		self.out = nn.Linear(hidden_dim, dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return self.out(self.dropout(F.silu(self.w1(hidden_states)) * self.w2(hidden_states)))


class SimpleMHA(nn.Module):
	def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		if d_model % n_heads != 0:
			raise ValueError("d_model must be divisible by n_heads")
		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model // n_heads
		self.q = nn.Linear(d_model, d_model)
		self.k = nn.Linear(d_model, d_model)
		self.v = nn.Linear(d_model, d_model)
		self.o = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(
		self,
		query_states: torch.Tensor,
		key_states: torch.Tensor,
		value_states: torch.Tensor,
		kv_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		batch_size, query_len, _ = query_states.shape
		key_len = key_states.shape[1]

		query_states = self.q(query_states).view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
		key_states = self.k(key_states).view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)
		value_states = self.v(value_states).view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)

		score = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
		if kv_mask is not None:
			score = score.masked_fill(~kv_mask[:, None, None, :].bool(), -1.0e4)
		attention = self.dropout(score.softmax(dim=-1))
		output = torch.matmul(attention, value_states).transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
		return self.o(output)


class CrossAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.query_norm = RMSNorm(d_model)
		self.key_value_norm = RMSNorm(d_model)
		self.attn = SimpleMHA(d_model, n_heads, dropout)
		self.dropout = nn.Dropout(dropout)

	def forward(self, query_states: torch.Tensor, key_value_states: torch.Tensor, kv_mask: torch.Tensor | None = None) -> torch.Tensor:
		return query_states + self.dropout(
			self.attn(self.query_norm(query_states), self.key_value_norm(key_value_states), self.key_value_norm(key_value_states), kv_mask)
		)


class SelfAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.norm = RMSNorm(d_model)
		self.attn = SimpleMHA(d_model, n_heads, dropout)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
		normalized = self.norm(hidden_states)
		return hidden_states + self.dropout(self.attn(normalized, normalized, normalized, token_mask))


class FeedForwardBlock(nn.Module):
	def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
		super().__init__()
		self.norm = RMSNorm(d_model)
		self.ff = SwiGLU(d_model, d_model * hidden_mult, dropout)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return hidden_states + self.dropout(self.ff(self.norm(hidden_states)))


class TokenMixer(nn.Module):
	def __init__(self, d_model: int, num_tokens: int, expansion: int = 2, dropout: float = 0.0) -> None:
		super().__init__()
		hidden = max(num_tokens, num_tokens * expansion)
		self.norm = RMSNorm(d_model)
		self.fc1 = nn.Linear(num_tokens, hidden)
		self.fc2 = nn.Linear(hidden, num_tokens)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		mixed = self.norm(hidden_states).transpose(1, 2)
		mixed = self.fc2(F.gelu(self.fc1(mixed))).transpose(1, 2)
		return hidden_states + self.dropout(mixed)


class MemoryCompressor(nn.Module):
	def __init__(self, d_model: int, memory_tokens: int, n_heads: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.memory = nn.Parameter(torch.randn(1, memory_tokens, d_model) * 0.02)
		self.cross = CrossAttentionBlock(d_model, n_heads, dropout)
		self.ff = FeedForwardBlock(d_model, dropout=dropout)

	def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
		memory = self.memory.expand(hidden_states.size(0), -1, -1)
		return self.ff(self.cross(memory, hidden_states, token_mask))


class UniScaleLayer(nn.Module):
	def __init__(self, d_model: int, n_heads: int, num_mix_tokens: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.cross = CrossAttentionBlock(d_model, n_heads, dropout)
		self.mix = TokenMixer(d_model, num_mix_tokens, 2, dropout)
		self.ff = FeedForwardBlock(d_model, 4, dropout)

	def forward(
		self,
		query_states: torch.Tensor,
		memory_states: torch.Tensor,
		memory_mask: torch.Tensor,
		static_states: torch.Tensor,
	) -> torch.Tensor:
		query_states = self.cross(query_states, memory_states, memory_mask)
		mixed = torch.cat([query_states, static_states], dim=1)
		mixed = self.ff(self.mix(mixed))
		return mixed[:, : query_states.size(1)]


class FMHead(nn.Module):
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		summed = hidden_states.sum(dim=1)
		return 0.5 * ((summed.pow(2) - hidden_states.pow(2).sum(dim=1)).sum(dim=-1, keepdim=True))


class UniScaleFormerModel(nn.Module):
	def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
		super().__init__()
		self.data_config = data_config
		self.model_config = model_config
		self.sequence_names = tuple(data_config.sequence_names)
		self.sequence_count = len(self.sequence_names)
		self.d_model = model_config.hidden_dim
		self.memory_tokens = max(1, model_config.memory_slots)
		self.num_queries = max(1, model_config.num_queries)
		self.base_static_count = 8
		self.extra_static_count = max(0, model_config.segment_count - self.base_static_count)
		self.static_token_count = self.base_static_count + self.extra_static_count
		self.last_aux_similarity: torch.Tensor | None = None

		self.token_embedding = nn.Embedding(
			num_embeddings=model_config.vocab_size,
			embedding_dim=model_config.embedding_dim,
			padding_idx=0,
		)
		self.token_projection = (
			nn.Identity()
			if model_config.embedding_dim == self.d_model
			else nn.Linear(model_config.embedding_dim, self.d_model)
		)
		self.dense_projection = nn.Sequential(
			nn.Linear(dense_dim, self.d_model),
			nn.LayerNorm(self.d_model),
			nn.GELU(),
		)
		self.extra_static_projection = None
		if self.extra_static_count > 0:
			self.extra_static_projection = nn.Sequential(
				nn.Linear(dense_dim, self.d_model * 2),
				nn.LayerNorm(self.d_model * 2),
				nn.GELU(),
				nn.Linear(self.d_model * 2, self.extra_static_count * self.d_model),
			)
		self.static_field_embedding = nn.Embedding(self.static_token_count, self.d_model)
		self.sequence_name_embedding = nn.Embedding(self.sequence_count + 1, self.d_model, padding_idx=0)
		self.position_embedding = nn.Embedding(data_config.max_seq_len, self.d_model)
		self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, self.d_model, padding_idx=0)
		self.event_projection = nn.Sequential(
			nn.LayerNorm(self.d_model * 5),
			nn.Linear(self.d_model * 5, self.d_model * 4),
			nn.GELU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(self.d_model * 4, self.d_model),
		)
		self.empty_sequence_tokens = nn.Parameter(torch.randn(self.sequence_count, self.d_model) * 0.02)

		self.static_encoder = nn.ModuleList(
			[
				SelfAttentionBlock(self.d_model, model_config.num_heads, model_config.dropout)
				for _ in range(max(1, model_config.static_layers))
			]
		)
		self.seq_encoder = nn.ModuleList(
			[
				SelfAttentionBlock(self.d_model, model_config.num_heads, model_config.dropout)
				for _ in range(max(1, model_config.sequence_layers))
			]
		)
		self.compressors = nn.ModuleList(
			[
				MemoryCompressor(self.d_model, self.memory_tokens, model_config.num_heads, model_config.dropout)
				for _ in range(self.sequence_count)
			]
		)
		self.query_seed = nn.Parameter(torch.randn(1, self.num_queries, self.d_model) * 0.02)
		self.query_proj = nn.Sequential(
			nn.Linear(self.d_model * 2, self.d_model),
			nn.GELU(),
			nn.Linear(self.d_model, self.d_model),
		)
		self.layers = nn.ModuleList(
			[
				UniScaleLayer(
					self.d_model,
					model_config.num_heads,
					self.num_queries + self.static_token_count,
					model_config.dropout,
				)
				for _ in range(max(1, model_config.num_layers))
			]
		)
		self.final_attn = SelfAttentionBlock(self.d_model, model_config.num_heads, model_config.dropout)
		self.fm = FMHead()
		head_hidden_dim = model_config.head_hidden_dim or self.d_model * 2
		self.head = nn.Sequential(
			nn.LayerNorm(self.d_model * 3 + 1),
			nn.Linear(self.d_model * 3 + 1, head_hidden_dim),
			nn.GELU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(head_hidden_dim, 1),
		)
		self._init_weights()

	def _init_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
				nn.init.normal_(module.weight, std=0.02)
				if module.padding_idx is not None:
					nn.init.zeros_(module.weight[module.padding_idx])

	def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
		if tensor is None:
			raise RuntimeError(f"Batch is missing required tensor: {name}")
		return tensor

	def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
		return self.token_projection(self.token_embedding(tokens))

	def _embed_hashed_indices(self, indices: torch.Tensor) -> torch.Tensor:
		vocab_limit = self.model_config.vocab_size - 1
		token_ids = (indices.remainder(vocab_limit) + 1).long()
		return self._embed_tokens(token_ids.unsqueeze(1)).squeeze(1)

	def _build_static_tokens(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		user_tokens = self._require(batch.user_tokens, "user_tokens")
		user_mask = self._require(batch.user_mask, "user_mask")
		candidate_post_tokens = self._require(batch.candidate_post_tokens, "candidate_post_tokens")
		candidate_post_mask = self._require(batch.candidate_post_mask, "candidate_post_mask")
		candidate_author_tokens = self._require(batch.candidate_author_tokens, "candidate_author_tokens")
		candidate_author_mask = self._require(batch.candidate_author_mask, "candidate_author_mask")

		user_id_token = self._embed_hashed_indices(batch.user_indices)
		item_id_token = self._embed_hashed_indices(batch.item_indices)
		user_summary = masked_mean(self._embed_tokens(user_tokens), user_mask)
		context_summary = masked_mean(self._embed_tokens(batch.context_tokens), batch.context_mask)
		candidate_summary = masked_mean(self._embed_tokens(batch.candidate_tokens), batch.candidate_mask)
		candidate_post_summary = masked_mean(self._embed_tokens(candidate_post_tokens), candidate_post_mask)
		candidate_author_summary = masked_mean(self._embed_tokens(candidate_author_tokens), candidate_author_mask)
		dense_summary = self.dense_projection(batch.dense_features)

		static_tokens = torch.stack(
			[
				user_id_token,
				item_id_token,
				user_summary,
				context_summary,
				candidate_summary,
				candidate_post_summary,
				candidate_author_summary,
				dense_summary,
			],
			dim=1,
		)
		if self.extra_static_projection is not None:
			extra_tokens = self.extra_static_projection(batch.dense_features).view(batch.batch_size, self.extra_static_count, self.d_model)
			static_tokens = torch.cat([static_tokens, extra_tokens], dim=1)
		field_ids = torch.arange(self.static_token_count, device=static_tokens.device)
		static_tokens = static_tokens + self.static_field_embedding(field_ids).unsqueeze(0)
		static_mask = torch.ones(batch.batch_size, self.static_token_count, dtype=torch.bool, device=static_tokens.device)
		return static_tokens, static_mask, item_id_token

	def _build_event_states(self, batch: BatchTensors) -> torch.Tensor:
		history_post_tokens = self._require(batch.history_post_tokens, "history_post_tokens")
		history_author_tokens = self._require(batch.history_author_tokens, "history_author_tokens")
		history_action_tokens = self._require(batch.history_action_tokens, "history_action_tokens")
		history_time_gap = self._require(batch.history_time_gap, "history_time_gap")

		history_hidden = self._embed_tokens(batch.history_tokens)
		post_hidden = self._embed_tokens(history_post_tokens)
		author_hidden = self._embed_tokens(history_author_tokens)
		action_hidden = self._embed_tokens(history_action_tokens)
		gap_hidden = self.time_gap_embedding(history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
		event_inputs = torch.cat([history_hidden, post_hidden, author_hidden, action_hidden, gap_hidden], dim=-1)
		return self.event_projection(event_inputs) * batch.history_mask.unsqueeze(-1).float()

	def _split_sequences(self, batch: BatchTensors, event_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		history_group_ids = self._require(batch.history_group_ids, "history_group_ids")
		batch_size = batch.batch_size
		max_seq_len = self.data_config.max_seq_len
		device = event_states.device
		sequence_states = event_states.new_zeros(batch_size, self.sequence_count, max_seq_len, self.d_model)
		sequence_mask = torch.zeros(batch_size, self.sequence_count, max_seq_len, dtype=torch.bool, device=device)

		for batch_index in range(batch_size):
			valid_positions = torch.nonzero(batch.history_mask[batch_index], as_tuple=False).squeeze(-1)
			for sequence_index in range(self.sequence_count):
				if valid_positions.numel() > 0:
					selected_positions = valid_positions[
						history_group_ids[batch_index, valid_positions] == (sequence_index + 1)
					]
				else:
					selected_positions = valid_positions
				selected_positions = selected_positions[-max_seq_len:]
				selected_count = int(selected_positions.numel())
				if selected_count == 0:
					sequence_states[batch_index, sequence_index, 0] = self.empty_sequence_tokens[sequence_index]
					sequence_mask[batch_index, sequence_index, 0] = True
					continue
				sequence_states[batch_index, sequence_index, :selected_count] = event_states[batch_index, selected_positions]
				sequence_mask[batch_index, sequence_index, :selected_count] = True

		positions = torch.arange(max_seq_len, device=device)
		sequence_states = sequence_states + self.position_embedding(positions).view(1, 1, max_seq_len, self.d_model)
		seq_name_ids = torch.arange(1, self.sequence_count + 1, device=device)
		sequence_states = sequence_states + self.sequence_name_embedding(seq_name_ids).view(1, self.sequence_count, 1, self.d_model)
		sequence_states = sequence_states * sequence_mask.unsqueeze(-1).float()
		return sequence_states, sequence_mask

	def forward(self, batch: BatchTensors) -> torch.Tensor:
		static_states, static_mask, item_id_token = self._build_static_tokens(batch)
		for block in self.static_encoder:
			static_states = block(static_states, static_mask)
		static_pool = masked_mean(static_states, static_mask)

		event_states = self._build_event_states(batch)
		sequence_states, sequence_mask = self._split_sequences(batch, event_states)

		memories = []
		sequence_pools = []
		for sequence_index in range(self.sequence_count):
			hidden_states = sequence_states[:, sequence_index]
			token_mask = sequence_mask[:, sequence_index]
			for block in self.seq_encoder:
				hidden_states = block(hidden_states, token_mask)
			memories.append(self.compressors[sequence_index](hidden_states, token_mask))
			sequence_pools.append(masked_mean(hidden_states, token_mask))

		memory_states = torch.cat(memories, dim=1)
		memory_mask = torch.ones(memory_states.shape[:2], dtype=torch.bool, device=memory_states.device)
		sequence_pool = masked_mean(torch.stack(sequence_pools, dim=1), sequence_mask.any(dim=-1))

		query_seed = self.query_seed.expand(static_states.size(0), -1, -1)
		query_states = query_seed + self.query_proj(torch.cat([static_pool, sequence_pool], dim=-1)).unsqueeze(1)
		for layer in self.layers:
			query_states = layer(query_states, memory_states, memory_mask, static_states)

		fused_states = torch.cat([query_states, static_states], dim=1)
		fused_mask = torch.cat(
			[
				torch.ones(static_states.size(0), query_states.size(1), dtype=torch.bool, device=static_states.device),
				static_mask,
			],
			dim=1,
		)
		fused_states = self.final_attn(fused_states, fused_mask)

		query_pool = query_states.mean(dim=1)
		fused_pool = masked_mean(fused_states, fused_mask)
		fm_cross = self.fm(torch.cat([query_states, static_states], dim=1))
		self.last_aux_similarity = F.cosine_similarity(query_pool, item_id_token, dim=-1)

		head_input = torch.cat([query_pool, static_pool, fused_pool, fm_cross], dim=-1)
		logits = self.head(head_input).squeeze(-1)
		return logits


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> UniScaleFormerModel:
	return UniScaleFormerModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)