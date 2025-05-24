from typing import List, Union, Dict

import torch
import pickle
import sys

from nanotron.parallel.pipeline_parallel.block import TensorPointer

from typing import Any, Dict, Optional
import nanotron.config
import nanotron.config
import nanotron.config.config
import nanotron.config.models_config
import nanotron.trainer
import torch
from torch import nn
import torch.distributed as dist
from packaging.version import Version
from pathlib import Path
from dataclasses import dataclass, field
import nanotron
from nanotron.s3_checkpoints import check_path_is_local
from safetensors.torch import safe_open
from tqdm import tqdm
from nanotron.config import (
    ParallelismArgs,
    LlamaConfig,
    ExistingCheckpointInit,
    RandomInit,
    SpectralMupInit,
)
from nanotron.generation.generate_store import AttachableStore
from torch.nn.parallel import DistributedDataParallel
from nanotron.constants import CHECKPOINT_VERSION
from nanotron.logging import log_rank
from nanotron import logging
from nanotron.distributed import get_global_rank
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.models import NanotronModel
from nanotron.models import llama
from nanotron.models.llama import (
    CoreAttention,
    logger,
)
from nanotron.parallel.tied_parameters import get_tied_id_to_param
from nanotron.parallel.parameters import NanotronParameter
from nanotron.serialize import weights as nt_weights
from nanotron.serialize.weights import (
    get_checkpoint_version,
    read_checkpoint_version_from_shard_file,
    CheckpointVersionFromShardFileException,
    load_sharded_param_latest,
)
from nanotron.parallel import ParallelContext
from nanotron.serialize.utils import (
    ObjectType,
    get_exp_tp_pp_rank_and_size_from,
    get_path,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward


def create_custom_apply_rotary_pos_emb(cfg):
    def apply_rotary_pos_emb_v0(self, q, k, cos, sin, unsqueeze_dim=2):
        # Full-RoPE
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v1(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the fastest-rotating (high-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        keep_dim = cfg["top_k_rope_dim"]
        if keep_dim <= 0:
            return q, k
        elif keep_dim >= q.size(-1):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q_embed[..., :keep_dim],
                q[..., keep_dim:half],
                q_embed[..., half : half + keep_dim],
                q[..., half + keep_dim :],
            ),
            -1,
        )
        k_embed = torch.cat(
            (
                k_embed[..., :keep_dim],
                k[..., keep_dim:half],
                k_embed[..., half : half + keep_dim],
                k[..., half + keep_dim :],
            ),
            -1,
        )
        return q_embed, k_embed

    def apply_rotary_pos_emb_v2(self, q, k, cos, sin, unsqueeze_dim=2):
        # select subspaces with equidistant intervals
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        indices = torch.arange(
            cfg["uniform_start_point"], q.size(-1), cfg["uniform_step"], device=q.device
        )
        q,k = q.clone(),k.clone()
        q[..., indices] = q_embed[..., indices]
        k[..., indices] = k_embed[..., indices]
        return q, k

    def apply_rotary_pos_emb_v3(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the fastest-rotating (high-frequency) subspaces and the slowest-rotating (low-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        top_k_dim, last_k_dim = cfg["top_k_rope_dim"], cfg["last_k_rope_dim"]
        # assert top_k_dim + last_k_dim <= q.size(-1)
        half = q.size(-1) // 2
        qs = [
            q_embed[..., :top_k_dim],
            q[..., top_k_dim : half - last_k_dim],
            q_embed[..., half - last_k_dim : half + top_k_dim],
            q[..., half + top_k_dim : half + half - last_k_dim],
            q_embed[..., half + half - last_k_dim :],
        ]
        ks = [
            k_embed[..., :top_k_dim],
            k[..., top_k_dim : half - last_k_dim],
            k_embed[..., half - last_k_dim : half + top_k_dim],
            k[..., half + top_k_dim : half + half - last_k_dim],
            k_embed[..., half + half - last_k_dim :],
        ]
        q_embed = torch.cat([q for q in qs if q != []], -1)
        k_embed = torch.cat([k for k in ks if k != []], -1)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v4(self, q, k, cos, sin, layer_idx=0, unsqueeze_dim=2):
        # retain the subspaces with higher 2-norm score
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        top_k_dim = cfg["top_k_rope_dim"]
        qk_tensor[layer_idx] = qk_tensor[layer_idx].to(q.device)
        topk_indices = torch.topk(qk_tensor[layer_idx], k=top_k_dim, dim=1)[1]
        mask = torch.zeros_like(qk_tensor[layer_idx])
        mask.scatter_(1, topk_indices, 1)
        mask_for_k = (
            torch.cat((mask, mask), dim=1).unsqueeze(0).unsqueeze(1).to(q.device)
        )
        mask_for_q = torch.repeat_interleave(
            input=mask_for_k, repeats=cfg["n_gqa_group"], dim=2
        ).to(q.device)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v5(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the slowest-rotating (low-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        last_k_dim = cfg["last_k_rope_dim"]
        half = q.size(-1) // 2
        qs = [
            q[..., : half - last_k_dim],
            q_embed[..., half - last_k_dim : half],
            q[..., half : half + half - last_k_dim],
            q_embed[..., half + half - last_k_dim :],
        ]
        ks = [
            k[..., : half - last_k_dim],
            k_embed[..., half - last_k_dim : half],
            k[..., half : half + half - last_k_dim],
            k_embed[..., half + half - last_k_dim :],
        ]
        q_embed = torch.cat([q for q in qs if q != []], -1)
        k_embed = torch.cat([k for k in ks if k != []], -1)
        return q_embed, k_embed
    
    def apply_rotary_pos_emb_v6(self, q, k, cos, sin, layer_idx=0, unsqueeze_dim=2):
        # retain the subspaces with higher 2-norm score and different head could have different r
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        mask = mask_tensor[layer_idx].to(q.device)
        mask_for_k = (
            torch.cat((mask, mask), dim=1).unsqueeze(0).unsqueeze(1).to(q.device)
        )
        mask_for_q = torch.repeat_interleave(
            input=mask_for_k, repeats=cfg["n_gqa_group"], dim=2
        ).to(q.device)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v7(self, q, k, cos, sin, unsqueeze_dim=2):
        # retain the fastest-rotating (high-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        keep_dim = cfg["top_k_rope_dim"]
        if keep_dim <= 0:
            return q, k
        elif keep_dim >= q.size(-1):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q[..., :half-keep_dim],
                q_embed[..., half-keep_dim:half+keep_dim],
                q[..., half+keep_dim:],
            ),
            -1,
        )
        k_embed = torch.cat(
            (
                k[..., :half-keep_dim],
                k_embed[..., half-keep_dim:half+keep_dim],
                k[..., half+keep_dim:],
            ),
            -1,
        )
        return q_embed, k_embed

    def apply_rotary_pos_emb_v8(self, q, k, cos, sin, unsqueeze_dim=2):
        # Keep 2 * top_k_rope_dim dimensions from first half, nothing from second half
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        keep_dim = cfg["top_k_rope_dim"]
        if keep_dim <= 0:
            return q, k
        elif keep_dim >= q.size(-1):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q_embed[..., :2*keep_dim],
                q[..., 2*keep_dim:],
            ),
            -1,
        )
        k_embed = torch.cat(
            (
                k_embed[..., :2*keep_dim],
                k[..., 2*keep_dim:],
            ),
            -1,
        )
        return q_embed, k_embed

    def apply_rotary_pos_emb_v9(self, q, k, cos, sin, unsqueeze_dim=2):
        # Keep top_k_rope_head dimensions from first half, nothing from second half
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        keep_head = cfg["top_k_rope_head"]
        if keep_head <= 0:
            return q, k
        elif keep_head >= q.size(-2):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q_embed[..., :4 * keep_head, :],
                q[..., 4 * keep_head:, :],
            ),
            -2,
        )
        k_embed = torch.cat(
            (
                k_embed[..., :keep_head, :],
                k[..., keep_head:, :],
            ),
            -2,
        )
        return q_embed, k_embed

    def apply_rotary_pos_emb_v10(self, q, k, cos, sin, unsqueeze_dim=2):
        # Keep top_k_rope_head dimensions from first half, nothing from second half
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        keep_head = cfg["last_k_rope_head"]
        if keep_head <= 0:
            return q, k
        elif keep_head >= q.size(-2):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q_embed[..., -keep_head:, :],
                q[..., keep_head:, :],
            ),
            -1,
        )
        k_embed = torch.cat(
            (
                k_embed[..., :keep_head, :],
                k[..., keep_head:, :],
            ),
            -1,
        )
        return q_embed, k_embed

    version = cfg["partial_rope_version"]
    if version == 4 or version == 6:
        with open(cfg["qk_tensor_path"], "rb") as fin:
            qk_tensor = torch.load(fin).cuda()
    if version == 6:
        flattened_qk = qk_tensor.view(qk_tensor.size(0), -1)
        total_dim = qk_tensor.size(1) * cfg["top_k_rope_dim"]
        _, top_indices = torch.topk(flattened_qk, total_dim, dim=1)
        mask_tensor = torch.zeros_like(flattened_qk, dtype=torch.bool)
        mask_tensor.scatter_(1, top_indices, 1)
        mask_tensor = mask_tensor.view_as(qk_tensor)

    versions = {
        0: apply_rotary_pos_emb_v0,
        1: apply_rotary_pos_emb_v1,
        2: apply_rotary_pos_emb_v2,
        3: apply_rotary_pos_emb_v3,
        4: apply_rotary_pos_emb_v4,
        5: apply_rotary_pos_emb_v5,
        6: apply_rotary_pos_emb_v6,
        7: apply_rotary_pos_emb_v7,
        8: apply_rotary_pos_emb_v8,
        9: apply_rotary_pos_emb_v9,
        10: apply_rotary_pos_emb_v10,
    }
    return versions.get(version, apply_rotary_pos_emb_v0)


def custom_forward_with_hidden_states_for_v4(
    self,
    input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
):
    # all tensors are optional as most ranks don't need anything from the dataloader.

    output = self.token_position_embeddings(input_ids=input_ids, input_mask=input_mask)

    hidden_encoder_states = {
        "hidden_states": output["input_embeds"],
        "sequence_mask": input_mask,
    }
    # solve module_input_keys not match error
    if "layer_idx" not in self.decoder[0].module_input_keys:
        for layer_idx, encoder_block in enumerate(self.decoder):
            encoder_block.module_input_keys.add("layer_idx")
    for layer_idx, encoder_block in enumerate(self.decoder):
        hidden_encoder_states = encoder_block(
            **hidden_encoder_states, layer_idx=layer_idx
        )

    hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])[
        "hidden_states"
    ]

    sharded_logits = self.lm_head(x=hidden_states)["logits"]

    fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

    return fp32_sharded_logits, hidden_states


def custom_decoder_forward_for_v4(
    self,
    hidden_states: Union[torch.Tensor, TensorPointer],
    sequence_mask: Union[torch.Tensor, TensorPointer],
    layer_idx: int,
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
    if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
        hidden_states, sequence_mask = self._checkpointed_forward(
            hidden_states, sequence_mask
        )
    else:
        hidden_states, sequence_mask = self._core_forward(
            hidden_states, sequence_mask, layer_idx
        )

    return {
        "hidden_states": hidden_states,
        "sequence_mask": sequence_mask,
    }


def custom_decoder_core_forward_for_v4(
    self,
    hidden_states: Union[torch.Tensor, TensorPointer],
    sequence_mask: Union[torch.Tensor, TensorPointer],
    layer_idx: int,
) -> List[Union[torch.Tensor, TensorPointer]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    output = self.attn(
        hidden_states=hidden_states, sequence_mask=sequence_mask, layer_idx=layer_idx
    )
    hidden_states = output["hidden_states"]
    hidden_states = hidden_states + residual

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
    hidden_states = hidden_states + residual

    return hidden_states, output["sequence_mask"]


def custom_llama_forward_for_v4(
    self,
    hidden_states,  # [seq_length, batch_size, hidden_size]
    sequence_mask,  # [batch_size, seq_length]
    layer_idx,
):
    from flash_attn import bert_padding
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    qkv_states = self.qkv_proj(
        hidden_states
    )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
    q_length, batch_size, _ = qkv_states.shape

    if self.is_gqa:
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [
                self.n_local_q_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
            ],
            dim=-1,
        )

        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )
        key_states = (
            key_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
        value_states = (
            value_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
    else:
        query_states, key_states, value_states = (
            qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]

    store = self.get_local_store()
    if store is not None:  # Inference case
        # Double check that we use store only at inference time
        assert key_states.requires_grad is False
        assert value_states.requires_grad is False
        if "position_offsets" in store:
            old_position_offsets = store["position_offsets"]
            position_ids = old_position_offsets[:, None] + sequence_mask
        else:
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        position_offsets = position_ids[:, -1]

        # Compute rotary embeddings
        # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
        old_rotary_embed_end = self.rotary_embedding.end
        # interleaved version.
        if self.rope_interleaved:
            query_states = self.rotary_embedding(
                query_states, position_ids=position_ids
            )
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)
        # non interleaved version.
        else:
            cos, sin = self.rotary_embedding(value_states, position_ids)
            query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if "key" not in store:
            # First inference iteration (Prefill)
            # TODO @nouamane: support custom masking
            # assert that [ False, False, False, False,  True,  True,  True,  True,  True,  True] is accepted
            # but [ False, False, False, False,  True,  True,  False,  False,  True,  True] is not (can't mask in the middle of sequence)
            assert ~(
                sequence_mask[:, :-1]
                & (~sequence_mask[:, 1:])  # True is never followed by False
            ).any(), "Can't mask in the middle of sequence, please make sure that pads are at the left of the sequence if existing"

            # preallocate k_cache, v_cache to self.prefill_kv_len
            k_cache = torch.zeros(
                (
                    batch_size,
                    self.prefill_kv_len,
                    self.n_local_kv_heads,
                    self.d_qk,
                ),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            v_cache = torch.zeros(
                (batch_size, self.prefill_kv_len, self.n_local_kv_heads, self.d_v),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            # Remove pad tokens from key_states and concatenate samples in key_unpad
            # cu_seqlens_k is the cumulative sequence lengths of key_states
            (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = (
                bert_padding.unpad_input(
                    query_states,
                    sequence_mask,
                )
            )
            (key_unpad, indices_k, cu_seqlens_k, max_seqlen_k) = (
                bert_padding.unpad_input(key_states, sequence_mask)
            )
            (value_unpad, _, _, _) = bert_padding.unpad_input(
                value_states, sequence_mask
            )

            # NOTE: this scale is for µTransfer,
            # in SP, we use sqrt(1/d_h)
            softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
            output_unpad = flash_attn_varlen_func(
                q=query_unpad,  # (total_q, n_local_q_heads, d_qk)
                k=key_unpad,  # (total_kv, n_local_kv_heads, d_qk)
                v=value_unpad,  # (total_kv, n_local_kv_heads, d_v)
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,  # True in prefill phase, False in subsequent phases
                return_attn_probs=False,
            )  # (total_unpadded, n_local_q_heads, d_v)

            attention_output = bert_padding.pad_input(
                output_unpad, indices_q, batch_size, q_length
            )  # (batch_size, q_length, n_local_q_heads, d_v)

            pad_to_right(key_states, sequence_mask, new_tensor=k_cache)
            pad_to_right(value_states, sequence_mask, new_tensor=v_cache)

        else:
            # Pull pre-computed key/value states
            # Subsequent inference iterations (q_length=1)
            k_cache = store["key"]
            v_cache = store["value"]

            # NOTE(fmom): According to flash_attn_with_kvcache, "If you pass in k / v, you must make sure that the cache is large enough to hold the new values"
            # Since rotary embedding has changed (to enable larger context), we need to enlarge k_cache and v_cache
            if self.rotary_embedding.end > old_rotary_embed_end:
                k_cache = torch.cat(
                    [
                        k_cache,
                        torch.zeros(
                            (
                                batch_size,
                                self.rotary_embedding.end - old_rotary_embed_end,
                                self.n_local_kv_heads,
                                self.d_qk,
                            ),
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                    ],
                    dim=1,
                )

                v_cache = torch.cat(
                    [
                        v_cache,
                        torch.zeros(
                            (
                                batch_size,
                                self.rotary_embedding.end - old_rotary_embed_end,
                                self.n_local_kv_heads,
                                self.d_v,
                            ),
                            dtype=query_states.dtype,
                            device=query_states.device,
                        ),
                    ],
                    dim=1,
                )

            assert (
                k_cache.shape[1] == self.rotary_embedding.end
            ), f"Cache size {k_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"
            assert (
                v_cache.shape[1] == self.rotary_embedding.end
            ), f"Cache size {v_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"

            # [batch_size, seq_length, num_heads, d_qk]
            query_states = query_states.view(
                batch_size, q_length, self.n_local_q_heads, self.d_qk
            )  # [batch_size, q_length, self.n_heads, d_qk]
            kv_length = key_states.shape[1]
            key_states = key_states.view(
                batch_size, kv_length, self.n_local_kv_heads, self.d_qk
            )  # [batch_size, kv_length, self.n_heads, d_qk]
            value_states = value_states.view(
                batch_size, kv_length, self.n_local_kv_heads, self.d_v
            )  # [batch_size, kv_length, self.n_heads, d_v]

            # NOTE: this scale is for µTransfer,
            # in SP, we use sqrt(1/d_h)
            softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
            attention_output = flash_attn_with_kvcache(
                query_states,
                k_cache,
                v_cache,
                key_states,
                value_states,
                rotary_cos=None,
                rotary_sin=None,
                # TODO @nouamane: seems like this doesn't help to indicate padding in (for first iteration it's just 0)
                cache_seqlens=position_offsets.contiguous(),
                softmax_scale=softmax_scale,
                causal=True,
                rotary_interleaved=False,  # the value is not used unless rotary_cos/sin is provided. https://github.com/Dao-AILab/flash-attention
            )

        store.update(
            {
                "key": k_cache,  # flash-attn has updated with new key_states using cache_seqlens
                "value": v_cache,
                "position_offsets": position_offsets,
            }
        )

    else:  # Training case
        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # # [2, batch_size, seq_length, num_heads, d_qk]
        # key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # # [batch_size, seq_length, 2, num_heads, d_qk]
        # key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        # query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
        # # [batch_size, seq_length, num_heads, d_qk]
        # key_states, value_states = torch.split(key_value_states, 1, dim=2)
        position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        cos, sin = self.rotary_embedding(value_states, position_ids)
        query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, layer_idx
        )
        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        # [batch_size, seq_length, num_heads, d_qk]
        # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
        query_states = query_states.view(
            batch_size * q_length, self.n_local_q_heads, self.d_qk
        )  # [batch_size * q_length, self.n_heads, d_qk]

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )  # [batch_size * kv_length, self.n_heads, d_qk]
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )  # [batch_size * kv_length, self.n_heads, d_v]

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

    attention_output = (
        attention_output.contiguous()
        .view(batch_size, q_length, self.n_local_q_heads * self.d_v)
        .transpose(0, 1)
    )
    output = self.o_proj(attention_output)

    return {"hidden_states": output, "sequence_mask": sequence_mask}


def custom_llama_forward(
    self,
    hidden_states,  # [seq_length, batch_size, hidden_size]
    sequence_mask,  # [batch_size, seq_length]
):
    from flash_attn import bert_padding
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    qkv_states = self.qkv_proj(
        hidden_states
    )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
    q_length, batch_size, _ = qkv_states.shape

    if self.is_gqa:
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [
                self.n_local_q_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
                self.n_local_kv_heads * self.d_qk,
            ],
            dim=-1,
        )

        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )
        key_states = (
            key_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
        value_states = (
            value_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )
    else:
        query_states, key_states, value_states = (
            qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]

    store = self.get_local_store()
    if store is not None:  # Inference case
        raise NotImplementedError("Only training is supported with nanotron framework.")
    else:  # Training case
        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # # [2, batch_size, seq_length, num_heads, d_qk]
        # key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # # [batch_size, seq_length, 2, num_heads, d_qk]
        # key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        # query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=key_value_states)
        # # [batch_size, seq_length, num_heads, d_qk]
        # key_states, value_states = torch.split(key_value_states, 1, dim=2)
        position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        cos, sin = self.rotary_embedding(value_states, position_ids)
        query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        # [batch_size, seq_length, num_heads, d_qk]
        # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
        query_states = query_states.view(
            batch_size * q_length, self.n_local_q_heads, self.d_qk
        )  # [batch_size * q_length, self.n_heads, d_qk]

        key_states = key_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_qk
        )  # [batch_size * kv_length, self.n_heads, d_qk]
        value_states = value_states.view(
            batch_size * kv_length, self.n_local_kv_heads, self.d_v
        )  # [batch_size * kv_length, self.n_heads, d_v]

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            q_sequence_mask=q_sequence_mask,
            kv_sequence_mask=kv_sequence_mask,
        )

    attention_output = (
        attention_output.contiguous()
        .view(batch_size, q_length, self.n_local_q_heads * self.d_v)
        .transpose(0, 1)
    )
    output = self.o_proj(attention_output)

    return {"hidden_states": output, "sequence_mask": sequence_mask}


def pad_to_right(tensor, mask, new_tensor=None):
    """Transform a left-padded tensor into a right-padded tensor. (Useful for prefilling key/value states)
    Args:
        tensor: (batch_size, seqlen, d1, d2)
        mask: (batch_size, seqlen)
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
    Returns:
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
        right_padded_mask: (batch_size, seqlen)
    """
    # First, we need to find the number of padding for each row
    unpad_seqlens = mask.sum(1)
    # Then, we need to find the maximum length of the tensor
    max_seqlen = mask.shape[1]
    # We can then create the indices to select the padded values
    # The indices are the same for each row
    indices = torch.arange(max_seqlen, device=mask.device)
    # We can then create the mask for the padded values
    right_padded_mask = indices < unpad_seqlens[:, None]
    # We select the useful values
    useful_values = tensor[mask]
    # We create the new tensor (if not provided)
    new_tensor = torch.zeros_like(tensor) if new_tensor is None else new_tensor
    # We fill the new tensor with the useful values
    new_tensor[:, : right_padded_mask.shape[1], :, :][right_padded_mask] = useful_values
    return new_tensor, right_padded_mask


@dataclass
class CustomLlamaConfig(LlamaConfig):
    RoPE: Dict = field(default_factory=dict)
    SVD: Dict = field(default_factory=dict)


@dataclass
class CustomModelArgs(nanotron.config.config.ModelArgs):
    model_config: CustomLlamaConfig


@dataclass
class CustomConfig(nanotron.config.config.Config):
    """Main configuration class"""

    model: CustomModelArgs


class CustomCausalSelfAttention(nn.Module, AttachableStore):
    def __init__(
        self,
        config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        assert (
            not config.rope_interleaved
        ), "MLA Causal attention does not support interleaved RoPE"
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.config = config
        self.parallel_config = parallel_config
        self.tp_pg = tp_pg
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = (
            config.num_attention_heads != config.num_key_value_heads
        )  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup
        self.layer_idx = layer_idx
        self.nope_mask = IndexForNope.get_index_for_nope(
            config.RoPE,
            head_dim=self.d_qk,
            head_num=self.n_local_kv_heads,
            layer_idx=self.layer_idx,
        )
        self.is_share_W_down = bool(config.SVD["method"] not in [2, 3])
        self.low_rank = config.SVD["low_rank"]

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = (
            parallel_config.tp_mode
            if parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication
            if parallel_config is not None
            else False
        )

        # build the slice config for self.qkv for save/load
        # shard are done within the contiguous chunk
        self.q_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.W_k_r = TensorParallelColumnLinear(
            self.d_model,
            (self.nope_mask == False).sum().item(),
            bias=False,
            pg=tp_pg,
            mode=tp_mode,
            async_communication=tp_linear_async_communication,
        )
        self.W_down_k = TensorParallelColumnLinear(
            self.d_model,
            self.low_rank * self.n_local_kv_heads,
            bias=False,
            pg=tp_pg,
            mode=tp_mode,
            async_communication=tp_linear_async_communication,
        )
        if not self.is_share_W_down:
            self.W_down_v = TensorParallelColumnLinear(
                self.d_model,
                self.low_rank * self.n_local_kv_heads,
                bias=False,
                pg=tp_pg,
                mode=tp_mode,
                async_communication=tp_linear_async_communication,
            )
        self.W_up_k = TensorParallelColumnLinear(
            self.low_rank * self.n_local_kv_heads,
            self.n_local_kv_heads * self.d_qk - (self.nope_mask == False).sum().item(),
            bias=False,
            pg=tp_pg,
            mode=tp_mode,
            async_communication=tp_linear_async_communication,
        )
        self.W_up_v = TensorParallelColumnLinear(
            self.low_rank * self.n_local_kv_heads,
            self.n_local_kv_heads * self.d_v,
            bias=False,
            pg=tp_pg,
            mode=tp_mode,
            async_communication=tp_linear_async_communication,
        )
        # TODO(kunhao): We want to have only one version per device and not one version per layer.
        self.rotary_embedding = llama.LlamaRotaryEmbedding(
            dim=self.d_qk,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.rope_interleaved = config.rope_interleaved

        # NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)
        self.flash_rotary_embedding = FlashRotaryEmbedding(
            dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved
        )

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = (
            config.max_position_embeddings
        )  # TODO @nouamane: compute based on free memory, because in rope we can surpass max_position_embeddings

    def forward(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
    ):
        from flash_attn import bert_padding
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        query_states = self.q_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk]
        q_length, batch_size, _ = query_states.shape

        query_states = (
            query_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
        )  # [batch_size, seq_length, n_local_q_heads, d_qk]
        k_r = self.W_k_r(hidden_states)  # [seq_len, bsz, -1]
        if self.is_share_W_down:
            c_kv = self.W_down_k(
                hidden_states
            )  # [seq_length, batch_size, low_rank * n_local_kv_heads]
            k_c = self.W_up_k(c_kv)  # [seq_len, bsz, -1]
            value_states = self.W_up_v(c_kv)  # [seq_len, bsz, -1]
        else:
            c_kv = torch.cat(
                [self.W_down_k(hidden_states), self.W_down_v(hidden_states)], dim=-1
            )  # [seq_length, batch_size, 2 * low_rank * n_local_kv_heads]
            c_k, c_v = c_kv.split(
                [
                    self.low_rank * self.n_local_kv_heads,
                    self.low_rank * self.n_local_kv_heads,
                ],
                dim=-1,
            )
            k_c = self.W_up_k(c_k)  # [seq_len, bsz, -1]
            value_states = self.W_up_v(c_v)  # [seq_len, bsz, -1]

        key_states = torch.zeros(
            (q_length, batch_size, self.n_local_kv_heads * self.d_qk),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        key_states[..., ~self.nope_mask] = k_r
        key_states[..., self.nope_mask] = k_c
        key_states = (
            key_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
        )  # [batch_size, seq_length, n_local_kv_heads, d_qk]
        c_kv = (
            c_kv.transpose(0, 1).contiguous().view(batch_size, q_length, -1)
        )  # [batch_size, seq_length, -1]
        value_states = (
            value_states.transpose(0, 1)
            .contiguous()
            .view(batch_size, q_length, self.n_local_kv_heads, self.d_v)
        )  # [batch_size, seq_length, n_local_kv_heads, d_v]

        store = self.get_local_store()  # In fact, collections.defaultdict?
        if store is not None:  # Inference case
            raise NotImplementedError("Inference case not implemented yet because there is no need for inference using nanotron framework.") 
        else:  # Training case
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
            cos, sin = self.rotary_embedding(value_states, position_ids)
            dbg_key_states = key_states

            if self.config.RoPE["partial_rope_version"] == 4:
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    layer_idx=self.layer_idx,
                )
            else:
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            # [batch_size, seq_length, n_local_kv_heads, d_qk]
            # assert torch.allclose(
            #     dbg_key_states.view(
            #         batch_size, q_length, self.n_local_kv_heads * self.d_qk
            #     )[..., self.nope_mask],
            #     key_states.view(
            #         batch_size, q_length, self.n_local_kv_heads * self.d_qk
            #     )[..., self.nope_mask],
            # )

            q_sequence_mask = sequence_mask
            kv_sequence_mask = sequence_mask

            kv_length = key_states.shape[1]
            # [batch_size, seq_length, num_heads, d_qk]
            # Shaping for use in `flash-attn` version of flash-attn: `flash_attn_unpadded_func`
            query_states = query_states.view(
                batch_size * q_length, self.n_local_q_heads, self.d_qk
            )  # [batch_size * q_length, self.n_heads, d_qk]

            key_states = key_states.view(
                batch_size * kv_length, self.n_local_kv_heads, self.d_qk
            )  # [batch_size * kv_length, self.n_heads, d_qk]
            value_states = value_states.view(
                batch_size * kv_length, self.n_local_kv_heads, self.d_v
            )  # [batch_size * kv_length, self.n_heads, d_v]

            attention_output = self.attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                q_sequence_mask=q_sequence_mask,
                kv_sequence_mask=kv_sequence_mask,
            )

        attention_output = (
            attention_output.contiguous()
            .view(batch_size, q_length, self.n_local_q_heads * self.d_v)
            .transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {
            "hidden_states": output,
            "sequence_mask": sequence_mask,
        }


class IndexForNope:
    _qk_tensor_path = None
    _qk_tensor_cache = None

    @staticmethod
    def get_index_for_nope_v0(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        nope_mask = torch.zeros((head_dim), dtype=torch.bool)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v1(rope_cfg, **kwargs):
        keep_dim = rope_cfg["top_k_rope_dim"]
        head_dim = kwargs["head_dim"]
        if keep_dim <= 0:
            nope_mask = torch.ones((head_dim), dtype=torch.bool)
        elif keep_dim >= head_dim:
            nope_mask = torch.zeros((head_dim), dtype=torch.bool)
        else:
            half = head_dim // 2
            nope_mask = torch.ones((half), dtype=torch.bool)
            nope_mask[:keep_dim] = False
            nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v2(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        indices_to_remove = torch.arange(
            rope_cfg["uniform_start_point"], head_dim, rope_cfg["uniform_step"]
        )
        nope_mask = torch.ones(head_dim, dtype=torch.bool)
        nope_mask[indices_to_remove] = False
        return nope_mask

    @staticmethod
    def get_index_for_nope_v3(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        top_k_dim, last_k_dim = rope_cfg["top_k_rope_dim"], rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        assert top_k_dim + last_k_dim <= half
        nope_mask = torch.zeros((half), dtype=torch.bool)
        nope_mask[top_k_dim : half - last_k_dim] = True
        nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v4(rope_cfg, **kwargs):
        if IndexForNope._qk_tensor_cache is None or rope_cfg["qk_tensor_path"] != IndexForNope._qk_tensor_path:
            with open(rope_cfg["qk_tensor_path"], "rb") as fin:
                IndexForNope._qk_tensor_cache = torch.load(fin)  # [layer_num, k_head_num, head_dim//2]
                IndexForNope._qk_tensor_path = rope_cfg["qk_tensor_path"]
                assert len(IndexForNope._qk_tensor_cache.shape) == 3
        qk_tensor = IndexForNope._qk_tensor_cache
        layer_idx = kwargs["layer_idx"]
        top_k_dim = rope_cfg["top_k_rope_dim"]
        topk_indices = torch.topk(qk_tensor[layer_idx], k=top_k_dim, dim=1)[1]
        nope_mask = torch.ones_like(qk_tensor[layer_idx], dtype=torch.bool)
        nope_mask.scatter_(1, topk_indices, False)
        nope_mask = torch.cat([nope_mask, nope_mask], dim=-1)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v5(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        last_k_rope_dim = rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        nope_mask = torch.ones((half), dtype=torch.bool)
        nope_mask[half-last_k_rope_dim:half] = False
        nope_mask = torch.cat([nope_mask, nope_mask], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v7(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        top_k_rope_dim = rope_cfg["top_k_rope_dim"]
        half = head_dim // 2
        nope_mask_first_half = torch.ones((half), dtype=torch.bool)
        # Keep last top_k_rope_dim dimensions of first half
        nope_mask_first_half[half-top_k_rope_dim:half] = False
        # Keep first top_k_rope_dim dimensions of second half
        nope_mask_second_half = torch.ones((half), dtype=torch.bool)
        nope_mask_second_half[:top_k_rope_dim] = False
        nope_mask = torch.cat([nope_mask_first_half, nope_mask_second_half], dim=0)
        return nope_mask

    @staticmethod
    def get_index_for_nope_v8(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        top_k_rope_dim = rope_cfg["top_k_rope_dim"]
        nope_mask = torch.ones((head_dim), dtype=torch.bool)
        nope_mask[:2*top_k_rope_dim] = False
        return nope_mask

    @staticmethod
    def get_index_for_nope_v9(rope_cfg, **kwargs):
        head_num = kwargs["head_num"]
        top_k_rope_head = rope_cfg["top_k_rope_head"]
        nope_mask = torch.ones((head_num), dtype=torch.bool)
        nope_mask[:top_k_rope_head] = False
        return nope_mask

    @staticmethod
    def get_index_for_nope_v10(rope_cfg, **kwargs):
        head_num = kwargs["head_num"]
        last_k_rope_head = rope_cfg["last_k_rope_head"]
        nope_mask = torch.ones((head_num), dtype=torch.bool)
        nope_mask[-last_k_rope_head:] = False
        return nope_mask

    @staticmethod
    def get_index_for_nope(rope_cfg, **kwargs):
        logger.info(f"rope_cfg: {rope_cfg}")
        version = rope_cfg["partial_rope_version"]
        versions = {
            0: IndexForNope.get_index_for_nope_v0,
            1: IndexForNope.get_index_for_nope_v1,
            2: IndexForNope.get_index_for_nope_v2,
            3: IndexForNope.get_index_for_nope_v3,
            4: IndexForNope.get_index_for_nope_v4,
            5: IndexForNope.get_index_for_nope_v5,
            7: IndexForNope.get_index_for_nope_v7,
            8: IndexForNope.get_index_for_nope_v8,
            9: IndexForNope.get_index_for_nope_v9,
            10: IndexForNope.get_index_for_nope_v10,
        }
        index_func = versions[version]
        nope_mask = index_func(rope_cfg, **kwargs)
        nope_mask = nope_mask.to(dtype=torch.bool)
        if version == 4:
            nope_mask = nope_mask.reshape(-1)
        elif version == 9 or version == 10:
            nope_mask = nope_mask.repeat_interleave(kwargs["head_dim"])
        else:
            nope_mask = nope_mask.repeat(repeats=(kwargs["head_num"],))
        return nope_mask


class SvdInit:
    @staticmethod
    def method_I(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down = (U_k[:, :r] + U_v[:, :r]) / 2
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_II(k, v, r=8):
        # Separately decompose W_k_nope and W_v into truncated SVDs, allocating dimensions to each
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down_k = U_k
        W_down_v = U_v
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down_k.t(), W_up_k.t(), W_down_v.t(), W_up_v.t()

    @staticmethod
    def method_III(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        Sigma_k_half = torch.diag(torch.sqrt(S_k))
        Sigma_v_half = torch.diag(torch.sqrt(S_v))
        W_down_k = U_k @ Sigma_k_half
        W_down_v = U_v @ Sigma_v_half
        W_up_k = Sigma_k_half @ V_k.t()
        W_up_v = Sigma_v_half @ V_v.t()
        return W_down_k.t(), W_up_k.t(), W_down_v.t(), W_up_v.t()

    @staticmethod
    def method_IV(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        Sigma_k_half = torch.diag(torch.sqrt(S_k))
        Sigma_v_half = torch.diag(torch.sqrt(S_v))
        W_down_k = U_k @ Sigma_k_half
        W_down_v = U_v @ Sigma_v_half
        W_down = (W_down_k + W_down_v) / 2
        W_up_k = Sigma_k_half @ V_k.t()
        W_up_v = Sigma_v_half @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_V(k, v, r=8):
        U_k, S_k, V_k = torch.svd(k)
        U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
        W_down = U_k
        W_down_pseudo_inv = torch.linalg.pinv(W_down)
        W_up_k = torch.diag(S_k) @ V_k.t()
        W_up_v = torch.matmul(W_down_pseudo_inv, v)
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_VI(k, v, r=8):
        U_v, S_v, V_v = torch.svd(v)
        U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
        W_down = U_v
        W_down_pseudo_inv = torch.linalg.pinv(W_down)
        W_up_k = torch.matmul(W_down_pseudo_inv, k)
        W_up_v = torch.diag(S_v) @ V_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def method_VII(k, v, r=8):
        # jointly factorize the con-catenated matrix
        U_kv, S_kv, V_kv = torch.svd(torch.cat([k, v], dim=1))
        U_kv, S_kv, V_kv = U_kv[:, :r], S_kv[:r], V_kv[:, :r]
        W_down = U_kv
        split_sizes = [k.size(1), v.size(1)]
        W_up_k, W_up_v = torch.split(V_kv, split_sizes, dim=0)
        W_up_k = torch.diag(S_kv) @ W_up_k.t()
        W_up_v = torch.diag(S_kv) @ W_up_v.t()
        return W_down.t(), W_up_k.t(), None, W_up_v.t()

    @staticmethod
    def init(k, v, svd_method=1, r=8):
        assert k.dtype == v.dtype, "k and v must have the same dtype"
        logger.info(f"Using SVD method {svd_method} with rank {r}")
        original_dtype = k.dtype
        k=k.to(torch.float32)
        v=v.to(torch.float32)
        versions = {
            1: SvdInit.method_I,
            2: SvdInit.method_II,
            3: SvdInit.method_III,
            4: SvdInit.method_IV,
            5: SvdInit.method_V,
            6: SvdInit.method_VI,
            7: SvdInit.method_VII,
        }
        W_down_k, W_up_k, W_down_v, W_up_v = versions[svd_method](k, v, r)
        W_down_k = W_down_k.to(original_dtype)
        W_up_k = W_up_k.to(original_dtype)
        if W_down_v is not None:
            W_down_v = W_down_v.to(original_dtype)
        W_up_v = W_up_v.to(original_dtype)
        return W_down_k, W_up_k, W_down_v, W_up_v


def custom_load_weights(
    model: nn.Module,
    parallel_context: ParallelContext,
    root_folder: Path,
    filtered_state_dict: Optional[Dict[str, Any]] = None,
):
    """Load weights from a checkpoint

    Args:
        model: model to load weights into
        parallel_context: distributed process groups
        root_folder: root folder of the checkpoint
        filtered_state_dict: state dict to load from (overrides model.state_dict()). if None, load from model.state_dict()
    """
    if all(
        ["W_down_k" not in module_name for module_name in model.state_dict()]
    ) or any(
        [
            "W_down_k" in str(file_name.absolute())
            for file_name in (root_folder / "model").glob("**/*")
        ]
    ):
        return nt_weights.original_load_weights(
            model, parallel_context, root_folder, filtered_state_dict
        )
    param_root_folder = root_folder / "model"

    module_id_to_prefix = {
        id(module): f"{module_name}." for module_name, module in model.named_modules()
    }
    # Fix the root_model
    module_id_to_prefix[id(model)] = ""

    checkpoint_version: Optional[Version] = None

    filtered_state_dict = (
        filtered_state_dict if filtered_state_dict is not None else model.state_dict()
    )
    param_shard_metadata = {}
    for name, param_or_buffer in tqdm(
        filtered_state_dict.items(),
        disable=dist.get_rank(parallel_context.world_pg) != 0,
        desc="Loading weights",
    ):
        if any(
            [
                mla_weight in name
                for mla_weight in [
                    "W_down_k",
                    "W_down_v",
                    "W_up_k",
                    "W_up_v",
                    "W_k_r",
                    "q_proj",
                ]
            ]
        ):
            continue
        # NOTE: extract how does the current model parameter are sharded
        # so that we can load optimizer checkpoints in this way
        param_shard_metadata[name] = {}
        # `state_dict` doesn't return a Param or a buffer, just a tensors which loses some metadata
        try:
            param = model.get_parameter(name)
        except AttributeError:
            param = None

        if isinstance(param, NanotronParameter):
            if param.is_tied:
                tied_info = param.get_tied_info()
                base_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                base_name = name

            if param.is_sharded:
                sharded_info = param.get_sharded_info()

                if param.is_tied:
                    # When params are tied only the first rank of tied param group stores weights (see save_weights)
                    group = parallel_context.world_ranks_to_pg[tied_info.global_ranks]
                    group_rank = 0
                else:
                    group = parallel_context.world_ranks_to_pg[
                        sharded_info.global_ranks
                    ]
                    group_rank = dist.get_rank(group)

                exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
                    world_rank=get_global_rank(group=group, group_rank=group_rank),
                    parallel_context=parallel_context,
                )
                # TODO @nouamane: do we consider exp_size=1 expert_sharded?
                is_expert_sharded = sharded_info.is_expert_sharded(parallel_context)
            else:
                exp_tp_pp_rank_and_size = None
                is_expert_sharded = False

            path = get_path(
                base_name,
                type=ObjectType.MODEL,
                exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
                prefix=param_root_folder,
                is_expert_sharded=is_expert_sharded,
            )

            if path.exists():
                with safe_open(path, framework="pt", device=str(param.device)) as fi:
                    # TODO @thomasw21: Choose only a slice if we switch the TP topology
                    param_or_buffer[:] = fi.get_tensor("data")

            elif not path.parent.exists():
                raise ValueError(
                    f"Checkpoint is empty or checkpoint structure is not matching the model architecture."
                    f"Couldn't find folder {path.parent} in checkpoint at {root_folder}"
                )
            else:
                # Let's assume that the topology changed and the param is sharded.
                # We search for all the files from the shards, concatenate the "unsharded" tensor
                # and load the specific shard we're interested in.
                if not param.is_sharded:
                    raise ValueError(
                        f"`{name}` is not a sharded parameter. It's possible you were expecting {path} to exist."
                    )
                # TODO @thomasw21: Make so that we don't need to code this logic somewhere else than in `get_path`
                sharded_info = param.get_sharded_info()
                suffix = base_name.rsplit(".", 1)[-1]
                shards_path = list(
                    path.parent.glob(f"{ObjectType.MODEL.value}_{suffix}*.safetensors")
                )
                if len(shards_path) <= 0:
                    raise ValueError(
                        f"Could not find any shards {ObjectType.MODEL.value}_{suffix}*.safetensors in {path.parent}."
                        f"If you notice `.safetensors` in the middle of the name of some of the checkpoints files. You need to run `scripts/fix_checkpoint_bad_naming.py`."
                    )

                if checkpoint_version is None:
                    checkpoint_version = get_checkpoint_version(
                        parallel_context, root_folder, param_save_path=shards_path[0]
                    )
                else:
                    current_checkpoint_version = None
                    try:
                        current_checkpoint_version = (
                            read_checkpoint_version_from_shard_file(
                                param_save_path=shards_path[0]
                            )
                        )
                    except CheckpointVersionFromShardFileException:
                        # The checkpoint version is read from the meta file
                        current_checkpoint_version = checkpoint_version
                    finally:
                        assert (
                            current_checkpoint_version == checkpoint_version
                        ), f"Checkpoint version mismatch at {shards_path[0]}."

                if checkpoint_version <= CHECKPOINT_VERSION:
                    load_sharded_param_latest(
                        param_or_buffer=param_or_buffer,
                        sharded_info=sharded_info,
                        shards_path=shards_path,
                        param_shard_metadata=param_shard_metadata[name],
                    )
                else:
                    raise ValueError(
                        f"Unsupported checkpoint version {checkpoint_version}"
                    )

        else:
            raise NotImplementedError(
                f"Parameters {param} should be a NanotronParameter"
            )

    for layer_idx in tqdm(range(len(model.model.decoder)), desc="Loading MLA weights"):
        base_name = f"model.decoder.{layer_idx}.pp_block.attn.qkv_proj.weight"
        exp_tp_pp_rank_and_size = get_exp_tp_pp_rank_and_size_from(
            world_rank=get_global_rank(group=group, group_rank=group_rank),
            parallel_context=parallel_context,
        )
        path = get_path(
            base_name,
            type=ObjectType.MODEL,
            exp_tp_pp_rank_and_size=exp_tp_pp_rank_and_size,
            prefix=param_root_folder,
            is_expert_sharded=True,
        )
        with safe_open(path, framework="pt", device=str(param.device)) as fi:
            qkv_proj = fi.get_tensor("data").t()
        attn_module_prefix = f"model.decoder.{layer_idx}.pp_block.attn"
        attn_module = model.model.decoder[layer_idx].pp_block.attn
        nope_mask = attn_module.nope_mask
        q_proj, k_proj, v_proj = qkv_proj.split(
            [
                attn_module.n_local_q_heads * attn_module.d_qk,
                attn_module.n_local_kv_heads * attn_module.d_qk,
                attn_module.n_local_kv_heads * attn_module.d_v,
            ],
            dim=-1,
        )
        filtered_state_dict[f"{attn_module_prefix}.q_proj.weight"][:] = q_proj.t()
        filtered_state_dict[f"{attn_module_prefix}.W_k_r.weight"][:] = k_proj[
            ..., ~nope_mask
        ].t()
        W_down_k, W_up_k, W_down_v, W_up_v = SvdInit.init(
            k_proj[..., nope_mask],
            v_proj,
            svd_method=model.config.SVD["method"],
            r=model.config.SVD["low_rank"] * model.config.num_key_value_heads,
        )
        filtered_state_dict[f"{attn_module_prefix}.W_down_k.weight"][:] = W_down_k
        filtered_state_dict[f"{attn_module_prefix}.W_up_k.weight"][:] = W_up_k
        if not attn_module.is_share_W_down:
            filtered_state_dict[f"{attn_module_prefix}.W_down_v.weight"][:] = W_down_v
        filtered_state_dict[f"{attn_module_prefix}.W_up_v.weight"][:] = W_up_v

    return param_shard_metadata


def partial_rope_monkey_patch(rope_cfg):
    if rope_cfg is not None:
        llama.LlamaRotaryEmbedding.apply_rotary_pos_emb = (
            create_custom_apply_rotary_pos_emb(rope_cfg)
        )
    if "CustomLlamaConfig" in nanotron.trainer.CONFIG_TO_MODEL_CLASS:
        return
    nanotron.config.models_config.LlamaConfig = CustomLlamaConfig
    nanotron.trainer.CONFIG_TO_MODEL_CLASS.update(
        {"CustomLlamaConfig": nanotron.trainer.CONFIG_TO_MODEL_CLASS["LlamaConfig"]}
    )
    if rope_cfg is not None and rope_cfg["partial_rope_version"]==4:
        from nanotron.models.llama import LlamaModel, LlamaDecoderLayer, CausalSelfAttention

        LlamaModel.forward_with_hidden_states = custom_forward_with_hidden_states_for_v4
        LlamaDecoderLayer.forward = custom_decoder_forward_for_v4
        LlamaDecoderLayer._core_forward = custom_decoder_core_forward_for_v4
        CausalSelfAttention.forward = custom_llama_forward_for_v4


def mla_monkey_patch(rope_cfg=None):
    llama.CausalSelfAttention = CustomCausalSelfAttention
    if not hasattr(nt_weights, "original_load_weights"):
        nt_weights.original_load_weights = nt_weights.load_weights
        nt_weights.load_weights = custom_load_weights
    nanotron.config.models_config.LlamaConfig = CustomLlamaConfig
    nanotron.trainer.CONFIG_TO_MODEL_CLASS.update(
        {"CustomLlamaConfig": nanotron.trainer.CONFIG_TO_MODEL_CLASS["LlamaConfig"]}
    )
    if rope_cfg["partial_rope_version"]==4:
        IndexForNope._qk_tensor_cache=torch.load(rope_cfg["qk_tensor_path"])
        IndexForNope._qk_tensor_path=rope_cfg["qk_tensor_path"]

    def custom_load_model_checkpoint(self, model: NanotronModel) -> NanotronModel:
        unwrapped_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )

        # Load or initialize model weights
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Load from a pre existing checkpoint
            if check_path_is_local(self.init_checkpoint_path):
                # Reload from a training checkpoint
                log_rank(
                    f"Loading weights from {self.init_checkpoint_path}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )
                self.param_shard_metadata = nt_weights.load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.init_checkpoint_path,
                )
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank(
                "No checkpoint path provided.",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint (without optimizer, lr_scheduler...)
                self.param_shard_metadata = nt_weights.load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(
                self.config.model.init_method, (RandomInit, SpectralMupInit)
            ):
                unwrapped_model.init_model_randomly(config=self.config)

                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for _, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(
                        param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg
                    )

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=unwrapped_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")
        return model

    nanotron.trainer.DistributedTrainer._load_model_checkpoint = (
        custom_load_model_checkpoint
    )

    if rope_cfg is not None:
        partial_rope_monkey_patch(rope_cfg)
