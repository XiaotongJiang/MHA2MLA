# copied from https://github.com/huggingface/nanotron/blob/main/examples/llama/convert_nanotron_to_hf.py
"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc_per_node=1 convert_nanotron_to_hf.py --checkpoint_path=nanotron-path --save_path=hf-path
"""
import json,os
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional

# Ensure the parent src directory is in Python path to allow absolute imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model
from nanotron.config import LlamaConfig as NanotronLlamaConfig
from nanotron.models import init_on_device_and_dtype
from nanotron.models.llama import LlamaForTraining
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

TEST_PROMPT = "What is the meaning of the word chutzpah?\nThe word chutzpah means"


def _handle_attention_block(
    qkv: torch.Tensor, part: Literal["q", "k", "v"], n_q_heads: int, n_kv_heads: int, d_qk: int
) -> torch.Tensor:
    # Huggingface Llama separates the q, k, v weights (as opposed to nanotron).
    # Furthermore, in the rotary embeddings in nanotron expects interleaved pairs of even
    # and odd dimensions GPT-J style, while the huggingface implementation expects
    # the whole 1st half and then the whole 2nd half GPT-NeoX style (for more information
    # see flash_attn.layers.rotary.RotaryEmbedding).
    # This function selects the proper chunk of the bundled qkv tensor and permutation
    # to ensure correct transformation to huggingface.

    def interleave(w: torch.Tensor):
        return w
        # w_new = []
        # for head_w in w.split(d_qk):
        #     head_w = head_w.view(d_qk // 2, 2, -1).transpose(0, 1).reshape(d_qk, -1)
        #     w_new.append(head_w)
        # return torch.cat(w_new)

    assert part in ["q", "k", "v"], "part must be one of [q, k, v]"

    index_end_q = n_q_heads * d_qk
    index_end_k = index_end_q + n_kv_heads * d_qk
    if part == "q":
        return interleave(qkv[:index_end_q])
    if part == "k":
        return interleave(qkv[index_end_q:index_end_k])
    return qkv[index_end_k:]


def _handle_gate_up_proj(gate_up_proj: torch.Tensor, gate: bool) -> torch.Tensor:
    # The gate and up projection are bundled in nanotron.
    # This function selects the proper chunk in the bundled weights to return
    # either the gate or the up projection only.
    weight_size = gate_up_proj.shape[0] // 2
    if gate:
        return gate_up_proj[:weight_size]
    else:
        return gate_up_proj[weight_size:]


def convert_nt_to_hf(nanotron_model: LlamaForTraining, hf_model: LlamaForCausalLM, model_config: NanotronLlamaConfig):
    """Converts the weights from the nanotron_model to hf_model, making modifications
    in-place."""

    nanotron_model_state_dict = nanotron_model.state_dict()

    hf_to_nt = get_weight_mapping(model_config, nt_to_hf=False)
    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            # Get the Nanotron parameter
            nanotron_key = hf_to_nt[f"{module_name_hf}.{param_name_hf}"]
            param = nanotron_model_state_dict[nanotron_key]

            if "qkv_proj" in nanotron_key:
                proj_name = module_name_hf.split(".")[4][0]
                param = _handle_attention_block(
                    param,
                    proj_name,
                    model_config.num_attention_heads,
                    model_config.num_key_value_heads,
                    model_config.hidden_size // model_config.num_attention_heads,
                )

            elif "gate_up_proj" in nanotron_key:
                gate = "gate" in module_name_hf
                param = _handle_gate_up_proj(param, gate)

            with torch.no_grad():
                param_hf.copy_(param)


def get_hf_config(config: NanotronLlamaConfig) -> HFLlamaConfig:
    """Converts a nanotron configuration to huggingface configuration."""
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=False).items() if hasattr(config, value)}
    return HFLlamaConfig(**attrs)

def shuffle_q_proj_based_on_pe_nope(attn, rope_config: dict, q_head_num: int):
    if rope_config['partial_rope_version'] == 1:
        original_q_proj = attn.q_proj.weight
        original_q_proj_per_head = original_q_proj.view(original_q_proj.shape[0], q_head_num, -1)

        original_w_k_r = attn.W_k_r.weight
        original_w_down_k = attn.W_down_k.weight
        original_w_up_k = attn.W_up_k.weight
        # shuffle
        keep_dim = rope_config['top_k_rope_dim']
        half = original_q_proj_per_head.size(-1) // 2
        q_proj_nope = torch.cat(
            (
                original_q_proj_per_head[..., keep_dim:half],
                original_q_proj_per_head[..., half + keep_dim :],
            ),
            dim=-1
        ).view(original_q_proj_per_head.shape[0], -1)
        # q_proj_nope_to_be_absorbed = q_proj_nope.view(q_proj_nope.shape[0], rope_config["n_gqa_group"], -1) # shape: (2048, 4, 256)
        # q_proj_nope_absorbed = torch.matmul(q_proj_nope_to_be_absorbed, original_w_up_k.T).reshape(q_proj_nope_to_be_absorbed.shape[0], -1)
        # # shape q_proj_nope_absorbed: (output 2048, input: 4, 256), original_w_up_k: (output 256, input 256) and q_proj_nope_absorbed is (2048, 1024)

        q_proj_rope = torch.cat(
            (
                original_q_proj_per_head[..., :keep_dim],
                original_q_proj_per_head[..., half : half + keep_dim],
            ),
            dim=-1
        ).view(original_q_proj_per_head.shape[0], -1)

        new_q_proj = torch.cat((q_proj_nope, q_proj_rope), dim=-1)
        attn.q_proj = torch.nn.Linear(new_q_proj.shape[1], new_q_proj.shape[0], bias=False)
        attn.q_proj.weight = torch.nn.Parameter(new_q_proj)

    else:
        # TODO implement for other partial_rope_version
        return

def post_convert_hf_model_for_sglang(hf_model: LlamaForCausalLM, model_config: NanotronLlamaConfig):
    # Combine w_up_v and o_proj weights for MLA optimization
    for layer in hf_model.model.layers:
        # Q_proj = [(Q_proj_nope) x (W_up_k)] + Q_proj_rope]
        shuffle_q_proj_based_on_pe_nope(layer.self_attn, model_config.RoPE, model_config.num_attention_heads)

        # kv_a_proj_with_mqa = [W_down_k, W_k_r]
        kv_a_proj_with_mqa = torch.cat([layer.self_attn.W_down_k.weight, layer.self_attn.W_k_r.weight], dim=0)
        layer.self_attn.kv_a_proj_with_mqa = torch.nn.Linear(kv_a_proj_with_mqa.shape[1], kv_a_proj_with_mqa.shape[0], bias=False)
        layer.self_attn.kv_a_proj_with_mqa.weight = torch.nn.Parameter(kv_a_proj_with_mqa)
        
        # kv_b_proj = [W_up_k, W_up_v]
        kv_b_proj = torch.cat([layer.self_attn.W_up_k.weight, layer.self_attn.W_up_v.weight], dim=0) # need to confirm??
        layer.self_attn.kv_b_proj = torch.nn.Linear(kv_b_proj.shape[1], kv_b_proj.shape[0], bias=False)
        layer.self_attn.kv_b_proj.weight = torch.nn.Parameter(kv_b_proj)

        # O_proj = o_proj * w_up_v
        o_proj = layer.self_attn.o_proj.weight
        w_up_v = layer.self_attn.W_up_v.weight.repeat(o_proj.shape[0] // layer.self_attn.W_up_v.weight.shape[0], 1) # repeat for group
        new_o_proj = torch.matmul(o_proj, w_up_v)
        layer.self_attn.o_proj = torch.nn.Linear(new_o_proj.shape[1], new_o_proj.shape[0], bias=False)
        layer.self_attn.o_proj.weight = torch.nn.Parameter(new_o_proj)


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path, tokenizer_name: Optional[str] = None, for_sglang: bool = False):
    """Loads the nanotron checkpoint in `checkpoint_path`, creates
    a new huggingface instance, copies the weights from the nanotron checkpoint
    and saves the transformed huggingface to `save_path`."""

    # Init nanotron model.
    with open(checkpoint_path / "model_config.json", "r") as f:
        attrs = json.load(f)
        model_config = NanotronLlamaConfig(**attrs)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
    )
    # Init huggingface model.
    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        model_config_hf = get_hf_config(model_config)
        hf_model = LlamaForCausalLM._from_config(model_config_hf)

    # Copy weights, initialize tokenizer and save model.
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(save_path)
    convert_nt_to_hf(nanotron_model, hf_model, model_config)
    post_convert_hf_model_for_sglang(hf_model, model_config)
    hf_model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


def check_converted_model_generation(save_path: Path):
    """Loads a huggingface model and tokenizer from `save_path` and
    performs a dummy text generation."""

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    input_ids = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].cuda()
    print("Inputs:", tokenizer.batch_decode(input_ids))

    model = LlamaForCausalLM.from_pretrained(save_path).cuda().bfloat16()
    out = model.generate(input_ids, max_new_tokens=100)
    print("Generation (converted): ", tokenizer.batch_decode(out))


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Nanotron weights to HF format")
    parser.add_argument("--checkpoint_path", type=Path, default="llama-7b", help="Path to the checkpoint")
    parser.add_argument("--save_path", type=Path, default="llama-7b-hf", help="Path to save the HF model")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--is_mla", action="store_true", help="Whether the model is an MLA model")
    parser.add_argument("--auto_encoder", action="store_true", help="Whether the model is using auto-encoder")
    parser.add_argument("--for_sglang", action="store_true", help="Whether the hf model will be served for sglang")
    args = parser.parse_args()
    with open(os.path.join(args.checkpoint_path,"model_config.json")) as f:
        config = json.load(f)
    if "RoPE" in config:
        # partial RoPE
        from mha2mla.monkey_patch import partial_rope_monkey_patch as partial_rope_monkey_patch_hf
        from mha2mla_nt.monkey_patch import CustomLlamaConfig,partial_rope_monkey_patch as partial_rope_monkey_patch_nt
        partial_rope_monkey_patch_hf(config["RoPE"])
        partial_rope_monkey_patch_nt(config["RoPE"])
        globals()["NanotronLlamaConfig"] = CustomLlamaConfig
    if args.is_mla:
        from mha2mla.monkey_patch import mla_monkey_patch as mla_monkey_patch_hf
        from mha2mla_nt.monkey_patch import mla_monkey_patch as mla_monkey_patch_nt
        with open(os.path.join(args.checkpoint_path,"model_config.json")) as f:
            config = json.load(f)
        mla_monkey_patch_hf(config["RoPE"])
        mla_monkey_patch_nt(config["RoPE"])

    if args.auto_encoder:
        with open(os.path.join(args.checkpoint_path,"model_config.json")) as f:
            config = json.load(f)
        from auto_encoder.patch_func_hf import ae_patch_func_hf
        from auto_encoder.patch_func_nt import ae_patch_func_nt,CustomLlamaConfig
        ae_patch_func_nt(config["RoPE"])
        ae_patch_func_hf(config["RoPE"])
        globals()["NanotronLlamaConfig"] = CustomLlamaConfig
    if not args.is_mla and not args.auto_encoder:
        from original_convert_weights import get_weight_mapping as original_get_weight_mapping
        from original_convert_weights import load_nanotron_model as original_load_nanotron_model
        get_weight_mapping = original_get_weight_mapping
        load_nanotron_model = original_load_nanotron_model
    # Convert Nanotron model to HF format.
    convert_checkpoint_and_save(
        checkpoint_path=args.checkpoint_path, save_path=args.save_path, tokenizer_name=args.tokenizer_name, for_sglang=args.for_sglang
    )

    # Check if the conversion was successful by generating some text.
    if args.tokenizer_name is not None:
        check_converted_model_generation(save_path=args.save_path)
