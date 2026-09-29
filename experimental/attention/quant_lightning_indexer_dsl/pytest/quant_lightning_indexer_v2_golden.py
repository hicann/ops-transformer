# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch_npu

from qli_test_utils import quant_lightning_indexer_golden
from qliv2_test_utils import PARAM_NAMES

PAGE_SIZE = 128
N1 = 32
N2 = 1
PACKED_D = 64
FP4_PACK_NUM = 2
E8M0_ONE_VALUE = 127
FP4_E2M1_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


def _range_pair(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid {name}: {value!r}")
    return low, high


def _make_mxfp4_bytes(logical_shape, data_range, data_dtype, generator):
    if data_dtype != torch.float4_e2m1fn_x2:
        raise TypeError(
            "MXFP4 requires qk_dtype=FLOAT4_E2M1/torch.float4_e2m1fn_x2, "
            f"got {data_dtype}"
        )
    low, high = _range_pair(data_range, "FP4 E2M1 numeric range")
    if logical_shape[-1] % FP4_PACK_NUM:
        raise ValueError(f"MXFP4 head_dim must be even, got {logical_shape[-1]}")
    valid_codes = (
        torch.nonzero(
            (FP4_E2M1_VALUES >= low) & (FP4_E2M1_VALUES <= high),
            as_tuple=False,
        )
        .flatten()
        .to(torch.uint8)
    )
    if valid_codes.numel() == 0:
        raise ValueError(
            f"FP4 E2M1 range contains no representable value: {data_range}"
        )
    raw_indices = torch.randint(
        valid_codes.numel(),
        tuple(logical_shape),
        generator=generator,
        dtype=torch.long,
    )
    raw = valid_codes[raw_indices]
    raw_rows = raw.reshape(-1, logical_shape[-1])
    coverage_count = min(valid_codes.numel(), logical_shape[-1])
    coverage = torch.arange(coverage_count).unsqueeze(0) + torch.arange(
        raw_rows.shape[0]
    ).unsqueeze(1)
    raw_rows[:, :coverage_count] = valid_codes[coverage % valid_codes.numel()]
    return (raw[..., 0::2] | (raw[..., 1::2] << 4)).contiguous()


def _make_e8m0_bytes(shape, scale_range, scale_dtype):
    if scale_dtype != torch.float8_e8m0fnu:
        raise TypeError(
            "MXFP4 requires dequant_dtype=FLOAT8_E8M0/torch.float8_e8m0fnu, "
            f"got {scale_dtype}"
        )
    low, high = _range_pair(scale_range, "E8M0 scale range")
    if low <= 0:
        raise ValueError(f"E8M0 scale range must be positive, got {scale_range}")
    min_code = max(0, math.ceil(math.log2(low)) + E8M0_ONE_VALUE)
    max_code = min(254, math.floor(math.log2(high)) + E8M0_ONE_VALUE)
    while min_code <= 254 and math.ldexp(1.0, min_code - E8M0_ONE_VALUE) < low:
        min_code += 1
    while max_code >= 0 and math.ldexp(1.0, max_code - E8M0_ONE_VALUE) > high:
        max_code -= 1
    if min_code > max_code:
        raise ValueError(
            f"E8M0 scale range contains no representable value: {scale_range}"
        )
    count = max_code - min_code + 1
    return (
        torch.arange(math.prod(shape), dtype=torch.int64)
        .remainder(count)
        .add(min_code)
        .reshape(shape)
        .to(torch.uint8)
    )


def _dsl_quant_mode(value):
    mode = int(value)
    if mode == 5:
        return 1
    if mode != 1:
        raise ValueError(f"MXFP4 quant_mode must be DSL 1 or legacy ASC 5, got {mode}")
    return mode


@dataclass
class QliV2InputData:
    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_q: torch.Tensor
    seqused_k: torch.Tensor
    cmp_residual_k: torch.Tensor | None
    block_table: torch.Tensor
    params: tuple
    cu_seqlens_k: torch.Tensor | None = None
    output_idx_offset: torch.Tensor | None = None


def _params_dict(test_data):
    if len(test_data) != len(PARAM_NAMES):
        raise ValueError(
            f"QLI V2 parameter count mismatch: got {len(test_data)}, "
            f"expected {len(PARAM_NAMES)}"
        )
    params = dict(zip(PARAM_NAMES, test_data))
    if params["k_t_size"] is None:
        params["k_t_size"] = params["k_seq"]
    return params


def _int_tensor(value, default):
    source = default if value is None else value
    return torch.tensor(source, dtype=torch.int32)


def _is_auto(value):
    return isinstance(value, str) and value.upper() == "AUTO"


def _with_axis0_gap(tensor):
    strides = list(tensor.stride())
    strides[0] *= 2
    result = torch.empty_strided(tensor.shape, tuple(strides), dtype=tensor.dtype)
    result.copy_(tensor)
    return result


def generate_qliv2_test_data(test_data):
    """Generate deterministic public-ABI tensors without invoking the NPU."""
    p = _params_dict(test_data)
    batch = int(p["batch_size"])
    s1 = int(p["q_seq"])
    s2 = int(p["k_seq"])
    total_q = int(p["q_t_size"])
    if total_q < 1:
        raise ValueError("q_t_size must be positive")
    if (int(p["q_head_num"]), int(p["k_head_num"]), int(p["head_dim"])) != (32, 1, 128):
        raise ValueError("fixed QLI axes must be N1=32, N2=1, D=128")
    if int(p["block_size"]) != PAGE_SIZE:
        raise ValueError("DSL transfer paramset fixes PA block_size at 128")
    if int(p["k_t_size"]) != s2:
        raise ValueError("k_t_size must equal k_seq for this transfer paramset")

    generator = torch.Generator(device="cpu").manual_seed(int(p["seed"]))
    pages = (s2 + PAGE_SIZE - 1) // PAGE_SIZE
    if p["layout_key"] != "TND" and int(p["block_num"]) != pages:
        raise ValueError(f"block_num must equal ceil(k_seq/128)={pages}")
    if p["weight_dtype"] != torch.float32:
        raise TypeError(
            f"weight_dtype must be FP32/torch.float32, got {p['weight_dtype']}"
        )
    if p["actual_seq_dtype"] != torch.int32:
        raise TypeError(
            f"actual_seq_dtype must be INT32/torch.int32, got {p['actual_seq_dtype']}"
        )
    _dsl_quant_mode(p["quant_mode"])
    q = _make_mxfp4_bytes(
        (total_q, N1, PACKED_D * FP4_PACK_NUM),
        p["query_datarange"],
        p["qk_dtype"],
        generator,
    )
    k = _make_mxfp4_bytes(
        (pages, PAGE_SIZE, N2, PACKED_D * FP4_PACK_NUM),
        p["key_datarange"],
        p["qk_dtype"],
        generator,
    )
    weight_low, weight_high = _range_pair(p["weights_datarange"], "weights_datarange")
    w = torch.rand((total_q, N1), generator=generator, dtype=torch.float32)
    w = w * (weight_high - weight_low) + weight_low
    q_descale = _make_e8m0_bytes(
        (total_q, N1, 2, 2), p["q_scale_datarange"], p["dequant_dtype"]
    )
    k_descale = _make_e8m0_bytes(
        (pages, PAGE_SIZE, N2, 2, 2),
        p["k_scale_datarange"],
        p["dequant_dtype"],
    )

    cu_key = None
    if p["layout_key"] == "TND":
        bounds = p["cu_seqlens_k"]
        if bounds is None or _is_auto(bounds):
            bounds = [index * s2 for index in range(batch + 1)]
        cu_key = torch.tensor(bounds, dtype=torch.int32)
        k = _make_mxfp4_bytes(
            (int(cu_key[-1]), N2, 128), p["key_datarange"], p["qk_dtype"], generator
        )
        k_descale = _make_e8m0_bytes(
            (int(cu_key[-1]), N2, 2, 2), p["k_scale_datarange"], p["dequant_dtype"]
        )
    storage_layout = str(p["storage_layout"])
    if storage_layout in ("key_axis0_noncontiguous", "axis0_noncontiguous"):
        k = _with_axis0_gap(k)
    if storage_layout in ("scale_axis0_noncontiguous", "axis0_noncontiguous"):
        k_descale = _with_axis0_gap(k_descale)

    if p["cu_seqlens_q"] is None and (batch != 1 or total_q != s1):
        raise ValueError(
            "cu_seqlens_q=None is only valid for batch_size=1 and q_t_size=q_seq"
        )
    default_cu = [index * s1 for index in range(batch + 1)]
    cu_value = p["cu_seqlens_q"]
    cu = (
        _int_tensor(default_cu, default_cu)
        if _is_auto(cu_value)
        else None
        if cu_value is None
        else _int_tensor(cu_value, cu_value)
    )
    if cu is not None and (
        tuple(cu.shape) != (batch + 1,) or int(cu[0]) != 0 or int(cu[-1]) != total_q
    ):
        raise ValueError(
            "cu_seqlens_q must have shape (B+1), start at 0, and end at q_t_size"
        )
    q_lengths = [total_q] if cu is None else (cu[1:] - cu[:-1]).tolist()
    if any(length < 0 or length > s1 for length in q_lengths):
        raise ValueError("each TND query length must be within [0, q_seq]")
    used_q_value = p["seqused_q"]
    used_q = (
        _int_tensor(q_lengths, q_lengths)
        if _is_auto(used_q_value)
        else None
        if used_q_value is None
        else _int_tensor(used_q_value, used_q_value)
    )
    if used_q is not None and (
        tuple(used_q.shape) != (batch,)
        or any(
            int(value) < 0 or int(value) > int(q_lengths[index])
            for index, value in enumerate(used_q.tolist())
        )
    ):
        raise ValueError("seqused_q must have shape (B,) and fit cu_seqlens_q")
    tails = (0, 1, 7, 9, 31, 63, 95, 127)
    default_used_k = [max(1, s2 - tails[index % len(tails)]) for index in range(batch)]
    used_k_value = p["seqused_k"]
    used_k = (
        _int_tensor(default_used_k, default_used_k)
        if _is_auto(used_k_value)
        else None
        if used_k_value is None
        else _int_tensor(used_k_value, used_k_value)
    )
    if used_k is not None and (
        tuple(used_k.shape) != (batch,)
        or any(int(value) < 0 or int(value) > s2 for value in used_k.tolist())
    ):
        raise ValueError("seqused_k must have shape (B,) and values in [0,k_seq]")
    ratio = int(p["cmp_ratio"])
    residual = p["cmp_residual_k"]
    if _is_auto(residual):
        residual = [index % ratio for index in range(batch)]
    residual_tensor = None if residual is None else _int_tensor(residual, residual)

    page_ids = torch.arange(pages, dtype=torch.int32)
    block_table_value = p["block_table"]
    block_table = (
        torch.stack([torch.roll(page_ids, shifts=index) for index in range(batch)])
        if _is_auto(block_table_value)
        else None
        if block_table_value is None
        else _int_tensor(block_table_value, block_table_value)
    )
    if block_table is not None and tuple(block_table.shape) != (batch, pages):
        raise ValueError(
            f"block_table must have shape ({batch},{pages}), got {tuple(block_table.shape)}"
        )
    return QliV2InputData(
        q,
        k,
        w,
        q_descale,
        k_descale,
        cu,
        used_q,
        used_k,
        residual_tensor,
        None if cu_key is not None else block_table,
        tuple(test_data),
        cu_key,
        None
        if p["output_idx_offset"] is None
        else torch.full((total_q, 1), int(p["output_idx_offset"]), dtype=torch.int32),
    )


def _operator_kwargs(p, data):
    return {
        "cu_seqlens_q": data.cu_seqlens_q,
        "cu_seqlens_k": data.cu_seqlens_k,
        "output_idx_offset": data.output_idx_offset,
        "seqused_q": data.seqused_q,
        "seqused_k": data.seqused_k,
        "cmp_residual_k": data.cmp_residual_k,
        "block_table": data.block_table,
        "topk": int(p["sparse_count"]),
        "quant_mode": _dsl_quant_mode(p["quant_mode"]),
        "max_seqlen_q": int(p["max_seqlen_q"]),
        "mask_mode": int(p["sparse_mode"]),
        "cmp_ratio": int(p["cmp_ratio"]),
        "layout_q": str(p["layout_query"]),
        "layout_k": str(p["layout_key"]),
        "return_value": bool(p["return_value"]),
        "candidate_topk_blocks": int(p["candidate_topk_blocks"]),
        "candidate_block_size": int(p["candidate_block_size"]),
    }


def _to_npu(value):
    return value.npu() if isinstance(value, torch.Tensor) else value


def _to_npu_preserving_strides(value):
    if value.is_contiguous():
        return value.npu()
    result = torch.empty_strided(
        value.shape, value.stride(), dtype=value.dtype, device="npu"
    )
    result.copy_(value)
    return result


def run_qliv2_case_data(data, device_id=0):
    """Run the independent CPU Golden and one fused NPU kernel call."""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from indexer_test_runtime import configure

    configure()
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl

    quant_lightning_indexer = (
        torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer
    )
    device_id = int(os.environ.get("QLIV2_DEVICE_ID", device_id))
    torch_npu.npu.set_device(device_id)
    p = _params_dict(data.params)
    kwargs = _operator_kwargs(p, data)
    cpu_result = quant_lightning_indexer_golden(
        data.q, data.k, data.w, data.q_descale, data.k_descale, **kwargs
    )
    if kwargs["candidate_topk_blocks"] == -1:
        # Match the Torch schema's disabled outputs; score references are unchanged.
        cpu_result["candidate_block_indices"] = cpu_result[
            "candidate_block_indices"
        ].reshape(-1)
        cpu_result["candidate_block_length"] = cpu_result[
            "candidate_block_length"
        ].reshape(-1)[:0]
    k_npu = _to_npu_preserving_strides(data.k)
    k_scale_npu = _to_npu_preserving_strides(data.k_descale)
    npu_kwargs = {name: _to_npu(value) for name, value in kwargs.items()}
    topk = npu_kwargs.pop("topk")
    quant_mode = npu_kwargs.pop("quant_mode")
    npu_kwargs["metadata"] = (
        torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            npu_kwargs["cu_seqlens_q"],
            npu_kwargs["cu_seqlens_k"],
            npu_kwargs["seqused_q"],
            npu_kwargs["seqused_k"],
            npu_kwargs["cmp_residual_k"],
            batch_size=(
                int(p["batch_size"]) if npu_kwargs["layout_k"] == "PA_BBND" else None
            ),
            max_seqlen_q=npu_kwargs["max_seqlen_q"],
            max_seqlen_k=int(p["k_seq"]),
            num_heads_q=int(p["q_head_num"]),
            num_heads_k=int(p["k_head_num"]),
            head_dim=int(p["head_dim"]),
            topk=topk,
            mask_mode=npu_kwargs["mask_mode"],
            cmp_ratio=npu_kwargs["cmp_ratio"],
            layout_q=npu_kwargs["layout_q"],
            layout_k=npu_kwargs["layout_k"],
            candidate_topk_blocks=npu_kwargs["candidate_topk_blocks"],
            candidate_block_size=npu_kwargs["candidate_block_size"],
        )
    )
    npu_outputs = quant_lightning_indexer(
        data.q.npu(),
        k_npu,
        data.w.npu(),
        data.q_descale.npu(),
        k_scale_npu,
        topk,
        quant_mode,
        **npu_kwargs,
    )
    torch.npu.synchronize()
    output_names = (
        "sparse_indices",
        "sparse_values",
        "candidate_block_indices",
        "candidate_block_length",
    )
    npu_result = {name: value.cpu() for name, value in zip(output_names, npu_outputs)}
    return cpu_result, npu_result


def qliv2_output_single(test_data):
    data = generate_qliv2_test_data(test_data)
    cpu_result, npu_result = run_qliv2_case_data(data)
    return (
        cpu_result,
        npu_result,
        cpu_result.get("_reference_scores"),
        cpu_result.get("sparse_values"),
        npu_result.get("sparse_values"),
    )


def generate_cpu_golden(input_data):
    p = _params_dict(input_data.params)
    return quant_lightning_indexer_golden(
        input_data.q,
        input_data.k,
        input_data.w,
        input_data.q_descale,
        input_data.k_descale,
        **_operator_kwargs(p, input_data),
    )


__all__ = [
    "QliV2InputData",
    "generate_cpu_golden",
    "generate_qliv2_test_data",
    "qliv2_output_single",
    "run_qliv2_case_data",
]
