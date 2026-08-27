#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden adapter for quant_block_sparse_attn TTK cases."""

import importlib.util
import math
import sys
from pathlib import Path

import torch

PYTEST_GOLDEN_MODULE = None
PYTEST_MXFP8_GOLDEN_MODULE = None
_MXFP8_MODULE_NAME = "_qbsa_mxfp8_pytest_golden"
_FP8_MODULE_NAME = "_qbsa_fp8_pytest_golden"


# ==================================================================================================
# Common helpers and TTK entry point
# ==================================================================================================


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [int(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def _numel(value):
    if value is None:
        return 0
    return int(value.numel()) if torch.is_tensor(value) else int(value.size)


def lengths_from_prefix(value):
    vals = to_list(value)
    return (
        [vals[i + 1] - vals[i] for i in range(len(vals) - 1)] if len(vals) > 1 else []
    )


def _format_golden_outputs(attention_out, softmax_lse, return_softmax_lse):
    """Keep the operator's two output slots and invalidate disabled LSE."""
    return [attention_out, softmax_lse if return_softmax_lse else None]


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def cpu_quant_block_sparse_attn(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    sparse_indices,
    sparse_seq_len,
    p_scale,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    block_table,
    atten_mask,
    metadata,
    *,
    quant_mode=1,
    softmax_scale=1.0,
    mask_mode=3,
    blocksize=0,
    sparse_block_size_q=128,
    sparse_block_size_kv=128,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_out="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    return_softmax_lse=False,
    quant_matmul=False,
    **kwargs,
):
    """Map the shared 16-input CSV signature to the existing pytest golden."""
    del metadata
    if _numel(cu_seqlens_kv) != 0:
        raise ValueError("cu_seqlens_kv must be empty")
    if _numel(seqused_q) != 0:
        raise ValueError("seqused_q must be empty")
    if p_scale is not None and _numel(p_scale) == 0:
        p_scale = None
    if int(quant_mode) != 2 and blocksize and int(key.shape[2]) != int(blocksize):
        raise ValueError("blocksize does not match key")

    common_kwargs = {
        "layout_kv": layout_kv,
        "layout_q": layout_q,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "mask_mode": mask_mode,
        "return_softmax_lse": return_softmax_lse,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "block_table": block_table,
        **kwargs,
    }
    if int(quant_mode) == 2:
        return _mxfp8_cpu_golden(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            softmax_scale,
            sparse_block_size_q,
            sparse_block_size_kv,
            quant_matmul=quant_matmul,
            **common_kwargs,
        )

    return _fp8_cpu_golden(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        p_scale,
        softmax_scale=softmax_scale,
        sparse_q_block_size=sparse_block_size_q,
        sparse_kv_block_size=sparse_block_size_kv,
        blocksize=blocksize,
        quant_mode=quant_mode,
        **common_kwargs,
    )


# ==================================================================================================
# MXFP8-only helpers
# ==================================================================================================


def _mxfp8_load_pytest_golden_module():
    """Load the legacy MXFP8 reference under the same name as input.py."""
    global PYTEST_MXFP8_GOLDEN_MODULE
    if PYTEST_MXFP8_GOLDEN_MODULE is not None:
        return PYTEST_MXFP8_GOLDEN_MODULE
    if _MXFP8_MODULE_NAME in sys.modules:
        PYTEST_MXFP8_GOLDEN_MODULE = sys.modules[_MXFP8_MODULE_NAME]
        return PYTEST_MXFP8_GOLDEN_MODULE

    pytest_dir = (
        Path(__file__).resolve().parents[2] / "pytest" / "bsa_fullquant_mxfp8_test"
    )
    module_path = pytest_dir / "qbsa_mxfp8_golden.py"
    spec = importlib.util.spec_from_file_location(_MXFP8_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MXFP8 golden module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MXFP8_MODULE_NAME] = module
    sys.path.insert(0, str(pytest_dir))
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MXFP8_MODULE_NAME, None)
        raise
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_MXFP8_GOLDEN_MODULE = module
    return module


def _mxfp8_normalize_data_range(data_range):
    """Preserve scalar radius or normalize an explicit [min, max] range."""
    if isinstance(data_range, (list, tuple)):
        if len(data_range) != 2:
            raise ValueError(
                f"data_range must be a scalar or [min, max], got: {data_range}"
            )
        low, high = float(data_range[0]), float(data_range[1])
        if math.isnan(low) or math.isnan(high):
            if not (math.isnan(low) and math.isnan(high)):
                raise ValueError(
                    "nan data_range bounds must describe a constant nan value, "
                    f"got: [{low}, {high}]"
                )
            return [low, high]
        if low > high:
            raise ValueError(
                f"data_range min must not exceed max, got: [{low}, {high}]"
            )
        return [low, high]

    radius = float(data_range)
    if not math.isfinite(radius):
        return radius
    if radius < 0:
        raise ValueError(
            f"scalar data_range must be finite and non-negative, got: {radius}"
        )
    return radius


def _mxfp8_cpu_golden(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    layout_kv="PA_BNBD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    mask_mode=3,
    return_softmax_lse=False,
    sparse_mode="dense",
    seed=0,
    quant_matmul=False,
    **kwargs,
):
    """Run the original pytest CPU reference on the exact customized inputs."""
    module = _mxfp8_load_pytest_golden_module()
    testcase_name = kwargs.get("testcase_name") or "__default__"
    cache = getattr(module, "_TTK_MXFP8_CACHE", {})
    cached = cache.pop(testcase_name, None)

    if cached is None:
        batch = int(sparse_indices.shape[0])
        q_prefix = to_list(cu_seqlens_q)
        q_lengths = lengths_from_prefix(cu_seqlens_q)
        kv_lengths = to_list(seqused_kv)
        case = {
            "B": batch,
            "N1": int(query.shape[1]),
            "N2": int(key.shape[1]),
            "D": int(query.shape[-1]),
            "cu_seqlens_q": q_prefix,
            "cu_seqlens_kv": to_list(cu_seqlens_kv),
            "seqused_q": to_list(seqused_q),
            "seqused_kv": kv_lengths,
            "s2_base_size": int(kwargs.get("s2_base_size", 512)),
            "blocknum": int(key.shape[0]),
            "max_block_per_batch": int(block_table.shape[1]),
            "block_size": int(key.shape[2]),
            "mask_mode": int(mask_mode),
            "fp8_dtype": torch.float8_e4m3fn,
            "scale_dtype": torch.float8_e8m0fnu,
            "quant_group_size": int(kwargs.get("quant_group_size", 32)),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "layout_sparse_indices": layout_sparse_indices,
            "layout_out": layout_out,
            "kv_cache_layout": "BnNBsD",
            "p_scale_value": (
                None
                if kwargs.get("p_scale_value") is None
                else float(kwargs["p_scale_value"])
            ),
            "softmax_scale": float(softmax_scale),
            "return_softmax_lse": bool(return_softmax_lse),
            "seed": int(seed),
            "data_range_q": _mxfp8_normalize_data_range(
                kwargs.get("data_range_q", 1.0)
            ),
            "data_range_k": _mxfp8_normalize_data_range(
                kwargs.get("data_range_k", 1.0)
            ),
            "data_range_v": _mxfp8_normalize_data_range(
                kwargs.get("data_range_v", 1.0)
            ),
            "device_id": 0,
            "sparse_q_block_size": int(sparse_q_block_size),
            "sparse_kv_block_size": int(sparse_kv_block_size),
            "sparse_mode": sparse_mode,
            "quant_mode": 2,
        }
        module.validate_mxfp8_case(case)
        module.set_active_case(case)
        data = {
            "query": to_cpu(query),
            "key": to_cpu(key),
            "value": to_cpu(value),
            "q_descale": to_cpu(q_descale),
            "k_descale": to_cpu(k_descale),
            "v_descale": to_cpu(v_descale),
            "p_scale": None if p_scale is None else to_cpu(p_scale),
            "sparse_indices": to_cpu(sparse_indices),
            "sparse_seq_len": to_cpu(sparse_seq_len),
            "block_table": to_cpu(block_table),
            "q_lengths": q_lengths,
            "kv_lengths": kv_lengths,
            "cu_seqlens_q": torch.tensor(q_prefix, dtype=torch.int32),
        }
    else:
        case = cached["case"]
        data = cached["data"]
        module.set_active_case(case)

    attention_out, softmax_lse = module.cpu_mxfp8_golden(
        data["query"],
        data["key"],
        data["value"],
        data["q_descale"],
        data["k_descale"],
        data["v_descale"],
        data["p_scale"],
        data["q_lengths"],
        data["kv_lengths"],
        data["cu_seqlens_q"],
        data["sparse_indices"],
        data["sparse_seq_len"],
        data["block_table"],
        use_quant_matmul=_as_bool(quant_matmul),
    )
    attention_out = attention_out.to(torch.bfloat16)
    return _format_golden_outputs(attention_out, softmax_lse, return_softmax_lse)


# ==================================================================================================
# FP8-only helpers
# ==================================================================================================


def _fp8_load_pytest_golden_module():
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    if _FP8_MODULE_NAME in sys.modules:
        PYTEST_GOLDEN_MODULE = sys.modules[_FP8_MODULE_NAME]
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "quant_block_sparse_attn_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(_FP8_MODULE_NAME, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load FP8 golden module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_FP8_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(_FP8_MODULE_NAME, None)
            raise
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return module


def _fp8_cpu_golden(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    p_scale=None,
    *,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    p_scale_value=1.0,
    cu_seqlens_q_value=None,
    cu_seqlens_kv_value=None,
    seqused_q_value=None,
    seqused_kv_value=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    blocksize=0,
    layout_kv="PA_BNBD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=1,
    mask_mode=3,
    return_softmax_lse=False,
    sparse_mode=None,
    seed=None,
    input_ranges=None,
    testcase_name=None,
    **kwargs,
):
    """CPU reference implementation wrapping the existing pytest golden."""
    del (
        value,
        q_descale,
        k_descale,
        v_descale,
        sparse_seq_len,
        atten_mask,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        quant_mode,
    )
    golden_module = _fp8_load_pytest_golden_module()
    cache_key = testcase_name or "__default__"
    cache = getattr(golden_module, "_TTK_FP8_CACHE", {})
    cached = cache.pop(cache_key, None)
    if cached is not None:
        golden = cached["data"]["golden"]
        return _format_golden_outputs(
            golden["attention_out"],
            golden["softmax_lse"],
            return_softmax_lse,
        )

    data_ranges = input_ranges or ()
    case = golden_module._fp8_assemble_case(
        query,
        key,
        sparse_indices,
        block_table,
        softmax_scale=softmax_scale,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_indices=layout_sparse_indices,
        layout_out=layout_out,
        mask_mode=mask_mode,
        return_softmax_lse=return_softmax_lse,
        sparse_mode=sparse_mode,
        cu_seqlens_q_value=cu_seqlens_q_value,
        cu_seqlens_kv_value=cu_seqlens_kv_value,
        seqused_q_value=seqused_q_value,
        seqused_kv_value=seqused_kv_value,
        p_scale_value=p_scale_value,
        seed=0 if seed is None else int(seed),
        blocksize=blocksize,
        data_range_q=data_ranges[0] if len(data_ranges) > 0 else 1.0,
        data_range_k=data_ranges[1] if len(data_ranges) > 1 else 1.0,
        data_range_v=data_ranges[2] if len(data_ranges) > 2 else 1.0,
        testcase_name=testcase_name,
        **kwargs,
    )
    generated = golden_module.generate_and_save_testdata(case)
    golden = generated["golden"]
    return _format_golden_outputs(
        golden["attention_out"], golden["softmax_lse"], return_softmax_lse
    )
