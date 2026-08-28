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

"""Input customization for quant_block_sparse_attn TTK cases.

E2E customize_inputs contract: modify tensors in-place via x.copy_(value).
No return value. Tensor shapes/dtypes are pre-allocated by TTK from CSV.
"""

import importlib.util
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

_MXFP8_MODULE_NAME = "_qbsa_mxfp8_pytest_golden"
_FP8_MODULE_NAME = "_qbsa_fp8_pytest_golden"


# ==================================================================================================
# Common helpers and TTK entry point
# ==================================================================================================


def _inplace_copy(destination, source):
    """Copy tensors while preserving the raw byte representation of FP8 dtypes."""
    if destination is None or source is None:
        return
    if isinstance(destination, torch.Tensor):
        source_tensor = source
        if isinstance(source, np.ndarray):
            if source.dtype.kind == "V" or "float8" in str(source.dtype):
                destination.view(torch.uint8).copy_(
                    torch.from_numpy(source.view(np.uint8))
                )
                return
            source_tensor = torch.from_numpy(source)
        if isinstance(source_tensor, torch.Tensor) and "float8" in str(
            source_tensor.dtype
        ):
            destination.view(torch.uint8).copy_(source_tensor.view(torch.uint8))
        else:
            destination.copy_(source_tensor)
        return

    source_tensor = (
        source.detach().cpu() if isinstance(source, torch.Tensor) else source
    )
    if isinstance(source_tensor, torch.Tensor) and "float8" in str(source_tensor.dtype):
        destination_view = (
            destination.view(np.uint8) if destination.dtype.kind == "V" else destination
        )
        destination_view[...] = source_tensor.view(torch.uint8).numpy()
    elif isinstance(source_tensor, torch.Tensor):
        destination[...] = source_tensor.numpy()
    else:
        destination[...] = source_tensor


def _numel(value):
    """Return element count for the NumPy or Torch buffers supplied by TTK."""
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, np.ndarray):
        return int(value.size)
    return int(np.size(value))


def _list_attr(value, name):
    if value is None:
        raise ValueError(f"{name} must be provided")
    return [int(item) for item in value]


def _require_empty_input(value, name):
    if value is not None and _numel(value) != 0:
        raise ValueError(f"{name} must be an empty tensor")


def customize_inputs(
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
    p_scale_value=1.0,
    cu_seqlens_q_value=None,
    cu_seqlens_kv_value=None,
    seqused_q_value=None,
    seqused_kv_value=None,
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
    sparse_mode=None,
    seed=None,
    input_ranges=None,
    testcase_name=None,
    **kwargs,
):
    """Convert the shared CSV attributes into the operator input tensors."""
    del quant_matmul  # Golden-only switch; it must not enter either input generator.
    _require_empty_input(metadata, "metadata")
    cu_q = _list_attr(cu_seqlens_q_value, "cu_seqlens_q_value")
    cu_kv_values = _list_attr(cu_seqlens_kv_value, "cu_seqlens_kv_value")
    seq_q_values = _list_attr(seqused_q_value, "seqused_q_value")
    kv_lengths = _list_attr(seqused_kv_value, "seqused_kv_value")

    explicit_seed = None if seed is None else int(seed)
    mxfp8_seed = (
        zlib.crc32(str(testcase_name or "qbsa").encode("utf-8")) & 0x7FFFFFFF
        if explicit_seed is None
        else explicit_seed
    )
    fp8_seed = 0 if explicit_seed is None else explicit_seed
    data_ranges = input_ranges or ()
    data_range_q = data_ranges[0] if len(data_ranges) > 0 else 1.0
    data_range_k = data_ranges[1] if len(data_ranges) > 1 else 1.0
    data_range_v = data_ranges[2] if len(data_ranges) > 2 else 1.0

    if int(quant_mode) == 2:
        _mxfp8_customize_inputs(
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
            softmax_scale=softmax_scale,
            sparse_q_block_size=sparse_block_size_q,
            sparse_kv_block_size=sparse_block_size_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            layout_out=layout_out,
            mask_mode=mask_mode,
            return_softmax_lse=return_softmax_lse,
            sparse_mode=sparse_mode,
            cu_seqlens_q_value=cu_q,
            cu_seqlens_kv_value=cu_kv_values,
            seqused_q_value=seq_q_values,
            seqused_kv_value=kv_lengths,
            p_scale_value=p_scale_value,
            seed=mxfp8_seed,
            blocknum=int(key.shape[0]),
            max_block_per_batch=int(block_table.shape[1]),
            data_range_q=data_range_q,
            data_range_k=data_range_k,
            data_range_v=data_range_v,
            testcase_name=testcase_name,
            **kwargs,
        )
        module = _mxfp8_load_golden_module()
        cached = module._TTK_MXFP8_CACHE[testcase_name or "__default__"]
        block_table_data = cached["data"]["block_table"]
    else:
        _fp8_customize_inputs_from_pytest(
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
            softmax_scale=softmax_scale,
            sparse_q_block_size=sparse_block_size_q,
            sparse_kv_block_size=sparse_block_size_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            layout_out=layout_out,
            mask_mode=mask_mode,
            return_softmax_lse=return_softmax_lse,
            sparse_mode=sparse_mode,
            cu_seqlens_q_value=cu_q,
            cu_seqlens_kv_value=cu_kv_values,
            seqused_q_value=seq_q_values,
            seqused_kv_value=kv_lengths,
            p_scale_value=p_scale_value,
            seed=fp8_seed,
            blocksize=blocksize,
            data_range_q=data_range_q,
            data_range_k=data_range_k,
            data_range_v=data_range_v,
            testcase_name=testcase_name,
            **kwargs,
        )
        return

    _inplace_copy(cu_seqlens_q, torch.tensor(cu_q, dtype=torch.int32))
    _inplace_copy(seqused_kv, torch.tensor(kv_lengths, dtype=torch.int32))
    _inplace_copy(block_table, block_table_data)


# ==================================================================================================
# MXFP8-only helpers
# ==================================================================================================


def _mxfp8_load_golden_module():
    """Load the legacy MXFP8 reference once so input and golden share one cache."""
    if _MXFP8_MODULE_NAME in sys.modules:
        return sys.modules[_MXFP8_MODULE_NAME]

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


def _mxfp8_assemble_case(
    query,
    key,
    sparse_indices,
    *,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    layout_q,
    layout_kv,
    layout_sparse_indices,
    layout_out,
    mask_mode,
    return_softmax_lse,
    cu_seqlens_q_value,
    cu_seqlens_kv_value,
    seqused_q_value,
    seqused_kv_value,
    sparse_mode,
    p_scale_value,
    seed,
    quant_group_size,
    s2_base_size,
    blocknum,
    max_block_per_batch,
    data_range_q,
    data_range_k,
    data_range_v,
):
    batch = int(sparse_indices.shape[0])
    cu_q = [int(value) for value in cu_seqlens_q_value]
    cu_kv = [int(value) for value in cu_seqlens_kv_value]
    seq_q = [int(value) for value in seqused_q_value]
    seq_kv = [int(value) for value in seqused_kv_value]
    normalized_blocknum = None if blocknum is None else int(blocknum)
    return {
        "B": batch,
        "N1": int(query.shape[1]),
        "N2": int(key.shape[1]),
        "D": int(query.shape[-1]),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": cu_kv,
        "seqused_q": seq_q,
        "seqused_kv": seq_kv,
        "s2_base_size": int(s2_base_size),
        "blocknum": normalized_blocknum,
        "max_block_per_batch": (
            None if max_block_per_batch is None else int(max_block_per_batch)
        ),
        "block_size": int(key.shape[2]),
        "mask_mode": int(mask_mode),
        "fp8_dtype": torch.float8_e4m3fn,
        "scale_dtype": torch.float8_e8m0fnu,
        "quant_group_size": int(quant_group_size),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "kv_cache_layout": "BnNBsD",
        "p_scale_value": (None if p_scale_value is None else float(p_scale_value)),
        "softmax_scale": float(softmax_scale),
        "return_softmax_lse": bool(return_softmax_lse),
        "seed": int(seed),
        "data_range_q": _mxfp8_normalize_data_range(data_range_q),
        "data_range_k": _mxfp8_normalize_data_range(data_range_k),
        "data_range_v": _mxfp8_normalize_data_range(data_range_v),
        "device_id": 0,
        "sparse_q_block_size": int(sparse_q_block_size),
        "sparse_kv_block_size": int(sparse_kv_block_size),
        "sparse_mode": sparse_mode,
        "quant_mode": 2,
    }


def _mxfp8_customize_inputs(
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
    *,
    softmax_scale=1.0,
    sparse_q_block_size=128,
    sparse_kv_block_size=128,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    mask_mode=3,
    return_softmax_lse=False,
    cu_seqlens_q_value=None,
    cu_seqlens_kv_value=None,
    seqused_q_value=None,
    seqused_kv_value=None,
    sparse_mode=None,
    p_scale_value=None,
    seed=0,
    quant_group_size=32,
    s2_base_size=512,
    blocknum=None,
    max_block_per_batch=None,
    data_range_q=1.0,
    data_range_k=1.0,
    data_range_v=1.0,
    testcase_name=None,
    **kwargs,
):
    module = _mxfp8_load_golden_module()
    case = _mxfp8_assemble_case(
        query,
        key,
        sparse_indices,
        softmax_scale=softmax_scale,
        sparse_q_block_size=sparse_q_block_size,
        sparse_kv_block_size=sparse_kv_block_size,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_indices=layout_sparse_indices,
        layout_out=layout_out,
        mask_mode=mask_mode,
        return_softmax_lse=return_softmax_lse,
        cu_seqlens_q_value=cu_seqlens_q_value,
        cu_seqlens_kv_value=cu_seqlens_kv_value,
        seqused_q_value=seqused_q_value,
        seqused_kv_value=seqused_kv_value,
        sparse_mode=(sparse_mode if sparse_mode in ("dense", "random") else "dense"),
        p_scale_value=p_scale_value,
        seed=seed,
        quant_group_size=quant_group_size,
        s2_base_size=s2_base_size,
        blocknum=blocknum,
        max_block_per_batch=max_block_per_batch,
        data_range_q=data_range_q,
        data_range_k=data_range_k,
        data_range_v=data_range_v,
    )
    # 根据 target p_scale tensor 的 dtype 注入 p_scale_type，使 golden 生成正确 dtype 的数据
    if p_scale is not None and _numel(p_scale) > 0 and p_scale.dtype == torch.float32:
        case["p_scale_type"] = "float32"
    # Golden owns MXFP8 validation and all input construction, including the
    # random block table. TTK only supplies CSV attributes and allocated shapes.
    data = module.generate_mxfp8_inputs(
        case,
        max_block_per_batch=max_block_per_batch,
        atten_mask_shape=(tuple(atten_mask.shape) if atten_mask is not None else None),
    )

    q_scale = module.fp32_to_e8m0fnu_safe(
        module.pack_q_scale_tnd_for_npu(data["q_descale"], data["q_lengths"]),
        "Q descale",
    )
    # The MXFP8 golden generator now creates the operator-facing physical PA
    # tensors directly.  Do not repack them through block_table a second time.
    key_pa = data["key"]
    value_pa = data["value"]
    k_scale_pa = data["k_descale"]
    v_scale_pa = data["v_descale"]

    _inplace_copy(query, data["query"])
    _inplace_copy(key, key_pa)
    _inplace_copy(value, value_pa)
    _inplace_copy(q_descale, q_scale)
    _inplace_copy(k_descale, module.fp32_to_e8m0fnu_safe(k_scale_pa, "K descale"))
    _inplace_copy(v_descale, module.fp32_to_e8m0fnu_safe(v_scale_pa, "V descale"))
    if p_scale is not None and _numel(p_scale) > 0:
        # TTK keeps a one-element placeholder for this optional input.  When
        # p_scale_value is omitted, materialize the operator's default scale
        # so that the placeholder is initialized and remains equivalent to 1.
        p_scale_source = data["p_scale"]
        if p_scale_source is None:
            p_scale_source = torch.ones((1,), dtype=torch.float32)
        if p_scale.dtype == torch.float32:
            _inplace_copy(p_scale, p_scale_source)
        else:
            _inplace_copy(
                p_scale,
                module.fp32_to_e8m0fnu_safe(p_scale_source, "P scale"),
            )
    _inplace_copy(sparse_indices, data["sparse_indices"])
    _inplace_copy(sparse_seq_len, data["sparse_seq_len"])
    # mask_mode=0 does not consume atten_mask.  TTK may intentionally allocate
    # an arbitrary-rank/shape uint8 tensor to cover the ignored-input contract,
    # so only materialize the fixed causal mask for mask_mode=3.
    if int(mask_mode) == 3 and atten_mask is not None and _numel(atten_mask) > 0:
        _inplace_copy(atten_mask, data["atten_mask"])

    cache = getattr(module, "_TTK_MXFP8_CACHE", {})
    cache[testcase_name or "__default__"] = {"case": case, "data": data}
    module._TTK_MXFP8_CACHE = cache


# ==================================================================================================
# FP8-only helpers
# ==================================================================================================


def _fp8_load_golden_module():
    """Load the pytest FP8 generator once so input and golden share one cache."""
    if _FP8_MODULE_NAME in sys.modules:
        return sys.modules[_FP8_MODULE_NAME]

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "quant_block_sparse_attn_golden.py"
    spec = importlib.util.spec_from_file_location(_FP8_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FP8 golden module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_FP8_MODULE_NAME] = module
    sys.path.insert(0, str(pytest_dir))
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
    return module


def _fp8_copy_prefix(destination, source, fill_value, name):
    """Copy a generated sparse tensor into a possibly larger CSV capacity."""
    if destination is None or source is None:
        return
    destination_shape = tuple(destination.shape)
    source_shape = tuple(source.shape)
    if len(destination_shape) != len(source_shape) or any(
        source_dim > destination_dim
        for source_dim, destination_dim in zip(source_shape, destination_shape)
    ):
        raise ValueError(
            f"{name} generated shape {source_shape} does not fit TTK shape {destination_shape}"
        )
    if destination_shape == source_shape:
        _inplace_copy(destination, source)
        return

    slices = tuple(slice(0, dim) for dim in source_shape)
    if isinstance(destination, torch.Tensor):
        destination.fill_(fill_value)
        destination[slices].copy_(source)
    else:
        destination.fill(fill_value)
        source_array = (
            source.detach().cpu().numpy() if torch.is_tensor(source) else source
        )
        destination[slices] = source_array


def _fp8_customize_inputs_from_pytest(
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
    *,
    testcase_name=None,
    **case_kwargs,
):
    """Generate one FP8 case through the canonical pytest implementation."""
    module = _fp8_load_golden_module()
    case = module._fp8_assemble_case(
        query,
        key,
        sparse_indices,
        block_table,
        testcase_name=testcase_name,
        **case_kwargs,
    )
    generated = module.generate_and_save_testdata(case)
    data = generated["input"]
    key_pa, value_pa, k_scale_pa = module.combined_kv_cache.make_combined_kv_views(
        data["kv_cache_storage"], data["kv_cache_meta"]
    )

    _inplace_copy(query, data["query"])
    _inplace_copy(key, key_pa)
    _inplace_copy(value, value_pa)
    _inplace_copy(q_descale, data["q_descale"])
    _inplace_copy(k_descale, k_scale_pa)
    _inplace_copy(v_descale, data["v_descale"])
    _fp8_copy_prefix(sparse_indices, data["sparse_indices"], -1, "sparse_indices")
    _fp8_copy_prefix(sparse_seq_len, data["sparse_seq_len"], 0, "sparse_seq_len")
    p_scale_numel = 0 if p_scale is None else _numel(p_scale)
    if data["p_scale"] is None:
        if p_scale_numel != 0:
            raise ValueError("p_scale must be empty when p_scale_value is None")
    else:
        if p_scale_numel == 0:
            raise ValueError("p_scale must contain one value when p_scale_value is set")
        _inplace_copy(p_scale, data["p_scale"])
    _inplace_copy(cu_seqlens_q, data["cu_seqlens_q"])
    _inplace_copy(cu_seqlens_kv, data["cu_seqlens_kv"])
    _inplace_copy(seqused_q, data["seqused_q"])
    _inplace_copy(seqused_kv, data["seqused_kv"])
    _inplace_copy(block_table, data["block_table"])
    if atten_mask is not None and _numel(atten_mask) > 0:
        _inplace_copy(atten_mask, data["atten_mask"])

    cache = getattr(module, "_TTK_FP8_CACHE", {})
    cache[testcase_name or "__default__"] = {
        "case": generated["params"],
        "data": generated,
    }
    module._TTK_FP8_CACHE = cache
