#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FFA (FusedFloydAttention) customize_inputs — 从 ATK init_by_input_data 移植

参数签名与 aclnnFusedFloydAttentionGetWorkspaceSize 一致（含输出 tensor 占位）。
in-place 修改 tensor，不返回值。
"""

import inspect as _inspect
import os

import numpy as np
import torch


def _parse_case_id(testcase_name):
    """从 testcase_name 中解析用例 id"""
    try:
        parts = testcase_name.rsplit("_", 1)
        if len(parts) == 2:
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return 0


_ND_SENTINEL_RANGE = (-100, 100)


def _is_nd_sentinel(rv):
    """检测 input_data_ranges 中的 (-100, 100) 哨兵范围

    TTK 解析后 rv 为 tuple (min, max)，检测是否为 nd 分布哨兵值。
    """
    if rv is None or not isinstance(rv, (tuple, list)):
        return False
    return (
        len(rv) == 2
        and rv[0] == _ND_SENTINEL_RANGE[0]
        and rv[1] == _ND_SENTINEL_RANGE[1]
    )


def _gen_nd_tensor(case_id, tensor_idx, shape, dtype, mean_range, std_range):
    """用与 ATK 完全相同的逻辑生成正态分布 tensor

    ATK 逻辑 (data_base.py get_mean_and_std + data_torch.py gen_nd_tensor_data):
      1. mean = np.random.uniform(mean[0], mean[1])
      2. std  = np.random.uniform(std[0], std[1])
      3. data = torch.normal(mean, std, shape).to(dtype)

    用 case_id * 100 + tensor_idx 作为 seed，确保每个 tensor 独立采样。
    """
    rng = np.random.RandomState(case_id * 100 + tensor_idx)
    mean = float(rng.uniform(mean_range[0], mean_range[1]))
    std = float(rng.uniform(std_range[0], std_range[1]))
    data = torch.normal(mean, std, tuple(shape)).to(dtype=dtype)
    return data


_INPUT_TENSOR_NAMES = ["query", "key1", "value1", "key2", "value2", "atten_mask"]


def customize_inputs(
    query,
    key1,
    value1,
    key2,
    value2,
    attenMaskOptional,
    scaleValue,
    softmaxMaxOut,
    softmaxSumOut,
    attentionOutOut,
    **kwargs,
):
    """定制输入修正

    参数签名与 aclnnFusedFloydAttentionGetWorkspaceSize 一致:
      inputs: query, key1, value1, key2, value2, attenMaskOptional
      attr:   scaleValue (double)
      outputs: softmaxMaxOut, softmaxSumOut, attentionOutOut (占位，不处理)

    TTK 已生成随机 tensor，此函数做原地确定性修正。
    必须 in-place 修改（用 copy_()），不返回值。

    对于 ATK JSON 中 range_values 为 nd 分布的 tensor，TTK CSV 中编码为
    "nd:mean_min,mean_max,std_min,std_max"，此处解析并用与 ATK 相同的
    torch.normal 逻辑重新生成数据，替换 TTK 默认的均匀分布数据。

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    testcase_name = kwargs.get("testcase_name", "")
    case_id = _parse_case_id(testcase_name)
    print(f"[CUSTOMIZE-FFA] case_id={case_id}, scaleValue={scaleValue}", flush=True)

    # scaleValue 随机参数处理: 如果是范围值 [min, max], 用 case_id 作为 seed 生成
    _caller = _inspect.currentframe().f_back
    _self_obj = _caller.f_locals.get("self") if _caller is not None else None
    _ctx_obj = getattr(_self_obj, "_ctx", None) if _self_obj is not None else None
    if _ctx_obj is not None and hasattr(_ctx_obj, "attributes"):
        _sv = _ctx_obj.attributes.get("scaleValue")
        if isinstance(_sv, (list, tuple)) and len(_sv) == 2:
            _rng = np.random.RandomState(case_id)
            _new_sv = float(_rng.uniform(_sv[0], _sv[1]))
            _ctx_obj.attributes["scaleValue"] = _new_sv
            print(
                f"[SCALAR-GEN-FFA] case_id={case_id}, scaleValue: {_sv} -> {_new_sv:.6f}",
                flush=True,
            )

    # nd 分布输入修正: 从 _ctx 获取 input_data_ranges，解析 nd: 前缀
    nd_tensors = {
        "query": query,
        "key1": key1,
        "value1": value1,
        "key2": key2,
        "value2": value2,
        "atten_mask": attenMaskOptional,
    }
    input_ranges = None
    flat_shapes = None
    if _ctx_obj is not None:
        input_ranges = getattr(_ctx_obj, "flat_input_data_ranges", None)
        flat_shapes = getattr(_ctx_obj, "flat_tensor_view_shapes", None)

    if input_ranges is not None and flat_shapes is not None:
        input_tensor_list = [query, key1, value1, key2, value2, attenMaskOptional]
        for idx, tensor in enumerate(input_tensor_list):
            if tensor is None or idx >= len(input_ranges):
                continue
            rv = input_ranges[idx]
            if not _is_nd_sentinel(rv):
                continue
            # ATK JSON 中 nd 分布参数: mean=[-100,100], std=[1,25]
            mean_range = (-100.0, 100.0)
            std_range = (1.0, 25.0)
            shape = flat_shapes[idx] if idx < len(flat_shapes) else list(tensor.shape)
            orig_dtype = tensor.dtype
            new_data = _gen_nd_tensor(
                case_id, idx, shape, orig_dtype, mean_range, std_range
            )
            tensor.copy_(new_data)
            print(
                f"[ND-GEN-FFA] case_id={case_id}, tensor_idx={idx} "
                f"({_INPUT_TENSOR_NAMES[idx]}), nd=({mean_range[0]},{mean_range[1]},{std_range[0]},{std_range[1]}), "
                f"shape={shape}, dtype={orig_dtype}",
                flush=True,
            )

    # bool 策略修正: 对齐 ATK 的 50/50 随机 True/False
    # ATK 对 bool dtype 有专路径，直接 np.random.choice([True, False])，无视 range
    if attenMaskOptional is not None and attenMaskOptional.dtype == torch.bool:
        _bool_rng = np.random.RandomState(case_id * 100 + 5)
        _bool_data = _bool_rng.choice(
            [True, False], size=tuple(attenMaskOptional.shape)
        )
        attenMaskOptional.copy_(torch.from_numpy(_bool_data))
        print(
            f"[BOOL-GEN-FFA] case_id={case_id}, atten_mask bool 50/50 random, "
            f"True ratio={_bool_data.mean():.4f}",
            flush=True,
        )

    # ===== dump 输入数据用于 ATK vs TTK 对比 =====
    try:
        _dump_dir = "/home/j00946653/ATK2TTK/temp/ttk_ffa_inputs"
        os.makedirs(_dump_dir, exist_ok=True)
        _input_marker = os.path.join(_dump_dir, f"case_{case_id}_input.done")
        if not os.path.exists(_input_marker):
            for _tname, _tensor in [
                ("query", query),
                ("key1", key1),
                ("value1", value1),
                ("key2", key2),
                ("value2", value2),
                ("atten_mask", attenMaskOptional),
            ]:
                if _tensor is not None:
                    _tpath = os.path.join(_dump_dir, f"case_{case_id}_{_tname}.pt")
                    torch.save(_tensor.cpu().clone(), _tpath)
            print(
                f"[TTK-FFA-INPUT-DUMP] case_{case_id} saved 6 tensors to {_dump_dir}",
                flush=True,
            )
            open(_input_marker, "w").close()
    except Exception as e:
        print(f"[TTK-FFA-INPUT-DUMP] save failed: {e}", flush=True)
    # ===== dump 结束 =====

    # 输出 tensor (softmaxMaxOut, softmaxSumOut, attentionOutOut) 是占位参数，不处理
    # 不返回任何值


def customize_inputs_e2e(
    query_ik,
    key_ij,
    value_ij,
    key_jk,
    value_jk,
    *,
    atten_mask=None,
    scale_value=1.0,
    **kwargs,
):
    """E2E 定制输入修正 — 与 torch_npu.npu_fused_floyd_attention 签名一致

    参数签名:
      query_ik, key_ij, value_ij, key_jk, value_jk (位置参数)
      atten_mask=None, scale_value=1. (关键字参数)

    不含输出占位 tensor (E2E 模式输出由 API 返回)。

    逻辑与 ACLNN 的 customize_inputs 完全一致:
      1. scale_value 随机参数处理
      2. nd 分布输入修正

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    testcase_name = kwargs.get("testcase_name", "")
    case_id = _parse_case_id(testcase_name)
    print(
        f"[CUSTOMIZE-FFA-E2E] case_id={case_id}, scale_value={scale_value}", flush=True
    )

    # scale_value 随机参数处理: 如果是范围值 [min, max], 用 case_id 作为 seed 生成
    _caller = _inspect.currentframe().f_back
    _self_obj = _caller.f_locals.get("self") if _caller is not None else None
    _ctx_obj = getattr(_self_obj, "_ctx", None) if _self_obj is not None else None
    if _ctx_obj is not None and hasattr(_ctx_obj, "attributes"):
        _sv = _ctx_obj.attributes.get("scale_value")
        if isinstance(_sv, (list, tuple)) and len(_sv) == 2:
            _rng = np.random.RandomState(case_id)
            _new_sv = float(_rng.uniform(_sv[0], _sv[1]))
            _ctx_obj.attributes["scale_value"] = _new_sv
            print(
                f"[SCALAR-GEN-FFA-E2E] case_id={case_id}, scale_value: {_sv} -> {_new_sv:.6f}",
                flush=True,
            )

    # nd 分布输入修正: 从 _ctx 获取 input_data_ranges
    input_ranges = None
    flat_shapes = None
    if _ctx_obj is not None:
        input_ranges = getattr(_ctx_obj, "flat_input_data_ranges", None)
        flat_shapes = getattr(_ctx_obj, "flat_tensor_view_shapes", None)

    if input_ranges is not None and flat_shapes is not None:
        input_tensor_list = [query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask]
        for idx, tensor in enumerate(input_tensor_list):
            if tensor is None or idx >= len(input_ranges):
                continue
            rv = input_ranges[idx]
            if not _is_nd_sentinel(rv):
                continue
            mean_range = (-100.0, 100.0)
            std_range = (1.0, 25.0)
            shape = flat_shapes[idx] if idx < len(flat_shapes) else list(tensor.shape)
            orig_dtype = tensor.dtype
            new_data = _gen_nd_tensor(
                case_id, idx, shape, orig_dtype, mean_range, std_range
            )
            tensor.copy_(new_data)
            print(
                f"[ND-GEN-FFA-E2E] case_id={case_id}, tensor_idx={idx} "
                f"({_INPUT_TENSOR_NAMES[idx]}), nd=({mean_range[0]},{mean_range[1]},{std_range[0]},{std_range[1]}), "
                f"shape={shape}, dtype={orig_dtype}",
                flush=True,
            )

    # bool 策略修正: 对齐 ATK 的 50/50 随机 True/False
    if atten_mask is not None and atten_mask.dtype == torch.bool:
        _bool_rng = np.random.RandomState(case_id * 100 + 5)
        _bool_data = _bool_rng.choice([True, False], size=tuple(atten_mask.shape))
        atten_mask.copy_(torch.from_numpy(_bool_data))
        print(
            f"[BOOL-GEN-FFA-E2E] case_id={case_id}, atten_mask bool 50/50 random, "
            f"True ratio={_bool_data.mean():.4f}",
            flush=True,
        )

    # 不返回任何值
