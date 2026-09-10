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

"""FFAG (FusedFloydAttentionGrad) 定制输入修正

与 ATK init_by_input_data 等价:
  1. 用 query/key1/value1/key2/value2/atten_mask/scale 运行 forward
  2. 将 forward 输出的 atten_in/softmax_max/softmax_sum in-place 覆写到输入 tensor

参数签名与 aclnnFusedFloydAttentionGradGetWorkspaceSize 一致 (不含 workspaceSize/executor)。
"""

import importlib.util
import inspect as _inspect
import os
import numpy as np
import torch

_IMPL_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name):
    """从同目录加载模块"""
    path = os.path.join(_IMPL_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_ffag_impl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_sibling("golden")
floyd_attn_forward = _golden_mod.floyd_attn_forward


def _parse_case_id(testcase_name):
    """从 testcase_name 尾部解析 case_id"""
    import re

    m = re.search(r"(\d+)\s*$", testcase_name or "")
    return int(m.group(1)) if m else 0


_ND_SENTINEL_RANGE = (-100, 100)

# ATK 同输入互喂（默认关闭，避免干扰日常随机 ND 测试）:
# 开启方式: export FFAG_ATK_INPUT_REPLAY=1
# 开启后若该 case 的 ATK dump 输入存在, 优先加载, 覆盖种子生成逻辑
# 用途: 归因对拍（详见 TTK_test/problem/ttk_input_bug/ffag_case_10_same_input_replay.md）
_ATK_DUMP_DIR = "/home/j00946653/ATK2TTK/temp/atk_ffag_nd_inputs"


def _replay_enabled():
    """ATK 输入回放开关, 默认关闭"""
    return os.environ.get("FFAG_ATK_INPUT_REPLAY", "0") == "1"


def _load_atk_dump(case_id, name):
    """加载 ATK 落盘输入, 不存在或开关关闭时返回 None"""
    if not _replay_enabled():
        return None
    path = os.path.join(_ATK_DUMP_DIR, f"case_{case_id}_{name}.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, tuple):
        data = data[0]
    return data


def _is_nd_sentinel(rv):
    """检测 input_data_ranges 中的 (-100, 100) 哨兵范围"""
    if rv is None or not isinstance(rv, (tuple, list)):
        return False
    return (
        len(rv) == 2
        and rv[0] == _ND_SENTINEL_RANGE[0]
        and rv[1] == _ND_SENTINEL_RANGE[1]
    )


_nd_rng_seeded = {}
_ND_MEAN_RANGE = (-100.0, 100.0)
_ND_STD_RANGE = (1.0, 25.0)


def _gen_nd_tensor(case_id, shape, dtype, mean_range, std_range, tensor_idx=0):
    """用与 ATK 完全相同的连续流种子策略生成正态分布 tensor

    ATK 在 base_dataset 中用 seed_everything(case_id) 一次播种全局 RNG，
    然后 5 个 ND tensor 从同一连续流依次取 mean/std。
    本函数复刻该行为: 第一次调用时 np.random.seed(case_id)+torch.manual_seed(case_id)，
    后续调用直接从已播种的连续流取值。
    """
    if case_id not in _nd_rng_seeded:
        np.random.seed(case_id)
        torch.manual_seed(case_id)
        _nd_rng_seeded[case_id] = True
    mean = float(np.random.uniform(mean_range[0], mean_range[1]))
    std = float(np.random.uniform(std_range[0], std_range[1]))
    data = torch.normal(mean, std, tuple(shape)).to(dtype=dtype)
    return data


_ND_TENSOR_NAMES = ["query", "key1", "value1", "key2", "value2"]


def _safe_to_tensor(arr):
    """numpy array -> torch tensor (bfloat16 安全转换)"""
    if arr is None:
        return None
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


def customize_inputs(
    query,
    key1,
    value1,
    key2,
    value2,
    dy,
    attenMaskOptional,
    softmaxMaxOptional,
    softmaxSumOptional,
    attentionInOptional,
    scaleValue,
    dqOut,
    dk1Out,
    dv1Out,
    dk2Out,
    dv2Out,
    **kwargs,
):
    """入参预处理 — 先运行 forward 生成 softmax_max/softmax_sum/atten_in

    参数签名与 aclnnFusedFloydAttentionGradGetWorkspaceSize 一致。
    输出 tensor (dqOut 等) 不参与输入修正，仅占位。

    TTK 已生成随机 tensor，此函数做原地确定性修正:
      1. 用 query/key1/value1/key2/value2/atten_mask/scaleValue 运行 forward
      2. 将 forward 输出的 atten_in/softmax_max/softmax_sum in-place 覆写

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              input_ranges, use_torch, short_soc_version
    """
    testcase_name = kwargs.get("testcase_name", "")
    case_id = _parse_case_id(testcase_name)

    # 处理 scaleValue (可能为范围值列表)
    if isinstance(scaleValue, (list, tuple)):
        if len(scaleValue) == 2:
            _rng = np.random.RandomState(case_id)
            scaleValue = float(_rng.uniform(scaleValue[0], scaleValue[1]))
        else:
            scaleValue = float(scaleValue[0]) if len(scaleValue) > 0 else 1.0
    else:
        scaleValue = float(scaleValue)

    # 获取输入 tensor 并转 CPU
    if not isinstance(query, torch.Tensor):
        query = _safe_to_tensor(query)
    if not isinstance(key1, torch.Tensor):
        key1 = _safe_to_tensor(key1)
    if not isinstance(value1, torch.Tensor):
        value1 = _safe_to_tensor(value1)
    if not isinstance(key2, torch.Tensor):
        key2 = _safe_to_tensor(key2)
    if not isinstance(value2, torch.Tensor):
        value2 = _safe_to_tensor(value2)

    query = query.cpu()
    key1 = key1.cpu()
    value1 = value1.cpu()
    key2 = key2.cpu()
    value2 = value2.cpu()

    atten_mask = attenMaskOptional
    if atten_mask is not None:
        if not isinstance(atten_mask, torch.Tensor):
            atten_mask = _safe_to_tensor(atten_mask)
        atten_mask = atten_mask.cpu()

    # nd 分布输入修正: 从 _ctx 获取 input_data_ranges，检测哨兵范围并重新生成正态分布数据
    _caller = _inspect.currentframe().f_back
    _self_obj = _caller.f_locals.get("self") if _caller is not None else None
    _ctx_obj = getattr(_self_obj, "_ctx", None) if _self_obj is not None else None
    if _ctx_obj is not None:
        input_ranges = getattr(_ctx_obj, "flat_input_data_ranges", None)
        flat_shapes = getattr(_ctx_obj, "flat_tensor_view_shapes", None)
        if input_ranges is not None and flat_shapes is not None:
            nd_tensors = [query, key1, value1, key2, value2]
            replayed = False
            for idx, tensor in enumerate(nd_tensors):
                if tensor is None or idx >= len(input_ranges):
                    continue
                rv = input_ranges[idx]
                if not _is_nd_sentinel(rv):
                    continue
                # 同输入互喂: 优先加载 ATK dump
                dumped = _load_atk_dump(case_id, _ND_TENSOR_NAMES[idx])
                if dumped is not None:
                    tensor.copy_(dumped.to(dtype=tensor.dtype))
                    replayed = True
                    print(
                        f"[ND-REPLAY-FFAG] case_id={case_id}, tensor_idx={idx} "
                        f"({_ND_TENSOR_NAMES[idx]}), loaded from ATK dump, dtype={tensor.dtype}",
                        flush=True,
                    )
                    continue
                mean_range = (-100.0, 100.0)
                std_range = (1.0, 25.0)
                shape = (
                    flat_shapes[idx] if idx < len(flat_shapes) else list(tensor.shape)
                )
                orig_dtype = tensor.dtype
                new_data = _gen_nd_tensor(
                    case_id, shape, orig_dtype, mean_range, std_range, tensor_idx=idx
                )
                tensor.copy_(new_data)
                print(
                    f"[ND-GEN-FFAG] case_id={case_id}, tensor_idx={idx} "
                    f"({_ND_TENSOR_NAMES[idx]}), shape={shape}, dtype={orig_dtype}",
                    flush=True,
                )
            if replayed:
                # dy 与 atten_mask 一并回放, 保证与 ATK 完全同输入
                # (ATK atten_mask range=1 → 全1; TTK CSV (0,0) → 全0, 必须回放)
                if isinstance(dy, torch.Tensor):
                    dumped = _load_atk_dump(case_id, "dy")
                    if dumped is not None:
                        dy.copy_(dumped.to(dtype=dy.dtype))
                        print(
                            f"[ND-REPLAY-FFAG] case_id={case_id}, dy loaded from ATK dump",
                            flush=True,
                        )
                if atten_mask is not None and isinstance(atten_mask, torch.Tensor):
                    dumped = _load_atk_dump(case_id, "atten_mask")
                    if dumped is not None:
                        atten_mask.copy_(dumped.to(dtype=atten_mask.dtype))
                        print(
                            f"[ND-REPLAY-FFAG] case_id={case_id}, atten_mask loaded from ATK dump",
                            flush=True,
                        )
            # dy 端点混入修正: TTK 框架的 _mix_expect_data 会向 dy 强制注入 ~25% 的
            # 区间端点值 (-2, 2)，而 ATK 使用 np.random.uniform(low, high, shape) 不混入端点。
            # 对 ND 用例（含哨兵范围），重新生成 dy 以消除端点混入，对齐 ATK 行为。
            if not replayed and isinstance(dy, torch.Tensor):
                dy_idx = 5  # dy 在 input_ranges 中的位置
                if dy_idx < len(input_ranges):
                    dy_rv = input_ranges[dy_idx]
                    if (
                        isinstance(dy_rv, (tuple, list))
                        and len(dy_rv) == 2
                        and dy_rv[0] != dy_rv[1]
                    ):
                        dy_shape = (
                            flat_shapes[dy_idx]
                            if dy_idx < len(flat_shapes)
                            else list(dy.shape)
                        )
                        dy_dtype = dy.dtype
                        # 使用独立 RNG 避免与 ND tensor 种子冲突
                        _dy_rng = np.random.RandomState(case_id * 100 + 99)
                        dy_low = float(dy_rv[0])
                        dy_high = float(dy_rv[1])
                        dy_data = _dy_rng.uniform(dy_low, dy_high, dy_shape).astype(
                            np.float16
                            if dy_dtype == torch.float16
                            else np.float32
                            if dy_dtype == torch.float32
                            else np.float64
                        )
                        dy.copy_(torch.from_numpy(dy_data).to(dtype=dy_dtype))
                        print(
                            f"[DY-FIX-FFAG] case_id={case_id}, regenerated dy without endpoint mixing, "
                            f"range=({dy_low},{dy_high}), dtype={dy_dtype}",
                            flush=True,
                        )

    # 输入 dump（用于 ATK vs TTK 逐元素对比验证）
    _DUMP_DIR = "/home/j00946653/ATK2TTK/temp/ttk_ffag_nd_inputs_dump"
    if os.environ.get("FFAG_INPUT_DUMP", "0") == "1":
        os.makedirs(_DUMP_DIR, exist_ok=True)
        _dump_tensors = {
            "query": query,
            "key1": key1,
            "value1": value1,
            "key2": key2,
            "value2": value2,
            "dy": dy,
        }
        if atten_mask is not None:
            _dump_tensors["atten_mask"] = atten_mask
        for _name, _tensor in _dump_tensors.items():
            if _tensor is not None:
                torch.save(
                    _tensor.cpu(), os.path.join(_DUMP_DIR, f"case_{case_id}_{_name}.pt")
                )
        print(
            f"[INPUT-DUMP-FFAG] case_id={case_id}, dumped {len(_dump_tensors)} tensors to {_DUMP_DIR}",
            flush=True,
        )

    # 保存原始 dtype
    origin_q_dtype = query.dtype
    origin_x_max_dtype = (
        softmaxMaxOptional.dtype if softmaxMaxOptional is not None else torch.float32
    )

    # 用 float64 计算 forward (与 ATK 一致)
    golden_dtype = torch.float64
    Q_f64 = query.to(golden_dtype)
    K1_f64 = key1.to(golden_dtype)
    V1_f64 = value1.to(golden_dtype)
    K2_f64 = key2.to(golden_dtype)
    V2_f64 = value2.to(golden_dtype)

    # 运行 forward
    atten_in, x_max, x_sum = floyd_attn_forward(
        Q_f64, K1_f64, K2_f64, V1_f64, V2_f64, atten_mask, scaleValue
    )

    # repeat 到 [B,H,N,M,8]
    x_max = x_max.repeat(1, 1, 1, 1, 8)
    x_sum = x_sum.repeat(1, 1, 1, 1, 8)

    # 转换 dtype (与 ATK 一致)
    x_max = x_max.to(origin_x_max_dtype)
    x_sum = x_sum.to(origin_x_max_dtype)
    atten_in = atten_in.to(origin_q_dtype)

    # in-place 覆写输入 tensor: softmaxMax, softmaxSum, attentionIn
    if softmaxMaxOptional is not None and x_max.shape == softmaxMaxOptional.shape:
        softmaxMaxOptional.copy_(
            x_max.to(softmaxMaxOptional.dtype).to(softmaxMaxOptional.device)
        )
    if softmaxSumOptional is not None and x_sum.shape == softmaxSumOptional.shape:
        softmaxSumOptional.copy_(
            x_sum.to(softmaxSumOptional.dtype).to(softmaxSumOptional.device)
        )
    if attentionInOptional is not None and atten_in.shape == attentionInOptional.shape:
        attentionInOptional.copy_(
            atten_in.to(attentionInOptional.dtype).to(attentionInOptional.device)
        )

    print(f"[CUSTOMIZE-FFAG] case_id={case_id}, scaleValue={scaleValue}", flush=True)
    print(
        f"[CUSTOMIZE-FFAG] Forward done: x_max.shape={tuple(x_max.shape)}, "
        f"atten_in.shape={tuple(atten_in.shape)}",
        flush=True,
    )


def customize_inputs_e2e(
    grad_output,
    query_ik,
    key_ij,
    value_ij,
    key_jk,
    value_jk,
    attention_out,
    softmax_max,
    softmax_sum,
    *,
    atten_mask=None,
    scale_value=1.0,
    **kwargs,
):
    """E2E 定制输入修正 — 与 torch_npu.npu_fused_floyd_attention_backward 签名一致

    参数顺序与 torch_npu API 一致 (grad_output 在第一位)。
    无输出占位 tensor。逻辑与 ACLNN customize_inputs 完全相同:
      1. 用 query/key/value/atten_mask/scale 运行 forward
      2. 将 forward 输出的 atten_in/softmax_max/softmax_sum in-place 覆写
    """
    testcase_name = kwargs.get("testcase_name", "")
    case_id = _parse_case_id(testcase_name)

    # 处理 scale_value (可能为范围值列表)
    if isinstance(scale_value, (list, tuple)):
        if len(scale_value) == 2:
            _rng = np.random.RandomState(case_id)
            scale_value = float(_rng.uniform(scale_value[0], scale_value[1]))
        else:
            scale_value = float(scale_value[0]) if len(scale_value) > 0 else 1.0
    else:
        scale_value = float(scale_value)

    # 获取输入 tensor 并转 CPU
    # E2E 参数名映射: query_ik→query, key_ij→key1, value_ij→value1, key_jk→key2, value_jk→value2
    if not isinstance(query_ik, torch.Tensor):
        query_ik = _safe_to_tensor(query_ik)
    if not isinstance(key_ij, torch.Tensor):
        key_ij = _safe_to_tensor(key_ij)
    if not isinstance(value_ij, torch.Tensor):
        value_ij = _safe_to_tensor(value_ij)
    if not isinstance(key_jk, torch.Tensor):
        key_jk = _safe_to_tensor(key_jk)
    if not isinstance(value_jk, torch.Tensor):
        value_jk = _safe_to_tensor(value_jk)

    query = query_ik.cpu()
    key1 = key_ij.cpu()
    value1 = value_ij.cpu()
    key2 = key_jk.cpu()
    value2 = value_jk.cpu()

    mask = atten_mask
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = _safe_to_tensor(mask)
        mask = mask.cpu()

    # nd 分布输入修正: 从 _ctx 获取 input_data_ranges，检测哨兵范围并重新生成正态分布数据
    _caller = _inspect.currentframe().f_back
    _self_obj = _caller.f_locals.get("self") if _caller is not None else None
    _ctx_obj = getattr(_self_obj, "_ctx", None) if _self_obj is not None else None
    if _ctx_obj is not None:
        input_ranges = getattr(_ctx_obj, "flat_input_data_ranges", None)
        flat_shapes = getattr(_ctx_obj, "flat_tensor_view_shapes", None)
        if input_ranges is not None and flat_shapes is not None:
            nd_tensors = [query, key1, value1, key2, value2]
            for idx, tensor in enumerate(nd_tensors):
                if tensor is None or idx >= len(input_ranges):
                    continue
                rv = input_ranges[idx]
                if not _is_nd_sentinel(rv):
                    continue
                mean_range = (-100.0, 100.0)
                std_range = (1.0, 25.0)
                shape = (
                    flat_shapes[idx] if idx < len(flat_shapes) else list(tensor.shape)
                )
                orig_dtype = tensor.dtype
                new_data = _gen_nd_tensor(
                    case_id, shape, orig_dtype, mean_range, std_range, tensor_idx=idx
                )
                tensor.copy_(new_data)
                print(
                    f"[ND-GEN-FFAG-E2E] case_id={case_id}, tensor_idx={idx} "
                    f"({_ND_TENSOR_NAMES[idx]}), shape={shape}, dtype={orig_dtype}",
                    flush=True,
                )
            # dy 端点混入修正: TTK 框架的 _mix_expect_data 会向 dy 强制注入 ~25% 的
            # 区间端点值，而 ATK 使用 np.random.uniform(low, high, shape) 不混入端点。
            if isinstance(grad_output, torch.Tensor):
                dy_idx = 5  # grad_output(dy) 在 input_ranges 中的位置
                if dy_idx < len(input_ranges):
                    dy_rv = input_ranges[dy_idx]
                    if (
                        isinstance(dy_rv, (tuple, list))
                        and len(dy_rv) == 2
                        and dy_rv[0] != dy_rv[1]
                    ):
                        dy_shape = (
                            flat_shapes[dy_idx]
                            if dy_idx < len(flat_shapes)
                            else list(grad_output.shape)
                        )
                        dy_dtype = grad_output.dtype
                        _dy_rng = np.random.RandomState(case_id * 100 + 99)
                        dy_low = float(dy_rv[0])
                        dy_high = float(dy_rv[1])
                        dy_data = _dy_rng.uniform(dy_low, dy_high, dy_shape).astype(
                            np.float16
                            if dy_dtype == torch.float16
                            else np.float32
                            if dy_dtype == torch.float32
                            else np.float64
                        )
                        grad_output.copy_(torch.from_numpy(dy_data).to(dtype=dy_dtype))
                        print(
                            f"[DY-FIX-FFAG-E2E] case_id={case_id}, regenerated dy without endpoint mixing, "
                            f"range=({dy_low},{dy_high}), dtype={dy_dtype}",
                            flush=True,
                        )

    # 保存原始 dtype
    origin_q_dtype = query.dtype
    origin_x_max_dtype = softmax_max.dtype if softmax_max is not None else torch.float32

    # 用 float64 计算 forward (与 ATK 一致)
    golden_dtype = torch.float64
    Q_f64 = query.to(golden_dtype)
    K1_f64 = key1.to(golden_dtype)
    V1_f64 = value1.to(golden_dtype)
    K2_f64 = key2.to(golden_dtype)
    V2_f64 = value2.to(golden_dtype)

    # 运行 forward
    atten_in, x_max, x_sum = floyd_attn_forward(
        Q_f64, K1_f64, K2_f64, V1_f64, V2_f64, mask, scale_value
    )

    # repeat 到 [B,H,N,M,8]
    x_max = x_max.repeat(1, 1, 1, 1, 8)
    x_sum = x_sum.repeat(1, 1, 1, 1, 8)

    # 转换 dtype (与 ATK 一致)
    x_max = x_max.to(origin_x_max_dtype)
    x_sum = x_sum.to(origin_x_max_dtype)
    atten_in = atten_in.to(origin_q_dtype)

    # in-place 覆写输入 tensor: softmax_max, softmax_sum, attention_out
    if softmax_max is not None and x_max.shape == softmax_max.shape:
        softmax_max.copy_(x_max.to(softmax_max.dtype).to(softmax_max.device))
    if softmax_sum is not None and x_sum.shape == softmax_sum.shape:
        softmax_sum.copy_(x_sum.to(softmax_sum.dtype).to(softmax_sum.device))
    if attention_out is not None and atten_in.shape == attention_out.shape:
        attention_out.copy_(atten_in.to(attention_out.dtype).to(attention_out.device))

    print(
        f"[CUSTOMIZE-FFAG-E2E] case_id={case_id}, scale_value={scale_value}", flush=True
    )
    print(
        f"[CUSTOMIZE-FFAG-E2E] Forward done: x_max.shape={tuple(x_max.shape)}, "
        f"atten_in.shape={tuple(atten_in.shape)}",
        flush=True,
    )
