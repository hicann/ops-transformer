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

"""FFA (FusedFloydAttention) TestSpec 入口

支持两种模式:
  1. ACLNN 模式 — aclnnFusedFloydAttention
  2. E2E 模式  — torch_npu.npu_fused_floyd_attention

ACLNN C API 参数顺序 (aclnnFusedFloydAttentionGetWorkspaceSize):
  inputs: query, key1, value1, key2, value2, attenMaskOptional
  attr:   scaleValue (double)
  outputs: softmaxMaxOut, softmaxSumOut, attentionOutOut

E2E torch_npu API 签名 (npu_fused_floyd_attention):
  inputs: query_ik, key_ij, value_ij, key_jk, value_jk
  kwargs: atten_mask=None, scale_value=1.
  returns: (softmax_max, softmax_sum, attention_out)
"""

import importlib.util
import os
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_impl(name):
    """从 impl/ 子目录加载模块 (spec loader 不注册包，不能用相对导入)"""
    path = os.path.join(_THIS_DIR, "impl", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_ffa_impl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_impl("golden")
_inputs_mod = _load_impl("inputs")
_compare_mod = _load_impl("compare")

floyd_attn_forward = _golden_mod.floyd_attn_forward
floyd_attn_forward_fp16 = _golden_mod.floyd_attn_forward_fp16
customize_inputs = _inputs_mod.customize_inputs
compare = _compare_mod.compare


def _golden(
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
    """CPU golden — 与 ATK floyd_attn_forward 等价

    参数签名与 aclnnFusedFloydAttentionGetWorkspaceSize 一致
    (不含 workspaceSize/executor)。输出 tensor 不参与 golden 计算。

    返回高精度golden (fp32 for fp16输入)。
    fp16标杆输出由compare函数通过 floyd_attn_forward_fp16 独立计算。

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    # 转为 CPU tensor
    query = query.detach().cpu() if isinstance(query, torch.Tensor) else query
    key1 = key1.detach().cpu() if isinstance(key1, torch.Tensor) else key1
    value1 = value1.detach().cpu() if isinstance(value1, torch.Tensor) else value1
    key2 = key2.detach().cpu() if isinstance(key2, torch.Tensor) else key2
    value2 = value2.detach().cpu() if isinstance(value2, torch.Tensor) else value2
    atten_mask = attenMaskOptional
    if atten_mask is not None and isinstance(atten_mask, torch.Tensor):
        atten_mask = atten_mask.detach().cpu()

    # 处理 scaleValue
    if isinstance(scaleValue, (list, tuple)):
        scaleValue = float(scaleValue[0]) if len(scaleValue) > 0 else 1.0
    else:
        scaleValue = float(scaleValue)

    testcase_name = kwargs.get("testcase_name", "")
    print(
        f"[GOLDEN-FFA] testcase_name={testcase_name}, scaleValue={scaleValue}",
        flush=True,
    )

    # 三方对比模式:
    #   golden = floyd_attn_forward(fp32输入) → fp32输出 (高精度参考)
    #     ATK cpu_benchmark (golden) 用 fp32 输入调用 floyd_attn_forward:
    #     - dtype=fp32 → golden_dtype=fp64, einsum用fp32计算
    #     - output.to(fp32) 最终输出fp32
    #   benchmark = floyd_attn_forward_fp16(原dtype输入) → 原dtype输出 (CPU标杆, 在compare中计算)
    #   NPU输出 = 原dtype
    # 将输入转为fp32产生高精度golden (与ATK cpu_benchmark一致)
    orig_dtype = query.dtype if isinstance(query, torch.Tensor) else torch.float16
    golden_dtype = (
        torch.float32
        if orig_dtype in (torch.float16, torch.bfloat16)
        else torch.float64
    )
    query_hp = query.to(golden_dtype) if isinstance(query, torch.Tensor) else query
    key1_hp = key1.to(golden_dtype) if isinstance(key1, torch.Tensor) else key1
    value1_hp = value1.to(golden_dtype) if isinstance(value1, torch.Tensor) else value1
    key2_hp = key2.to(golden_dtype) if isinstance(key2, torch.Tensor) else key2
    value2_hp = value2.to(golden_dtype) if isinstance(value2, torch.Tensor) else value2

    softmax_max, softmax_sum, attention_out = floyd_attn_forward(
        query_hp, key1_hp, key2_hp, value1_hp, value2_hp, atten_mask, scaleValue
    )

    # softmax_max/sum 需要 repeat 到 [B,H,N,M,8] 以匹配 NPU 输出
    softmax_max_out = softmax_max.repeat(1, 1, 1, 1, 8)
    softmax_sum_out = softmax_sum.repeat(1, 1, 1, 1, 8)

    # 返回 [softmaxMaxOut, softmaxSumOut, attentionOutOut]
    # softmaxMaxOut 和 softmaxSumOut 不比较 (compare 中 golden 设为 None)
    # attention_out 保持高精度 (fp32 for fp16输入)
    return [softmax_max_out, softmax_sum_out, attention_out]


def _golden_e2e(
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
    """CPU golden for E2E — 与 torch_npu.npu_fused_floyd_attention 签名一致

    参数签名与 torch_npu.npu_fused_floyd_attention 一致:
      query_ik, key_ij, value_ij, key_jk, value_jk (位置参数)
      atten_mask=None, scale_value=1. (关键字参数)

    不含输出占位 tensor (E2E 模式输出由 API 返回)。

    返回高精度golden (fp32 for fp16输入)。
    fp16标杆输出由compare函数通过 floyd_attn_forward_fp16 独立计算。

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    # 转为 CPU tensor
    query = query_ik.detach().cpu() if isinstance(query_ik, torch.Tensor) else query_ik
    key1 = key_ij.detach().cpu() if isinstance(key_ij, torch.Tensor) else key_ij
    value1 = value_ij.detach().cpu() if isinstance(value_ij, torch.Tensor) else value_ij
    key2 = key_jk.detach().cpu() if isinstance(key_jk, torch.Tensor) else key_jk
    value2 = value_jk.detach().cpu() if isinstance(value_jk, torch.Tensor) else value_jk
    atten_mask_t = atten_mask
    if atten_mask_t is not None and isinstance(atten_mask_t, torch.Tensor):
        atten_mask_t = atten_mask_t.detach().cpu()

    # 处理 scale_value
    if isinstance(scale_value, (list, tuple)):
        scale_value = float(scale_value[0]) if len(scale_value) > 0 else 1.0
    else:
        scale_value = float(scale_value)

    testcase_name = kwargs.get("testcase_name", "")
    print(
        f"[GOLDEN-FFA-E2E] testcase_name={testcase_name}, scale_value={scale_value}",
        flush=True,
    )

    # 将输入转为fp32产生高精度golden (与ATK cpu_benchmark一致)
    orig_dtype = query.dtype if isinstance(query, torch.Tensor) else torch.float16
    golden_dtype = (
        torch.float32
        if orig_dtype in (torch.float16, torch.bfloat16)
        else torch.float64
    )
    query_hp = query.to(golden_dtype) if isinstance(query, torch.Tensor) else query
    key1_hp = key1.to(golden_dtype) if isinstance(key1, torch.Tensor) else key1
    value1_hp = value1.to(golden_dtype) if isinstance(value1, torch.Tensor) else value1
    key2_hp = key2.to(golden_dtype) if isinstance(key2, torch.Tensor) else key2
    value2_hp = value2.to(golden_dtype) if isinstance(value2, torch.Tensor) else value2

    softmax_max, softmax_sum, attention_out = floyd_attn_forward(
        query_hp, key1_hp, key2_hp, value1_hp, value2_hp, atten_mask_t, scale_value
    )

    # softmax_max/sum 需要 repeat 到 [B,H,N,M,8] 以匹配 NPU 输出
    softmax_max_out = softmax_max.repeat(1, 1, 1, 1, 8)
    softmax_sum_out = softmax_sum.repeat(1, 1, 1, 1, 8)

    return [softmax_max_out, softmax_sum_out, attention_out]


class FusedFloydAttentionTestSpec:
    """FFA 算子测试规范 — ACLNN 模式

    三方对比模式:
      - Actual:   NPU输出 vs 高精度golden
      - Benchmark: fp16标杆输出 vs 高精度golden
      - ratio = Actual误差 / max(Benchmark误差, error_thd)
    """

    # CPU golden 参考实现 — 参数签名与 aclnn 头文件一致
    golden = staticmethod(_golden)

    # 定制输入修正 — scaleValue 随机参数处理
    customize_inputs = staticmethod(customize_inputs)

    # 自定义精度比较 — ATK cv_fused_double_benchmark 三方对比
    compare = staticmethod(compare)

    # 精度标准
    tolerance = {
        "bfloat16": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


class FusedFloydAttentionE2ESpec:
    """FFA 算子测试规范 — E2E (torch_npu 直调) 模式

    参数签名与 torch_npu.npu_fused_floyd_attention 一致:
      query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask, scale_value

    复用 ACLNN 的 compare 逻辑和 floyd_attn_forward golden 计算。
    """

    # CPU golden — 签名与 torch_npu API 一致 (无输出占位)
    golden = staticmethod(_golden_e2e)

    # 定制输入修正 — E2E 版本 (无输出占位 tensor)
    customize_inputs = staticmethod(_inputs_mod.customize_inputs_e2e)

    # 自定义精度比较 — 复用 ACLNN 的三方对比
    compare = staticmethod(compare)

    # 精度标准
    tolerance = {
        "bfloat16": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


__spec__ = {
    "aclnnFusedFloydAttention": "FusedFloydAttentionTestSpec",
    "torch_npu.npu_fused_floyd_attention": "FusedFloydAttentionE2ESpec",
}
