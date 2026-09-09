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

"""FFAG (FusedFloydAttentionGrad) TestSpec 入口

ACLNN 模式 TestSpec，从 impl/ 加载各模块。

C API 参数顺序 (aclnnFusedFloydAttentionGradGetWorkspaceSize):
  inputs: query, key1, value1, key2, value2, dy, attenMaskOptional,
          softmaxMaxOptional, softmaxSumOptional, attentionInOptional
  attr:   scaleValue (double)
  outputs: dqOut, dk1Out, dv1Out, dk2Out, dv2Out
"""

import importlib.util
import os
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_impl(name):
    """从 impl/ 子目录加载模块 (spec loader 不注册包，不能用相对导入)"""
    path = os.path.join(_THIS_DIR, "impl", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_ffag_impl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_impl("golden")
_inputs_mod = _load_impl("inputs")
_compare_mod = _load_impl("compare")

floyd_attn_forward = _golden_mod.floyd_attn_forward
floyd_attn_backward = _golden_mod.floyd_attn_backward
floyd_attn_backward_fp16 = _golden_mod.floyd_attn_backward_fp16
customize_inputs = _inputs_mod.customize_inputs
compare = _compare_mod.compare


def _golden(
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
    """CPU golden — 高精度参考, 与 ATK cpu_benchmark 等价

    参数签名与 aclnnFusedFloydAttentionGradGetWorkspaceSize 一致
    (不含 workspaceSize/executor)。输出 tensor (dqOut 等) 不参与 golden 计算。

    三方对比模式:
      - golden = 高精度输入完整计算forward+backward (模拟 ATK cpu_benchmark)
      - benchmark = 原dtype输入完整计算forward+backward (在compare中计算, 模拟 cpu_0)
      - NPU输出 = 原dtype

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    # 转为 CPU tensor
    query = query.detach().cpu() if isinstance(query, torch.Tensor) else query
    key1 = key1.detach().cpu() if isinstance(key1, torch.Tensor) else key1
    value1 = value1.detach().cpu() if isinstance(value1, torch.Tensor) else value1
    key2 = key2.detach().cpu() if isinstance(key2, torch.Tensor) else key2
    value2 = value2.detach().cpu() if isinstance(value2, torch.Tensor) else value2
    grad = dy.detach().cpu() if isinstance(dy, torch.Tensor) else dy
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
        f"[GOLDEN-FFAG] testcase_name={testcase_name}, scaleValue={scaleValue}",
        flush=True,
    )

    # 将输入提升到 fp64 (与 ATK init_by_input_data 一致)
    # 然后完整计算 forward+backward, 得到高精度 golden
    golden_dtype = torch.float64
    query_hp = query.to(golden_dtype)
    key1_hp = key1.to(golden_dtype)
    value1_hp = value1.to(golden_dtype)
    key2_hp = key2.to(golden_dtype)
    value2_hp = value2.to(golden_dtype)

    # 用 fp64 计算 forward (与 ATK init_by_input_data 一致)
    atten_in_hp, x_max_hp, x_sum_hp = floyd_attn_forward(
        query_hp, key1_hp, key2_hp, value1_hp, value2_hp, atten_mask, scaleValue
    )
    x_max_hp = x_max_hp.repeat(1, 1, 1, 1, 8)
    x_sum_hp = x_sum_hp.repeat(1, 1, 1, 1, 8)

    # ATK 把 forward 中间结果转回原 dtype (init_by_input_data 中的关键步骤)
    origin_q_dtype = query.dtype if isinstance(query, torch.Tensor) else torch.float16
    origin_x_max_dtype = (
        softmaxMaxOptional.dtype if softmaxMaxOptional is not None else torch.float32
    )
    atten_in_final = atten_in_hp.to(origin_q_dtype)
    x_max_final = x_max_hp.to(origin_x_max_dtype)
    x_sum_final = x_sum_hp.to(origin_x_max_dtype)

    # ATK __call__ 中: Q/K1/...保持 fp64, grad 保持原 dtype, atten_in/x_max/x_sum 用转换后的
    # floyd_attn_backward 内部: dtype=fp64, golden_dtype=fp32, einsum 用 Q.to(fp64)
    dQ, dK1, dV1, dK2, dV2 = floyd_attn_backward(
        query_hp,
        key1_hp,
        key2_hp,
        value1_hp,
        value2_hp,
        grad,
        atten_mask,
        x_max_final,
        x_sum_final,
        atten_in_final,
        scaleValue,
    )

    return [
        dQ.contiguous(),
        dK1.contiguous(),
        dV1.contiguous(),
        dK2.contiguous(),
        dV2.contiguous(),
    ]


class FusedFloydAttentionGradTestSpec:
    """FFAG 算子测试规范 — 适配 ATK golden 到 TTK ACLNN 模式

    三方对比模式:
      - Actual:   NPU输出 vs 高精度golden
      - Benchmark: 原dtype标杆输出 vs 高精度golden
      - ratio = Actual误差 / max(Benchmark误差, error_thd)
    """

    # CPU golden 参考实现 — 参数签名与 aclnn 头文件一致
    golden = staticmethod(_golden)

    # 定制输入修正 — 先运行 forward 覆写 softmax_max/sum/atten_in
    customize_inputs = staticmethod(customize_inputs)

    # 自定义精度比较 — 纯Python白盒 cv_fused_double_benchmark 三方对比
    compare = staticmethod(compare)

    # 精度标准
    tolerance = {
        "bfloat16": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


def _golden_e2e(
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
    """CPU golden — E2E 模式, 与 torch_npu.npu_fused_floyd_attention_backward 签名一致

    参数顺序与 torch_npu API 一致 (grad_output 在第一位)。
    无输出占位 tensor, 返回 5 个梯度。

    复用 ACLNN golden 的核心计算逻辑: fp64 forward + backward。
    """
    # 转为 CPU tensor
    query = query_ik.detach().cpu() if isinstance(query_ik, torch.Tensor) else query_ik
    key1 = key_ij.detach().cpu() if isinstance(key_ij, torch.Tensor) else key_ij
    value1 = value_ij.detach().cpu() if isinstance(value_ij, torch.Tensor) else value_ij
    key2 = key_jk.detach().cpu() if isinstance(key_jk, torch.Tensor) else key_jk
    value2 = value_jk.detach().cpu() if isinstance(value_jk, torch.Tensor) else value_jk
    grad = (
        grad_output.detach().cpu()
        if isinstance(grad_output, torch.Tensor)
        else grad_output
    )
    mask = atten_mask
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()

    # 处理 scale_value
    if isinstance(scale_value, (list, tuple)):
        scale_value = float(scale_value[0]) if len(scale_value) > 0 else 1.0
    else:
        scale_value = float(scale_value)

    testcase_name = kwargs.get("testcase_name", "")
    print(
        f"[GOLDEN-FFAG-E2E] testcase_name={testcase_name}, scale_value={scale_value}",
        flush=True,
    )

    # 用 fp64 计算 forward+backward (与 ACLNN golden 逻辑一致)
    golden_dtype = torch.float64
    query_hp = query.to(golden_dtype)
    key1_hp = key1.to(golden_dtype)
    value1_hp = value1.to(golden_dtype)
    key2_hp = key2.to(golden_dtype)
    value2_hp = value2.to(golden_dtype)

    atten_in_hp, x_max_hp, x_sum_hp = floyd_attn_forward(
        query_hp, key1_hp, key2_hp, value1_hp, value2_hp, mask, scale_value
    )
    x_max_hp = x_max_hp.repeat(1, 1, 1, 1, 8)
    x_sum_hp = x_sum_hp.repeat(1, 1, 1, 1, 8)

    origin_q_dtype = query.dtype if isinstance(query, torch.Tensor) else torch.float16
    origin_x_max_dtype = softmax_max.dtype if softmax_max is not None else torch.float32
    atten_in_final = atten_in_hp.to(origin_q_dtype)
    x_max_final = x_max_hp.to(origin_x_max_dtype)
    x_sum_final = x_sum_hp.to(origin_x_max_dtype)

    dQ, dK1, dV1, dK2, dV2 = floyd_attn_backward(
        query_hp,
        key1_hp,
        key2_hp,
        value1_hp,
        value2_hp,
        grad,
        mask,
        x_max_final,
        x_sum_final,
        atten_in_final,
        scale_value,
    )

    return [
        dQ.contiguous(),
        dK1.contiguous(),
        dV1.contiguous(),
        dK2.contiguous(),
        dV2.contiguous(),
    ]


class FusedFloydAttentionGradE2ESpec:
    """FFAG 算子测试规范 — E2E (torch_npu 直调) 模式

    参数签名与 torch_npu.npu_fused_floyd_attention_backward 一致:
      grad_output, query_ik, key_ij, value_ij, key_jk, value_jk,
      attention_out, softmax_max, softmax_sum, *, atten_mask=None, scale_value=1.

    复用 ACLNN 的 compare 逻辑和 floyd_attn_backward golden 计算。
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
    "aclnnFusedFloydAttentionGrad": "FusedFloydAttentionGradTestSpec",
    "torch_npu.npu_fused_floyd_attention_backward": "FusedFloydAttentionGradE2ESpec",
}
