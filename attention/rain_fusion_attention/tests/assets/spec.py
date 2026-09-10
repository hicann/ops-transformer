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

"""RFA (RainFusionAttention) TestSpec 入口

ACLNN 模式 TestSpec，从 impl/ 加载各模块。

C API 参数顺序 (aclnnRainFusionAttentionGetWorkspaceSize):
  inputs: query, key, value, selectIdx, selectNumIdx, blockShape,
          attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
          blockTableOptional
  attr:   qInputLayout, kvInputLayout, numKeyValueHeads, maskType,
          scaleValue (double), innerPrecise, blockSize
  outputs: attentionOut, softmaxLseOptional
"""

import importlib.util
import os
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_impl(name):
    """从 impl/ 子目录加载模块 (spec loader 不注册包，不能用相对导入)"""
    path = os.path.join(_THIS_DIR, "impl", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_rfa_impl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_golden_mod = _load_impl("golden")
_inputs_mod = _load_impl("inputs")
_compare_mod = _load_impl("compare")

rain_fusion_attention_forward = _golden_mod.rain_fusion_attention_forward
rain_fusion_attention_forward_fp16 = _golden_mod.rain_fusion_attention_forward_fp16
customize_inputs = _inputs_mod.customize_inputs
compare = _compare_mod.compare


def _golden(
    query,
    key,
    value,
    selectIdx,
    selectNumIdx,
    blockShape,
    attenMaskOptional,
    actualSeqLengthsOptional,
    actualSeqLengthsKvOptional,
    blockTableOptional,
    qInputLayout,
    kvInputLayout,
    numKeyValueHeads,
    maskType,
    scaleValue,
    innerPrecise,
    blockSize,
    attentionOut,
    softmaxLseOptional,
    **kwargs,
):
    """CPU golden — 高精度参考, 与 ATK cpu_benchmark 等价

    参数签名与 aclnnRainFusionAttentionGetWorkspaceSize 一致
    (不含 workspaceSize/executor)。输出 tensor 不参与 golden 计算。

    三方对比模式:
      - golden = 高精度输入完整计算 (模拟 ATK cpu_benchmark)
      - benchmark = 原dtype输入完整计算 (在compare中计算, 模拟 cpu_0)
      - NPU输出 = 原dtype

    **kwargs: testcase_name, tensor_dtypes, tensor_formats, scalar_dtypes,
              use_torch, short_soc_version
    """
    # 转为 CPU tensor
    query = query.detach().cpu() if isinstance(query, torch.Tensor) else query
    key = key.detach().cpu() if isinstance(key, torch.Tensor) else key
    value = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    selectIdx = (
        selectIdx.detach().cpu() if isinstance(selectIdx, torch.Tensor) else selectIdx
    )
    selectNumIdx = (
        selectNumIdx.detach().cpu()
        if isinstance(selectNumIdx, torch.Tensor)
        else selectNumIdx
    )

    # 处理 scaleValue
    if isinstance(scaleValue, (list, tuple)):
        scaleValue = float(scaleValue[0]) if len(scaleValue) > 0 else 1.0
    else:
        scaleValue = float(scaleValue)

    # 处理 blockShape
    if isinstance(blockShape, (list, tuple)):
        block_shape = [int(blockShape[0]), int(blockShape[1])]
    else:
        block_shape = [64, 128]

    # 处理 actualSeqLengths
    q_seqlen_list = (
        list(actualSeqLengthsOptional) if actualSeqLengthsOptional is not None else []
    )
    kv_seqlen_list = (
        list(actualSeqLengthsKvOptional)
        if actualSeqLengthsKvOptional is not None
        else []
    )

    testcase_name = kwargs.get("testcase_name", "")
    print(
        f"[GOLDEN-RFA] testcase_name={testcase_name}, scaleValue={scaleValue}",
        flush=True,
    )

    # 与 ATK golden 一致: ATK double benchmark 中 golden 和 benchmark 使用不同 dtype
    # ATK 框架在两次 __call__ 调用中传入不同 dtype 的输入:
    #   - 输入 bf16/fp16: golden upcast 到 fp32 (高精度), benchmark 保持原 dtype
    #   - 输入 fp32: golden downcast 到低精度, benchmark 保持 fp32
    #     downcast 目标: innerPrecise=1 → fp16, innerPrecise=0 → bf16
    # rain_fusion_attention_forward 内部根据 query.dtype 选择高/低精度路径
    orig_dtype = query.dtype if isinstance(query, torch.Tensor) else torch.float32
    if orig_dtype == torch.float32:
        golden_dtype = torch.float16 if innerPrecise == 1 else torch.bfloat16
    else:
        golden_dtype = torch.float32
    query_hp = query.to(golden_dtype) if isinstance(query, torch.Tensor) else query
    key_hp = key.to(golden_dtype) if isinstance(key, torch.Tensor) else key
    value_hp = value.to(golden_dtype) if isinstance(value, torch.Tensor) else value

    atten_out = rain_fusion_attention_forward(
        query_hp,
        key_hp,
        value_hp,
        selectIdx,
        selectNumIdx,
        block_shape,
        q_seqlen_list,
        kv_seqlen_list,
        scaleValue,
        qInputLayout,
        kvInputLayout,
        innerPrecise,
    )

    # 返回 [attentionOut, softmaxLse(None)]
    # softmaxLse 在 ATK 中设为 null，这里返回 None
    return [atten_out, None]


class RainFusionAttentionTestSpec:
    """RFA 算子测试规范 — 适配 ATK golden 到 TTK ACLNN 模式

    三方对比模式:
      - Actual:   NPU输出 vs 高精度golden
      - Benchmark: 原dtype标杆输出 vs 高精度golden
      - ratio = Actual误差 / max(Benchmark误差, error_thd)
    """

    # CPU golden 参考实现 — 参数签名与 aclnn 头文件一致
    golden = staticmethod(_golden)

    # 定制输入修正 — selectIdx 重生成 + scaleValue 随机参数处理
    customize_inputs = staticmethod(customize_inputs)

    # 自定义精度比较 — 纯Python白盒 cv_fused_double_benchmark 三方对比
    compare = staticmethod(compare)

    # 精度标准
    tolerance = {
        "bfloat16": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


__spec__ = {"aclnnRainFusionAttention": "RainFusionAttentionTestSpec"}
