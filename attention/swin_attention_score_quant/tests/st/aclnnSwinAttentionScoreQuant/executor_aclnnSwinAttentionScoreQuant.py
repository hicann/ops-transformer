#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ctypes

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


def _decode_dequant_u64(t):
    """解码 uint64 反量化张量为 (scale, offset)。

    语义假设（依据同仓 fused_infer_attention_score FIA V4 ST golden 与
    mc2/mega_moe 文档中 uint64 fixpipe dequant 的约定推导，算子文档未明示编码细节）：
    uint64 低 32 位为 fp32 的缩放因子 scale 的位模式，高 32 位为 int32 的偏移量 offset。
    反量化公式: dequant(acc) = acc * scale + offset（acc 为 int32 累加结果，
    bias 以 int32 形式在反量化前累加进 acc）。
    ATK 数据生成按 range_values 生成原始整数，本函数按位解码保证与 NPU 侧
    硬件 dequant 对同一份原始数据解释一致。
    """
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    if arr.dtype == np.int64:
        arr = arr.view(np.uint64)
    elif arr.dtype != np.uint64:
        arr = arr.astype(np.uint64)
    low32 = (arr & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    high32 = (arr >> np.uint64(32)).astype(np.uint32)
    scale = low32.view(np.float32).astype(np.float32)
    offset = high32.view(np.int32).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(offset)


@register("executor_swin_attention_score_quant")
class SwinAttentionScoreQuantApi(BaseApi):
    """aclnnSwinAttentionScoreQuant CPU golden。

    计算公式（见 docs/aclnnSwinAttentionScoreQuant.md）:
        out = Softmax(QK^T + bias1 + bias2)V
    全量化流程:
        1) int8 的 Q/K/V 经 scale_dequant1(uint64: 低32位 fp32 scale, 高32位 int32 offset)
           及 int32 的 bias_dequant1 反量化: score = (QK^T + bias_dequant1) * scale1 + offset1，
           scale/offset 按 key 序号 s2（QK^T 矩阵的列）广播;
        2) score(fp16) 加 padding_mask1（即公式中的 bias1），padding_mask2 为预留参数
           （文档说明当前仅支持 nullptr），golden 对非 None 的 padding_mask2 直接跳过并注释;
        3) 对 softmax_axes 指定的维度（当前仅支持最后一维）做 softmax;
        4) softmax 归一化结果经 scale_quant/bias_quant 量化回 int8:
           p_int8 = round(p * scale_quant + bias_quant)（按 s2 广播, fp16 精度, 就近取整）;
        5) p_int8 与 int8 的 V 矩乘，经 bias_dequant2/scale_dequant2 反量化:
           out = (p_int8 @ V + bias_dequant2) * scale2 + offset2，按 head 序号 h 广播, 输出 fp16。
    转置语义假设（文档未明确定义转置后输入 shape，且 tiling 始终按物理 [N,C,S,H] 解析）:
        query/key/value_transpose 为 true 时，输入以物理 shape [N,C,H,S] 给出
        （S/H 两维互换），golden 先转置回逻辑 [N,C,S,H] 再计算；输出恒为 [N,C,S,H]。
        scaleQuant/scaleDequant1/biasQuant/biasDequant1 的 [1,S] 与 paddingMask1 的
        [1,C,S,S] 中的 S 均为逻辑序列长度。
    """

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            kwargs = input_data.kwargs
            query = kwargs["query"]
            key = kwargs["key"]
            value = kwargs["value"]
            scale_quant = kwargs["scaleQuant"]
            scale_dequant1 = kwargs["scaleDequant1"]
            scale_dequant2 = kwargs["scaleDequant2"]
            bias_quant = kwargs.get("biasQuantOptional")
            bias_dequant1 = kwargs.get("biasDequant1Optional")
            bias_dequant2 = kwargs.get("biasDequant2Optional")
            padding_mask1 = kwargs.get("paddingMask1Optional")
            padding_mask2 = kwargs.get("paddingMask2Optional")
            query_transpose = kwargs["queryTranspose"]
            key_transpose = kwargs["keyTranspose"]
            value_transpose = kwargs["valueTranspose"]
            softmax_axes = kwargs["softmaxAxes"]

            # 1. 转置处理: true 时输入物理 shape 为 [N,C,H,S]，转回逻辑 [N,C,S,H]
            q = query.transpose(-2, -1) if query_transpose else query
            k = key.transpose(-2, -1) if key_transpose else key
            v = value.transpose(-2, -1) if value_transpose else value

            # 2. QK^T: int8 矩乘的 int32 累加结果用 fp32 表达（|acc| <= 127*127*64 < 2^24, 精确）
            qk = torch.matmul(
                q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)
            )

            # 3. 反量化1: (QK^T + bias_dequant1) * scale1 + offset1, 沿列 s2 广播
            scale1, offset1 = _decode_dequant_u64(scale_dequant1)
            if bias_dequant1 is not None:
                qk = qk + bias_dequant1.to(torch.float32).reshape(1, -1)
            score = (qk * scale1.reshape(1, -1) + offset1.reshape(1, -1)).to(
                torch.float16
            )

            # 4. 加 bias1（padding_mask1, fp16, [1,C,S,S] 广播）
            if padding_mask1 is not None:
                score = score + padding_mask1.to(torch.float16)
            # padding_mask2 为预留参数（文档: 当前仅支持输入 nullptr），非 None 时跳过不参与计算

            # 5. softmax（softmax_axes 归一化到 [0, ndim)，当前仅支持最后一维）
            axes = softmax_axes if softmax_axes >= 0 else score.dim() + softmax_axes
            prob = torch.softmax(score.to(torch.float32), dim=axes).to(torch.float16)

            # 6. softmax 后量化: p_int8 = round(p * scale_quant + bias_quant), 沿 s2 广播, fp16 精度
            prob_q = prob * scale_quant.to(torch.float16).reshape(1, -1)
            if bias_quant is not None:
                prob_q = prob_q + bias_quant.to(torch.float16).reshape(1, -1)
            prob_int8 = (
                torch.round(prob_q.to(torch.float32)).clamp_(-128, 127).to(torch.int8)
            )

            # 7. PV: int8 矩乘（|acc| <= 127*127*1024 < 2^24, fp32 精确），反量化2 沿 h 广播
            pv = torch.matmul(prob_int8.to(torch.float32), v.to(torch.float32))
            scale2, offset2 = _decode_dequant_u64(scale_dequant2)
            if bias_dequant2 is not None:
                pv = pv + bias_dequant2.to(torch.float32).reshape(1, -1)
            out = (pv * scale2.reshape(1, -1) + offset2.reshape(1, -1)).to(
                torch.float16
            )
            return out
        return None


@register("executor_aclnn_swin_attention_score_quant")
class SwinAttentionScoreQuantAclnnApi(AclnnBaseApi):
    """aclnnSwinAttentionScoreQuant NPU 透传类。

    ATK backend.convert_input_data 对 range_values 为 ["None"] 的可选 tensor
    转换出的对象可能无法通过 C 签名检查（如 c_void_p(None)）。参考已验证的
    GBSA 模式：遍历 input_data.kwargs 逐项 convert 并记录 name_to_idx，
    对 kwargs 中为 None 的可选输入在 input_args 中手动置空为 null_tensor_ptr，
    非 None 的保持原样；输出参数 out 走正常 convert_output_data，不做改动。
    """

    OPTIONAL_TENSOR_INPUTS = (
        "biasQuantOptional",
        "biasDequant1Optional",
        "biasDequant2Optional",
        "paddingMask1Optional",
        "paddingMask2Optional",
    )

    def init_by_input_data(self, input_data):
        input_args = []
        name_to_idx = {}
        for name, kwarg in input_data.kwargs.items():
            start = len(input_args)
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
            if data:
                name_to_idx[name] = start

        output_packages = []
        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)

        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)

        for name in self.OPTIONAL_TENSOR_INPUTS:
            if name in name_to_idx and input_data.kwargs.get(name) is None:
                input_args[name_to_idx[name]] = null_tensor_ptr

        return input_args, output_packages

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *scaleQuant, const aclTensor *scaleDequant1, "
            "const aclTensor *scaleDequant2, const aclTensor *biasQuantOptional, "
            "const aclTensor *biasDequant1Optional, const aclTensor *biasDequant2Optional, "
            "const aclTensor *paddingMask1Optional, const aclTensor *paddingMask2Optional, "
            "bool queryTranspose, bool keyTranspose, bool valueTranspose, int64_t softmaxAxes, "
            "aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
