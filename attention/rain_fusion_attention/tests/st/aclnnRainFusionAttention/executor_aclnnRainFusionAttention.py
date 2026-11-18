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

"""aclnnRainFusionAttention ST executor。

CPU golden 逻辑改造自 tests/assets/impl/golden.py 的 RFA 参考实现:
按 selectIdx/selectNumIdx 选块做稀疏 attention (block-sparse + online softmax),
attentionOut = Softmax(scale * query * key^T + atten_mask) * value,
支持 TND/BNSD 布局与 GQA, 全程 float32 计算, 最终转回输入 dtype。
attenMaskOptional 按 add 到 QK^T 分数上的方式处理, blockTableOptional(PagedAttention)
与 softmaxLseOptional 当前不支持, golden 侧忽略。
"""

import ctypes

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray, AclTensor


def safe_to_tensor(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.cpu()
    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


def _as_scalar(value, default=None):
    """把 attr 标量 kwargs 值归一为 python 标量, 兼容单元素 list/dict 包裹。"""
    if value is None:
        return default
    if isinstance(value, dict):
        return _as_scalar(value.get("range_values"), default)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return _as_scalar(value[0], default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return value.cpu().flatten().tolist()[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.flatten().tolist()[0]
    return value


def _as_int_list(value):
    """把 aclIntArray 类 kwargs 值归一为 int list, 兼容 list/dict/tensor 包裹。"""
    if value is None:
        return None
    if isinstance(value, dict):
        return _as_int_list(value.get("range_values"))
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.cpu().flatten().tolist()]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.flatten().tolist()]
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, dict):
                nested = _as_int_list(item.get("range_values"))
                if nested:
                    result.extend(nested)
            elif isinstance(item, (list, tuple, torch.Tensor, np.ndarray)):
                result.extend(_as_int_list(item))
            elif isinstance(item, (int, float)):
                result.append(int(item))
        return result
    return None


def _as_string(value, default="TND"):
    if value is None:
        return default
    value = _as_scalar(value, default)
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _change_bnsd_to_tnd(tensor, seqlen_list):
    """BNSD [B, N, S, D] -> TND [sum(seqlen), N, D]"""
    head_dim = tensor.shape[-1]
    head_num = tensor.shape[1]
    token_num = sum(seqlen_list)
    batch = len(seqlen_list)
    res = torch.zeros((token_num, head_num, head_dim), dtype=tensor.dtype)
    count = 0
    for i in range(batch):
        res[count : count + seqlen_list[i], :, :] = tensor[
            i, :, : seqlen_list[i], :
        ].permute(1, 0, 2)
        count += seqlen_list[i]
    return res


def _change_tnd_to_bnsd(tensor, seqlen_list, max_seqlen):
    """TND [sum(seqlen), N, D] -> BNSD [B, N, max_seqlen, D]"""
    head_num = tensor.shape[1]
    head_dim = tensor.shape[2]
    batch = len(seqlen_list)
    res = torch.zeros((batch, head_num, max_seqlen, head_dim), dtype=tensor.dtype)
    count = 0
    for i in range(batch):
        res[i, :, : seqlen_list[i], :] = tensor[
            count : count + seqlen_list[i], :, :
        ].permute(1, 0, 2)
        count += seqlen_list[i]
    return res


def _mask_to_float(mask):
    """mask 转 float32 加法掩码。

    bool mask 采用 torch 约定: True 表示参与注意力, False 表示屏蔽(置为
    float32 最小值); 其他 dtype 按 add 方式直接转 float32。
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        result = torch.zeros(tuple(mask.shape), dtype=torch.float32)
        result[~mask] = torch.finfo(torch.float32).min
        return result
    return mask.to(torch.float32)


def _online_softmax_attention_fp32(q_block, kv_blocks, scale):
    """FlashAttention 风格 online softmax, 全程 float32 计算。

    kv_blocks: [(k_block, v_block, mask_block_or_None), ...]
      k_block: [1, D, kv_len], v_block: [1, kv_len, D], mask_block: [q_len, kv_len]
      或 [1, kv_len](广播)。
    """
    q_block = q_block.cpu().to(torch.float32)
    q_len = q_block.shape[1]
    head_size = q_block.shape[2]
    m_i = torch.full((1, q_len, 1), -float("inf"), dtype=torch.float32)
    l_i = torch.zeros((1, q_len, 1), dtype=torch.float32)
    o_i = torch.zeros((1, q_len, head_size), dtype=torch.float32)
    for k_block, v_block, mask_block in kv_blocks:
        k_block = k_block.cpu().to(torch.float32)
        v_block = v_block.cpu().to(torch.float32)
        s_i = torch.matmul(q_block, k_block) * scale
        if mask_block is not None:
            s_i = s_i + mask_block.to(torch.float32)
        m_block_max = torch.max(s_i, dim=-1, keepdim=True)[0]
        m_new = torch.maximum(m_i, m_block_max)
        alpha = torch.exp(m_i - m_new)
        p_i = torch.exp(s_i - m_new)
        l_i = alpha * l_i + torch.sum(p_i, dim=-1, keepdim=True)
        o_i = alpha * o_i + torch.matmul(p_i, v_block)
        m_i = m_new
    return o_i / l_i


def _ref_select_idx_attention(
    query,
    key,
    value,
    scale,
    select_idx_list,
    select_num_idx_list,
    s_block_x,
    s_block_y,
    total_q_blocks,
    max_kv_block_num,
    q_seqlen_list,
    kv_seqlen_list,
    batch,
    atten_mask,
):
    """按 selectIdx/selectNumIdx 稀疏选块的 attention 参考计算 (TND 输入)。

    query: [total_q_tokens, num_heads, head_size]
    key/value: [total_kv_tokens, kv_heads, head_size]
    selectIdx: [total_q_blocks, num_heads, max_kv_block_num], 有效索引在前、-1 填充
    selectNumIdx: [total_q_blocks, num_heads]
    """
    query = query.permute(1, 0, 2)  # [N, T, D]
    key = key.permute(1, 2, 0)  # [N2, D, Tkv]
    value = value.permute(1, 0, 2)  # [N2, Tkv, D]
    num_heads = query.shape[0]
    kv_heads = key.shape[0]
    total_q_tokens = query.shape[1]
    head_size = query.shape[2]
    out = torch.zeros((num_heads, total_q_tokens, head_size), dtype=torch.float32)
    group_size = max(1, num_heads // kv_heads)

    atten_mask_f = _mask_to_float(atten_mask) if atten_mask is not None else None

    q_token_offset = 0
    kv_token_offset = 0
    q_block_offset = 0
    for batch_idx in range(batch):
        q_seqlen = q_seqlen_list[batch_idx]
        kv_seqlen = kv_seqlen_list[batch_idx]
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x

        for t_local in range(s_block_num_q):
            t_global = q_block_offset + t_local
            q_start_global = q_token_offset + t_local * s_block_x
            q_end_global = q_token_offset + min((t_local + 1) * s_block_x, q_seqlen)
            if q_start_global >= q_end_global:
                continue

            for head in range(num_heads):
                select_idx_offset = (
                    t_global * num_heads * max_kv_block_num + head * max_kv_block_num
                )
                select_num_offset = t_global * num_heads + head
                select_num = select_num_idx_list[select_num_offset]
                if select_num <= 0:
                    continue
                selected_kv_blocks = select_idx_list[
                    select_idx_offset : select_idx_offset + max_kv_block_num
                ]
                q_block = query[head : head + 1, q_start_global:q_end_global, :]
                kv_head_idx = head // group_size

                k_blocks = []
                v_blocks = []
                m_blocks = []
                for kv_block_idx in selected_kv_blocks[:select_num]:
                    if kv_block_idx < 0:  # 跳过 -1 填充
                        continue
                    k_start_local = kv_block_idx * s_block_y
                    k_end_local = min((kv_block_idx + 1) * s_block_y, kv_seqlen)
                    if k_start_local >= k_end_local:
                        continue
                    k_start_global = kv_token_offset + k_start_local
                    k_end_global = kv_token_offset + k_end_local
                    k_blocks.append(
                        key[
                            kv_head_idx : kv_head_idx + 1,
                            :,
                            k_start_global:k_end_global,
                        ]
                    )
                    v_blocks.append(
                        value[
                            kv_head_idx : kv_head_idx + 1,
                            k_start_global:k_end_global,
                            :,
                        ]
                    )
                    if atten_mask_f is not None:
                        if atten_mask_f.shape[0] > 1:
                            m_blocks.append(
                                atten_mask_f[
                                    q_start_global:q_end_global,
                                    k_start_global:k_end_global,
                                ]
                            )
                        else:
                            m_blocks.append(
                                atten_mask_f[:, k_start_global:k_end_global]
                            )

                if not k_blocks:
                    continue

                k_com = torch.cat(k_blocks, dim=2)
                v_com = torch.cat(v_blocks, dim=1)
                m_com = torch.cat(m_blocks, dim=1) if m_blocks else None

                kv_blocks = []
                chunk = 512
                kv_total_len = k_com.shape[2]
                if kv_total_len <= chunk:
                    kv_blocks.append((k_com, v_com, m_com))
                else:
                    num_chunks = kv_total_len // chunk
                    remainder = kv_total_len % chunk
                    for i in range(num_chunks):
                        start = i * chunk
                        end = start + chunk
                        mask_chunk = None if m_com is None else m_com[:, start:end]
                        kv_blocks.append(
                            (k_com[:, :, start:end], v_com[:, start:end, :], mask_chunk)
                        )
                    if remainder > 0:
                        mask_chunk = None if m_com is None else m_com[:, -remainder:]
                        kv_blocks.append(
                            (
                                k_com[:, :, -remainder:],
                                v_com[:, -remainder:, :],
                                mask_chunk,
                            )
                        )

                out_block = _online_softmax_attention_fp32(q_block, kv_blocks, scale)
                out[head : head + 1, q_start_global:q_end_global, :] = out_block

        q_token_offset += q_seqlen
        kv_token_offset += kv_seqlen
        q_block_offset += s_block_num_q

    out = out.permute(1, 0, 2)
    return out


def rain_fusion_attention_golden(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    block_shape,
    atten_mask,
    q_seqlen_list,
    kv_seqlen_list,
    scale_value,
    q_input_layout,
    kv_input_layout,
):
    """RFA 前向 golden: float32 计算, 输出转回输入 dtype, shape 与 query 一致。"""
    query = safe_to_tensor(query)
    key = safe_to_tensor(key)
    value = safe_to_tensor(value)
    select_idx = safe_to_tensor(select_idx)
    select_num_idx = safe_to_tensor(select_num_idx)
    atten_mask = safe_to_tensor(atten_mask)

    orig_dtype = query.dtype
    s_block_x = int(block_shape[0])
    s_block_y = int(block_shape[1])
    q_seqlen_list = [int(v) for v in q_seqlen_list]
    kv_seqlen_list = [int(v) for v in kv_seqlen_list]
    batch = len(q_seqlen_list)

    is_bnsd = q_input_layout == "BNSD" and kv_input_layout == "BNSD"
    if is_bnsd:
        max_q_seqlen = query.shape[2]
        query_tnd = _change_bnsd_to_tnd(query.to(torch.float32), q_seqlen_list)
        key_tnd = _change_bnsd_to_tnd(key.to(torch.float32), kv_seqlen_list)
        value_tnd = _change_bnsd_to_tnd(value.to(torch.float32), kv_seqlen_list)
    else:
        query_tnd = query.to(torch.float32)
        key_tnd = key.to(torch.float32)
        value_tnd = value.to(torch.float32)

    total_q_blocks = select_idx.shape[0]
    max_kv_block_num = select_idx.shape[2]
    select_idx_list = select_idx.cpu().flatten().tolist()
    select_num_idx_list = select_num_idx.cpu().flatten().tolist()

    out_tnd = _ref_select_idx_attention(
        query_tnd,
        key_tnd,
        value_tnd,
        float(scale_value),
        select_idx_list,
        select_num_idx_list,
        s_block_x,
        s_block_y,
        total_q_blocks,
        max_kv_block_num,
        q_seqlen_list,
        kv_seqlen_list,
        batch,
        atten_mask,
    )

    if is_bnsd:
        out = _change_tnd_to_bnsd(out_tnd, q_seqlen_list, max_q_seqlen)
    else:
        out = out_tnd
    return out.to(orig_dtype)


@register("executor_rain_fusion_attention")
class RainFusionAttentionApi(BaseApi):
    """aclnnRainFusionAttention CPU golden。"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = input_data.kwargs
        query = kwargs["query"]
        key = kwargs["key"]
        value = kwargs["value"]
        select_idx = kwargs["selectIdx"]
        select_num_idx = kwargs["selectNumIdx"]
        block_shape = _as_int_list(kwargs["blockShape"])
        atten_mask = kwargs["attenMaskOptional"]
        q_seqlen_list = _as_int_list(kwargs["actualSeqLengthsOptional"])
        kv_seqlen_list = _as_int_list(kwargs["actualSeqLengthsKvOptional"])
        q_layout = _as_string(kwargs["qInputLayout"])
        kv_layout = _as_string(kwargs["kvInputLayout"])
        scale_value = float(_as_scalar(kwargs["scaleValue"], 1.0))

        # TND/BNSD 场景 seqlen 必选, 仅在缺失时按整包兜底推断
        if not q_seqlen_list:
            if q_layout == "BNSD":
                q_seqlen_list = [int(query.shape[2])] * int(query.shape[0])
            else:
                q_seqlen_list = [int(query.shape[0])]
        if not kv_seqlen_list:
            if kv_layout == "BNSD":
                kv_seqlen_list = [int(key.shape[2])] * int(key.shape[0])
            else:
                kv_seqlen_list = [int(key.shape[0])]

        attention_out_golden = rain_fusion_attention_golden(
            query,
            key,
            value,
            select_idx,
            select_num_idx,
            block_shape,
            atten_mask,
            q_seqlen_list,
            kv_seqlen_list,
            scale_value,
            q_layout,
            kv_layout,
        )
        return attention_out_golden, torch.ones(0)


@register("executor_aclnn_rain_fusion_attention")
class RainFusionAttentionApiNPU(AclnnBaseApi):
    """aclnnRainFusionAttention NPU 两段式调用。"""

    def __init__(self, task_result: TaskResult, backend):
        super(RainFusionAttentionApiNPU, self).__init__(task_result, backend)
        self.input = None

    def init_by_input_data(self, input_data: InputDataset):
        # GBSA 参考模式: 手动两段式组包, 记录 name_to_idx, 按名字置空 None 可选输入,
        # 避免 backend.convert_input_data 对 ["None"] 可选 tensor 产生的
        # c_void_p(None) 无法通过 C 签名检查。
        self.input = input_data
        input_args = []
        name_to_idx = {}
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
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
        out_start = len(input_args)
        input_args.extend(output_packages)

        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        null_int_array_ptr = ctypes.cast(null_void_ptr, ctypes.POINTER(AclIntArray))

        # softmaxLseOptional 为输出参数(签名最后一位 tensor): 一律置空并移除其输出包
        if len(output_packages) >= 2:
            input_args[out_start + len(output_packages) - 1] = null_tensor_ptr
            output_packages.pop()
        elif len(output_packages) == 1:
            # 缺失 lse 槽位时补空指针, 保持与 C 签名参数个数一致
            input_args.append(null_tensor_ptr)

        # json 中声明为 ["None"] 的可选输入(kwargs 值为 None)按名字统一置空
        optional_tensor_names = (
            "attenMaskOptional",
            "blockTableOptional",
            "selectIdx",
            "selectNumIdx",
        )
        for name in optional_tensor_names:
            if name in name_to_idx and input_data.kwargs.get(name) is None:
                input_args[name_to_idx[name]] = null_tensor_ptr
        optional_int_array_names = (
            "actualSeqLengthsOptional",
            "actualSeqLengthsKvOptional",
        )
        for name in optional_int_array_names:
            if name in name_to_idx and input_data.kwargs.get(name) is None:
                input_args[name_to_idx[name]] = null_int_array_ptr

        return input_args, output_packages

    def after_call(self, output_packages):
        result = super().after_call(output_packages)
        if isinstance(result, (list, tuple)):
            attention_out = result[0]
        else:
            attention_out = result
        return attention_out, torch.ones(0)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnRainFusionAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *selectIdx, const aclTensor *selectNumIdx, const aclIntArray *blockShape, const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"
