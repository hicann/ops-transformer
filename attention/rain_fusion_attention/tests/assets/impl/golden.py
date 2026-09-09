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

"""RFA (RainFusionAttention) CPU Golden 实现 — 从 ATK 移植

纯计算模块，不含 TTK 框架依赖。

C API 参数顺序 (aclnnRainFusionAttentionGetWorkspaceSize):
  inputs: query, key, value, selectIdx, selectNumIdx, blockShape,
          attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
          blockTableOptional
  attr:   qInputLayout, kvInputLayout, numKeyValueHeads, maskType,
          scaleValue (double), innerPrecise, blockSize
  outputs: attentionOut, softmaxLseOptional
"""

import numpy as np
import torch


def safe_to_tensor(arr):
    if arr is None:
        return None
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


def _online_softmax_attention_torch_high(q_block, kv_blocks, scale):
    q_block = q_block.cpu()
    device = torch.device("cpu")
    q_len = q_block.shape[1]
    head_size = q_block.shape[2]

    m_i = torch.full((1, q_len, 1), -float("inf"), dtype=torch.float32, device=device)
    l_i = torch.zeros((1, q_len, 1), dtype=torch.float32, device=device)
    O_i = torch.zeros((1, q_len, head_size), dtype=torch.float32, device=device)

    for k_block, v_block in kv_blocks:
        k_block = k_block.cpu()
        v_block = v_block.cpu()
        S_i = torch.matmul(q_block.to(torch.float32), k_block.to(torch.float32))
        S_i = S_i * scale
        m_block, _ = torch.max(S_i, dim=-1, keepdim=True)
        m_new = torch.maximum(m_i, m_block)
        alpha = torch.exp(m_i - m_new)
        P_i = torch.exp(S_i - m_new)
        l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)
        O_i = alpha * O_i + torch.matmul(
            P_i.to(torch.float32), v_block.to(torch.float32)
        )
        m_i = m_new

    O_final = O_i / l_i
    return O_final


def _online_softmax_attention_torch(q_block, kv_blocks, scale, torch_dtype):
    q_block = q_block.cpu()
    device = torch.device("cpu")
    q_len = q_block.shape[1]
    head_size = q_block.shape[2]

    m_i = torch.full((1, q_len, 1), -float("inf"), dtype=torch.float32, device=device)
    l_i = torch.zeros((1, q_len, 1), dtype=torch.float32, device=device)
    O_i = torch.zeros((1, q_len, head_size), dtype=torch.float32, device=device)

    for k_block, v_block in kv_blocks:
        k_block = k_block.cpu()
        v_block = v_block.cpu()
        S_i = torch.matmul(q_block, k_block).to(torch_dtype)
        S_i = S_i * scale
        m_block, _ = torch.max(S_i, dim=-1, keepdim=True)
        m_new = torch.maximum(m_i, m_block)
        alpha = torch.exp(m_i - m_new).to(torch_dtype)
        P_i = torch.exp(S_i - m_new).to(torch_dtype)
        l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)
        O_i = alpha * O_i + torch.matmul(P_i, v_block).to(torch_dtype)
        m_i = m_new

    O_final = O_i / l_i
    return O_final


def _ref_select_idx_attention_torch(
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
    torch_dtype,
    inner_precise,
):
    if query.device.type != "cpu":
        query = query.cpu()
    if key.device.type != "cpu":
        key = key.cpu()
    if value.device.type != "cpu":
        value = value.cpu()

    device = torch.device("cpu")
    query = query.permute(1, 0, 2)
    key = key.permute(1, 2, 0)
    value = value.permute(1, 0, 2)
    num_heads = query.shape[0]
    kv_heads = key.shape[0]

    total_q_tokens = query.shape[1]
    head_size = query.shape[2]
    out = torch.zeros(
        (num_heads, total_q_tokens, head_size), dtype=query.dtype, device=device
    )

    q_token_offset = 0
    kv_token_offset = 0
    q_block_offset = 0

    for batch_idx in range(batch):
        q_seqlen = q_seqlen_list[batch_idx]
        kv_seqlen = kv_seqlen_list[batch_idx]
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y

        for t_local in range(s_block_num_q):
            t_global = q_block_offset + t_local
            q_block_idx = t_local
            q_start_local = q_block_idx * s_block_x
            q_end_local = min((q_block_idx + 1) * s_block_x, q_seqlen)
            q_start_global = q_token_offset + q_start_local
            q_end_global = q_token_offset + q_end_local

            for head in range(num_heads):
                select_idx_offset = (
                    t_global * num_heads * max_kv_block_num + head * max_kv_block_num
                )
                select_num_offset = t_global * num_heads + head
                select_num = select_num_idx_list[select_num_offset]
                selected_kv_blocks = select_idx_list[
                    select_idx_offset : select_idx_offset + max_kv_block_num
                ]

                q_block = query[head : head + 1, q_start_global:q_end_global, :]
                group_size = num_heads // kv_heads
                kv_head_idx = head // group_size

                kv_blocks = []
                k_blocks = []
                v_blocks = []
                if select_num == 0:
                    continue

                for kv_block_idx in selected_kv_blocks[:select_num]:
                    if kv_block_idx == -1:
                        continue
                    k_start_local = kv_block_idx * s_block_y
                    k_end_local = min((kv_block_idx + 1) * s_block_y, kv_seqlen)
                    if k_start_local >= k_end_local:
                        continue
                    k_start_global = kv_token_offset + k_start_local
                    k_end_global = kv_token_offset + k_end_local
                    k_block = key[
                        kv_head_idx : kv_head_idx + 1, :, k_start_global:k_end_global
                    ]
                    v_block = value[
                        kv_head_idx : kv_head_idx + 1, k_start_global:k_end_global, :
                    ]
                    k_blocks.append(k_block)
                    v_blocks.append(v_block)

                if not k_blocks:
                    continue

                k_block_com = torch.cat(k_blocks, dim=2)
                v_block_com = torch.cat(v_blocks, dim=1)
                k_total_len = k_block_com.shape[2]

                if k_total_len <= 512:
                    kv_blocks.append((k_block_com, v_block_com))
                else:
                    num_chunks = k_total_len // 512
                    remainder = k_total_len % 512
                    for i in range(num_chunks):
                        start = i * 512
                        end = start + 512
                        kv_blocks.append(
                            (k_block_com[:, :, start:end], v_block_com[:, start:end, :])
                        )
                    if remainder > 0:
                        kv_blocks.append(
                            (
                                k_block_com[:, :, -remainder:],
                                v_block_com[:, -remainder:, :],
                            )
                        )

                if query.dtype == torch.float32:
                    out_block = _online_softmax_attention_torch_high(
                        q_block, kv_blocks, scale
                    )
                else:
                    out_block = _online_softmax_attention_torch(
                        q_block, kv_blocks, scale, torch_dtype
                    )

                out[head : head + 1, q_start_global:q_end_global, :] = out_block

        q_token_offset += q_seqlen
        kv_token_offset += kv_seqlen
        q_block_offset += s_block_num_q

    out = out.permute(1, 0, 2)
    return out


def rain_fusion_attention_forward(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    block_shape,
    q_seqlen_list,
    kv_seqlen_list,
    scale_value,
    q_input_layout,
    kv_input_layout,
    inner_precise,
):
    """RFA 前向计算 — 与 ATK TestRainFusionAttentionTorch.calc_data 等价

    输入 shape (TND layout):
      query:         [total_q_tokens, num_heads, head_size]
      key:           [total_kv_tokens, num_heads, head_size]
      value:         [total_kv_tokens, num_heads, head_size]
      select_idx:    [total_q_blocks, num_heads, max_kv_block_num]
      select_num_idx:[total_q_blocks, num_heads]
      block_shape:   [block_shape_x, block_shape_y]
      q_seqlen_list: [batch] — 每个 batch 的 q 序列长度
      kv_seqlen_list:[batch] — 每个 batch 的 kv 序列长度

    返回:
      attention_out: [total_q_tokens, num_heads, head_size] (float32)
    """
    if not isinstance(query, torch.Tensor):
        query = safe_to_tensor(query).cpu()
    if not isinstance(key, torch.Tensor):
        key = safe_to_tensor(key).cpu()
    if not isinstance(value, torch.Tensor):
        value = safe_to_tensor(value).cpu()
    if not isinstance(select_idx, torch.Tensor):
        select_idx = safe_to_tensor(select_idx).cpu()
    if not isinstance(select_num_idx, torch.Tensor):
        select_num_idx = safe_to_tensor(select_num_idx).cpu()

    query = query.cpu()
    key = key.cpu()
    value = value.cpu()
    select_idx = select_idx.cpu()
    select_num_idx = select_num_idx.cpu()

    embedding_size = query.shape[2]
    num_heads = query.shape[1]
    batch_size = len(q_seqlen_list)

    if inner_precise == 1:
        scale_value = np.float16(scale_value)

    if q_input_layout == "TND" and kv_input_layout == "TND":
        if isinstance(q_seqlen_list, (list, tuple)):
            num_tokens = sum(q_seqlen_list)
        else:
            num_tokens = torch.tensor(q_seqlen_list).sum().item()
        head_size_vo = embedding_size

        shape_out = (num_tokens, num_heads, head_size_vo)

        query_dtype = query.dtype
        if isinstance(query_dtype, torch.dtype):
            torch_dtype = query_dtype
        elif query_dtype == np.float32 or str(query_dtype) == "float32":
            torch_dtype = torch.float32
        elif query_dtype == np.float16 or str(query_dtype) == "float16":
            torch_dtype = torch.float16
        elif query_dtype == np.bfloat16 or str(query_dtype) == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        if inner_precise == 1:
            torch_dtype = torch.float16

        total_q_blocks = select_idx.shape[0]
        max_kv_block_num = select_idx.shape[2]
        s_block_x = block_shape[0]
        s_block_y = block_shape[1]
        select_idx_list = select_idx.flatten().tolist()
        select_num_idx_list = select_num_idx.flatten().tolist()

        ref_output = _ref_select_idx_attention_torch(
            query,
            key,
            value,
            scale_value,
            select_idx_list,
            select_num_idx_list,
            s_block_x,
            s_block_y,
            total_q_blocks,
            max_kv_block_num,
            q_seqlen_list,
            kv_seqlen_list,
            batch_size,
            torch_dtype,
            inner_precise,
        )
        if query.dtype != torch.float32:
            ref_output_h = ref_output.to(torch.float32)
            return ref_output_h
        else:
            return ref_output


def rain_fusion_attention_forward_fp16(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    block_shape,
    q_seqlen_list,
    kv_seqlen_list,
    scale_value,
    q_input_layout,
    kv_input_layout,
    inner_precise,
):
    """原dtype精度版本的 RFA 前向计算

    模拟ATK CPU后端 (cpu_0) 用原dtype输入完整计算的标杆输出。
    与 rain_fusion_attention_forward 使用完全相同的代码, 但输入是原dtype而非高精度。

    ATK的三方对比:
      - cpu_benchmark (golden): fp32输入 → fp32输出 (高精度参考)
      - cpu_0 (benchmark):      fp16输入 → fp16输出 (与NPU相同dtype)
      - pyaclnn_0 (NPU):        fp16输入 → fp16输出
    """
    # 直接调用 rain_fusion_attention_forward, 用原dtype输入
    # 函数内部根据 query.dtype 选择计算精度, 输出转fp32
    # 再转回原dtype作为benchmark输出
    orig_dtype = query.dtype if isinstance(query, torch.Tensor) else torch.float16
    out = rain_fusion_attention_forward(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        block_shape,
        q_seqlen_list,
        kv_seqlen_list,
        scale_value,
        q_input_layout,
        kv_input_layout,
        inner_precise,
    )
    return out.to(orig_dtype)
