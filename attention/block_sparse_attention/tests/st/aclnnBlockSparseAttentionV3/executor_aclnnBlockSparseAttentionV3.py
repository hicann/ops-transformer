# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import logging
import logging as _logging
import math
import numpy as np
import os
import random
import struct
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import zlib
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import (
    ACLTYPE_TO_CTYPE,
    AclDataType,
    AclFormat,
    AclIntArray,
    AclTensor,
    AclTensorStruct,
    Int64,
    TORCH_TO_ACLTYPE,
    TensorPtr,
    ascendcl,
    nnopbase,
)
from dataclasses import dataclass
from ml_dtypes import bfloat16
from typing import List, Optional, Sequence, Tuple


def inspect_kwargs(kwargs):
    """
    遍历打印 kwargs 中的 key 和 value。
    如果是 Tensor/Array，只打印 shape 和 dtype。
    如果是标量或普通列表，打印具体值。
    """
    print(f"\n{'=' * 20} DEBUG: Inspect Kwargs {'=' * 20}")

    if not isinstance(kwargs, dict):
        print(f"Error: Expected dict, got {type(kwargs)}")
        return

    for key, value in kwargs.items():
        # 尝试获取类型名称
        type_name = type(value).__name__

        # 1. 处理 PyTorch Tensor
        if isinstance(value, torch.Tensor):
            print(
                f"[Tensor]  Key: {key:<25} | Shape: {list(value.shape)} | Dtype: {value.dtype} | Device: {value.device}"
            )

        # 2. 处理 Numpy Array
        elif isinstance(value, np.ndarray):
            print(
                f"[Numpy]   Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}"
            )

        # 3. 处理包含 Tensor 的列表 (常见于 List[Tensor])
        elif (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and (isinstance(value[0], torch.Tensor) or isinstance(value[0], np.ndarray))
        ):
            shape_info = list(value[0].shape) if hasattr(value[0], "shape") else "N/A"
            print(
                f"[List_T]  Key: {key:<25} | Len: {len(value)} | Item0_Shape: {shape_info}"
            )

        # 4. 处理自定义对象 (如果有 shape 属性)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            print(
                f"[Custom]  Key: {key:<25} | Shape: {value.shape} | Dtype: {value.dtype}"
            )

        # 5. 其他标量或普通对象，直接打印值
        else:
            # 如果值太长，截断一下
            str_val = str(value)
            if len(str_val) > 100:
                str_val = str_val[:100] + "..."
            print(f"[Scalar]  Key: {key:<25} | Value: {str_val}")

    print(f"{'=' * 62}\n")


def safe_to_tensor(arr):
    if arr is None:
        return None
    # 如果是 list，先转为 numpy array
    if isinstance(arr, list):
        arr = np.array(arr)
    # 如果是 bfloat16，先转 float32 再转 torch
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    # 其他类型直接转换
    return torch.from_numpy(arr)


@register("aclnn_blocksparseattentionv3inputprocess")
class BlockSparseAttentionV3InputProcess(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(BlockSparseAttentionV3InputProcess, self).__init__(task_result, backend)
        self.qSeqlenList = []

    @classmethod
    def change_bnsd_to_tnd(self, tensor, seqlenList):
        """
        把BNSD格式的tensor转换成TND格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        tokenNum = sum(seqlenList)
        batch = len(seqlenList)
        res = torch.zeros((tokenNum, headNum, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[count : count + seqlenList[i], :, :] = tensor[
                i, :, : seqlenList[i], :
            ].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res

    def init_by_input_data(self, input_data):
        np.random.seed(10)
        torch.npu.synchronize()
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型

        # 将tensor重新to到npu上，防止流同步失败
        query = self.acl_tensor_to_torch(input_args[0]).to("npu")
        input_args[0] = self.torch_tensor_to_acl(query)
        key = self.acl_tensor_to_torch(input_args[1]).to("npu")
        input_args[1] = self.torch_tensor_to_acl(key)
        value = self.acl_tensor_to_torch(input_args[2]).to("npu")
        input_args[2] = self.torch_tensor_to_acl(value)
        blockSparseMask = self.acl_tensor_to_torch(input_args[3]).to("npu")
        input_args[3] = self.torch_tensor_to_acl(blockSparseMask)
        output_packages = []
        input_args[4] = null_tensor_ptr  # attenMask
        self.qSeqlenList = input_data.kwargs["actualSeqLengths"]
        input_args[8] = null_tensor_ptr  # blockTable
        input_args[9] = null_tensor_ptr  # qDequantScale
        input_args[10] = null_tensor_ptr  # kDequantScale
        input_args[11] = null_tensor_ptr  # vDequantScale
        input_args[12] = null_tensor_ptr  # pQuantScale
        input_args[26] = null_tensor_ptr  # softmaxLse

        input_args.pop()
        output_packages.append(input_args[-2])

        return input_args, output_packages

    def __call__(self):
        """算子调用逻辑"""
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(
                dtype=torch.float32
            )
            output.append(temp_output_pack)
        batch = len(self.qSeqlenList)
        count = 0
        tokenNum = sum(self.qSeqlenList)

        if output[0].dim() == 4:
            outputTemp = torch.zeros(
                (tokenNum, output[0].shape[1], output[0].shape[-1]),
                dtype=output[0].dtype,
            )
            outputTensor = output[0]
            for i in range(batch):
                outputTemp[count : count + self.qSeqlenList[i], :, :] = outputTensor[
                    i, :, : self.qSeqlenList[i], :
                ].permute(1, 0, 2)
                count += self.qSeqlenList[i]
            return [outputTemp]
        else:
            return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMaskOptional, const aclTensor *attenMaskOptional, const aclIntArray *blockShapeOptional, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode, double dstTypeMax, aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"


class TestBlockSparseAttentionTorch:
    @classmethod
    def online_softmax_attention_torch_high(cls, q_block, kv_blocks, scale):
        # 确保在 CPU 上运行
        q_block = q_block.cpu()
        device = torch.device("cpu")
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]

        # 初始化状态量（确保在 CPU 上）
        m_i = torch.full(
            (1, q_len, 1), -float("inf"), dtype=torch.float32, device=device
        )  # running max
        l_i = torch.zeros(
            (1, q_len, 1), dtype=torch.float32, device=device
        )  # running sum
        O_i = torch.zeros(
            (1, q_len, head_size), dtype=torch.float32, device=device
        )  # running output
        is_first = 1
        # 逐块处理
        for k_block, v_block in kv_blocks:
            # 确保 kv blocks 在 CPU 上
            k_block = k_block.cpu()
            v_block = v_block.cpu()
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            S_i = torch.matmul(
                q_block.to(torch.float32), k_block.to(torch.float32)
            )  # (1, q_len, k_len)

            S_i = S_i * scale

            # 2. 计算当前块的最大值
            m_block, _ = torch.max(S_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 3. 计算新的全局最大值
            m_new = torch.maximum(m_i, m_block)  # (1, q_len, 1)

            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = torch.exp(m_i - m_new)  # (1, q_len, 1)

            # 5. 计算当前块的稳定 softmax 分子
            P_i = torch.exp(S_i - m_new)  # (1, q_len, k_len)

            # 6. 更新 running sum
            l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 7. 更新 running output
            # O_new = alpha * O_old + P_i @ V_i
            O_i = alpha * O_i + torch.matmul(
                P_i.to(torch.float32), v_block.to(torch.float32)
            )  # (1, q_len, head_size)

            # 8. 更新 running max
            m_i = m_new

        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)

        return O_final

    @classmethod
    def online_softmax_attention_torch(
        cls, q_block, kv_blocks, scale, torch_dtype, input_dtype, inner_precise
    ):
        # 确保在 CPU 上运行
        q_block = q_block.cpu()
        device = torch.device("cpu")
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]

        # 初始化状态量（确保在 CPU 上）
        m_i = torch.full(
            (1, q_len, 1), -float("inf"), dtype=torch.float32, device=device
        )  # running max
        l_i = torch.zeros(
            (1, q_len, 1), dtype=torch.float32, device=device
        )  # running sum
        O_i = torch.zeros(
            (1, q_len, head_size), dtype=torch.float32, device=device
        )  # running output
        is_first = 1
        # 逐块处理
        for k_block, v_block in kv_blocks:
            # 确保 kv blocks 在 CPU 上
            k_block = k_block.cpu()
            v_block = v_block.cpu()
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            # CPU不支持half精度matmul，需要转换为float32计算
            S_i = torch.matmul(q_block.float(), k_block.float()).to(
                torch_dtype
            )  # (1, q_len, k_len)

            S_i = S_i * scale

            # 2. 计算当前块的最大值
            m_block, _ = torch.max(S_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 3. 计算新的全局最大值
            m_new = torch.maximum(m_i, m_block)  # (1, q_len, 1)

            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = torch.exp(m_i - m_new).to(torch_dtype)  # (1, q_len, 1)

            # 5. 计算当前块的稳定 softmax 分子
            P_i = torch.exp(S_i - m_new).to(torch_dtype)  # (1, q_len, k_len)

            # 6. 更新 running sum
            l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 7. 更新 running output
            # CPU不支持half精度matmul，需要转换为float32计算
            O_i = alpha * O_i + torch.matmul(P_i.float(), v_block.float()).to(
                torch_dtype
            )  # (1, q_len, head_size)

            # 8. 更新 running max
            m_i = m_new

        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)

        return O_final

    def ref_select_idx_attention_torch(
        self,
        query,
        key,
        value,
        scale: float,
        select_idx_list: list,
        select_num_idx_list: list,
        s_block_x: int,
        s_block_y: int,
        total_q_blocks: int,
        max_kv_block_num: int,
        q_seqlen_list: list,
        kv_seqlen_list: list,
        batch: int,
        torch_dtype,
        inner_precise,
    ):
        """
        PyTorch 版本的 ref_select_idx_attention
        确保所有计算在 CPU 上进行
        """
        # 确保输入在 CPU 上
        if query.device.type != "cpu":
            query = query.cpu()
        if key.device.type != "cpu":
            key = key.cpu()
        if value.device.type != "cpu":
            value = value.cpu()

        device = torch.device("cpu")

        # 转置操作
        query = query.permute(
            1, 0, 2
        )  # (total_q_tokens, num_heads, head_size) -> (num_heads, total_q_tokens, head_size)
        key = key.permute(
            1, 2, 0
        )  # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, head_size, total_kv_tokens)
        value = value.permute(
            1, 0, 2
        )  # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, total_kv_tokens, head_size)
        num_heads = query.shape[0]
        kv_heads = key.shape[0]

        # 初始化输出 - 注意这里应该是total_q_tokens而不是max_q_seqlen
        total_q_tokens = query.shape[1]
        head_size = query.shape[2]
        out_high = torch.zeros(
            (num_heads, total_q_tokens, head_size), dtype=torch.float32, device=device
        )
        out = torch.zeros(
            (num_heads, total_q_tokens, head_size), dtype=query.dtype, device=device
        )

        # 【关键修复】：添加batch级别的累计偏移量
        q_token_offset = 0  # Q方向token累计偏移
        kv_token_offset = 0  # KV方向token累计偏移
        q_block_offset = 0  # Q块累计偏移（用于selectIdx索引）

        for batch_idx in range(batch):
            q_seqlen = q_seqlen_list[batch_idx]
            kv_seqlen = kv_seqlen_list[batch_idx]

            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y

            for t_local in range(s_block_num_q):
                t_global = q_block_offset + t_local
                q_block_idx = t_local  # 当前batch内的Q块索引

                # 【关键修复】：batch内的相对位置
                q_start_local = q_block_idx * s_block_x
                q_end_local = min((q_block_idx + 1) * s_block_x, q_seqlen)

                # 【关键修复】：加上batch偏移得到全局位置
                q_start_global = q_token_offset + q_start_local
                q_end_global = q_token_offset + q_end_local

                for head in range(num_heads):
                    # 获取该头对应的selectIdx
                    select_idx_offset = (
                        t_global * num_heads * max_kv_block_num
                        + head * max_kv_block_num
                    )
                    select_num_offset = t_global * num_heads + head

                    select_num = select_num_idx_list[select_num_offset]
                    selected_kv_blocks = select_idx_list[
                        select_idx_offset : select_idx_offset + max_kv_block_num
                    ]

                    # 【GQA修复】：在循环之前提取 q_block 和计算 GQA 参数（只需要一次）
                    q_block = query[
                        head : head + 1, q_start_global:q_end_global, :
                    ]  # (1, q_block_size, head_size)

                    # 处理 group attention 的情况
                    group_size = num_heads // kv_heads
                    kv_head_idx = head // group_size

                    # 收集所有选中的KV块数据（作为 (K, V) 元组列表）
                    kv_blocks = []
                    k_blocks = []
                    v_blocks = []
                    if select_num == 0:
                        continue

                    for kv_block_idx in selected_kv_blocks[:select_num]:
                        if kv_block_idx == -1:  # 跳过填充的-1
                            continue

                        # 【关键修复】：batch内的相对位置
                        k_start_local = kv_block_idx * s_block_y
                        k_end_local = min((kv_block_idx + 1) * s_block_y, kv_seqlen)

                        # 【关键修复】：加上batch偏移得到全局位置
                        k_start_global = kv_token_offset + k_start_local
                        k_end_global = kv_token_offset + k_end_local

                        # 【修复】：使用全局位置访问key和value
                        k_block = key[
                            kv_head_idx : kv_head_idx + 1,
                            :,
                            k_start_global:k_end_global,
                        ]  # (1, head_size, k_block_size)
                        v_block = value[
                            kv_head_idx : kv_head_idx + 1,
                            k_start_global:k_end_global,
                            :,
                        ]  # (1, k_block_size, head_size)

                        k_blocks.append(k_block)
                        v_blocks.append(v_block)

                    k_block_com = torch.cat(k_blocks, dim=2)
                    v_block_com = torch.cat(v_blocks, dim=1)

                    k_total_len = k_block_com.shape[2]
                    v_total_len = v_block_com.shape[1]

                    if k_total_len <= 512:
                        kv_blocks.append((k_block_com, v_block_com))
                    else:
                        num_chunks = k_total_len // 512
                        remainder = k_total_len % 512
                        for i in range(num_chunks):
                            start = i * 512
                            end = start + 512
                            k_chunk = k_block_com[:, :, start:end]
                            v_chunk = v_block_com[:, start:end, :]
                            kv_blocks.append((k_chunk, v_chunk))
                        if remainder > 0:
                            k_last_chunk = k_block_com[:, :, -remainder:]
                            v_last_chunk = v_block_com[:, -remainder:, :]
                            kv_blocks.append((k_last_chunk, v_last_chunk))
                    # 使用 Online Softmax 计算注意力（FlashAttention 风格）
                    if query.dtype == torch.float32:
                        out_block = self.online_softmax_attention_torch_high(
                            q_block, kv_blocks, scale
                        )  # (1, q_block_size, head_size)
                    else:
                        out_block = self.online_softmax_attention_torch(
                            q_block,
                            kv_blocks,
                            scale,
                            torch_dtype,
                            query.dtype,
                            inner_precise,
                        )
                    # 【修复】：输出到全局位置
                    # out_high[head:head+1, q_start_global:q_end_global, :] = out_block_high
                    out[head : head + 1, q_start_global:q_end_global, :] = out_block
            # 【关键修复】：更新累计偏移量，为下一个batch做准备
            q_token_offset += q_seqlen
            kv_token_offset += kv_seqlen
            q_block_offset += s_block_num_q

        out = out.permute(1, 0, 2)
        return out

    @classmethod
    def change_bnsd_to_tnd(self, tensor, seqlenList):
        """
        把BNSD格式的tensor转换成TND格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        tokenNum = sum(seqlenList)
        batch = len(seqlenList)
        res = torch.zeros((tokenNum, headNum, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[count : count + seqlenList[i], :, :] = tensor[
                i, :, : seqlenList[i], :
            ].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res

    @classmethod
    def change_tnd_to_bnsd(self, tensor, seqlenList, maxQSeqlen):
        """
        把TND格式的tensor转换成BNSD格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        batch = len(seqlenList)
        res = torch.zeros((batch, headNum, maxQSeqlen, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[i, :, : seqlenList[i], :] = tensor[
                count : count + seqlenList[i], :, :
            ].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res

    def calc_data(
        self,
        query_dtype,
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
        """
        PyTorch 版本的 calc_data
        确保所有计算在 CPU 上进行
        """
        # 确保输入是 torch.Tensor，如果是 numpy 数组则转换
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

        # 确保在 CPU 上
        query = query.cpu()
        key = key.cpu()
        value = value.cpu()
        select_idx = select_idx.cpu()
        select_num_idx = select_num_idx.cpu()
        maxQSeqlen = 0
        if q_input_layout == "BNSD":
            maxQSeqlen = query.shape[-2]
            query = self.change_bnsd_to_tnd(query, q_seqlen_list)
            key = self.change_bnsd_to_tnd(key, kv_seqlen_list)
            value = self.change_bnsd_to_tnd(value, kv_seqlen_list)
        embedding_size = query.shape[2]
        num_heads = query.shape[1]
        batch_size = len(q_seqlen_list)

        if inner_precise == 1:
            scale_value = np.float16(scale_value)

        # if q_input_layout == 'TND' and kv_input_layout == 'TND':
        # 使用 torch 计算总和
        if isinstance(q_seqlen_list, (list, tuple)):
            num_tokens = sum(q_seqlen_list)
        else:
            num_tokens = torch.tensor(q_seqlen_list).sum().item()
        head_size_vo = embedding_size

        shape_out = (num_tokens, num_heads, head_size_vo)
        # 根据 query_dtype 确定 torch dtype
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

        input_type = torch_dtype
        if inner_precise == 1:
            torch_dtype = torch.float16

        ref_output = torch.zeros(
            shape_out, dtype=torch_dtype, device=torch.device("cpu")
        )
        ref_output_high = torch.zeros(
            shape_out, dtype=torch.float32, device=torch.device("cpu")
        )

        total_q_blocks = select_idx.shape[0]
        max_kv_block_num = select_idx.shape[2]
        s_block_x = block_shape[0]
        s_block_y = block_shape[1]
        select_idx_list = select_idx.flatten().tolist()
        select_num_idx_list = select_num_idx.flatten().tolist()
        ref_output = self.ref_select_idx_attention_torch(
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


@register("aclnn_blocksparseattentionv3")
class BlockSparseAttentionV3Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionV3Api, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["query"]
        query_dtype = query.dtype
        key = input_data.kwargs["key"]
        value = input_data.kwargs["value"]
        block_shape = input_data.kwargs["blockShape"]
        block_sparse_mask = input_data.kwargs["blockSparseMask"]
        atten_mask = input_data.kwargs["attenMask"]
        q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        block_table = input_data.kwargs["blockTable"]
        q_input_layout = input_data.kwargs["qInputLayout"]
        kv_input_layout = input_data.kwargs["kvInputLayout"]
        kv_heads = input_data.kwargs["numKeyValueHeads"]
        mask_type = input_data.kwargs["maskType"]
        scale_value = input_data.kwargs["scaleValue"]
        inner_precise = input_data.kwargs["innerPrecise"]
        block_size = input_data.kwargs["blockSize"]
        q_input_value = query.cpu()
        k_input_value = key.cpu()
        v_input_value = value.cpu()
        # 从blockSparseMask转换为selectIdx和selectNumIdx
        batch = len(q_seqlen_list)
        select_idx_input, select_num_idx_input = (
            self.change_block_sparsemask_to_selectidx_selctnumidx(
                block_sparse_mask, q_seqlen_list, kv_seqlen_list, block_shape, batch
            )
        )
        testObj = TestBlockSparseAttentionTorch()
        atten_out_golden = testObj.calc_data(
            query_dtype,
            q_input_value,
            k_input_value,
            v_input_value,
            select_idx_input,
            select_num_idx_input,
            block_shape,
            q_seqlen_list,
            kv_seqlen_list,
            scale_value,
            q_input_layout,
            kv_input_layout,
            inner_precise,
        )
        return atten_out_golden

    def change_block_sparsemask_to_selectidx_selctnumidx(
        self, block_sparse_mask, q_seqlen_list, kv_seqlen_list, block_shape, batch
    ):
        """
        把blocksparseMask转为selectIdx和selectNumIdx
        其中blockSparseMask为[b, headNum, maxQBlockNum, maxKvBlockNum]
        selectIdx为[QBlockNum, headNum, maxKvBlockNum]维度，其中第一维表示每个batch块的合轴
        selectNumIdx为[QBlockNum, headNum]维度表示每个Q方向块有效的KV块数
        """
        import numpy as np

        s_block_x = block_shape[0]
        s_block_y = block_shape[1]
        # blockSparseMask shape: [b, headNum, maxQBlockNum, maxKvBlockNum]
        num_heads = block_sparse_mask.shape[1]
        max_q_block_num = block_sparse_mask.shape[2]
        max_kv_block_num = block_sparse_mask.shape[3]
        # 计算总的Q块数
        total_q_blocks = 0
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            total_q_blocks += s_block_num_q
        # 初始化selectIdx和selectNumIdx列表
        select_idx_list = []
        select_num_idx_list = []
        # 遍历每个batch
        q_block_offset = 0
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            kv_seqlen = kv_seqlen_list[b]
            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            # 遍历当前batch的每个Q块
            for q_block_idx in range(s_block_num_q):
                # 遍历每个head
                for head in range(num_heads):
                    # 从blockSparseMask中提取当前位置的mask: [maxKvBlockNum]
                    mask_row = block_sparse_mask[b, head, q_block_idx, :]
                    # 找出所有非零的KV块索引（稀疏mask中非零表示该KV块被选中）
                    selected_kv_blocks = []
                    for kv_block_idx in range(s_block_num_kv):
                        if (
                            kv_block_idx < max_kv_block_num
                            and mask_row[kv_block_idx] != 0
                        ):
                            selected_kv_blocks.append(kv_block_idx)
                    # 填充到max_kv_block_num长度，不足的用-1填充
                    padded_blocks = selected_kv_blocks + [-1] * (
                        max_kv_block_num - len(selected_kv_blocks)
                    )
                    # 添加到selectIdx
                    select_idx_list.extend(padded_blocks)
                # 为当前Q块的每个head记录选中的KV块数量
                for head in range(num_heads):
                    mask_row = block_sparse_mask[b, head, q_block_idx, :]
                    num_selected = 0
                    for kv_block_idx in range(s_block_num_kv):
                        if (
                            kv_block_idx < max_kv_block_num
                            and mask_row[kv_block_idx] != 0
                        ):
                            num_selected += 1
                    select_num_idx_list.append(num_selected)
            q_block_offset += s_block_num_q
        # 将列表转换为tensor并reshape为正确的shape
        select_idx_tensor = torch.tensor(select_idx_list, dtype=torch.int32).view(
            total_q_blocks, num_heads, max_kv_block_num
        )
        select_num_idx_tensor = torch.tensor(
            select_num_idx_list, dtype=torch.int32
        ).view(total_q_blocks, num_heads)
        return select_idx_tensor, select_num_idx_tensor

    def gen_select_idx_data(
        self,
        q_seqlen_list,
        kv_seqlen_list,
        s_block_x,
        s_block_y,
        batch,
        num_heads,
        select_ratio,
    ):
        """
        @brief 生成新的selectIdx数据格式，支持变长序列
        @param q_seqlen_list  每个batch的Q序列长度列表
        @param kv_seqlen_list 每个batch的KV序列长度列表
        @param s_block_x      Q方向分块大小
        @param s_block_y      KV方向分块大小
        @param batch          batch数
        @param num_heads      注意力头数
        @param sparsity_ratio 稀疏度比例，控制每个Q块选择多少比例的KV块 (默认0.3表示30%的KV块)
        @return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num
        """
        select_idx_list = []
        select_num_idx_list = []
        # 计算T: 所有batch中Q方向切块的总数 (直接按照x基本块计算)
        total_q_blocks = 0
        max_kv_block_num = 0
        # 首先计算总的Q块数和最大KV块数
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            kv_seqlen = kv_seqlen_list[b]
            # Q方向分块计算: 直接按照x基本块计算
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            total_q_blocks += s_block_num_q
            max_kv_block_num = max(max_kv_block_num, s_block_num_kv)
        # 为每个Q块和每个头生成selectIdx
        q_block_offset = 0
        for b in range(batch):
            q_seqlen = q_seqlen_list[b]
            kv_seqlen = kv_seqlen_list[b]
            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y
            for t_local in range(s_block_num_q):
                t_global = q_block_offset + t_local
                q_block_idx = t_local  # 当前batch内的Q块索引
                batch_select_idx = []
                batch_select_num = []
                for head in range(num_heads):
                    # 为每个头生成稀疏的KV块选择（模拟实际稀疏注意力模式）
                    selected_kv_blocks = []
                    # 全部采用随机稀疏，随机选择若干KV块，不区分对角线/额外块
                    if s_block_num_kv > 0:
                        # 为不同头和Q块设置不同的随机种子，确保多样性
                        random.seed(head * 1000 + q_block_idx)
                        # 随机选择1~int(s_block_num_kv * select_ratio)个KV块
                        if select_ratio == 1:
                            select_ratio = np.random.random()
                        num_select = max(1, int(s_block_num_kv * select_ratio))
                        selected_kv_blocks = random.sample(
                            range(s_block_num_kv), num_select
                        )
                    else:
                        selected_kv_blocks = []
                    selected_kv_blocks.sort()
                    # 填充到max_kv_block_num长度
                    padded_blocks = selected_kv_blocks + [-1] * (
                        max_kv_block_num - len(selected_kv_blocks)
                    )
                    batch_select_idx.extend(padded_blocks)
                    batch_select_num.append(len(selected_kv_blocks))
                select_idx_list.extend(batch_select_idx)
                select_num_idx_list.extend(batch_select_num)

            q_block_offset += s_block_num_q

        return select_idx_list, select_num_idx_list, total_q_blocks, max_kv_block_num

    def init_by_input_data(self, input_data: InputDataset):
        np.random.seed(10)
        q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        block_shape = input_data.kwargs["blockShape"]
        blockSparseMask = input_data.kwargs["blockSparseMask"]
        batch = blockSparseMask.shape[0]
        head_num = blockSparseMask.shape[1]
        max_q_block_num = blockSparseMask.shape[2]
        max_kv_block_num = blockSparseMask.shape[3]
        blockSparseMask = torch.zeros(
            (batch, head_num, max_q_block_num, max_kv_block_num), dtype=torch.int8
        )
        for i in range(batch):
            for j in range(head_num):
                q_block_num = (q_seqlen_list[i] + block_shape[0] - 1) // block_shape[0]
                for k in range(q_block_num):
                    kv_block_num = (
                        kv_seqlen_list[i] + block_shape[1] - 1
                    ) // block_shape[1]
                    for l in range(kv_block_num):
                        blockSparseMask[i, j, k, l] = np.random.randint(0, 2)
        if self.device == "cpu":
            input_data.kwargs["blockSparseMask"] = blockSparseMask
        if self.device == "pyaclnn" or self.device == "npu":
            input_data.kwargs["blockSparseMask"] = blockSparseMask.npu()


# ----------------------------------------------------------------------
# BSA tiling-key 覆盖用例 golden（原独立文件 bsa_golden.py 内联）
# 覆盖 json id 200-283：api_type=aclnn_blocksparseattention_v3（CPU 参考）、
# aclnn_api_type=aclnn_blocksparseattentioninputprocess（NPU 打包，按 quantMode 分发）。
# 与上方 executor 同名的两个调度器类已加后缀 _tk，注册名不变。
# ----------------------------------------------------------------------
np.random.seed(1)
np.set_printoptions(threshold=np.inf, linewidth=500, suppress=True)

# 全局开关
DEBUG_nonquant = False
# 重新定义print
if not DEBUG_nonquant:
    print = lambda *args, **kwargs: None


def gen_block_sparse_pattern_nonquant(
    q_seqlen_list: any,
    kv_seqlen_list: any,
    block_shape: any,
    batch: int,
    num_heads: int,
    sparsity_ratio: float,
    is_varied_sparsity: int,
):
    q_s_aligned = max(q_seqlen_list)
    kv_s_aligned = max(kv_seqlen_list)
    s_block_x = block_shape[0]
    s_block_y = block_shape[1]
    q_s_block_num_aligned = (q_s_aligned + s_block_x - 1) // s_block_x
    kv_s_block_num_aligned = (kv_s_aligned + s_block_y - 1) // s_block_y

    block_sparse_mask = np.zeros(
        (batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int8
    )
    block_sparse_idx = np.ones(
        (batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned),
        dtype=np.int32,
    )
    # sparse idx 无效位置为-1，便于和idx=0做区分
    block_sparse_idx = -block_sparse_idx
    block_sparse_count = np.zeros(
        (batch, num_heads, q_s_block_num_aligned), dtype=np.int32
    )

    for b in range(batch):
        q_s_block_num_aval = (q_seqlen_list[b] + s_block_x - 1) // s_block_x
        kv_s_block_num_aval = (kv_seqlen_list[b] + s_block_y - 1) // s_block_y
        max_rsvd_kv_s_block_num = max(1, int(kv_s_block_num_aval * sparsity_ratio))
        for n1 in range(num_heads):
            for q_blk_id in range(q_s_block_num_aval):
                cur_row_sparse_mask = np.zeros(kv_s_block_num_aval, dtype=np.int8)
                # 当前rainfusion稀疏算法的原理：每个BN组合的所有X方向分块都采取同一个topK的方式选择Y方向的块
                # is_varied_sparsity是依据上述算法发散的进一步泛化场景：每个BN组合的每个X方向分块的topK的值也不相同
                rsvd_kv_s_block_num = max_rsvd_kv_s_block_num
                if is_varied_sparsity:
                    random.seed(
                        b * num_heads * q_s_block_num_aval
                        + n1 * q_s_block_num_aval
                        + q_blk_id
                    )
                    rsvd_kv_s_block_num = random.randint(1, max_rsvd_kv_s_block_num)
                # 依据随机出的保留块计数生成随机的稀疏pattern以及保留块下标
                cur_row_sparse_idx = sorted(
                    np.random.choice(
                        kv_s_block_num_aval, rsvd_kv_s_block_num, replace=False
                    )
                )
                for idx in cur_row_sparse_idx:
                    cur_row_sparse_mask[idx] = 1
                # 将生成的mask，idx，count填入对应位置
                block_sparse_mask[b, n1, q_blk_id, 0:kv_s_block_num_aval] = (
                    cur_row_sparse_mask
                )
                block_sparse_idx[b, n1, q_blk_id, 0:rsvd_kv_s_block_num] = (
                    cur_row_sparse_idx
                )
                block_sparse_count[b, n1, q_blk_id] = rsvd_kv_s_block_num
    return block_sparse_mask, block_sparse_idx, block_sparse_count


class TestBlockSparseAttentionGenBmGt_nonquant:
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        embedding_size: int
        dtype: any
        inner_prec: int
        scale: float
        q_layout: str
        kv_layout: str

    @dataclass
    class AttentionInputs:
        query: any
        key: any
        value: any
        block_sparse_idx: any
        block_sparse_count: any
        block_sparse_shape: any
        q_seqlen_list: any
        kv_seqlen_list: any
        aux_attrs: any

    def base_tile_mm(self, left, right, mm_k_tile):
        res = None
        # 此处与kernel代码中l0 buffer上的切分相同
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (
                (left.shape[1] - mm1_k_loop_idx * mm_k_tile)
                if (mm1_k_loop_idx == mm_k_loop - 1)
                else mm_k_tile
            )
            left_slice = left[
                :, mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k
            ]
            right_slice = right[
                mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k, :
            ]
            res_slice = np.matmul(left_slice, right_slice).astype(np.float32)
            res = res_slice if res is None else (res + res_slice)
        return res

    def online_softmax(
        self, qk_tile_res, gm, is_first_tile, interm_dtype_sm=np.float32
    ):
        sim = qk_tile_res.astype(interm_dtype_sm)
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first_tile:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        p = np.exp(sim_sub.astype(interm_dtype_sm))
        ll = np.sum(p, axis=-1, keepdims=True)
        return p, ll, dm, gm, lm

    def rescale_o(self, lo, ll, dm, go, gl, is_first_tile, interm_dtype_re=np.float32):
        if is_first_tile:
            gl = ll
            go = lo
        else:
            dm = np.exp(dm)
            gl = gl * dm
            gl = gl + ll
            go = go * dm.astype(interm_dtype_re)
            go = go + lo
        return go, gl.astype(interm_dtype_re)

    def ref_flash_block_sparse_attention(
        self, query, key, value, softmax_scale, inner_prec, attn_mask=None
    ):
        # key: [D, S2]
        # value: [S2, D]
        cur_kv_s_gathered = key.shape[1]
        kv_s_base_tile = 256 if inner_prec == 4 else 128
        interm_dtype_sm = np.float32 if inner_prec == 0 else query.dtype
        interm_dtype_re = query.dtype if inner_prec == 1 else np.float32
        softmax_scale = (
            np.asarray(softmax_scale, dtype=query.dtype)
            if inner_prec == 4
            else np.asarray(softmax_scale, dtype=np.float32)
        )
        gl = None
        go = None
        for kv_s_start in range(0, cur_kv_s_gathered, kv_s_base_tile):
            cur_kv_s_tile = (
                (cur_kv_s_gathered - kv_s_start)
                if (kv_s_start + kv_s_base_tile > cur_kv_s_gathered)
                else kv_s_base_tile
            )
            key_tile = key[:, kv_s_start : kv_s_start + cur_kv_s_tile]
            value_tile = value[kv_s_start : kv_s_start + cur_kv_s_tile, :]
            # q*k^t
            mm1_k_tile = 128
            qk_tile_res = self.base_tile_mm(query, key_tile, mm1_k_tile).astype(
                interm_dtype_sm
            )
            qk_tile_res = qk_tile_res * softmax_scale
            # online softmax
            if kv_s_start == 0:
                gm = None
            p, ll, dm, gm, lm = self.online_softmax(
                qk_tile_res, gm, kv_s_start == 0, interm_dtype_sm
            )
            p = p.astype(query.dtype)
            # p*v
            mm2_k_tile = 128
            lo = self.base_tile_mm(p, value_tile, mm2_k_tile).astype(interm_dtype_re)
            # rescale O
            go, gl = self.rescale_o(
                lo, ll, dm, go, gl, kv_s_start == 0, interm_dtype_re
            )
        go = go / gl
        go = go.astype(query.dtype)  # [S1,D]
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32)  # [S1]
        return go, lse

    def ref_attention(
        self, query, key, value, softmax_scale, inner_prec, attn_mask=None
    ):
        s = np.matmul(query.astype(np.float32), key.astype(np.float32)).astype(
            np.float32
        )
        s = s * np.float32(softmax_scale)
        row_max = np.max(s, axis=-1, keepdims=True)
        s_sub = s - row_max
        s_sub = np.exp(s_sub)
        row_sum = np.sum(s_sub, axis=-1, keepdims=True)
        p = s_sub / row_sum
        # Independent double-precision LSE truth for ATK double benchmark.
        # FP32 log(sum(exp())) can lose accuracy near zero; using the same
        # FP32 path as the benchmark can also make its error spuriously zero.
        s_lse = np.matmul(query.astype(np.float64), key.astype(np.float64))
        s_lse = s_lse * np.float64(softmax_scale)
        lse = np.logaddexp.reduce(s_lse, axis=-1)
        o = np.matmul(p.astype(np.float32), value.astype(np.float32)).astype(np.float32)
        return o, lse

    def compute_output(self, attention_inputs: AttentionInputs, is_benchmark):
        attn_out_bm = None
        attn_out_gt = None
        lse_bm = None
        lse_gt = None
        if not is_benchmark:
            attn_out_bm = np.zeros_like(attention_inputs.query)
        else:
            attn_out_gt = np.zeros_like(attention_inputs.query).astype(np.float32)

        input_layout = attention_inputs.aux_attrs.q_layout
        if input_layout == "TND":
            lse_shape = (
                attention_inputs.query.shape[0],
                attention_inputs.query.shape[1],
                1,
            )
        else:
            lse_shape = (*attention_inputs.query.shape[:3], 1)
        if not is_benchmark:
            lse_bm = np.zeros(lse_shape, dtype=np.float32)
        else:
            lse_gt = np.zeros(lse_shape, dtype=np.float32)

        query = attention_inputs.query
        key = attention_inputs.key
        value = attention_inputs.value
        block_sparse_idx = attention_inputs.block_sparse_idx
        block_sparse_count = attention_inputs.block_sparse_count
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        softmax_scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        group_size = int(num_heads / kv_heads)

        q_s_aligned = max(q_seqlen_list)
        kv_s_aligned = max(kv_seqlen_list)
        q_s_block_num_aligned = (
            q_s_aligned + block_sparse_shape[0] - 1
        ) // block_sparse_shape[0]
        kv_s_block_num_aligned = (
            kv_s_aligned + block_sparse_shape[1] - 1
        ) // block_sparse_shape[1]
        for b in range(batch):
            q_s_block_num_aval = (
                q_seqlen_list[b] + block_sparse_shape[0] - 1
            ) // block_sparse_shape[0]
            kv_s_block_num_aval = (
                kv_seqlen_list[b] + block_sparse_shape[1] - 1
            ) // block_sparse_shape[1]
            for n1 in range(num_heads):
                n2 = int(n1 // group_size)
                for q_s_blk_idx in range(q_s_block_num_aval):
                    # 获取当前q的稀疏分块
                    cur_q_s_blk_size = (
                        (q_seqlen_list[b] - q_s_blk_idx * block_sparse_shape[0])
                        if q_s_blk_idx == (q_s_block_num_aval - 1)
                        else block_sparse_shape[0]
                    )
                    q_blk = None
                    if q_layout == "TND":
                        t_offset = (
                            sum(q_seqlen_list[:b]) + q_s_blk_idx * block_sparse_shape[0]
                        )
                        q_blk = query[
                            t_offset : t_offset + cur_q_s_blk_size, n1, :
                        ]  # numpy默认将整数索引降维
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        q_blk = query[
                            b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                        ]  # numpy默认将整数索引降维
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        q_blk = query[
                            b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                        ]  # numpy默认将整数索引降维
                    # 获取这个q分块对应的kv分块的稀疏pattern
                    cur_kv_s_rsvd_blk_count = block_sparse_count[b, n1, q_s_blk_idx]
                    cur_kv_s_sparse_idx = (
                        block_sparse_idx[b, n1, q_s_blk_idx, :cur_kv_s_rsvd_blk_count]
                        .reshape(cur_kv_s_rsvd_blk_count)
                        .tolist()
                    )
                    # 获取将所有保留的kv块拼接起来的结果
                    key_gathered = None
                    value_gathered = None
                    count = 0
                    for kv_s_blk_idx in cur_kv_s_sparse_idx:
                        cur_kv_s_blk_size = (
                            (kv_seqlen_list[b] - kv_s_blk_idx * block_sparse_shape[1])
                            if kv_s_blk_idx == (kv_s_block_num_aval - 1)
                            else block_sparse_shape[1]
                        )
                        cur_k_blk = None
                        cur_v_blk = None
                        if kv_layout == "TND":
                            t_offset = (
                                sum(kv_seqlen_list[:b])
                                + kv_s_blk_idx * block_sparse_shape[1]
                            )
                            cur_k_blk = key[
                                t_offset : t_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_v_blk = value[
                                t_offset : t_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))
                        elif kv_layout == "BNSD":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[
                                b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                            ]
                            cur_v_blk = value[
                                b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                            ]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))
                        elif kv_layout == "BSND":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[
                                b, s2_offset : s2_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_v_blk = value[
                                b, s2_offset : s2_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_k_blk = np.transpose(cur_k_blk, (1, 0))

                        if count == 0:
                            key_gathered = cur_k_blk
                            value_gathered = cur_v_blk
                        else:
                            key_gathered = np.concatenate(
                                [key_gathered, cur_k_blk], axis=1
                            )
                            value_gathered = np.concatenate(
                                [value_gathered, cur_v_blk], axis=0
                            )
                        count += 1
                    if not is_benchmark:
                        attn_out_bm_slice, lse_bm_slice = (
                            self.ref_flash_block_sparse_attention(
                                q_blk,
                                key_gathered,
                                value_gathered,
                                softmax_scale,
                                inner_prec,
                            )
                        )
                    else:
                        attn_out_gt_slice, lse_gt_slice = self.ref_attention(
                            q_blk,
                            key_gathered,
                            value_gathered,
                            softmax_scale,
                            inner_prec,
                        )

                    if q_layout == "TND":
                        t_offset = (
                            sum(q_seqlen_list[:b]) + q_s_blk_idx * block_sparse_shape[0]
                        )
                        if not is_benchmark:
                            attn_out_bm[
                                t_offset : t_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_bm_slice
                            lse_bm[t_offset : t_offset + cur_q_s_blk_size, n1, :] = (
                                lse_bm_slice.reshape(-1, 1)
                            )
                        else:
                            attn_out_gt[
                                t_offset : t_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_gt_slice
                            lse_gt[t_offset : t_offset + cur_q_s_blk_size, n1, :] = (
                                lse_gt_slice.reshape(-1, 1)
                            )
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        if not is_benchmark:
                            attn_out_bm[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = attn_out_bm_slice
                            lse_bm[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = lse_bm_slice.reshape(-1, 1)
                        else:
                            attn_out_gt[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = attn_out_gt_slice
                            lse_gt[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = lse_gt_slice.reshape(-1, 1)
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_sparse_shape[0]
                        if not is_benchmark:
                            attn_out_bm[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_bm_slice
                            lse_bm[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = lse_bm_slice.reshape(-1, 1)
                        else:
                            attn_out_gt[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_gt_slice
                            lse_gt[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = lse_gt_slice.reshape(-1, 1)

        return attn_out_bm, attn_out_gt, lse_bm, lse_gt


def tensor_to_numpy_array_nonquant(t):
    if t is None:
        return None
    if t.dtype is torch.bfloat16:
        return t.to(torch.float32).numpy().astype(bfloat16)
    else:
        return t.numpy()


def numpy_array_to_tensor_nonquant(arr):
    if arr is None:
        return None
    # 如果是 list，先转为 numpy array
    if isinstance(arr, list):
        arr = np.array(arr)
    # 如果是 bfloat16，先转 float32 再转 torch
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    # 其他类型直接转换
    return torch.from_numpy(arr)


@register("bsa_nonquant_inputprocess_reference")
class BlockSparseAttentionInputProcess_nonquant(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(BlockSparseAttentionInputProcess_nonquant, self).__init__(
            task_result, backend
        )

    def init_by_input_data(self, input_data):
        np.random.seed(10)
        torch.npu.synchronize()
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型

        # 将tensor重新to到npu上，防止流同步失败
        query = self.acl_tensor_to_torch(input_args[0]).to("npu")
        input_args[0] = self.torch_tensor_to_acl(query)
        key = self.acl_tensor_to_torch(input_args[1]).to("npu")
        input_args[1] = self.torch_tensor_to_acl(key)
        value = self.acl_tensor_to_torch(input_args[2]).to("npu")
        input_args[2] = self.torch_tensor_to_acl(value)
        blockSparseMask = self.acl_tensor_to_torch(input_args[3]).to("npu")
        input_args[3] = self.torch_tensor_to_acl(blockSparseMask)

        self.input_layout = input_data.kwargs["qInputLayout"]

        self.qSeqlenList = input_data.kwargs["actualSeqLengths"]
        if self.qSeqlenList is None:
            self.actseq_aval = 0
        else:
            self.actseq_aval = 1

        output_packages = []
        input_args[4] = null_tensor_ptr  # attenMask

        input_args[8] = null_tensor_ptr  # blockTable
        input_args[20] = null_tensor_ptr
        if input_data.kwargs["softmaxLseFlag"] == 1:
            input_args.pop()
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            input_args.pop()
            output_packages.append(input_args[-2])
        return input_args, output_packages

    def __call__(self):
        """算子调用逻辑"""
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(
                dtype=torch.float32
            )
            output.append(temp_output_pack)

        if output[0].dim() == 4:
            outputTemp = torch.zeros(
                (
                    output[0].shape[0],
                    output[0].shape[1],
                    output[0].shape[2],
                    output[0].shape[3],
                ),
                dtype=output[0].dtype,
            )
            outputTensor = output[0]
            if self.qSeqlenList is not None:
                for i in range(output[0].shape[0]):
                    if self.input_layout == "BNSD":
                        outputTemp[i, :, : self.qSeqlenList[i], :] = outputTensor[
                            i, :, : self.qSeqlenList[i], :
                        ]
                    elif self.input_layout == "BSND":
                        outputTemp[i, : self.qSeqlenList[i], :, :] = outputTensor[
                            i, : self.qSeqlenList[i], :, :
                        ]
                return [outputTemp]
        else:
            return output[0]

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMask, const aclTensor *attenMaskOptional, const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)"


@register("aclnn_blocksparseattention")
class BlockSparseAttentionApi_nonquant(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionApi_nonquant, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query_golden = tensor_to_numpy_array_nonquant(self.query.cpu())
        key_golden = tensor_to_numpy_array_nonquant(self.key.cpu())
        value_golden = tensor_to_numpy_array_nonquant(self.value.cpu())

        testObj = TestBlockSparseAttentionGenBmGt_nonquant()
        aux_attrs = testObj.AuxAttrs(
            self.num_heads,
            self.kv_heads,
            self.headDim,
            self.query_dtype,
            self.inner_precise,
            self.scale_value,
            self.q_input_layout,
            self.kv_input_layout,
        )
        attention_inputs = testObj.AttentionInputs(
            query_golden,
            key_golden,
            value_golden,
            self.block_sparse_idx,
            self.block_sparse_count,
            self.block_shape,
            self.q_seqlen_list,
            self.kv_seqlen_list,
            aux_attrs,
        )
        attn_out_bm_np, attn_out_gt_np, lse_bm_np, lse_gt_np = testObj.compute_output(
            attention_inputs, self.task_result.is_benchmark_task
        )
        attn_out_bm = numpy_array_to_tensor_nonquant(attn_out_bm_np)
        attn_out_gt = numpy_array_to_tensor_nonquant(attn_out_gt_np)
        lse_bm = numpy_array_to_tensor_nonquant(lse_bm_np)
        lse_gt = numpy_array_to_tensor_nonquant(lse_gt_np)
        # attn_out_bm = torch.empty_like(numpy_array_to_tensor_nonquant(attention_inputs.query))
        # attn_out_gt = torch.empty_like(numpy_array_to_tensor_nonquant(attention_inputs.query)).to(torch.float32)
        if not self.task_result.is_benchmark_task:
            # 标杆
            if self.softmax_lse_flag:
                return attn_out_bm.to(torch.float32), lse_bm.to(torch.float32)
            return attn_out_bm.to(torch.float32)
        else:
            # 真值
            if self.softmax_lse_flag:
                return attn_out_gt.to(torch.float32), lse_gt.to(torch.float32)
            return attn_out_gt.to(torch.float32)

    def init_by_input_data(self, input_data: InputDataset):
        self.query = input_data.kwargs["query"]
        self.query_dtype = self.query.dtype
        self.key = input_data.kwargs["key"]
        self.value = input_data.kwargs["value"]
        self.block_shape = input_data.kwargs["blockShape"]
        self.q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        self.kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        self.kv_heads = input_data.kwargs["numKeyValueHeads"]
        self.scale_value = input_data.kwargs["scaleValue"]
        self.inner_precise = input_data.kwargs["innerPrecise"]
        self.softmax_lse_flag = input_data.kwargs["softmaxLseFlag"]
        # 根据可控的稀疏率重新生成sparse pattern，此处稀疏率选取为(0, 1)
        random.seed(10)
        sparsity_ratio = random.uniform(0.0, 1.0)
        np.random.seed(10)
        # sparsity_ratio = 1
        is_varied_sparsity = 0

        if self.q_input_layout == "TND":
            self.batch = len(self.q_seqlen_list)
            self.num_heads = self.query.shape[1]
            self.headDim = self.query.shape[2]
        elif self.q_input_layout == "BNSD":
            self.batch = self.query.shape[0]
            self.num_heads = self.query.shape[1]
            self.headDim = self.query.shape[3]
        elif self.q_input_layout == "BSND":
            self.batch = self.query.shape[0]
            self.num_heads = self.query.shape[2]
            self.headDim = self.query.shape[3]

        block_sparse_mask_np, self.block_sparse_idx, self.block_sparse_count = (
            gen_block_sparse_pattern_nonquant(
                self.q_seqlen_list,
                self.kv_seqlen_list,
                self.block_shape,
                self.batch,
                self.num_heads,
                sparsity_ratio,
                is_varied_sparsity,
            )
        )

        self.block_sparse_mask = numpy_array_to_tensor_nonquant(block_sparse_mask_np)
        if self.device == "cpu":
            input_data.kwargs["blockSparseMask"] = self.block_sparse_mask
        if self.device == "pyaclnn" or self.device == "npu":
            input_data.kwargs["blockSparseMask"] = self.block_sparse_mask.npu()


# ----------------------------------------------------------------------
# FP8 全量化参考实现（原 bsa_golden_fullquant_fp8.py）
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


# ATK acl wrapper may miss FP8; ACL_FLOAT8_E4M3FN=36, E5M2=35
TORCH_TO_ACLTYPE.setdefault("torch.float8_e4m3fn", 36)
TORCH_TO_ACLTYPE.setdefault("torch.float8_e5m2", 35)

random.seed(1)  # Python 内置 random
np.random.seed(1)  # NumPy 随机数
# torch.manual_seed(1) # PyTorch 随机数（如果其他地方用到）
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_printoptions(threshold=float("inf"))

# 全局开关
DEBUG_fp8 = False

# FP32 -> FP8_E4M3FN conversion constants.  These implement the same RNE
# (round-to-nearest, ties-to-even) bit path used by DAV_3510 vcvt ROUND_R.
_E4M3_FP8_MAX_BITS_fp8 = np.uint32(1087 << 20)  # 480.0f: first non-finite E4M3FN code
_E4M3_DENORM_MASK_fp8 = np.uint32(141 << 23)
_E4M3_DENORM_THRESH_fp8 = np.uint32(121 << 23)  # 2**-6: minimum normal value
_E4M3_BIAS_ROUND_fp8 = np.uint32((((7 - 127) << 23) + 0x7FFFF) & 0xFFFFFFFF)
_E4M3_SIGN_fp8 = np.uint32(0x80000000)

# 重新定义print
if not DEBUG_fp8:
    print = lambda *args, **kwargs: None


def gen_block_sparse_pattern_fp8(
    q_seqlen_list: any,
    kv_seqlen_list: any,
    block_shape: any,
    batch: int,
    num_heads: int,
    sparsity_ratio: float,
    is_varied_sparsity: int,
):
    q_s_aligned = max(q_seqlen_list)
    kv_s_aligned = max(kv_seqlen_list)
    s_block_x = block_shape[0]
    s_block_y = block_shape[1]
    q_s_block_num_aligned = (q_s_aligned + s_block_x - 1) // s_block_x
    kv_s_block_num_aligned = (kv_s_aligned + s_block_y - 1) // s_block_y

    block_sparse_mask = np.zeros(
        (batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int8
    )
    block_sparse_idx = np.ones(
        (batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned),
        dtype=np.int32,
    )
    # sparse idx 无效位置为-1，便于和idx=0做区分
    block_sparse_idx = -block_sparse_idx
    block_sparse_count = np.zeros(
        (batch, num_heads, q_s_block_num_aligned), dtype=np.int32
    )

    for b in range(batch):
        q_s_block_num_aval = (q_seqlen_list[b] + s_block_x - 1) // s_block_x
        kv_s_block_num_aval = (kv_seqlen_list[b] + s_block_y - 1) // s_block_y
        max_rsvd_kv_s_block_num = max(1, int(kv_s_block_num_aval * sparsity_ratio))
        for n1 in range(num_heads):
            for q_blk_id in range(q_s_block_num_aval):
                cur_row_sparse_mask = np.zeros(kv_s_block_num_aval, dtype=np.int8)
                # 当前rainfusion稀疏算法的原理：每个BN组合的所有X方向分块都采取同一个topK的方式选择Y方向的块
                # is_varied_sparsity是依据上述算法发散的进一步泛化场景：每个BN组合的每个X方向分块的topK的值也不相同
                rsvd_kv_s_block_num = max_rsvd_kv_s_block_num
                if is_varied_sparsity:
                    random.seed(
                        b * num_heads * q_s_block_num_aval
                        + n1 * q_s_block_num_aval
                        + q_blk_id
                    )
                    rsvd_kv_s_block_num = random.randint(1, max_rsvd_kv_s_block_num)
                # 依据随机出的保留块计数生成随机的稀疏pattern以及保留块下标
                cur_row_sparse_idx = sorted(
                    np.random.choice(
                        kv_s_block_num_aval, rsvd_kv_s_block_num, replace=False
                    )
                )
                for idx in cur_row_sparse_idx:
                    cur_row_sparse_mask[idx] = 1
                # 将生成的mask，idx，count填入对应位置
                block_sparse_mask[b, n1, q_blk_id, 0:kv_s_block_num_aval] = (
                    cur_row_sparse_mask
                )
                block_sparse_idx[b, n1, q_blk_id, 0:rsvd_kv_s_block_num] = (
                    cur_row_sparse_idx
                )
                block_sparse_count[b, n1, q_blk_id] = rsvd_kv_s_block_num
    return block_sparse_mask, block_sparse_idx, block_sparse_count


class TestBlockSparseAttentionGenBmGt_fp8:
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        embedding_size: int
        dtype: any
        inner_prec: int
        scale: float
        q_layout: str
        kv_layout: str
        attention_out_dtype: any

    @dataclass
    class AttentionInputs:
        query: any
        key: any
        value: any
        block_sparse_idx: any
        block_sparse_count: any
        block_sparse_shape: any
        q_seqlen_list: any
        kv_seqlen_list: any
        q_dequant_scale: any
        k_dequant_scale: any
        v_dequant_scale: any
        aux_attrs: any

    def fp32_to_fp8_RINT(
        self, tensor: np.ndarray, round_mode: str = "rint"
    ) -> np.ndarray:
        """Emulate FP32 -> FP8_E4M3FN CAST_RINT + SAT, then decode to FP32."""
        original_shape = np.shape(tensor)
        flat = np.ascontiguousarray(tensor, dtype=np.float32).reshape(-1)
        bits = self._float32_to_fp8_e4m3_vec(flat, round_mode)
        result = self._fp8_e4m3_to_float32_vec(bits)
        return result.reshape(original_shape)

    def _float32_to_fp8_e4m3_vec(
        self, arr: np.ndarray, round_mode: str = "rint"
    ) -> np.ndarray:
        """Encode FP32 as E4M3FN using CANN CAST_RINT (RNE) and SAT semantics."""
        if round_mode != "rint":
            raise ValueError(f"unsupported FP8 cast round mode: {round_mode}")

        # CANN's SatMode::SAT clamps finite overflow to +/-448 before ROUND_R.
        value = np.ascontiguousarray(arr, dtype=np.float32)
        value = np.clip(value, np.float32(-448.0), np.float32(448.0))
        value_bits = value.view(np.uint32)
        sign = value_bits & _E4M3_SIGN_fp8
        magnitude_bits = value_bits ^ sign

        # For subnormals, adding 2**14 shifts the E4M3 mantissa into integer
        # bits and the subtraction restores the E4M3 exponent encoding.  For
        # normal values, the bias plus the retained-mantissa LSB implements
        # round-to-nearest with ties going to an even mantissa.
        overflow = magnitude_bits >= _E4M3_FP8_MAX_BITS_fp8
        denormal = magnitude_bits < _E4M3_DENORM_THRESH_fp8
        denormal_value = magnitude_bits.view(np.float32) + np.float32(16384.0)
        denormal_result = denormal_value.view(np.uint32) - _E4M3_DENORM_MASK_fp8
        retained_mantissa_odd = (magnitude_bits >> np.uint32(20)) & np.uint32(1)
        normal_result = (
            magnitude_bits + _E4M3_BIAS_ROUND_fp8 + retained_mantissa_odd
        ) >> np.uint32(20)

        result = np.where(
            overflow,
            np.uint32(0x7F),
            np.where(denormal, denormal_result, normal_result),
        )
        result = (result | (sign >> np.uint32(24))) & np.uint32(0xFF)
        return result.astype(np.uint8, copy=False)

    def _fp8_e4m3_to_float32_vec(self, bits: np.ndarray) -> np.ndarray:
        """Decode E4M3FN bit patterns to FP32 values."""
        encoded = np.asarray(bits, dtype=np.uint8)
        sign = (encoded >> np.uint8(7)) & np.uint8(1)
        exponent = (encoded >> np.uint8(3)) & np.uint8(0xF)
        mantissa = encoded & np.uint8(0x7)

        is_nan = (exponent == 15) & (mantissa == 7)
        is_zero = (exponent == 0) & (mantissa == 0)
        is_denormal = (exponent == 0) & (mantissa != 0)
        exponent_fp32 = exponent.astype(np.float32)
        mantissa_fp32 = mantissa.astype(np.float32)
        result = np.where(
            is_zero,
            np.float32(0.0),
            np.where(
                is_denormal,
                mantissa_fp32 * np.float32(2.0**-9),
                (np.float32(1.0) + mantissa_fp32 * np.float32(0.125))
                * np.exp2(exponent_fp32 - np.float32(7.0)),
            ),
        )
        result = np.where(sign.astype(bool), -result, result)
        return np.where(is_nan, np.float32("nan"), result).astype(
            np.float32, copy=False
        )

    @staticmethod
    def fp32_to_my19bit(tensor: np.ndarray):
        """Truncate an FP32 scalar to the 19-bit precision used by Fixpipe."""
        bits = np.asarray(tensor, dtype=np.float32).reshape(()).view(np.uint32).item()
        result = bits & 0xFFFFE000
        return np.array(result, dtype=np.uint32).view(np.float32)

    def base_tile_mm(self, left, right, mm_k_tile):
        res = None
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (
                (left.shape[1] - mm1_k_loop_idx * mm_k_tile)
                if (mm1_k_loop_idx == mm_k_loop - 1)
                else mm_k_tile
            )
            left_slice = left[
                :, mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k
            ]
            right_slice = right[
                mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k, :
            ]
            res_slice = np.matmul(left_slice, right_slice).astype(np.float32)
            res = res_slice if res is None else (res + res_slice)
        return res

    def base_tile_mm_particular(self, left, right, mm_k_tile, dtype):
        res = None
        mm_k_loop = (left.shape[1] + mm_k_tile - 1) // mm_k_tile
        for mm1_k_loop_idx in range(mm_k_loop):
            sub_k = (
                (left.shape[1] - mm1_k_loop_idx * mm_k_tile)
                if (mm1_k_loop_idx == mm_k_loop - 1)
                else mm_k_tile
            )
            left_slice = left[
                :, mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k
            ].astype(np.float32)
            right_slice = right[
                mm1_k_loop_idx * mm_k_tile : mm1_k_loop_idx * mm_k_tile + sub_k, :
            ].astype(np.float32)
            res_slice = np.matmul(left_slice, right_slice)
            res = res_slice if res is None else (res + res_slice)
        return res

    def online_softmax(
        self, qk_tile_res, gm, is_first_tile, interm_dtype_sm=np.float32
    ):
        sim = qk_tile_res.astype(interm_dtype_sm)
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first_tile:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        p = np.exp(sim_sub.astype(interm_dtype_sm))
        ll = np.sum(p, axis=-1, keepdims=True)
        return p, ll, dm, gm, lm

    def rescale_o(self, lo, ll, dm, go, gl, is_first_tile, interm_dtype_re=np.float32):
        if is_first_tile:
            gl = ll
            go = lo
        else:
            dm = np.exp(dm)
            gl = gl * dm
            gl = gl + ll
            go = go * dm.astype(interm_dtype_re)
            go = go + lo
        return go, gl.astype(interm_dtype_re)

    @staticmethod
    def _largest_divisor_at_most(value, max_bound):
        value = int(value)
        max_bound = int(max_bound)
        if value == 0 or max_bound == 0:
            return 1
        if max_bound >= value:
            return value
        result = 1
        d = 1
        while d * d <= value:
            if value % d == 0:
                if d <= max_bound:
                    result = max(result, d)
                paired = value // d
                if paired <= max_bound:
                    result = max(result, paired)
            d += 1
        return result

    @staticmethod
    def calc_kv_base_tile_a5(inner_prec, block_shape_y, embedding_size):
        # Match host CalcFp8KvBaseTile
        preferred = 256 if (inner_prec == 4 and embedding_size <= 128) else 128
        y = int(block_shape_y)
        if y % preferred == 0:
            return preferred
        if preferred > 128 and y % 128 == 0:
            return 128
        if y > preferred:
            return TestBlockSparseAttentionGenBmGt_fp8._largest_divisor_at_most(
                y, preferred
            )
        return y

    def ref_flash_block_sparse_attention(
        self,
        query,
        key,
        value,
        softmax_scale,
        inner_prec,
        q_deq_scale,
        k_deq_scale_gathered,
        v_deq_scale_gathered,
        block_shape_y,
        attention_out_dtype,
        embedding_size=128,
        attn_mask=None,
    ):
        cur_kv_s_gathered = key.shape[1]
        kv_s_base_tile = self.calc_kv_base_tile_a5(
            inner_prec, block_shape_y, embedding_size
        )
        interm_dtype_sm = attention_out_dtype
        interm_dtype_re = query.dtype if inner_prec == 1 else np.float32
        softmax_scale = np.asarray(softmax_scale, dtype=query.dtype)
        gl = None
        go = None

        # loop_count = block_shape_y // kv_s_base_tile
        deq_scale_index = 0
        already_deq_scale_time = 0

        for kv_s_start in range(0, cur_kv_s_gathered, kv_s_base_tile):
            cur_kv_s_tile = (
                (cur_kv_s_gathered - kv_s_start)
                if (kv_s_start + kv_s_base_tile > cur_kv_s_gathered)
                else kv_s_base_tile
            )
            key_tile = key[:, kv_s_start : kv_s_start + cur_kv_s_tile]
            value_tile = value[kv_s_start : kv_s_start + cur_kv_s_tile, :]
            mm1_k_tile = 128

            qk_tile_res = self.base_tile_mm_particular(
                query, key_tile, mm1_k_tile, query.dtype
            )

            y_blk_idx = kv_s_start // block_shape_y
            combined_deq_scale = (
                q_deq_scale * k_deq_scale_gathered[y_blk_idx] * softmax_scale
            )
            combined_deq_scale_fixpipe = self.fp32_to_my19bit(combined_deq_scale)
            qk_tile_res = qk_tile_res * combined_deq_scale_fixpipe
            qk_tile_res = qk_tile_res.astype(interm_dtype_sm)

            if kv_s_start == 0:
                gm = None
            p, ll, dm, gm, lm = self.online_softmax(
                qk_tile_res, gm, kv_s_start == 0, interm_dtype_sm
            )
            p = p * 448
            p = self.fp32_to_fp8_RINT(p)
            p = p.astype(query.dtype)
            mm2_k_tile = 128
            lo = self.base_tile_mm_particular(p, value_tile, mm2_k_tile, np.float32)
            v_deq_scale_fixpipe = self.fp32_to_my19bit(v_deq_scale_gathered[y_blk_idx])
            lo = lo * v_deq_scale_fixpipe
            go, gl = self.rescale_o(
                lo, ll, dm, go, gl, kv_s_start == 0, interm_dtype_re
            )
        go = go / gl
        go = go * (1.0 / 448.0)
        go = go.astype(query.dtype)
        lse = np.squeeze(np.log(gl) + gm, axis=-1).astype(np.float32)
        return go, lse

    def ref_attention(
        self, query, key, value, softmax_scale, inner_prec, attn_mask=None
    ):
        s = np.matmul(query.astype(np.float32), key.astype(np.float32)).astype(
            np.float32
        )
        s = s * np.float32(softmax_scale)
        row_max = np.max(s, axis=-1, keepdims=True)
        s_sub = s - row_max
        s_sub = np.exp(s_sub)
        row_sum = np.sum(s_sub, axis=-1, keepdims=True)
        p = s_sub / row_sum
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)
        o = np.matmul(p.astype(np.float32), value.astype(np.float32)).astype(np.float32)
        return o, lse

    def compute_output(self, attention_inputs: AttentionInputs, is_benchmark):
        attn_out_bm = None
        attn_out_gt = None
        lse_bm = None
        lse_gt = None
        if not is_benchmark:
            attn_out_bm = np.zeros_like(attention_inputs.query)
        else:
            attn_out_gt = np.zeros_like(attention_inputs.query).astype(np.float32)

        input_layout = attention_inputs.aux_attrs.q_layout
        if input_layout == "TND":
            if not is_benchmark:
                lse_bm = np.zeros(
                    (
                        attention_inputs.query.shape[0],
                        attention_inputs.query.shape[1],
                        1,
                    ),
                    dtype=np.float32,
                )
            else:
                lse_gt = np.zeros(
                    (
                        attention_inputs.query.shape[0],
                        attention_inputs.query.shape[1],
                        1,
                    ),
                    dtype=np.float32,
                )
        elif input_layout == "BNSD" or input_layout == "BSND":
            if not is_benchmark:
                lse_bm = np.zeros(
                    (
                        attention_inputs.query.shape[0],
                        attention_inputs.query.shape[1],
                        attention_inputs.query.shape[2],
                        1,
                    ),
                    dtype=np.float32,
                )
            else:
                lse_gt = np.zeros(
                    (
                        attention_inputs.query.shape[0],
                        attention_inputs.query.shape[1],
                        attention_inputs.query.shape[2],
                        1,
                    ),
                    dtype=np.float32,
                )

        query = attention_inputs.query
        key = attention_inputs.key
        value = attention_inputs.value
        block_sparse_idx = attention_inputs.block_sparse_idx
        block_sparse_count = attention_inputs.block_sparse_count
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list
        q_dequant_scale = attention_inputs.q_dequant_scale
        k_dequant_scale = attention_inputs.k_dequant_scale
        v_dequant_scale = attention_inputs.v_dequant_scale

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        softmax_scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        group_size = int(num_heads / kv_heads)
        attention_out_dtype = attention_inputs.aux_attrs.attention_out_dtype
        deq_scale_index = 0
        already_deq_scale_time = 0
        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]
        # loop_count = block_shape_y // 256

        q_s_aligned = max(q_seqlen_list)
        kv_s_aligned = max(kv_seqlen_list)
        q_s_block_num_aligned = (q_s_aligned + block_shape_x - 1) // block_shape_x
        kv_s_block_num_aligned = (kv_s_aligned + block_shape_y - 1) // block_shape_y

        q_offsets_sum = [0]
        for i in range(len(q_seqlen_list) - 1):
            q_offsets_sum.append(q_offsets_sum[-1] + q_seqlen_list[i])

        kv_offsets_sum = [0]
        for i in range(len(kv_seqlen_list) - 1):
            kv_offsets_sum.append(kv_offsets_sum[-1] + kv_seqlen_list[i])

        for b in range(batch):
            q_s_block_num_aval = (q_seqlen_list[b] + block_shape_x - 1) // block_shape_x
            kv_s_block_num_aval = (
                kv_seqlen_list[b] + block_shape_y - 1
            ) // block_shape_y
            for n1 in range(num_heads):
                n2 = int(n1 // group_size)
                for q_s_blk_idx in range(q_s_block_num_aval):
                    cur_q_s_blk_size = (
                        (q_seqlen_list[b] - q_s_blk_idx * block_shape_x)
                        if q_s_blk_idx == (q_s_block_num_aval - 1)
                        else block_shape_x
                    )
                    q_blk = None
                    q_deq_scale = q_dequant_scale[b, n1, q_s_blk_idx]
                    if q_layout == "TND":
                        t_offset = q_offsets_sum[b] + q_s_blk_idx * block_shape_x
                        q_blk = query[t_offset : t_offset + cur_q_s_blk_size, n1, :]
                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_shape_x
                        q_blk = query[
                            b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                        ]
                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_shape_x
                        q_blk = query[
                            b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                        ]
                    q_blk_antiquant = q_blk.astype(np.float32) * q_deq_scale
                    cur_kv_s_rsvd_blk_count = block_sparse_count[b, n1, q_s_blk_idx]
                    # cur_kv_s_rsvd_blk_count = block_sparse_count[b, n1, q_s_blk_idx].item()
                    cur_kv_s_sparse_idx = (
                        block_sparse_idx[b, n1, q_s_blk_idx, :cur_kv_s_rsvd_blk_count]
                        .reshape(cur_kv_s_rsvd_blk_count)
                        .tolist()
                    )
                    key_gathered = None
                    value_gathered = None
                    k_deq_scale_gathered = None
                    v_deq_scale_gathered = None
                    key_gathered_antiquant = None
                    value_gathered_antiquant = None
                    key_parts = []
                    value_parts = []
                    k_scale_parts = []
                    v_scale_parts = []
                    key_gathered_antiquant_parts = []
                    value_gathered_antiquant_parts = []

                    for kv_s_blk_idx in cur_kv_s_sparse_idx:
                        cur_kv_s_blk_size = (
                            (kv_seqlen_list[b] - kv_s_blk_idx * block_sparse_shape[1])
                            if kv_s_blk_idx == (kv_s_block_num_aval - 1)
                            else block_sparse_shape[1]
                        )
                        cur_k_blk = None
                        cur_v_blk = None
                        cur_k_blk_antiquant = None
                        cur_v_blk_antiquant = None
                        cur_k_deq_scale_blk = k_dequant_scale[b, n2, kv_s_blk_idx]
                        cur_v_deq_scale_blk = v_dequant_scale[b, n2, kv_s_blk_idx]

                        if kv_layout == "TND":
                            t_offset = (
                                sum(kv_seqlen_list[:b])
                                + kv_s_blk_idx * block_sparse_shape[1]
                            )
                            cur_k_blk = key[
                                t_offset : t_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_v_blk = value[
                                t_offset : t_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_k_blk = cur_k_blk.transpose(1, 0)
                            cur_k_blk_antiquant = (
                                key[
                                    t_offset : t_offset + cur_kv_s_blk_size, n2, :
                                ].astype(np.float32)
                                * cur_k_deq_scale_blk
                            )
                            cur_v_blk_antiquant = (
                                value[
                                    t_offset : t_offset + cur_kv_s_blk_size, n2, :
                                ].astype(np.float32)
                                * cur_v_deq_scale_blk
                            )
                            cur_k_blk_antiquant = cur_k_blk_antiquant.transpose(1, 0)
                        elif kv_layout == "BNSD":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[
                                b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                            ]
                            cur_v_blk = value[
                                b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                            ]
                            cur_k_blk = cur_k_blk.transpose(1, 0)
                            cur_k_blk_antiquant = (
                                key[
                                    b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                                ].astype(np.float32)
                                * cur_k_deq_scale_blk
                            )
                            cur_v_blk_antiquant = (
                                value[
                                    b, n2, s2_offset : s2_offset + cur_kv_s_blk_size, :
                                ].astype(np.float32)
                                * cur_v_deq_scale_blk
                            )
                            cur_k_blk_antiquant = cur_k_blk_antiquant.transpose(1, 0)
                        elif kv_layout == "BSND":
                            s2_offset = kv_s_blk_idx * block_sparse_shape[1]
                            cur_k_blk = key[
                                b, s2_offset : s2_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_v_blk = value[
                                b, s2_offset : s2_offset + cur_kv_s_blk_size, n2, :
                            ]
                            cur_k_blk = cur_k_blk.transpose(1, 0)
                            cur_k_blk_antiquant = (
                                key[
                                    b, s2_offset : s2_offset + cur_kv_s_blk_size, n2, :
                                ].astype(np.float32)
                                * cur_k_deq_scale_blk
                            )
                            cur_v_blk_antiquant = (
                                value[
                                    b, s2_offset : s2_offset + cur_kv_s_blk_size, n2, :
                                ].astype(np.float32)
                                * cur_v_deq_scale_blk
                            )
                            cur_k_blk_antiquant = cur_k_blk_antiquant.transpose(1, 0)

                        # 收集到列表中（不立即拼接）
                        key_parts.append(cur_k_blk)
                        value_parts.append(cur_v_blk)
                        k_scale_parts.append(cur_k_deq_scale_blk)
                        v_scale_parts.append(cur_v_deq_scale_blk)
                        key_gathered_antiquant_parts.append(cur_k_blk_antiquant)
                        value_gathered_antiquant_parts.append(cur_v_blk_antiquant)

                    # 循环结束后，一次性拼接所有部分
                    if key_parts:
                        key_gathered = np.concatenate(key_parts, axis=1)
                        value_gathered = np.concatenate(value_parts, axis=0)
                        k_deq_scale_gathered = np.concatenate(k_scale_parts, axis=0)
                        v_deq_scale_gathered = np.concatenate(v_scale_parts, axis=0)
                        key_gathered_antiquant = np.concatenate(
                            key_gathered_antiquant_parts, axis=1
                        )
                        value_gathered_antiquant = np.concatenate(
                            value_gathered_antiquant_parts, axis=0
                        )
                    else:
                        key_gathered = None
                        value_gathered = None
                        k_deq_scale_gathered = None
                        v_deq_scale_gathered = None
                        key_gathered_antiquant = None
                        value_gathered_antiquant = None

                    if not is_benchmark:
                        attn_out_bm_slice, lse_bm_slice = (
                            self.ref_flash_block_sparse_attention(
                                q_blk,
                                key_gathered,
                                value_gathered,
                                softmax_scale,
                                inner_prec,
                                q_deq_scale,
                                k_deq_scale_gathered,
                                v_deq_scale_gathered,
                                block_shape_y,
                                attention_out_dtype,
                                embedding_size,
                            )
                        )
                    else:
                        q_blk_antiquant = q_blk_antiquant  # 已经是 numpy 数组
                        key_gathered_antiquant = (
                            key_gathered_antiquant  # 已经是 numpy 数组
                        )
                        value_gathered_antiquant = (
                            value_gathered_antiquant  # 已经是 numpy 数组
                        )

                        attn_out_gt_slice, lse_gt_slice = self.ref_attention(
                            q_blk_antiquant,
                            key_gathered_antiquant,
                            value_gathered_antiquant,
                            softmax_scale,
                            inner_prec,
                        )

                    # 根据 layout 和 is_benchmark 进行赋值
                    if q_layout == "TND":
                        t_offset = q_offsets_sum[b] + q_s_blk_idx * block_shape_x
                        if not is_benchmark:
                            # Benchmark 模式：写入 attn_out_bm
                            attn_out_bm[
                                t_offset : t_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_bm_slice
                            lse_bm[t_offset : t_offset + cur_q_s_blk_size, n1, :] = (
                                lse_bm_slice.reshape(-1, 1)
                            )
                        else:
                            # 非 Benchmark 模式：写入 attn_out_gt
                            attn_out_gt[
                                t_offset : t_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_gt_slice
                            lse_gt[t_offset : t_offset + cur_q_s_blk_size, n1, :] = (
                                lse_gt_slice.reshape(-1, 1)
                            )

                    elif q_layout == "BNSD":
                        s1_offset = q_s_blk_idx * block_shape_x
                        if not is_benchmark:
                            # Benchmark 模式：写入 attn_out_bm
                            attn_out_bm[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = attn_out_bm_slice
                            lse_bm[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = lse_bm_slice.reshape(-1, 1)
                        else:
                            # 非 Benchmark 模式：写入 attn_out_gt
                            attn_out_gt[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = attn_out_gt_slice
                            lse_gt[
                                b, n1, s1_offset : s1_offset + cur_q_s_blk_size, :
                            ] = lse_gt_slice.reshape(-1, 1)

                    elif q_layout == "BSND":
                        s1_offset = q_s_blk_idx * block_shape_x
                        if not is_benchmark:
                            # Benchmark 模式：写入 attn_out_bm
                            attn_out_bm[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_bm_slice
                            lse_bm[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = lse_bm_slice.reshape(-1, 1)
                        else:
                            # 非 Benchmark 模式：写入 attn_out_gt
                            attn_out_gt[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = attn_out_gt_slice
                            lse_gt[
                                b, s1_offset : s1_offset + cur_q_s_blk_size, n1, :
                            ] = lse_gt_slice.reshape(-1, 1)

        return attn_out_bm, attn_out_gt, lse_bm, lse_gt


def tensor_to_numpy_array_fp8(t):
    if t is None:
        return None
    if t.dtype is torch.bfloat16:
        return t.to(torch.float32).numpy().astype(bfloat16)
    else:
        return t.numpy()


def numpy_array_to_tensor_fp8(arr):
    if arr is None:
        return None
    # 如果是 list，先转为 numpy array
    if isinstance(arr, list):
        arr = np.array(arr)
    # 如果是 bfloat16，先转 float32 再转 torch
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    # 其他类型直接转换
    return torch.from_numpy(arr)


@register("aclnn_blocksparseattentioninputprocess")
class BlockSparseAttentionInputProcess_fp8(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(BlockSparseAttentionInputProcess_fp8, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data):
        torch.manual_seed(10)
        torch.npu.synchronize()
        input_args = []
        output_packages = []

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        appended_output_count = len(output_packages)
        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)

        query = self.acl_tensor_to_torch(input_args[0]).to("npu")
        input_args[0] = self.torch_tensor_to_acl(query)
        key = self.acl_tensor_to_torch(input_args[1]).to("npu")
        input_args[1] = self.torch_tensor_to_acl(key)
        value = self.acl_tensor_to_torch(input_args[2]).to("npu")
        input_args[2] = self.torch_tensor_to_acl(value)
        qDequantScale = self.acl_tensor_to_torch(input_args[9]).to("npu")
        input_args[9] = self.torch_tensor_to_acl(qDequantScale)
        kDequantScale = self.acl_tensor_to_torch(input_args[10]).to("npu")
        input_args[10] = self.torch_tensor_to_acl(kDequantScale)
        vDequantScale = self.acl_tensor_to_torch(input_args[11]).to("npu")
        input_args[11] = self.torch_tensor_to_acl(vDequantScale)

        self.input_layout = input_data.kwargs["qInputLayout"]
        self.softmax_lse_flag = input_data.kwargs["softmaxLseFlag"]

        self.qSeqlenList = input_data.kwargs["actualSeqLengths"]
        kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        block_shape = input_data.kwargs["blockShape"]
        query_host = input_data.kwargs["query"]
        if self.input_layout == "TND":
            batch = len(self.qSeqlenList)
            num_heads = int(query_host.shape[1])
        elif self.input_layout == "BNSD":
            batch = int(query_host.shape[0])
            num_heads = int(query_host.shape[1])
        else:
            batch = int(query_host.shape[0])
            num_heads = int(query_host.shape[2])
        random.seed(10)
        sparsity_ratio = random.uniform(0.0, 1.0)
        np.random.seed(10)
        mask_np, _, _ = gen_block_sparse_pattern_fp8(
            self.qSeqlenList,
            kv_seqlen_list,
            block_shape,
            batch,
            num_heads,
            sparsity_ratio,
            0,
        )
        input_args[3] = self.torch_tensor_to_acl(
            numpy_array_to_tensor_fp8(mask_np).npu()
        )
        if self.qSeqlenList is None:
            self.actseq_aval = 0
        else:
            self.actseq_aval = 1

        output_packages = []
        input_args[4] = null_tensor_ptr  # blockTable
        input_args[12] = null_tensor_ptr  # V3 pQuantScale

        # ATK 会把 output_info_list 转换出的输出追加到参数尾部。先移除
        # 这些副本，再使用 V3 固定的 25/26 位置作为输出参数。
        if appended_output_count:
            del input_args[-appended_output_count:]

        if input_data.kwargs["softmaxLseFlag"] == 1:
            output_packages.extend((input_args[25], input_args[26]))
        else:
            # V3 始终保留 softmaxLseOptional 形参；关闭 LSE 时传 typed nullptr。
            input_args[26] = null_tensor_ptr
            output_packages.append(input_args[25])
        return input_args, output_packages

    def __call__(self):
        """算子调用逻辑"""
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack).to(
                dtype=torch.float32
            )
            output.append(temp_output_pack)

        if self.input_layout == "BNSD":
            outputTemp = torch.zeros(
                (
                    output[0].shape[0],
                    output[0].shape[1],
                    output[0].shape[2],
                    output[0].shape[3],
                ),
                dtype=output[0].dtype,
            )
            outputTensor = output[0]
            if self.qSeqlenList is not None:
                for i in range(output[0].shape[0]):
                    outputTemp[i, :, : self.qSeqlenList[i], :] = outputTensor[
                        i, :, : self.qSeqlenList[i], :
                    ]
                if len(output) == 2:
                    lseTemp = torch.zeros(
                        (output[1].shape[0], output[1].shape[1], output[1].shape[2], 1),
                        dtype=output[1].dtype,
                    )
                    lseTensor = output[1]
                    for i in range(output[1].shape[0]):
                        lseTemp[i, :, : self.qSeqlenList[i], :] = lseTensor[
                            i, :, : self.qSeqlenList[i], :
                        ]
                    return [outputTemp, lseTemp]
                else:
                    return [outputTemp]
        elif self.input_layout == "BSND":
            outputTemp = torch.zeros(
                (
                    output[0].shape[0],
                    output[0].shape[1],
                    output[0].shape[2],
                    output[0].shape[3],
                ),
                dtype=output[0].dtype,
            )
            outputTensor = output[0]
            if self.qSeqlenList is not None:
                for i in range(output[0].shape[0]):
                    outputTemp[i, : self.qSeqlenList[i], :, :] = outputTensor[
                        i, : self.qSeqlenList[i], :, :
                    ]
                if len(output) == 2:
                    lseTemp = torch.zeros(
                        (output[1].shape[0], output[1].shape[1], output[1].shape[2], 1),
                        dtype=output[1].dtype,
                    )
                    lseTensor = output[1]
                    for i in range(output[1].shape[0]):
                        lseTemp[i, : self.qSeqlenList[i], :, :] = lseTensor[
                            i, : self.qSeqlenList[i], :, :
                        ]
                    return [outputTemp, lseTemp]
                else:
                    return [outputTemp]
        else:
            if len(output) == 2:
                return output[0], output[1]
            else:
                return output[0]

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *blockSparseMask, const aclTensor *attenMaskOptional, "
            "const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional, "
            "const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, "
            "const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, "
            "const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, "
            "char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, "
            "double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, "
            "int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode, double dstTypeMax, "
            "aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )


@register("aclnn_blocksparseattention_v3")
class BlockSparseAttentionApi_fp8(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionApi_fp8, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query_golden = self.query.cpu()
        key_golden = self.key.cpu()
        value_golden = self.value.cpu()
        query_golden = tensor_to_numpy_array_fp8(
            query_golden.to(self.attention_out_dtype)
        )
        key_golden = tensor_to_numpy_array_fp8(key_golden.to(self.attention_out_dtype))
        value_golden = tensor_to_numpy_array_fp8(
            value_golden.to(self.attention_out_dtype)
        )

        q_dequant_scale_golden = tensor_to_numpy_array_fp8(self.q_dequant_scale.cpu())
        k_dequant_scale_golden = tensor_to_numpy_array_fp8(self.k_dequant_scale.cpu())
        v_dequant_scale_golden = tensor_to_numpy_array_fp8(self.v_dequant_scale.cpu())

        testObj = TestBlockSparseAttentionGenBmGt_fp8()
        aux_attrs = testObj.AuxAttrs(
            self.num_heads,
            self.kv_heads,
            self.headDim,
            self.query_dtype,
            self.inner_precise,
            self.scale_value,
            self.q_input_layout,
            self.kv_input_layout,
            query_golden.dtype,
        )
        attention_inputs = testObj.AttentionInputs(
            query_golden,
            key_golden,
            value_golden,
            self.block_sparse_idx,
            self.block_sparse_count,
            self.block_shape,
            self.q_seqlen_list,
            self.kv_seqlen_list,
            q_dequant_scale_golden,
            k_dequant_scale_golden,
            v_dequant_scale_golden,
            aux_attrs,
        )
        attn_out_bm_np, attn_out_gt_np, lse_bm_np, lse_gt_np = testObj.compute_output(
            attention_inputs, self.task_result.is_benchmark_task
        )
        attn_out_bm = numpy_array_to_tensor_fp8(attn_out_bm_np)
        attn_out_gt = numpy_array_to_tensor_fp8(attn_out_gt_np)
        lse_bm = numpy_array_to_tensor_fp8(lse_bm_np)
        lse_gt = numpy_array_to_tensor_fp8(lse_gt_np)
        if not self.task_result.is_benchmark_task:
            if self.softmax_lse_flag:
                return attn_out_bm.to(torch.float32), lse_bm.to(torch.float32)
            else:
                return attn_out_bm.to(torch.float32)
        else:
            if self.softmax_lse_flag:
                return attn_out_gt.to(torch.float32), lse_gt.to(torch.float32)
            else:
                return attn_out_gt.to(torch.float32)

    def init_by_input_data(self, input_data: InputDataset):
        self.query = input_data.kwargs["query"]
        self.query_dtype = self.query.dtype
        self.key = input_data.kwargs["key"]
        self.value = input_data.kwargs["value"]
        self.block_shape = input_data.kwargs["blockShape"]
        self.q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        self.kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        self.kv_heads = input_data.kwargs["numKeyValueHeads"]
        self.scale_value = input_data.kwargs["scaleValue"]
        self.inner_precise = input_data.kwargs["innerPrecise"]
        self.q_dequant_scale = input_data.kwargs["qDequantScale"]
        self.k_dequant_scale = input_data.kwargs["kDequantScale"]
        self.v_dequant_scale = input_data.kwargs["vDequantScale"]
        self.attention_out_dtype = input_data.kwargs["attentionOut"].dtype
        self.softmax_lse_flag = input_data.kwargs["softmaxLseFlag"]
        # Keep Python + NumPy RNG in lockstep with InputProcess (mask uses np.random.choice).
        random.seed(10)
        sparsity_ratio = random.uniform(0.0, 1.0)
        np.random.seed(10)
        is_varied_sparsity = 0

        if self.q_input_layout == "TND":
            self.batch = len(self.q_seqlen_list)
            self.num_heads = self.query.shape[1]
            self.headDim = self.query.shape[2]
        elif self.q_input_layout == "BNSD":
            self.batch = self.query.shape[0]
            self.num_heads = self.query.shape[1]
            self.headDim = self.query.shape[3]
        elif self.q_input_layout == "BSND":
            self.batch = self.query.shape[0]
            self.num_heads = self.query.shape[2]
            self.headDim = self.query.shape[3]

        # self.block_sparse_mask = block_sparse_mask
        block_sparse_mask_np, self.block_sparse_idx, self.block_sparse_count = (
            gen_block_sparse_pattern_fp8(
                self.q_seqlen_list,
                self.kv_seqlen_list,
                self.block_shape,
                self.batch,
                self.num_heads,
                sparsity_ratio,
                is_varied_sparsity,
            )
        )

        self.block_sparse_mask = numpy_array_to_tensor_fp8(block_sparse_mask_np)

        if self.device == "cpu":
            input_data.kwargs["blockSparseMask"] = self.block_sparse_mask
        if self.device == "pyaclnn" or self.device == "npu":
            input_data.kwargs["blockSparseMask"] = self.block_sparse_mask.npu()


# ----------------------------------------------------------------------
# MXFP4 参考实现（原 bsa_golden_fullquant_mxfp4.py）
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


_logging = Logger().get_logger()

_FLOAT4_TORCH_STORAGE_ATTRS = ("float4_e2m1fn_x2", "float4_e2m1fn", "float4_e2m1")
_FLOAT4_NPU_STORAGE_ATTRS = (
    "float4_e2m1fn_x2",
    "float4_e2m1fn",
    "float4_e2m1fnu",
    "float4_e2m1",
)
_NOT_LIMITED = 2147483647


def ensure_atk_float8_e8m0_support():
    if not hasattr(AclDataType, "ACL_FLOAT8_E8M0"):
        return
    key = "torch.float8_e8m0fnu"
    if key not in TORCH_TO_ACLTYPE:
        TORCH_TO_ACLTYPE[key] = AclDataType.ACL_FLOAT8_E8M0
        _logging.info(f"ATK dtype map: register {key} -> ACL_FLOAT8_E8M0")
    if AclDataType.ACL_FLOAT8_E8M0 not in ACLTYPE_TO_CTYPE:
        ACLTYPE_TO_CTYPE[AclDataType.ACL_FLOAT8_E8M0] = ctypes.c_ubyte


def ensure_atk_float4_e2m1_support():
    if not hasattr(AclDataType, "ACL_FLOAT4_E2M1"):
        return
    acl_dt = AclDataType.ACL_FLOAT4_E2M1
    for attr in _FLOAT4_TORCH_STORAGE_ATTRS:
        dt = getattr(torch, attr, None)
        if dt is None:
            continue
        key = f"torch.{attr}"
        if key not in TORCH_TO_ACLTYPE:
            TORCH_TO_ACLTYPE[key] = acl_dt
            _logging.info(f"ATK dtype map: register {key} -> ACL_FLOAT4_E2M1")
    if acl_dt not in ACLTYPE_TO_CTYPE:
        ACLTYPE_TO_CTYPE[acl_dt] = ctypes.c_ubyte


try:
    ensure_atk_float8_e8m0_support()
    ensure_atk_float4_e2m1_support()
except Exception as _dtype_reg_err:
    _logging.warning(f"BSA MXFP4 aclnn dtype map register skipped: {_dtype_reg_err}")


def _is_real_torch_dtype(dt) -> bool:
    """torch_npu 会把 ACL 枚举常量(如 float4_e2m1fn_x2 == 296)挂在同名属性上,
    必须确认拿到的是真正的 torch.dtype, 否则 view(296) 会报 shape 错误。"""
    return isinstance(dt, torch.dtype)


def get_float4_storage_dtype():
    for attr in _FLOAT4_TORCH_STORAGE_ATTRS:
        dt = getattr(torch, attr, None)
        if dt is not None and _is_real_torch_dtype(dt):
            return dt
    try:
        import torch_npu

        for attr in _FLOAT4_NPU_STORAGE_ATTRS:
            dt = getattr(torch_npu, attr, None)
            if dt is not None and _is_real_torch_dtype(dt):
                return dt
    except ImportError:
        pass
    return None


def prepare_fp4_tensor_on_npu(t: torch.Tensor) -> torch.Tensor:
    """uint8 nibble-pack 上 NPU 后 view 为 float4 storage (对齐 GMM aclnn 路径)."""
    dt = get_float4_storage_dtype()
    if dt is None:
        return t.npu().contiguous() if not t.is_npu else t.contiguous()
    if t.dtype == dt:
        u8 = t.view(torch.uint8).contiguous()
    elif t.dtype == torch.uint8:
        u8 = t.contiguous()
    else:
        u8 = t.to(torch.uint8).contiguous()
    return u8.npu().view(dt)


def to_float8_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """uint8 存储的 e8m0 指数 -> float8_e8m0fnu view。

    没有该 dtype 的环境保持 uint8, 由 aclCreateTensor 侧按 FLOAT8_E8M0 解读。
    """
    e8 = getattr(torch, "float8_e8m0fnu", None)
    if e8 is None:
        return scale.contiguous()
    if scale.dtype == e8:
        return scale.contiguous()
    return scale.to(torch.uint8).contiguous().view(e8)


def _fp4_logical_shape(
    storage_shape: Tuple[int, ...], head_dim: int
) -> Tuple[int, ...]:
    logical = list(storage_shape)
    logical[-1] = int(head_dim)
    return tuple(logical)


def _contiguous_strides_mxfp4(shape: Tuple[int, ...]) -> List[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _create_acl_tensor_mxfp4(
    npu_tensor: torch.Tensor, acl_dtype: int, logical_shape: Tuple[int, ...]
) -> AclTensorStruct:
    """按给定 ACL dtype 与逻辑 shape 建 aclTensor。

    view dims 与 storage dims 都取 logical_shape: FP4 的缓冲区是 uint8 [..., D/2],
    而算子要看到 [..., D]; 若 storage 填物理 shape, ACL 会按 0.5B/元素再算一遍偏移,
    读到的地址就错位了。
    """
    if npu_tensor.device.type != "npu":
        npu_tensor = npu_tensor.npu().contiguous()
    ndim = len(logical_shape)
    dims = (Int64 * ndim)(*[int(x) for x in logical_shape])
    strides = (Int64 * ndim)(
        *_contiguous_strides_mxfp4(tuple(int(x) for x in logical_shape))
    )
    addr = ctypes.c_void_p(npu_tensor.data_ptr())
    acl_tensor = nnopbase.aclCreateTensor(
        dims, ndim, acl_dtype, strides, 0, AclFormat.ACL_FORMAT_ND, dims, ndim, addr
    )
    if not acl_tensor:
        raise RuntimeError(
            f"aclCreateTensor 失败: dtype={acl_dtype} shape={tuple(logical_shape)}"
        )
    return AclTensorStruct(
        acl_tensor, addr.value, npu_tensor, ascendcl.aclDataTypeSize(acl_dtype)
    )


def create_fp4_acl_tensor(packed_uint8: torch.Tensor, head_dim: int) -> AclTensorStruct:
    """uint8 nibble-pack [..., D/2] -> FLOAT4_E2M1 aclTensor [..., D]。"""
    npu_t = prepare_fp4_tensor_on_npu(packed_uint8)
    logical = _fp4_logical_shape(tuple(int(x) for x in npu_t.shape), head_dim)
    return _create_acl_tensor_mxfp4(npu_t, AclDataType.ACL_FLOAT4_E2M1, logical)


def create_e8m0_acl_tensor(scale: torch.Tensor) -> AclTensorStruct:
    """e8m0 scale -> FLOAT8_E8M0 aclTensor。

    先按 uint8 上 NPU 再 view: float8_e8m0fnu 的 CPU->NPU 直拷在部分 torch_npu
    版本上不支持, 与 FP4 同样的处理。
    """
    u8 = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    npu_t = to_float8_e8m0(u8.contiguous().npu())
    return _create_acl_tensor_mxfp4(
        npu_t, AclDataType.ACL_FLOAT8_E8M0, tuple(int(x) for x in npu_t.shape)
    )


def create_int64_array_mxfp4(values):
    """int 序列 -> aclIntArray*; None 表示不配置(空指针)。

    空指针必须带正确的 ctypes 类型: ATK before_call 会拿 input_args 的类型逐个与
    C++ 原型比对, 用 aclTensor* 的空指针冒充 aclIntArray* 会当场判成类型不匹配。
    """
    if values is None:
        return ctypes.POINTER(AclIntArray)()
    return nnopbase.create_x_list([Int64(int(v)) for v in values])


random.seed(21)
np.random.seed(21)
torch.manual_seed(21)
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_printoptions(threshold=float("inf"))

DEBUG_mxfp4 = False
if not DEBUG_mxfp4:
    print = lambda *args, **kwargs: None


# =====================================================================
# MXFP4 (E2M1) 量化/反量化/打包工具
# =====================================================================
QUANT_MODE_OCP = 2
QUANT_MODE_CX = 3
CX_DEFAULT_DST_TYPE_MAX = 6.0
E8M0_BIAS_mxfp4 = 127
FP4_E2M1_EMAX = 2

FP4_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_FP4_POS_LEVELS_FP32 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_FP4_POS_MIDPOINTS_FP32 = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)


def _resolve_cx_params(dst_type_max):
    cx_val = (
        CX_DEFAULT_DST_TYPE_MAX
        if (dst_type_max is None or dst_type_max == 0)
        else float(dst_type_max)
    )
    log2_cx = math.log2(cx_val)
    ceil_log2_cx = math.ceil(log2_cx)
    return log2_cx, ceil_log2_cx


def generate_fp4_e2m1_encoding_table():
    encodings, values = [], []
    for s in [0, 1]:
        for e in range(4):
            for m in range(2):
                enc = (s << 3) | (e << 1) | m
                val = (
                    (-1) ** s * (2.0 ** (e - 1)) * (1.0 + m / 2.0)
                    if e != 0
                    else (-1) ** s * (2.0**0) * (m / 2.0)
                )
                encodings.append(enc)
                values.append(val)
    fp32_to_encoding = {}
    for enc, val in zip(encodings, values):
        if val not in fp32_to_encoding:
            fp32_to_encoding[val] = enc
    return fp32_to_encoding


FP32_TO_FP4 = generate_fp4_e2m1_encoding_table()


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        expanded = tensor.unsqueeze(-1)
        idx = torch.argmin(
            torch.abs(expanded - FP4_E2M1_VALUES.to(tensor.device)), dim=-1
        )
        return FP4_E2M1_VALUES.to(tensor.device)[idx]


# quantize_fp4 会把最后一维展开成 16 档做 argmin, 峰值内存是输入的 16 倍;
# 大序列用例下按块数分片, 把峰值压到百 MB 量级(数值结果与整体处理完全一致)。
_QUANT_FP4_CHUNK_BLOCKS = 1 << 16


def _quantize_fp4_chunked(blocks: torch.Tensor) -> torch.Tensor:
    if blocks.shape[0] <= _QUANT_FP4_CHUNK_BLOCKS:
        return quantize_fp4(blocks)
    out = torch.empty_like(blocks)
    for lo in range(0, blocks.shape[0], _QUANT_FP4_CHUNK_BLOCKS):
        hi = min(lo + _QUANT_FP4_CHUNK_BLOCKS, blocks.shape[0])
        out[lo:hi] = quantize_fp4(blocks[lo:hi])
    return out


def compute_e8m0_scale_blocks(
    blocks: torch.Tensor,
    mode: str = "baseline",
    quant_mode: int = QUANT_MODE_OCP,
    log2_cx: float = 0.0,
) -> torch.Tensor:
    """blocks: [n_blk, block_size] -> scales: [n_blk]。逐块版的向量化形式，语义等价。"""
    max_abs = blocks.abs().amax(dim=-1)
    log_max = torch.log2(max_abs)
    if quant_mode == QUANT_MODE_CX:
        exp = torch.ceil(log_max - log2_cx)
    else:
        exp = (
            torch.floor(log_max) if mode == "baseline" else torch.ceil(log_max)
        ) - FP4_E2M1_EMAX
    exp = torch.clamp(exp, -127.0, 127.0)
    scales = torch.pow(torch.tensor(2.0, dtype=exp.dtype, device=exp.device), exp)
    # 全零块: log2(0)=-inf 会被 clamp 成 -127, 这里按逐块版语义统一取 1.0
    return torch.where(max_abs < 1e-30, torch.ones_like(scales), scales)


def mx_quantize(
    tensor: torch.Tensor,
    block_size: int = 32,
    mode: str = "baseline",
    quant_mode: int = QUANT_MODE_OCP,
    log2_cx: float = 0.0,
):
    original_shape = tensor.shape
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    last = tensor.shape[-1]
    pad = 0
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad), value=0.0)
    reshaped = tensor.reshape(-1, block_size)
    scales = compute_e8m0_scale_blocks(
        reshaped, mode=mode, quant_mode=quant_mode, log2_cx=log2_cx
    )
    fp4 = _quantize_fp4_chunked(reshaped / scales.unsqueeze(-1))
    fp4 = fp4.view(tensor.shape)
    if pad > 0:
        fp4 = fp4[..., :-pad]
    return fp4, scales


def mx_dequantize(
    fp4_vals: torch.Tensor, scales: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    original_shape = fp4_vals.shape
    last = original_shape[-1] if fp4_vals.dim() > 0 else 1
    pad = 0
    if fp4_vals.dim() > 0 and last % block_size != 0:
        pad = block_size - (last % block_size)
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad), value=0.0)
    reshaped = fp4_vals.reshape(-1, block_size)
    dq = reshaped * scales.unsqueeze(1).to(reshaped.device)
    dq = dq.view(fp4_vals.shape)
    if pad > 0:
        dq = dq[..., :last]
    return dq


def mxfp4_qdq(
    t: torch.Tensor,
    axis: int,
    block_size: int = 32,
    mode: str = "baseline",
    quant_mode: int = QUANT_MODE_OCP,
    log2_cx: float = 0.0,
) -> torch.Tensor:
    t_last = t.movedim(axis, -1).contiguous()
    fp4_sim, scales = mx_quantize(
        t_last, block_size=block_size, mode=mode, quant_mode=quant_mode, log2_cx=log2_cx
    )
    dq = mx_dequantize(fp4_sim, scales, block_size=block_size)
    return dq.movedim(-1, axis).contiguous()


def convert_fp32_simulated_to_fp4_encoding(fp32_sim: torch.Tensor) -> torch.Tensor:
    flat = fp32_sim.flatten().cpu()
    enc = torch.zeros_like(flat, dtype=torch.uint8)
    for fp32_val, encoding in FP32_TO_FP4.items():
        enc[flat == fp32_val] = encoding
    return enc.to(fp32_sim.device).reshape(fp32_sim.shape)


def pack_fp4_to_uint8(fp4_enc: torch.Tensor) -> torch.Tensor:
    shape = fp4_enc.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {shape[-1]}")
    reshaped = fp4_enc.reshape(-1, 2)
    packed = (reshaped[:, 1] << 4) | (reshaped[:, 0] & 0x0F)
    return packed.reshape(*shape[:-1], shape[-1] // 2)


def mxfp4_quantize_pack_last(
    tensor: torch.Tensor,
    quant_axis: int,
    block_size: int = 32,
    mode: str = "baseline",
    quant_mode: int = QUANT_MODE_OCP,
    log2_cx: float = 0.0,
):
    nd = tensor.dim()
    qax = quant_axis % nd
    assert tensor.shape[-1] % 2 == 0, f"last dim must be even, got {tensor.shape[-1]}"
    t_q_last = tensor.movedim(qax, -1).contiguous()
    fp4_sim, scales_flat = mx_quantize(
        t_q_last,
        block_size=block_size,
        mode=mode,
        quant_mode=quant_mode,
        log2_cx=log2_cx,
    )
    codes = (
        convert_fp32_simulated_to_fp4_encoding(fp4_sim).movedim(-1, qax).contiguous()
    )
    packed = pack_fp4_to_uint8(codes)
    qdim = tensor.shape[qax]
    n_blk = (qdim + block_size - 1) // block_size
    rest = list(t_q_last.shape[:-1])
    scales_real = scales_flat.reshape(*rest, n_blk).movedim(-1, qax).contiguous()
    exp_int = torch.log2(scales_real).round().long()
    e8m0 = (exp_int + E8M0_BIAS_mxfp4).clamp(0, 254).to(torch.uint8)
    return packed, e8m0


def _e8m0_qk_to_v3_mxfp4(e8m0_uint8):
    *lead, nd = e8m0_uint8.shape
    paired = e8m0_uint8.reshape(*lead, nd // 2, 2)
    return paired.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3_mxfp4(e8m0_uint8):
    B, N2, nb, D = e8m0_uint8.shape
    if nb % 2 != 0:
        pad_block = torch.full((B, N2, 1, D), E8M0_BIAS_mxfp4, dtype=torch.uint8)
        e8m0_uint8 = torch.cat([e8m0_uint8, pad_block], dim=2)
        nb += 1
    return (
        e8m0_uint8.reshape(B, N2, nb // 2, 2, D)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )


# =====================================================================
# blockwise_snap_local P量化
# =====================================================================
def _pack_fp4_e2m1_along_last(idx_tensor):
    *batch, n = idx_tensor.shape
    if n % 2 == 1:
        pad = torch.zeros(*batch, 1, dtype=idx_tensor.dtype, device=idx_tensor.device)
        idx_tensor = torch.cat([idx_tensor, pad], dim=-1)
        n += 1
    pairs = idx_tensor.reshape(*batch, n // 2, 2)
    low = pairs[..., 0].to(torch.uint8) & 0xF
    high = pairs[..., 1].to(torch.uint8) & 0xF
    return (high << 4) | low


def _blockwise_snap_local_quantize_p_eff_mxfp4(
    S: torch.Tensor,
    m_running,
    seqk_dim: int,
    mx_block_size: int = 32,
    mx_mode: str = "baseline",
    is_cx: bool = False,
    neg_log2_cx: float = 0.0,
    p_quant_exp_shift: float = 2.0,
):
    NEG_INF = float("-inf")
    device = S.device
    src_dtype = S.dtype
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[skp]
    LN2 = math.log(2.0)

    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(S.shape)
        pshape[skp] = pad
        S_padded = torch.cat(
            [S, torch.full(pshape, NEG_INF, dtype=src_dtype, device=device)], dim=skp
        )
    else:
        S_padded = S
    S_moved = S_padded.movedim(skp, -1)
    other = S_moved.shape[:-1]
    n_blk = S_moved.shape[-1] // mx_block_size
    S_r = S_moved.reshape(*other, n_blk, mx_block_size)

    m_block_raw = S_r.max(dim=-1).values
    m_block_raw_safe = torch.where(
        torch.isinf(m_block_raw) & (m_block_raw < 0),
        torch.zeros_like(m_block_raw),
        m_block_raw,
    )
    if is_cx:
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)
        neg_log2_cx_fp16 = torch.tensor(neg_log2_cx, dtype=torch.float16, device=device)
        k_raw = m_block_raw_safe * INV_LN2_fp16 + neg_log2_cx_fp16
        K_block = torch.ceil(k_raw)
        m_block_snap = K_block * LN2_fp16
    else:
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)
        K_block = torch.floor(m_block_raw_safe * INV_LN2_fp16)
        m_block_snap = (K_block - 2) * LN2_fp16

    P_local = torch.exp(S_r - m_block_snap.unsqueeze(-1))
    P_local_bf = P_local.to(torch.bfloat16).to(P_local.dtype)

    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    idx = torch.bucketize(P_local_bf.contiguous(), midpoints, right=True)
    P_q = levels[idx]

    K_ij = K_block.max(dim=-1).values
    K_ij_fp32 = K_ij.float()
    m_new = K_ij_fp32 if m_running is None else torch.maximum(m_running, K_ij_fp32)
    K_new_s = m_new.to(src_dtype)

    if is_cx:
        K_diff = K_block - K_new_s.unsqueeze(-1)
        corr = torch.exp2(
            K_diff - torch.tensor(p_quant_exp_shift, dtype=torch.float16, device=device)
        )
    else:
        K_diff = K_block - K_new_s.unsqueeze(-1)
        corr = torch.exp2(K_diff - 2.0)

    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(
        -1, skp
    )
    P_q_moved = P_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(
        -1, skp
    )
    P_eff = P_q_moved * corr_broad
    return P_eff, m_new


# =====================================================================
# s_dtype 解析
# =====================================================================
_S_DTYPE_MAP_mxfp4 = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_OUT_DTYPE_MAP_mxfp4 = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _resolve_s_dtype(s_dtype):
    if s_dtype not in _S_DTYPE_MAP_mxfp4:
        raise ValueError(
            f"s_dtype must be one of {list(_S_DTYPE_MAP_mxfp4)}, got '{s_dtype}'"
        )
    return _S_DTYPE_MAP_mxfp4[s_dtype]


def _resolve_out_dtype(out_dtype):
    if out_dtype not in _OUT_DTYPE_MAP_mxfp4:
        raise ValueError(
            f"out_dtype must be one of {list(_OUT_DTYPE_MAP_mxfp4)}, got '{out_dtype}'"
        )
    return _OUT_DTYPE_MAP_mxfp4[out_dtype]


# =====================================================================
# block sparse pattern 生成
# =====================================================================
# 稀疏率候选。必须含 1.0: 每行保留块数按 ceil(ck * ratio) 算, ratio=1.0 才能取到 dense(mask 全 1),
# 而 dense 是最主流形态。1.0 给双份权重, 其余取值覆盖不同密度。
SPARSITY_RATIO_CHOICES_mxfp4 = (1.0, 0.9, 0.8, 0.75, 0.5, 0.4, 0.25, 0.1)


def build_block_sparse_mask(
    sparsity_ratio, q_seqlen_list, kv_seqlen_list, block_shape, batch, num_heads
):
    """按 per-batch 实际长度分块生成 blockSparseMask [B, N1, ceil_q, ceil_kv] int8。

    每个 q-block 随机 attend ceil(该 batch 的 kv 块数 * sparsity_ratio) 个 kv 块(至少 1 个)。
    维度用 max(实际长度) 算; 某 batch 实际块数之外的位置保持 0(该 batch 没有对应 token)。
    调用方负责再把维度 0-pad 到 CeilDiv(S_padded, block) 以满足 host 校验。
    """
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (max(q_seqlen_list) + block_x - 1) // block_x
    ceil_kv = (max(kv_seqlen_list) + block_y - 1) // block_y
    mask = np.zeros((batch, num_heads, ceil_q, ceil_kv), dtype=np.int8)

    for b in range(batch):
        cq_b = (q_seqlen_list[b] + block_x - 1) // block_x
        ck_b = (kv_seqlen_list[b] + block_y - 1) // block_y
        n_attend = min(ck_b, max(1, math.ceil(ck_b * sparsity_ratio)))
        if n_attend == ck_b:
            mask[b, :, :cq_b, :ck_b] = 1
            continue
        for n1 in range(num_heads):
            for q_blk_id in range(cq_b):
                sel = np.random.choice(ck_b, n_attend, replace=False)
                mask[b, n1, q_blk_id, sel] = 1
    return mask


# =====================================================================
# 数据生成辅助
# =====================================================================
def as_dim(value, name="dim"):
    """把维度值转成 Python 原生 int, 并挡住负值。

    维度非法时不同环境的报错差异很大: 有的 torch 版本对负维度直接说 "negative dimension",
    有的却报 "SymIntArrayRef expected to contain only concrete integers", 完全看不出是负数。
    在造数入口自己校验一遍, 把上游给的坏值当场暴露出来。
    """
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        dim = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} 无法转为 int: type={type(value)} value={value!r}"
        ) from exc
    if dim < 0:
        raise ValueError(f"{name} 为负数({dim}), 上游传入的维度非法")
    return dim


def as_shape(shape):
    return tuple(as_dim(s, f"shape[{i}]") for i, s in enumerate(shape))


def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    data_range = [-x, x]
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(
        as_shape(shape), dtype=dtype, device=device
    )


# 造数幅度。上限取 10.0(与参考脚本默认一致): 再放大后 S=Q·Kᵀ·scale 的量级为 sqrt(D)·x²,
# 在 s_dtype=fp16 下会溢出, 那是造数问题而非算子问题。
DATA_X_CHOICES = (1.0, 10.0, 0.1, 4.0)


def _canon_seed_part(part):
    """把种子输入规范化成与具体数值类型无关的字符串。

    不能直接 repr(): np.int64(2) / torch 标量 / Python int 的 repr 各不相同, 而两个后端喂过来的
    kwargs 类型未必一致, 那样 cpu 与 npu 会算出不同的种子, 造出不同的数据。
    """
    if isinstance(part, (list, tuple)):
        return "[" + ",".join(_canon_seed_part(p) for p in part) + "]"
    if isinstance(part, str):
        return part
    if isinstance(part, float):
        return f"{part:.6g}"
    return str(int(part))


def case_derived_seed(*parts):
    """由用例参数派生随机种子。

    cpu / npu 两个节点各自独立执行 init_by_input_data, 必须拿到同一份数据, 所以种子只能来自
    用例参数; 但固定成常量会让所有用例共用同一份数据(稀疏度、数值幅度都退化成单点取值)。
    用 crc32 而非内置 hash(): 后者对 str 逐进程加盐, 跨进程不稳定。
    """
    key = "|".join(_canon_seed_part(p) for p in parts)
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def _act_seq_given(value):
    """判断 actualSeqLengths 这类 attr_tuple 是否真的配置了。

    "不配置"的表达形式不统一: None、字符串 "None"、空序列, 以及把未配置的整型 attr 物化成
    负数哨兵。最后一种最危险 —— 当成真实长度用会一路传到张量维度上, 而某些 torch 版本对负
    维度报的是 "SymIntArrayRef expected to contain only concrete integers", 根本看不出根因。
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value != "None"
    if len(value) == 0:
        return False
    if all(isinstance(v, str) for v in value):
        return not all(v == "None" for v in value)
    try:
        return min(int(v) for v in value) >= 1
    except (TypeError, ValueError):
        return False


def zero_degenerate_(kv_parts, n_tokens=32):
    """把每个 batch 的前 n_tokens 个 kv token 置 0(就地)。

    目的是造出"整个 32 元素量化块全零"的输入, 覆盖 e8m0 scale 的退化分支
    (max_abs≈0 时取 scale=1.0)。只置零一小段而非全部, 输出仍非平凡, 精度指标才有分辨力。
    """
    for t in kv_parts:
        t[:, :, : min(n_tokens, t.shape[2]), :] = 0.0


# =====================================================================
# CPU golden — BNSD block-sparse MXFP4 分块 flash attention
# =====================================================================
def cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    scale_value,
    num_kv_heads,
    p_quant="snap_local",
    s_dtype="fp16",
    out_dtype="bfloat16",
    quant_mode=QUANT_MODE_OCP,
    dst_type_max=0.0,
):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    B, N, S, D = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    group_size = N // N2
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S2 + block_y - 1) // block_y
    s_torch_dtype = _resolve_s_dtype(s_dtype)

    is_cx = quant_mode == QUANT_MODE_CX
    neg_log2_cx = 0.0
    log2_cx = 0.0
    if is_cx:
        log2_cx, ceil_log2_cx = _resolve_cx_params(dst_type_max)
        neg_log2_cx = -log2_cx
        p_quant_exp_shift = float(ceil_log2_cx)
    else:
        p_quant_exp_shift = 2.0

    q_q = mxfp4_qdq(q, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
    k_q = mxfp4_qdq(k, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
    v_q = mxfp4_qdq(v, axis=2, quant_mode=quant_mode, log2_cx=log2_cx)

    out = torch.zeros((B, N, S, D), dtype=torch.float32)
    lse = torch.full((B, N, S, 1), float("-inf"), dtype=torch.float32)
    scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)

    for b in range(B):
        for n in range(N):
            n2 = n // group_size
            for qb in range(ceil_q):
                q_lo = qb * block_x
                q_hi = min(q_lo + block_x, S)
                if q_lo >= S:
                    break
                sel_j = [
                    j for j in range(ceil_kv) if block_sparse_mask[b, n, qb, j] != 0
                ]
                if not sel_j:
                    continue
                chunks = [sel_j[i : i + 2] for i in range(0, len(sel_j), 2)]
                Q_blk = q_q[b, n, q_lo:q_hi]
                nq = q_hi - q_lo
                m_run = None
                l_run = torch.zeros(nq, dtype=torch.float32)
                acc = torch.zeros((nq, D), dtype=torch.float32)

                for chunk in chunks:
                    positions = []
                    for j in chunk:
                        st = j * block_y
                        ed = min(st + block_y, S2)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    K_blk = k_q[b, n2][pos]
                    V_blk = v_q[b, n2][pos]

                    S_fp32 = (K_blk @ Q_blk.t()).to(torch.float32)
                    S_s = (S_fp32 * scale_s).to(s_torch_dtype)

                    if p_quant == "snap_local":
                        P_eff_s, m_new = _blockwise_snap_local_quantize_p_eff_mxfp4(
                            S_s,
                            m_running=m_run,
                            seqk_dim=0,
                            is_cx=is_cx,
                            neg_log2_cx=neg_log2_cx,
                            p_quant_exp_shift=p_quant_exp_shift,
                        )
                        P_eff_s = torch.nan_to_num(P_eff_s, nan=0.0)
                        P_fp32 = P_eff_s.float()
                        alpha = (
                            torch.zeros(nq, dtype=torch.float32)
                            if m_run is None
                            else torch.exp2(m_run.float() - m_new.float())
                        )
                    else:
                        m_s = S_s.max(dim=0).values
                        m_safe_s = torch.where(
                            torch.isinf(m_s) & (m_s < 0), torch.zeros_like(m_s), m_s
                        )
                        P_ij = torch.exp(S_s - m_safe_s.unsqueeze(0))
                        P_fp32 = torch.nan_to_num(P_ij, nan=0.0).float()
                        m_new = m_safe_s.float()
                        alpha = (
                            torch.zeros(nq, dtype=torch.float32)
                            if m_run is None
                            else torch.exp(m_run - m_new)
                        )

                    reduce_sum = P_fp32.sum(dim=0)
                    l_run = alpha * l_run + reduce_sum
                    acc = alpha.unsqueeze(-1) * acc + (P_fp32.t() @ V_blk)
                    m_run = m_new

                l_safe = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                acc = acc / l_safe.unsqueeze(-1)
                out[b, n, q_lo:q_hi] = acc
    return out.to(_resolve_out_dtype(out_dtype)), lse.to(torch.float32)


# =====================================================================
# 精度比较
# =====================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        return 100.0, "Pass"
    if real_data.size != data_compe.size:
        return 0.0, "Failed"
    if data_type in ("bfloat16", "float16"):
        rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0
    else:
        rtol, atol, max_diff_hd = 0.005, 0.000025, 10.0
    diff_result = np.isclose(
        real_data.astype(np.float32),
        data_compe.astype(np.float32),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    err_idx = np.where(diff_result != np.array((True,)))[0]
    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    diff_abs = np.abs(data_compe - real_data)
    b = np.add(np.maximum(np.abs(real_data), np.abs(data_compe)), 10e-10)
    err_diff = (diff_abs / b)[err_idx]
    pct_thd_val = (1 - pct_thd) * 100.0
    result_str = "Pass" if fulfill_percent >= pct_thd_val else "Failed"
    if len(err_diff) > 0 and np.max(err_diff) >= max_diff_hd:
        result_str = "Failed"
    return fulfill_percent, result_str


# =====================================================================
# ATK golden 计算类
# =====================================================================
class TestBlockSparseAttentionMxfp4GenBmGt:
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        embedding_size: int
        inner_prec: int
        scale: float
        q_layout: str
        kv_layout: str
        attention_out_dtype: str
        quant_mode: int
        dst_type_max: float

    @dataclass
    class AttentionInputs:
        # q/k/v_parts: per-batch 实际长度(未 padding)的 BNSD 规范张量列表
        #   q_parts[bi]: [1, N1, actualSeqLen[bi], D]; k/v_parts[bi]: [1, N2, actualSeqLenKv[bi], D]
        q_parts: any
        k_parts: any
        v_parts: any
        block_sparse_mask: any
        block_sparse_shape: any
        q_seqlen_list: any
        kv_seqlen_list: any
        padded_s1: any  # BNSD/BSND 输出张量 S 维(padded 大小); TND 为 None
        aux_attrs: any

    def compute_output(self, attention_inputs):
        """Benchmark 分支（is_benchmark=False）：MXFP4 qdq + snap_local P量化 分块flash"""
        block_sparse_mask_np = attention_inputs.block_sparse_mask
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        out_dtype_str = attention_inputs.aux_attrs.attention_out_dtype
        quant_mode = attention_inputs.aux_attrs.quant_mode
        dst_type_max = attention_inputs.aux_attrs.dst_type_max
        group_size = int(num_heads / kv_heads)

        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]

        q_parts = attention_inputs.q_parts
        k_parts = attention_inputs.k_parts
        v_parts = attention_inputs.v_parts

        mask_tensor = torch.from_numpy(block_sparse_mask_np)

        cpu_chunks = []
        for bi in range(batch):
            cq_b = (q_seqlen_list[bi] + block_shape_x - 1) // block_shape_x
            ck_b = (kv_seqlen_list[bi] + block_shape_y - 1) // block_shape_y
            mask_b = mask_tensor[bi : bi + 1, :, :cq_b, :ck_b]
            out_b, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
                q_parts[bi],
                k_parts[bi],
                v_parts[bi],
                mask_b,
                [block_shape_x, block_shape_y],
                scale,
                kv_heads,
                p_quant="snap_local",
                s_dtype="fp16",
                out_dtype=out_dtype_str,
                quant_mode=quant_mode,
                dst_type_max=dst_type_max,
            )
            cpu_chunks.append(out_b)

        return self._assemble_result(
            cpu_chunks,
            q_layout,
            batch,
            num_heads,
            embedding_size,
            q_seqlen_list,
            attention_inputs.padded_s1,
        )

    def compute_output_benchmark(self, attention_inputs):
        """Benchmark 真值分支（is_benchmark=True）：用量化前的原始 fp32 QKV 做标准 BSA"""
        block_sparse_mask_np = attention_inputs.block_sparse_mask
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        scale = attention_inputs.aux_attrs.scale
        group_size = int(num_heads / kv_heads)

        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]

        # 直接用原始 fp32 数据（不做 mxfp4 qdq）
        q_parts = attention_inputs.q_parts
        k_parts = attention_inputs.k_parts
        v_parts = attention_inputs.v_parts

        mask_tensor = torch.from_numpy(block_sparse_mask_np)

        cpu_chunks = []
        for bi in range(batch):
            cq_b = (q_seqlen_list[bi] + block_shape_x - 1) // block_shape_x
            ck_b = (kv_seqlen_list[bi] + block_shape_y - 1) // block_shape_y
            mask_b = mask_tensor[bi : bi + 1, :, :cq_b, :ck_b]

            q_b = q_parts[bi]  # [1, N, S1, D] fp32
            k_b = k_parts[bi]  # [1, N2, S2, D] fp32
            v_b = v_parts[bi]  # [1, N2, S2, D] fp32
            B_b, N_b, S1_b, D_b = q_b.shape
            N2_b = k_b.shape[1]
            S2_b = k_b.shape[2]

            out_b = torch.zeros((1, N_b, S1_b, D_b), dtype=torch.float32)

            for n in range(N_b):
                n2 = n // group_size
                for qb_idx in range(cq_b):
                    q_lo = qb_idx * block_shape_x
                    q_hi = min(q_lo + block_shape_x, S1_b)
                    if q_lo >= S1_b:
                        break
                    sel_j = [j for j in range(ck_b) if mask_b[0, n, qb_idx, j] != 0]
                    if not sel_j:
                        continue
                    positions = []
                    for j in sel_j:
                        st = j * block_shape_y
                        ed = min(st + block_shape_y, S2_b)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    Q_blk = q_b[0, n, q_lo:q_hi]  # [nq, D]
                    K_blk = k_b[0, n2][pos]  # [npos, D]
                    V_blk = v_b[0, n2][pos]  # [npos, D]

                    # fp32 标准 attention
                    s = (Q_blk.float() @ K_blk.float().t()) * scale
                    row_max = s.max(dim=-1, keepdim=True).values
                    p = torch.exp(s - row_max)
                    p = p / p.sum(dim=-1, keepdim=True)
                    o = p.float() @ V_blk.float()
                    out_b[0, n, q_lo:q_hi] = o

            cpu_chunks.append(out_b)

        return self._assemble_result(
            cpu_chunks,
            q_layout,
            batch,
            num_heads,
            embedding_size,
            q_seqlen_list,
            attention_inputs.padded_s1,
        )

    def _assemble_result(
        self,
        cpu_chunks,
        q_layout,
        batch,
        num_heads,
        embedding_size,
        q_seqlen_list,
        padded_s1=None,
    ):
        """将 per-batch BNSD 结果拼回原始 layout。

        BNSD/BSND 变长(padded)场景: 输出张量 S 维为 padded 大小, [actualSeqLen, S_padded) 区间
        算子不写(kernel 不清零 padding 区), golden 与 NPU 侧都显式置 0 后再比对。
        """
        if q_layout == "TND":
            result_parts = []
            for bi in range(batch):
                result_parts.append(cpu_chunks[bi].permute(0, 2, 1, 3).flatten(0, 1))
            result = torch.cat(result_parts, dim=0)
        elif q_layout == "BNSD":
            out_s = padded_s1 if padded_s1 is not None else max(q_seqlen_list)
            result = torch.zeros(
                as_shape((batch, num_heads, out_s, embedding_size)),
                dtype=cpu_chunks[0].dtype,
            )
            for bi in range(batch):
                result[bi, :, : q_seqlen_list[bi]] = cpu_chunks[bi][0]
        elif q_layout == "BSND":
            out_s = padded_s1 if padded_s1 is not None else max(q_seqlen_list)
            result = torch.zeros(
                as_shape((batch, out_s, num_heads, embedding_size)),
                dtype=cpu_chunks[0].dtype,
            )
            for bi in range(batch):
                result[bi, : q_seqlen_list[bi]] = cpu_chunks[bi][0].permute(1, 0, 2)

        return result.to(torch.float32)


def tensor_to_numpy_array_mxfp4(t):
    if t is None:
        return None
    if t.dtype is torch.bfloat16:
        return t.to(torch.float32).numpy().astype(bfloat16)
    else:
        return t.numpy()


def numpy_array_to_tensor_mxfp4(arr):
    if arr is None:
        return None
    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


# =====================================================================
# ATK 接口注册: 本类只承担 CPU 标杆 + 造数(prep)。被测节点走
# PyAclnnBlockSparseAttentionV3Api_mxfp4(pyaclnn 直调两段式 aclnn)。
# 不 import cann_ops_transformer 的 torch 封装——import 会触发
# OpBuilder/torch.utils.cpp_extension.load 的 JIT 编译。
# =====================================================================
@register("aclnn_blocksparseattention_v3")
class BlockSparseAttentionV3Api_mxfp4(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionV3Api_mxfp4, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        # _logging.info(f"=======================================init_by_input_data=======================================")
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        # 标量属性一律归一成 Python 原生类型: 后端给过来的可能是 np.int64 / 张量标量 / SymInt,
        # 直接拿去构造 torch shape 会报 "expected to contain only concrete integers",
        # 且 cpu/npu 两个后端的类型未必一致, 会让 case_derived_seed 算出不同的种子。
        self.kv_heads = int(input_data.kwargs["numKeyValueHeads"])
        self.scale_value = float(input_data.kwargs["scaleValue"])
        self.inner_precise = int(input_data.kwargs["innerPrecise"])
        self.softmax_lse_flag = int(input_data.kwargs["softmaxLseFlag"])
        self.quant_mode = int(input_data.kwargs.get("quantMode", QUANT_MODE_OCP))
        self.dst_type_max = float(input_data.kwargs.get("dstTypeMax", 0.0))
        self.block_shape = [int(v) for v in input_data.kwargs["blockShape"]]
        raw_act_q = input_data.kwargs["actualSeqLengths"]
        raw_act_kv = input_data.kwargs["actualSeqLengthsKv"]
        # actualSeqLengths/Kv 允许同时不配置(doc: 二者必须同时配置或同时不配置), 此时算子按张量
        # S 维全长计算。TND 下二者必选, 只有 BNSD/BSND 会走这条分支。
        self.pass_act_seq = _act_seq_given(raw_act_q) and _act_seq_given(raw_act_kv)

        attention_out = input_data.kwargs["attentionOut"]
        if attention_out.dtype == torch.bfloat16:
            self.attention_out_dtype_str = "bfloat16"
            self.attention_out_torch_dtype = torch.bfloat16
        else:
            self.attention_out_dtype_str = "float16"
            self.attention_out_torch_dtype = torch.float16

        query_packed = input_data.kwargs["query"]
        if self.q_input_layout == "TND":
            assert self.pass_act_seq, "TND 下 actualSeqLengths/Kv 为必选"
            self.batch = len(raw_act_q)
            self.num_heads = int(query_packed.shape[1])
            self.headDim = int(query_packed.shape[2]) * 2
        elif self.q_input_layout == "BNSD":
            self.batch = int(query_packed.shape[0])
            self.num_heads = int(query_packed.shape[1])
            self.headDim = int(query_packed.shape[3]) * 2
        elif self.q_input_layout == "BSND":
            self.batch = int(query_packed.shape[0])
            self.num_heads = int(query_packed.shape[2])
            self.headDim = int(query_packed.shape[3]) * 2

        if self.pass_act_seq:
            self.q_seqlen_list = tuple(int(x) for x in raw_act_q)
            self.kv_seqlen_list = tuple(int(x) for x in raw_act_kv)
        else:
            # 不配置时等价于每 batch 都取满张量 S 维, golden 侧仍需要具体长度
            key_shape = input_data.kwargs["key"].shape
            s_dim = 2 if self.q_input_layout == "BNSD" else 1
            self.q_seqlen_list = tuple([int(query_packed.shape[s_dim])] * self.batch)
            self.kv_seqlen_list = tuple([int(key_shape[s_dim])] * self.batch)

        assert min(self.q_seqlen_list) >= 1 and min(self.kv_seqlen_list) >= 1, (
            f"actualSeqLengths 必须为正: q={self.q_seqlen_list} kv={self.kv_seqlen_list} "
            f"(上游原值 q={raw_act_q!r} kv={raw_act_kv!r})"
        )
        assert (
            len(self.q_seqlen_list) == self.batch
            and len(self.kv_seqlen_list) == self.batch
        ), (
            f"actualSeqLengths 长度须等于 batch({self.batch}): "
            f"q={self.q_seqlen_list} kv={self.kv_seqlen_list}"
        )

        self.log2_cx = 0.0
        if self.quant_mode == QUANT_MODE_CX:
            self.log2_cx, _ = _resolve_cx_params(self.dst_type_max)

        q_s_list = self.q_seqlen_list
        kv_s_list = self.kv_seqlen_list

        # 种子必须只由用例参数决定(cpu/npu 两节点各自造数, 结果要逐位一致), 但不能是常量,
        # 否则所有用例共用同一份数据分布。稀疏率与造数幅度同样由它派生。
        case_seed = case_derived_seed(
            self.q_input_layout,
            self.batch,
            self.num_heads,
            self.kv_heads,
            self.headDim,
            q_s_list,
            kv_s_list,
            self.block_shape,
            self.quant_mode,
            self.dst_type_max,
            self.attention_out_dtype_str,
        )
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)

        sparsity_ratio = SPARSITY_RATIO_CHOICES_mxfp4[
            case_seed % len(SPARSITY_RATIO_CHOICES_mxfp4)
        ]
        data_x = DATA_X_CHOICES[
            (case_seed // len(SPARSITY_RATIO_CHOICES_mxfp4)) % len(DATA_X_CHOICES)
        ]
        zero_degenerate = (
            case_seed // (len(SPARSITY_RATIO_CHOICES_mxfp4) * len(DATA_X_CHOICES))
        ) % 4 == 0
        _logging.info(
            f"=======case_seed:{case_seed};sparsity_ratio:{sparsity_ratio};data_x:{data_x};"
            f"zero_degenerate:{zero_degenerate};pass_act_seq:{self.pass_act_seq};========"
        )

        # BNSD/BSND 变长场景: 张量 S 维为 padded 大小, actualSeqLengths 指示每 batch 实际有效长度
        # (V3 doc: BNSD/BSND 配置 actualSeqLengths 后算子按实际长度处理)。
        # 造数策略对齐 test_bsa_mxfp4_golden.py 的 padded 分支:
        #   per-batch 造"未 padding"的 fp32 数据 -> golden 直接用; NPU 侧对同一份数据 0-pad 到张量 S 维再打包。
        #   量化一致性靠 0-padding: Q/K 量化轴=D 不跨 seq; V 量化轴=seq, 末尾不满 32 的块在
        #   unpadded(mx_quantize 内部 0-pad) 与 0-padded(外部 0-pad) 两种情形下填充值均为 0 -> scale 相同。
        if self.q_input_layout == "TND":
            self.padded_s1 = None
            self.padded_s2 = None
            T1 = sum(q_s_list)
            T2 = sum(kv_s_list)
            q_tnd = randn_x((T1, self.num_heads, self.headDim), data_x)
            k_tnd = randn_x((T2, self.kv_heads, self.headDim), data_x)
            v_tnd = randn_x((T2, self.kv_heads, self.headDim), data_x)
            self.q_parts, self.k_parts, self.v_parts = [], [], []
            q_offset, kv_offset = 0, 0
            for bi in range(self.batch):
                sq, skv = q_s_list[bi], kv_s_list[bi]
                self.q_parts.append(
                    q_tnd[q_offset : q_offset + sq]
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                self.k_parts.append(
                    k_tnd[kv_offset : kv_offset + skv]
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                self.v_parts.append(
                    v_tnd[kv_offset : kv_offset + skv]
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                q_offset += sq
                kv_offset += skv
        else:
            key_packed = input_data.kwargs["key"]
            if self.q_input_layout == "BNSD":
                self.padded_s1 = int(query_packed.shape[2])
                self.padded_s2 = int(key_packed.shape[2])
            else:
                self.padded_s1 = int(query_packed.shape[1])
                self.padded_s2 = int(key_packed.shape[1])
            assert (
                max(q_s_list) <= self.padded_s1 and max(kv_s_list) <= self.padded_s2
            ), (
                f"actualSeqLengths 须 <= 张量 S 维: q={q_s_list} vs {self.padded_s1}, "
                f"kv={kv_s_list} vs {self.padded_s2}"
            )
            self.q_parts = [
                randn_x((1, self.num_heads, q_s_list[bi], self.headDim), data_x)
                for bi in range(self.batch)
            ]
            self.k_parts = [
                randn_x((1, self.kv_heads, kv_s_list[bi], self.headDim), data_x)
                for bi in range(self.batch)
            ]
            self.v_parts = [
                randn_x((1, self.kv_heads, kv_s_list[bi], self.headDim), data_x)
                for bi in range(self.batch)
            ]

        if zero_degenerate:
            zero_degenerate_(self.k_parts)
            zero_degenerate_(self.v_parts)

        self.block_sparse_mask_np = build_block_sparse_mask(
            sparsity_ratio,
            q_s_list,
            kv_s_list,
            self.block_shape,
            self.batch,
            self.num_heads,
        )

        # mask 按 per-batch 实际长度分块生成, 再 0-pad 到 host 校验要求的 CeilDiv(S_padded, block)
        # (host CheckSparsePattern 只校验维度, padding 块内容置 0 表示"无 token 可 attend")
        if self.padded_s1 is not None:
            ceil_q_pad = (self.padded_s1 + self.block_shape[0] - 1) // self.block_shape[
                0
            ]
            ceil_kv_pad = (
                self.padded_s2 + self.block_shape[1] - 1
            ) // self.block_shape[1]
            pad_q = max(0, ceil_q_pad - self.block_sparse_mask_np.shape[2])
            pad_kv = max(0, ceil_kv_pad - self.block_sparse_mask_np.shape[3])
            if pad_q > 0 or pad_kv > 0:
                self.block_sparse_mask_np = np.pad(
                    self.block_sparse_mask_np,
                    ((0, 0), (0, 0), (0, pad_q), (0, pad_kv)),
                    constant_values=0,
                )

        self.block_sparse_mask_tensor = numpy_array_to_tensor_mxfp4(
            self.block_sparse_mask_np
        )

        # 打包 QKV 为 MXFP4 格式（NPU 侧输入）
        if self.q_input_layout == "TND":
            q_p_ch, k_p_ch, v_p_ch = [], [], []
            q_s_ch, k_s_ch, v_s_ch = [], [], []
            for bi in range(self.batch):
                q_b = self.q_parts[bi]
                k_b = self.k_parts[bi]
                v_b = self.v_parts[bi]
                q_p, q_e = mxfp4_quantize_pack_last(
                    q_b, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx
                )
                k_p, k_e = mxfp4_quantize_pack_last(
                    k_b, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx
                )
                v_p, v_e = mxfp4_quantize_pack_last(
                    v_b, quant_axis=-2, quant_mode=self.quant_mode, log2_cx=self.log2_cx
                )
                q_p_ch.append(q_p.permute(0, 2, 1, 3).flatten(0, 1))
                k_p_ch.append(k_p.permute(0, 2, 1, 3).flatten(0, 1))
                v_p_ch.append(v_p.permute(0, 2, 1, 3).flatten(0, 1))
                q_s_ch.append(
                    _e8m0_qk_to_v3_mxfp4(q_e).permute(0, 2, 1, 3, 4).flatten(0, 1)
                )
                k_s_ch.append(
                    _e8m0_qk_to_v3_mxfp4(k_e).permute(0, 2, 1, 3, 4).flatten(0, 1)
                )
                v_s_ch.append(
                    _e8m0_v_to_v3_mxfp4(v_e).permute(0, 2, 1, 3, 4).flatten(0, 1)
                )
            self.q_packed = torch.cat(q_p_ch, dim=0).contiguous()
            self.k_packed = torch.cat(k_p_ch, dim=0).contiguous()
            self.v_packed = torch.cat(v_p_ch, dim=0).contiguous()
            self.q_scale = torch.cat(q_s_ch, dim=0).contiguous()
            self.k_scale = torch.cat(k_s_ch, dim=0).contiguous()
            self.v_scale = torch.cat(v_s_ch, dim=0).contiguous()
        else:
            # per-batch 实际长度数据 0-pad 到张量 S 维后 cat 成 BNSD [B, N, S_padded, D]
            q_bnsd = torch.cat(
                [
                    torch.nn.functional.pad(
                        self.q_parts[bi], (0, 0, 0, self.padded_s1 - q_s_list[bi])
                    )
                    for bi in range(self.batch)
                ],
                dim=0,
            ).contiguous()
            k_bnsd = torch.cat(
                [
                    torch.nn.functional.pad(
                        self.k_parts[bi], (0, 0, 0, self.padded_s2 - kv_s_list[bi])
                    )
                    for bi in range(self.batch)
                ],
                dim=0,
            ).contiguous()
            v_bnsd = torch.cat(
                [
                    torch.nn.functional.pad(
                        self.v_parts[bi], (0, 0, 0, self.padded_s2 - kv_s_list[bi])
                    )
                    for bi in range(self.batch)
                ],
                dim=0,
            ).contiguous()

            q_packed_t, q_e8m0 = mxfp4_quantize_pack_last(
                q_bnsd, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx
            )
            k_packed_t, k_e8m0 = mxfp4_quantize_pack_last(
                k_bnsd, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx
            )
            v_packed_t, v_e8m0 = mxfp4_quantize_pack_last(
                v_bnsd, quant_axis=-2, quant_mode=self.quant_mode, log2_cx=self.log2_cx
            )
            self.q_scale = _e8m0_qk_to_v3_mxfp4(q_e8m0)
            self.k_scale = _e8m0_qk_to_v3_mxfp4(k_e8m0)
            self.v_scale = _e8m0_v_to_v3_mxfp4(v_e8m0)

            if self.q_input_layout == "BSND":
                self.q_packed = q_packed_t.permute(0, 2, 1, 3).contiguous()
                self.k_packed = k_packed_t.permute(0, 2, 1, 3).contiguous()
                self.v_packed = v_packed_t.permute(0, 2, 1, 3).contiguous()
                self.q_scale = self.q_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.k_scale = self.k_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.v_scale = self.v_scale.permute(0, 2, 1, 3, 4).contiguous()
            else:
                self.q_packed = q_packed_t
                self.k_packed = k_packed_t
                self.v_packed = v_packed_t

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _logging.info(
            f"=======q_input_layout:{self.q_input_layout};self.q_packed:{self.q_packed.shape};q_seqlen_list:{self.q_seqlen_list};;self.k_packed:{self.k_packed.shape};kv_seqlen_list:{self.kv_seqlen_list};padded_s1:{self.padded_s1};padded_s2:{self.padded_s2};========"
        )
        if self.device == "cpu":
            testObj = TestBlockSparseAttentionMxfp4GenBmGt()
            aux_attrs = testObj.AuxAttrs(
                self.num_heads,
                self.kv_heads,
                self.headDim,
                self.inner_precise,
                self.scale_value,
                self.q_input_layout,
                self.kv_input_layout,
                self.attention_out_dtype_str,
                self.quant_mode,
                self.dst_type_max,
            )

            attention_inputs = testObj.AttentionInputs(
                self.q_parts,
                self.k_parts,
                self.v_parts,
                self.block_sparse_mask_np,
                self.block_shape,
                self.q_seqlen_list,
                self.kv_seqlen_list,
                self.padded_s1,
                aux_attrs,
            )

            if self.task_result.is_benchmark_task:
                result = testObj.compute_output_benchmark(attention_inputs)
            else:
                result = testObj.compute_output(attention_inputs)
            return result.to(torch.float32)

        raise ValueError(
            f"不支持的 device: {self.device}; pyaclnn 节点请使用 "
            f"aclnn_api_type=pyaclnn_aclnn_blocksparseattention_v3"
        )


# =====================================================================
# pyaclnn 节点: 直调两段式 aclnnBlockSparseAttentionV3
# =====================================================================
def _bsa_out_shape(prep):
    """attentionOut 的 shape, 与 query 的 layout 对齐(doc L148-L152)。"""
    if prep.q_input_layout == "TND":
        return (sum(prep.q_seqlen_list), prep.num_heads, prep.headDim)
    if prep.q_input_layout == "BNSD":
        return (prep.batch, prep.num_heads, prep.padded_s1, prep.headDim)
    return (prep.batch, prep.padded_s1, prep.num_heads, prep.headDim)


@register("pyaclnn_aclnn_blocksparseattention_v3")
class PyAclnnBlockSparseAttentionV3Api_mxfp4(AclnnBaseApi):
    """按 C++ 原型逐个位置自己组装入参, 不走 ATK 的 kwargs->convert 通道。

    这么做的原因: MXFP4 的 query/key/value 缓冲区是 uint8 nibble-pack, 而算子要按
    FLOAT4_E2M1 看到两倍长的最后一维; dequantScale 又必须是 FLOAT8_E8M0。这两类
    "storage dtype 与逻辑 dtype 不一致"的入参没法靠 kwargs 的 dtype 表达, 只能自己
    建 aclTensor(与 Gmm_mxa8w4 的 aclnn 工程同一思路)。同时原型里还有 kwargs 侧不存在
    的 pQuantScale, 位置也只能显式补 nullptr。
    """

    def init_by_input_data(self, input_data: InputDataset):
        prep = self._get_prepared(input_data)
        self.prep = prep

        out_shape = _bsa_out_shape(prep)
        self.out_npu = torch.zeros(
            out_shape, dtype=prep.attention_out_torch_dtype, device="npu"
        )
        out_pack = _create_acl_tensor_mxfp4(
            self.out_npu,
            TORCH_TO_ACLTYPE[str(prep.attention_out_torch_dtype)],
            out_shape,
        )

        act_q = list(prep.q_seqlen_list) if prep.pass_act_seq else None
        act_kv = list(prep.kv_seqlen_list) if prep.pass_act_seq else None

        # 顺序严格对齐 docs/aclnnBlockSparseAttentionV3.md 的一段式原型
        input_args = [
            create_fp4_acl_tensor(prep.q_packed, prep.headDim),
            create_fp4_acl_tensor(prep.k_packed, prep.headDim),
            create_fp4_acl_tensor(prep.v_packed, prep.headDim),
            nnopbase.create_acl_tensor(
                prep.block_sparse_mask_tensor.contiguous().npu(),
                AclFormat.ACL_FORMAT_ND,
            ),
            TensorPtr(),  # attenMaskOptional: 仅支持 nullptr
            create_int64_array_mxfp4(list(prep.block_shape)),
            create_int64_array_mxfp4(act_q),
            create_int64_array_mxfp4(act_kv),
            TensorPtr(),  # blockTableOptional: 仅支持 nullptr
            create_e8m0_acl_tensor(prep.q_scale),
            create_e8m0_acl_tensor(prep.k_scale),
            create_e8m0_acl_tensor(prep.v_scale),
            TensorPtr(),  # pQuantScaleOptional
            ctypes.c_char_p(prep.q_input_layout.encode("utf-8")),
            ctypes.c_char_p(prep.kv_input_layout.encode("utf-8")),
            Int64(prep.kv_heads),
            Int64(0),  # maskType: 仅支持 0
            ctypes.c_double(prep.scale_value),
            Int64(prep.inner_precise),
            Int64(0),  # blockSize: 仅支持 0
            Int64(_NOT_LIMITED),  # preTokens
            Int64(_NOT_LIMITED),  # nextTokens
            Int64(0),  # softmaxLseFlag: MXFP4 不支持 lse
            Int64(prep.quant_mode),
            ctypes.c_double(prep.dst_type_max),
            out_pack,
            TensorPtr(),  # softmaxLseOptional
        ]

        _logging.info(
            f"pyaclnn BSA V3 MXFP4 | layout={prep.q_input_layout} "
            f"q={tuple(prep.q_packed.shape)} k={tuple(prep.k_packed.shape)} "
            f"out={out_shape} quantMode={prep.quant_mode} dstTypeMax={prep.dst_type_max} "
            f"actSeq={'on' if prep.pass_act_seq else 'off'}"
        )

        torch.npu.synchronize()
        return input_args, [out_pack]

    def _get_prepared(self, input_data: InputDataset):
        """拿到造好 MXFP4 数据的 golden 侧对象。

        造数只由用例参数派生(case_derived_seed), 所以 cpu 与 pyaclnn 两个节点各自造出的
        数据逐位一致。后端已经先对 api_executor 调过一次 init_by_input_data, 能复用就复用,
        省掉一遍量化打包。
        """
        cached = getattr(self.backend, "api_executor", None)
        if isinstance(cached, BlockSparseAttentionV3Api_mxfp4) and hasattr(
            cached, "q_packed"
        ):
            return cached
        prep = BlockSparseAttentionV3Api_mxfp4(self.task_result)
        prep.init_by_input_data(input_data)
        return prep

    def after_call(self, output_packages):
        prep = self.prep
        out = self.out_npu.cpu().to(torch.float32)
        # BNSD/BSND 变长: kernel 不写 [actualSeqLen, S_padded), golden 侧该区间为 0
        if prep.padded_s1 is not None:
            for bi in range(prep.batch):
                act_q = prep.q_seqlen_list[bi]
                if act_q >= prep.padded_s1:
                    continue
                if prep.q_input_layout == "BNSD":
                    out[bi, :, act_q:] = 0
                else:
                    out[bi, act_q:] = 0
        return [out]

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
            "const aclTensor *query, "
            "const aclTensor *key, "
            "const aclTensor *value, "
            "const aclTensor *blockSparseMaskOptional, "
            "const aclTensor *attenMaskOptional, "
            "const aclIntArray *blockShapeOptional, "
            "const aclIntArray *actualSeqLengthsOptional, "
            "const aclIntArray *actualSeqLengthsKvOptional, "
            "const aclTensor *blockTableOptional, "
            "const aclTensor *qDequantScaleOptional, "
            "const aclTensor *kDequantScaleOptional, "
            "const aclTensor *vDequantScaleOptional, "
            "const aclTensor *pQuantScaleOptional, "
            "char *qInputLayout, "
            "char *kvInputLayout, "
            "int64_t numKeyValueHeads, "
            "int64_t maskType, "
            "double scaleValue, "
            "int64_t innerPrecise, "
            "int64_t blockSize, "
            "int64_t preTokens, "
            "int64_t nextTokens, "
            "int64_t softmaxLseFlag, "
            "int64_t quantMode, "
            "double dstTypeMax, "
            "aclTensor *attentionOut, "
            "aclTensor *softmaxLseOptional, "
            "uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )


# ----------------------------------------------------------------------
# MXFP8 参考实现（原 bsa_golden_fullquant_mxfp8.py）
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""BSA MXFP8 OCP (quantMode=4) ATK golden + pyaclnn V3 packing.

Actual 切块与 MXFP4 ATK golden 同构: 选中的 Y 块每 2 个拼一段, P-snap 后立刻
corr + PV + rescale。Q/K/V 仍走 e4m3+e8m0 qdq; P 仍走 e4m3 snap
(m=(K-7.8046875)·ln2, corr 按 emax=8)。lse = m_run·ln2 + ln(ℓ) - Δ·ln2（Δ=7.8046875-8）。
双标杆:
  - is_benchmark_task=False: 上述 snap golden
  - is_benchmark_task=True: 量化前 fp32 QKV 标准 BSA（无 qdq / 无 P-snap）
QKV 逻辑 dtype 是 FLOAT8_E4M3FN; scale 是 FLOAT8_E8M0。CPU golden 在
head 维上 batched matmul; P-snap 对整块 score 做一次 e4m3 round-trip,
不再按 head 循环调 ``tensor.to(float8)``。不注册 ATK 名,
由 bsa_golden.py 按 quantMode=4 委派。
"""


_logging = Logger().get_logger()
TORCH_TO_ACLTYPE.setdefault("torch.float8_e4m3fn", 36)
try:
    ensure_atk_float8_e8m0_support()
except Exception as exc:
    _logging.warning("MXFP8 e8m0 dtype map skipped: %s", exc)

QUANT_MODE_MXFP8_OCP = 4
E8M0_BIAS_mxfp8 = 127
FP8_E4M3_EMAX = 8
FP8_E4M3_MAX = 448.0
P_SNAP_EMAX = 7.8046875
MX_GROUP = 32
_S_DTYPE_MAP_mxfp8 = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_OUT_DTYPE_MAP_mxfp8 = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _acl_fp8_e4m3():
    return getattr(AclDataType, "ACL_FLOAT8_E4M3FN", 36)


def create_fp8_acl_tensor(tensor: torch.Tensor):
    if tensor.dtype != torch.float8_e4m3fn:
        tensor = tensor.view(torch.uint8).view(torch.float8_e4m3fn)
    u8 = tensor.view(torch.uint8).contiguous()
    npu_t = u8.npu().view(torch.float8_e4m3fn)
    return _create_acl_tensor_mxfp4(
        npu_t, _acl_fp8_e4m3(), tuple(int(x) for x in npu_t.shape)
    )


def mxfp8_quantize_native(tensor, quant_axis, group_size=MX_GROUP):
    nd = tensor.dim()
    qax = quant_axis % nd
    t = tensor.movedim(qax, -1).contiguous().float()
    *lead, d = t.shape
    ng = math.ceil(d / group_size)
    pad = ng * group_size - d
    x = F.pad(t, (0, pad)) if pad else t
    g = x.reshape(*lead, ng, group_size)
    all_zero = torch.all(g == 0, dim=-1)
    max_vals = torch.max(torch.abs(g), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - FP8_E4M3_EMAX
    two = torch.tensor(2.0, dtype=torch.float32, device=t.device)
    scale = torch.where(
        all_zero, torch.ones_like(shared_exp), torch.pow(two, shared_exp)
    )
    q = (
        (g / scale.unsqueeze(-1))
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    q_out = q.reshape(*lead, ng * group_size)[..., :d].movedim(-1, qax).contiguous()
    dq = (q.float() * scale.unsqueeze(-1)).reshape(*lead, ng * group_size)[..., :d]
    dq = dq.movedim(-1, qax).contiguous()
    e8m0 = (shared_exp.round().long() + E8M0_BIAS_mxfp8).clamp(0, 254).to(torch.uint8)
    e8m0 = e8m0.movedim(-1, qax).contiguous()
    return q_out, e8m0, dq


def mxfp8_qdq(t, axis, group_size=MX_GROUP):
    """QKV mxfp8 qdq for CPU golden. Same scale as ``mxfp8_quantize_native``,
    element rounding via software e4m3 (matches ``.to(float8_e4m3fn).float()``).
    """
    nd = t.dim()
    qax = axis % nd
    x = t.movedim(qax, -1).contiguous().float()
    *lead, d = x.shape
    ng = math.ceil(d / group_size)
    pad = ng * group_size - d
    g = F.pad(x, (0, pad)) if pad else x
    g = g.reshape(*lead, ng, group_size)
    all_zero = torch.all(g == 0, dim=-1)
    max_vals = torch.max(torch.abs(g), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - FP8_E4M3_EMAX
    two = torch.tensor(2.0, dtype=torch.float32, device=t.device)
    scale = torch.where(
        all_zero, torch.ones_like(shared_exp), torch.pow(two, shared_exp)
    )
    q_fp = _e4m3fn_round_trip_fp32(g / scale.unsqueeze(-1))
    dq = (q_fp * scale.unsqueeze(-1)).reshape(*lead, ng * group_size)[..., :d]
    return dq.movedim(-1, qax).contiguous()


def _e8m0_qk_to_v3_mxfp8(e8m0_uint8):
    *lead, nd = e8m0_uint8.shape
    paired = e8m0_uint8.reshape(*lead, nd // 2, 2)
    return paired.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3_mxfp8(e8m0_uint8):
    bsz, n2, nb, dim = e8m0_uint8.shape
    if nb % 2 != 0:
        pad_block = torch.full((bsz, n2, 1, dim), E8M0_BIAS_mxfp8, dtype=torch.uint8)
        e8m0_uint8 = torch.cat([e8m0_uint8, pad_block], dim=2)
        nb += 1
    return (
        e8m0_uint8.reshape(bsz, n2, nb // 2, 2, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )


_E4M3_FP8_MAX_BITS_mxfp8 = np.uint32(1087 << 20)  # 480.0f, first value not in e4m3fn
_E4M3_DENORM_MASK_mxfp8 = np.uint32(141 << 23)
_E4M3_DENORM_THRESH_mxfp8 = np.uint32(121 << 23)  # 2^{-6}
_E4M3_BIAS_ROUND_mxfp8 = np.uint32((((7 - 127) << 23) + 0x7FFFF) & 0xFFFFFFFF)
_E4M3_SIGN_mxfp8 = np.uint32(0x80000000)


def _e4m3fn_round_trip_fp32(x: torch.Tensor) -> torch.Tensor:
    """Bit-exact ``clamp(-448, 448).to(float8_e4m3fn).float()`` without CPU float8.

    Same rounding as torch ``c10::detail::fp8e4m3fn_from_fp32_value`` (RNE).
    CPU ``tensor.to(float8_e4m3fn)`` is ~200x slower than this numpy path.
    """
    orig_shape = tuple(x.shape)
    y = np.ascontiguousarray(x.detach().float().cpu().numpy(), dtype=np.float32)
    y = np.clip(y, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    u = y.view(np.uint32)
    sign = u & _E4M3_SIGN_mxfp8
    f_bits = u ^ sign
    overflow = f_bits >= _E4M3_FP8_MAX_BITS_mxfp8
    denorm = f_bits < _E4M3_DENORM_THRESH_mxfp8
    denorm_f = f_bits.view(np.float32) + np.float32(16384.0)
    small_result = denorm_f.view(np.uint32) - _E4M3_DENORM_MASK_mxfp8
    mant_odd = (f_bits >> np.uint32(20)) & np.uint32(1)
    normal_result = (f_bits + _E4M3_BIAS_ROUND_mxfp8 + mant_odd) >> np.uint32(20)
    result = np.where(
        overflow, np.uint32(0x7F), np.where(denorm, small_result, normal_result)
    )
    result = (result | (sign >> np.uint32(24))) & np.uint32(0xFF)
    s = (result >> np.uint32(7)) & np.uint32(1)
    e = (result >> np.uint32(3)) & np.uint32(0xF)
    m = result & np.uint32(0x7)
    is_nan = (e == 15) & (m == 7)
    is_zero = (e == 0) & (m == 0)
    is_den = (e == 0) & (m != 0)
    e_f = e.astype(np.float32)
    m_f = m.astype(np.float32)
    val = np.where(
        is_zero,
        np.float32(0.0),
        np.where(
            is_den,
            m_f * np.float32(2.0**-9),
            (np.float32(1.0) + m_f * np.float32(0.125))
            * np.exp2(e_f - np.float32(7.0)),
        ),
    )
    val = np.where(s.astype(bool), -val, val)
    val = np.where(is_nan, np.float32("nan"), val).astype(np.float32, copy=False)
    out = torch.from_numpy(np.ascontiguousarray(val)).view(orig_shape)
    return out.to(device=x.device, dtype=torch.float32)


def _packed_selected_ids(mask_bn_ck: torch.Tensor):
    """Increasing selected Y-block ids per row, padded. Matches the old list-comp order."""
    bn, ck = mask_bn_ck.shape
    if ck == 0 or bn == 0:
        empty_i = mask_bn_ck.new_empty((bn, 0), dtype=torch.long)
        empty_b = mask_bn_ck.new_empty((bn, 0), dtype=torch.bool)
        return empty_i, empty_b
    sel_mask = mask_bn_ck != 0
    max_sel = int(sel_mask.sum(dim=-1).max().item())
    if max_sel == 0:
        empty_i = mask_bn_ck.new_zeros((bn, 0), dtype=torch.long)
        empty_b = mask_bn_ck.new_zeros((bn, 0), dtype=torch.bool)
        return empty_i, empty_b
    idx = torch.arange(ck, device=mask_bn_ck.device).expand(bn, ck)
    keyed = torch.where(sel_mask, idx, idx.new_full(idx.shape, ck))
    packed = torch.sort(keyed, dim=-1).values[:, :max_sel]
    valid = packed < ck
    return packed.clamp(max=max(ck - 1, 0)), valid


def _gather_kv_blocks(k_flat, v_flat, block_ids, block_valid, block_y, s2):
    """Gather Y-blocks per row. k/v: [BN, S2, D], ids/valid: [BN, nb] -> [BN, npos, D]."""
    bn, nb = block_ids.shape
    dim = k_flat.shape[-1]
    if nb == 0 or s2 <= 0:
        empty = k_flat.new_zeros((bn, 0, dim))
        return empty, empty, k_flat.new_zeros((bn, 0), dtype=torch.bool)
    offs = torch.arange(block_y, device=k_flat.device)
    pos = block_ids.unsqueeze(-1) * block_y + offs
    in_range = (pos >= 0) & (pos < s2)
    tok_valid = block_valid.unsqueeze(-1) & in_range
    pos_f = pos.clamp(0, s2 - 1).reshape(bn, -1)
    valid_f = tok_valid.reshape(bn, -1)
    idx = pos_f.unsqueeze(-1).expand(-1, -1, dim)
    k_blk = torch.gather(k_flat, 1, idx)
    v_blk = torch.gather(v_flat, 1, idx)
    zero = k_blk.new_zeros(())
    k_blk = torch.where(valid_f.unsqueeze(-1), k_blk, zero)
    v_blk = torch.where(valid_f.unsqueeze(-1), v_blk, zero)
    return k_blk, v_blk, valid_f


def _blockwise_snap_local_raw(score, seqk_dim=0, mx_block_size=32, seqk_valid=None):
    neg_inf = float("-inf")
    device = score.device
    src_dtype = score.dtype
    ndim = score.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = score.shape[skp]
    ln2 = math.log(2.0)
    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(score.shape)
        pshape[skp] = pad
        s_padded = torch.cat(
            [score, torch.full(pshape, neg_inf, dtype=src_dtype, device=device)],
            dim=skp,
        )
    else:
        s_padded = score
    s_moved = s_padded.movedim(skp, -1)
    other = s_moved.shape[:-1]
    n_blk = s_moved.shape[-1] // mx_block_size
    s_r = s_moved.reshape(*other, n_blk, mx_block_size)
    m_block_raw = s_r.max(dim=-1).values
    m_block_raw_safe = torch.where(
        torch.isinf(m_block_raw) & (m_block_raw < 0),
        torch.zeros_like(m_block_raw),
        m_block_raw,
    )
    inv_ln2 = torch.tensor(1.0 / ln2, dtype=torch.float16, device=device)
    ln2_fp16 = torch.tensor(ln2, dtype=torch.float16, device=device)
    k_block = torch.floor(m_block_raw_safe * inv_ln2)
    snap_emax = torch.tensor(P_SNAP_EMAX, dtype=torch.float16, device=device)
    m_block_snap = (k_block - snap_emax) * ln2_fp16
    p_local = torch.exp(s_r.float() - m_block_snap.float().unsqueeze(-1))
    p_q = _e4m3fn_round_trip_fp32(p_local).to(src_dtype)
    k_ij = k_block.max(dim=-1).values.float()
    if seqk_valid is not None:
        v = seqk_valid
        if pad > 0:
            pshape = list(v.shape)
            pshape[skp] = pad
            v = torch.cat(
                [v, torch.zeros(pshape, dtype=torch.bool, device=device)], dim=skp
            )
        v_r = v.movedim(skp, -1).reshape(*other, n_blk, mx_block_size)
        valid_blk = v_r.any(dim=-1)
        k_ij = (
            torch.where(valid_blk, k_block, k_block.new_full((), float("-inf")))
            .max(dim=-1)
            .values.float()
        )
    p_q_moved = p_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(
        -1, skp
    )
    return p_q_moved, k_block, k_ij


def _pscale_corr(p_q_moved, k_block, k_g, seqk_dim=0, mx_block_size=32):
    skp = seqk_dim if seqk_dim >= 0 else p_q_moved.dim() + seqk_dim
    seqk_len = p_q_moved.shape[skp]
    k_new_s = k_g.to(p_q_moved.dtype)
    k_diff = k_block - k_new_s.unsqueeze(-1)
    corr = torch.exp2(k_diff - float(FP8_E4M3_EMAX))
    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(
        -1, skp
    )
    return p_q_moved * corr_broad


def _blockwise_snap_local_quantize_p_eff_mxfp8(
    score,
    m_running,
    seqk_dim=0,
    mx_block_size=32,
    seqk_valid=None,
    first_mask=None,
):
    """与 MXFP4 `_blockwise_snap_local_quantize_p_eff_mxfp8` 同控制流: 本 chunk 的 K_ij
    立刻和 running max 取 max, 再 corr。P 量化仍是 e4m3 + emax=8。

    first_mask: per-row bool, True = 该行尚未见过有效 chunk, 等价旧循环 m_run is None。
    seqk_valid: 与 score 同形的 bool, 全无效的 32-block 不参与 K_ij, 避免 packed pad 把 K_ij 抬成 0。
    """
    p_q, k_block, k_ij = _blockwise_snap_local_raw(
        score, seqk_dim=seqk_dim, mx_block_size=mx_block_size, seqk_valid=seqk_valid
    )
    if m_running is None:
        m_new = k_ij
    elif first_mask is not None:
        fm = (
            first_mask.unsqueeze(-1)
            if first_mask.dim() + 1 == k_ij.dim()
            else first_mask
        )
        m_new = torch.where(fm, k_ij, torch.maximum(m_running, k_ij))
    else:
        m_new = torch.maximum(m_running, k_ij)
    p_eff = _pscale_corr(
        p_q, k_block, m_new, seqk_dim=seqk_dim, mx_block_size=mx_block_size
    )
    return p_eff, m_new


def _flatten_bn_kv(k, v, n_q, group_size):
    """k/v [B, Nkv, S, D] -> [B*Nq, S, D] with GQA broadcast onto query heads."""
    bsz = k.shape[0]
    n2_idx = (
        (torch.arange(n_q, device=k.device) // group_size).view(1, n_q).expand(bsz, n_q)
    )
    b_idx = torch.arange(bsz, device=k.device).view(bsz, 1).expand(bsz, n_q)
    k_flat = k[b_idx.reshape(-1), n2_idx.reshape(-1)]
    v_flat = v[b_idx.reshape(-1), n2_idx.reshape(-1)]
    return k_flat, v_flat


def cpu_block_sparse_attention_bnsd_mxfp8(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    scale_value,
    num_kv_heads,
    s_dtype="fp16",
    out_dtype="bfloat16",
    return_lse=False,
):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    bsz, n_q, s1, dim = q.shape
    n2 = k.shape[1]
    s2 = k.shape[2]
    group_size = n_q // n2
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    s_torch_dtype = _S_DTYPE_MAP_mxfp8[s_dtype]
    q_q = mxfp8_qdq(q, axis=-1)
    k_q = mxfp8_qdq(k, axis=-1)
    v_q = mxfp8_qdq(v, axis=2)
    out = torch.zeros((bsz, n_q, s1, dim), dtype=torch.float32)
    lse = torch.full((bsz, n_q, s1), float("-inf"), dtype=torch.float32)
    ln2 = math.log(2.0)
    scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype, device=q.device)
    mask = torch.as_tensor(block_sparse_mask, device=q.device)
    bn = bsz * n_q
    q_flat = q_q.reshape(bn, s1, dim)
    k_flat, v_flat = _flatten_bn_kv(k_q, v_q, n_q, group_size)
    for qb in range(ceil_q):
        q_lo = qb * block_x
        q_hi = min(q_lo + block_x, s1)
        if q_lo >= s1:
            break
        nq = q_hi - q_lo
        q_blk = q_flat[:, q_lo:q_hi]
        ids, valid = _packed_selected_ids(mask[:, :, qb, :].reshape(bn, ceil_kv))
        max_sel = ids.shape[1]
        if max_sel == 0:
            continue
        m_run = q_flat.new_zeros((bn, nq))
        l_run = q_flat.new_zeros((bn, nq))
        acc = q_flat.new_zeros((bn, nq, dim))
        seen = torch.zeros(bn, dtype=torch.bool, device=q.device)
        for c0 in range(0, max_sel, 2):
            nb = min(2, max_sel - c0)
            k_blk, v_blk, tok_valid = _gather_kv_blocks(
                k_flat,
                v_flat,
                ids[:, c0 : c0 + nb],
                valid[:, c0 : c0 + nb],
                block_y,
                s2,
            )
            head_has = tok_valid.any(dim=-1)
            if not bool(head_has.any()):
                continue
            s_fp32 = torch.matmul(k_blk, q_blk.transpose(-1, -2)).to(torch.float32)
            s_fp32 = s_fp32.masked_fill(~tok_valid.unsqueeze(-1), float("-inf"))
            s_s = (s_fp32 * scale_s).to(s_torch_dtype)
            first = ~seen
            p_eff_s, m_new = _blockwise_snap_local_quantize_p_eff_mxfp8(
                s_s,
                m_running=m_run,
                seqk_dim=1,
                seqk_valid=tok_valid.unsqueeze(-1).expand_as(s_s),
                first_mask=first,
            )
            # First valid chunk per row matches the old `m_run is None` path (alpha=0).
            # Rows with no tokens this chunk are a no-op (alpha=1, p=0).
            alpha = torch.where(
                first.unsqueeze(-1),
                torch.zeros_like(m_new),
                torch.exp2(m_run.float() - m_new.float()),
            )
            alpha = torch.where(head_has.unsqueeze(-1), alpha, torch.ones_like(alpha))
            p_fp32 = torch.nan_to_num(p_eff_s, nan=0.0).float()
            p_fp32 = torch.where(
                tok_valid.unsqueeze(-1), p_fp32, torch.zeros_like(p_fp32)
            )
            p_fp32 = p_fp32 * head_has.view(bn, 1, 1)
            l_run = alpha * l_run + p_fp32.sum(dim=1)
            acc = alpha.unsqueeze(-1) * acc + torch.matmul(
                p_fp32.transpose(-1, -2), v_blk
            )
            m_run = torch.where(head_has.unsqueeze(-1), m_new.float(), m_run)
            seen = seen | head_has
        l_safe = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
        out_blk = (acc / l_safe.unsqueeze(-1)).view(bsz, n_q, nq, dim)
        out[:, :, q_lo:q_hi] = out_blk
        ln_l = torch.where(
            l_run > 0, torch.log(l_run), l_run.new_full(l_run.shape, float("-inf"))
        )
        lse_blk = torch.where(
            l_run > 0,
            ln_l + m_run * ln2 + (FP8_E4M3_EMAX - P_SNAP_EMAX) * ln2,
            l_run.new_full(l_run.shape, float("-inf")),
        )
        lse_blk = torch.where(
            seen.unsqueeze(-1), lse_blk, lse_blk.new_full(lse_blk.shape, float("-inf"))
        )
        lse[:, :, q_lo:q_hi] = lse_blk.view(bsz, n_q, nq)
    out = out.to(_OUT_DTYPE_MAP_mxfp8[out_dtype])
    if return_lse:
        return out, lse
    return out


def cpu_block_sparse_attention_bnsd_fp32_standard(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    scale_value,
    num_kv_heads,
    return_lse=False,
):
    """ATK 双标杆真值腿：量化前 fp32 QKV + 标准 softmax，无 mxfp8 qdq / P-snap。

    与 MXFP4 `compute_output_benchmark` 同构：把选中的 Y 块拼成一段 KV 再做一次
    fp32 attention。lse 为该 mask 下的 logsumexp。
    """
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    bsz, n_q, s1, dim = q.shape
    n2 = k.shape[1]
    s2 = k.shape[2]
    group_size = n_q // n2
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    scale = float(scale_value)
    out = torch.zeros((bsz, n_q, s1, dim), dtype=torch.float32)
    lse = torch.full((bsz, n_q, s1), float("-inf"), dtype=torch.float32)
    mask = torch.as_tensor(block_sparse_mask, device=q.device)
    bn = bsz * n_q
    q_flat = q.reshape(bn, s1, dim)
    k_flat, v_flat = _flatten_bn_kv(k, v, n_q, group_size)
    for qb in range(ceil_q):
        q_lo = qb * block_x
        q_hi = min(q_lo + block_x, s1)
        if q_lo >= s1:
            break
        nq = q_hi - q_lo
        ids, valid = _packed_selected_ids(mask[:, :, qb, :].reshape(bn, ceil_kv))
        if ids.shape[1] == 0:
            continue
        k_blk, v_blk, tok_valid = _gather_kv_blocks(
            k_flat, v_flat, ids, valid, block_y, s2
        )
        q_blk = q_flat[:, q_lo:q_hi]
        s = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * scale
        s = s.masked_fill(~tok_valid.unsqueeze(1), float("-inf"))
        row_max = s.max(dim=-1, keepdim=True).values
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        p = torch.exp(s - row_max)
        p = torch.where(tok_valid.unsqueeze(1), p, torch.zeros_like(p))
        row_sum = p.sum(dim=-1, keepdim=True)
        p = p / torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
        out_blk = torch.matmul(p, v_blk)
        out[:, :, q_lo:q_hi] = out_blk.view(bsz, n_q, nq, dim)
        lse_blk = row_max.squeeze(-1) + torch.log(
            torch.where(
                row_sum.squeeze(-1) > 0,
                row_sum.squeeze(-1),
                row_sum.new_ones(row_sum.shape[:-1]),
            )
        )
        empty = ~tok_valid.any(dim=-1)
        lse_blk = torch.where(
            empty.unsqueeze(-1), lse_blk.new_full(lse_blk.shape, float("-inf")), lse_blk
        )
        lse[:, :, q_lo:q_hi] = lse_blk.view(bsz, n_q, nq)
    if return_lse:
        return out, lse
    return out


def _assemble_layout(
    chunks,
    lse_chunks,
    q_layout,
    batch,
    num_heads,
    dim,
    q_seqlen_list,
    padded_s1,
    want_lse,
):
    if q_layout == "TND":
        out = torch.cat(
            [c.permute(0, 2, 1, 3).flatten(0, 1) for c in chunks], dim=0
        ).to(torch.float32)
        if not want_lse:
            return out
        lse = torch.cat(
            [c.permute(0, 2, 1).flatten(0, 1).unsqueeze(-1) for c in lse_chunks], dim=0
        )
        return out, lse
    # BNSD/BSND: [actualSeqLen, S) 是 padding。kernel 不写这块，有效行里空 KV 的 LSE 仍是 -inf。
    # padding 的 attention 和 LSE 两侧都置 0 再比对。
    if q_layout == "BNSD":
        out_s = padded_s1 if padded_s1 is not None else max(q_seqlen_list)
        out = torch.zeros(as_shape((batch, num_heads, out_s, dim)), dtype=torch.float32)
        lse = torch.zeros((batch, num_heads, out_s, 1), dtype=torch.float32)
        for bi in range(batch):
            slen = q_seqlen_list[bi]
            out[bi, :, :slen] = chunks[bi][0].float()
            if want_lse:
                lse[bi, :, :slen, 0] = lse_chunks[bi][0]
        return (out, lse) if want_lse else out
    out_s = padded_s1 if padded_s1 is not None else max(q_seqlen_list)
    out = torch.zeros(as_shape((batch, out_s, num_heads, dim)), dtype=torch.float32)
    lse = torch.zeros((batch, out_s, num_heads, 1), dtype=torch.float32)
    for bi in range(batch):
        slen = q_seqlen_list[bi]
        out[bi, :slen] = chunks[bi][0].permute(1, 0, 2).float()
        if want_lse:
            lse[bi, :slen, :, 0] = lse_chunks[bi][0].permute(1, 0)
    return (out, lse) if want_lse else out


def _lse_shape(prep):
    if prep.q_input_layout == "TND":
        return (sum(prep.q_seqlen_list), prep.num_heads, 1)
    if prep.q_input_layout == "BNSD":
        return (prep.batch, prep.num_heads, prep.padded_s1, 1)
    return (prep.batch, prep.padded_s1, prep.num_heads, 1)


def _out_shape(prep):
    if prep.q_input_layout == "TND":
        return (sum(prep.q_seqlen_list), prep.num_heads, prep.headDim)
    if prep.q_input_layout == "BNSD":
        return (prep.batch, prep.num_heads, prep.padded_s1, prep.headDim)
    return (prep.batch, prep.padded_s1, prep.num_heads, prep.headDim)


class BlockSparseAttentionV3Api_mxfp8(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        self.kv_heads = int(input_data.kwargs["numKeyValueHeads"])
        self.scale_value = float(input_data.kwargs["scaleValue"])
        self.inner_precise = int(input_data.kwargs["innerPrecise"])
        self.softmax_lse_flag = int(input_data.kwargs["softmaxLseFlag"])
        self.quant_mode = int(input_data.kwargs.get("quantMode", QUANT_MODE_MXFP8_OCP))
        self.dst_type_max = float(input_data.kwargs.get("dstTypeMax", 0.0))
        self.block_shape = [int(v) for v in input_data.kwargs["blockShape"]]
        raw_act_q = input_data.kwargs["actualSeqLengths"]
        raw_act_kv = input_data.kwargs["actualSeqLengthsKv"]
        self.pass_act_seq = _act_seq_given(raw_act_q) and _act_seq_given(raw_act_kv)

        attention_out = input_data.kwargs["attentionOut"]
        if attention_out.dtype == torch.bfloat16:
            self.attention_out_dtype_str = "bfloat16"
            self.attention_out_torch_dtype = torch.bfloat16
        else:
            self.attention_out_dtype_str = "float16"
            self.attention_out_torch_dtype = torch.float16

        query = input_data.kwargs["query"]
        if self.q_input_layout == "TND":
            assert self.pass_act_seq, "TND 下 actualSeqLengths/Kv 为必选"
            self.batch = len(raw_act_q)
            self.num_heads = int(query.shape[1])
            self.headDim = int(query.shape[2])
        elif self.q_input_layout == "BNSD":
            self.batch = int(query.shape[0])
            self.num_heads = int(query.shape[1])
            self.headDim = int(query.shape[3])
        else:
            self.batch = int(query.shape[0])
            self.num_heads = int(query.shape[2])
            self.headDim = int(query.shape[3])

        if self.pass_act_seq:
            self.q_seqlen_list = tuple(int(x) for x in raw_act_q)
            self.kv_seqlen_list = tuple(int(x) for x in raw_act_kv)
        else:
            key_shape = input_data.kwargs["key"].shape
            s_dim = 2 if self.q_input_layout == "BNSD" else 1
            self.q_seqlen_list = tuple([int(query.shape[s_dim])] * self.batch)
            self.kv_seqlen_list = tuple([int(key_shape[s_dim])] * self.batch)

        q_s_list = self.q_seqlen_list
        kv_s_list = self.kv_seqlen_list
        case_seed = case_derived_seed(
            self.q_input_layout,
            self.batch,
            self.num_heads,
            self.kv_heads,
            self.headDim,
            q_s_list,
            kv_s_list,
            self.block_shape,
            self.quant_mode,
            self.dst_type_max,
            self.attention_out_dtype_str,
            self.softmax_lse_flag,
        )
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        sparsity_ratio = SPARSITY_RATIO_CHOICES_mxfp4[
            case_seed % len(SPARSITY_RATIO_CHOICES_mxfp4)
        ]
        data_x = DATA_X_CHOICES[
            (case_seed // len(SPARSITY_RATIO_CHOICES_mxfp4)) % len(DATA_X_CHOICES)
        ]
        zero_degenerate = (
            case_seed // (len(SPARSITY_RATIO_CHOICES_mxfp4) * len(DATA_X_CHOICES))
        ) % 4 == 0

        if self.q_input_layout == "TND":
            self.padded_s1 = None
            self.padded_s2 = None
            t1, t2 = sum(q_s_list), sum(kv_s_list)
            q_tnd = randn_x((t1, self.num_heads, self.headDim), data_x)
            k_tnd = randn_x((t2, self.kv_heads, self.headDim), data_x)
            v_tnd = randn_x((t2, self.kv_heads, self.headDim), data_x)
            self.q_parts, self.k_parts, self.v_parts = [], [], []
            q_off, kv_off = 0, 0
            for bi in range(self.batch):
                sq, skv = q_s_list[bi], kv_s_list[bi]
                self.q_parts.append(
                    q_tnd[q_off : q_off + sq]
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                self.k_parts.append(
                    k_tnd[kv_off : kv_off + skv]
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                self.v_parts.append(
                    v_tnd[kv_off : kv_off + skv]
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                q_off += sq
                kv_off += skv
        else:
            key = input_data.kwargs["key"]
            if self.q_input_layout == "BNSD":
                self.padded_s1 = int(query.shape[2])
                self.padded_s2 = int(key.shape[2])
            else:
                self.padded_s1 = int(query.shape[1])
                self.padded_s2 = int(key.shape[1])
            self.q_parts = [
                randn_x((1, self.num_heads, q_s_list[bi], self.headDim), data_x)
                for bi in range(self.batch)
            ]
            self.k_parts = [
                randn_x((1, self.kv_heads, kv_s_list[bi], self.headDim), data_x)
                for bi in range(self.batch)
            ]
            self.v_parts = [
                randn_x((1, self.kv_heads, kv_s_list[bi], self.headDim), data_x)
                for bi in range(self.batch)
            ]

        if zero_degenerate:
            zero_degenerate_(self.k_parts)
            zero_degenerate_(self.v_parts)

        self.block_sparse_mask_np = build_block_sparse_mask(
            sparsity_ratio,
            q_s_list,
            kv_s_list,
            self.block_shape,
            self.batch,
            self.num_heads,
        )
        if self.padded_s1 is not None:
            ceil_q_pad = (self.padded_s1 + self.block_shape[0] - 1) // self.block_shape[
                0
            ]
            ceil_kv_pad = (
                self.padded_s2 + self.block_shape[1] - 1
            ) // self.block_shape[1]
            pad_q = max(0, ceil_q_pad - self.block_sparse_mask_np.shape[2])
            pad_kv = max(0, ceil_kv_pad - self.block_sparse_mask_np.shape[3])
            if pad_q or pad_kv:
                self.block_sparse_mask_np = np.pad(
                    self.block_sparse_mask_np,
                    ((0, 0), (0, 0), (0, pad_q), (0, pad_kv)),
                    constant_values=0,
                )
        self.block_sparse_mask_tensor = numpy_array_to_tensor_mxfp4(
            self.block_sparse_mask_np
        )

        if self.q_input_layout == "TND":
            q_p, k_p, v_p, q_s, k_s, v_s = [], [], [], [], [], []
            for bi in range(self.batch):
                q_fp8, q_e, _ = mxfp8_quantize_native(self.q_parts[bi], -1)
                k_fp8, k_e, _ = mxfp8_quantize_native(self.k_parts[bi], -1)
                v_fp8, v_e, _ = mxfp8_quantize_native(self.v_parts[bi], -2)
                q_p.append(q_fp8.permute(0, 2, 1, 3).flatten(0, 1))
                k_p.append(k_fp8.permute(0, 2, 1, 3).flatten(0, 1))
                v_p.append(v_fp8.permute(0, 2, 1, 3).flatten(0, 1))
                q_s.append(
                    _e8m0_qk_to_v3_mxfp8(q_e).permute(0, 2, 1, 3, 4).flatten(0, 1)
                )
                k_s.append(
                    _e8m0_qk_to_v3_mxfp8(k_e).permute(0, 2, 1, 3, 4).flatten(0, 1)
                )
                v_s.append(
                    _e8m0_v_to_v3_mxfp8(v_e).permute(0, 2, 1, 3, 4).flatten(0, 1)
                )
            self.q_fp8 = torch.cat(q_p, dim=0).contiguous()
            self.k_fp8 = torch.cat(k_p, dim=0).contiguous()
            self.v_fp8 = torch.cat(v_p, dim=0).contiguous()
            self.q_scale = torch.cat(q_s, dim=0).contiguous()
            self.k_scale = torch.cat(k_s, dim=0).contiguous()
            self.v_scale = torch.cat(v_s, dim=0).contiguous()
        else:
            q_bnsd = torch.cat(
                [
                    F.pad(self.q_parts[bi], (0, 0, 0, self.padded_s1 - q_s_list[bi]))
                    for bi in range(self.batch)
                ],
                dim=0,
            ).contiguous()
            k_bnsd = torch.cat(
                [
                    F.pad(self.k_parts[bi], (0, 0, 0, self.padded_s2 - kv_s_list[bi]))
                    for bi in range(self.batch)
                ],
                dim=0,
            ).contiguous()
            v_bnsd = torch.cat(
                [
                    F.pad(self.v_parts[bi], (0, 0, 0, self.padded_s2 - kv_s_list[bi]))
                    for bi in range(self.batch)
                ],
                dim=0,
            ).contiguous()
            q_fp8, q_e, _ = mxfp8_quantize_native(q_bnsd, -1)
            k_fp8, k_e, _ = mxfp8_quantize_native(k_bnsd, -1)
            v_fp8, v_e, _ = mxfp8_quantize_native(v_bnsd, -2)
            self.q_scale = _e8m0_qk_to_v3_mxfp8(q_e)
            self.k_scale = _e8m0_qk_to_v3_mxfp8(k_e)
            self.v_scale = _e8m0_v_to_v3_mxfp8(v_e)
            if self.q_input_layout == "BSND":
                self.q_fp8 = q_fp8.permute(0, 2, 1, 3).contiguous()
                self.k_fp8 = k_fp8.permute(0, 2, 1, 3).contiguous()
                self.v_fp8 = v_fp8.permute(0, 2, 1, 3).contiguous()
                self.q_scale = self.q_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.k_scale = self.k_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.v_scale = self.v_scale.permute(0, 2, 1, 3, 4).contiguous()
            else:
                self.q_fp8, self.k_fp8, self.v_fp8 = q_fp8, k_fp8, v_fp8

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device != "cpu":
            raise ValueError(
                f"不支持的 device: {self.device}; pyaclnn 请走 quantMode=4 的 V3 inputprocess 委派"
            )
        mask = torch.from_numpy(self.block_sparse_mask_np)
        out_chunks, lse_chunks = [], []
        bx, by = self.block_shape
        want_lse = bool(self.softmax_lse_flag)
        use_fp32_bm = bool(getattr(self.task_result, "is_benchmark_task", False))
        for bi in range(self.batch):
            cq_b = (self.q_seqlen_list[bi] + bx - 1) // bx
            ck_b = (self.kv_seqlen_list[bi] + by - 1) // by
            mask_b = mask[bi : bi + 1, :, :cq_b, :ck_b]
            if use_fp32_bm:
                out_b, lse_b = cpu_block_sparse_attention_bnsd_fp32_standard(
                    self.q_parts[bi],
                    self.k_parts[bi],
                    self.v_parts[bi],
                    mask_b,
                    self.block_shape,
                    self.scale_value,
                    self.kv_heads,
                    return_lse=True,
                )
            else:
                out_b, lse_b = cpu_block_sparse_attention_bnsd_mxfp8(
                    self.q_parts[bi],
                    self.k_parts[bi],
                    self.v_parts[bi],
                    mask_b,
                    self.block_shape,
                    self.scale_value,
                    self.kv_heads,
                    s_dtype="fp16",
                    out_dtype=self.attention_out_dtype_str,
                    return_lse=True,
                )
            out_chunks.append(out_b)
            lse_chunks.append(lse_b)
        packed = _assemble_layout(
            out_chunks,
            lse_chunks,
            self.q_input_layout,
            self.batch,
            self.num_heads,
            self.headDim,
            self.q_seqlen_list,
            self.padded_s1,
            want_lse,
        )
        if want_lse:
            out, lse = packed
            return out.to(torch.float32), lse.to(torch.float32)
        return packed.to(torch.float32)


class PyAclnnBlockSparseAttentionV3Api_mxfp8(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prep = self._get_prepared(input_data)
        self.prep = prep
        out_shape = _out_shape(prep)
        self.out_npu = torch.zeros(
            out_shape, dtype=prep.attention_out_torch_dtype, device="npu"
        )
        out_pack = _create_acl_tensor_mxfp4(
            self.out_npu,
            TORCH_TO_ACLTYPE[str(prep.attention_out_torch_dtype)],
            out_shape,
        )
        act_q = list(prep.q_seqlen_list) if prep.pass_act_seq else None
        act_kv = list(prep.kv_seqlen_list) if prep.pass_act_seq else None
        if prep.softmax_lse_flag:
            lse_shape = _lse_shape(prep)
            self.lse_npu = torch.zeros(lse_shape, dtype=torch.float32, device="npu")
            lse_pack = _create_acl_tensor_mxfp4(
                self.lse_npu, TORCH_TO_ACLTYPE[str(torch.float32)], lse_shape
            )
        else:
            self.lse_npu = None
            lse_pack = TensorPtr()
        input_args = [
            create_fp8_acl_tensor(prep.q_fp8),
            create_fp8_acl_tensor(prep.k_fp8),
            create_fp8_acl_tensor(prep.v_fp8),
            nnopbase.create_acl_tensor(
                prep.block_sparse_mask_tensor.contiguous().npu(),
                AclFormat.ACL_FORMAT_ND,
            ),
            TensorPtr(),
            create_int64_array_mxfp4(list(prep.block_shape)),
            create_int64_array_mxfp4(act_q),
            create_int64_array_mxfp4(act_kv),
            TensorPtr(),
            create_e8m0_acl_tensor(prep.q_scale),
            create_e8m0_acl_tensor(prep.k_scale),
            create_e8m0_acl_tensor(prep.v_scale),
            TensorPtr(),
            ctypes.c_char_p(prep.q_input_layout.encode("utf-8")),
            ctypes.c_char_p(prep.kv_input_layout.encode("utf-8")),
            Int64(prep.kv_heads),
            Int64(0),
            ctypes.c_double(prep.scale_value),
            Int64(prep.inner_precise),
            Int64(0),
            Int64(_NOT_LIMITED),
            Int64(_NOT_LIMITED),
            Int64(prep.softmax_lse_flag),
            Int64(prep.quant_mode),
            ctypes.c_double(prep.dst_type_max),
            out_pack,
            lse_pack,
        ]
        torch.npu.synchronize()
        outputs = [out_pack, lse_pack] if prep.softmax_lse_flag else [out_pack]
        return input_args, outputs

    def _get_prepared(self, input_data: InputDataset):
        cached = getattr(self.backend, "api_executor", None)
        if isinstance(cached, BlockSparseAttentionV3Api_mxfp8) and hasattr(
            cached, "q_fp8"
        ):
            return cached
        prep = BlockSparseAttentionV3Api_mxfp8(self.task_result)
        prep.init_by_input_data(input_data)
        return prep

    def after_call(self, output_packages):
        prep = self.prep
        out = self.out_npu.cpu().to(torch.float32)
        lse = self.lse_npu.cpu().to(torch.float32) if prep.softmax_lse_flag else None
        if prep.padded_s1 is not None:
            for bi in range(prep.batch):
                act_q = prep.q_seqlen_list[bi]
                if act_q >= prep.padded_s1:
                    continue
                if prep.q_input_layout == "BNSD":
                    out[bi, :, act_q:] = 0
                    if lse is not None:
                        lse[bi, :, act_q:] = 0
                else:
                    out[bi, act_q:] = 0
                    if lse is not None:
                        lse[bi, act_q:] = 0
        if lse is not None:
            return [out, lse]
        return [out]

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *blockSparseMaskOptional, const aclTensor *attenMaskOptional, "
            "const aclIntArray *blockShapeOptional, "
            "const aclIntArray *actualSeqLengthsOptional, "
            "const aclIntArray *actualSeqLengthsKvOptional, "
            "const aclTensor *blockTableOptional, "
            "const aclTensor *qDequantScaleOptional, "
            "const aclTensor *kDequantScaleOptional, "
            "const aclTensor *vDequantScaleOptional, "
            "const aclTensor *pQuantScaleOptional, "
            "char *qInputLayout, char *kvInputLayout, "
            "int64_t numKeyValueHeads, int64_t maskType, double scaleValue, "
            "int64_t innerPrecise, int64_t blockSize, int64_t preTokens, "
            "int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode, "
            "double dstTypeMax, aclTensor *attentionOut, aclTensor *softmaxLseOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )


# ----------------------------------------------------------------------
# HIF8 参考实现（原 bsa_golden_fullquant_hif8.py）
# ----------------------------------------------------------------------
"""HIF8 codec and BlockSparseAttention V3 full-quantization golden.

The codec is embedded so each quantization mode remains self-contained.
"""

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


FP32_FRACTION_BITS = 23  # fp32尾数位数

HIF8_EXP_ZERO_THRESHOLD = -23  # 边界值
HIF8_EXP_DML_MIN = -22  # DML最小指数
HIF8_EXP_DML_MAX = -15  # DML最大指数
HIF8_EXP_D0 = 0  # D0指数值
HIF8_EXP_D1_BOUNDARY = 1  # D1指数值
HIF8_EXP_D2_MIN, HIF8_EXP_D2_MAX = 2, 3  # D2指数范围
HIF8_EXP_D3_MIN, HIF8_EXP_D3_MAX = 4, 7  # D3指数范围
HIF8_EXP_D4_MIN, HIF8_EXP_D4_MAX = 8, 15  # D4指数范围

HIF8_DOT_DML = 0  # DML: Denormal Low, 指数范围 -22 ~ -16, 0位尾数
HIF8_DOT_D0 = 1  # D0: 指数为0，3位尾数（最高精度）
HIF8_DOT_D1 = 2  # D1: 指数为±1，3位尾数
HIF8_DOT_D2 = 4  # D2: 指数为±2 ~ ±3，3位尾数
HIF8_DOT_D3 = 8  # D3: 指数为±4 ~ ±7，2位尾数
HIF8_DOT_D4 = 12  # D4: 指数为±8 ~ ±15，1位尾数（最低精度）
HIF8_DOT_INVALID = -1  # 无效状态

HIF8_FRAC_BITS_DML = 0  # DML档位尾数位数
HIF8_FRAC_BITS_D0 = 3  # D0档位尾数位数
HIF8_FRAC_BITS_D1 = 3  # D1档位尾数位数
HIF8_FRAC_BITS_D2 = 3  # D2档位尾数位数
HIF8_FRAC_BITS_D3 = 2  # D3档位尾数位数
HIF8_FRAC_BITS_D4 = 1  # D4档位尾数位数

HIF8_EXP_BITS_DML = 3  # DML档位指数位数
HIF8_EXP_BITS_D0 = 0  # D0档位指数位数
HIF8_EXP_BITS_D1 = 1  # D1档位指数位数
HIF8_EXP_BITS_D2 = 2  # D2档位指数位数
HIF8_EXP_BITS_D3 = 3  # D3档位指数位数
HIF8_EXP_BITS_D4 = 4  # D4档位指数位数

HIF8_ZERO = 0
HIF8_NAN = 128  # 0b10000000, NaN
HIF8_NEG_INF = 239  # 0b11101111, -inf
HIF8_NEG_MAX = 238  # 0b11101110, 负极大值
HIF8_POS_INF = 111  # 0b01101111, +inf
HIF8_POS_MAX = 110  # 0b01101110, 正极大值

HIF8_SIGN_MASK = 128  # 0b10000000, 符号位掩码
HIF8_DOT_MASK = 120  # 0b01110000, dot值掩码
HIF8_FRAC_MASK_3BIT = 7  # 0b00000111, 3位尾数掩码（D0/D1/D2）
HIF8_FRAC_MASK_2BIT = 3  # 0b00000011, 2位尾数掩码（D3）
HIF8_FRAC_MASK_1BIT = 1  # 0b00000001, 1位尾数掩码（D4）
HIF8_EXP_MASK_DML = 7  # 0b00000111, DML指数掩码（bit0-2）
HIF8_EXP_MASK_D4 = 30  # 0b00011110, D4指数掩码（bit1-4）
HIF8_EXP_MASK_D3 = 28  # 0b00011100, D3指数掩码（bit2-4）
HIF8_EXP_MASK_D2 = 24  # 0b00011000, D2指数掩码（bit3-4）
HIF8_EXP_SIGN_MASK_D1 = 8  # 0b00001000, D1指数掩码（bit3）

HIF8_DOT_BIT_SHIFT = 3  # Dot值在HiF8中的起始位置(bit3)
HIF8_DML_EXP_OFFSET = 23  # DML指数偏移值
HIF8_OVERFLOW_SCALE = 1.25  # 溢出阈值缩放因子
HIF8_MAX_FINITE_VALUE = 32768  # 最大有限值（非饱和模式下的边界值, 2^15

SSR_T14_MASK = 16383  # 0b0011 1111 1111 1111, 14位低位掩码
SSR_F14_OFFSET = 8192  # 0b0010 0000 0000 0000, F14偏移值
SSR_DML_SHIFT = 10  # SSR舍入移位值
SSR_RESERVED_BITS = 14  # SSR舍入保留位数
HYBRID_ROUND_EXP_THRESHOLD = 4  # 混合舍入的指数分界点


def fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        return True, 0
    hif8_value_tmp = fraction32_int >> (FP32_FRACTION_BITS - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1


def fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = SSR_T14_MASK
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        f14_values = (fraction32_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (FP32_FRACTION_BITS - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (FP32_FRACTION_BITS - hif8_bits_num))
        f14_values = f14_t14 >> (FP32_FRACTION_BITS - hif8_bits_num - SSR_RESERVED_BITS)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value


def get_hif8_fraction_bits_number(exponent):
    if exponent < HIF8_EXP_DML_MIN:
        return HIF8_DOT_INVALID, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if HIF8_EXP_DML_MIN <= exponent < HIF8_EXP_DML_MAX:
        return HIF8_DOT_DML, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if exponent == HIF8_EXP_D0:
        return HIF8_DOT_D0, HIF8_EXP_BITS_D0, HIF8_FRAC_BITS_D0
    if abs(exponent) == HIF8_EXP_D1_BOUNDARY:
        return HIF8_DOT_D1, HIF8_EXP_BITS_D1, HIF8_FRAC_BITS_D1
    if HIF8_EXP_D2_MIN <= abs(exponent) <= HIF8_EXP_D2_MAX:
        return HIF8_DOT_D2, HIF8_EXP_BITS_D2, HIF8_FRAC_BITS_D2
    if HIF8_EXP_D3_MIN <= abs(exponent) <= HIF8_EXP_D3_MAX:
        return HIF8_DOT_D3, HIF8_EXP_BITS_D3, HIF8_FRAC_BITS_D3
    if HIF8_EXP_D4_MIN <= abs(exponent) <= HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_FRAC_BITS_D4
    if exponent > HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_DOT_INVALID


def cvt_float32_to_hifuint8(x, round_mode="round", over_mode=True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    ec = 0
    over_value = HIF8_OVERFLOW_SCALE * pow(2.0, HIF8_EXP_D4_MAX + ec)
    if x < 0.0:
        sign = True
        sign_int_value = HIF8_SIGN_MASK
    if torch.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return HIF8_NEG_INF
            else:
                return HIF8_NEG_MAX
        else:
            if over_mode:
                return HIF8_POS_INF
            else:
                return HIF8_POS_MAX
    if torch.isnan(x):
        if over_mode:
            return HIF8_NAN
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(
        x_abs * pow(2, FP32_FRACTION_BITS) * pow(2, -exponent)
        - pow(2, FP32_FRACTION_BITS)
    )
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = (
        get_hif8_fraction_bits_number(exponent)
    )
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = fp32_ta_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = fp32_ssr_round_to_hif8(
            fraction_int, fraction_hif8_bits, exponent
        )
    else:
        print("unknown round type")
        return 0

    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = (
            get_hif8_fraction_bits_number(exponent)
        )
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < HIF8_EXP_ZERO_THRESHOLD:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= HIF8_EXP_ZERO_THRESHOLD:
            return 0
        else:
            return sign_int_value + exponent + HIF8_DML_EXP_OFFSET
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = (
            sign_int_value
            + dot_int_value
            + sig_exp
            + exponent_int_value
            + hif8_frac_value
        )
    return hif8_int_value


def cvt_hifuint8_to_float32(x, over_mode=True):
    x = int(x)
    if x == HIF8_ZERO:
        return float(0)
    elif x == HIF8_NAN:
        if over_mode:
            return float("nan")
        else:
            return float(0)
    elif x == HIF8_NEG_INF:
        if over_mode:
            return -torch.inf
        else:
            return -HIF8_MAX_FINITE_VALUE
    elif x == HIF8_POS_INF:
        if over_mode:
            return torch.inf
        else:
            return HIF8_MAX_FINITE_VALUE
    else:
        if x >= HIF8_NAN:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & HIF8_DOT_MASK
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= HIF8_DOT_D4:
            exponent = x & HIF8_EXP_MASK_D4
            exponent_int = exponent >> 1
            if exponent_int >= 8:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 8

            fra_int = x & HIF8_FRAC_MASK_1BIT
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= HIF8_DOT_D3:
            exponent = x & HIF8_EXP_MASK_D3
            exponent_int = exponent >> 2
            if exponent_int >= 4:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 4

            fra_int = x & HIF8_FRAC_MASK_2BIT
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= HIF8_DOT_D2:
            exponent = x & HIF8_EXP_MASK_D2
            exponent_int = exponent >> 3
            if exponent_int >= 2:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 2

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= HIF8_DOT_D1:
            exponent = x & HIF8_EXP_SIGN_MASK_D1
            exponent_sign = exponent >> 3
            if exponent_sign >= 1:
                exponent_value = -1
            else:
                exponent_value = 1

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_D0:
            exponent_value = 0
            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_DML:
            m_value = 1
            exponent_value = (x & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET
        else:
            print("error, dot error")
            m_value = 0.0
            exponent_value = 0
        return sign * pow(2.0, exponent_value) * m_value


def trans_hifuint8_tensor_to_float_point(in_tensor):
    shape = in_tensor.shape

    flat = in_tensor.reshape(-1)

    out = torch.tensor(
        [cvt_hifuint8_to_float32(x) for x in flat],
        dtype=torch.float32,
        device=in_tensor.device,
    )

    return out.reshape(shape)


def trans_float_tensor_to_hifuint8_point(in_tensor, round_mode="round", over_mode=True):
    shape = in_tensor.shape

    flat = in_tensor.reshape(-1)

    out = torch.tensor(
        [cvt_float32_to_hifuint8(x, round_mode, over_mode) for x in flat],
        dtype=torch.uint8,
        device=in_tensor.device,
    )

    return out.reshape(shape)


def trans_hifuint8_tensor_to_float(in_tensor, over_mode=True):
    """
    将 HiF8 编码的 uint8 Tensor 批量转换为 float32 Tensor (支持 CPU/GPU 矢量化)
    """
    shape = in_tensor.shape
    x = in_tensor.reshape(-1).to(torch.int32)
    out = torch.zeros_like(x, dtype=torch.float32)

    # 1. 特殊值处理 (Masks)
    mask_zero = x == HIF8_ZERO
    mask_nan = x == HIF8_NAN
    mask_ninf = x == HIF8_NEG_INF
    mask_pinf = x == HIF8_POS_INF

    if over_mode:
        out = torch.where(mask_nan, torch.tensor(float("nan"), device=x.device), out)
        out = torch.where(mask_ninf, torch.tensor(-torch.inf, device=x.device), out)
        out = torch.where(mask_pinf, torch.tensor(torch.inf, device=x.device), out)
    else:
        out = torch.where(mask_nan, 0.0, out)
        out = torch.where(mask_ninf, float(-HIF8_MAX_FINITE_VALUE), out)
        out = torch.where(mask_pinf, float(HIF8_MAX_FINITE_VALUE), out)

    # 正常数值的 Mask (排除特殊值)
    mask_normal = ~(mask_zero | mask_nan | mask_ninf | mask_pinf)
    if not mask_normal.any():
        return out.reshape(shape)

    # 提取正常数值子集进行计算
    x_norm = x[mask_normal]

    # 符号位计算
    sign = torch.where(x_norm >= HIF8_NAN, -1.0, 1.0)

    # 提取 dot 档位
    dot_4_value = (x_norm & HIF8_DOT_MASK) >> 3

    # 初始化指数和尾数乘子
    exponent_value = torch.zeros_like(x_norm, dtype=torch.float32)
    m_value = torch.zeros_like(x_norm, dtype=torch.float32)

    # --- 档位 D4 ---
    m_d4 = dot_4_value >= HIF8_DOT_D4
    if m_d4.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D4) >> 1
        exponent_value = torch.where(
            m_d4,
            torch.where(exp_int >= 8, -exp_int, exp_int + 8).float(),
            exponent_value,
        )
        m_value = torch.where(m_d4, 1.0 + (x_norm & HIF8_FRAC_MASK_1BIT) * 0.5, m_value)

    # --- 档位 D3 ---
    m_d3 = (~m_d4) & (dot_4_value >= HIF8_DOT_D3)
    if m_d3.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D3) >> 2
        exponent_value = torch.where(
            m_d3,
            torch.where(exp_int >= 4, -exp_int, exp_int + 4).float(),
            exponent_value,
        )
        m_value = torch.where(
            m_d3, 1.0 + (x_norm & HIF8_FRAC_MASK_2BIT) * 0.25, m_value
        )

    # --- 档位 D2 ---
    m_d2 = (~(m_d4 | m_d3)) & (dot_4_value >= HIF8_DOT_D2)
    if m_d2.any():
        exp_int = (x_norm & HIF8_EXP_MASK_D2) >> 3
        exponent_value = torch.where(
            m_d2,
            torch.where(exp_int >= 2, -exp_int, exp_int + 2).float(),
            exponent_value,
        )
        m_value = torch.where(
            m_d2, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value
        )

    # --- 档位 D1 ---
    m_d1 = (~(m_d4 | m_d3 | m_d2)) & (dot_4_value >= HIF8_DOT_D1)
    if m_d1.any():
        exp_sign = (x_norm & HIF8_EXP_SIGN_MASK_D1) >> 3
        exponent_value = torch.where(
            m_d1, torch.where(exp_sign >= 1, -1.0, 1.0), exponent_value
        )
        m_value = torch.where(
            m_d1, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value
        )

    # --- 档位 D0 ---
    m_d0 = dot_4_value == HIF8_DOT_D0
    if m_d0.any():
        exponent_value = torch.where(m_d0, 0.0, exponent_value)
        m_value = torch.where(
            m_d0, 1.0 + (x_norm & HIF8_FRAC_MASK_3BIT) * 0.125, m_value
        )

    # --- 档位 DML ---
    m_dml = dot_4_value == HIF8_DOT_DML
    if m_dml.any():
        exponent_value = torch.where(
            m_dml,
            ((x_norm & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET).float(),
            exponent_value,
        )
        m_value = torch.where(m_dml, 1.0, m_value)

    # 计算正常值结果并写回
    norm_res = sign * torch.pow(2.0, exponent_value) * m_value
    out[mask_normal] = norm_res

    return out.reshape(shape)


def trans_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    """
    通过向量操作，将 float32 Tensor 批量转换为 HiF8 编码的 uint8 Tensor
    """

    shape = in_tensor.shape
    x = in_tensor.reshape(-1).to(torch.float32)

    # 先用int32作为输出类型，避免出现赋值错误
    out = torch.zeros_like(x, dtype=torch.int32)

    # 1. 符号位与绝对值提取
    sign_mask = x < 0.0
    sign_int_value = torch.where(sign_mask, HIF8_SIGN_MASK, 0)
    x_abs = torch.abs(x)

    # 2. 溢出与边界条件判断 (Masks)
    over_value = HIF8_OVERFLOW_SCALE * (2.0**HIF8_EXP_D4_MAX)
    mask_inf_or_over = torch.isinf(x) | (x_abs >= over_value)
    mask_nan = torch.isnan(x)
    mask_zero = x_abs == 0.0

    # 处理特殊边界填值
    if over_mode:
        out = torch.where(
            mask_inf_or_over, torch.where(sign_mask, HIF8_NEG_INF, HIF8_POS_INF), out
        )
        out = torch.where(mask_nan, HIF8_NAN, out)
    else:
        out = torch.where(
            mask_inf_or_over, torch.where(sign_mask, HIF8_NEG_MAX, HIF8_POS_MAX), out
        )
        out = torch.where(mask_nan, 0, out)
    out = torch.where(mask_zero, 0, out)

    # 提取正常数字的 Mask
    mask_normal = ~(mask_inf_or_over | mask_nan | mask_zero)
    if not mask_normal.any():
        return out.reshape(shape).to(torch.uint8)

    x_norm = x_abs[mask_normal]
    sign_norm = sign_int_value[mask_normal]

    # Extract the binary exponent exactly. FP32 log2 can round a value just
    # below a power of two up to that power's exponent, changing HIF8 rounding
    # (e.g. nextafter(2**-23, 0) must encode as zero, not the minimum value).
    mantissa, binary_exponent = torch.frexp(x_norm)
    exponent = binary_exponent.to(torch.int32) - 1

    # 确定截断模式 (TA / SSR)
    if round_mode == "hybrid":
        cut_bit_is_ta = torch.abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD
    elif round_mode == "round":
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)
    elif round_mode == "storound":
        cut_bit_is_ta = torch.zeros_like(exponent, dtype=torch.bool)
    else:
        cut_bit_is_ta = torch.ones_like(exponent, dtype=torch.bool)

    # 计算 fraction_int
    fraction_int = ((mantissa * 2.0 - 1.0) * (2.0**FP32_FRACTION_BITS)).to(torch.int32)

    # 批量获取档位属性 (根据 exponent 映射)
    abs_exp = torch.abs(exponent)

    dot = torch.full_like(exponent, HIF8_DOT_INVALID)
    exp_bits = torch.zeros_like(exponent)
    frac_bits = torch.zeros_like(exponent)

    # 条件区间映射
    m1 = exponent < HIF8_EXP_DML_MIN
    dot = torch.where(m1, HIF8_DOT_INVALID, dot)
    exp_bits = torch.where(m1, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m1, HIF8_FRAC_BITS_DML, frac_bits)

    m2 = (~m1) & (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX)
    dot = torch.where(m2, HIF8_DOT_DML, dot)
    exp_bits = torch.where(m2, HIF8_EXP_BITS_DML, exp_bits)
    frac_bits = torch.where(m2, HIF8_FRAC_BITS_DML, frac_bits)

    m3 = exponent == HIF8_EXP_D0
    dot = torch.where(m3, HIF8_DOT_D0, dot)
    exp_bits = torch.where(m3, HIF8_EXP_BITS_D0, exp_bits)
    frac_bits = torch.where(m3, HIF8_FRAC_BITS_D0, frac_bits)

    m4 = abs_exp == HIF8_EXP_D1_BOUNDARY
    dot = torch.where(m4, HIF8_DOT_D1, dot)
    exp_bits = torch.where(m4, HIF8_EXP_BITS_D1, exp_bits)
    frac_bits = torch.where(m4, HIF8_FRAC_BITS_D1, frac_bits)

    m5 = (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX)
    dot = torch.where(m5, HIF8_DOT_D2, dot)
    exp_bits = torch.where(m5, HIF8_EXP_BITS_D2, exp_bits)
    frac_bits = torch.where(m5, HIF8_FRAC_BITS_D2, frac_bits)

    m6 = (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX)
    dot = torch.where(m6, HIF8_DOT_D3, dot)
    exp_bits = torch.where(m6, HIF8_EXP_BITS_D3, exp_bits)
    frac_bits = torch.where(m6, HIF8_FRAC_BITS_D3, frac_bits)

    m7 = (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX)
    dot = torch.where(m7, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m7, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m7, HIF8_FRAC_BITS_D4, frac_bits)

    m8 = exponent > HIF8_EXP_D4_MAX
    dot = torch.where(m8, HIF8_DOT_D4, dot)
    exp_bits = torch.where(m8, HIF8_EXP_BITS_D4, exp_bits)
    frac_bits = torch.where(m8, HIF8_DOT_INVALID, frac_bits)

    # ------------------ TA 舍入分支 ------------------
    carry_ta = torch.zeros_like(exponent, dtype=torch.bool)
    frac_val_ta = torch.zeros_like(exponent)

    m_zero_thresh = exponent == HIF8_EXP_ZERO_THRESHOLD
    carry_ta = torch.where(m_zero_thresh, True, carry_ta)

    m_ta_norm = ~m_zero_thresh
    shift_bits = torch.clamp(FP32_FRACTION_BITS - (frac_bits + 1), min=0)
    hif8_val_tmp = fraction_int >> shift_bits

    pow_frac = torch.pow(2, frac_bits + 1) - 1
    m_carry = m_ta_norm & (hif8_val_tmp == pow_frac)
    carry_ta = torch.where(m_carry, True, carry_ta)

    m_odd = m_ta_norm & (~m_carry) & (hif8_val_tmp != 0) & (hif8_val_tmp % 2 == 1)
    frac_val_ta = torch.where(m_odd, (hif8_val_tmp + 1) >> 1, frac_val_ta)

    m_even = m_ta_norm & (~m_carry) & (hif8_val_tmp != 0) & (hif8_val_tmp % 2 == 0)
    frac_val_ta = torch.where(m_even, hif8_val_tmp >> 1, frac_val_ta)

    # ------------------ SSR 舍入分支 ------------------
    carry_ssr = torch.zeros_like(exponent, dtype=torch.bool)
    frac_val_ssr = torch.zeros_like(exponent)

    f14_v1 = (fraction_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
    t14_v1 = fraction_int & SSR_T14_MASK
    hif8_v1 = torch.zeros_like(fraction_int)

    s_bits = torch.clamp(FP32_FRACTION_BITS - frac_bits, min=0)
    hif8_v2 = fraction_int >> s_bits
    f14_t14 = fraction_int - (hif8_v2 << s_bits)
    s_bits_f14 = torch.clamp(FP32_FRACTION_BITS - frac_bits - SSR_RESERVED_BITS, min=0)
    f14_v2 = f14_t14 >> s_bits_f14
    t14_v2 = f14_t14 & SSR_T14_MASK

    f14_values = torch.where(m_zero_thresh, f14_v1, f14_v2)
    t14_values = torch.where(m_zero_thresh, t14_v1, t14_v2)
    hif8_value = torch.where(m_zero_thresh, hif8_v1, hif8_v2)

    m_ge = f14_values >= t14_values
    pow_frac_ssr = torch.pow(2, frac_bits) - 1
    m_ssr_carry = m_ge & (hif8_value == pow_frac_ssr)
    carry_ssr = torch.where(m_ssr_carry, True, carry_ssr)
    frac_val_ssr = torch.where(m_ge & (~m_ssr_carry), hif8_value + 1, frac_val_ssr)
    frac_val_ssr = torch.where(~m_ge, hif8_value, frac_val_ssr)

    # ------------------ 合并舍入结果 ------------------
    carry_exp_status = torch.where(cut_bit_is_ta, carry_ta, carry_ssr)
    hif8_frac_value = torch.where(cut_bit_is_ta, frac_val_ta, frac_val_ssr)

    exponent = torch.where(carry_exp_status, exponent + 1, exponent)
    abs_exp = torch.abs(exponent)

    dot = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_DOT_INVALID, dot),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_DOT_DML,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status, torch.where(exponent == HIF8_EXP_D0, HIF8_DOT_D0, dot), dot
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_DOT_D1, dot),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_DOT_D2,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_DOT_D3,
            dot,
        ),
        dot,
    )
    dot = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_DOT_D4,
            dot,
        ),
        dot,
    )

    frac_bits = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_FRAC_BITS_DML, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_FRAC_BITS_DML,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(exponent == HIF8_EXP_D0, HIF8_FRAC_BITS_D0, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_FRAC_BITS_D1, frac_bits),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_FRAC_BITS_D2,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_FRAC_BITS_D3,
            frac_bits,
        ),
        frac_bits,
    )
    frac_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_FRAC_BITS_D4,
            frac_bits,
        ),
        frac_bits,
    )

    exp_bits = torch.where(
        carry_exp_status,
        torch.where(exponent < HIF8_EXP_DML_MIN, HIF8_EXP_BITS_DML, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (exponent >= HIF8_EXP_DML_MIN) & (exponent < HIF8_EXP_DML_MAX),
            HIF8_EXP_BITS_DML,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(exponent == HIF8_EXP_D0, HIF8_EXP_BITS_D0, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(abs_exp == HIF8_EXP_D1_BOUNDARY, HIF8_EXP_BITS_D1, exp_bits),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D2_MIN) & (abs_exp <= HIF8_EXP_D2_MAX),
            HIF8_EXP_BITS_D2,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D3_MIN) & (abs_exp <= HIF8_EXP_D3_MAX),
            HIF8_EXP_BITS_D3,
            exp_bits,
        ),
        exp_bits,
    )
    exp_bits = torch.where(
        carry_exp_status,
        torch.where(
            (abs_exp >= HIF8_EXP_D4_MIN) & (abs_exp <= HIF8_EXP_D4_MAX),
            HIF8_EXP_BITS_D4,
            exp_bits,
        ),
        exp_bits,
    )

    # ------------------ 组合输出编码 ------------------
    hif8_int_value = torch.zeros_like(exponent)
    sig_exp = torch.where(exponent < 0, 1, 0)

    # 分支 A: dot <= 0
    m_a = dot <= 0
    val_a = torch.where(
        exponent <= HIF8_EXP_ZERO_THRESHOLD,
        0,
        sign_norm + exponent + HIF8_DML_EXP_OFFSET,
    )
    hif8_int_value = torch.where(m_a, val_a, hif8_int_value)

    # 分支 B: dot == 1
    m_b = dot == 1
    val_b = sign_norm + (dot << HIF8_DOT_BIT_SHIFT) + hif8_frac_value
    hif8_int_value = torch.where(m_b, val_b, hif8_int_value)

    # 分支 C: dot > 1
    m_c = dot > 1
    abs_exponent = torch.abs(exponent)
    abs_exponent = abs_exponent - torch.pow(2, exp_bits - 1)
    exponent_int_value = abs_exponent << frac_bits
    sig_exp_shifted = sig_exp << (exp_bits - 1 + frac_bits)
    dot_int_value = dot << HIF8_DOT_BIT_SHIFT
    val_c = (
        sign_norm
        + dot_int_value
        + sig_exp_shifted
        + exponent_int_value
        + hif8_frac_value
    )
    hif8_int_value = torch.where(m_c, val_c, hif8_int_value)

    hif8_int_value = torch.where(exponent < HIF8_EXP_ZERO_THRESHOLD, 0, hif8_int_value)

    out[mask_normal] = hif8_int_value

    return out.reshape(shape).to(torch.uint8)


# BlockSparseAttention V3 HIF8 golden and ACL tensor adapter.

_logging = Logger().get_logger()
HIF8_MAX = 32768.0
Q_BASE_TILE = 128
KV_BASE_TILE = 256
MIN_FLOAT = -3.402823466e38
QUANT_MODE_HIF8 = 5
NOT_LIMITED = 2147483647
SPARSITY_RATIO_CHOICES_hif8 = (1.0, 1.0, 0.9, 0.8, 0.75, 0.5, 0.4, 0.25, 0.1)
DATA_RANGE_CHOICES = (0.77, 0.91, 1.0, 2.0, 5.0)
P_SCALE_CHOICES = (1.0, 2.0, 7.0, 16.0, 32.0, 64.0)


def _f32(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.item()
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _f32_mul(lhs, rhs) -> float:
    return _f32(_f32(lhs) * _f32(rhs))


def mm1_dequant_scale(q_scale, k_scale, attention_scale) -> float:
    """Match ``float q * float k * float attentionScale``.

    The MM1 FixPipe runs with NO_QUANT (hardware dual-destination mode) and the
    online-softmax epilogue applies this scale instead, so the 19-bit scalar
    truncation no longer exists.
    """
    return _f32_mul(_f32_mul(q_scale, k_scale), attention_scale)


def quantize_hif8(src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    src = src.to(torch.float32).contiguous()
    max_abs = float(src.abs().max()) if src.numel() else 0.0
    scale = max_abs / HIF8_MAX if max_abs else 1.0
    encoded = trans_float_tensor_to_hifuint8(
        src / scale, round_mode="round", over_mode=True
    )
    return encoded.contiguous(), torch.tensor([scale], dtype=torch.float32)


def decode_hif8(src: torch.Tensor) -> torch.Tensor:
    return trans_hifuint8_tensor_to_float(
        src.to(torch.uint8).contiguous(), over_mode=True
    ).to(torch.float32)


def cast_hif8(src: torch.Tensor) -> torch.Tensor:
    bits = trans_float_tensor_to_hifuint8(
        src.to(torch.float32).contiguous(), round_mode="round", over_mode=True
    )
    return decode_hif8(bits)


def _contiguous_strides_hif8(shape: Sequence[int]) -> List[int]:
    strides = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * int(shape[idx + 1])
    return strides


def _create_acl_tensor_hif8(
    npu_tensor: torch.Tensor, acl_dtype, logical_shape
) -> AclTensorStruct:
    npu_tensor = npu_tensor.contiguous()
    if npu_tensor.device.type != "npu":
        npu_tensor = npu_tensor.npu()
    logical_shape = tuple(int(x) for x in logical_shape)
    ndim = len(logical_shape)
    dims = (Int64 * ndim)(*logical_shape)
    strides = (Int64 * ndim)(*_contiguous_strides_hif8(logical_shape))
    addr = ctypes.c_void_p(npu_tensor.data_ptr())
    acl_tensor = nnopbase.aclCreateTensor(
        dims,
        ndim,
        acl_dtype,
        strides,
        0,
        AclFormat.ACL_FORMAT_ND,
        dims,
        ndim,
        addr,
    )
    if not acl_tensor:
        raise RuntimeError(
            f"aclCreateTensor failed: dtype={acl_dtype}, shape={logical_shape}"
        )
    return AclTensorStruct(
        acl_tensor, addr.value, npu_tensor, ascendcl.aclDataTypeSize(acl_dtype)
    )


def create_hif8_acl_tensor(storage: torch.Tensor) -> AclTensorStruct:
    if not hasattr(AclDataType, "ACL_HIFLOAT8"):
        raise RuntimeError("this ATK/CANN acl wrapper does not expose ACL_HIFLOAT8")
    storage = storage.to(torch.uint8).contiguous().npu()
    return _create_acl_tensor_hif8(storage, AclDataType.ACL_HIFLOAT8, storage.shape)


def create_acl_tensor(tensor: torch.Tensor) -> AclTensorStruct:
    tensor = tensor.contiguous().npu()
    return _create_acl_tensor_hif8(
        tensor, TORCH_TO_ACLTYPE[str(tensor.dtype)], tensor.shape
    )


def create_int64_array_hif8(values):
    if values is None:
        return ctypes.POINTER(AclIntArray)()
    return nnopbase.create_x_list([Int64(int(v)) for v in values])


def _case_seed(*values) -> int:
    return zlib.crc32(repr(values).encode("utf-8")) & 0x7FFFFFFF


def _as_lengths(raw, default: int, batch: int) -> Tuple[int, ...]:
    if raw is None or (isinstance(raw, str) and raw == "None"):
        return tuple([int(default)] * batch)
    values = tuple(int(x) for x in raw)
    if len(values) != batch:
        raise ValueError(f"actual sequence length count {len(values)} != batch {batch}")
    return values


def _bnsd_to_layout(tensor: torch.Tensor, layout: str, lengths) -> torch.Tensor:
    if layout == "BNSD":
        return tensor.contiguous()
    bsnd = tensor.permute(0, 2, 1, 3).contiguous()
    if layout == "BSND":
        return bsnd
    return torch.cat(
        [bsnd[b, : int(lengths[b])] for b in range(tensor.shape[0])], dim=0
    ).contiguous()


def _lse_to_layout(tensor: torch.Tensor, layout: str, lengths) -> torch.Tensor:
    if layout == "BNSD":
        return tensor.unsqueeze(-1).contiguous()
    if layout == "BSND":
        return tensor.permute(0, 2, 1).unsqueeze(-1).contiguous()
    return (
        torch.cat(
            [
                tensor[b, :, : int(lengths[b])].transpose(0, 1)
                for b in range(tensor.shape[0])
            ],
            dim=0,
        )
        .unsqueeze(-1)
        .contiguous()
    )


def build_sparse_mask(
    batch,
    heads,
    q_storage,
    kv_storage,
    q_lengths,
    kv_lengths,
    block_shape,
    ratio,
    seed,
    layout,
):
    bx, by = (int(x) for x in block_shape)
    mask_q = max(q_lengths) if layout == "TND" else q_storage
    mask_kv = max(kv_lengths) if layout == "TND" else kv_storage
    cq = (mask_q + bx - 1) // bx
    ck = (mask_kv + by - 1) // by
    mask = torch.zeros((batch, heads, cq, ck), dtype=torch.int8)
    gen = torch.Generator().manual_seed(seed)
    for b in range(batch):
        valid_cq = (q_lengths[b] + bx - 1) // bx
        valid_ck = (kv_lengths[b] + by - 1) // by
        keep = max(1, math.ceil(valid_ck * ratio))
        for h in range(heads):
            for qb in range(valid_cq):
                chosen = torch.randperm(valid_ck, generator=gen)[:keep]
                mask[b, h, qb, chosen] = 1
    return mask


def _gather_positions(mask_row, kv_len, block_y):
    result = []
    for block_id, enabled in enumerate(mask_row.tolist()):
        if enabled:
            begin = block_id * block_y
            result.extend(range(begin, min(begin + block_y, kv_len)))
    return result


def _mm2_tiled(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((p.shape[0], v.shape[1]), dtype=torch.float32)
    for begin in range(0, p.shape[1], 128):
        end = min(begin + 128, p.shape[1])
        out = (out + torch.matmul(p[:, begin:end], v[begin:end])).to(torch.float32)
    return out


def hif8_attention_block(
    q_raw, k_raw, v_raw, q_scale, k_scale, v_scale, p_scale, attention_scale
):
    mm1_scale = mm1_dequant_scale(q_scale, k_scale, attention_scale)
    mm2_scale = _f32(v_scale)
    ln_p_scale = torch.log(torch.tensor(_f32(p_scale), dtype=torch.float32))
    out = torch.zeros((q_raw.shape[0], v_raw.shape[1]), dtype=torch.float32)
    lse = torch.full((q_raw.shape[0],), float("inf"), dtype=torch.float32)
    for q0 in range(0, q_raw.shape[0], Q_BASE_TILE):
        q1 = min(q0 + Q_BASE_TILE, q_raw.shape[0])
        q = q_raw[q0:q1]
        running_max = torch.full((q.shape[0],), MIN_FLOAT, dtype=torch.float32)
        running_sum = torch.zeros((q.shape[0],), dtype=torch.float32)
        running_out = torch.zeros((q.shape[0], v_raw.shape[1]), dtype=torch.float32)
        for k0 in range(0, k_raw.shape[0], KV_BASE_TILE):
            k1 = min(k0 + KV_BASE_TILE, k_raw.shape[0])
            raw_s = torch.matmul(k_raw[k0:k1], q.transpose(0, 1)).transpose(0, 1)
            scores = (raw_s * mm1_scale).to(torch.float32)
            tile_max = scores.max(dim=-1).values.to(torch.float32)
            new_max = torch.maximum(running_max, tile_max).to(torch.float32)
            exp_max = torch.exp(running_max - new_max).to(torch.float32)
            p_fp32 = torch.exp(scores - (new_max - ln_p_scale)[:, None]).to(
                torch.float32
            )
            new_sum = (exp_max * running_sum + p_fp32.sum(dim=-1)).to(torch.float32)
            p_hif8 = cast_hif8(p_fp32)
            # MM2 FixPipe runs with NO_QUANT: UB holds the raw P @ V, and the V dequant
            # scale is applied by the rescale-O epilogue on the normalized output.
            tile_out = _mm2_tiled(p_hif8, v_raw[k0:k1]).to(torch.float32)
            running_out = (exp_max[:, None] * running_out + tile_out).to(torch.float32)
            running_max, running_sum = new_max, new_sum
        out[q0:q1] = (running_out / running_sum[:, None] * mm2_scale).to(torch.float32)
        lse[q0:q1] = running_max + torch.log(running_sum) - ln_p_scale
    return out, lse


def bsa_hif8_reference(
    q_bits,
    k_bits,
    v_bits,
    mask,
    block_shape,
    q_lengths,
    kv_lengths,
    scales,
    attention_scale,
    out_dtype,
    layout,
):
    """Reference branch matching ``fp8_test_bsa.py::ref_attention``.

    The ATK CPU node is called twice: the ordinary task is the quantized
    benchmark path above, while the benchmark task must use the anti-quantized
    Q/K/V values and a plain block-sparse attention calculation.  HIF8 storage
    is decoded first and the corresponding dequant scale is applied, exactly
    like ``q_blk_antiquant = q_blk.astype(float32) * q_deq_scale`` in the
    reference ATK implementation.
    """
    q_raw = decode_hif8(q_bits).to(torch.float32) * _f32(scales[0])
    k_raw = decode_hif8(k_bits).to(torch.float32) * _f32(scales[1])
    v_raw = decode_hif8(v_bits).to(torch.float32) * _f32(scales[2])
    batch, q_heads, q_storage, dim = q_raw.shape
    kv_heads = k_raw.shape[1]
    group = q_heads // kv_heads
    bx, by = (int(x) for x in block_shape)
    out = torch.zeros((batch, q_heads, q_storage, dim), dtype=torch.float32)
    lse = torch.zeros((batch, q_heads, q_storage), dtype=torch.float32)
    for b in range(batch):
        for qh in range(q_heads):
            kvh = qh // group
            for qb in range((q_lengths[b] + bx - 1) // bx):
                positions = _gather_positions(mask[b, qh, qb], kv_lengths[b], by)
                index = torch.tensor(positions, dtype=torch.long)
                q0, q1 = qb * bx, min((qb + 1) * bx, q_lengths[b])
                q = q_raw[b, qh, q0:q1]
                k = k_raw[b, kvh].index_select(0, index)
                v = v_raw[b, kvh].index_select(0, index)
                scores = (
                    torch.matmul(q, k.transpose(0, 1)) * _f32(attention_scale)
                ).to(torch.float32)
                row_lse = torch.logsumexp(scores, dim=-1)
                p = torch.softmax(scores, dim=-1).to(torch.float32)
                out[b, qh, q0:q1] = torch.matmul(p, v)
                lse[b, qh, q0:q1] = row_lse
    return (
        _bnsd_to_layout(out.to(out_dtype), layout, q_lengths).to(torch.float32),
        _lse_to_layout(lse, layout, q_lengths).to(torch.float32),
    )


def bsa_hif8_golden(
    q_bits,
    k_bits,
    v_bits,
    mask,
    block_shape,
    q_lengths,
    kv_lengths,
    scales,
    p_scale,
    attention_scale,
    out_dtype,
    layout,
):
    q_raw, k_raw, v_raw = decode_hif8(q_bits), decode_hif8(k_bits), decode_hif8(v_bits)
    batch, q_heads, q_storage, dim = q_raw.shape
    kv_heads = k_raw.shape[1]
    group = q_heads // kv_heads
    bx, by = (int(x) for x in block_shape)
    out = torch.zeros((batch, q_heads, q_storage, dim), dtype=torch.float32)
    lse = torch.full((batch, q_heads, q_storage), float("inf"), dtype=torch.float32)
    for b in range(batch):
        for qh in range(q_heads):
            kvh = qh // group
            for qb in range((q_lengths[b] + bx - 1) // bx):
                positions = _gather_positions(mask[b, qh, qb], kv_lengths[b], by)
                index = torch.tensor(positions, dtype=torch.long)
                q0, q1 = qb * bx, min((qb + 1) * bx, q_lengths[b])
                block_out, block_lse = hif8_attention_block(
                    q_raw[b, qh, q0:q1],
                    k_raw[b, kvh].index_select(0, index),
                    v_raw[b, kvh].index_select(0, index),
                    scales[0],
                    scales[1],
                    scales[2],
                    p_scale,
                    attention_scale,
                )
                out[b, qh, q0:q1] = block_out
                lse[b, qh, q0:q1] = block_lse
        # The kernel does not define padded rows.  Normalize both references
        # to zero so ATK compares only the semantically valid prefix.
        lse[b, :, q_lengths[b] :] = 0
    return (
        _bnsd_to_layout(out.to(out_dtype), layout, q_lengths).to(torch.float32),
        _lse_to_layout(lse, layout, q_lengths).to(torch.float32),
    )


class Hif8Prepared:
    def prepare(self, input_data: InputDataset):
        kw = input_data.kwargs
        self.layout, self.kv_layout = str(kw["qInputLayout"]), str(kw["kvInputLayout"])
        if self.layout != self.kv_layout or self.layout not in ("BNSD", "BSND", "TND"):
            raise ValueError("HIF8 ATK requires matching BNSD/BSND/TND Q/KV layouts")
        q_desc, k_desc = kw["query"], kw["key"]
        self.kv_heads = int(kw["numKeyValueHeads"])
        self.block_shape = tuple(int(x) for x in kw["blockShape"])
        self.scale_value = float(kw["scaleValue"])
        self.inner_precise = int(kw["innerPrecise"])
        self.softmax_lse_flag = int(kw["softmaxLseFlag"])
        self.quant_mode = int(kw["quantMode"])
        self.dst_type_max = float(kw["dstTypeMax"])
        if self.quant_mode != QUANT_MODE_HIF8:
            raise ValueError(f"HIF8 ATK requires quantMode=5, got {self.quant_mode}")
        if self.layout == "TND":
            self.batch = len(kw["actualSeqLengths"])
            self.q_heads, self.dim = int(q_desc.shape[1]), int(q_desc.shape[2])
            q_storage = max(int(x) for x in kw["actualSeqLengths"])
            kv_storage = max(int(x) for x in kw["actualSeqLengthsKv"])
        elif self.layout == "BNSD":
            self.batch, self.q_heads = int(q_desc.shape[0]), int(q_desc.shape[1])
            q_storage, self.dim = int(q_desc.shape[2]), int(q_desc.shape[3])
            kv_storage = int(k_desc.shape[2])
        else:
            self.batch, q_storage = int(q_desc.shape[0]), int(q_desc.shape[1])
            self.q_heads, self.dim = int(q_desc.shape[2]), int(q_desc.shape[3])
            kv_storage = int(k_desc.shape[1])
        if self.dim not in (64, 128) or self.q_heads % self.kv_heads:
            raise ValueError(
                "HIF8 quantMode=5 requires D in {64, 128} and Nq divisible by Nkv"
            )
        self.q_storage, self.kv_storage = q_storage, kv_storage
        self.q_lengths = _as_lengths(kw.get("actualSeqLengths"), q_storage, self.batch)
        self.kv_lengths = _as_lengths(
            kw.get("actualSeqLengthsKv"), kv_storage, self.batch
        )
        if any(x <= 0 or x > q_storage for x in self.q_lengths) or any(
            x <= 0 or x > kv_storage for x in self.kv_lengths
        ):
            raise ValueError(
                "actual sequence lengths must be positive and within storage shapes"
            )
        if (
            self.block_shape[0] <= 0
            or self.block_shape[1] <= 0
            or self.block_shape[0] % 16
            or self.block_shape[1] % 16
        ):
            raise ValueError("HIF8 blockShape X/Y must be positive multiples of 16")
        self.out_dtype = (
            torch.bfloat16
            if kw["attentionOut"].dtype == torch.bfloat16
            else torch.float16
        )
        seed = _case_seed(
            self.layout,
            self.batch,
            self.q_heads,
            self.kv_heads,
            self.q_lengths,
            self.kv_lengths,
            self.block_shape,
            self.out_dtype,
            self.scale_value,
        )
        amp = DATA_RANGE_CHOICES[(seed // 7) % len(DATA_RANGE_CHOICES)]
        p_value = P_SCALE_CHOICES[(seed // 17) % len(P_SCALE_CHOICES)]
        ratio = SPARSITY_RATIO_CHOICES_hif8[seed % len(SPARSITY_RATIO_CHOICES_hif8)]
        gen = torch.Generator().manual_seed(seed)
        q_src = (
            torch.rand((self.batch, self.q_heads, q_storage, self.dim), generator=gen)
            * 2
            - 1
        ) * amp
        k_src = (
            torch.rand((self.batch, self.kv_heads, kv_storage, self.dim), generator=gen)
            * 2
            - 1
        ) * amp
        v_src = (
            torch.rand((self.batch, self.kv_heads, kv_storage, self.dim), generator=gen)
            * 2
            - 1
        ) * amp
        for b in range(self.batch):
            q_src[b, :, self.q_lengths[b] :] = 0
            k_src[b, :, self.kv_lengths[b] :] = 0
            v_src[b, :, self.kv_lengths[b] :] = 0
        self.q_bits_bnsd, self.q_scale = quantize_hif8(q_src)
        self.k_bits_bnsd, self.k_scale = quantize_hif8(k_src)
        self.v_bits_bnsd, self.v_scale = quantize_hif8(v_src)
        self.q_bits = _bnsd_to_layout(self.q_bits_bnsd, self.layout, self.q_lengths)
        self.k_bits = _bnsd_to_layout(self.k_bits_bnsd, self.layout, self.kv_lengths)
        self.v_bits = _bnsd_to_layout(self.v_bits_bnsd, self.layout, self.kv_lengths)
        self.p_scale = torch.tensor([p_value], dtype=torch.float32)
        self.mask = build_sparse_mask(
            self.batch,
            self.q_heads,
            q_storage,
            kv_storage,
            self.q_lengths,
            self.kv_lengths,
            self.block_shape,
            ratio,
            seed,
            self.layout,
        )
        _logging.info(
            f"BSA HIF8 seed={seed} layout={self.layout} B={self.batch} "
            f"Nq={self.q_heads} Nkv={self.kv_heads} q={self.q_lengths} "
            f"kv={self.kv_lengths} block={self.block_shape} ratio={ratio} p={p_value}"
        )

    def out_shape(self):
        if self.layout == "TND":
            return (sum(self.q_lengths), self.q_heads, self.dim)
        if self.layout == "BNSD":
            return (self.batch, self.q_heads, self.q_storage, self.dim)
        return (self.batch, self.q_storage, self.q_heads, self.dim)

    def lse_shape(self):
        if self.layout == "TND":
            return (sum(self.q_lengths), self.q_heads, 1)
        if self.layout == "BNSD":
            return (self.batch, self.q_heads, self.q_storage, 1)
        return (self.batch, self.q_storage, self.q_heads, 1)

    def golden(self, is_benchmark=False):
        scales = (self.q_scale.item(), self.k_scale.item(), self.v_scale.item())
        if is_benchmark:
            return bsa_hif8_reference(
                self.q_bits_bnsd,
                self.k_bits_bnsd,
                self.v_bits_bnsd,
                self.mask,
                self.block_shape,
                self.q_lengths,
                self.kv_lengths,
                scales,
                self.scale_value,
                self.out_dtype,
                self.layout,
            )
        return bsa_hif8_golden(
            self.q_bits_bnsd,
            self.k_bits_bnsd,
            self.v_bits_bnsd,
            self.mask,
            self.block_shape,
            self.q_lengths,
            self.kv_lengths,
            scales,
            self.p_scale.item(),
            self.scale_value,
            self.out_dtype,
            self.layout,
        )


class BlockSparseAttentionV3Api_hif8(BaseApi, Hif8Prepared):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        self.prepare(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        out, lse = self.golden(self.task_result.is_benchmark_task)
        return (out, lse) if self.softmax_lse_flag else out


class PyAclnnBlockSparseAttentionV3Api_hif8(AclnnBaseApi):
    def _get_prepared(self, input_data):
        cached = getattr(self.backend, "api_executor", None)
        if isinstance(cached, BlockSparseAttentionV3Api_hif8) and hasattr(
            cached, "q_bits"
        ):
            return cached
        prep = BlockSparseAttentionV3Api_hif8(self.task_result)
        prep.init_by_input_data(input_data)
        return prep

    def init_by_input_data(self, input_data: InputDataset):
        p = self._get_prepared(input_data)
        self.prep = p
        # These tensors are output buffers and are fully written by aclnn BSA.
        # ``torch.zeros(..., device="npu")`` launches the unrelated ZerosLike
        # operator, which is unavailable with the user-readable OPP overlay on
        # this host.  Empty allocation needs no auxiliary kernel; padded tails
        # are normalized on CPU in ``after_call`` below.
        self.out_npu = torch.empty(p.out_shape(), dtype=p.out_dtype, device="npu")
        out_pack = create_acl_tensor(self.out_npu)
        if p.softmax_lse_flag:
            self.lse_npu = torch.empty(p.lse_shape(), dtype=torch.float32, device="npu")
            lse_pack = create_acl_tensor(self.lse_npu)
        else:
            self.lse_npu, lse_pack = None, TensorPtr()
        args = [
            create_hif8_acl_tensor(p.q_bits),
            create_hif8_acl_tensor(p.k_bits),
            create_hif8_acl_tensor(p.v_bits),
            create_acl_tensor(p.mask),
            TensorPtr(),
            create_int64_array_hif8(p.block_shape),
            create_int64_array_hif8(p.q_lengths),
            create_int64_array_hif8(p.kv_lengths),
            TensorPtr(),
            create_acl_tensor(p.q_scale),
            create_acl_tensor(p.k_scale),
            create_acl_tensor(p.v_scale),
            create_acl_tensor(p.p_scale),
            ctypes.c_char_p(p.layout.encode()),
            ctypes.c_char_p(p.kv_layout.encode()),
            Int64(p.kv_heads),
            Int64(0),
            ctypes.c_double(p.scale_value),
            Int64(p.inner_precise),
            Int64(0),
            Int64(NOT_LIMITED),
            Int64(NOT_LIMITED),
            Int64(p.softmax_lse_flag),
            Int64(QUANT_MODE_HIF8),
            ctypes.c_double(p.dst_type_max),
            out_pack,
            lse_pack,
        ]
        torch.npu.synchronize()
        return args, [out_pack, lse_pack] if p.softmax_lse_flag else [out_pack]

    def after_call(self, output_packages):
        p = self.prep
        result = [self.out_npu.cpu().to(torch.float32)]
        if p.softmax_lse_flag:
            result.append(self.lse_npu.cpu().to(torch.float32))
        if p.layout != "TND":
            for b, valid_q in enumerate(p.q_lengths):
                if p.layout == "BNSD":
                    result[0][b, :, valid_q:] = 0
                    if p.softmax_lse_flag:
                        result[1][b, :, valid_q:] = 0
                else:
                    result[0][b, valid_q:] = 0
                    if p.softmax_lse_flag:
                        result[1][b, valid_q:] = 0
        return result

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *blockSparseMaskOptional, const aclTensor *attenMaskOptional, "
            "const aclIntArray *blockShapeOptional, const aclIntArray *actualSeqLengthsOptional, "
            "const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, "
            "const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, "
            "const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, "
            "char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, "
            "double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, "
            "int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode, double dstTypeMax, "
            "aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )


# ----------------------------------------------------------------------
# BSA tiling-key 覆盖用例调度器（原 bsa_golden.py）
# ----------------------------------------------------------------------
"""Unified V3 ATK golden for executable BSA tiling-key coverage cases.

The existing non-quant, FP8, MXFP4, MXFP8 and HIF8 golden implementations remain
the numerical references.  This module supplies one V3 entry and dispatches
both the reference calculation and ACL tensor preparation by ``quantMode``.
"""


V3_SIGNATURE = (
    "aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
    "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
    "const aclTensor *blockSparseMask, const aclTensor *attenMaskOptional, "
    "const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional, "
    "const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, "
    "const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, "
    "const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, "
    "char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, "
    "double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, "
    "int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode, double dstTypeMax, "
    "aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize, "
    "aclOpExecutor **executor)"
)


@register("aclnn_blocksparseattentioninputprocess")
class BlockSparseAttentionV3InputProcess_tk(AclnnBaseApi):
    """Positional V3 packing shared by all supported quantization modes.

    V3 indices are fixed by its C signature: q/k/v dequant scales are 9/10/11,
    pQuantScale is 12, and attentionOut/softmaxLse are 25/26.
    """

    def __init__(self, task_result: TaskResult, backend):
        super(BlockSparseAttentionV3InputProcess_tk, self).__init__(
            task_result, backend
        )
        self._delegate = None

    def init_by_input_data(self, input_data):
        quant_mode = int(input_data.kwargs["quantMode"])
        if quant_mode in (2, 3):
            # MXFP4 uses packed uint8 storage for logical FLOAT4_E2M1 Q/K/V
            # and uint8 storage for logical FLOAT8_E8M0 scales.  The generic
            # ATK converter cannot express those logical/storage dtype pairs.
            self._delegate = PyAclnnBlockSparseAttentionV3Api_mxfp4(
                self.task_result, self.backend
            )
            return self._delegate.init_by_input_data(input_data)
        elif quant_mode == 4:
            # MXFP8 uses logical FLOAT8_E4M3FN Q/K/V and FLOAT8_E8M0 scales;
            # reuse its dedicated ACL tensor construction and V3 packing.
            self._delegate = PyAclnnBlockSparseAttentionV3Api_mxfp8(
                self.task_result, self.backend
            )
            return self._delegate.init_by_input_data(input_data)
        elif quant_mode == 5:
            # HIF8 uses uint8 physical storage for logical ACL_HIFLOAT8 Q/K/V
            # and supplies per-tensor dequant/P-quant scales.
            self._delegate = PyAclnnBlockSparseAttentionV3Api_hif8(
                self.task_result, self.backend
            )
            return self._delegate.init_by_input_data(input_data)

        torch.manual_seed(10)
        torch.npu.synchronize()
        input_args, output_packages = [], []
        for index, arg in enumerate(input_data.args):
            input_args.extend(self.backend.convert_input_data(arg, index=index))
        for name, kwarg in input_data.kwargs.items():
            input_args.extend(self.backend.convert_input_data(kwarg, name=name))
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_packages.extend(self.backend.convert_output_data(output_data, index))
        appended_output_count = len(output_packages)
        input_args.extend(output_packages)

        null_ptr = ctypes.cast(ctypes.c_void_p(None), ctypes.POINTER(AclTensor))
        # Keep required tensors on NPU.  The FP8 scales must likewise be NPU
        # tensors; V3 requires null scales for quantMode=0.
        for index in (0, 1, 2, 3):
            tensor = self.acl_tensor_to_torch(input_args[index]).to("npu")
            input_args[index] = self.torch_tensor_to_acl(tensor)
        if quant_mode == 1:
            for index in (9, 10, 11):
                tensor = self.acl_tensor_to_torch(input_args[index]).to("npu")
                input_args[index] = self.torch_tensor_to_acl(tensor)
        else:
            input_args[9] = null_ptr
            input_args[10] = null_ptr
            input_args[11] = null_ptr

        # All generated coverage cases are mask-free/non-paged and V3 currently
        # treats pQuantScale as a reserved null parameter.
        input_args[4] = null_ptr
        input_args[8] = null_ptr
        input_args[12] = null_ptr

        # Drop exactly the converted output packages appended above.  The
        # legacy non-quant golden may describe only attentionOut even when V3
        # has softmaxLseFlag enabled, so the flag cannot determine this count.
        if appended_output_count:
            del input_args[-appended_output_count:]

        output_packages = []
        if input_data.kwargs["softmaxLseFlag"] == 1:
            output_packages.extend((input_args[-2], input_args[-1]))
        else:
            # Keep the V3 softmaxLseOptional slot, but make its null value a
            # typed aclTensor pointer.  JSON's optional-tensor conversion
            # yields c_void_p(None), which fails ATK's C-signature checker.
            input_args[-1] = null_ptr
            output_packages.append(input_args[-2])
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        if self._delegate is not None:
            return self._delegate.after_call(output_packages)
        outputs = [
            self.acl_tensor_to_torch(item).to(dtype=torch.float32)
            for item in output_packages
        ]
        return tuple(outputs) if len(outputs) == 2 else outputs[0]

    def get_cpp_func_signature_type(self):
        if self._delegate is not None:
            return self._delegate.get_cpp_func_signature_type()
        return V3_SIGNATURE


@register("aclnn_blocksparseattention_v3")
class BlockSparseAttentionV3Api_tk(BaseApi):
    """Select the existing reference implementation without splitting the JSON."""

    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionV3Api_tk, self).__init__(task_result)
        self._reference = None

    def init_by_input_data(self, input_data: InputDataset):
        quant_mode = input_data.kwargs["quantMode"]
        if quant_mode == 0:
            reference_cls = BlockSparseAttentionApi_nonquant
        elif quant_mode == 1:
            reference_cls = BlockSparseAttentionApi_fp8
        elif quant_mode in (2, 3):
            reference_cls = BlockSparseAttentionV3Api_mxfp4
        elif quant_mode == 4:
            reference_cls = BlockSparseAttentionV3Api_mxfp8
        elif quant_mode == 5:
            reference_cls = BlockSparseAttentionV3Api_hif8
        else:
            raise RuntimeError(f"unsupported BSA quantMode: {quant_mode}")
        self._reference = reference_cls(self.task_result)
        self._reference.init_by_input_data(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self._reference is None:
            self.init_by_input_data(input_data)
        return self._reference(input_data, with_output)
