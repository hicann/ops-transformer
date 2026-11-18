#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ctypes

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


def _per_batch_seq_lens(act_seq_len, act_seq_len_type):
    """
    按文档语义还原每个batch的序列长度:
    act_seq_len_type == 0: actSeqLenOptional 中数值为前继batch序列大小的cumsum结果(累积和);
    act_seq_len_type == 1: actSeqLenOptional 中数值为每个batch的序列大小。
    """
    if act_seq_len_type == 0:
        seq_lens = []
        prev = 0
        for value in act_seq_len:
            seq_lens.append(int(value) - prev)
            prev = int(value)
        return seq_lens
    return [int(value) for value in act_seq_len]


def nsa_compress_with_cache_golden(
    input_tensor,
    weight,
    slot_mapping,
    act_seq_len,
    block_table,
    compress_block_size,
    compress_stride,
    act_seq_len_type,
    page_block_size,
):
    """
    aclnnNsaCompressWithCache CPU golden:
      compressIdx = (s - compressBlockSize) / compressStride
      outputCacheRef[slotMapping[b]] =
          input[compressIdx*stride : compressIdx*stride + compressBlockSize] * weight[:]
    其中 s 为batch b的当前序列长度, 窗口即该batch的最后一compressBlockSize个token;
    仅当 s >= compressBlockSize 且 (s - compressBlockSize) % compressStride == 0 时该batch才会被压缩。
    weight shape = [compressBlockSize, N], weight[p, h] 与窗口内第p个token的head h按元素相乘后
    沿窗口维求和(headDim维广播), 计算在float32上进行后写回input的dtype。
    写cache规则:
      - blockTableOptional 为 None(非paged场景): input按[T, N, D]的TND token流语义展开,
        batch b的第t个token位于展开后的 (sum(actSeqLen[:b]) + t) 位置;
      - blockTableOptional 不为 None(paged场景): batch b的第t个token位于
        input[blockTable[b][t // pageBlockSize], t % pageBlockSize];
      - 每个满足压缩条件的batch向 outputCache[slotMapping[b]] 写入一个压缩token,
        未写入的位置保持为0; outputCache shape = [len(slotMapping), N, D]。
    actSeqLenOptional 为 None 时, 无batch可压缩, outputCache 保持全0。
    """
    head_num = input_tensor.shape[2]
    head_dim = input_tensor.shape[3]
    result_len = (
        int(slot_mapping.shape[0])
        if hasattr(slot_mapping, "shape")
        else len(slot_mapping)
    )

    if act_seq_len is None:
        return torch.zeros((result_len, head_num, head_dim), dtype=input_tensor.dtype)

    seq_lens = _per_batch_seq_lens(act_seq_len, act_seq_len_type)
    batch_size = min(len(seq_lens), result_len)

    output_cache = torch.zeros((result_len, head_num, head_dim), dtype=torch.float32)
    flat_tokens = input_tensor.reshape(-1, head_num, head_dim)
    total_tokens = flat_tokens.shape[0]

    token_base = 0
    for batch_idx in range(batch_size):
        seq_len = seq_lens[batch_idx]
        if (
            seq_len >= compress_block_size
            and (seq_len - compress_block_size) % compress_stride == 0
        ):
            compress_idx = (seq_len - compress_block_size) // compress_stride
            window_start = compress_idx * compress_stride
            if block_table is None:
                # 非paged场景: TND紧密排列的token流
                global_start = token_base + window_start
                if (
                    global_start < 0
                    or global_start + compress_block_size > total_tokens
                ):
                    raise IndexError(
                        "compress window [%d, %d) out of range of %d tokens"
                        % (
                            global_start,
                            global_start + compress_block_size,
                            total_tokens,
                        )
                    )
                window = flat_tokens[global_start : global_start + compress_block_size]
            else:
                # paged场景: 通过blockTable将batch内token映射到物理page
                window_tokens = []
                for offset in range(window_start, window_start + compress_block_size):
                    page_idx = int(block_table[batch_idx][offset // page_block_size])
                    page_offset = offset % page_block_size
                    window_tokens.append(input_tensor[page_idx][page_offset])
                window = torch.stack(window_tokens, dim=0)
            weight_fp32 = weight.to(torch.float32)
            # weight[p, h] 广播到headDim维, 沿窗口维求和
            compressed = (window.to(torch.float32) * weight_fp32.unsqueeze(-1)).sum(
                dim=0
            )
            output_cache[int(slot_mapping[batch_idx])] = compressed
        token_base += seq_len
    return output_cache.to(input_tensor.dtype)


@register("executor_nsa_compress_with_cache")
class NsaCompressWithCacheApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # ATK 依据 range_values 随机生成的 slotMapping 可能含重复槽位, 与算子
        # "slotMapping 无重复"约束冲突, 会导致 CPU/NPU 写 cache 顺序不一致;
        # 此处统一覆盖为确定性槽位 [0, B), 与 NPU 侧 init_by_input_data 重写逻辑一致
        batch = len(input_data.kwargs["slotMapping"])
        input_data.kwargs["slotMapping"] = list(range(batch))
        input_tensor = input_data.kwargs["input"]
        weight = input_data.kwargs["weight"]
        slot_mapping = input_data.kwargs["slotMapping"]
        act_seq_len = input_data.kwargs["actSeqLenOptional"]
        block_table = input_data.kwargs["blockTableOptional"]
        compress_block_size = int(input_data.kwargs["compressBlockSize"])
        compress_stride = int(input_data.kwargs["compressStride"])
        act_seq_len_type = int(input_data.kwargs["actSeqLenType"])
        page_block_size = int(input_data.kwargs["pageBlockSize"])

        return nsa_compress_with_cache_golden(
            input_tensor,
            weight,
            slot_mapping,
            act_seq_len,
            block_table,
            compress_block_size,
            compress_stride,
            act_seq_len_type,
            page_block_size,
        )


@register("executor_aclnn_nsa_compress_with_cache")
class NsaCompressWithCacheAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """先按基类逻辑打包输入, 再修正可选参数与 slotMapping。

        1. ATK 依据 range_values 随机生成的 slotMapping 可能含重复槽位, 违反算子
           "slotMapping 无重复"约束, 会导致 NPU 写 cache 顺序与 golden 不一致;
           将 input_args[2] 重写为无重复槽位 [0, B)。
        2. 可选参数为 None 时 ATK 的转换产物无法通过 C 签名检查, 统一置为
           对应类型的空指针(参考 GBSA_RDV/aclnn_genericblocksparseattention.py):
           input_args[3](actSeqLenOptional, aclIntArray)、
           input_args[4](blockTableOptional, aclTensor)。
        C 签名下标: input=0, weight=1, slotMapping=2, actSeqLenOptional=3,
        blockTableOptional=4, layoutOptional=5, compressBlockSize=6,
        compressStride=7, actSeqLenType=8, pageBlockSize=9, outputCache=10。
        """
        input_args, output_packages = super().init_by_input_data(input_data)

        # slotMapping 重写为无重复槽位
        slot_mapping = self.acl_tensor_to_torch(input_args[2])
        new_slot = torch.arange(slot_mapping.shape[0], dtype=torch.int32)
        if slot_mapping.device.type != "cpu":
            new_slot = new_slot.to(slot_mapping.device)
        input_args[2] = self.torch_tensor_to_acl(new_slot)

        # 可选参数为 None 时置为对应类型的空指针, 非 None 保持基类转换结果
        null_tensor_ptr = ctypes.cast(ctypes.c_void_p(None), ctypes.POINTER(AclTensor))
        null_int_array_ptr = ctypes.cast(
            ctypes.c_void_p(None), ctypes.POINTER(AclIntArray)
        )
        if input_data.kwargs["actSeqLenOptional"] is None:
            input_args[3] = null_int_array_ptr
        if input_data.kwargs["blockTableOptional"] is None:
            input_args[4] = null_tensor_ptr
        return input_args, output_packages

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(const aclTensor *input, "
            "const aclTensor *weight, const aclTensor *slotMapping, const aclIntArray *actSeqLenOptional, "
            "const aclTensor *blockTableOptional, char *layoutOptional, int64_t compressBlockSize, "
            "int64_t compressStride, int64_t actSeqLenType, int64_t pageBlockSize, aclTensor *outputCache, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
