#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ctypes

import torch
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclIntArray
from atk.common.log import Logger

logging = Logger().get_logger()


class NsaSelectedAttentionInferGolden:
    """aclnnNsaSelectedAttentionInfer CPU golden.

    语义（与算子文档对齐）：
    - query: TND 时为 [totalQTokens, numHeads, 192]（Dq = 192，含 rope 部分）；
      BSH 时为 [B, S, numHeads * 192]（3 维）；BSND 时为 [B, S, numHeads, 192]。
    - key/value: paged KV cache，blocknum 个物理块，
      key: [numBlocks, pageBlockSize, numKVHeads * 192]（3 维）或
           [numBlocks, pageBlockSize, numKVHeads, 192]（4 维，BSND）；
      value: [numBlocks, pageBlockSize, numKVHeads * 128] 或
             [numBlocks, pageBlockSize, numKVHeads, 128]。
    - topkIndices: TND 时 [T, numKVHeads, selectBlockCount]，
      BSH/BSND 时 [B, S, numKVHeads, selectBlockCount]，元素为 selectBlockSize
      粒度的逻辑块号，对应 KV 绝对位置 [blockId * selectBlockSize, +selectBlockSize)。
    - blockTableOptional: [B, maxBlockNumPerBatch] int32，逻辑页号 -> 物理页号映射；
      为 None 时逻辑页号直取物理页号（非 paged 直取）。
    - 每 token 按 topkIndices 逐块取出选中 KV 并拼接（GQA 按 numHeads // numKVHeads
      分组），scale 乘、softmax(QK^T) 后与 value_topk 相乘，内部 fp32 计算后转回
      输入 dtype。
    - attenMaskOptional/sparseMode 为预留参数（文档：暂未使用），golden 不处理。
    """

    @staticmethod
    def _to_int_list(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        if isinstance(value, torch.Tensor):
            return [int(v) for v in value.reshape(-1).tolist()]
        if isinstance(value, np.ndarray):
            return [int(v) for v in value.reshape(-1).tolist()]
        return [int(value)]

    @classmethod
    def _attend_tokens(
        cls,
        q_tokens,
        topk_tokens,
        key4,
        value4,
        block_table_row,
        out_tokens,
        group,
        select_block_size,
        page_block_size,
        scale_value,
        dtype,
        kv_heads,
    ):
        """q_tokens: [S, numHeads, Dq]; topk_tokens: [S, kvHeads, selectBlockCount];
        out_tokens: [S, numHeads, Dv]。"""
        s_len = q_tokens.shape[0]
        offsets = torch.arange(select_block_size, dtype=torch.int64)
        for t in range(s_len):
            for h in range(kv_heads):
                block_ids = topk_tokens[t, h].reshape(-1).to(torch.int64)
                positions = (block_ids * select_block_size).unsqueeze(
                    1
                ) + offsets.unsqueeze(0)
                positions = positions.reshape(-1)
                pages = torch.div(positions, page_block_size, rounding_mode="floor")
                offsets_in_page = positions % page_block_size
                if block_table_row is not None:
                    pages = block_table_row.reshape(-1).to(torch.int64)[pages]
                k_selected = key4[pages, offsets_in_page, h, :].to(torch.float32)
                v_selected = value4[pages, offsets_in_page, h, :].to(torch.float32)
                q_group = q_tokens[t, h * group : (h + 1) * group].to(torch.float32)
                sim = torch.matmul(q_group, k_selected.transpose(0, 1)) * scale_value
                sim = sim - torch.max(sim, dim=-1, keepdim=True).values
                prob = torch.exp(sim)
                prob = prob / torch.sum(prob, dim=-1, keepdim=True)
                prob_low = prob.to(dtype).to(torch.float32)
                out = torch.matmul(prob_low, v_selected)
                out_tokens[t, h * group : (h + 1) * group] = out.to(dtype)

    @classmethod
    def compute(
        cls,
        query,
        key,
        value,
        topk_indices,
        block_table,
        actual_q_seq_len,
        actual_kv_seq_len,
        layout,
        num_heads,
        num_key_value_heads,
        select_block_size,
        select_block_count,
        page_block_size,
        scale_value,
    ):
        q_lens = cls._to_int_list(actual_q_seq_len)
        kv_lens = cls._to_int_list(actual_kv_seq_len)
        layout = str(layout)
        num_heads = int(num_heads)
        kv_heads = int(num_key_value_heads)
        select_block_size = int(select_block_size)
        page_block_size = int(page_block_size)
        scale_value = float(scale_value)
        dtype = query.dtype
        group = num_heads // kv_heads

        def split_page_cache(cache):
            if cache.dim() == 3:
                nb, bs, hidden = cache.shape
                return cache.reshape(nb, bs, kv_heads, hidden // kv_heads)
            return cache

        key4 = split_page_cache(key)
        value4 = split_page_cache(value)
        value_dim = value4.shape[-1]

        if layout == "TND":
            # query: [T, numHeads, Dq]; topkIndices: [T, kvHeads, selectBlockCount]
            q_tokens_all = query
            topk_all = topk_indices
            total_tokens = q_tokens_all.shape[0]
            out = torch.zeros((total_tokens, num_heads, value_dim), dtype=dtype)
            token_offset = 0
            for batch_idx, q_len in enumerate(q_lens):
                bt_row = None if block_table is None else block_table[batch_idx]
                cls._attend_tokens(
                    q_tokens_all[token_offset : token_offset + q_len],
                    topk_all[token_offset : token_offset + q_len],
                    key4,
                    value4,
                    bt_row,
                    out[token_offset : token_offset + q_len],
                    group,
                    select_block_size,
                    page_block_size,
                    scale_value,
                    dtype,
                    kv_heads,
                )
                token_offset += q_len
            return out

        # BSH / BSND
        if layout == "BSH":
            batch, s_max, hidden = query.shape
            q4 = query.reshape(batch, s_max, num_heads, hidden // num_heads)
        else:  # BSND
            q4 = query
            batch, s_max = q4.shape[0], q4.shape[1]
        topk_all = topk_indices  # [B, S, kvHeads, selectBlockCount]
        out = torch.zeros((batch, s_max, num_heads, value_dim), dtype=dtype)
        for batch_idx in range(batch):
            bt_row = None if block_table is None else block_table[batch_idx]
            cls._attend_tokens(
                q4[batch_idx],
                topk_all[batch_idx],
                key4,
                value4,
                bt_row,
                out[batch_idx],
                group,
                select_block_size,
                page_block_size,
                scale_value,
                dtype,
                kv_heads,
            )
        if layout == "BSH":
            return out.reshape(batch, s_max, num_heads * value_dim)
        return out


@register("executor_nsa_selected_attention_infer")
class NsaSelectedAttentionInferApi(BaseApi):
    """CPU golden executor for aclnnNsaSelectedAttentionInfer."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = input_data.kwargs
        query = kwargs["query"]
        key = kwargs["key"]
        value = kwargs["value"]
        topk_indices = kwargs["topkIndices"]
        block_table = kwargs.get("blockTableOptional", None)
        actual_q_seq_len = kwargs["actualQSeqLenOptional"]
        actual_kv_seq_len = kwargs["actualKvSeqLenOptional"]
        layout = kwargs["layoutOptional"]
        num_heads = kwargs["numHeads"]
        num_key_value_heads = kwargs["numKeyValueHeads"]
        select_block_size = kwargs["selectBlockSize"]
        select_block_count = kwargs["selectBlockCount"]
        page_block_size = kwargs["pageBlockSize"]
        scale_value = kwargs["scaleValue"]
        # attenMaskOptional/sparseMode 为预留参数，golden 不使用
        golden_out = NsaSelectedAttentionInferGolden.compute(
            query,
            key,
            value,
            topk_indices,
            block_table,
            actual_q_seq_len,
            actual_kv_seq_len,
            layout,
            num_heads,
            num_key_value_heads,
            select_block_size,
            select_block_count,
            page_block_size,
            scale_value,
        )
        return golden_out


@register("executor_aclnn_nsa_selected_attention_infer")
class NsaSelectedAttentionInferAclnnApi(AclnnBaseApi):
    """aclnnNsaSelectedAttentionInfer NPU 两段式调用。

    GBSA 参考模式：手动两段式组包并记录 name_to_idx，对 json 中
    range_values 为 ["None"] 的可选输入（kwargs 值为 None）按名字将对应
    input_args 下标置空为类型正确的空指针，避免 backend.convert_input_data
    产生的 c_void_p(None) 无法通过 C 签名检查。
    blockTableOptional 为必选输入（全部用例均有真实值），不置空；
    输出参数 output 不改动。
    """

    def init_by_input_data(self, input_data: InputDataset):
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
        input_args.extend(output_packages)

        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        null_int_array_ptr = ctypes.cast(null_void_ptr, ctypes.POINTER(AclIntArray))

        # json 中声明为 ["None"] 的可选输入（kwargs 值为 None）按名字统一置空；
        # 非 None 保持原样。blockTableOptional 必选有真实值，不在置空列表。
        if (
            name_to_idx.get("attenMaskOptional") is not None
            and input_data.kwargs.get("attenMaskOptional") is None
        ):
            input_args[name_to_idx["attenMaskOptional"]] = null_tensor_ptr
        for name in ("actualQSeqLenOptional", "actualKvSeqLenOptional"):
            if name in name_to_idx and input_data.kwargs.get(name) is None:
                input_args[name_to_idx[name]] = null_int_array_ptr

        return input_args, output_packages

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *topkIndices, const aclTensor *attenMaskOptional, const aclTensor *blockTableOptional, const aclIntArray *actualQSeqLenOptional, const aclIntArray *actualKvSeqLenOptional, char *layoutOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize, int64_t selectBlockCount, int64_t pageBlockSize, double scaleValue, int64_t sparseMode, aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor)"
