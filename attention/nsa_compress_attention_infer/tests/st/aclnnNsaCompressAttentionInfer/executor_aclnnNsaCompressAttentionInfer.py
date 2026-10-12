#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import torch
import numpy as np
import torch_npu
from dataclasses import dataclass
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.common.log import Logger

logging = Logger().get_logger()
TOPK_CHECK_OUT_SUFFIX = 2


def golden_add_gen(input_shape, l, d, ll, x=None):
    q_seqlen, S1, S2 = input_shape
    pcmp = torch.randn(input_shape, dtype=torch.float)
    if x is not None:
        pcmp = x
    pcmp_transpose = pcmp.transpose(2, 1)
    pslc = torch.zeros([q_seqlen, S2, S1], dtype=torch.float)
    for i in range(q_seqlen):
        for j in range(1, S2):
            for m in range(int(ll / d)):
                for n in range(int(l / d)):
                    idx = int(ll / d * j) - m - n
                    if idx >= S2:
                        continue
                    if idx < 0:
                        continue
                    pslc[i, j - 1, :] += pcmp_transpose[i, idx, :]
    return pslc


def golden_reduce_sum(pslc, q_seqlen, s2, g, s2_new):
    pslc[:, 0, :] = np.inf
    pslc[:, s2_new - 2 : s2_new, :] = np.inf
    p = pslc.reshape(q_seqlen, s2, -1, g)
    res = p.sum(-1)
    return res.transpose(2, 1)[:, :, :s2_new]


def sort_and_select(input_tensor, k):
    # 获取输入tensor的形状 (行数, 列数)
    q_seqlen, rows, cols = input_tensor.shape

    # 结果矩阵，用来存储每行处理后的结果
    result = np.zeros(
        (q_seqlen, rows, k), dtype=np.int32
    )  # 每行保留k个元素，指定 dtype 为 int32

    # 每行多存几个结果，用于topk校验
    topk_check = np.zeros((q_seqlen, rows, k + TOPK_CHECK_OUT_SUFFIX), dtype=np.int32)

    for s in range(q_seqlen):
        for i in range(rows):
            # 获取当前行
            row = input_tensor[s, i, :]

            # 保留第一个和最后两个元素
            first = row[0]
            last_two = row[-2:]

            # 排序其余元素，去除第一个和最后两个元素
            middle_elements = row[1:-2]
            # sorted_middle = np.argsort(middle_elements)[::-1]  # 返回排序后的索引（降序）
            sorted_middle = np.argsort(
                -middle_elements, kind="mergesort"
            )  # 返回排序后的索引（降序）

            # 获取前k-3个最大的数的索引
            top_k_indices = sorted_middle[: k - 3]

            # 每行多存几个结果，用于topk校验
            topk_check_indices = sorted_middle[: k - 3 + TOPK_CHECK_OUT_SUFFIX]

            # 构建新的行索引
            result_row_indices = np.concatenate(
                ([0], [cols - 2, cols - 1], top_k_indices + 1)
            )  # 修改这里，选择最后两个元素
            # 每行多存几个结果，用于topk校验
            result_check_row_indices = np.concatenate(
                ([0], [cols - 2, cols - 1], topk_check_indices + 1)
            )

            # 将结果存储到 result 中
            result[s, i, :] = result_row_indices.astype(np.int32)  # 转换为 int32 类型
            # 每行多存几个结果，用于topk校验
            topk_check[s, i, :] = result_check_row_indices.astype(np.int32)

    return result, topk_check


class TestPagedMLAttention:
    @dataclass
    class AttentionInputs:
        query: any
        key_cache: any
        value_cache: any
        block_tables: any
        q_seqlen_list: any
        k_seqlen_list: any
        global_mask: any
        mask_type: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        num_heads: int
        kv_heads: int
        head_size: int
        head_size_rope: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any

    @classmethod
    def check_attr(
        cls,
        batch: int,
        q_seqlen_list: int,
        kv_seqlen: int,
        num_blocks: int,
        block_size: int,
    ):
        if batch * kv_seqlen > num_blocks * block_size:
            logging(
                "[ERROR] the number of K and V tokens is too big to fit in the paged cache."
            )
            sys.exit()

        if block_size > 128:
            logging("[ERROR] blockSize > 128 is not supported.")
            sys.exit()

        if min(q_seqlen_list) < 1 or max(q_seqlen_list) > 4:
            logging("[ERROR] q_seqlen is support [1, 4].")
            sys.exit()

    @classmethod
    def check_attr1(
        cls,
        batch: int,
        q_seqlen: int,
        max_kv_seqlen: int,
        min_kv_seqlen: int,
        l: int,
        d: int,
        ll: int,
        num_blocks: int,
        block_size: int,
        topk: int,
    ):
        if batch * max_kv_seqlen > num_blocks * block_size:
            logging(
                "[ERROR] the number of K and V tokens is too big to fit in the paged cache."
            )
            return False

        if block_size > 128:
            logging("[ERROR] blockSize > 128 is not supported.")
            return False

        selectKvSeqlenMax = (max_kv_seqlen - 1) * d + l
        selectBlockNumMax = (selectKvSeqlenMax + ll - 1) / ll
        if selectBlockNumMax > 4096:
            logging(
                f"[ERROR] selectBlockNum must be within (0, 4096]. selectBlockNumMax = {selectBlockNumMax}"
            )
            return False
        selectKvSeqlenMin = (min_kv_seqlen - 1) * d + l
        selectBlockNumMin = (selectKvSeqlenMin + ll - 1) // ll
        if selectBlockNumMin < topk + TOPK_CHECK_OUT_SUFFIX:
            return False
        return True

    @classmethod
    def group_matmul(cls, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = torch.matmul(
                A[i * group_num : (i + 1) * group_num, :, :].to(torch.float32),
                B[i : (i + 1), :, :].to(torch.float32),
            )
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def ref_masked_attention(
        self,
        query,  # (q_seqlen, num_heads, head_size)
        key,  # (k_seqlen, kv_heads, head_size)
        value,
        scale: float,
        mask,  # (q_seqlen, k_seqlen)
        index,
        l,
        d,
        ll,
        top_k,
        g,
    ):
        query = query

        query = torch.permute(query, (1, 0, 2))  # [1, 64, 192]   -> [64, 1, 192]
        key = torch.permute(key, (1, 2, 0))  # [4096, 4, 192] -> [4, 192, 4096]

        sim_high = self.group_matmul(
            query.shape[0], key.shape[0], query, key
        )  # (head_num, q_seqlen, k_seqlen)

        sim_high = sim_high * scale

        if mask is not None and self.q_seqlen > 1:
            # 需要知道当前的batch qseqlen
            lastValidStart = (self.selectKvseqlen - l) // d * d
            lastValidEnd = lastValidStart + l - 1
            tailKvTokens = self.selectKvseqlen - lastValidEnd - 1

            for i in range(self.q_seqlen):
                if self.q_seqlen - i - 1 > tailKvTokens:
                    sim_high[:, i, self.k_seqlen - 1] = -10000

        p_high = self.softmax_numpy(sim_high)
        p = torch.from_numpy(p_high).to(query.dtype)
        p_high = torch.from_numpy(p_high).to(torch.float32)

        value = torch.permute(value, (1, 0, 2))
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)

        out_high = torch.permute(out_high, (1, 0, 2))
        out = torch.permute(out, (1, 0, 2))
        out = out.to(query.dtype)

        ## 计算importance score
        x = torch.permute(p_high, (1, 0, 2))
        q_seqlen = x.shape[0]
        s1 = x.shape[1]
        s2 = x.shape[2]
        pslc_add = golden_add_gen([q_seqlen, s1, s2], l, d, ll, x)
        s2_new = int(np.ceil(((s2 - 1) * d + l) / ll))  # 杨东更改
        pslc = golden_reduce_sum(pslc_add, q_seqlen, s2, g, s2_new)

        topk_output, topk_check_output = sort_and_select(pslc.numpy(), top_k)
        return (
            out,
            out_high,
            p_high,
            p,
            sim_high,
            pslc,
            topk_output,
            s2_new,
            topk_check_output,
        )

    def ref_single_query_cached_kv_attention(
        self,
        attention_inputs: AttentionInputs,
        output,
        true_out,
        topk_out,
        pslc_out,
        topk_check_out,
        scale,
        l,
        d,
        ll,
        top_k,
        g,
        sel_kv_seqlen,
    ):
        num_heads = attention_inputs.query.shape[1]
        kv_heads = attention_inputs.value_cache.shape[2]
        head_size_qk = attention_inputs.key_cache.shape[3]
        head_size_vo = attention_inputs.value_cache.shape[3]
        block_size = attention_inputs.value_cache.shape[1]

        batch = len(attention_inputs.q_seqlen_list)
        cu_seqlen = 0
        p_high_out = torch.ones(
            (batch, num_heads, 1, max(attention_inputs.k_seqlen_list)),
            dtype=torch.float32,
        )
        p_out = torch.ones(
            (batch, num_heads, 1, max(attention_inputs.k_seqlen_list)),
            dtype=attention_inputs.query.dtype,
        )
        for i in range(batch):
            q_seqlen = int(attention_inputs.q_seqlen_list[i])
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            self.q_seqlen = q_seqlen
            self.k_seqlen = k_seqlen
            self.selectKvseqlen = sel_kv_seqlen[i]
            q = attention_inputs.query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
            block_table = attention_inputs.block_tables[i]
            keys = []
            values = []
            for j in range(k_seqlen):  # j 每个k token拼接
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size
                k = attention_inputs.key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = attention_inputs.value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)

            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            if attention_inputs.mask_type == 1:
                mask = torch.zeros((128, 128), dtype=torch.bool)
            else:
                mask = None
            (
                out,
                out_high,
                p_high,
                p,
                sim_high,
                pslc,
                topk_output,
                s2_new,
                topk_check_output,
            ) = self.ref_masked_attention(
                q, keys, values, scale, mask, i, l, d, ll, top_k, g
            )

            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            pslc = pslc.reshape(-1, kv_heads, s2_new)
            output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out_high
            pslc_out[cu_seqlen : cu_seqlen + q_seqlen, :, :s2_new] = pslc
            topk_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = topk_output
            topk_check_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = topk_check_output
            cu_seqlen += attention_inputs.q_seqlen_list[i]

        return scale

    def calc_data(
        self,
        gen_data_params: GenDataParams,
        query,
        key_cache,
        value_cache,
        block_tables,
        kv_seqlen_list,
        sel_kv_seqlen,
        num_head,
        kv_heads,
        ll,
        top_k,
        l,
        d,
        scale,
        block_size,
        input_dtype,
    ):
        head_size_qk = gen_data_params.head_size + gen_data_params.head_size_rope
        head_size_vo = gen_data_params.head_size
        num_tokens = np.array(gen_data_params.q_seqlen_list).sum()
        batch_size = len(gen_data_params.q_seqlen_list)
        mask = None

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        ref_output = torch.zeros(shape_out, dtype=gen_data_params.dtype)
        true_out = torch.zeros(shape_out, dtype=torch.float32)
        s2 = max(kv_seqlen_list)
        s2_new_after_score = int(np.ceil(((s2 - 1) * d + l) / ll))  # s2是k_seq_len
        pslc_out_shape_out = (num_tokens, kv_heads, s2_new_after_score)
        pslc_out = np.zeros(pslc_out_shape_out, dtype=np.float32)
        topk_shape_out = (num_tokens, gen_data_params.kv_heads, top_k)
        topk_out = np.zeros(topk_shape_out, dtype=np.float32)
        # 用于辅助topk校验
        topk_check_shape_out = (
            num_tokens,
            gen_data_params.kv_heads,
            top_k + TOPK_CHECK_OUT_SUFFIX,
        )
        topk_check_out = np.zeros(topk_check_shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(
            query,
            key_cache,
            value_cache,
            block_tables,
            gen_data_params.q_seqlen_list,
            gen_data_params.k_seqlen_list,
            mask,
            gen_data_params.mask_type,
        )
        g = num_head // kv_heads
        scale = self.ref_single_query_cached_kv_attention(
            attention_inputs,
            ref_output,
            true_out,
            topk_out,
            pslc_out,
            topk_check_out,
            scale,
            l,
            d,
            ll,
            top_k,
            g,
            sel_kv_seqlen,
        )

        atten_mask = np.zeros(
            (len(gen_data_params.k_seqlen_list), gen_data_params.k_seqlen_list[0]),
            dtype=np.float32,
        )

        key_cache_reshape = key_cache.reshape(
            gen_data_params.num_blocks,
            gen_data_params.block_size,
            gen_data_params.kv_heads * head_size_qk,
        )

        value_cache_reshape = value_cache.reshape(
            gen_data_params.num_blocks,
            gen_data_params.block_size,
            gen_data_params.kv_heads * head_size_vo,
        )

        return (
            query,
            key_cache_reshape,
            value_cache_reshape,
            torch.from_numpy(np.array(block_tables).astype(np.int32)),
            np.array(gen_data_params.k_seqlen_list).astype(np.int64),
            torch.from_numpy(atten_mask),
            ref_output,
            true_out,
            torch.from_numpy(topk_out.astype(np.int32)),
            torch.from_numpy(topk_check_out.astype(np.int32)),
            scale,
        )

    def cpu_result(
        query,
        key_cache,
        value_cache,
        block_table,
        q_seqlen_list,
        kv_seqlen_list,
        sel_kv_seqlen,
        num_head,
        kv_heads,
        ll,
        top_k,
        l,
        d,
        scale,
        block_size,
        layout,
        is_benchmark_task,
    ):
        input_dtype = query.dtype  # torch.float16 / torch.bfloat16
        # 计算cpu结果
        batch = block_table.size(0)
        # q_seqlen_list = [1] * batch
        num_blocks = key_cache.size(0)
        key_cache = key_cache.reshape(num_blocks, block_size, kv_heads, -1)
        value_cache = value_cache.reshape(num_blocks, block_size, kv_heads, -1)
        embedding_size = value_cache.size(-1)
        embedding_size_rope = key_cache.size(-1) - value_cache.size(-1)
        mask_type = 1
        max_kv_seqlen = max(kv_seqlen_list)
        min_kv_seqlen = min(kv_seqlen_list)

        testObj = TestPagedMLAttention()
        testObj.check_attr(batch, q_seqlen_list, max_kv_seqlen, num_blocks, block_size)
        testObj.check_attr1(
            batch,
            1,
            max_kv_seqlen,
            min_kv_seqlen,
            l,
            d,
            ll,
            num_blocks,
            block_size,
            top_k,
        )
        gen_data_params = testObj.GenDataParams(
            q_seqlen_list,
            kv_seqlen_list,
            num_head,
            kv_heads,
            embedding_size,
            embedding_size_rope,
            num_blocks,
            block_size,
            mask_type,
            input_dtype,
        )

        (
            _,
            _,
            _,
            _,
            _,
            _,
            golden_atten_out,
            high_percision_atten_out,
            golden_topk_out,
            topk_check_out,
            _,
        ) = testObj.calc_data(
            gen_data_params,
            query,
            key_cache,
            value_cache,
            block_table,
            kv_seqlen_list,
            sel_kv_seqlen,
            num_head,
            kv_heads,
            ll,
            top_k,
            l,
            d,
            scale,
            block_size,
            input_dtype,
        )

        if not is_benchmark_task:
            # 返回标杆
            return golden_atten_out, golden_topk_out
        else:
            # 返回真值
            return high_percision_atten_out, topk_check_out


@register("executor_nsa_compress_attention_infer")
class NsaCompressAttentionInferApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        from atk.common.connection import RemoteManager

        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = []
        for node in self.task_result.nodes.nodes:
            if node != post_node:
                remote_nodes.append(node)
        remote_manager = RemoteManager(
            self.task_result.case_config, post_node, remote_nodes[0]
        )
        local_dir, remote_dir = remote_manager.get_local_and_remote_dir()
        logging.info(f"local_dir{local_dir}, remote_dir{remote_dir}")

        if self.device == "cpu":
            cpu_result_out, _ = TestPagedMLAttention.cpu_result(
                self.query,
                self.key,
                self.value,
                self.block_table,
                self.actual_q_seq_len,
                self.actual_cmp_seq_kvlen,
                self.actual_sel_seq_kvlen,
                self.head_num,
                self.key_value_head_num,
                self.select_block_size,
                self.select_block_count,
                self.compress_block_size,
                self.compress_stride,
                self.scale_value,
                self.page_block_size,
                layout="TND",
                is_benchmark_task=self.task_result.is_benchmark_task,
            )
            cpu_result_topk = torch.zeros_like(_)
            return cpu_result_out, cpu_result_topk

        elif self.device == "npu":
            real_atten_out, real_topk_out = torch_npu.npu_nsa_compress_attention_infer(
                self.query,
                self.key,
                self.value,
                self.scale_value,
                self.head_num,
                self.key_value_head_num,
                self.select_block_size,
                self.select_block_count,
                self.page_block_size,
                self.compress_block_size,
                self.compress_stride,
                atten_mask=None,
                block_table=self.block_table,
                topk_mask=None,
                actual_seq_qlen=None,
                actual_cmp_seq_kvlen=self.actual_cmp_seq_kvlen,
                actual_sel_seq_kvlen=None,
            )
            result = [real_atten_out.cpu(), real_atten_out.dtype]
            torch.save(result, f"{local_dir}/result.bin")
            return real_atten_out

    def init_by_input_data(self, input_data: InputDataset):
        self.query = input_data.kwargs["query"]
        self.key = input_data.kwargs["key"]
        self.value = input_data.kwargs["value"]
        self.scale_value = input_data.kwargs["scaleValue"]
        self.head_num = input_data.kwargs["numHeads"]
        self.key_value_head_num = input_data.kwargs["numKeyValueHeads"]
        self.select_block_size = input_data.kwargs["selectBlockSize"]
        self.select_block_count = input_data.kwargs["selectBlockCount"]
        self.page_block_size = input_data.kwargs["pageBlockSize"]
        self.compress_block_size = input_data.kwargs["compressBlockSize"]
        self.compress_stride = input_data.kwargs["compressBlockStride"]
        self.block_table = input_data.kwargs["blockTableOptional"]
        self.actual_q_seq_len = input_data.kwargs["actualQSeqLenOptional"]
        # 新增sel actual kv seqlen
        self.actual_sel_seq_kvlen = input_data.kwargs["actualSelKvSeqLenOptional"]
        self.actual_cmp_seq_kvlen = input_data.kwargs["actualCmpKvSeqLenOptional"]
        self.embedding_size = self.value.shape[2] / self.key_value_head_num
        self.embedding_size_rope = self.query.shape[2] - self.embedding_size
        self.max_kv_seqlen = max(self.actual_cmp_seq_kvlen)


@register("executor_aclnn_nsa_compress_attention_infer")
class SampleAclnnApi(AclnnBaseApi):
    def after_call(self, output_packages):
        output1, output2 = super().after_call(output_packages)
        from atk.common.connection import RemoteManager

        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = []
        for node in self.task_result.nodes.nodes:
            if node != post_node:
                remote_nodes.append(node)
        remote_manager = RemoteManager(
            self.task_result.case_config, post_node, remote_nodes[0]
        )
        local_dir, remote_dir = remote_manager.get_local_and_remote_dir()
        logging.info(f"local_dir{local_dir}, remote_dir{remote_dir}")
        result = [output1.cpu(), output1.dtype]
        output2.zero_()
        # torch.save(result, f"{local_dir}/result.bin")
        return output1, output2

    def get_cpp_func_signature_type(self):
        return """aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(
        const aclTensor *query,
        const aclTensor *key,
        const aclTensor *value,
        const aclTensor *attentionMaskOptional,
        const aclTensor *blockTableOptional,
        const aclIntArray *actualQSeqLenOptional,
        const aclIntArray *actualCmpKvSeqLenOptional,
        const aclIntArray *actualSelKvSeqLenOptional,
        const aclTensor *topKMaskOptional,
        int64_t numHeads,
        int64_t numKeyValueHeads,
        int64_t selectBlockSize,
        int64_t selectBlockCount,
        int64_t compressBlockSize,
        int64_t compressBlockStride,
        double scaleValue,
        char *layoutOptional,
        int64_t pageBlockSize,
        int64_t sparseMode,
        const aclTensor *output,
        const aclTensor *topKOutput,
        uint64_t *workspaceSize,
        aclOpExecutor **executor)"""
