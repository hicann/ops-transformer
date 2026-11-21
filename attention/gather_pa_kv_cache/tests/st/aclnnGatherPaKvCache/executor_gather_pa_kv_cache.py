# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import random
import numpy as np
import logging
import ctypes
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import torch.nn as nn


@register("function_GatherPaKvCache")
class GatherPaKvCacheApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(GatherPaKvCacheApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        random.seed(self.task_result.case_config.id)
        np.random.seed(self.task_result.case_config.id)
        torch.manual_seed(self.task_result.case_config.id)
        dtype = random.choice([np.float16])
        isSeqLensCumsum = input_data.kwargs["isSeqLensCumsum"]
        batch = random.randint(1, 4)

        seq_len = random.randint(1, 3)
        num_heads = random.choice([32, 64])  # num_heads生成1到128之间的随机数

        head_size_k = random.randint(1, 12)  # head_size_k 生成1到128之间的随机数

        head_size_v = random.randint(1, 12)  # head_size_v 生成1到128之间的随机数

        block_size = random.randint(1, 2) * 16  # block_size 生成1到128之间的随机数

        num_blocks = random.randint(1, 12)  # num_blocks 生成1到128之间的随机数
        num_tokens = batch * seq_len
        key_cache = np.random.randint(
            1, 11, size=(num_blocks, block_size, num_heads, head_size_k)
        ).astype(dtype)
        value_cache = np.random.randint(
            1, 11, size=(num_blocks, block_size, num_heads, head_size_v)
        ).astype(dtype)

        if key_cache.dtype == "float32":
            key_cache = torch.from_numpy(key_cache).type(torch.bfloat16)
            value_cache = torch.from_numpy(value_cache).type(torch.bfloat16)
        else:
            key_cache = torch.from_numpy(key_cache)
            value_cache = torch.from_numpy(value_cache)
        if self.device == "cpu":
            input_data.kwargs["keyCache"] = key_cache
            input_data.kwargs["valueCache"] = value_cache
        elif self.device == "pyaclnn":
            input_data.kwargs["keyCache"] = key_cache.npu()
            input_data.kwargs["valueCache"] = value_cache.npu()
        if isSeqLensCumsum:
            context_lens = [random.randint(1, 1024) for _ in range(num_tokens)]
            max_context_len = max(context_lens)
            cu_context_lens = [0]
            for elem in context_lens:
                cu_context_lens.append(cu_context_lens[-1] + elem)
            max_num_blocks_per_req = (
                max_context_len + block_size - 1
            ) // block_size + 4
            block_tables = []
            for _ in range(num_tokens):
                block_table = [
                    random.randint(0, num_blocks - 1)
                    for _ in range(max_num_blocks_per_req)
                ]
                block_tables.append(block_table)
            seq_starts = []
            seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]
            seq_starts = np.array(seq_starts).astype(np.int32)
            block_tables = np.array(block_tables).astype(np.int32)
            context_lens = np.array(cu_context_lens).astype(np.int32)
            sum_context_lens = context_lens[-1]

            key_expect = torch.zeros(
                (sum_context_lens, num_heads, head_size_k), dtype=key_cache.dtype
            )
            value_expect = torch.zeros(
                (sum_context_lens, num_heads, head_size_v), dtype=value_cache.dtype
            )
            key = torch.zeros(
                (sum_context_lens, num_heads, head_size_k), dtype=key_cache.dtype
            )
            value = torch.zeros(
                (sum_context_lens, num_heads, head_size_v), dtype=value_cache.dtype
            )
            if self.device == "cpu":
                input_data.kwargs["seqOffset"] = torch.from_numpy(seq_starts)
                input_data.kwargs["blockTables"] = torch.from_numpy(block_tables)
                input_data.kwargs["seqLens"] = torch.from_numpy(context_lens)
                input_data.kwargs["keyRef"] = key
                input_data.kwargs["valueRef"] = value
            elif self.device == "pyaclnn":
                input_data.kwargs["seqOffset"] = torch.from_numpy(seq_starts).npu()
                input_data.kwargs["blockTables"] = torch.from_numpy(block_tables).npu()
                input_data.kwargs["seqLens"] = torch.from_numpy(context_lens).npu()
                input_data.kwargs["keyRef"] = key.npu()
                input_data.kwargs["valueRef"] = value.npu()
        else:
            context_lens = [random.randint(1, 1024) for _ in range(num_tokens)]
            max_context_len = max(context_lens)
            max_num_blocks_per_req = (
                max_context_len + block_size - 1
            ) // block_size + 4
            block_tables = []
            for _ in range(num_tokens):
                block_table = [
                    random.randint(0, num_blocks - 1)
                    for _ in range(max_num_blocks_per_req)
                ]
                block_tables.append(block_table)
            seq_starts = []
            seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]
            seq_starts = np.array(seq_starts).astype(np.int32)
            block_tables = np.array(block_tables).astype(np.int32)
            context_lens = np.array(context_lens).astype(np.int32)

            sum_context_lens = sum(context_lens)

            key_expect = torch.zeros(
                (sum_context_lens, num_heads, head_size_k), dtype=key_cache.dtype
            )
            value_expect = torch.zeros(
                (sum_context_lens, num_heads, head_size_v), dtype=value_cache.dtype
            )
            key = torch.zeros(
                (sum_context_lens, num_heads, head_size_k), dtype=key_cache.dtype
            )
            value = torch.zeros(
                (sum_context_lens, num_heads, head_size_v), dtype=value_cache.dtype
            )
            if self.device == "cpu":
                input_data.kwargs["seqOffset"] = torch.from_numpy(seq_starts)
                input_data.kwargs["blockTables"] = torch.from_numpy(block_tables)
                input_data.kwargs["seqLens"] = torch.from_numpy(context_lens)
                input_data.kwargs["keyRef"] = key
                input_data.kwargs["valueRef"] = value
            elif self.device == "pyaclnn":
                input_data.kwargs["seqOffset"] = torch.from_numpy(seq_starts).npu()
                input_data.kwargs["blockTables"] = torch.from_numpy(block_tables).npu()
                input_data.kwargs["seqLens"] = torch.from_numpy(context_lens).npu()
                input_data.kwargs["keyRef"] = key.npu()
                input_data.kwargs["valueRef"] = value.npu()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        key_cache = input_data.kwargs["keyCache"]
        value_cache = input_data.kwargs["valueCache"]
        block_tables = input_data.kwargs["blockTables"]
        context_lens = input_data.kwargs["seqLens"]
        keyRef = input_data.kwargs["keyRef"]
        valueRef = input_data.kwargs["valueRef"]
        seq_starts = input_data.kwargs["seqOffset"]
        cache_mode = input_data.kwargs["cacheMode"]
        is_seq_lens_cumsum = input_data.kwargs["isSeqLensCumsum"]
        num_tokens = block_tables.shape[0]
        block_size = key_cache.shape[1]
        key_expect = keyRef
        value_expect = valueRef
        if is_seq_lens_cumsum:
            kv_rslt_id = 0
            context_start = 0
            for i in range(num_tokens):
                block_table = block_tables[i]
                context_end = int(context_lens[i + 1])
                context_len = context_end - context_start
                context_start = context_end
                block_table_offset = 0
                block_table_offset = seq_starts[i] // block_size
                for j in range(context_len):
                    block_id = int(block_table[block_table_offset + j // block_size])
                    block_offset = j % block_size
                    if block_id < 0:
                        continue
                    temp_k = key_cache[block_id][block_offset]
                    temp_v = value_cache[block_id][block_offset]
                    key_expect[kv_rslt_id] = temp_k
                    value_expect[kv_rslt_id] = temp_v
                    kv_rslt_id += 1
            return key_expect, value_expect
        else:
            kv_rslt_id = 0
            for i in range(num_tokens):
                block_table = block_tables[i]
                context_len = int(context_lens[i])
                block_table_offset = 0
                block_table_offset = seq_starts[i] // block_size
                for j in range(context_len):
                    block_id = int(block_table[block_table_offset + j // block_size])
                    block_offset = j % block_size
                    if block_id < 0:
                        continue
                    temp_k = key_cache[block_id][block_offset]
                    temp_v = value_cache[block_id][block_offset]
                    key_expect[kv_rslt_id] = temp_k
                    value_expect[kv_rslt_id] = temp_v
                    kv_rslt_id += 1
            return key_expect, value_expect


@register("function_aclnnGatherPaKvCache")
class FunctionaclnnGatherPaKvCache(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        output_packages[0] = input_args[4]
        output_packages[1] = input_args[5]

        return input_args, output_packages
