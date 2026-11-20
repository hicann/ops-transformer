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

"""准备 MLA v4 用例的 CPU 输入。

输入张量与生成的 NumPy 存储共享内存，钩子在 Golden 前原地改写，返回值不被使用。
cache_index_rule 指定确定性的槽位：unique 为每个 token 分配唯一槽位；
last 令第一个 token 写最后一个合法槽位；negative 全部写为 -1，跳过 Cache 写入；
upper 全部写为 block_number*block_size，用于越界检查。
合法非负槽位必须唯一且处于 [0, block_number*block_size) 范围内。
"""

import torch


def _rewrite_cache_index(cache_index, kv_cache, rule):
    tokens = int(cache_index.numel())
    block_number, block_size = int(kv_cache.shape[0]), int(kv_cache.shape[1])
    capacity = block_number * block_size
    if rule in (None, "", "unique"):
        if tokens > capacity:
            return
        cache_index.copy_(torch.arange(tokens, dtype=torch.int64))
    elif rule == "last":
        if tokens > capacity:
            return
        slots = torch.arange(tokens, dtype=torch.int64)
        slots[0] = capacity - 1
        cache_index.copy_(slots)
    elif rule == "negative":
        cache_index.fill_(-1)
    elif rule == "upper":
        cache_index.fill_(capacity)
    else:
        raise ValueError(f"unsupported cache_index_rule: {rule}")


def customize_inputs(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    cache_index_rule="unique",
    **kwargs,
):
    """按用例规则将 cache_index 改写为确定性的槽位索引。"""
    import mla_prolog_v4_ttk_ops as adapter

    transport = adapter._transport()
    # 语义 HiF8 经 CPU 参考编码后，以 uint8 载体传入 Cache 接口。
    if transport.is_hif8(kv_cache):
        transport.normalize_hif8(kv_cache)
        transport.remember_cache(kv_cache, kwargs.get("testcase_name"))
    cache_index = transport.to_torch(cache_index)
    kv_cache = transport.to_torch(kv_cache)
    if not torch.is_tensor(cache_index) or not torch.is_tensor(kv_cache):
        return
    _rewrite_cache_index(cache_index, kv_cache, cache_index_rule)
