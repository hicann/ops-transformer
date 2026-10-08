#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""op_summary CSV 解析(最小版, 抽取自 ops-transformer-testkit core/parser.py)。

按 OP Type 过滤目标算子的 (start_time, duration) 条目;
热启动统计: 丢弃第一次(预热), 其余取均值。
"""

import csv
from typing import List, Tuple

# msprof CSV 中量化 MLA 主算子的 "OP Type" 列值(AI Core/MIX_AIC)。
# metadata 算子(QuantFlashMlaWithKvcacheMetadata, AI_CPU)不会被匹配。
OP_TYPE = "QuantFlashMlaWithKvcache"


def parse_op_summary(csv_path, op_type: str = OP_TYPE) -> List[Tuple[float, float]]:
    """返回按开始时间排序的 (start_time_us, duration_us) 列表。"""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    entries = [
        (float(r["Task Start Time(us)"]), float(r["Task Duration(us)"]))
        for r in rows
        if r["OP Type"] == op_type
    ]
    entries.sort(key=lambda x: x[0])
    return entries


def compute_hot_avg(durations: List[float]) -> float:
    """热启动: 丢弃第一次(预热), 其余取均值; 只有一条时直接返回。"""
    if not durations:
        return -1.0
    if len(durations) > 1:
        keep = durations[1:]
        return sum(keep) / len(keep)
    return durations[0]
