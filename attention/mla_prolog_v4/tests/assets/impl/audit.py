# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""在 TTK 计时和图捕获前执行参数契约检查。

底层存储、步长和偏移均由 CSV 声明，不在此构造 Cache 布局。
"""

import importlib.util
import logging
from pathlib import Path

_DONE = set()


def sibling(stem):
    path = Path(__file__).with_name(stem + ".py")
    spec = importlib.util.spec_from_file_location(str(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(adapter, args, attrs):
    helper = sibling("audit_reference")
    sample = adapter._load_under_test()
    key = id(sample)
    if key not in _DONE:
        helper.check_enum_contract(sample)
        _DONE.add(key)
    cache = args[7]
    logging.info(
        "cache: shape=%s stride=%s offset=%s",
        tuple(cache.shape),
        cache.stride(),
        cache.storage_offset(),
    )
    # 由 TTK 控制被测算子的首次真实执行，此处不执行计算审计。
    logging.info("非计算型契约检查通过；全部位于 TTK 计时之外")
