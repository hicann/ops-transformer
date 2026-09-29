# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 本目录为 experimental LIV2 的 torch 接口，仅在 --experimental 单独打包形态下入包
# （setup.py TORCH_EXTENSION_EXPERIMENTAL 门控，PR #12715：带 --experimental 只收 experimental
# 目录、不带则完全不收），与现网 attention/lightning_indexer_v2 的 torch_extension 互斥收录，
# 现网包恒为现网原版、无覆盖面。本包自包含导出全部接口（lightning_indexer /
# lightning_indexer_metadata 与现网版同名同 schema，lightning_indexer 旧入口内部固定以
# candidate 关闭路径调用新签名 aclnn；lightning_indexer_candidate 为 experimental 扩展）；
# 注册方式与仓库其他算子一致（无条件 define/impl，不做逐算子幂等守卫）
__all__ = [
    "lightning_indexer",
    "lightning_indexer_v2",
    "lightning_indexer_metadata",
    "lightning_indexer_candidate",
]

from .lightning_indexer import (
    lightning_indexer,
    lightning_indexer_metadata,
    lightning_indexer_candidate,
)

lightning_indexer_v2 = lightning_indexer

from . import graph_convert_lightning_indexer
