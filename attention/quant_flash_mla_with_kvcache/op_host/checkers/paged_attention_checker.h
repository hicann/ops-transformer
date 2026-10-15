/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_attention_checker.h
 * \brief Checker for block_table, cache_seqlens, k_cache non-contiguous ( Paged Attention参数组)
 */

#ifndef QMLA_PAGED_ATTENTION_CHECKER_H
#define QMLA_PAGED_ATTENTION_CHECKER_H

#include "base_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

class PagedAttentionChecker : public QmlaBaseChecker {
public:
    PagedAttentionChecker() = default;
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const QmlaTilingInfo& qmlaInfo) override;
    ge::graphStatus CheckFeature(const QmlaTilingInfo& qmlaInfo) override;

private:
    ge::graphStatus CheckSingleParaBlockTable(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckSingleParaCacheSeqlens(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckBlockSize(const QmlaTilingInfo& qmlaInfo);
    // --- Feature: k_cache 非连续 Tensor 支持校验 ---
    ge::graphStatus CheckNonContiguousSupport(const QmlaTilingInfo& qmlaInfo);
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
#endif // QMLA_PAGED_ATTENTION_CHECKER_H
