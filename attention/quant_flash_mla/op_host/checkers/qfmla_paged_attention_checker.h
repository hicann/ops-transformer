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
 * \file qfmla_paged_attention_checker.h
 * \brief Checker for block_table, cache_seqlens, k_cache non-contiguous ( Paged Attention参数组)
 */

#ifndef QFMLA_PAGED_ATTENTION_CHECKER_H
#define QFMLA_PAGED_ATTENTION_CHECKER_H

#include "qfmla_base_checker.h"

namespace optiling {
namespace quant_flash_mla {

class PagedAttentionChecker : public QfmlaBaseChecker {
public:
    PagedAttentionChecker() = default;
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckFeature(const QfmlaTilingInfo& qfmlaInfo) override;

private:
    ge::graphStatus CheckSingleParaBlockTable(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckSingleParaCacheSeqlens(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckBlockSize(const QfmlaTilingInfo& qfmlaInfo);
    // --- Feature: k_cache 非连续 Tensor 支持校验 ---
    ge::graphStatus CheckNonContiguousSupport(const QfmlaTilingInfo& qfmlaInfo);
};

} // namespace quant_flash_mla
} // namespace optiling
#endif // QFMLA_PAGED_ATTENTION_CHECKER_H
