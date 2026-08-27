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
 * \file paged_attention_checker_quant_flash_attn.h
 * \brief Checker for block_table ( Paged Attention参数组)
 */

#ifndef PAGED_ATTENTION_CHECKER_QUANT_FLASH_ATTN_H
#define PAGED_ATTENTION_CHECKER_QUANT_FLASH_ATTN_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {

class PagedAttentionChecker : public QfaBaseChecker {
public:
    PagedAttentionChecker() = default;
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo) override;

private:
    ge::graphStatus CheckSingleParaBlockTable(const QfaTilingInfo &qfaInfo);
    // --- Feature: 非连续 Tensor 支持校验 ---
    // 仅 PA 场景(layout_kv 为 PA_BNBD 或 PA_NZ)时，k/v/k_descale/v_descale
    // 仅支持 0 轴和 1 轴非连续，其余轴必须连续；非 PA 场景均不支持非连续。
    ge::graphStatus CheckNonContiguousSupport(const QfaTilingInfo &qfaInfo) const;
    // 仅通过 layout_kv 判断是否为 PA 场景 (PA_BBND/PA_BNBD/PA_NZ)
    bool IsPageAttention(const QfaTilingInfo &qfaInfo) const;
    // --- Feature: blockSize 场景化校验 ---
    ge::graphStatus CheckBlockSizeMxFp8(const QfaTilingInfo &qfaInfo) const;
    ge::graphStatus CheckBlockSizeGqaFp8(const QfaTilingInfo &qfaInfo) const;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // PAGED_ATTENTION_CHECKER_QUANT_FLASH_ATTN_H
