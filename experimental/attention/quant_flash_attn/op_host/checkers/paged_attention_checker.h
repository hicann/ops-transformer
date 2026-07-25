/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_attention_checker.h
 * \brief Checker for block_table (文档约束: Paged Attention参数组)
 */

#ifndef QUANT_FLASH_ATTN_PAGED_ATTENTION_CHECKER_H
#define QUANT_FLASH_ATTN_PAGED_ATTENTION_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace quant_flash_attn {

class PagedAttentionChecker : public QfaBaseChecker {
public:
    PagedAttentionChecker() = default;
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo) override;

private:
    ge::graphStatus CheckSingleParaBlockTable(const QuantFlashAttnTilingInfo &qfaInfo);
    // 仅通过 layout_kv 判断是否为 PA 场景 (BnBsH/BnNBsD/NZ)
    bool IsPageAttention(const QuantFlashAttnTilingInfo &qfaInfo) const;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_PAGED_ATTENTION_CHECKER_H
