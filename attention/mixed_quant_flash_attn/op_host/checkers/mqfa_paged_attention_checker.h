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
 * \file mqfa_paged_attention_checker.h
 * \brief Checker for PagedAttention parameters (文档约束: Paged Attention参数组)
 */

#ifndef MQFA_PAGED_ATTENTION_CHECKER_H
#define MQFA_PAGED_ATTENTION_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "mqfa_base_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {

class PagedAttentionChecker : public FABaseChecker {
public:
    PagedAttentionChecker() = default;
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo& faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo& faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo& faInfo) override;
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MQFA_PAGED_ATTENTION_CHECKER_H
