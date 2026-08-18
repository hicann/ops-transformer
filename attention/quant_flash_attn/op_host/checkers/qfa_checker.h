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
 * \file qfa_checker.h
 * \brief QfaChecker using Composite Pattern - manages all checkers uniformly
 */

#ifndef QUANT_FLASH_ATTN_QFA_CHECKER_H
#define QUANT_FLASH_ATTN_QFA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"

#include "./common_checker_quant_flash_attn.h"
#include "./quant_checker.h"
#include "./mask_checker_quant_flash_attn.h"
#include "./metadata_checker_quant_flash_attn.h"
#include "./paged_attention_checker_quant_flash_attn.h"
#include "./seq_len_checker_quant_flash_attn.h"
#include "./sinks_checker_quant_flash_attn.h"
#include "./softmax_lse_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {

class QfaChecker {
public:
    QfaChecker() = default;
    ~QfaChecker() = default;

    ge::graphStatus Init(const QfaTilingInfo &qfaInfo);
    ge::graphStatus Process(const QfaTilingInfo &qfaInfo);

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo);

private:
    using CheckMethod = std::function<ge::graphStatus(QfaBaseChecker *, const QfaTilingInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const QfaTilingInfo &qfaInfo);

    void RegisterCheckers();

    std::vector<std::unique_ptr<QfaBaseChecker>> checkers_;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QFA_CHECKER_H
