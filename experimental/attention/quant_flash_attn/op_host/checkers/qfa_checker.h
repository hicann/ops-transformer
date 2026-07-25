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
 * \file qfa_checker.h
 * \brief QfaChecker using Composite Pattern - manages all checkers uniformly
 */

#ifndef QUANT_FLASH_ATTN_QFA_CHECKER_H
#define QUANT_FLASH_ATTN_QFA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"
#include "base_checker.h"

#include "./common_checker.h"
#include "./quant_checker.h"
#include "./mask_checker.h"
#include "./seq_len_checker.h"
#include "./paged_attention_checker.h"
#include "./sinks_checker.h"
#include "./softmax_lse_checker.h"

namespace optiling {
namespace quant_flash_attn {

class QfaChecker {
public:
    QfaChecker() = default;
    ~QfaChecker() = default;

    ge::graphStatus Init(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus Process(const QuantFlashAttnTilingInfo &qfaInfo);

    ge::graphStatus CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo);

private:
    using CheckMethod = std::function<ge::graphStatus(QfaBaseChecker *, const QuantFlashAttnTilingInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const QuantFlashAttnTilingInfo &qfaInfo);

    void RegisterCheckers();

    std::vector<std::unique_ptr<QfaBaseChecker>> checkers_;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QFA_CHECKER_H
