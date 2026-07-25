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
 * \file common_checker.h
 * \brief Common checker for layout, shape, dtype parameters (文档约束: 公共参数组)
 */

#ifndef QUANT_FLASH_ATTN_COMMON_CHECKER_H
#define QUANT_FLASH_ATTN_COMMON_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace quant_flash_attn {

class CommonChecker : public QfaBaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo) override;

private:
    // --- Layout checks ---
    ge::graphStatus CheckSingleParaLayout(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- Dtype/shapeDim checks (SinglePara) ---
    ge::graphStatus CheckSingleParaDtype(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaShapeDim(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- Attr checks (SinglePara) ---
    ge::graphStatus CheckSingleParaSoftmaxScale(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- ParaExistence: metadata (公共参数组) ---
    ge::graphStatus CheckMetadataExistence(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- MultiPara: dtype consistency ---
    ge::graphStatus CheckDtypeConsistency(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- Feature: axis & headNum cross-check ---
    ge::graphStatus CheckAxis(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckHeadNum(const QuantFlashAttnTilingInfo &qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_COMMON_CHECKER_H
