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
 * \file mask_checker_quant_flash_attn.h
 * \brief Checker for mask_mode, attn_mask, win_left, win_right ( Mask参数组)
 */

#ifndef MASK_CHECKER_QUANT_FLASH_ATTN_H
#define MASK_CHECKER_QUANT_FLASH_ATTN_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {

class MaskChecker : public QfaBaseChecker {
public:
    MaskChecker() = default;
    ~MaskChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo) override;

private:
    ge::graphStatus CheckSingleParaMaskMode(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaAttnMask(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaWindowParams(const QfaTilingInfo &qfaInfo);

    // --- MultiPara: mask_mode 与 attn_mask 的存在性关系校验 ---
    ge::graphStatus CheckMaskModeAttnMaskConsistency(const QfaTilingInfo &qfaInfo);

    // --- Feature: MxFP8 场景下 mask_mode 取值约束 ---
    ge::graphStatus CheckMaskModeQuantMode(const QfaTilingInfo &qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // MASK_CHECKER_QUANT_FLASH_ATTN_H
