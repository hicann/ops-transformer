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
 * \file quant_checker.h
 * \brief Checker for quant_mode, q_descale, k_descale, v_descale, p_scale, layout_q_descale
 *        (文档约束: 全量化参数组)
 */

#ifndef QUANT_FLASH_ATTN_QUANT_CHECKER_H
#define QUANT_FLASH_ATTN_QUANT_CHECKER_H

#include <map>
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace quant_flash_attn {

class QuantChecker : public QfaBaseChecker {
public:
    QuantChecker() = default;
    ~QuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckParaExistence(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckFeature(const QuantFlashAttnTilingInfo &qfaInfo) override;
    ge::graphStatus CheckMultiPara(const QuantFlashAttnTilingInfo &qfaInfo) override;

private:
    // --- SinglePara ---
    ge::graphStatus CheckSingleParaQuantMode(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaQDescale(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaKDescale(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaVDescale(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaPScale(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckSingleParaLayoutQDescale(const QuantFlashAttnTilingInfo &qfaInfo);

    // --- MultiPara: descale shape consistency (descale_shape匹配关系表) ---
    // 注: v_descale shape 不做校验, 仅校验 q_descale / k_descale
    ge::graphStatus CheckDescaleShape(const QuantFlashAttnTilingInfo &qfaInfo);
    ge::graphStatus CheckQDescaleShape(const QuantFlashAttnTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKDescaleShape(const QuantFlashAttnTilingInfo &qfaInfo) const;

    // --- MultiPara: descale dtype 校验 (descale_dtype匹配关系表) ---
    // MxFP8/MxFP4 场景下, q/k/v descale 的 tensor_type 仅支持 FLOAT8_E8M0
    ge::graphStatus CheckDescaleDtype(const QuantFlashAttnTilingInfo &qfaInfo) const;

    // 校验实际 shape 与期望 shape 的每个维度是否相等
    ge::graphStatus CheckShapeEqual(const gert::StorageShape &actual, const std::vector<int64_t> &expected,
                                    const std::string &paraName, const char *opName) const;

    // --- Feature: layout 匹配关系校验 (文档: layout匹配关系表) ---
    ge::graphStatus CheckLayoutConstraint(const QuantFlashAttnTilingInfo &qfaInfo) const;

    // --- Feature: q/k/v/attn_out shape 校验 (文档: q/k/v/attn_out shape匹配关系表) ---
    ge::graphStatus CheckShapeMatch(const QuantFlashAttnTilingInfo &qfaInfo) const;
    ge::graphStatus CheckQueryShape(const QuantFlashAttnTilingInfo &qfaInfo) const;
    ge::graphStatus CheckKVShape(const QuantFlashAttnTilingInfo &qfaInfo) const;
    ge::graphStatus CheckAttnOutShape(const QuantFlashAttnTilingInfo &qfaInfo) const;

    // --- Feature: MxFP4 场景特殊约束 ---
    ge::graphStatus CheckMxFp4Constraint(const QuantFlashAttnTilingInfo &qfaInfo) const;
    ge::graphStatus CheckMxFp4QkvDtype(const QuantFlashAttnTilingInfo &qfaInfo) const;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QUANT_CHECKER_H
