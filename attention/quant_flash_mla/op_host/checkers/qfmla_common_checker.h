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
 * \file qfmla_common_checker.h
 * \brief Common checker for layout, dtype, shape, axis parameters ( 公共参数组)
 */

#ifndef QFMLA_COMMON_CHECKER_H
#define QFMLA_COMMON_CHECKER_H

#include "qfmla_base_checker.h"

namespace optiling {
namespace quant_flash_mla {

class CommonChecker : public QfmlaBaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckParaExistence(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckFeature(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo) override;

private:
    // --- SinglePara: layout 校验 ---
    ge::graphStatus CheckSingleParaLayout(const QfmlaTilingInfo& qfmlaInfo);
    // --- SinglePara: dtype 校验 ---
    ge::graphStatus CheckSingleParaDtype(const QfmlaTilingInfo& qfmlaInfo);
    // --- SinglePara: shape dim 校验 ---
    ge::graphStatus CheckSingleParaShapeDim(const QfmlaTilingInfo& qfmlaInfo);
    // --- SinglePara: softmax_scale 校验 ---
    ge::graphStatus CheckSingleParaSoftmaxScale(const QfmlaTilingInfo& qfmlaInfo);

    // --- Feature: 轴校验 (head_dim_qk, head_dim_v, KV_N, Q_N, S, B, Q_T) ---
    ge::graphStatus CheckAxis(const QfmlaTilingInfo& qfmlaInfo);
    // --- Feature: q shape 维度值校验 ---
    ge::graphStatus CheckQueryShape(const QfmlaTilingInfo& qfmlaInfo);
    // --- Feature: attn_out shape 维度值校验 ---
    ge::graphStatus CheckAttnOutShape(const QfmlaTilingInfo& qfmlaInfo);

    // --- MultiPara: q/k_cache dtype 一致性 ---
    ge::graphStatus CheckDtypeConsistency(const QfmlaTilingInfo& qfmlaInfo);
};

} // namespace quant_flash_mla
} // namespace optiling
#endif // QFMLA_COMMON_CHECKER_H
