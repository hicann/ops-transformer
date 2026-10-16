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
 * \file qfmla_quant_checker.h
 * \brief Checker for quant_mode, q_descale, k_descale ( 全量化参数组)
 */

#ifndef QFMLA_QUANT_CHECKER_H
#define QFMLA_QUANT_CHECKER_H

#include "qfmla_base_checker.h"

namespace optiling {
namespace quant_flash_mla {

class QuantChecker : public QfmlaBaseChecker {
public:
    QuantChecker() = default;
    ~QuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckParaExistence(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckFeature(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo) override;

private:
    // --- SinglePara ---
    ge::graphStatus CheckSingleParaQuantMode(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckSingleParaQDescale(const QfmlaTilingInfo& qfmlaInfo);
    ge::graphStatus CheckSingleParaKDescale(const QfmlaTilingInfo& qfmlaInfo);

    // --- MultiPara: q_descale shape 与 layout_q 匹配 ---
    ge::graphStatus CheckQDescaleShape(const QfmlaTilingInfo& qfmlaInfo);
    // --- MultiPara: k_descale shape 校验 (1,) ---
    ge::graphStatus CheckKDescaleShape(const QfmlaTilingInfo& qfmlaInfo);
};

} // namespace quant_flash_mla
} // namespace optiling
#endif // QFMLA_QUANT_CHECKER_H
