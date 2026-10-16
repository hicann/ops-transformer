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
 * \file qfmla_softmax_lse_checker.h
 * \brief Checker for return_softmax_lse, softmax_lse ( SoftmaxLSE参数组)
 */

#ifndef QFMLA_SOFTMAX_LSE_CHECKER_H
#define QFMLA_SOFTMAX_LSE_CHECKER_H

#include "qfmla_base_checker.h"

namespace optiling {
namespace quant_flash_mla {

class SoftmaxLSEChecker : public QfmlaBaseChecker {
public:
    SoftmaxLSEChecker() = default;
    ~SoftmaxLSEChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo) override;
    ge::graphStatus CheckMultiPara(const QfmlaTilingInfo& qfmlaInfo) override;
};

} // namespace quant_flash_mla
} // namespace optiling
#endif // QFMLA_SOFTMAX_LSE_CHECKER_H
