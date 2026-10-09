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
 * \file sinks_checker.h
 * \brief Checker for sinks parameter (文档参数名: sinks, 文档约束: LearnableSink参数组)
 */

#ifndef MIXED_QUANT_FLASH_ATTN_SINKS_CHECKER_H
#define MIXED_QUANT_FLASH_ATTN_SINKS_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "mqfa_base_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {

class SinksChecker : public FABaseChecker {
public:
    SinksChecker() = default;
    ~SinksChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo& faInfo) override;
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MIXED_QUANT_FLASH_ATTN_SINKS_CHECKER_H
