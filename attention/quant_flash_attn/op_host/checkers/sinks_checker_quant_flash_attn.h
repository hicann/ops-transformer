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
 * \file sinks_checker_quant_flash_attn.h
 * \brief Checker for sinks parameter ( Sinks参数组)
 */

#ifndef SINKS_CHECKER_QUANT_FLASH_ATTN_H
#define SINKS_CHECKER_QUANT_FLASH_ATTN_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {

class SinksChecker : public QfaBaseChecker {
public:
    SinksChecker() = default;
    ~SinksChecker() override = default;

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo) override;
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // SINKS_CHECKER_QUANT_FLASH_ATTN_H
