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
 * \file sinks_checker.cpp
 * \brief Checker for sinks parameter ( Sinks参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "sinks_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

ge::graphStatus SinksChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验列): 暂不支持
    // 传入 sinks 时直接报错
    if (qfaInfo.sinksFlag) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, "sinks", "provided", "sinks is currently not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
