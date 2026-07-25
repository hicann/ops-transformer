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
 * \file softmax_lse_checker.cpp
 * \brief Checker for return_softmax_lse, softmax_lse (文档约束: SoftmaxLSE参数组)
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../quant_flash_attn_tiling_info.h"
#include "softmax_lse_checker.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;

ge::graphStatus SoftmaxLSEChecker::CheckSingleParaReturnSoftmaxLse(const QuantFlashAttnTilingInfo &qfaInfo)
{
    const bool *returnSoftmaxLse = qfaInfo.opParamInfo.returnSoftMaxLse;
    if (returnSoftmaxLse == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    bool returnSoftmaxLseVal = *returnSoftmaxLse;
    OP_CHECK_IF(returnSoftmaxLseVal != false && returnSoftmaxLseVal != true,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, RETURN_SOFTMAX_LSE_NAME.c_str(),
                                                      (returnSoftmaxLseVal ? "true" : "false"),
                                                      "The value of return_softmax_lse must be True or False"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckSingleParaSoftmaxLse(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (qfaInfo.returnSoftmaxLse == 0) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *lseOutDesc = qfaInfo.opParamInfo.lseOut.desc;
    if (lseOutDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(lseOutDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SOFTMAX_LSE_NAME.c_str(),
                                          DataTypeToSerialStr(lseOutDesc->GetDataType()).c_str(), "FLOAT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(lseOutDesc, SOFTMAX_LSE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const QuantFlashAttnTilingInfo &qfaInfo)
{
    if (CheckSingleParaReturnSoftmaxLse(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSingleParaSoftmaxLse(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
