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
 * \file softmax_lse_checker.cpp
 * \brief Checker for return_softmax_lse, softmax_lse ( SoftmaxLSE参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "softmax_lse_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    // 约束(单参数校验列):
    // return_softmax_lse: data_type 仅支持 BOOL, 值仅支持 True/False
    // softmax_lse: data_type 仅支持 FLOAT32
    const bool *returnSoftmaxLse = qfaInfo.opParamInfo.returnSoftMaxLse;
    if (returnSoftmaxLse != nullptr) {
        bool returnSoftmaxLseVal = *returnSoftmaxLse;
        OP_CHECK_IF(returnSoftmaxLseVal != false && returnSoftmaxLseVal != true,
                    OP_LOGE_FOR_INVALID_VALUE(qfaInfo.opName, RETURN_SOFTMAX_LSE_NAME.c_str(),
                                              (returnSoftmaxLseVal ? "true" : "false"), "True or False"),
                    return ge::GRAPH_FAILED);
    }

    if (!qfaInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc *lseOutDesc = qfaInfo.opParamInfo.lseOut.desc;
    if (lseOutDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(lseOutDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SOFTMAX_LSE_NAME.c_str(),
                                          DataTypeToSerialString(lseOutDesc->GetDataType()).c_str(), "FLOAT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(lseOutDesc, SOFTMAX_LSE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    // 约束(一致性校验列):
    // 当 return_softmax_lse 为 True 时, softmax_lse 必须非空
    if (qfaInfo.softmaxLseFlag) {
        const gert::StorageShape *lseOutShape = qfaInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(
            lseOutShape == nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, SOFTMAX_LSE_NAME.c_str(), "empty",
                                                  "When return_softmax_lse is True, softmax_lse cannot be empty"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
