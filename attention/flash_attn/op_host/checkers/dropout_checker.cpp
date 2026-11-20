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
 * \file dropout_checker.cpp
 * \brief Checker for dropout parameter (文档约束: Dropout参数组 dropout_p/seed/offset)
 */

#include <cmath>
#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "dropout_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

static ge::graphStatus CheckRngTensor(const FaTilingInfo& faInfo, const FAOptionalParaInfo& param,
                                      const std::string& name)
{
    if (param.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(param.desc == nullptr || param.desc->GetDataType() != ge::DT_UINT64,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, name.c_str(), "expected a uint64 Tensor"),
                return ge::GRAPH_FAILED);
    const auto& shape = param.tensor->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || shape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, name.c_str(), "expected shape (1,)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropoutChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    auto& dropoutPPtr = faInfo.opParamInfo.dropoutP;
    if (dropoutPPtr != nullptr) {
        float dropoutP = *dropoutPPtr;
        OP_CHECK_IF(!std::isfinite(dropoutP) || dropoutP < 0.0f || dropoutP >= 1.0f,
                    OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, DROPOUT_P_NAME, std::to_string(dropoutP).c_str(),
                                              "finite value in [0, 1)"),
                    return ge::GRAPH_FAILED);
    }

    if (CheckRngTensor(faInfo, faInfo.opParamInfo.seed, SEED_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckRngTensor(faInfo, faInfo.opParamInfo.offset, OFFSET_NAME);
}

ge::graphStatus DropoutChecker::CheckMultiPara(const FaTilingInfo& faInfo)
{
    OP_CHECK_IF(
        !faInfo.hasDrop && (faInfo.opParamInfo.seed.tensor != nullptr || faInfo.opParamInfo.offset.tensor != nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "seed, offset",
                                                 "seed and offset must be absent when dropout_p is 0"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.hasDrop && faInfo.pageAttentionFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    faInfo.opName, "layout_kv", "dropout does not support paged KV cache (PA_BBND/PA_BNBD/PA_NZ)"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
