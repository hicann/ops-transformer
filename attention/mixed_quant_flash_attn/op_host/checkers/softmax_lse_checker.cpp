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
 * \brief Checker for softmax_lse parameter (文档约束: SoftmaxLSE参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../mqfa_fa_tiling_info.h"
#include "mqfa_softmax_lse_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc* lseOutDesc = faInfo.opParamInfo.lseOut.desc;
    if (lseOutDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(lseOutDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "lse_out",
                                          Ops::Base::ToString(lseOutDesc->GetDataType()).c_str(), "FP32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(lseOutDesc, SOFTMAX_LSE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckParaExistence(const FaTilingInfo& faInfo)
{
    if (faInfo.softmaxLseFlag) {
        const gert::StorageShape* lseOutShape = faInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(lseOutShape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "lse_out", "empty",
                                                          "When softmaxLSE is enabled, lse_out cannot be empty"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiPara(const FaTilingInfo& faInfo)
{
    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::StorageShape* lseOutShape = faInfo.opParamInfo.lseOut.shape;
    if (lseOutShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const auto& lseShape = lseOutShape->GetStorageShape();
    uint32_t dimNum = lseShape.GetDimNum();

    if (faInfo.qLayout == FaLayout::TND) {
        // (Q_T, Q_N) — 2D
        OP_CHECK_IF(dimNum != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "softmax_lse", (std::to_string(dimNum) + "D").c_str(),
                                                 "2D (Q_T, Q_N) for TND layout"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(0) != faInfo.qTSize,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                                                  ("dim[0] must be Q_T=" + std::to_string(faInfo.qTSize)).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(1) != faInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                                                  ("dim[1] must be Q_N=" + std::to_string(faInfo.n1Size)).c_str()),
            return ge::GRAPH_FAILED);
    } else {
        // (B, Q_N, Q_S) — 3D for BNSD/BSND
        OP_CHECK_IF(dimNum != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "softmax_lse", (std::to_string(dimNum) + "D").c_str(),
                                                 "3D (B, Q_N, Q_S) for non-TND layout"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseShape.GetDim(0) != faInfo.bSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                                                          ("dim[0] must be B=" + std::to_string(faInfo.bSize)).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(1) != faInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                                                  ("dim[1] must be Q_N=" + std::to_string(faInfo.n1Size)).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(2) != faInfo.s1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                                                  ("dim[2] must be Q_S=" + std::to_string(faInfo.s1Size)).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
