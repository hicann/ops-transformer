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
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "softmax_lse",
                                          DataTypeToSerialString(lseOutDesc->GetDataType()).c_str(), "FLOAT"),
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
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "softmax_lse",
                                                             "When return_softmax_lse is true, softmax_lse "
                                                             "cannot be empty"),
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
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "softmax_lse",
                                                             (std::to_string(dimNum) + "D").c_str(),
                                                             "When layout_q is TND, the shape dim of softmax_lse "
                                                             "must be 2D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(0) != faInfo.qTSize,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                                                  "When layout_q is TND, the size of the 0th axis of softmax_lse "
                                                  "must be equal to the size of the 0th axis of q"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(1) != faInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                "When layout_q is TND, the size of the 1st axis of softmax_lse must be equal to the size of the "
                "1st axis of q"),
            return ge::GRAPH_FAILED);
    } else {
        // (B, Q_N, Q_S) — 3D for BNSD/BSND
        OP_CHECK_IF(dimNum != 3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(faInfo.opName, "softmax_lse",
                                                             (std::to_string(dimNum) + "D").c_str(),
                                                             "When layout_q is not TND, the shape dim of softmax_lse "
                                                             "must be 3D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseShape.GetDim(0) != faInfo.bSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                        "When layout_q is not TND, the size of the 0th axis of softmax_lse must be equal to the "
                        "size of the 0th axis of q"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            lseShape.GetDim(1) != faInfo.n1Size,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                "When layout_q is not TND, the size of the 1st axis of softmax_lse must be equal to num heads of "
                "q"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(lseShape.GetDim(2) != faInfo.s1Size,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        faInfo.opName, "softmax_lse", ToStringRaw(lseShape).c_str(),
                        "When layout_q is not TND, the size of the 2nd axis of softmax_lse must be equal to sequence "
                        "length of q"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
