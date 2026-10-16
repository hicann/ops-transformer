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
 * \file qfmla_metadata_checker.cpp
 * \brief Checker for metadata parameter ( 公共参数组 - metadata)
 */

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "../qfmla_tiling_info.h"
#include "../qfmla_tiling_info_parser.h"
#include "qfmla_metadata_checker.h"

namespace optiling {
namespace quant_flash_mla {

using namespace ge;

// ============================================================================
// SinglePara — metadata dtype/format/shape 校验
// ============================================================================

ge::graphStatus MetadataChecker::CheckSinglePara(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    // metadata 传入时校验 dtype=int32
    if (qfmlaInfo.opParamInfo.metadata.desc != nullptr) {
        OP_CHECK_IF(
            qfmlaInfo.opParamInfo.metadata.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName, METADATA_NAME,
                ge::TypeUtils::DataTypeToSerialString(qfmlaInfo.opParamInfo.metadata.desc->GetDataType()).c_str(),
                "The dtype of metadata must be int32"),
            return ge::GRAPH_FAILED);

        // format: ND
        if (CheckFormatSupport(qfmlaInfo.opParamInfo.metadata.desc, METADATA_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        // metadata 必须为 2D: (2, max_schedule_size)
        // dim0=2: 第一维存 AICore 调度数据, 第二维存 AIVCore 调度数据
        // dim1=max_schedule_size: 由 quant_flash_mla_metadata 动态计算, 必须 > 0
        const auto& metaShape = qfmlaInfo.opParamInfo.metadata.shape->GetStorageShape();
        OP_CHECK_IF(metaShape.GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        opName, METADATA_NAME, (std::to_string(metaShape.GetDimNum()) + "D").c_str(),
                        "The shape dim of metadata must be 2D (2, max_schedule_size)"),
                    return ge::GRAPH_FAILED);

        // dim0 必须为 2 (AICore + AIVCore)
        OP_CHECK_IF(metaShape.GetDim(0) != 2,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, METADATA_NAME, ("[" + std::to_string(metaShape.GetDim(0)) + ", ...]").c_str(),
                        "The dim0 of metadata must be 2 (AICore + AIVCore)"),
                    return ge::GRAPH_FAILED);

        // dim1 (max_schedule_size) 必须 > 0
        OP_CHECK_IF(metaShape.GetDim(1) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName, METADATA_NAME, ("[..., " + std::to_string(metaShape.GetDim(1)) + "]").c_str(),
                        "The dim1 (max_schedule_size) of metadata must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence — metadata 必须传入
// ============================================================================

ge::graphStatus MetadataChecker::CheckParaExistence(const QfmlaTilingInfo& qfmlaInfo)
{
    const char* opName = qfmlaInfo.opName;
    OP_CHECK_IF(!qfmlaInfo.metadataFlag,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, METADATA_NAME, "provided",
                                                      "metadata must be provided, it is required for scheduling"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_mla
} // namespace optiling
