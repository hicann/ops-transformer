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
 * \file metadata_checker.cpp
 * \brief Checker for metadata parameter ( 公共参数组 - metadata)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../qfa_tiling_info.h"
#include "metadata_checker_quant_flash_attn.h"

namespace optiling {
namespace quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35QFA;

ge::graphStatus MetadataChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    // 单参数校验：仅当 metadata 传入时校验其自身属性（dtype/format/shape）。
    // 存在性校验（必须传入）由 CheckParaExistence 负责。
    const gert::Tensor *metadataTensor = qfaInfo.opParamInfo.metadata.tensor;
    const gert::CompileTimeTensorDesc *metadataDesc = qfaInfo.opParamInfo.metadata.desc;
    if (metadataTensor == nullptr || metadataDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // dtype 仅支持 INT32
    OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, METADATA_NAME.c_str(),
                                          DataTypeToSerialString(metadataDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    // format 仅支持 ND
    if (CheckFormatSupport(metadataDesc, METADATA_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // shape dim 支持 1 或 2: 1D 为 (max_schedule_size,), 仅正向 FA 调度数据;
    // 2D 为 (2, max_schedule_size), 第一维存正向 FA 调度数据, 第二维存反向 FAG 调度数据
    uint32_t dimNum = metadataTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != DIM_NUM_1 && dimNum != DIM_NUM_2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, METADATA_NAME.c_str(),
                                             (std::to_string(dimNum) + "D").c_str(), "1D or 2D"),
                return ge::GRAPH_FAILED);

    // shape dim0 必须 > 0（shape 由 quant_flash_attn_metadata 动态计算，不应为空）
    int64_t dim0 = metadataTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(dim0 <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qfaInfo.opName, METADATA_NAME.c_str(),
                                                      ("[" + std::to_string(dim0) + "]").c_str(),
                                                      "The value of dim0 of metadata shape must be greater than 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// ParaExistence — metadata 必须传入
// ============================================================================

ge::graphStatus MetadataChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    // 当前不支持不传入 metadata，未传入将发出拦截报警
    OP_CHECK_IF(qfaInfo.opParamInfo.metadata.tensor == nullptr || qfaInfo.opParamInfo.metadata.desc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, METADATA_NAME.c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
