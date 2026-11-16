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
 * \brief Checker for metadata parameter (文档: INT32, shape=(max_schedule_size,))
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../mqfa_fa_tiling_info.h"
#include "mqfa_metadata_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus MetadataChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    auto& metadataTensor = faInfo.opParamInfo.metadata.tensor;
    if (metadataTensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "metadata", "metadata cannot be empty");
        return ge::GRAPH_FAILED;
    }

    const gert::CompileTimeTensorDesc* metadataDesc = faInfo.opParamInfo.metadata.desc;
    OP_CHECK_IF(metadataDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "TensorDesc of metadata"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(metadataDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "metadata",
                                          DataTypeToSerialString(metadataDesc->GetDataType()).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(metadataDesc, METADATA_NAME)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = metadataTensor->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "metadata", (std::to_string(dimNum) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    int64_t dim0 = metadataTensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(dim0 <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "metadata",
                                                      GetShapeStr(metadataTensor->GetStorageShape()).c_str(),
                                                      "The 0th axis of metadata must be greater than 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MetadataChecker::CheckMultiPara(const FaTilingInfo& faInfo)
{
    int64_t expectedSize = 0;
    int64_t dim0 = faInfo.opParamInfo.metadata.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(CheckScheduleSizeConsistency(faInfo, expectedSize) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    faInfo.opName, METADATA_NAME.c_str(), std::to_string(dim0).c_str(),
                    ("The shape size of metadata must be equal to " + std::to_string(expectedSize) +
                     ". The possible cause is that the parameters batch_size and num_heads_kv of"
                     " MixedQuantFlashAttnMetadata do not match the batch_size and num_heads_kv"
                     " obtained from q and k")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MetadataChecker::CheckScheduleSizeConsistency(const FaTilingInfo& faInfo, int64_t& expectedSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo.platformInfo);
    const int64_t coreSum =
        static_cast<int64_t>(ascendcPlatform.GetCoreNumAic()) + static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    constexpr int64_t metadataUnit = 16;    // head/FA/FD每个调度单元均为16个int32
    constexpr int64_t metadataAlign = 4096; // int32元素个数对齐粒度, 与分配侧保持一致
    const int64_t rawSize = (coreSum * faInfo.bSize * faInfo.n2Size + 1) * metadataUnit;
    expectedSize = (rawSize + metadataAlign - 1) / metadataAlign * metadataAlign;
    const int64_t dim0 = faInfo.opParamInfo.metadata.tensor->GetStorageShape().GetDim(0);
    return (dim0 == expectedSize) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
