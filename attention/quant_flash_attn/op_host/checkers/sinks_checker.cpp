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

ge::graphStatus SinksChecker::CheckSinglePara(const QfaTilingInfo& qfaInfo)
{
    // 未传入 sinks 时直接通过
    if (!qfaInfo.sinksFlag) {
        return ge::GRAPH_SUCCESS;
    }

    // 特性交叉校验: sinks 仅在 DAV_3510 且 MXFP8 / GQA FP8 / HIF8 量化模式下支持
    const bool supportedArch = (qfaInfo.npuArch == NpuArch::DAV_3510);
    const bool supportedQuantMode =
        qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 ||
        qfaInfo.quantMode ==
            QfaQuantMode::A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 ||
        qfaInfo.quantMode == QfaQuantMode::A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32;
    OP_CHECK_IF(
        !supportedArch || !supportedQuantMode,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(qfaInfo.opName, SINKS_NAME.c_str(), "provided",
                                              "sinks is only supported on DAV_3510 with MXFP8/GQA FP8/HIF8 quant mode"),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* sinksDesc = qfaInfo.opParamInfo.sinks.desc;
    const gert::Tensor* sinksTensor = qfaInfo.opParamInfo.sinks.tensor;
    OP_CHECK_IF(sinksDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, "TensorDesc of sinks"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinksTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(qfaInfo.opName, "Tensor of sinks"),
                return ge::GRAPH_FAILED);

    // dtype 校验
    OP_CHECK_IF(sinksDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(qfaInfo.opName, SINKS_NAME.c_str(),
                                          DataTypeToSerialString(sinksDesc->GetDataType()).c_str(), "FLOAT32"),
                return ge::GRAPH_FAILED);

    // format 校验
    if (CheckFormatSupport(sinksDesc, SINKS_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& shape = sinksTensor->GetStorageShape();
    size_t dimNum = shape.GetDimNum();
    OP_CHECK_IF(
        dimNum != DIM_NUM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(qfaInfo.opName, SINKS_NAME.c_str(), (std::to_string(dimNum) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(sinksTensor->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(qfaInfo.opName, SINKS_NAME.c_str(), "non-ND", "ND"),
                return ge::GRAPH_FAILED);

    // shape 校验: 长度必须等于 n1 (query head 数)
    int64_t dim0 = static_cast<int64_t>(shape.GetDim(0));
    OP_CHECK_IF(dim0 != qfaInfo.n1Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    qfaInfo.opName, SINKS_NAME.c_str(), ("[" + std::to_string(dim0) + "]").c_str(),
                    ("The length of sinks must be " + std::to_string(qfaInfo.n1Size) + " (N1)").c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
