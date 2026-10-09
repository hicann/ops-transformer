/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "log/log.h"
#include "../mqfa_fa_tiling_info.h"
#include "mqfa_quant_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
using namespace ge;
using namespace arch35FA;

ge::graphStatus QuantChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    const std::vector<int64_t> supportedQuantModes = {1, 2};
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckValueSupport(static_cast<int64_t>(faInfo.faQuantMode), supportedQuantModes),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "quant_compute_mode",
                                                      std::to_string(static_cast<int64_t>(faInfo.faQuantMode)).c_str(),
                                                      "quant_compute_mode must be 1 or 2"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        faInfo.kDescaleType != ge::DT_FLOAT8_E8M0 && faInfo.kDescaleType != ge::DT_HIFLOAT4_SCALE,
        OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "k_descale", DataTypeToSerialString(faInfo.kDescaleType).c_str(),
                                  "FLOAT8_E8M0 or HIFLOAT4_SCALE"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        faInfo.vDescaleType != ge::DT_FLOAT8_E8M0 && faInfo.vDescaleType != ge::DT_HIFLOAT4_SCALE,
        OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "v_descale", DataTypeToSerialString(faInfo.vDescaleType).c_str(),
                                  "FLOAT8_E8M0 or HIFLOAT4_SCALE"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.inputKvType != ge::DT_FLOAT4_E2M1 && faInfo.inputKvType != ge::DT_HIFLOAT4,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "k/v", DataTypeToSerialString(faInfo.inputKvType).c_str(),
                                          "FLOAT4_E2M1 or HIFLOAT4"),
                return ge::GRAPH_FAILED);

    // Shape dimension checks (single-para: only dim count, no layout dependency)
    const auto& kDescaleShape = faInfo.opParamInfo.kDescale.shape->GetStorageShape();
    const auto& vDescaleShape = faInfo.opParamInfo.vDescale.shape->GetStorageShape();
    OP_CHECK_IF(kDescaleShape.GetDimNum() != 4 && kDescaleShape.GetDimNum() != 5 && kDescaleShape.GetDimNum() != 6,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "k_descale",
                                             std::to_string(kDescaleShape.GetDimNum()).c_str(), "4D 、5D or 6D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vDescaleShape.GetDimNum() != 4 && vDescaleShape.GetDimNum() != 5 && vDescaleShape.GetDimNum() != 6,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "v_descale",
                                             std::to_string(vDescaleShape.GetDimNum()).c_str(), "4D 、5D or 6D"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckParaExistence(const FaTilingInfo& faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.quantComputeMode == nullptr,
                OP_LOGE(faInfo.opName, "quant_compute_mode", "quant_compute_mode must be provided"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.kDescale.desc == nullptr || faInfo.opParamInfo.kDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "k_descale"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.vDescale.desc == nullptr || faInfo.opParamInfo.vDescale.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "v_descale"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantChecker::CheckMultiPara(const FaTilingInfo& faInfo)
{
    OP_CHECK_IF(
        faInfo.kDescaleType != faInfo.vDescaleType,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            faInfo.opName, "v_descale", DataTypeToSerialString(faInfo.vDescaleType).c_str(),
            ("v_descale dtype must match k_descale dtype " + DataTypeToSerialString(faInfo.kDescaleType)).c_str()),
        return ge::GRAPH_FAILED);

    const auto& kDescaleShape = faInfo.opParamInfo.kDescale.shape->GetStorageShape();
    const auto& vDescaleShape = faInfo.opParamInfo.vDescale.shape->GetStorageShape();

    if (faInfo.kvLayout == FaLayout::PA_NZ) {
        if (faInfo.faQuantMode == FaQuantMode::A16C4_KV_MXFP4_SOFTMAX_FP32) {
            // k_descale: (num_blocks, KV_N, block_size/16, ceil(D/64), 16, 2)
            // v_descale: (num_blocks, KV_N, D/16, ceil(block_size/64), 16, 2)
            uint32_t n2Size = faInfo.n2Size;
            uint32_t blockSize = faInfo.blockSize;
            uint32_t headDim = faInfo.qkHeadDim;
            uint32_t ceilD64 = (headDim + 63) / 64;
            uint32_t ceilBs64 = (blockSize + 63) / 64;

            // k_descale: (num_blocks, KV_N, block_size/16, ceil(D/64), 16, 2)
            OP_CHECK_IF(kDescaleShape.GetDim(0) != faInfo.totalBlockNum,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                            ("dim[0] must be num_blocks=" + std::to_string(faInfo.totalBlockNum)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                kDescaleShape.GetDim(1) != n2Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                                                      ("dim[1] must be KV_N=" + std::to_string(n2Size)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(kDescaleShape.GetDim(2) != blockSize / 16,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                            ("dim[2] must be block_size/16=" + std::to_string(blockSize / 16)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                kDescaleShape.GetDim(3) != ceilD64,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                                                      ("dim[3] must be ceil(D/64)=" + std::to_string(ceilD64)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                kDescaleShape.GetDim(4) != 16 || kDescaleShape.GetDim(5) != 2,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                                                      "dim[4] must be 16, dim[5] must be 2"),
                return ge::GRAPH_FAILED);

            // v_descale: (num_blocks, KV_N, D/16, ceil(block_size/64), 16, 2)
            OP_CHECK_IF(vDescaleShape.GetDim(0) != faInfo.totalBlockNum,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                            ("dim[0] must be num_blocks=" + std::to_string(faInfo.totalBlockNum)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                vDescaleShape.GetDim(1) != n2Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                                                      ("dim[1] must be KV_N=" + std::to_string(n2Size)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                vDescaleShape.GetDim(2) != headDim / 16,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                                                      ("dim[2] must be D/16=" + std::to_string(headDim / 16)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(vDescaleShape.GetDim(3) != ceilBs64,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                            ("dim[3] must be ceil(block_size/64)=" + std::to_string(ceilBs64)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                vDescaleShape.GetDim(4) != 16 || vDescaleShape.GetDim(5) != 2,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                                                      "dim[4] must be 16, dim[5] must be 2"),
                return ge::GRAPH_FAILED);
        } else if (faInfo.faQuantMode == FaQuantMode::A16C4_KV_HIF4_SOFTMAX_FP32) {
            // k_descale: (num_blocks, KV_N, block_size/16, ceil(D/64), 16)
            // v_descale: (num_blocks, KV_N, D/16, ceil(block_size/64), 16)
            uint32_t n2Size = faInfo.n2Size;
            uint32_t blockSize = faInfo.blockSize;
            uint32_t headDim = faInfo.qkHeadDim;
            uint32_t ceilD64 = (headDim + 63) / 64;
            uint32_t ceilBs64 = (blockSize + 63) / 64;

            // k_descale: (num_blocks, KV_N, block_size/16, ceil(D/64), 16)
            OP_CHECK_IF(kDescaleShape.GetDim(0) != faInfo.totalBlockNum,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                            ("dim[0] must be num_blocks=" + std::to_string(faInfo.totalBlockNum)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                kDescaleShape.GetDim(1) != n2Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                                                      ("dim[1] must be KV_N=" + std::to_string(n2Size)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(kDescaleShape.GetDim(2) != blockSize / 16,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                            ("dim[2] must be block_size/16=" + std::to_string(blockSize / 16)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                kDescaleShape.GetDim(3) != ceilD64,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_descale", ToStringRaw(kDescaleShape).c_str(),
                                                      ("dim[3] must be ceil(D/64)=" + std::to_string(ceilD64)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(kDescaleShape.GetDim(4) != 16,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "k_descale",
                                                              ToStringRaw(kDescaleShape).c_str(), "dim[4] must be 16"),
                        return ge::GRAPH_FAILED);

            // v_descale: (num_blocks, KV_N, D/16, ceil(block_size/64), 16)
            OP_CHECK_IF(vDescaleShape.GetDim(0) != faInfo.totalBlockNum,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                            ("dim[0] must be num_blocks=" + std::to_string(faInfo.totalBlockNum)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                vDescaleShape.GetDim(1) != n2Size,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                                                      ("dim[1] must be KV_N=" + std::to_string(n2Size)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                vDescaleShape.GetDim(2) != headDim / 16,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                                                      ("dim[2] must be D/16=" + std::to_string(headDim / 16)).c_str()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(vDescaleShape.GetDim(3) != ceilBs64,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            faInfo.opName, "v_descale", ToStringRaw(vDescaleShape).c_str(),
                            ("dim[3] must be ceil(block_size/64)=" + std::to_string(ceilBs64)).c_str()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(vDescaleShape.GetDim(4) != 16,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "v_descale",
                                                              ToStringRaw(vDescaleShape).c_str(), "dim[4] must be 16"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace mixed_quant_flash_attn
} // namespace optiling
