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
 * \file quant_reduce_scatter_tiling.cpp
 * \brief host侧tiling实现
 */
#include <register/op_def_registry.h>
#include "../../op_kernel/quant_reduce_scatter_tiling_key.h"
#include "../../op_kernel/quant_reduce_scatter_tiling_data.h"
#include "common/quant_reduce_scatter_util_tiling.h"

namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

/**
 * @brief 打印tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void PrintTilingDataInfo(const gert::TilingContext *context, QuantReduceScatterTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "bs is %lu in quant_reduce_scatter.", tilingData.quantReduceScatterTilingInfo.bs);
    OP_LOGD(nodeName, "hiddenSize is %u in quant_reduce_scatter.", tilingData.quantReduceScatterTilingInfo.hiddenSize);
    OP_LOGD(nodeName, "scaleHiddenSize is %lu in quant_reduce_scatter.",
            tilingData.quantReduceScatterTilingInfo.scaleHiddenSize);
    OP_LOGD(nodeName, "aivNum is %u in quant_reduce_scatter.", tilingData.quantReduceScatterTilingInfo.aivNum);
}

/**
 * @brief 设置tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void SetTilingData(gert::TilingContext *context, QuantReduceScatterTilingData &tilingData)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // set tilingData
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum));
    tilingData.quantReduceScatterTilingInfo.aivNum = aivNum;
    uint64_t xValueBS = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    // 3d场景，context->GetInputShape在函数CheckInputTensorDim中已经校验
    if (context->GetInputShape(X_INDEX)->GetStorageShape().GetDimNum() == THREE_DIMS) {
        xValueBS = xValueBS * context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    }
    tilingData.quantReduceScatterTilingInfo.bs = xValueBS;
    tilingData.quantReduceScatterTilingInfo.hiddenSize = xValueH;
    tilingData.quantReduceScatterTilingInfo.scaleHiddenSize = scalesValueH;
    tilingData.quantReduceScatterTilingInfo.totalWinSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
}

// 基于 TARGET_ITER 公式计算 host 推荐的 xPerBlock（先除 rankSize 再反推），写入 tilingData
static void SetXPerBlock(QuantReduceScatterTilingData &tilingData, const TilingRunInfo &runInfo)
{
    constexpr uint32_t TARGET_ITER = 3U;    // 与 QAR 对称
    constexpr uint32_t MIN_BLOCK = 2048U;   // QRS comm-bound：per_core 太小公式失效，MIN 兜到 2048
    constexpr uint32_t ALIGN_BLOCK = 1024U; // 与 kernel X_BLOCK_ALIGN_NUM 对齐
    uint64_t xNums = tilingData.quantReduceScatterTilingInfo.bs * tilingData.quantReduceScatterTilingInfo.hiddenSize;
    uint64_t aivNum = tilingData.quantReduceScatterTilingInfo.aivNum;
    uint64_t rankSize = static_cast<uint64_t>(runInfo.rankSize);
    uint64_t xSliceSizeNums = xNums / rankSize;
    uint64_t perCoreElem = (xSliceSizeNums + aivNum - 1U) / aivNum;
    uint64_t xPerBlock = (perCoreElem + TARGET_ITER - 1U) / TARGET_ITER;
    xPerBlock = std::max<uint64_t>(xPerBlock, MIN_BLOCK);
    xPerBlock = (xPerBlock / ALIGN_BLOCK) * ALIGN_BLOCK;
    tilingData.quantReduceScatterTilingInfo.xPerBlock = static_cast<uint32_t>(xPerBlock);
    tilingData.quantReduceScatterTilingInfo.alignBlock = ALIGN_BLOCK;
}

/**
 * @brief 设置tilingKey
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 设置tilingKey模板参数
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MTE_COMM);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in quant_reduce_scatter.", tilingKey);
}

/**
 * @brief quant_reduce_scatter算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QuantReduceScatterTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("quant_reduce_scatter", "context"),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT("quant_reduce_scatter", "nodeName"),
                    return ge::GRAPH_FAILED);

    QuantReduceScatterTilingData *tilingData = context->GetTilingData<QuantReduceScatterTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    TilingRunInfo runInfo = {};
    OP_TILING_CHECK(QuantReduceScatterUtilTiling::CheckNpuArch(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "npuArch", "non-DAV_3510",
                                                          "The value of npuArch must be DAV_3510"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(QuantReduceScatterUtilTiling::CheckTilingFunc(context, runInfo, OpType::OP_QUANT_REDUCE_SCATTER) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tiling check failed in quant_reduce_scatter."), return ge::GRAPH_FAILED);

    SetTilingData(context, *tilingData);
    tilingData->quantReduceScatterTilingInfo.hcclBufferSize = runInfo.hcclBufferSize;
    SetXPerBlock(*tilingData, runInfo);
    SetTilingKey(context);
    PrintTilingDataInfo(context, *tilingData);
    return ge::GRAPH_SUCCESS;
}

struct QuantReduceScatterCompileInfo {};

ge::graphStatus TilingParseForQuantReduceScatter(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantReduceScatter)
    .Tiling(QuantReduceScatterTilingFunc)
    .TilingParse<QuantReduceScatterCompileInfo>(TilingParseForQuantReduceScatter);

} // namespace MC2Tiling
