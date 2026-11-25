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
 * \file all_gather_add_tiling.cc
 * \brief
 */

#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "hccl/hccl_types.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "../../op_kernel/all_gather_add_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t TILE_NUM = 1;
constexpr uint32_t COMM_TURN = 2;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
} // namespace
namespace optiling {

static ge::graphStatus AllGatherParamsCheck(const gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    uint64_t inputDim0 = aShape->GetStorageShape().GetDim(0);
    uint64_t inputDim1 = aShape->GetStorageShape().GetDim(1);

    OP_CHECK_IF(inputDim0 == 0 || inputDim1 == 0, OP_LOGE(context->GetNodeName(), "the value is invalid"),
                return ge::GRAPH_FAILED);

    if (context->GetAttrs() == nullptr) {
        OP_LOGE(context->GetNodeName(), "get attrs failed");
        return ge::GRAPH_FAILED;
    }
    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    OP_CHECK_IF(group == nullptr, OP_LOGE(context->GetNodeName(), "group is nullptr. "), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 设置 hccl 高阶 API Tiling 结构体。
// 引擎按代际选择：A2/A3 用默认引擎（AICPU 高层 Hccl），A5 显式设 MTE/AIV(3)。
static void InitHcclParam(AllGatherAddTilingData* tilingData, const char* group, const std::string& socVersion)
{
    bool isA3 = (socVersion.find("910_93") != std::string::npos);
    bool isA5 = (socVersion.find("950") != std::string::npos);
    std::string algConfig = isA3 ? "AllGather=level0:doublering" : "AllGather=level0:fullmesh";
    Mc2CcTilingConfig mc2CcTilingConfig(group, HCCL_CMD_ALLGATHER, algConfig);
    if (isA5) {
        mc2CcTilingConfig.SetCommEngine(3); // 仅 A5 显式选 MTE/AIV 引擎；A2/A3 用默认引擎
    }
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
}

static ge::graphStatus AllGatherAddTilingFunc(gert::TilingContext* context)
{
    // 1.对参数进行校验
    OP_CHECK_IF(AllGatherParamsCheck(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "param is invalid"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAiv());
    // batch mode：所有核同时启动。MTE 裸窗口的跨卡 flag 同步依赖所有核同时启动。
    context->SetScheduleMode(1);

    // 2.获取WorkspaceSize信息
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 3.设置TilingData
    AllGatherAddTilingData* tilingData = context->GetTilingData<AllGatherAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    tilingData->commTurn =
        COMM_TURN; // 通信轮次（A2/A3 高层 Hccl 按轮次通算；A5 MTE 单轮发送，仅参与 blockElemNum 换算）
    tilingData->tileNum = TILE_NUM; // 仅 A2/A3 使用；A5 在下方按 UB 封顶重算
    tilingData->totalElemNum = context->GetInputTensor(1)->GetShapeSize();
    tilingData->blockElemNum = tilingData->totalElemNum / tilingData->commTurn /
                               context->GetBlockDim(); // 每次Add计算只处理前一次通信结果长度的数据
    tilingData->addTileElemNum = tilingData->blockElemNum / tilingData->tileNum;
    uint32_t rankSize = *context->GetAttrs()->GetAttrPointer<uint32_t>(static_cast<int>(1));
    tilingData->rankSize = rankSize; // all_gather 的 rankDim，kernel 用其切分每 rank 数据量
    tilingData->addCoresPerRank =
        context->GetBlockDim() / rankSize; // 仅 A2/A3 使用：每个 rank 分到的核数（计算地址偏移用）
    tilingData->gatherTileElemNum =
        tilingData->totalElemNum / rankSize / tilingData->commTurn; // 仅 A2/A3 使用：每轮通信数据长度

    // 读 socVersion（供 A5 专属校验和引擎选择使用）
    auto platformInfo = context->GetPlatformInfo();
    std::string socVersion;
    platformInfo->GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    bool isA5 = (socVersion.find("950") != std::string::npos);

    // A5 专属整除/对齐校验：A5 MTE kernel 里 aElemsPerRank、numBlocks、blockElemNum 都是整数除法，
    // 不整除会静默截断；240 half(480B) 为一块数据。addTileElemNum 由 maxTileHalf 对齐保证，尾块也 240 对齐。
    // A2/A3 走高层 Hccl，不使用 240-half 块，不套用这些校验（A2 核数不同会误报 561002）。
    if (isA5) {
        constexpr uint32_t BLOCK_DATA_HALF = 240; // 每块数据 240 half（与 kernel SPLIT_BLOCK_DATA_HALF 一致）
        constexpr uint32_t ADD_BUFFER_NUM = 2;    // 与 kernel ADD_BUFFER_NUM 一致
        uint64_t blockDim = context->GetBlockDim();
        OP_CHECK_IF(tilingData->totalElemNum % (COMM_TURN * blockDim) != 0,
                    OP_LOGE(context->GetNodeName(), "totalElemNum %u not divisible by commTurn*blockDim %lu",
                            tilingData->totalElemNum, COMM_TURN * blockDim),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tilingData->totalElemNum % rankSize != 0,
                    OP_LOGE(context->GetNodeName(), "totalElemNum %u not divisible by rankSize %u",
                            tilingData->totalElemNum, rankSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((tilingData->totalElemNum / rankSize) % BLOCK_DATA_HALF != 0,
                    OP_LOGE(context->GetNodeName(), "aElemsPerRank %u not divisible by 240",
                            tilingData->totalElemNum / rankSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((tilingData->blockElemNum * COMM_TURN) % BLOCK_DATA_HALF != 0,
                    OP_LOGE(context->GetNodeName(),
                            "blockElemNumTotal %u not divisible by 240, per-core gather boundary misaligned",
                            tilingData->blockElemNum * COMM_TURN),
                    return ge::GRAPH_FAILED);
        // 完整分块：把每核总数据按 UB 容量封顶的 tile 切分，kernel 外层循环 tileNum 次处理尾块。
        // UB 用量 = 3 组队列(gatherQ_/bQ_/cQ_) × ADD_BUFFER_NUM buffer × addTileElemNum × sizeof(half)
        //            + sendBuf_ + sendQ_（2 × SYNC_FLAG_NUM × SPLIT_BLOCK_HALF × sizeof(half)）
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "get UB size failed"), return ge::GRAPH_FAILED);
        constexpr uint64_t HALF_BYTES = sizeof(uint16_t);               // half = 2 字节
        constexpr uint64_t SEND_UB_BYTES = 2ULL * 8 * 256 * HALF_BYTES; // sendBuf_+sendQ_ = 8192B
        uint64_t maxTileHalf = (ubSize - SEND_UB_BYTES) / (3ULL * ADD_BUFFER_NUM * HALF_BYTES);
        maxTileHalf = (maxTileHalf / BLOCK_DATA_HALF) * BLOCK_DATA_HALF; // 按 240 half 对齐
        OP_CHECK_IF(maxTileHalf < BLOCK_DATA_HALF,
                    OP_LOGE(context->GetNodeName(), "UB too small for one tile: %lu", maxTileHalf),
                    return ge::GRAPH_FAILED);
        uint64_t blockElemNumTotal = static_cast<uint64_t>(tilingData->blockElemNum) * COMM_TURN;
        uint64_t mteTile = blockElemNumTotal < maxTileHalf ? blockElemNumTotal : maxTileHalf;
        uint32_t mteTileNum = static_cast<uint32_t>((blockElemNumTotal + mteTile - 1) / mteTile);
        tilingData->addTileElemNum = static_cast<uint32_t>(mteTile); // 封顶后的 tile
        tilingData->tileNum = mteTileNum;                            // tile 数（含尾块）
    }

    auto group = context->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    InitHcclParam(tilingData, group, socVersion);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherAdd).Tiling(AllGatherAddTilingFunc);
} // namespace optiling
