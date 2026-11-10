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
 * \file matmul_reduce_scatter_v2_tiling.cpp
 * \brief
 */
#include "matmul_reduce_scatter_v2_tiling.h"
#include <algorithm>
#include <queue>
#include <stdlib.h>
#include <sys/types.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "reduce_scatter_comm_algo_table.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_tiling_utils.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "op_host/tiling_templates_registry.h"
#include "reduce_scatter_fit_balance_tiling.h"
#include "../reduce_scatter_formulaic_tiling.h"
#include "../../../op_kernel/matmul_reduce_scatter_v2_apt_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Log;
using namespace Mc2Tiling;

namespace optiling {
namespace {
constexpr char ACLNN_CCU_PEER_ONLY_MODE[] = "ccu_peer_only";
}

constexpr uint32_t X1SCALE_INDEX = 3;
constexpr uint32_t X2SCALE_INDEX = 4;
constexpr uint32_t COMMMODE_INDEX = 10;
// 新功能从这里开始
bool MatmulReduceScatterV2Tiling::IsCapable()
{
    if ((npuArch_ == Ops::Base::DAV_3510 || npuArch_ == Ops::Base::DAV_9201) && // A6复用A5 tiling路径
        ((args_.geAType == ge::DT_BF16) || (args_.geAType == ge::DT_FLOAT16))) {
        OP_LOGI(opName_, "start with MatmulReduceScatterV2Tiling tiling.");
        return true;
    }

    OP_LOGI(opName_, "skip MatmulReduceScatterV2Tiling tiling when input datatype is not fp16 or bf16.");
    return false;
}

void PrintMMV3TilingData(const std::string& opName, Mc2MatMulV3TilingData& tiling)
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, "tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, "tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, "tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, "tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, "tiling.mBaseTailSplitCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, "tiling.nBaseTailSplitCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, "tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, "tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, "tiling.aswWindowLen %d", tiling.aswWindowLen);
}

void MatmulReduceScatterV2Tiling::PrintAllTilingData() const
{
    if (matmulReduceScatterV2TilingData_->param.rankID == 0) {
        PrintRCSTilingData(context_->GetNodeName(), matmulReduceScatterV2TilingData_->param);
        OP_LOGD(opName_, "MutableMC2MmV3TileTilingData matmulTiling");
        PrintMMV3TilingData(context_->GetNodeName(), matmulReduceScatterV2TilingData_->mC2Mmv3TileTilingData);
        if (matmulReduceScatterV2TilingData_->param.tailM > 0) {
            OP_LOGD(opName_, "MutableMC2MmV3TileTilingData matmulTiling");
            PrintMMV3TilingData(context_->GetNodeName(), matmulReduceScatterV2TilingData_->mC2Mmv3TailTilingData);
        }
    }
}

ge::graphStatus MatmulReduceScatterV2Tiling::CheckInput()
{
    auto x1ScaleShape = context_->GetOptionalInputShape(X1SCALE_INDEX);
    OP_TILING_CHECK(x1ScaleShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1Scale", "not nullptr",
                                                          "If the dtype of x is fp16 or bf16, x1Scale must be nullptr"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleShape = context_->GetOptionalInputShape(X2SCALE_INDEX);
    OP_TILING_CHECK(x2ScaleShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x2Scale", "not nullptr",
                                                          "If the dtype of x is fp16 or bf16, x2Scale must be nullptr"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::SetMc2Hcomm()
{
    const uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    const uint32_t opType = isA2APath_ ? static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL) :
                                         static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_REDUCE_SCATTER);

    const std::string rsConfig = isA2APath_ ? "AlltoAll=level0:fullmesh" : "ReduceScatter=level0:fullmesh";

    int index = 0;
    auto group = context_->GetAttrs()->GetAttrPointer<char>(index++);
    OP_TILING_CHECK(group == nullptr || std::strlen(group) == 0, OP_LOGE_WITH_INVALID_INPUT(opName_, "group"),
                    return ge::GRAPH_FAILED);
    uint32_t dataType = static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType));

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, rsConfig, reduceType, dataType, dataType);
    if (commMode_ == TPL_AICPU_COMM_MODE) {
        mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_AICPU_TS_ENGINE);
        OP_LOGD(opName_, "[SetCommEngine] Set CommEngine to AiCPU for matmul_reduce_scatter_v2_tiling.");
    } else {
        mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
        OP_LOGD(opName_, "[SetCommEngine] Set CommEngine to CCU for matmul_reduce_scatter_v2_tiling.");
    }
    const auto commMode = context_->GetAttrs()->GetAttrPointer<char>(COMMMODE_INDEX);
    const bool enablePeerOnly = std::strcmp(commMode, ACLNN_CCU_PEER_ONLY_MODE) == 0;
    // APACE 保留本地 rank 的结果拷贝，仅禁用 PeerOnly 优化，不禁用 CCU。
    isPeerOnly_ = !usingApaceImpl_ && enablePeerOnly && !isA2APath_ && commMode_ == TPL_CCU_COMM_MODE &&
                  args_.rankDim == PEER_ONLY_RANK_SIZE && mc2tiling::Mc2TilingUtils::GetDebugMode() == 0;
    if (isPeerOnly_) {
        mc2CcTilingConfig.SetAlgConfig(PEER_ONLY_ALGORITHM);
        // The paired PeerOnly Matmul kernel writes the local contribution directly to output.
        OP_TILING_CHECK(mc2CcTilingConfig.SetSkipLocalRankCopy(1) != 0,
                        OP_LOGE(opName_, "SetSkipLocalRankCopy failed for peer-only MMRS"), return ge::GRAPH_FAILED);
    } else {
        // 通信算法统一由算法表管理(组网/卡数/数据量/优先级)：单层Mesh+CLOS混联形态命中具体表项用组网算法，
        // 16p非混联形态命中benchmark优选表项，其余命中通路对应的fullmesh兜底表项；无匹配回退rsConfig
        uint8_t commEngine =
            (commMode_ == TPL_CCU_COMM_MODE) ? mc2tiling::A5_CCU_ENGINE : mc2tiling::A5_AICPU_TS_ENGINE;
        std::string algoName =
            SelectReduceScatterAlgoName(opName_, group, commEngine, args_.orgMValue, args_.orgNValue,
                                        args_.outputDtypeSize, static_cast<uint32_t>(args_.rankDim), rsConfig);
        OP_LOGI(opName_, "[SetMc2Hcomm] selected algoName=%s, group=%s", algoName.c_str(), group);
        mc2CcTilingConfig.SetAlgConfig(algoName);
    }
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulReduceScatterV2TilingData_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(matmulReduceScatterV2TilingData_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg,
                                                              Mc2MMRegisterCfg& registerCfg,
                                                              Mc2MatMulV3TilingData& tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim);
    OP_LOGD(opName_, "execute DoMatmulV3Tiling!");
    tilingCfg.SetMatMulV3TilingData(tilingData);
    OP_TILING_CHECK(
        Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "do tiling failed"), return ge::GRAPH_FAILED);
    if (!usingApaceImpl_) {
        return ge::GRAPH_SUCCESS;
    }
    const auto& cube = tilingData.tCubeTiling;
    if (static_cast<uint64_t>(cube.M) != mmV3Args_.mValue || cube.M <= 0 || cube.N <= 0 || cube.Ka <= 0 ||
        cube.Kb != cube.Ka || static_cast<uint64_t>(cube.N) != args_.orgNValue ||
        static_cast<uint64_t>(cube.Ka) != args_.orgKValue || cube.baseM <= 0 || cube.baseN <= 0 || cube.baseK <= 0 ||
        cube.stepKa <= 0 || cube.stepKb <= 0 || cube.usedCoreNum <= 0 ||
        static_cast<uint64_t>(cube.usedCoreNum) > args_.aicCoreNum || cube.baseM % BASIC_BLOCK_SIZE_16 != 0 ||
        cube.baseN % BASIC_BLOCK_SIZE_16 != 0 || cube.baseK % BASIC_BLOCK_SIZE_16 != 0 || tilingData.mTailCnt == 0 ||
        tilingData.nTailCnt == 0) {
        OP_LOGE(opName_, "Matmul tiling does not fit APACE Basic scheduling");
        return ge::GRAPH_FAILED;
    }
    const uint64_t baseM = cube.baseM;
    const uint64_t baseN = cube.baseN;
    const uint64_t baseK = cube.baseK;
    const auto validTail = [](uint64_t length, uint64_t base, uint64_t splitCount, uint64_t tailMain) {
        const uint64_t blocks = (length + base - 1) / base;
        if (splitCount == 0 || splitCount > blocks) {
            return false;
        }
        const uint64_t tail = length - (blocks - splitCount) * base;
        return splitCount == 1 || (tailMain > 0 && tailMain <= base && tailMain * (splitCount - 1) < tail &&
                                   tail - tailMain * (splitCount - 1) <= base);
    };
    if (!validTail(cube.M, baseM, tilingData.mBaseTailSplitCnt, tilingData.mTailMain) ||
        !validTail(cube.N, baseN, tilingData.nBaseTailSplitCnt, tilingData.nTailMain)) {
        OP_LOGE(opName_, "Matmul tiling does not fit APACE Basic scheduling");
        return ge::GRAPH_FAILED;
    }
    // Basic 按实际启动核数分配尾轮，原来的尾块细分必须仍落在同一轮内。
    const uint64_t blocks =
        ((static_cast<uint64_t>(cube.M) + baseM - 1) / baseM) * ((static_cast<uint64_t>(cube.N) + baseN - 1) / baseN);
    const uint64_t tailBlocks = blocks % args_.aicCoreNum;
    const uint64_t tailSplit = static_cast<uint64_t>(tilingData.mTailCnt) * tilingData.nTailCnt;
    if (tailSplit > 1 && (tailBlocks == 0 || tailSplit > args_.aicCoreNum / tailBlocks)) {
        OP_LOGE(opName_, "Matmul tiling does not fit APACE Basic scheduling");
        return ge::GRAPH_FAILED;
    }

    // 与 Basic BlockMmad 的两份 L1、L0A/L0B 和 BiasTable 分配保持一致。
    const uint64_t kL1 = baseK * static_cast<uint64_t>(std::min(cube.stepKa, cube.stepKb));
    const uint64_t l1SlotBytes = compileInfo_.l1Size / DB_SIZE;
    const uint64_t biasBytes = args_.isBias ? baseN * ge::GetSizeByDataType(args_.geBiasType) : 0;
    const uint64_t l1RowBytes = (baseM + baseN) * args_.inputDtypeSize;
    const uint64_t l0cStages = cube.dbL0C > DB_OFF_SIZE ? DB_SIZE : DB_OFF_SIZE;
    OP_TILING_CHECK(l1RowBytes == 0 || biasBytes > l1SlotBytes || kL1 > (l1SlotBytes - biasBytes) / l1RowBytes ||
                        baseM * baseK * args_.inputDtypeSize > compileInfo_.l0ASize / DB_SIZE ||
                        baseN * baseK * args_.inputDtypeSize > compileInfo_.l0BSize / DB_SIZE ||
                        baseM * baseN * sizeof(float) > compileInfo_.l0CSize / l0cStages ||
                        (args_.isBias && baseN * sizeof(float) > compileInfo_.btSize / DB_SIZE),
                    OP_LOGE(opName_, "Matmul tiling exceeds APACE Basic buffer capacity"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::DoAllMatmulTiling()
{
    OP_LOGD(opName_, "execute DoAllMatmulTiling!");
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "get platform info failed"), return ge::GRAPH_FAILED);
    // 获取compileInfo
    OP_TILING_CHECK(mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "init compile info failed"), return ge::GRAPH_FAILED);

    // 根据芯片型号获取策略模板
    std::vector<int32_t> priorities;
    OP_TILING_CHECK(mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch_, priorities, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "get mmv3 priority policy failed"), return ge::GRAPH_FAILED);
    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch_, priorities};
    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 获取tileTiling
    mmV3Args_.mValue = tileMValue_ * args_.rankDim;
    OP_LOGD(opName_, "Do Mc2MatMulV3 tile tiling!");
    // APACE 调度完整逻辑 M，跨 rank 的块由 Kernel 分段；不再套用单 rank 的尾轮修正。
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(static_cast<const void*>(&compileInfo_),
                                                      static_cast<const void*>(&mmV3Args_),
                                                      usingApaceImpl_ ? 0UL : tileMValue_);
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MmV3TileTilingData()));
    if (tailMValue_ != 0UL) {
        mmV3Args_.mValue = tailMValue_ * args_.rankDim;
        OP_LOGD(opName_, "Do Mc2MatMulV3 tail tiling!");
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(static_cast<const void*>(&compileInfo_),
                                                          static_cast<const void*>(&mmV3Args_),
                                                          usingApaceImpl_ ? 0UL : tailMValue_);
        MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tailTilingCfg, registerCfg, MutableMC2MmV3TailTilingData()));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::DoOpTiling()
{
    usingApaceImpl_ = IsApaceSupported();
    if (usingApaceImpl_) {
        isA2APath_ = false;
    }
    MC2_CHECK_LOG_RET(opName_, CheckHCCLSize());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    OP_TILING_CHECK(SetMc2Hcomm() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Tiling SetHcommCfg failed."),
                    return ge::GRAPH_FAILED);
    SetRcsTilingData(matmulReduceScatterV2TilingData_->param);
    MC2_CHECK_LOG_RET(opName_, DoSplitMTiling(matmulReduceScatterV2TilingData_->param));
    MC2_CHECK_LOG_RET(opName_,
                      AdjustHCCLLimit(matmulReduceScatterV2TilingData_->param, mc2tiling::Mc2QuantMode::DEFAULT));
    MC2_CHECK_LOG_RET(opName_, DoAllMatmulTiling());
    SetTilingResult(matmulReduceScatterV2TilingData_->param, MutableMC2MmV3TileTilingData().tCubeTiling,
                    MutableMC2MmV3TailTilingData().tCubeTiling, matmulReduceScatterV2TilingData_->debugMode,
                    matmulReduceScatterV2TilingData_->dataType);
    if (usingApaceImpl_) {
        const auto& cfg = matmulReduceScatterV2TilingData_->param;
        const uint64_t tailM = cfg.tailCnt > 0 ? tailMValue_ : 0;
        OP_TILING_CHECK(cfg.rankDim == 0 || cfg.tileCnt == 0 || cfg.rankM % cfg.rankDim != 0,
                        OP_LOGE(opName_, "invalid APACE nonquant split"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(static_cast<uint64_t>(cfg.tileCnt) + cfg.tailCnt > REDUCE_SCATTER_V2_HCCL_NUM_LIMIT,
                        OP_LOGE(opName_, "APACE communication count %u + %u exceeds %lu", cfg.tileCnt, cfg.tailCnt,
                                REDUCE_SCATTER_V2_HCCL_NUM_LIMIT),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            tileMValue_ == 0 || tileMValue_ * cfg.tileCnt + tailM * cfg.tailCnt != cfg.rankM / cfg.rankDim ||
                (cfg.tailCnt > 0 && (tailM == 0 || cfg.tailM != tailM)),
            OP_LOGE(opName_, "APACE nonquant slices do not cover all output rows"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(std::max(tileMValue_, tailM) * args_.orgNValue * args_.outputDtypeSize >
                            mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT,
                        OP_LOGE(opName_, "APACE communication slice exceeds 256 MiB"), return ge::GRAPH_FAILED);
    }
    OP_LOGI(opName_, "FP16/BF16 APACE selected: %d, commMode=%u, tileCnt=%u, tailCnt=%u", usingApaceImpl_, commMode_,
            matmulReduceScatterV2TilingData_->param.tileCnt, matmulReduceScatterV2TilingData_->param.tailCnt);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulReduceScatterV2Tiling::PostTiling()
{
#if MC2_DFX_ENABLE
    matmulReduceScatterV2TilingData_->dumpInfo.workspaceLayout = workspaceLayout_;
#endif
    auto rawTilingDataPtr = context_->GetRawTilingData();
    OP_TILING_CHECK((rawTilingDataPtr == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "rawTilingDataPtr"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ",
            sizeof(MatmulReduceScatterV2TilingData), rawTilingDataPtr->GetCapacity());
    rawTilingDataPtr->SetDataSize(sizeof(MatmulReduceScatterV2TilingData));

    OP_TILING_CHECK(sizeof(MatmulReduceScatterV2TilingData) % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", sizeof(MatmulReduceScatterV2TilingData)),
                    return ge::GRAPH_FAILED);
    PrintAllTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

CutResult MatmulReduceScatterV2Tiling::GetTilingResult()
{
    if (!usingApaceImpl_ && mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_)) {
        MMReduceScatterFitBalanceTiling scatterTiling(args_, KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL,
                                                      TopoType::STANDARD_CARD, (commMode_ == TPL_AICPU_COMM_MODE));
        return scatterTiling.GetTiling();
    } else if (!usingApaceImpl_ && mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        MMReduceScatterFitBalanceTiling scatterTiling(args_, KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL,
                                                      TopoType::EIGHT_P, (commMode_ == TPL_AICPU_COMM_MODE));
        return scatterTiling.GetTiling();
    } else {
        MMPlusReduceScatter scatterTiling(args_, args_.rankDim, KernelType::REDUCE_SCATTER, npuArch_, false,
                                          (commMode_ == TPL_AICPU_COMM_MODE));
        scatterTiling.GetTiling();
        CutResult cutRes = scatterTiling.tilingM_.cutRes;
        if (usingApaceImpl_) {
            AlignCutResForApace(cutRes, args_.mValue);
        }
        return cutRes;
    }
}

// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulReduceScatterV2, MatmulReduceScatterV2Tiling,
                                   std::vector<int32_t>({static_cast<int32_t>(Ops::Base::DAV_3510),
                                                         static_cast<int32_t>(Ops::Base::DAV_9201)}),
                                   0);

} // namespace optiling
