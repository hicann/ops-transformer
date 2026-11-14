/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_matmul_tiling_v2.cpp
 * \brief
 */
#include <algorithm>
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "all_gather_matmul_tiling_v2.h"
#include "../../../op_kernel/all_gather_matmul_v2_apt_tiling_key.h"
#include "all_gather_comm_algo_table.h"
#include "all_gather_fit_balance_tiling.h"
#include "all_gather_hccl_utils.h"
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_comm_utils.h"
#include "mc2_log.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "register/op_def_registry.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace optiling {
static constexpr int64_t CMP_MAX_LEN = 7;
static constexpr uint64_t APACE_KERNEL_ALIGN_M = 256;
// fragment(L1-A nz / L0C nz 分形)行对齐粒度：tail 路径 paddedTailM 按 16 对齐（同 V3 M_TAIL_ALIGN，
// 保证 L0C 拷出首地址 1024B 对齐）；head tileM 非 16 对齐时 256 行调度块会跨越 fragment 边界，
// 产生分形中间起始的 L1/L0C 拷贝（MTE2/fixpipe 非法地址 → aicore error），须改走纯尾块路径
static constexpr uint64_t APACE_KERNEL_FRACTAL_ALIGN_M = 16;
static constexpr uint32_t APACE_STEPK_THRESHOLD = 4;
static constexpr uint8_t APACE_L1_BUFFER_NUM = 2;

bool AllGatherMatmulTilingV2::IsCapable()
{
    if ((npuArch_ == Ops::Base::DAV_3510 || npuArch_ == Ops::Base::DAV_9201) && inputIsBf16Fp16_) {
        OP_LOGI(opName_, "Start with AllGatherMatmulTilingV2 tiling.");
        return true;
    }

    OP_LOGI(opName_, "Skip AllGatherMatmulTilingV2 tiling when input datatype is not fp16 or bf16.");
    return false;
}

ge::graphStatus AllGatherMatmulTilingV2::SetRawTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK((rawTilingData == nullptr), OP_LOGE(opName_, "Fail to get rawTilingData."),
                    return ge::GRAPH_FAILED);
    allGatherMatmulTilingDataV2_ = context_->GetTilingData<AllGatherMatmulTilingDataV2>();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::FitApaceMmTileToL1(uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                                            uint32_t& stepK) const
{
    uint64_t l1Size = 0UL;
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    }
    if (l1Size == 0UL) {
        l1Size = 512UL * 1024UL; // 兜底与 kernel 侧 TOTAL_L1_SIZE(arch35) 一致
    }
    // 半区预算与 mmTile.nBufferNum 同源（APACE_L1_BUFFER_NUM），避免缓冲数调整时公式静默失准
    const uint64_t halfL1 = l1Size / APACE_L1_BUFFER_NUM;
    const uint64_t c0Elems = 32UL / args_.inputDtypeSize;                     // fp16/bf16: 16 元素/分形
    const uint64_t biasElemSize = args_.isBias ? args_.outputDtypeSize : 0UL; // BiasType = CType

    auto l1Usage = [this, baseK, c0Elems, biasElemSize](uint32_t curBaseM, uint32_t curBaseN, uint32_t curStepK) {
        uint64_t kL1 = mc2tiling::AlignUp(static_cast<uint64_t>(curStepK) * baseK, c0Elems);
        uint64_t aL1 = static_cast<uint64_t>(curBaseM) * kL1 * args_.inputDtypeSize;
        uint64_t bL1 = static_cast<uint64_t>(curBaseN) * kL1 * args_.inputDtypeSize;
        uint64_t biasL1 = static_cast<uint64_t>(curBaseN) * biasElemSize;
        return aL1 + bL1 + biasL1;
    };

    const uint32_t oriStepK = stepK;
    while ((stepK > 1UL) && (l1Usage(baseM, baseN, stepK) > halfL1)) {
        stepK--;
    }
    // stepK=1 时 A+B+bias ≤ L0A+L0B+bias < L1/2（L0 tile 硬约束 baseM·baseK·dtype ≤ L0A、
    // baseN·baseK·dtype ≤ L0B，v3 档位对自身 kernel 合法即满足），仍超限仅当借用档位违反
    // L0 约束（不可达），报错拒绝（同 3rd gmm_qbmm_tiling "L1 space overflow" 处置）
    OP_TILING_CHECK(
        (l1Usage(baseM, baseN, stepK) > halfL1),
        OP_LOGE(opName_,
                "[APACE] L1 capacity overflow: baseM=%u, baseN=%u, baseK=%u, stepK=%u, usage=%lu, "
                "halfL1=%lu, isBias=%d.",
                baseM, baseN, baseK, stepK, l1Usage(baseM, baseN, stepK), halfL1, static_cast<int32_t>(args_.isBias)),
        return ge::GRAPH_FAILED);
    if (stepK != oriStepK) {
        OP_LOGW(opName_, "[APACE] L1 capacity fit: stepK %u->%u (baseM=%u, baseN=%u, baseK=%u, isBias=%d, halfL1=%lu).",
                oriStepK, stepK, baseM, baseN, baseK, static_cast<int32_t>(args_.isBias), halfL1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::FillApaceMmTile(const Mc2MatMulV3TilingData& localTiling)
{
    auto& mmTile = hcommTilingDataSelf_.mmTile;
    const auto& tCube = localTiling.tCubeTiling;
    mmTile = {};
    mmTile.m = static_cast<uint32_t>(args_.mValue);
    mmTile.n = static_cast<uint32_t>(args_.nValue);
    mmTile.k = static_cast<uint32_t>(args_.kValue);
    mmTile.baseM = tCube.baseM;
    mmTile.baseN = tCube.baseN;
    mmTile.baseK = tCube.baseK;
    mmTile.scaleKL1 = 0; // 非量化使用该字段
    mmTile.mTailTile = localTiling.mTailCnt;
    mmTile.nTailTile = localTiling.nTailCnt;
    mmTile.mBaseTailSplitCnt = localTiling.mBaseTailSplitCnt;
    mmTile.nBaseTailSplitCnt = localTiling.nBaseTailSplitCnt;
    mmTile.mTailMain = localTiling.mTailMain;
    mmTile.nTailMain = localTiling.nTailMain;
    mmTile.usedCoreNum = tCube.usedCoreNum;
    uint32_t stepK = std::min(tCube.stepKa, tCube.stepKb);
    stepK = std::min(stepK, APACE_STEPK_THRESHOLD);
    // apace L1 半区(A|B|bias)容量复核：零余量档位叠加 bias 时 bias 槽会越界/重叠，超限仅降 stepK
    OP_TILING_CHECK((FitApaceMmTileToL1(mmTile.baseM, mmTile.baseN, mmTile.baseK, stepK) != ge::GRAPH_SUCCESS),
                    OP_LOGE(opName_, "Fail to fit apace mm tile to L1."), return ge::GRAPH_FAILED);
    mmTile.stepK = static_cast<uint8_t>(stepK);
    mmTile.nBufferNum = APACE_L1_BUFFER_NUM;
    mmTile.dbL0c = static_cast<uint8_t>(tCube.dbL0C);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoApaceTiling()
{
    auto& rcsCfg = MutableRCSTilingData();
    uint64_t apaceTileM = tileMValue_;
    uint64_t apaceTileCnt = rcsCfg.tileCnt;
    uint64_t apaceTailM = rcsCfg.tailM;
    uint64_t apaceTailCnt = rcsCfg.tailCnt;
    if ((apaceTileM != 0) && (args_.mValue != 0)) {
        uint64_t alignedTileM = mc2tiling::AlignUp(apaceTileM, APACE_KERNEL_ALIGN_M);
        if (alignedTileM >= args_.mValue) {
            if (args_.mValue % APACE_KERNEL_FRACTAL_ALIGN_M == 0) {
                apaceTileM = args_.mValue;
                apaceTileCnt = 1;
                apaceTailM = 0;
                apaceTailCnt = 0;
            } else {
                // 整 M 单 head tile 且 M 非分形(16 行)对齐：改走纯尾块路径（tileCnt=0，kernel 跳过 head 提交）。
                // tail fragment 按 paddedTailM=AlignUp(M,16) 对齐分布（realFragmentSize=M 抑制补齐行写 C），
                // 使 256 行调度块跨界切片的 L1-A/L0C 侧偏移恒为 16 对齐，避免 aicore L1 error（rank>2 卡死根因）
                apaceTileM = 0;
                apaceTileCnt = 0;
                apaceTailM = args_.mValue;
                apaceTailCnt = 1;
            }
        } else {
            apaceTileM = alignedTileM;
            apaceTileCnt = args_.mValue / alignedTileM;
            uint64_t remainder = args_.mValue % alignedTileM;
            apaceTailM = (remainder > 0) ? remainder : 0;
            apaceTailCnt = (remainder > 0) ? 1 : 0;
        }
    }
    auto& commTile = hcommTilingDataSelf_.commTile;
    commTile.splitAxisTileSize = apaceTileM;
    commTile.splitAxisTileCnt = apaceTileCnt;
    commTile.splitAxisTailSize = apaceTailM;
    commTile.splitAxisTailCnt = apaceTailCnt;
    commTile.nonSplitAxisSize = args_.kValue;

    OP_TILING_CHECK((FillApaceMmTile(MutableMC2MatmulV3LocalTilingData()) != ge::GRAPH_SUCCESS),
                    OP_LOGE(opName_, "Fail to fill apace mm tile."), return ge::GRAPH_FAILED);

    hcommTilingDataSelf_.mc2InitTiling = allGatherMatmulTilingDataV2_->mc2InitTiling;
    hcommTilingDataSelf_.mc2CcTiling = allGatherMatmulTilingDataV2_->mc2CcTiling;

    hcommTilingDataSelf_.gatherLen = rcsCfg.gatherLen;

    OP_LOGI(opName_,
            "[APACE] commTile: tileM=%lu, tileCnt=%lu, tailM=%lu, tailCnt=%lu, k=%lu; mmTile: baseM=%u, baseN=%u, "
            "baseK=%u, stepK=%u, nBufferNum=%u, dbL0c=%u, mTailTile=%u, nTailTile=%u; gatherLen=%lu.",
            commTile.splitAxisTileSize, commTile.splitAxisTileCnt, commTile.splitAxisTailSize,
            commTile.splitAxisTailCnt, commTile.nonSplitAxisSize, hcommTilingDataSelf_.mmTile.baseM,
            hcommTilingDataSelf_.mmTile.baseN, hcommTilingDataSelf_.mmTile.baseK, hcommTilingDataSelf_.mmTile.stepK,
            hcommTilingDataSelf_.mmTile.nBufferNum, hcommTilingDataSelf_.mmTile.dbL0c,
            hcommTilingDataSelf_.mmTile.mTailTile, hcommTilingDataSelf_.mmTile.nTailTile,
            hcommTilingDataSelf_.gatherLen);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckHCCLSize());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    MC2_CHECK_LOG_RET(opName_, SetRawTilingData());
    SetRcsTilingData(MutableRCSTilingData());
    DoSplitMTiling(MutableRCSTilingData());
    // 通信配置放在切分之后：单轮通信量依赖切分结果 tileMValue_（对齐 alltoall 先切分后配置的时序）
    OP_TILING_CHECK(SetMc2Hcomm(MutableRCSTilingData()) != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Fail to set Mc2Hcomm."),
                    return ge::GRAPH_FAILED);
    MC2_CHECK_LOG_RET(opName_, DoVersion2Tiling());
    DoAllGatherTiling(MutableRCSTilingData(), MutableMC2MatmulV3TileTilingData().tCubeTiling,
                      MutableMC2MatmulV3TailTilingData().tCubeTiling, allGatherMatmulTilingDataV2_->debugMode,
                      allGatherMatmulTilingDataV2_->dataType);
    OP_TILING_CHECK(DoApaceTiling() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "Fail to do apace tiling."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void Mc2PrintMMV3TilingData(const std::string& opName, Mc2MatMulV3TilingData& tiling)
{
    PrintTCubeTilingData(opName, tiling.tCubeTiling);
    OP_LOGD(opName, " tiling.mTailCnt %d", tiling.mTailCnt);
    OP_LOGD(opName, " tiling.nTailCnt %d", tiling.nTailCnt);
    OP_LOGD(opName, " tiling.kTailCnt %d", tiling.kTailCnt);
    OP_LOGD(opName, " tiling.isHf32 %d", tiling.isHf32);
    OP_LOGD(opName, " tiling.mBaseTailSplitCnt %d", tiling.mBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.nBaseTailSplitCnt %d", tiling.nBaseTailSplitCnt);
    OP_LOGD(opName, " tiling.mTailMain %d", tiling.mTailMain);
    OP_LOGD(opName, " tiling.nTailMain %d", tiling.nTailMain);
    OP_LOGD(opName, " tiling.aswWindowLen %d", tiling.aswWindowLen);
}

void AllGatherMatmulTilingV2::PrintAllTilingData()
{
    if (MutableRCSTilingData().rankID != 0) {
        return;
    }
    PrintRCSTilingData(context_->GetNodeName(), MutableRCSTilingData());
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3LocalTilingData());
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3TileTilingData());
    if (MutableRCSTilingData().tailM <= 0) {
        return;
    }
    OP_LOGD(opName_, "Matmul has tail.");
    Mc2PrintMMV3TilingData(context_->GetNodeName(), MutableMC2MatmulV3TailTilingData());
}

ge::graphStatus AllGatherMatmulTilingV2::PostTiling()
{
    auto* outTilingData = context_->GetTilingData<Apace::hcommAllGatherMatmulTilingData>();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Fail to get hcomm tiling data from context"),
                    return ge::GRAPH_FAILED);
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((tilingBufCap < sizeof(hcommTilingDataSelf_)),
                    OP_LOGE(opName_, "TilingBuffer capacity is too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(hcommTilingDataSelf_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &hcommTilingDataSelf_, sizeof(hcommTilingDataSelf_));
    if (ret != EOK) {
        OP_LOGE(opName_, "PostTiling: memcpy_s hcomm tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    const size_t registeredMaxTilingSize = sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2);
    const size_t apaceTilingSize = sizeof(Apace::hcommAllGatherMatmulTilingData);
    const size_t finalSize = (registeredMaxTilingSize > apaceTilingSize) ? registeredMaxTilingSize : apaceTilingSize;
    OP_TILING_CHECK((finalSize % sizeof(uint64_t) != 0),
                    OP_LOGE(opName_, "Tiling data size[%zu] not aligned to 8.", finalSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (tilingBufCap < finalSize),
        OP_LOGE(opName_, "TilingBuffer capacity %zu is smaller than final tiling size %zu.", tilingBufCap, finalSize),
        return ge::GRAPH_FAILED);
    if ((tilingBufCap >= finalSize) && (finalSize > apaceTilingSize)) {
        auto* tilingBuf = static_cast<uint8_t*>(context_->GetRawTilingData()->GetData());
        errno_t zeroRet =
            memset_s(tilingBuf + apaceTilingSize, tilingBufCap - apaceTilingSize, 0, finalSize - apaceTilingSize);
        if (zeroRet != EOK) {
            OP_LOGE(opName_, "PostTiling: memset_s hcomm tiling padding failed, ret=%d.", zeroRet);
            return ge::GRAPH_FAILED;
        }
    }
    context_->GetRawTilingData()->SetDataSize(finalSize);
    OP_LOGI(opName_, "Final apace hcomm tiling data size=%zu (v2=%zu, apace=%zu), context capacity size=%zu.",
            finalSize, registeredMaxTilingSize, apaceTilingSize, tilingBufCap);
    context_->SetBlockDim(args_.aicCoreNum);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

CutResult AllGatherMatmulTilingV2::GetTilingResult()
{
    AllGatherMMFitBalanceTiling tileFormulate(args_, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    CutResult result = tileFormulate.GetTiling();

    bool isCcuMode = (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0);
    if (isCcuMode) {
        CommSizeInfo commSize{args_.kValue, args_.inputDtypeSize, args_.rankDim};
        AdjustCutResultForCCU(result, args_.mValue, commSize, opName_);
    } else {
        AdjustCutResultForAICPU(result, args_.mValue, tileFormulate.matmulPerf_.GetBaseM(), opName_);
    }

    return result;
}

ge::graphStatus AllGatherMatmulTilingV2::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg& tilingCfg,
                                                          Mc2MMRegisterCfg& registerCfg,
                                                          Mc2MatMulV3TilingData& tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim - 1);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (args_.nValue != 0)
        if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName_, "DoMatmulV3Tiling failed.");
            return ge::GRAPH_FAILED;
        }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::DoVersion2Tiling()
{
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Fail to get platformInfo."), return ge::GRAPH_FAILED);

    // 获取 compileInfo
    if (mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fail to Init Compile Info!");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatForm = platform_ascendc::PlatformAscendC(platformInfo);

    // 根据芯片型号获取策略模板
    platform_ascendc::SocVersion socVersion = ascendcPlatForm.GetSocVersion();
    NpuArch npuArch = ascendcPlatForm.GetCurNpuArch();

    std::vector<int32_t> priorities;
    MC2_CHECK_LOG_RET(opName_, mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch, priorities, opName_));

    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch, priorities};

    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 计算 local 块 tiling
    Mc2MatmulHelper::Mc2MatmulTilingCfg localTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                       reinterpret_cast<const void*>(&mmV3Args_));
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(localTilingCfg, registerCfg, MutableMC2MatmulV3LocalTilingData()));

    // 计算 tile 块 tiling
    mmV3Args_.mValue = tileMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tileCnt);
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                      reinterpret_cast<const void*>(&mmV3Args_), tileMValue_);
    tileTilingCfg.SetCommCnt(MutableRCSTilingData().tileCnt);
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MatmulV3TileTilingData()));
    MutableMC2MatmulV3TileTilingData().tCubeTiling.M = (tileMValue_ * (args_.rankDim - 1));

    if (tailMValue_ > 0) {
        // 计算 tail 块 tiling
        mmV3Args_.mValue = tailMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tailCnt);
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void*>(&compileInfo_),
                                                          reinterpret_cast<const void*>(&mmV3Args_), tailMValue_);
        tailTilingCfg.SetCommCnt(MutableRCSTilingData().tailCnt);
        MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tailTilingCfg, registerCfg, MutableMC2MatmulV3TailTilingData()));
        MutableMC2MatmulV3TailTilingData().tCubeTiling.M = (tailMValue_ * (args_.rankDim - 1));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::SetMc2Hcomm(Mc2Tiling::RCSTiling& rcsCfg)
{
    // Set hccl comm engine with comm_mode
    uint8_t commEngine = Mc2Comm::ENGINE_AICPU;
    if (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0) {
        commEngine = Mc2Comm::ENGINE_CCU_SCHED;
    }
    OP_LOGD(opName_, "Tiling SetMc2Hcom commMode_: %s", commMode_);
    OP_LOGD(opName_, "Tiling SetMc2Hcom commEngine: %d", commEngine);
    // 单轮通信量 = 每轮切分的 tileM × K × dtypeSize（对齐 alltoall 口径），算法表为空时回退默认算法
    std::string algConfig =
        Mc2Tiling::SelectAllGatherAlgoNameByArch(opName_, group_, commEngine, tileMValue_, args_.kValue,
                                                 args_.inputDtypeSize, args_.rankDim, static_cast<uint32_t>(npuArch_));
    OP_LOGI(opName_, "[SetMc2Hcomm] selected algConfig=%s, group=%s", algConfig.c_str(), group_);
    Mc2CcTilingConfig mc2CcTilingConfig(
        group_, static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER), algConfig, 0,
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)),
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
    mc2CcTilingConfig.SetCommEngine(commEngine);
    uint8_t skipBufferWindowCopy = (allGatherMatmulTilingDataV2_->param.gatherLen == 0) ?
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
    mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataV2_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingDataV2_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::CheckInput()
{
    auto x1ScaleShape = context_->GetOptionalInputShape(SCALE_INV1);
    OP_TILING_CHECK(
        x1ScaleShape != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1Scale", "not nullptr",
                                              "If the dtype of x1 and x2 is fp16 or bf16, x1Scale must be nullptr"),
        return ge::GRAPH_FAILED);
    auto x2ScaleShape = context_->GetOptionalInputShape(SCALE_INV2);
    OP_TILING_CHECK(
        x2ScaleShape != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x2Scale", "not nullptr",
                                              "If the dtype of x1 and x2 is fp16 or bf16, x2Scale must be nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTilingV2::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed."), return ge::GRAPH_FAILED);
    workspaceSize_ = libApiWorkSpaceSize_ + storageA_;
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "[APACE] workspaces[0] size=%zu, libApi=%u, gatherLen(storageA)=%u.", workspaces[0],
            libApiWorkSpaceSize_, storageA_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AllGatherMatmulTilingV2::GetTilingKey() const
{
    uint8_t outputType = (outputIsFp8_) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);

    uint8_t commMode = Mc2Comm::COMM_MODE_AICPU;
    if (std::strncmp(commMode_, "ccu", CMP_MAX_LEN) == 0) {
        commMode = Mc2Comm::COMM_MODE_CCU;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputIsBf16Fp16_, args_.isBTrans, outputType, TPL_DEFAULT_MODE,
                                                  SCALE_TYPE_NOT_IS_MX, commMode);
    OP_LOGD(opName_, "AllGatherMatmulV2(apace), inputIsBf16Fp16_, args_.isBTrans, outputType, commMode: [%d,%d,%u,%u]",
            inputIsBf16Fp16_, args_.isBTrans, outputType, commMode);
    OP_LOGD(opName_, "tilingKey=%lu", tilingKey);
    return tilingKey;
}

AllGatherMatmulTilingV2::AllGatherMatmulTilingV2(gert::TilingContext* context)
    : AllGatherMatmulTilingBase(context),
      allGatherMatmulTilingDataV2_(&allGatherMatmulTilingDataV2Self_)
{}
// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AllGatherMatmulV2, AllGatherMatmulTilingV2,
                                   std::vector<int32_t>({static_cast<int32_t>(Ops::Base::DAV_3510),
                                                         static_cast<int32_t>(Ops::Base::DAV_9201)}),
                                   0);
} // namespace optiling
