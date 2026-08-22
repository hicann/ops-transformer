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
 * \file all_gather_matmul_tiling_base.cpp
 * \brief
 */
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>

#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "util/math_util.h"
#include "all_gather_fit_balance_tiling.h"
#include "all_gather_matmul_tiling_base.h"
#include "../../op_kernel/all_gather_matmul_v2_apt_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
const std::set<int> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64};
constexpr uint64_t BLOCK_SIZE_INDEX = 6;

void AllGatherMatmulTilingBase::SetTilingArgsDim()
{
    const gert::StorageShape *x1Shape = context_->GetInputShape(INPUT_X1);
    const gert::StorageShape *x2Shape = context_->GetInputShape(INPUT_X2);
    uint64_t x1Dim0 = x1Shape->GetStorageShape().GetDim(0);
    uint64_t x1Dim1 = x1Shape->GetStorageShape().GetDim(1);
    uint64_t x2Dim0 = x2Shape->GetStorageShape().GetDim(0);
    uint64_t x2Dim1 = x2Shape->GetStorageShape().GetDim(1);

    args_.orgMValue = x1Dim0;
    args_.orgNValue = (x1Dim1 == x2Dim0) ? x2Dim1 : x2Dim0;
    args_.orgKValue = x1Dim1;
    args_.mValue = x1Dim0;
    args_.nValue = (x1Dim1 == x2Dim0) ? x2Dim1 : x2Dim0;
    args_.kValue = x1Dim1;
    return;
}

void AllGatherMatmulTilingBase::SetTilingArgsDataType()
{
    const gert::StorageShape *matrixBias = context_->GetOptionalInputShape(BIAS);
    ge::DataType aType = context_->GetInputDesc(INPUT_X1)->GetDataType();
    ge::DataType bType = context_->GetInputDesc(INPUT_X2)->GetDataType();
    ge::DataType biasType;
    bool isBias = true;
    auto cType = aType;

    if (matrixBias == nullptr) {
        isBias = false;
        biasType = cType;
    } else {
        biasType = context_->GetOptionalInputDesc(BIAS)->GetDataType();
    }

    args_.inputDtypeSize = mc2tiling::GetDataTypeSize(opName_, aType);
    args_.outputDtypeSize = mc2tiling::GetDataTypeSize(opName_, cType);

    args_.isBias = isBias;
    args_.geAType = aType;
    args_.geBType = bType;
    args_.geCType = cType;
    args_.geBiasType = biasType;
    args_.aType = mc2tiling::ConvertGeTypeToMmType(opName_, aType);
    args_.bType = mc2tiling::ConvertGeTypeToMmType(opName_, bType);
    args_.cType = mc2tiling::ConvertGeTypeToMmType(opName_, cType);
    args_.biasType = mc2tiling::ConvertGeTypeToMmType(opName_, biasType); // 因为bias可能不存在，先采用biasType规避
    inputIsBf16Fp16_ = ((aType == ge::DT_BF16) || (aType == ge::DT_FLOAT16)) ? true : false;
    return;
}

void AllGatherMatmulTilingBase::SetTilingArgsGatherStatus()
{
    auto gatherOutShape = context_->GetOutputShape(GATHER_OUT);
    args_.isStorageGather = true;
    if (gatherOutShape != nullptr) {
        int64_t mulGatherShape = 1;
        for (uint32_t i = 0; i < gatherOutShape->GetStorageShape().GetDimNum(); i++) {
            mulGatherShape = mulGatherShape * gatherOutShape->GetStorageShape().GetDim(i);
            OP_LOGD("AllGatherMatmul", "gatherOutShape StorageShape=%ld, Dim=%u.",
                    gatherOutShape->GetStorageShape().GetDim(i), i);
        }
        if (mulGatherShape == 0) {
            args_.isStorageGather = false;
        }
    } else {
        args_.isStorageGather = false;
    }
    return;
}

bool AllGatherMatmulTilingBase::AnalyzeInputs()
{
    args_.enablePad = false;
    args_.enableSplitK = false;
    SetTilingArgsDim();
    SetTilingArgsDataType();
    SetTilingArgsGatherStatus();
    return true;
}

ge::graphStatus AllGatherMatmulTilingBase::AnalyzeShapeAttr()
{
    opName_ = context_->GetNodeName();
    OP_TILING_CHECK(((!AnalyzeAttrs()) || (!AnalyzeInputs()) || (!SetCommAlgo())),
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to analyze context info"), return ge::GRAPH_FAILED);
    commAlgorithm_ = static_cast<uint64_t>(args_.commAlg);
    return ge::GRAPH_SUCCESS;
}

void AllGatherMatmulTilingBase::SetMC2AllGatherDataInfo(Mc2Tiling::RCSTiling &rcsCfg, ::TCubeTiling &mmTiling,
                                                        ::TCubeTiling &tailTiling)
{
    // 只通信不计算模式下，如果没有gatherOut且K > N, recvOff和sendCnt需要根据N计算
    auto columnNum = args_.orgKValue;
    OP_LOGD(opName_, "Gather out flag is %d, K is %lu, N is %lu.", (rcsCfg.gatherLen == 0), args_.orgKValue,
            args_.orgNValue);
}

ge::graphStatus AllGatherMatmulTilingBase::AdjustHCCLLimit(Mc2Tiling::RCSTiling &rcfCfg,
                                                           mc2tiling::Mc2QuantMode quantMmMode)
{
    if (tileMValue_ * args_.kValue * sizeof(args_.geAType) * args_.rankDim <= mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_LOG_I(opName_,
              "The result of formulaic tiling result does not meet the hccl restriction,"
              " current splitting: tileM [%ld], tileCnt [%ld], tailM [%ld], tailCnt [%ld]. start re-splitM.",
              tileMValue_, rcfCfg.tileCnt, tailMValue_, rcfCfg.tailCnt);

    OP_TILING_CHECK(
        (quantMmMode == mc2tiling::Mc2QuantMode::PERBLOCK_MODE),
        OP_LOGE(opName_, "Unsupported x1 size. Even after formulaic splitting, the size still exceeds 256MB."),
        return ge::GRAPH_FAILED);

    uint64_t minSplitPart = Ops::Base::CeilDiv(args_.mValue * args_.kValue * sizeof(args_.geAType) * args_.rankDim,
                                               mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT);
    tileMValue_ = Ops::Base::CeilDiv(args_.mValue, minSplitPart);
    rcfCfg.tileCnt = Ops::Base::FloorDiv(args_.mValue, tileMValue_);
    rcfCfg.tailM = args_.mValue - rcfCfg.tileCnt * tileMValue_;
    tailMValue_ = rcfCfg.tailM;
    if (tailMValue_ == 0) {
        rcfCfg.tailCnt = 0;
    } else {
        rcfCfg.tailCnt = 1;
    }
    OPS_LOG_I(opName_,
              "Because the result of formulaic tiling result does not meet the hccl restriction,"
              " the re-splitM result: tileM [%ld], tileCnt [%ld], tailM [%ld], tailCnt [%ld]. end re-splitM.",
              tileMValue_, rcfCfg.tileCnt, tailMValue_, rcfCfg.tailCnt);
    return ge::GRAPH_SUCCESS;
}

// tiling

void AllGatherMatmulTilingBase::DoAllGatherTiling(Mc2Tiling::RCSTiling &rcsCfg, ::TCubeTiling &mmTiling,
                                                  ::TCubeTiling &tailTiling, uint32_t &dataType)
{
    SetMC2AllGatherDataInfo(rcsCfg, mmTiling, tailTiling);

    dataType = (static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType))); // hccl数据类型

    // 计算一下额外申请的内存
    storageA_ = GetStorageA(rcsCfg);
}

void AllGatherMatmulTilingBase::SetRcsTilingData(Mc2Tiling::RCSTiling &rcsCfg)
{
    rcsCfg.rankDim = args_.rankDim;
    rcsCfg.isTransposeA = args_.isATrans;
    rcsCfg.isTransposeB = args_.isBTrans;
    rcsCfg.commtype = (static_cast<uint32_t>(args_.cmdType));
    OP_LOGD(opName_, "AllGatherMatmul SetRcsTilingData, args_.orgMValue=%lu, args_.orgNValue=%lu, args_.orgKValue=%lu.",
            args_.orgMValue, args_.orgNValue, args_.orgKValue);
    rcsCfg.rankM = args_.orgMValue;
    rcsCfg.rankN = args_.orgNValue;
    rcsCfg.rankK = args_.orgKValue;
    rcsCfg.aicCoreNum = args_.aicCoreNum;
    rcsCfg.storageGather = 0;
    if (args_.isStorageGather) {
        rcsCfg.storageGather = 1;
    }

    if (args_.isBias && (args_.bType == matmul_tiling::DataType::DT_BFLOAT16)) {
        biasLen_ = mc2tiling::AlignUp(args_.orgNValue, mc2tiling::SHAPE_ALIGN_SIZE) * sizeof(float);
    }
    rcsCfg.biasLen = biasLen_;
}

bool AllGatherMatmulTilingBase::SetCommAlgo()
{
    args_.commAlg = mc2tiling::Mc2GetCommAlgo(rankSize_, args_.orgMValue, group_, context_);
    if (args_.commAlg != mc2tiling::COMM_ALG_FULL_MESH) {
        OP_LOGE(opName_, "CommAlgo %u is not supported.", args_.commAlg);
        return false;
    }
    return true;
}

uint32_t AllGatherMatmulTilingBase::AllGatherSplitM(mc2tiling::TilingArgs &args, uint32_t maxTileCnt = 64)
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;
    }

    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue / args.commTurn;
    }

    if (args.inputDtypeSize == 2) {                          // 数据长度为2, 则向 2*64 = 128，则向128对齐
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64); // align size
    } else if (args.inputDtypeSize == 4) {                   // 4 is float32 type size
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32); // align size
    }
    if (args.mValue > tileLen) {
        return tileLen;
    }
    return args.mValue;
}

CutResult AllGatherMatmulTilingBase::GetTilingResult()
{
    AllGatherMMFitBalanceTiling tileFormulate(args_, KernelType::ALL_GATHER, TopoType::STANDARD_CARD);
    return tileFormulate.GetTiling();
}

void AllGatherMatmulTilingBase::DoSplitMTiling(Mc2Tiling::RCSTiling &rcfCfg)
{
    // cmdType = HCCL_CMD_ALLGATHER, 是允许切K
    if (args_.enableSplitK) { // 只有1份
        OP_LOGI(opName_, "enableSplitK is true.");
        rcfCfg.tileCnt = 1;
        rcfCfg.tailCnt = 0;
        rcfCfg.tailM = 0;
    } else if (args_.commTurn != 0) {
        OP_LOGI(opName_, "commTurn is %lu.", args_.commTurn);
        uint64_t splite = AllGatherSplitM(args_);

        // 现在找到1个合适的切分
        auto tileCnt = args_.mValue / splite;  // 切的份数
        auto tileTail = args_.mValue % splite; // 尾巴

        rcfCfg.tileCnt = tileCnt;
        tileMValue_ = splite;
        rcfCfg.tailCnt = 0;
        rcfCfg.tailM = tileTail;
        if (tileTail != 0) {
            tailMValue_ = tileTail;
        }
    } else {
        CutResult mCutAllgather = GetTilingResult();
        rcfCfg.tileCnt = mCutAllgather.numLongTile;
        tileMValue_ = mCutAllgather.longTileLen;
        rcfCfg.tailCnt = 0;
        rcfCfg.tailM = 0;
        if (mCutAllgather.numShortTile > 0) {
            rcfCfg.tailM = mCutAllgather.shortTileLen;
            tailMValue_ = mCutAllgather.shortTileLen;
            rcfCfg.tailCnt = mCutAllgather.numShortTile;
        }
    }
}

void AllGatherMatmulTilingBase::Reset()
{
    tileMValue_ = 0UL;
    tailMValue_ = 0UL;
    rankSize_ = 0L;
    outputIsFp8_ = false;
    inputIsBf16Fp16_ = true;
    commAlgorithm_ = 0U;
    enableNd2Nz_ = true;
    castBias_ = false;
    biasLen_ = 0U;
    storageA_ = 0U;
    gatherIndex_ = 0U;
}

bool AllGatherMatmulTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK((attrs == nullptr), VECTOR_INNER_ERR_REPORT_TILING(opName_, "failed to get attrs"), return false);
    group_ = attrs->GetAttrPointer<char>(GROUP);
    auto isTransA = attrs->GetAttrPointer<bool>(IS_TRANS_A);
    auto isTransB = attrs->GetAttrPointer<bool>(IS_TRANS_B);
    auto gatherIndexPtr = attrs->GetAttrPointer<int>(GATHER_IDX);
    auto commTurn = attrs->GetAttrPointer<int>(COMM_TURN);
    OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group_, rankSize_),
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "GetRankSize failed."), return false);
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankSize_) == SUPPORT_RANK_SIZE.end(),
        VECTOR_INNER_ERR_REPORT_TILING(
            opName_, "world_size should be 2 or 4 or 8 or 16 or 32 or 64, but the actual value is %ld.", rankSize_),
        return false);
    OP_TILING_CHECK(commTurn == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "commTurn is nullptr!"), return false);
    OP_TILING_CHECK(*commTurn != 0,
                    VECTOR_INNER_ERR_REPORT_TILING(
                        opName_, "The expected value of commTurn is 0, but the actual value is %d.", *commTurn),
                    return false);
    args_.isATrans = isTransA ? *isTransA : 0;
    args_.isBTrans = isTransB ? *isTransB : 0;
    args_.cmdType = mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER;
    args_.rankDim = static_cast<uint32_t>(rankSize_);
    args_.commTurn = commTurn ? *commTurn : 0;
    gatherIndex_ = gatherIndexPtr ? *gatherIndexPtr : 0;
    OP_TILING_CHECK((args_.isATrans != 0),
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "the isTransA should be false, but real value is true"),
                    return false);
    OP_TILING_CHECK(
        (gatherIndex_ != 0),
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "the gatherIndex should be 0, but real value is %u", gatherIndex_),
        return false);
    auto blockSize = *context_->GetAttrs()->GetAttrPointer<int>(BLOCK_SIZE_INDEX);
    OP_TILING_CHECK(
        blockSize != 0,
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "blockSize should be 0, but the actual value is %u.", blockSize),
        return false);
    OP_LOGD(opName_,
            " group=%s, rankSize=%ld, is_trans_a=%u, is_trans_b=%d, gather_index=%u,"
            " comm_turn=%lu",
            group_, rankSize_, args_.isATrans, args_.isBTrans, gatherIndex_, args_.commTurn);

    return true;
}

ge::graphStatus AllGatherMatmulTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    args_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    // 未来可以在此处校验平台是否支持当前输入
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus AllGatherMatmulTilingBase::GetShapeAttrsInfo() { return AnalyzeShapeAttr(); };
ge::graphStatus AllGatherMatmulTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AllGatherMatmulTilingBase::GetStorageA(Mc2Tiling::RCSTiling &rcsCfg)
{
    constexpr uint64_t alignAddrLen = 512;
    uint32_t gatherIndex = rcsCfg.gatherIndex;
    uint64_t nd2nzLen = 0;
    uint64_t storageA = 0;

    // step1: ND2NZ
    if (gatherIndex == 0U) { // 转置B
        // 计算ND2NZ需使用空间方法保持与MMV3 tiling计算逻辑一致
        uint64_t alignByte = 256 / args_.inputDtypeSize; // 256B 对齐shape
        uint64_t kALign = ops::CeilAlign(static_cast<uint64_t>(rcsCfg.rankK), alignByte);
        uint64_t nALign = ops::CeilAlign(static_cast<uint64_t>(rcsCfg.rankN), alignByte);
        nd2nzLen = kALign * nALign * args_.inputDtypeSize;
    } else {
        auto alignM = rcsCfg.rankM + 16;
        auto alignK = rcsCfg.rankK + 16;
        nd2nzLen = mc2tiling::AlignUp(alignM * alignK * args_.inputDtypeSize, alignAddrLen);
    }

    if (args_.cmdType == mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER) {
        uint64_t gmcFloat = 0; // allgatherMm 通信后数据只需放在gatherLen对应的workspace或者gatherout中，不需要gmcFloat
        uint64_t gatherLen = 0;
        if (args_.isStorageGather == false) {
            if (gatherIndex == 0U) { // A矩阵
                gatherLen = mc2tiling::AlignUp(rcsCfg.rankM * rcsCfg.rankK * args_.inputDtypeSize, alignAddrLen);
            } else {
                gatherLen = mc2tiling::AlignUp(rcsCfg.rankK * rcsCfg.rankN * args_.inputDtypeSize, alignAddrLen);
            }
            gatherLen *= rcsCfg.rankDim;
        }

        rcsCfg.nd2NzWorkLen = nd2nzLen;
        rcsCfg.cToFloatLen = gmcFloat;
        rcsCfg.gatherLen = gatherLen;

        storageA = nd2nzLen + gmcFloat + gatherLen; // 需要计算存放的A矩阵
    }
    return storageA;
}

ge::graphStatus AllGatherMatmulTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "get workspace failed"),
                    return ge::GRAPH_FAILED);

    workspaceSize_ = libApiWorkSpaceSize_ + storageA_ + biasLen_;
    workspaces[0] = workspaceSize_;
    OP_LOGD(opName_, "workspaces[0] size=%ld, biasLen=%d", workspaces[0], biasLen_);

    return ge::GRAPH_SUCCESS;
}

uint64_t AllGatherMatmulTilingBase::GetTilingKey() const
{
    uint8_t outputType = (outputIsFp8_) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(inputIsBf16Fp16_, args_.isBTrans, outputType, TPL_DEFAULT_MODE, SCALE_TYPE_NOT_IS_MX);
    return tilingKey;
}

} // namespace optiling
