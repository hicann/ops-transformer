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
 * \file weight_quant_matmul_all_reduce_tiling_950.cc
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_

#include "weight_quant_matmul_all_reduce_tiling_950.h"
#include "common/utils/op_mc2.h"
#include "mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_apt_tiling_key.h"
#include "all_reduce_fit_balance_tiling.h"
#include "mc2_comm_utils.h"

using namespace Mc2Tiling;
namespace optiling {
constexpr int64_t ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;
constexpr uint64_t STANDARD_CARD_CGMPAD_WORKSPACE_CNT = 3;
constexpr uint64_t FIVE_ONE_TWO = 512;
constexpr uint64_t WEIGHT_QUANT_A2APATH_CRITICAL_K = 2048;
bool WeightQuantMatmulAllReduceTilingA5::IsCapable()
{
    if (isA16W8_ || isA16W4_) {
        OP_LOGI(opName_, "Start with weight quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip weight quant tiling as dtype not support.");
    return false;
}

ge::graphStatus WeightQuantTilingTransferHelperA5::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start fill matmul info.");
    auto &&tilingArgs = tilingProcesser_.args_;
    opName_ = tilingProcesser_.opName_;
    try {
        matmulInfoPtr_ = std::make_unique<Mc2WeightQuantBatchMatmulInfo>();
    } catch (const std::bad_alloc &e) {
        OP_LOGE(opName_, "Failed to create matmul info.");
        return ge::GRAPH_FAILED;
    }
    matmulInfoPtr_->transA = tilingArgs.isATrans;
    matmulInfoPtr_->transB = tilingArgs.isBTrans;
    matmulInfoPtr_->hasBias = tilingArgs.isBias;
    matmulInfoPtr_->hasAntiQuantOffset = tilingProcesser_.HasAntiQuantOffset();
    matmulInfoPtr_->mSize = tilingArgs.mValue;
    matmulInfoPtr_->kSize = tilingArgs.kValue;
    matmulInfoPtr_->nSize = tilingArgs.nValue;
    matmulInfoPtr_->aDtype = tilingArgs.geAType;
    matmulInfoPtr_->bDtype = tilingArgs.geBType;
    matmulInfoPtr_->cDtype = tilingArgs.geCType;
    matmulInfoPtr_->biasDtype = tilingArgs.geBiasType;
    matmulInfoPtr_->antiQuantType = tilingProcesser_.antiQuantType_;
    matmulInfoPtr_->groupSize = tilingProcesser_.antiGroupSize_;
    matmulInfoPtr_->quantType = tilingProcesser_.quantType_;
    matmulInfoPtr_->bFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(tilingProcesser_.mmrCtxInfo_.x2->GetStorageFormat()));
    OP_TILING_CHECK((matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) &&
                        (matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_CHANNEL),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        matmulInfoPtr_->opName, "antiQuant",
                        std::to_string(static_cast<int>(matmulInfoPtr_->antiQuantType)).c_str(),
                        "When the weight format is Nz, the value of antiQuant must be per-channel"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

WeightQuantMMAllReduceTilingKeyParams WeightQuantTilingTransferHelperA5::GetWeightQuantMMAllReduceTPLParam()
{
    WeightQuantMMAllReduceTilingKeyParams tplParam;
    tplParam.transB = matmulInfoPtr_->transB;
    tplParam.antiQuantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tplParam.hasAntiQuantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    return tplParam;
}

ge::graphStatus WeightQuantAsTilingTransferHelper::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start fill weight fp8/hif8 matmul info.");
    auto &&fp8Hif8TilingArgs = tilingProcesser_.args_;
    opName_ = tilingProcesser_.opName_;
    try {
        matmulInfoPtr_ = std::make_unique<Mc2WeightQuantBatchMatmulInfo>();
    } catch (const std::bad_alloc &e) {
        OP_LOGE(opName_, "Failed to create matmul info.");
        return ge::GRAPH_FAILED;
    }
    matmulInfoPtr_->transA = fp8Hif8TilingArgs.isATrans;
    matmulInfoPtr_->transB = fp8Hif8TilingArgs.isBTrans;
    matmulInfoPtr_->hasBias = fp8Hif8TilingArgs.isBias;
    matmulInfoPtr_->hasAntiQuantOffset = tilingProcesser_.HasAntiQuantOffset();
    matmulInfoPtr_->mSize = fp8Hif8TilingArgs.mValue;
    matmulInfoPtr_->kSize = fp8Hif8TilingArgs.kValue;
    matmulInfoPtr_->nSize = fp8Hif8TilingArgs.nValue;
    matmulInfoPtr_->aDtype = fp8Hif8TilingArgs.geAType;
    matmulInfoPtr_->bDtype = fp8Hif8TilingArgs.geBType;
    matmulInfoPtr_->cDtype = fp8Hif8TilingArgs.geCType;
    matmulInfoPtr_->biasDtype = fp8Hif8TilingArgs.geBiasType;
    matmulInfoPtr_->antiQuantScaleDtype = fp8Hif8TilingArgs.antiquantscaleDType;
    matmulInfoPtr_->antiQuantType = tilingProcesser_.antiQuantType_;
    matmulInfoPtr_->groupSize = tilingProcesser_.antiGroupSize_;
    matmulInfoPtr_->quantType = tilingProcesser_.quantType_;
    matmulInfoPtr_->bFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(tilingProcesser_.mmrCtxInfo_.x2->GetStorageFormat()));
    PrintTilingInputParam(matmulInfoPtr_);
    OP_TILING_CHECK((matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ),
                    OP_LOGE_FOR_INVALID_FORMAT(matmulInfoPtr_->opName, "x2",
                                               std::to_string(static_cast<int32_t>(matmulInfoPtr_->bFormat)).c_str(),
                                               std::to_string(static_cast<int32_t>(ge::FORMAT_FRACTAL_NZ)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

WeightQuantMMAllReduceTilingKeyParams WeightQuantAsTilingTransferHelper::GetWeightQuantASMMAllReduceTPLParam()
{
    WeightQuantMMAllReduceTilingKeyParams tplParam;
    tplParam.transB = matmulInfoPtr_->transB;
    tplParam.templateCustom = static_cast<uint8_t>(mte2Config_);
    tplParam.antiQuantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tplParam.quantType = static_cast<uint8_t>(matmulInfoPtr_->quantType);
    tplParam.hasAntiQuantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    if (matmulInfoPtr_->biasDtype == ge::DT_FLOAT && matmulInfoPtr_->hasBias) {
        tplParam.biasIsExist = true;
        tplParam.isBiasFp32 = true;
    }
    return tplParam;
}

void WeightQuantAsTilingTransferHelper::PrintTilingInputParam(
    std::unique_ptr<Mc2WeightQuantBatchMatmulInfo> &matmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, "The transA_=%d, transB_=%d, hasBias_=%d, hasAntiQuantOffset_=%d.",
            matmulInfo->transA, matmulInfo->transB, matmulInfo->hasBias, matmulInfo->hasAntiQuantOffset);
    OP_LOGD(tilingProcesser_.opName_, "The mSize_=%ld, kSize_=%ld, nSize_=%ld, groupSize_=%ld.", matmulInfo->mSize,
            matmulInfo->kSize, matmulInfo->nSize, matmulInfo->groupSize);
    OP_LOGD(tilingProcesser_.opName_, "The aDtype_=%d, bDtype_=%d, cDtype_=%d, biasDtype_=%d.",
            static_cast<int32_t>(matmulInfo->aDtype), static_cast<int32_t>(matmulInfo->bDtype),
            static_cast<int32_t>(matmulInfo->cDtype), static_cast<int32_t>(matmulInfo->biasDtype));
    OP_LOGD(tilingProcesser_.opName_, "The antiQuantType_=%d, antiQuantScaleType_=%d, quantType_=%d, bFormat=%d.",
            static_cast<int32_t>(matmulInfo->antiQuantType), static_cast<int32_t>(matmulInfoPtr_->antiQuantScaleDtype),
            static_cast<int32_t>(matmulInfo->quantType), static_cast<int32_t>(matmulInfo->bFormat));
}

ge::graphStatus WeightQuantTilingTransferHelperA5::PostTiling()
{
    PrintCVTilingData(true);
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, "Set matmul workspace size=%lu to mc2.", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantAsTilingTransferHelper::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, "Weight quant fp8/hif8 set mm workspace size=%lu to mc2.",
            tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantMatmulAllReduceTilingA5::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().M = args_.orgMValue;
    MutableTCubeTileTilingData().isBias = args_.isBias;
    MutableTCubeTileTilingData().usedCoreNum = 1;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA16W8());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    MC2_CHECK_LOG_RET(opName_, SetMc2Hcomm());
    DoRCSTiling();
    DoSplitMTiling();
    if (isKZero_) {
        DoEmptyTensorTiling();
        DoAllReduceTiling(true);
        return ge::GRAPH_SUCCESS;
    }

    if (antiQuantType_ != AntiQuantType::PER_GROUP) {
        MC2_CHECK_LOG_RET(opName_, DoWeightQuantAsTiling());
    } else {
        MC2_CHECK_LOG_RET(opName_, DoWeightQuantTiling());
    }

    DoAllReduceTiling();
    return ge::GRAPH_SUCCESS;
}

uint64_t WeightQuantMatmulAllReduceTilingA5::GetTilingKey() const
{
    if (unlikely(isKZero_)) {
        const uint64_t tilingKey =
            GET_TPL_TILING_KEY(MMTYPE_WEIQUANT_NULL_TENSOR, false, false, false, SET_NOT_USE_FP_MM_TILING,
                               SET_NOT_USE_QUANT_MM_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
        return tilingKey;
    }
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    uint8_t commMode = GetCommMode();
    if (commMode == Mc2Comm::COMM_MODE_CCU && k >= WEIGHT_QUANT_A2APATH_CRITICAL_K &&
        mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        isUseA2APath = false;
    }
    bool isA2ARSAG = isUseA2APath;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        MMTYPE_WEIGHT_QUANT_MM, WeightQuantTPLPatams_.transB, WeightQuantTPLPatams_.biasIsExist, isA2ARSAG, commMode,
        SET_NOT_USE_FP_MM_TILING, SET_NOT_USE_QUANT_MM_TILING, WeightQuantTPLPatams_.templateCustom,
        WeightQuantTPLPatams_.antiQuantType, WeightQuantTPLPatams_.quantType, WeightQuantTPLPatams_.hasAntiQuantOffset,
        WeightQuantTPLPatams_.isBiasFp32);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: transB, biasIsExist is: [%d,%d].", WeightQuantTPLPatams_.transB,
            WeightQuantTPLPatams_.biasIsExist);
    OP_LOGD(opName_,
            "Mc2MatmulAllReduce: templateCustom, antiQuantType, quantType, "
            "hasAntiQuantOffset, isBiasFp32 is: [%u, %u,%u,%d,%d].",
            WeightQuantTPLPatams_.templateCustom, WeightQuantTPLPatams_.antiQuantType, WeightQuantTPLPatams_.quantType,
            WeightQuantTPLPatams_.hasAntiQuantOffset, WeightQuantTPLPatams_.isBiasFp32);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: weight_quant_TilingKey=%lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::GetWorkspaceSizeForA2ARSAG()
{
    uint64_t commWorkSpace = 0UL;
    uint64_t cgmPadLen = 0UL;
    uint64_t tileM = MutableTCubeTileTilingData().M;
    uint64_t tailM = MutableTCubeTailTilingData().M;
    uint64_t tempTileSize = tileM * MutableTCubeTileTilingData().N;
    uint64_t tempTailSize = tailM * MutableTCubeTailTilingData().N;

    cgmPadLen = (MutableRCSTilingData().tileCnt + MutableRCSTilingData().tailCnt) * FIVE_ONE_TWO;
    commWorkSpace =
        (tempTileSize * MutableRCSTilingData().tileCnt + tempTailSize * MutableRCSTilingData().tailCnt + cgmPadLen) *
        static_cast<uint64_t>(args_.outputDtypeSize);
    OP_LOGI(opName_, "WeightQuantMatmulAllReduceTilingA5 Set commWorkSpace size=%lu to context.", commWorkSpace);

    // MatMul输出存储+alltoall输出存储+reduceSum输出存储
    uint64_t workspaceSizeCount = STANDARD_CARD_CGMPAD_WORKSPACE_CNT;
    myWorkSpaceSize_ = myWorkSpaceSize_ + commWorkSpace * workspaceSizeCount + commWorkSpace / args_.rankDim;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::GetWorkspaceSize()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::GetWorkspaceSize());
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910B) {
        myWorkSpaceSize_ = myWorkSpaceSize_ + MutableRCSTilingData().biasLen;
        if (isKZero_) {
            myWorkSpaceSize_ = myWorkSpaceSize_ + libApiWorkSpaceSize_;
            OP_LOGD(opName_, "Empty tensor k is 0, set workspace size=%lu to context.", myWorkSpaceSize_);
        }
    } else {
        bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
        const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
        uint8_t commMode = GetCommMode();
        if (commMode == Mc2Comm::COMM_MODE_CCU && k >= WEIGHT_QUANT_A2APATH_CRITICAL_K &&
            mc2tiling::Is8P(args_.rankDim, npuArch_)) {
            isUseA2APath = false;
        }
        if (isUseA2APath) {
            OP_TILING_CHECK(GetWorkspaceSizeForA2ARSAG() != ge::GRAPH_SUCCESS,
                            OP_LOGE(opName_, "get workspace size By GetWorkspaceSizeForA2ARSAG failed."),
                            return ge::GRAPH_FAILED);
        }
        myWorkSpaceSize_ += workspaceSize_;
    }
    OP_LOGI(opName_, "Set max workspace size=%lu to context.", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

void WeightQuantMatmulAllReduceTilingA5::PrintMatmulAsTiling(bool isTail)
{
    auto &tiling = weightQuantMatmulAllReduceA5Fp8TilingData_.tileMmASTiling;
    if (isTail) {
        tiling = weightQuantMatmulAllReduceA5Fp8TilingData_.tailMmASTiling;
    }

    OP_LOGD(opName_, "Tiling.cubeNumBlocksN=%u.", tiling.cubeBlockDimN);
    OP_LOGD(opName_, "Tiling.cubeNumBlocksM=%u.", tiling.cubeBlockDimM);
    OP_LOGD(opName_, "Tiling.hasBias=%u.", tiling.hasBias);
    OP_LOGD(opName_, "Tiling.firstTailBlockCoun=%u.", tiling.firstTailBlockCount);
    OP_LOGD(opName_, "Tiling.secondTailBlockCount=%u.", tiling.secondTailBlockCount);
    OP_LOGD(opName_, "Tiling.weightL2Cacheable=%u.", tiling.weightL2Cacheable);

    OP_LOGD(opName_, "Tiling.mainBlockL1Size=%u.", tiling.mainBlockL1Size);
    OP_LOGD(opName_, "Tiling.firstTailBlockL1Size=%u.", tiling.firstTailBlockL1Size);
    OP_LOGD(opName_, "Tiling.secondTailBlockL1Size=%u.", tiling.secondTailBlockL1Size);
    OP_LOGD(opName_, "Tiling.aPreloadSize=%u.", tiling.aPreloadSize);
    OP_LOGD(opName_, "Tiling.groupSize=%lu.", tiling.groupSize);
    OP_LOGD(opName_, "Tiling.mainBlockCount=%lu.", tiling.mainBlockCount);
    OP_LOGD(opName_, "Tiling.mSize=%lu.", tiling.mSize);
    OP_LOGD(opName_, "Tiling.kSize=%lu.", tiling.kSize);
    OP_LOGD(opName_, "Tiling.nSize=%lu.", tiling.nSize);
}

void WeightQuantMatmulAllReduceTilingA5::PrintExtendMatmulTiling(bool isTail)
{
    if (antiQuantType_ != AntiQuantType::PER_GROUP) {
        PrintMatmulAsTiling(isTail);
        return;
    }

    auto &tiling = weightQuantMatmulAllReduceA5TilingData_.tileRegBaseMmTiling;
    if (isTail) {
        tiling = weightQuantMatmulAllReduceA5TilingData_.tailRegBaseMmTiling;
    }

    OP_LOGD(opName_, "Tiling.cubeNumBlocksN=%u.", tiling.cubeBlockDimN);
    OP_LOGD(opName_, "Tiling.cubeNumBlocksM=%u.", tiling.cubeBlockDimM);
    OP_LOGD(opName_, "Tiling.vecCoreParallel=%u.", tiling.vecCoreParallel);
    OP_LOGD(opName_, "Tiling.reserve1=%u.", tiling.reserve1);
    OP_LOGD(opName_, "Tiling.AL1Pingpong=%u.", tiling.AL1Pingpong);
    OP_LOGD(opName_, "Tiling.BL1Pingpong=%u.", tiling.BL1Pingpong);

    OP_LOGD(opName_, "Tiling.kSize=%lu.", tiling.kSize);
    OP_LOGD(opName_, "Tiling.nSize=%lu.", tiling.nSize);
    OP_LOGD(opName_, "Tiling.groupSize=%lu.", tiling.groupSize);
    OP_LOGD(opName_, "Tiling.mSize=%lu.", tiling.mSize);
    OP_LOGD(opName_, "Tiling.nBubSize=%lu.", tiling.nBubSize);
    OP_LOGD(opName_, "Tiling.kBubSize=%lu.", tiling.kBubSize);
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::PostTiling()
{
    size_t dataSize = 0;
    if (antiQuantType_ != AntiQuantType::PER_GROUP) {
        dataSize = sizeof(WeightQuantMatmulAllReduceA5Fp8TilingData);
    } else {
        dataSize = sizeof(WeightQuantMatmulAllReduceA5TilingData);
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", dataSize,
            context_->GetRawTilingData()->GetCapacity());

    OP_TILING_CHECK(dataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", dataSize), return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(dataSize);

    if (antiQuantType_ != AntiQuantType::PER_GROUP) {
        errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                               reinterpret_cast<void *>(&weightQuantMatmulAllReduceA5Fp8TilingData_), dataSize);
        if (ret != EOK) {
            OP_LOGE(context_->GetNodeName(), "Memcpy_s failed, ret=%d", ret);
            return ge::GRAPH_FAILED;
        }
    } else {
        errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                               reinterpret_cast<void *>(&weightQuantMatmulAllReduceA5TilingData_), dataSize);
        if (ret != EOK) {
            OP_LOGE(context_->GetNodeName(), "Memcpy_s failed, ret=%d", ret);
            return ge::GRAPH_FAILED;
        }
    }

    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::SetMc2HcommAllReduce(const char *groupName,
                                                                         const uint32_t reduceType)
{
    const std::string algConfig = "AllReduce=level0:fullmesh";
    const uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    const uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType, dataType, dataType);
    uint8_t commModeVal = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commModeVal), OP_LOGE(opName_, "Get commMode failed."),
                    return ge::GRAPH_FAILED);
    if (commModeVal == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commModeVal == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    if (antiQuantType_ != AntiQuantType::PER_GROUP) {
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5Fp8TilingData_.mc2InitTiling) != 0,
                        OP_LOGE(opName_, "Get mc2InitTiling from weightQuantMatmulAllReduceA5Fp8TilingData failed."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5Fp8TilingData_.mc2CcTiling) != 0,
                        OP_LOGE(opName_, "Get mc2CcTiling from weightQuantMatmulAllReduceA5Fp8TilingData failed."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5TilingData_.mc2InitTiling) != 0,
                        OP_LOGE(opName_, "Get mc2InitTiling from weightQuantMatmulAllReduceA5Fp8TilingData failed."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5TilingData_.mc2CcTiling) != 0,
                        OP_LOGE(opName_, "Get mc2CcTiling from weightQuantMatmulAllReduceA5Fp8TilingData failed."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::SetMc2HcommTwoShot(const char *groupName, const uint32_t reduceType)
{
    uint32_t opType1 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLTOALL);
    uint32_t opType2 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));
    const std::string algConfig1 = "AlltoAll=level0:fullmesh";
    const std::string algConfig2 = "AllGather=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType1, algConfig1, reduceType, dataType, dataType);
    uint8_t commModeVal = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commModeVal), OP_LOGE(opName_, "Get commMode failed."),
                    return ge::GRAPH_FAILED);
    if (commModeVal == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commModeVal == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    if (antiQuantType_ != AntiQuantType::PER_GROUP) {
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5Fp8TilingData_.mc2InitTiling),
                        OP_LOGE(opName_, "Get mc2InitTiling from weightQuantMatmulAllReduceA5Fp8TilingData_ failed."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5Fp8TilingData_.mc2CcTiling),
                        OP_LOGE(opName_, "Get mc2CcTiling from weightQuantMatmulAllReduceA5Fp8TilingData_ failed."),
                        return ge::GRAPH_FAILED);
        mc2CcTilingConfig.SetGroupName(groupName);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfig2);
        mc2CcTilingConfig.SetReduceType(reduceType, dataType, dataType);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5Fp8TilingData_.mc2CcTilingComm),
                        OP_LOGE(opName_, "Get mc2CcTilingComm from weightQuantMatmulAllReduceA5Fp8TilingData_ failed."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5TilingData_.mc2InitTiling),
                        OP_LOGE(opName_, "Get mc2InitTiling from weightQuantMatmulAllReduceA5TilingData_ failed."),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5TilingData_.mc2CcTiling),
                        OP_LOGE(opName_, "Get mc2CcTiling from weightQuantMatmulAllReduceA5TilingData_ failed."),
                        return ge::GRAPH_FAILED);
        mc2CcTilingConfig.SetGroupName(groupName);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfig2);
        mc2CcTilingConfig.SetReduceType(reduceType, dataType, dataType);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(weightQuantMatmulAllReduceA5TilingData_.mc2CcTilingComm),
                        OP_LOGE(opName_, "Get mc2CcTilingComm from weightQuantMatmulAllReduceA5TilingData_ failed."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::SetMc2Hcomm()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "y", Ops::Base::ToString(args_.geCType).c_str(),
                                              "FLOAT16, BF16, FLOAT or INT8"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetAttrs() == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"),
                    return ge::GRAPH_FAILED);
    const uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    const char *groupName = context_->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    uint8_t commMode = GetCommMode();
    if (commMode == Mc2Comm::COMM_MODE_CCU && k >= WEIGHT_QUANT_A2APATH_CRITICAL_K &&
        mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        isUseA2APath = false;
    }
    if (isUseA2APath) {
        OP_TILING_CHECK(
            SetMc2HcommTwoShot(groupName, reduceType) != ge::GRAPH_SUCCESS,
            OP_LOGE(opName_, "WeightQuantMatmulAllReduceTilingA5 set Mc2Hcomm config By SetMc2HcommTwoShot failed."),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            SetMc2HcommAllReduce(groupName, reduceType) != ge::GRAPH_SUCCESS,
            OP_LOGE(opName_, "WeightQuantMatmulAllReduceTilingA5 set Mc2Hcomm config By SetMc2HcommAllReduce failed."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::DoWeightQuantTiling()
{
    args_.mValue = tileMValue_;
    WeightQuantTilingTransferHelperA5 mmTile(*this, weightQuantMatmulAllReduceA5TilingData_.tileRegBaseMmTiling);
    if (args_.enableSplitK) {
        ge::graphStatus curStatus = mmTile.MatmulDoTiling();
        WeightQuantTPLPatams_ = mmTile.GetWeightQuantMMAllReduceTPLParam();
        return curStatus;
    } else {
        MC2_CHECK_LOG_RET(opName_, mmTile.MatmulDoTiling());
        if (MutableRCSTilingData().tailCnt == 0) {
            WeightQuantTPLPatams_ = mmTile.GetWeightQuantMMAllReduceTPLParam();
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        WeightQuantTilingTransferHelperA5 mmTail(*this, weightQuantMatmulAllReduceA5TilingData_.tailRegBaseMmTiling);
        ge::graphStatus curStatus = mmTail.MatmulDoTiling();
        WeightQuantTPLPatams_ = mmTile.GetWeightQuantMMAllReduceTPLParam();
        return curStatus;
    }
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::CheckAxisSize()
{
    uint64_t alignDim = 32;
    if (isA16W4_) {
        alignDim = 64UL;
    }
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(m > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", std::to_string(m).c_str(),
                                                          "The value of m of x1 exceeds the upper limit INT32_MAX"),
                    return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(k > static_cast<uint64_t>(UINT16_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", std::to_string(k).c_str(),
                                                          "The value of k of x1 exceeds the upper limit UINT16_MAX"),
                    return ge::GRAPH_FAILED);
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(n > static_cast<uint64_t>(UINT16_MAX),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x2", std::to_string(n).c_str(),
                                                          "The size of n-axis exceeds the upper limit UINT16_MAX"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isA16W8_ && (antiQuantType_ == AntiQuantType::PER_GROUP) &&
                        ((n % alignDim != 0) || (k % alignDim != 0)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "x2", ("k=" + std::to_string(k) + ", n=" + std::to_string(n)).c_str(),
                        "When the quantization mode is A16W8 or F8 pergroup, K and N of x2 must align to 32"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isA16W4_ && (antiQuantType_ == AntiQuantType::PER_GROUP) &&
                        ((n % alignDim != 0) || (k % alignDim != 0)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "x2", ("k=" + std::to_string(k) + ", n=" + std::to_string(n)).c_str(),
                        "When the quantization mode is A16W4 pergroup, K and N of x2 must align to 64"),
                    return ge::GRAPH_FAILED);
    return CheckWeightQuantEmptyTensor();
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::DoWeightQuantAsTiling()
{
    args_.mValue = tileMValue_;
    WeightQuantAsTilingTransferHelper mmTile(*this, weightQuantMatmulAllReduceA5Fp8TilingData_.tileMmASTiling);
    MC2_CHECK_LOG_RET(opName_, mmTile.MatmulDoTiling());
    if (MutableRCSTilingData().tailCnt == 0) {
        WeightQuantTPLPatams_ = mmTile.GetWeightQuantASMMAllReduceTPLParam();
        return ge::GRAPH_SUCCESS;
    }
    args_.mValue = tailMValue_;
    WeightQuantAsTilingTransferHelper mmTail(*this, weightQuantMatmulAllReduceA5Fp8TilingData_.tailMmASTiling);
    ge::graphStatus curStatus = mmTail.MatmulDoTiling();
    WeightQuantTPLPatams_ = mmTail.GetWeightQuantASMMAllReduceTPLParam();
    return curStatus;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::CheckBiasInput()
{
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    if (mmrCtxInfo_.bias_shape != nullptr) {
        const auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(x1Type != biasType,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "bias",
                                                              std::to_string(static_cast<int32_t>(biasType)).c_str(),
                                                              "The value of bias dtype must be the same as that of x1"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTilingA5::CheckInput()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::CheckInput());
    const size_t x2DimNum = (static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtxInfo_.x2->GetStorageFormat())) ==
                                     ge::Format::FORMAT_FRACTAL_NZ ?
                                 4 :
                                 2);
    const size_t actualX2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x2DimNum != actualX2DimNum,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "x2", (std::to_string(actualX2DimNum) + "D").c_str(),
                        ("The shape dim of x2 must be " + std::to_string(x2DimNum)).c_str()),
                    return ge::GRAPH_FAILED);
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    OP_TILING_CHECK(!((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1",
                                                          std::to_string(static_cast<int32_t>(x1Type)).c_str(),
                                                          "The value of x1 dtype must be fp16 or bf16"),
                    return ge::GRAPH_FAILED);
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(!((x2Type == ge::DT_INT8) || (x2Type == ge::DT_INT4) || (x2Type == ge::DT_FLOAT8_E4M3FN) ||
                      (x2Type == ge::DT_HIFLOAT8)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x2",
                                                          std::to_string(static_cast<int32_t>(x2Type)).c_str(),
                                                          "The value of x2 dtype must be int8, int4, fp8_e4m3 or hif8"),
                    return ge::GRAPH_FAILED);

    if ((x2Type == ge::DT_FLOAT8_E4M3FN) || (x2Type == ge::DT_HIFLOAT8)) {
        isWeightFp8Hif8_ = true;
        OP_TILING_CHECK(mmrCtxInfo_.antiquant_offset != nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "antiquant_offset"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(CheckBiasInput(), OP_LOGE(context_->GetNodeName(), "Check bias failed."), return ge::GRAPH_FAILED);
    // x1,antiquantScale数据类型相同
    auto antiquantScaleType = mmrCtxInfo_.antiquant_scale->GetDataType();
    OP_TILING_CHECK(
        antiquantScaleType != x1Type,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "antiquant_scale",
                                              std::to_string(static_cast<int32_t>(antiquantScaleType)).c_str(),
                                              "The value of antiquant_scale dtype must be the same as that of x1"),
        return ge::GRAPH_FAILED);
    // antiquantScale和antiquantOffset数据类型相同
    if (mmrCtxInfo_.antiquant_offset_shape != nullptr) {
        auto antiquantOffsetType = mmrCtxInfo_.antiquant_offset->GetDataType();
        OP_TILING_CHECK(antiquantOffsetType != antiquantScaleType,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "antiquant_offset",
                            std::to_string(static_cast<int32_t>(antiquantOffsetType)).c_str(),
                            "The value of antiquant_offset dtype must be the same as that of antiquantScale"),
                        return ge::GRAPH_FAILED);
    }
    // antiquantgroupsize 校验
    uint64_t kValue = GetKValue();
    if (kValue != 0 && mmrCtxInfo_.antiquantGroupSizePtr != nullptr) {
        const int64_t groupSize = *(mmrCtxInfo_.antiquantGroupSizePtr);
        OP_TILING_CHECK(((groupSize != 0) && ((groupSize % ANTIQUANT_GROUP_SIZE_MIN_VALUE != 0) ||
                                              (groupSize < ANTIQUANT_GROUP_SIZE_MIN_VALUE) ||
                                              (groupSize > std::min(static_cast<int32_t>(kValue - 1), INT32_MAX)))),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "antiquantGroupSize", std::to_string(groupSize).c_str(),
                            "The value of antiquantGroupSize must be in the range [32, min(k-1, INT32_MAX)]"),
                        return ge::GRAPH_FAILED);
    }
    // pergroup场景下不支持fp8和hif8,增加校验
    OP_TILING_CHECK((antiQuantType_ == AntiQuantType::PER_GROUP) &&
                        ((x2Type == ge::DT_FLOAT8_E4M3FN) || (x2Type == ge::DT_HIFLOAT8)),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "x2",
                                              ge::TypeUtils::DataTypeToSerialString(x2Type).c_str(),
                                              "DT_FLOAT8_E4M3FN and DT_HIFLOAT8 not supported in per-group"),
                    return ge::GRAPH_FAILED);

    // pertensor场景下不支持fp8和hif8,增加校验
    OP_TILING_CHECK((antiQuantType_ == AntiQuantType::PER_TENSOR) &&
                        ((x2Type == ge::DT_FLOAT8_E4M3FN) || (x2Type == ge::DT_HIFLOAT8)),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "x2",
                                              ge::TypeUtils::DataTypeToSerialString(x2Type).c_str(),
                                              "DT_FLOAT8_E4M3FN and DT_HIFLOAT8 not supported in per-tensor"),
                    return ge::GRAPH_FAILED);

    return CheckAxisSize();
}
WeightQuantMatmulAllReduceTilingA5::WeightQuantMatmulAllReduceTilingA5(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context),
      weightQuantMatmulAllReduceA5TilingData_(weightQuantMatmulAllReduceA5TilingDataSelf_),
      weightQuantMatmulAllReduceA5Fp8TilingData_(weightQuantMatmulAllReduceA5Fp8TilingDataSelf_)
{
}

CutResult WeightQuantMatmulAllReduceTilingA5::GetTilingResult()
{
    CutResult mCutAllreduceOutput;
    SocVersion inputSocVer = SocVersion::SOC910_B;
    SetMCutSocVersion(inputSocVer);
    const gert::StorageShape *commQuantScaleFirst = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleSecond = mmrCtxInfo_.comm_quant_scale_2_shape;
    if (mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_)) {
        MMAllReduceFitBalanceTiling allReduceTilingHcclInst(args_, KernelType::ALL_REDUCE_VIA_TWO_SHOT,
                                                            TopoType::STANDARD_CARD);
        mCutAllreduceOutput = allReduceTilingHcclInst.GetTiling();
    } else if (mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
        uint8_t commMode = GetCommMode();
        KernelType kernelType;
        if (commMode == Mc2Comm::COMM_MODE_CCU && k >= WEIGHT_QUANT_A2APATH_CRITICAL_K) {
            kernelType = KernelType::ALL_REDUCE;
        } else {
            kernelType = KernelType::ALL_REDUCE_VIA_TWO_SHOT;
        }
        MMAllReduceFitBalanceTiling allReduceTilingHcclInst(args_, kernelType, TopoType::EIGHT_P);
        mCutAllreduceOutput = allReduceTilingHcclInst.GetTiling();
    } else {
        MMPlusAllReduce allReduceTilingHcclInst(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVer, isPerBlock_);
        allReduceTilingHcclInst.GetTiling();
        mCutAllreduceOutput = allReduceTilingHcclInst.tilingM_.cutRes;
    }
    return mCutAllreduceOutput;
}

// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAllReduce, WeightQuantMatmulAllReduceTilingA5,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
