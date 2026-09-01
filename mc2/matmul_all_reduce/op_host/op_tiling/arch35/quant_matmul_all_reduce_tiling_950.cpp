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
 * \file quant_matmul_all_reduce_tiling_950.cc
 * \brief
 */
#ifndef _QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
#define _QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
#include "quant_matmul_all_reduce_tiling_950.h"
#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "util/math_util.h"
#include "mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_apt_tiling_key.h"
#include "all_reduce_fit_balance_tiling.h"
#include "mc2_comm_utils.h"

using namespace Mc2Log;
using namespace Mc2Tiling;
namespace optiling {
constexpr uint64_t HCOMM_CNT = 2;
constexpr uint64_t INT8_WORKSPACE_CNT = 3;
constexpr uint64_t PERTILE_FP8_WORKSPACE_CNT = 3;
constexpr uint64_t PERTILE_FP32_WORKSPACE_CNT = 2;
constexpr uint64_t STANDARD_CARD_CGMPAD_WORKSPACE_CNT = 3;
constexpr uint64_t FIVE_ONE_TWO = 512;
constexpr uint64_t GROUP_M_OFFSET = 32;
constexpr uint64_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr uint64_t PERTILE_TILELEN = 128;
constexpr uint64_t DOUBLE_BUFFER = 2;
constexpr uint64_t QUANT_MODE_FP8 = 2;
constexpr uint32_t ALIGN_DATA_SIZE = 32;
constexpr uint64_t CCU_ALLTOALL_MAX_DATACNT = 200 * 1024 * 1024;
constexpr uint64_t QUANT_A2APATH_CRITICAL_K = 4608;

namespace {
const gert::Shape defaultShape = gert::Shape();
gert::StorageShape defaultStorageShape = gert::StorageShape();
} // namespace

bool QuantMatmulAllReduceTilingA5::IsPerTensorDequantScale(const gert::Shape &dequantShape) const
{
    return dequantShape.GetDimNum() > 0 && dequantShape.GetDim(dequantShape.GetDimNum() - 1) == 1;
}

bool QuantMatmulAllReduceTilingA5::IsCapable()
{
    if (isA8W8_ || (scenario_ == AllReduceScenario::FP8HIF8) || (scenario_ == AllReduceScenario::MXFP4) ||
        (scenario_ == AllReduceScenario::MXFP8)) {
        OP_LOGI(opName_, "Start with matmulAllReduce 950 quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip matmulAllReduce 950 quant tiling as dtype not support.");
    return false;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::SetMc2HcommAllReduce(const char *groupName, const uint32_t reduceType)
{
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));
    const std::string algConfig = "AllReduce=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType, dataType, dataType);
    uint8_t commMode = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commMode), OP_LOGE(opName_, "Get commMode failed."), return ge::GRAPH_FAILED);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2InitTiling),
                    OP_LOGE(opName_, "Get mc2InitTiling from quantMatmulAllReduceTilingData failed."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2CcTiling),
                    OP_LOGE(opName_, "Get mc2CcTiling from quantMatmulAllReduceTilingData failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::SetMc2HcommTwoShot(const char *groupName, const uint32_t reduceType,
                                                                 const uint8_t dataType)
{
    uint32_t opType1 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLTOALL);
    uint32_t opType2 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);

    const std::string algConfig1 = "AlltoAll=level0:fullmesh";
    const std::string algConfig2 = "AllGather=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType1, algConfig1, reduceType, dataType, dataType);
    uint8_t commMode = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commMode), OP_LOGE(opName_, "Get commMode failed."), return ge::GRAPH_FAILED);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(
        mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2InitTiling),
        OP_LOGE(opName_, "Get mc2InitTiling from quantMatmulAllReduceTilingData by SetMc2HcommTwoShot failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2CcTiling),
        OP_LOGE(opName_, "Get mc2CcTiling from quantMatmulAllReduceTilingData by SetMc2HcommTwoShot failed."),
        return ge::GRAPH_FAILED);
    mc2CcTilingConfig.SetGroupName(groupName);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfig2);
    mc2CcTilingConfig.SetReduceType(reduceType, dataType, dataType);
    OP_TILING_CHECK(
        mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2CcTilingComm),
        OP_LOGE(opName_, "Get mc2CcTilingComm from quantMatmulAllReduceTilingData by SetMc2HcommTwoShot failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::SetMc2HcommRSAG(const char *groupName, const uint32_t reduceType)
{
    uint32_t opType1 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    uint32_t opType2 = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    uint8_t dataType1 = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_INT8));
    uint8_t dataType2 = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_FLOAT));
    const std::string algConfig1 = "ReduceScatter=level0:fullmesh";
    const std::string algConfig2 = "AllGather=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType1, algConfig1, reduceType, dataType2, dataType1);
    uint8_t commMode = 0;
    OP_TILING_CHECK(GetAndConvertCommMode(commMode), OP_LOGE(opName_, "Get commMode failed."), return ge::GRAPH_FAILED);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        mc2CcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(
        mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2InitTiling),
        OP_LOGE(opName_, "Get mc2InitTiling from quantMatmulAllReduceTilingData by SetMc2HcommRSAG failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2CcTiling),
                    OP_LOGE(opName_, "Get mc2CcTiling from quantMatmulAllReduceTilingData by SetMc2HcommRSAG failed."),
                    return ge::GRAPH_FAILED);
    mc2CcTilingConfig.SetGroupName(groupName);
    mc2CcTilingConfig.SetOpType(opType2);
    mc2CcTilingConfig.SetAlgConfig(algConfig2);
    mc2CcTilingConfig.SetReduceType(reduceType, dataType1, dataType1);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(quantMatmulAllReduceTilingData_.mc2CcTilingComm),
                    OP_LOGE(opName_, "Get mc2CcTilingComm from quantMatmulAllReduceTilingData failed."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::SetMc2Hcomm()
{
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    uint8_t commMode = GetCommMode();
    if (commMode == Mc2Comm::COMM_MODE_CCU && k >= QUANT_A2APATH_CRITICAL_K &&
        mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        isUseA2APath = false;
    }
    OP_TILING_CHECK(
        mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType) == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "y", Ops::Base::ToString(args_.geCType).c_str(),
                                  "FLOAT16, BF16, FLOAT or INT8"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetAttrs() == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "comm_mode"),
                    return ge::GRAPH_FAILED);
    const char *groupName = context_->GetAttrs()->GetAttrPointer<char>(static_cast<int>(0));
    const uint32_t reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    if (isUseA2APath && !MutableRCSTilingData().isInputCommQuantScale) {
        uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geCType));
        OP_TILING_CHECK(SetMc2HcommTwoShot(groupName, reduceType, dataType) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName_, "set Mc2Hcomm config By SetMc2HcommTwoShot failed."), return ge::GRAPH_FAILED);
    } else {
        if (MutableRCSTilingData().isInputCommQuantScale == 1) {
            uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, ge::DataType::DT_INT8));
            OP_TILING_CHECK(SetMc2HcommTwoShot(groupName, reduceType, dataType) != ge::GRAPH_SUCCESS,
                            OP_LOGE(opName_, "set Mc2Hcomm config By SetMc2HcommTwoShot failed."),
                            return ge::GRAPH_FAILED);
        } else if (MutableRCSTilingData().isInputCommQuantScale == QUANT_MODE_FP8) {
            uint8_t dataType = static_cast<uint8_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType));
            OP_TILING_CHECK(SetMc2HcommTwoShot(groupName, reduceType, dataType) != ge::GRAPH_SUCCESS,
                            OP_LOGE(opName_, "set Mc2Hcomm config By SetMc2HcommTwoShot failed."),
                            return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK(SetMc2HcommAllReduce(groupName, reduceType) != ge::GRAPH_SUCCESS,
                            OP_LOGE(opName_, "set Mc2Hcomm config By SetMc2HcommAllReduce failed."),
                            return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA8W8());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    DoRCSTiling();
    MC2_CHECK_LOG_RET(opName_, CheckHCCLSize());
    DoSplitMTiling();

    MC2_CHECK_LOG_RET(opName_, AdjustHCCLLimit());
    MC2_CHECK_LOG_RET(opName_, DoQuantTiling());
    if (MutableRCSTilingData().isInputCommQuantScale == 1) {
        isCommInt8Enable_ = true;
    }
    MC2_CHECK_LOG_RET(opName_, SetMc2Hcomm());
    DoAllReduceTiling(true);
    if (MutableRCSTilingData().isInputCommQuantScale == QUANT_MODE_FP8) {
        isCommFp8Enable_ = true;
        MC2_CHECK_LOG_RET(opName_, GetDynamicQuantTempBuffSize());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckHCCLSize()
{
    auto &&param = MutableRCSTilingData();
    if (param.isInputCommQuantScale != QUANT_MODE_FP8) {
        return ge::GRAPH_SUCCESS;
    }
    // 如果1行数据就超通信数据量限制，那么任何M切分方式都无法满足
    uint64_t sizeOfSingleM = static_cast<uint64_t>(param.rankN) * args_.rankDim;
    OP_TILING_CHECK(
        sizeOfSingleM > mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "matmul output size", std::to_string(sizeOfSingleM).c_str(),
                                              "The value of matmul output size must not exceed 256MB"),
        return ge::GRAPH_FAILED);
    // 如果按通信最大次数切分，能够满足通信数据量限制，那么继续做tiling
    uint64_t sizeOfSplitM =
        Ops::Base::CeilDiv(static_cast<uint64_t>(param.rankM), mc2tiling::ALL_GATHER_HCCL_NUM_LIMIT) * sizeOfSingleM;
    OP_TILING_CHECK(sizeOfSplitM > mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1 size", std::to_string(sizeOfSplitM).c_str(),
                                                          "The value of x1 size must not exceed 256MB"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::AdjustHCCLLimit()
{
    auto &&param = MutableRCSTilingData();
    // 当前仅校验pertile量化场景，pertile量化是低bit通信，通信的数据类型仅占1字节
    if (param.isInputCommQuantScale != QUANT_MODE_FP8) {
        return ge::GRAPH_SUCCESS;
    }
    // matmulAllReduce在pertile量化场景下，通信分为alltoall和allgather两个阶段
    // alltoall和allgather单次集合通信均不能超过256M,而allgather单次集合通信数据量和rankDim成正比
    uint64_t singleTileDataSize = tileMValue_ * static_cast<uint64_t>(param.rankN) * args_.rankDim;
    if (singleTileDataSize <= mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGI(opName_,
            "The formulaic tiling result does not meet the hccl restriction,"
            " current splitting: tileM [%lu], tileCnt [%u], tailM [%lu], tailCnt [%u].",
            tileMValue_, param.tileCnt, tailMValue_, param.tailCnt);
    // 如果公式化tiling的结果不满足通信量限制，那么重新切分，按单次最大通信量限制进行切分
    uint64_t totalDataSize = static_cast<uint64_t>(param.rankM) * static_cast<uint64_t>(param.rankN) * args_.rankDim;
    uint64_t minSplitPart = Ops::Base::CeilDiv(totalDataSize, mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT);
    tileMValue_ = Ops::Base::CeilDiv(static_cast<uint64_t>(param.rankM), minSplitPart);
    param.tileCnt = static_cast<uint32_t>(Ops::Base::FloorDiv(static_cast<uint64_t>(param.rankM), tileMValue_));
    param.tailM =
        static_cast<uint32_t>(static_cast<uint64_t>(param.rankM) - static_cast<uint64_t>(param.tileCnt) * tileMValue_);
    tailMValue_ = param.tailM;
    param.tailCnt = (tailMValue_ == 0) ? 0 : 1;
    OP_LOGI(opName_,
            "Because the formulaic tiling result does not meet the hccl restriction,"
            " the re-splitM result: tileM [%lu], tileCnt [%u], tailM [%lu], tailCnt [%u]. end re-splitM.",
            tileMValue_, param.tileCnt, tailMValue_, param.tailCnt);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::GetDynamicQuantTempBuffSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    OP_TILING_CHECK(ascendcPlatform.GetCoreNumAiv() == 0, OP_LOGE(opName_, "GetCoreNumAiv Failed! Invalid AivNum."),
                    return ge::GRAPH_FAILED);
    auto aivNum = ascendcPlatform.GetCoreNumAiv() > 0 ? ascendcPlatform.GetCoreNumAiv() : 1;
    int64_t procRowsRaw = MutableTCubeTileTilingData().M < aivNum ? 1 : MutableTCubeTileTilingData().M / aivNum;
    uint64_t ubSize = static_cast<uint64_t>(aicoreParams_.ubSize);
    uint64_t fp8Size = 1;
    uint64_t outSize = args_.geCType == ge::DataType::DT_FLOAT ? sizeof(float) : 2;
    uint64_t ubDenomQuant =
        3 * PERTILE_TILELEN * sizeof(float) + PERTILE_TILELEN * fp8Size + sizeof(float) + sizeof(uint8_t);
    ubDenomQuant *= DOUBLE_BUFFER;
    int64_t procRows = ((ubSize / ubDenomQuant) < procRowsRaw) ? (ubSize / ubDenomQuant) : procRowsRaw;
    int64_t procRowTileCnt = (ubSize / ubDenomQuant) / procRows;
    OP_LOGD(opName_, "Quant: ubSize=%ld, ubDenomQuant=%ld, procRowsRaw=%ld, procRows=%ld, procRowTileCnt=%ld.", ubSize,
            ubDenomQuant, procRowsRaw, procRows, procRowTileCnt);
    std::vector<int64_t> srcShapeVec = {procRowTileCnt, 1};
    std::vector<int64_t> dstShapeVec = {procRowTileCnt, PERTILE_TILELEN};
    ge::Shape srcShape(srcShapeVec);
    ge::Shape dstShape(dstShapeVec);
    uint32_t maxValBroadCast{0};
    uint32_t minValBroadCast{0};
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcShape, dstShape, sizeof(float), false, maxValBroadCast,
                                       minValBroadCast); // Quant 广播Scale计算量化结果时需要的额外空间
    uint32_t minValBroadCastDequant{0};
    uint64_t ubDenomDequant =
        PERTILE_TILELEN * fp8Size + sizeof(float) + PERTILE_TILELEN * outSize + 2 * PERTILE_TILELEN * sizeof(float);
    ubDenomDequant *= DOUBLE_BUFFER;
    procRows = ((ubSize / ubDenomDequant) < procRowsRaw) ? (ubSize / ubDenomDequant) : procRowsRaw;
    procRowTileCnt = (ubSize / ubDenomDequant) / procRows;
    OP_LOGD(opName_, "Dequant: ubSize=%ld, ubDenomDequant=%ld, procRowsRaw=%ld, procRows=%ld, procRowTileCnt=%ld.",
            ubSize, ubDenomDequant, procRowsRaw, procRows, procRowTileCnt);
    std::vector<int64_t> srcShapeDequantVec = {procRowTileCnt, 1};
    std::vector<int64_t> dstShapeDequantVec = {procRowTileCnt, PERTILE_TILELEN};
    ge::Shape srcShapeDequant(srcShapeDequantVec);
    ge::Shape dstShapeDequant(dstShapeDequantVec);
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcShapeDequant, dstShapeDequant, sizeof(float), false,
                                       maxValBroadCast, minValBroadCastDequant); // Dequant 广播Scale计算反量化
    uint32_t tempBuffSize = std::max(minValBroadCast, minValBroadCastDequant);
    tempBuffSize = Ops::Base::CeilDiv(tempBuffSize, ALIGN_DATA_SIZE) * ALIGN_DATA_SIZE;
    MutableRCSTilingData().dynamicQuantTempBuffSize = tempBuffSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t QuantMatmulAllReduceTilingA5::GetTilingKey() const
{
    uint8_t commDtype = COMMDTPYE_DEFAULT;
    if (isCommInt8Enable_ == true) {
        commDtype = COMMDTPYE_INT8; // 适配int8 通信;
    } else if (isCommFp8Enable_ == true) {
        commDtype = COMMDTPYE_FP8; // 适配fp8 通信;
    }
    bool scenarioIsMXFP8 = (scenario_ == AllReduceScenario::MXFP8); // 区分MXFP8 和 FP8HIF8场景
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    uint8_t commMode = GetCommMode();
    if (commMode == Mc2Comm::COMM_MODE_CCU && k >= QUANT_A2APATH_CRITICAL_K &&
        mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        isUseA2APath = false;
    }
    bool isA2ARSAG = (isUseA2APath && (commDtype == COMMDTPYE_DEFAULT));
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(MMTYPE_QUANT_MM, quantTPlparam_.transB, false, isA2ARSAG, commMode, SET_NOT_USE_FP_MM_TILING,
                           quantTPlparam_.kernelType, commDtype, scenarioIsMXFP8, SET_NOT_USE_WEIGHT_QUANT_MM_TILING);
    OP_LOGD(opName_,
            "TransB, kernelType,"
            "commDtype, scenarioIsMXFP8 is:[%d,%u,%u,%d].",
            quantTPlparam_.transB, quantTPlparam_.kernelType, commDtype, scenarioIsMXFP8);
    OP_LOGD(opName_, "Mc2MatmulAllReduce: quant_TilingKey=%lu.", tilingKey);
    return tilingKey;
}

void QuantMatmulAllReduceTilingA5::PrintExtendMatmulTiling(bool isTail)
{
    auto &tiling = quantMatmulAllReduceTilingData_.tilematmulTiling;
    if (isTail) {
        tiling = quantMatmulAllReduceTilingData_.tailmatmulTiling;
    }

    OP_LOGD(opName_, "QuantBmmV3Params.batchA=%u.", tiling.params.batchA);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB=%u.", tiling.params.batchB);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC=%u.", tiling.params.batchC);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA1=%u.", tiling.params.batchA1);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA2=%u.", tiling.params.batchA2);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA3=%u.", tiling.params.batchA3);
    OP_LOGD(opName_, "QuantBmmV3Params.batchA4=%u.", tiling.params.batchA4);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB1=%u.", tiling.params.batchB1);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB2=%u.", tiling.params.batchB2);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB3=%u.", tiling.params.batchB3);
    OP_LOGD(opName_, "QuantBmmV3Params.batchB4=%u.", tiling.params.batchB4);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC1=%u.", tiling.params.batchC1);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC2=%u.", tiling.params.batchC2);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC3=%u.", tiling.params.batchC3);
    OP_LOGD(opName_, "QuantBmmV3Params.batchC4=%u.", tiling.params.batchC4);
    OP_LOGD(opName_, "QuantBmmV3Params.singleCoreBatch=%u.", tiling.params.singleCoreBatch);
    OP_LOGD(opName_, "QuantBmmV3Params.isPerTensor=%u.", tiling.params.isPerTensor);
    OP_LOGD(opName_, "QuantBmmV3Params.isPertoken=%u.", tiling.params.isPertoken);
    OP_LOGD(opName_, "QuantBmmV3Params.isDoubleScale=%u.", tiling.params.isDoubleScale);
    OP_LOGD(opName_, "QuantBmmV3Params.biasThreeDim=%u.", tiling.params.biasThreeDim);
    OP_LOGD(opName_, "QuantBmmV3Params.ubCalcM=%u.", tiling.params.ubCalcM);
    OP_LOGD(opName_, "QuantBmmV3Params.ubCalcN=%u.", tiling.params.ubCalcN);
    OP_LOGD(opName_, "QuantBmmV3Params.needUbBuffer=%u.", tiling.params.needUbBuffer);
    OP_LOGD(opName_, "QuantBmmV3Params.realSingleCoreM=%u.", tiling.params.realSingleCoreM);
    OP_LOGD(opName_, "QuantBmmV3Params.realSingleCoreN=%u.", tiling.params.realSingleCoreN);
    OP_LOGD(opName_, "QuantBmmV3Params.biasDtype=%u.", tiling.params.biasDtype);
    OP_LOGD(opName_, "QuantBmmV3Params.ubSize=%u.", tiling.params.ubSize);
    OP_LOGD(opName_, "QuantBmmV3Params.isMClash=%u.", tiling.params.isMClash);
    OP_LOGD(opName_, "QuantBmmV3Params.isNClash=%u.", tiling.params.isNClash);
    OP_LOGD(opName_, "QuantBmmV3Params.groupSizeM=%u.", tiling.params.groupSizeM);
    OP_LOGD(opName_, "QuantBmmV3Params.groupSizeK=%u.", tiling.params.groupSizeK);
    OP_LOGD(opName_, "QuantBmmV3Params.groupSizeN=%u.", tiling.params.groupSizeN);

    OP_LOGD(opName_, "TileL2cacheTiling.mTileCntL2=%u.", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName_, "TileL2cacheTiling.nTileCntL2=%u.", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName_, "TileL2cacheTiling.mTileBlock=%u.", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName_, "TileL2cacheTiling.nTileBlock=%u.", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName_, "TileL2cacheTiling.calOrder=%u.", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName_, "TileL2cacheTiling.isBasicTiling=%u.", tiling.tileL2cacheTiling.isBasicTiling);

    OP_LOGD(opName_, "AdaptiveSlidingWin.mTailTile=%u.", tiling.adaptiveSlidingWin.mTailTile);
    OP_LOGD(opName_, "AdaptiveSlidingWin.nTailTile=%u.", tiling.adaptiveSlidingWin.nTailTile);
}

ge::graphStatus QuantMatmulAllReduceTilingA5::GetWorkspaceSizeForA2ARSAG(const uint64_t gmcFloat)
{
    uint64_t commLen = 0UL;
    uint64_t cgmPadLen = 0UL;
    uint64_t commWorkSpace = 0UL;

    uint64_t rankM = static_cast<uint64_t>(MutableRCSTilingData().rankM);
    uint64_t rankN = static_cast<uint64_t>(MutableRCSTilingData().rankN);
    commLen = rankM * rankN;
    cgmPadLen = (MutableRCSTilingData().tileCnt + MutableRCSTilingData().tailCnt) * FIVE_ONE_TWO;
    commWorkSpace = (commLen + cgmPadLen) * static_cast<uint64_t>(args_.outputDtypeSize);
    OP_LOGI(opName_, "Set commWorkSpace size=%lu to context.", commWorkSpace);

    myWorkSpaceSize_ = myWorkSpaceSize_ + gmcFloat;
    uint64_t workspaceSizeCount = STANDARD_CARD_CGMPAD_WORKSPACE_CNT;
    myWorkSpaceSize_ = myWorkSpaceSize_ + commWorkSpace * workspaceSizeCount + commWorkSpace / args_.rankDim;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::GetWorkspaceSizeOfCommQuantScaleOrFP8(const uint64_t gmcFloat)
{
    uint64_t commInt8WorkSpace = 0UL;
    uint64_t commFp32WorkSpace = 0UL;
    bool isFp8 = MutableRCSTilingData().isInputCommQuantScale == QUANT_MODE_FP8;
    if (MutableRCSTilingData().isInputCommQuantScale == 1 || isFp8) {
        uint64_t padTileM = MutableTCubeTileTilingData().M;
        uint64_t padTailM = MutableTCubeTailTilingData().M;
        if (padTileM % args_.rankDim != 0) {
            padTileM += args_.rankDim - (padTileM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTileM = padTileM * MutableTCubeTileTilingData().N * sizeof(int8_t);
        if (padTailM % args_.rankDim != 0) {
            padTailM += args_.rankDim - (padTailM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTailM = padTailM * MutableTCubeTailTilingData().N * sizeof(int8_t);
        commFp32WorkSpace =
            (tempPadTileM * MutableRCSTilingData().tileCnt + tempPadTailM * MutableRCSTilingData().tailCnt) *
            sizeof(float);
        if (isFp8) {
            uint64_t tileN = MutableTCubeTileTilingData().N;
            uint64_t tailN = MutableTCubeTailTilingData().N;
            tileN += Ops::Base::CeilDiv(tileN, PERTILE_TILELEN);
            tailN += Ops::Base::CeilDiv(tailN, PERTILE_TILELEN);
            tempPadTileM = padTileM * tileN;
            tempPadTailM = padTailM * tailN;
        }
        commInt8WorkSpace =
            tempPadTileM * MutableRCSTilingData().tileCnt + tempPadTailM * MutableRCSTilingData().tailCnt;
        commInt8WorkSpace *= isFp8 ? sizeof(float) : sizeof(int8_t);
        OP_LOGI(opName_, "Set commInt8WorkSpace size=%lu, commFp32WorkSpace size=%lu to context.", commInt8WorkSpace,
                commFp32WorkSpace);
        MutableRCSTilingData().commInt8WorkSpace =
            (commInt8WorkSpace); // int8 通信用于存放reduceScatter输入 workspace 的开销
    }
    if (isFp8) {
        // 存放Matmul输出+quant输出+alltoall输出+(dequant+reduce+quant)混合输出+allgather输出+dequant输出
        myWorkSpaceSize_ = myWorkSpaceSize_ + PERTILE_FP8_WORKSPACE_CNT * commInt8WorkSpace +
                           commInt8WorkSpace / args_.rankDim + PERTILE_FP32_WORKSPACE_CNT * commFp32WorkSpace;
    } else {
        myWorkSpaceSize_ = myWorkSpaceSize_ + gmcFloat + INT8_WORKSPACE_CNT * commInt8WorkSpace;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    uint64_t commWorkSpace = myWorkSpaceSize_ - libApiWorkSpaceSize_;
    MutableRCSTilingData().commWorkSpaceSize = (commWorkSpace); // myWorkSpaceSize_去除系统空间后剩余大小
    uint64_t gmcFloat = static_cast<uint64_t>(MutableRCSTilingData().rankM) *
                        static_cast<uint64_t>(MutableRCSTilingData().rankN) *
                        static_cast<uint64_t>(args_.outputDtypeSize);
    bool isUseA2APath = mc2tiling::IsUseA2APath(args_.rankDim, npuArch_);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    uint8_t commMode = GetCommMode();
    if (commMode == Mc2Comm::COMM_MODE_CCU && k >= QUANT_A2APATH_CRITICAL_K &&
        mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        isUseA2APath = false;
    }
    if (isUseA2APath && !MutableRCSTilingData().isInputCommQuantScale) {
        OP_TILING_CHECK(GetWorkspaceSizeForA2ARSAG(gmcFloat) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName_, "get workspace size By GetWorkspaceSizeForA2ARSAG failed."),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(GetWorkspaceSizeOfCommQuantScaleOrFP8(gmcFloat) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName_, "get workspace size By GetWorkspaceSizeOfCommQuantScaleOrFP8 failed."),
                        return ge::GRAPH_FAILED);
    }
    OP_LOGI(opName_, "Set max workspace size=%lu to context.", myWorkSpaceSize_);
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::PostTiling()
{
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.",
            sizeof(QuantMatmulAllReduceTilingDataA5), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(QuantMatmulAllReduceTilingDataA5));

    OP_TILING_CHECK(
        (sizeof(QuantMatmulAllReduceTilingDataA5) % sizeof(uint64_t)) != 0,
        OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", sizeof(QuantMatmulAllReduceTilingDataA5)),
        return ge::GRAPH_FAILED);

    errno_t ret =
        memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                 reinterpret_cast<void *>(&quantMatmulAllReduceTilingData_), sizeof(QuantMatmulAllReduceTilingDataA5));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Memcpy_s failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }

    PrintTilingData();
    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

Mc2Tiling::RCSTiling &QuantMatmulAllReduceTilingA5::MutableRCSTilingData()
{
    return quantMatmulAllReduceTilingData_.param;
}
::TCubeTiling &QuantMatmulAllReduceTilingA5::MutableTCubeTileTilingData()
{
    return quantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}
::TCubeTiling &QuantMatmulAllReduceTilingA5::MutableTCubeTailTilingData()
{
    return quantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::DoQuantTiling()
{
    args_.mValue = tileMValue_;
    QuantTilingTransferHelperA5 mmTile(*this, quantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OP_LOGD(opName_, "Enable SplitK Tiling.");
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
        quantTPlparam_ = mmTile.GetQuantMMAllReduceTPLParam(mmTile.GetKernelType());
        OP_LOGD(opName_, "QuantmmAllReduce get kernelType: %d.", mmTile.GetKernelType());
        return ge::GRAPH_SUCCESS;
    } else {
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
        ;
        if (MutableRCSTilingData().tailCnt == 0) {
            quantTPlparam_ = mmTile.GetQuantMMAllReduceTPLParam(mmTile.GetKernelType());
            OP_LOGD(opName_, "QuantmmAllReduce get kernelType: %d.", mmTile.GetKernelType());
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        QuantTilingTransferHelperA5 mmTail(*this, quantMatmulAllReduceTilingData_.tailmatmulTiling);
        MC2_CHECK_LOG_RET(opName_, mmTail.DoTiling());

        quantTPlparam_ = mmTail.GetQuantMMAllReduceTPLParam(mmTail.GetKernelType());
        OP_LOGD(opName_, "QuantmmAllReduce get kernelType: %d.", mmTail.GetKernelType());
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckAxisSize()
{
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
    // A2有ND2NZ指令的长度约束65535，为了兼容A2，x2最后一维限制65535
    uint64_t x2FirstDim = mmrCtxInfo_.x2_shape->GetStorageShape().GetDim(0);
    uint64_t x2LastDim = mmrCtxInfo_.x2_shape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((x2FirstDim > static_cast<uint64_t>(INT32_MAX)) || (x2LastDim > static_cast<uint64_t>(UINT16_MAX)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "x2",
                        ("[first=" + std::to_string(x2FirstDim) + ", last=" + std::to_string(x2LastDim) + "]").c_str(),
                        "The shape of x2 must not exceed upper limit"),
                    return ge::GRAPH_FAILED);

    return CheckQuantEmptyTensor();
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckA8W8ScenarioScaleType()
{
    if (scenario_ == AllReduceScenario::A8W8) {
        auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
        auto yType = mmrCtxInfo_.y->GetDataType();
        auto pertokenScaleShape = mmrCtxInfo_.pertoken_scale_shape;
        OP_LOGD(opName_, "DequantScaleType=%d, yType=%d.", dequantScaleType, yType);
        // 1. y = bf16 时，dequantScale = bf16
        // 2. y = fp16 且 protoken 不存在时， dequantScale = int64、uint64
        // 3. y = fp16 且 protoken 存在时，dequantScale = fp32
        if (yType == ge::DT_BF16) {
            OP_TILING_CHECK(
                dequantScaleType != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                      ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str(),
                                                      "The value of dequantScale dtype must be bf16"),
                return ge::GRAPH_FAILED);
        } else if (pertokenScaleShape == nullptr) {
            OP_TILING_CHECK(
                !((dequantScaleType == ge::DT_UINT64) || (dequantScaleType == ge::DT_INT64)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                      ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str(),
                                                      "The value of dequantScale dtype must be uint64 or int64"),
                return ge::GRAPH_FAILED);
        } else {
            OP_TILING_CHECK(
                dequantScaleType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                      ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str(),
                                                      "The value of dequantScale dtype must be bf16"),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckDequantScaleType()
{
    // dequantScale数据类型范围
    auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
    OP_TILING_CHECK(
        !((dequantScaleType == ge::DT_UINT64) || (dequantScaleType == ge::DT_BF16) ||
          (dequantScaleType == ge::DT_INT64) || (dequantScaleType == ge::DT_FLOAT) ||
          (dequantScaleType == ge::DT_FLOAT8_E8M0)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                              ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str(),
                                              "The dtype of dequantScale must be uint64, int64, bf16 or float32"),
        return ge::GRAPH_FAILED);
    if (scenario_ == AllReduceScenario::FP8HIF8) {
        if (dequantScaleType == ge::DT_FLOAT) {
            // fp8/hif8场景下，pertoken+pertensor/perchannel场景必须提供两个scale，类型为float
            OP_TILING_CHECK(mmrCtxInfo_.pertoken_scale == nullptr,
                            OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "per_token_scale"),
                            return ge::GRAPH_FAILED);
            auto perTokenScaleType = mmrCtxInfo_.pertoken_scale->GetDataType();
            OP_TILING_CHECK(
                perTokenScaleType != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "per_token_scale",
                                                      ge::TypeUtils::DataTypeToSerialString(perTokenScaleType).c_str(),
                                                      "The dtype of per_token_scale must be float"),
                return ge::GRAPH_FAILED);
        } else {
            // fp8/hif8场景下，单路pertensor场景必须提供1个scale，类型为uint64
            OP_TILING_CHECK(
                dequantScaleType != ge::DT_UINT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                      ge::TypeUtils::DataTypeToSerialString(dequantScaleType).c_str(),
                                                      "The dtype of dequantScale must be uint64"),
                return ge::GRAPH_FAILED);
        }
    }

    OP_TILING_CHECK(CheckMXFPScenarioScaleType() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Check scale type failed."), return ge::GRAPH_FAILED);
    return CheckA8W8ScenarioScaleType();
} // namespace optiling

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckMXFPScenarioScaleType()
{
    auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
    if ((scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8)) {
        OP_TILING_CHECK(mmrCtxInfo_.pertoken_scale == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "per_token_scale"),
                        return ge::GRAPH_FAILED);
        auto perTokenScaleType = mmrCtxInfo_.pertoken_scale->GetDataType();
        OP_TILING_CHECK((dequantScaleType != ge::DT_FLOAT8_E8M0) || (perTokenScaleType != ge::DT_FLOAT8_E8M0),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), "dequantScale/pertokenScale",
                            (std::string(ge::TypeUtils::DataTypeToSerialString(dequantScaleType)) + "/" +
                             std::string(ge::TypeUtils::DataTypeToSerialString(perTokenScaleType)))
                                .c_str(),
                            "The dtype of dequantScale and pertokenScale must be float8_e8m0"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckBias()
{
    // bias数据类型为int32, fp8/hif8,MX场景下为float
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        if (isA8W8_) {
            OP_TILING_CHECK(biasType != ge::DT_INT32,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                context_->GetNodeName(), "bias", std::to_string(static_cast<int32_t>(biasType)).c_str(),
                                "The dtype of bias must be int32"),
                            return ge::GRAPH_FAILED);
        } else if ((scenario_ == AllReduceScenario::FP8HIF8) || (scenario_ == AllReduceScenario::MXFP4) ||
                   (scenario_ == AllReduceScenario::MXFP8)) {
            OP_TILING_CHECK(biasType != ge::DT_FLOAT,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                context_->GetNodeName(), "bias", std::to_string(static_cast<int32_t>(biasType)).c_str(),
                                "The dtype of bias must be float"),
                            return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckCommQuantScale()
{
    // comm_quant_scale不为空时校验数据类型
    if ((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) && (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)) {
        auto commQuantScaleType1 = mmrCtxInfo_.comm_quant_scale_1->GetDataType();
        auto commQuantScaleType2 = mmrCtxInfo_.comm_quant_scale_2->GetDataType();
        auto cType = mmrCtxInfo_.y->GetDataType();
        OP_TILING_CHECK(
            ((commQuantScaleType1 != cType) || (commQuantScaleType2 != cType)),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "comm_quant_scale",
                                                  (std::to_string(static_cast<int32_t>(commQuantScaleType1)) + "/" +
                                                   std::to_string(static_cast<int32_t>(commQuantScaleType2)))
                                                      .c_str(),
                                                  "The dtype of comm_quant_scale must be the same as that of output"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckQuantGroupSize()
{
    OP_TILING_CHECK(mmrCtxInfo_.groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "groupSize"),
                    return false);
    auto groupSizePtr = mmrCtxInfo_.groupSizePtr;
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    mc2tiling::Mc2MatmulShapeInfo shapeInfo = {mmrCtxInfo_.x1_shape,
                                               mmrCtxInfo_.x2_shape,
                                               mmrCtxInfo_.pertoken_scale_shape,
                                               mmrCtxInfo_.dequant_scale_shape,
                                               false,
                                               args_.isBTrans,
                                               opName_};

    if (isPerBlock_) {
        OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                        OP_LOGE(opName_, "Failed to execute inferGroupSize."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupSizeM != SUPPORTED_BLOCK_SIZE) || (groupSizeN != SUPPORTED_BLOCK_SIZE) ||
                            (groupSizeK != SUPPORTED_BLOCK_SIZE),
                        OP_LOGE_WITH_INVALID_ATTR(opName_, "groupSize",
                                                  (std::string("[groupSizeM=") + std::to_string(groupSizeM) +
                                                   ", groupSizeN=" + std::to_string(groupSizeN) +
                                                   ", groupSizeK=" + std::to_string(groupSizeK) + "]")
                                                      .c_str(),
                                                  "[groupSizeM=128, groupSizeN=128, groupSizeK=128]"),
                        return ge::GRAPH_FAILED);
    } else if ((scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8)) {
        shapeInfo.isMxfp = true;
        OP_TILING_CHECK(!mc2tiling::Mc2TilingUtils::InferGroupSize(shapeInfo, groupSizeM, groupSizeN, groupSizeK),
                        OP_LOGE(opName_, "Failed to execute inferGroupSize."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (groupSizeM != MX_GROUP_SIZE_M) || (groupSizeN != MX_GROUP_SIZE_N) || (groupSizeK != MX_GROUP_SIZE_K),
            OP_LOGE_WITH_INVALID_ATTR(opName_, "groupSize",
                                      (std::string("[groupSizeM=") + std::to_string(groupSizeM) + ", groupSizeN=" +
                                       std::to_string(groupSizeN) + ", groupSizeK=" + std::to_string(groupSizeK) + "]")
                                          .c_str(),
                                      "[groupSizeM=1, groupSizeN=1, groupSizeK=32]"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckX1X2()
{
    // x2 shape 为 2 维
    size_t x2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(x2DimNum != DIM_NUM_TWO && x2DimNum != DIM_NUM_FOUR,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x2",
                                                             (std::to_string(x2DimNum) + "D").c_str(),
                                                             "The shape dim of x2 must be 2 or 4"),
                    return ge::GRAPH_FAILED);
    // x1，x2数据类型相同
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK((isA8W8_ && (x1Type != x2Type)),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x2",
                                                          std::to_string(static_cast<int32_t>(x2Type)).c_str(),
                                                          "The dtype of x2 must be the same as that of x1"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((scenario_ == AllReduceScenario::FP8HIF8) && (x1Type != x2Type) &&
                     ((x1Type == ge::DT_HIFLOAT8) || (x2Type == ge::DT_HIFLOAT8))),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "x2", std::to_string(static_cast<int32_t>(x2Type)).c_str(),
                        "If the dtype of x1 is HIFLOAT8, the dtype of x2 must be the same as that of x1"),
                    return ge::GRAPH_FAILED);
    if ((scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8)) {
        OP_TILING_CHECK(args_.isBTrans == false,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "x2", std::to_string(args_.isBTrans).c_str(),
                            "When the quantization mode is MXfp4 or MXfp8, x2 must be transposed"),
                        return ge::GRAPH_FAILED);
    }
    if (scenario_ == AllReduceScenario::MXFP4) {
        uint64_t x1K = GetKValue();
        OP_TILING_CHECK(
            Ops::Base::CeilDiv(x1K, MX_GROUP_SIZE_K) % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", ("k=" + std::to_string(x1K)).c_str(),
                                                  "The value of k of x1 ceildivided by 32 must be even"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTilingA5::CheckInput()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::CheckInput());
    OP_TILING_CHECK(CheckX1X2() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Check input_X failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckBias() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Check bias failed."),
                    return ge::GRAPH_FAILED);
    // dequantScale数据类型范围
    MC2_CHECK_LOG_RET(opName_, CheckDequantScaleType());
    // comm_quant_scale不为空时校验数据类型
    OP_TILING_CHECK(CheckCommQuantScale() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Check commQuantScale failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckQuantGroupSize() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Check groupSize failed."), return ge::GRAPH_FAILED);

    return CheckAxisSize();
}

QuantMatmulAllReduceTilingA5::QuantMatmulAllReduceTilingA5(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context),
      quantMatmulAllReduceTilingData_(quantMatmulAllReduceTilingDataSelf_)
{}

// 使用外部传入的tilingdata和ctxinfo
QuantMatmulAllReduceTilingA5::QuantMatmulAllReduceTilingA5(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                           QuantMatmulAllReduceTilingDataA5 *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo),
      quantMatmulAllReduceTilingData_(*out)
{}

const gert::Shape QuantTilingTransferHelperA5::GetX1Shape(const size_t index)
{
    (void)index;
    // 判断原始的m是否128对齐,对于perblock三维场景m轴非128对齐按照3维传值，和matmul保持一致
    bool isNotBatchOne = (tilingProcesser_.args_.batchValue != 1ULL);
    bool is128Aligned = ((tilingProcesser_.args_.orgMValue / tilingProcesser_.args_.batchValue) & 127ULL) == 0;
    // perblock场景下，若x1为（b,m,k)时，量化参数shape为[b,ceilDiv(m,128),ceilDiv(k,128)],仅当ceilDiv(m,128)为整数时，才能保证
    // b*ceilDiv(m,128) = ceilDiv(b*m,128)
    // mxfp场景下，若x1为（b,m,k)时，量化参数shape为[m,ceilDiv(k,64),2],若batch合轴，量化参数shape为[b*m,ceilDiv(k,64),2]，原shape无法满足计算
    if ((tilingProcesser_.isPerBlock_ && isNotBatchOne && !is128Aligned) ||
        ((tilingProcesser_.scenario_ == AllReduceScenario::MXFP8) && isNotBatchOne) ||
        ((tilingProcesser_.scenario_ == AllReduceScenario::MXFP4) && isNotBatchOne)) {
        return gert::Shape({static_cast<int64_t>(tilingProcesser_.args_.batchValue),
                            static_cast<int64_t>(tilingProcesser_.args_.mValue) /
                                static_cast<int64_t>(tilingProcesser_.args_.batchValue),
                            static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    } else {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.mValue), static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
}
const gert::Shape QuantTilingTransferHelperA5::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.args_.isBTrans) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.nValue), static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
    return gert::Shape(
        {static_cast<int64_t>(tilingProcesser_.args_.kValue), static_cast<int64_t>(tilingProcesser_.args_.nValue)});
}

const gert::Shape &QuantTilingTransferHelperA5::GetScaleShape(const size_t index)
{
    (void)index;
    OP_TILING_CHECK(tilingProcesser_.mmrCtxInfo_.dequant_scale_shape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "dequant_scale"), return defaultShape);
    return tilingProcesser_.mmrCtxInfo_.dequant_scale_shape->GetStorageShape();
}

const gert::StorageShape *QuantTilingTransferHelperA5::GetOffsetShape(const size_t index)
{
    (void)index;
    return nullptr;
}

const gert::StorageShape *QuantTilingTransferHelperA5::GetPertokenShape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.mmrCtxInfo_.pertoken_scale_shape == nullptr) {
        return nullptr;
    }

    if (tilingProcesser_.isPerBlock_) {
        return tilingProcesser_.mmrCtxInfo_.pertoken_scale_shape;
    }

    defaultStorageShape = gert::StorageShape({static_cast<int64_t>(tilingProcesser_.args_.mValue)},
                                             {static_cast<int64_t>(tilingProcesser_.args_.mValue)});
    return &defaultStorageShape;
}

const gert::StorageShape *QuantTilingTransferHelperA5::GetBiasShape(const size_t index)
{
    (void)index;
    return tilingProcesser_.mmrCtxInfo_.bias_shape;
}

ge::graphStatus QuantTilingTransferHelperA5::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling.");
    auto &&tilingArgs = tilingProcesser_.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transA = tilingArgs.isATrans;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    inputParams_.cDtype = tilingArgs.geCType;
    inputParams_.outDtype = static_cast<int64_t>(tilingArgs.geCType);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    inputParams_.scaleDtype = tilingProcesser_.mmrCtxInfo_.dequant_scale->GetDataType();
    if (tilingProcesser_.mmrCtxInfo_.pertoken_scale != nullptr) {
        inputParams_.perTokenScaleDtype = tilingProcesser_.mmrCtxInfo_.pertoken_scale->GetDataType();
    }

    if ((tilingProcesser_.scenario_ == AllReduceScenario::MXFP4) ||
        (tilingProcesser_.scenario_ == AllReduceScenario::MXFP8)) {
        inputParams_.groupSizeM = MX_GROUP_SIZE_M;
        inputParams_.groupSizeN = MX_GROUP_SIZE_N;
        inputParams_.groupSizeK = MX_GROUP_SIZE_K;
    }

    if (tilingProcesser_.isPerBlock_) {
        inputParams_.groupSizeM = SUPPORTED_BLOCK_SIZE;
        inputParams_.groupSizeN = SUPPORTED_BLOCK_SIZE;
        inputParams_.groupSizeK = SUPPORTED_BLOCK_SIZE;
    }
    // optiling::PlatformInfo::GetInstance().intrinsic_fix_pipe_l0c2out = tilingProcesser_.supportL0c2Out_;
    MC2_CHECK_TRUE_RET(tilingProcesser_.opName_, AnalyzeInputs());
    inputParams_.isPerTensor = (tilingProcesser_.quantType_ == Mc2QuantType::PER_TENSOR);
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

QuantMMAllReduceTPLParam QuantTilingTransferHelperA5::GetQuantMMAllReduceTPLParam(const uint64_t kernelType)
{
    QuantMMAllReduceTPLParam param;
    param.transB = inputParams_.transB;
    param.kernelType = kernelType;
    return param;
}

ge::graphStatus QuantTilingTransferHelperA5::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, "Set mm workspace size=%lu to mc2.", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

void QuantTilingTransferHelperA5::PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, "The transA_=%d, transB_=%d, hasBias_=%d.", quantBatchMatmulInfo.transA,
            quantBatchMatmulInfo.transB, quantBatchMatmulInfo.hasBias);
    OP_LOGD(tilingProcesser_.opName_, "The mSize_=%ld, kSize_=%ld, nSize_=%ld, libApiWorkSpaceSize=%u.",
            quantBatchMatmulInfo.mSize, quantBatchMatmulInfo.kSize, quantBatchMatmulInfo.nSize,
            quantBatchMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_, "The aDtype_=%d, bDtype_=%d, cDtype_=%d, biasDtype_=%d, outDtype=%ld.",
            static_cast<int32_t>(quantBatchMatmulInfo.aDtype), static_cast<int32_t>(quantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.cDtype), static_cast<int32_t>(quantBatchMatmulInfo.biasDtype),
            quantBatchMatmulInfo.outDtype);
    OP_LOGD(tilingProcesser_.opName_,
            "The batchA=%lu, batchA1-A4=[%lu:%lu:%lu:%lu], "
            "batchB=%lu, batchB1-B4=[%lu:%lu:%lu:%lu], batchC=%lu, batchBias=%lu.",
            quantBatchMatmulInfo.batchA, quantBatchMatmulInfo.batchA1, quantBatchMatmulInfo.batchA2,
            quantBatchMatmulInfo.batchA3, quantBatchMatmulInfo.batchA4, quantBatchMatmulInfo.batchB,
            quantBatchMatmulInfo.batchB1, quantBatchMatmulInfo.batchB2, quantBatchMatmulInfo.batchB3,
            quantBatchMatmulInfo.batchB4, quantBatchMatmulInfo.batchC, quantBatchMatmulInfo.batchBias);
    OP_LOGD(tilingProcesser_.opName_, "Check isperTensor=%d.", static_cast<int32_t>(quantBatchMatmulInfo.isPerTensor));
}

CutResult QuantMatmulAllReduceTilingA5::GetTilingResult()
{
    CutResult mCutAllreduce;
    SocVersion inputSocVersion = SocVersion::SOC910_B;
    SetMCutSocVersion(inputSocVersion);
    const gert::StorageShape *commQuantScaleShape1 = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleShape2 = mmrCtxInfo_.comm_quant_scale_2_shape;
    auto &&param = MutableRCSTilingData();
    uint8_t commMode = GetCommMode();
    bool isPertileFp8 = (param.isInputCommQuantScale == QUANT_MODE_FP8);
    bool isAlign = (isPerBlock_ || (scenario_ == AllReduceScenario::MXFP4) || (scenario_ == AllReduceScenario::MXFP8) ||
                    isPertileFp8);
    if (mc2tiling::IsStandardCard4P(args_.rankDim, npuArch_)) {
        MMAllReduceFitBalanceTiling allReduceTilingHccl(args_, KernelType::ALL_REDUCE_VIA_TWO_SHOT,
                                                        TopoType::STANDARD_CARD, SocVersion::SOC950, commMode,
                                                        isPertileFp8);
        allReduceTilingHccl.SetIsAlign(isAlign);
        mCutAllreduce = allReduceTilingHccl.GetTiling();
    } else if (mc2tiling::Is8P(args_.rankDim, npuArch_)) {
        const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
        KernelType kernelType;
        if (commMode == Mc2Comm::COMM_MODE_CCU && k >= QUANT_A2APATH_CRITICAL_K && !param.isInputCommQuantScale) {
            kernelType = KernelType::ALL_REDUCE;
        } else {
            kernelType = KernelType::ALL_REDUCE_VIA_TWO_SHOT;
        }
        MMAllReduceFitBalanceTiling allReduceTilingHccl(args_, kernelType, TopoType::EIGHT_P, SocVersion::SOC950,
                                                        commMode, isPertileFp8);
        allReduceTilingHccl.SetIsAlign(isAlign);
        mCutAllreduce = allReduceTilingHccl.GetTiling();
    } else {
        MMPlusAllReduce allReduceTilingHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVersion, isPerBlock_);
        allReduceTilingHccl.GetTiling();
        mCutAllreduce = allReduceTilingHccl.tilingM_.cutRes;
    }
    return mCutAllreduce;
}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAllReduce, QuantMatmulAllReduceTilingA5,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 0);
} // namespace optiling

#endif //_QUANT_MATMUL_ALL_REDUCE_TILING_950_CC_
