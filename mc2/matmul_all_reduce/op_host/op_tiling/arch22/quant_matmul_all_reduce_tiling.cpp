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
 * \file quant_matmul_all_reduce_tiling.cc
 * \brief
 */
#ifndef _QUANT_MATMUL_ALL_REDUCE_TILING_CC_
#define _QUANT_MATMUL_ALL_REDUCE_TILING_CC_
#include "quant_matmul_all_reduce_tiling.h"
#include "common/utils/op_mc2.h"

using namespace Mc2Tiling;
namespace optiling {
namespace {
const gert::Shape defaultShape = gert::Shape();
const gert::StorageShape defaultStorageShape = gert::StorageShape();
} // namespace
bool QuantMatmulAllReduceTiling::IsCapable()
{
    if (isA8W8_) {
        OP_LOGI(opName_, "start with quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip quant tiling as dtype not support");
    return false;
}
ge::graphStatus QuantMatmulAllReduceTiling::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA8W8());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    MC2_CHECK_LOG_RET(opName_, CheckCommModeA2());
    DoRCSTiling();
    DoSplitMTiling();
    MC2_CHECK_LOG_RET(opName_, DoQuantTiling());
    if (MutableRCSTilingData().isInputCommQuantScale == 1) {
        isCommInt8Enable_ = true;
    }
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}
uint64_t QuantMatmulAllReduceTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(ASCEND_910B),
                                            static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_QUANT_MATMUL),
                                            MATMUL_ALLREDUCE_EMPTY_INPUT_F, isCommInt8Enable_,
                                            0UL, // ENABLE_L2_CACHE
                                            0UL, // SHARE_MM
                                            SET_NOT_USE_FM_MM_TPL_TILING, quantMatmulTPLParam_.trans,
                                            quantMatmulTPLParam_.isPertoken, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
    OP_LOGI(opName_, " tilingKey %lu", tilingKey);
    return tilingKey;
}
ge::graphStatus QuantMatmulAllReduceTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    uint64_t int8WorkSpace = 0UL;
    if (MutableRCSTilingData().isInputCommQuantScale == 1) {
        uint64_t padTileM = MutableTCubeTileTilingData().M;
        uint64_t padTailM = MutableTCubeTailTilingData().M;
        if (padTileM % args_.rankDim != 0) {
            padTileM += args_.rankDim - (padTileM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTileM = padTileM * MutableTCubeTileTilingData().N * sizeof(uint8_t);
        if (padTailM % args_.rankDim != 0) {
            padTailM += args_.rankDim - (padTailM % args_.rankDim); // args_.rankDim :1/2/4/8 不会为0
        }
        uint64_t tempPadTailM = padTailM * MutableTCubeTailTilingData().N * sizeof(uint8_t);
        int8WorkSpace = tempPadTileM * MutableRCSTilingData().tileCnt + tempPadTailM * MutableRCSTilingData().tailCnt;
        OP_LOGI(opName_, " set int8WorkSpace size %lu to context", int8WorkSpace);
    }
    uint64_t commWorkSpace = myWorkSpaceSize_ - libApiWorkSpaceSize_;
    MutableRCSTilingData().commWorkSpaceSize = commWorkSpace; // myWorkSpaceSize_去除系统空间后剩余大小
    myWorkSpaceSize_ = myWorkSpaceSize_ + int8WorkSpace;
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QuantMatmulAllReduceTiling::PostTiling()
{
    size_t tilingDataSize = sizeof(QuantMatmulAllReduceTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ", tilingDataSize,
            context_->GetRawTilingData()->GetCapacity());
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    errno_t memcpyRet = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 reinterpret_cast<void *>(&quantMatmulAllReduceTilingData_), tilingDataSize);
    if (memcpyRet != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", memcpyRet);
        return ge::GRAPH_FAILED;
    }
    PrintTilingData();

    context_->SetBlockDim(args_.aicCoreNum);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batchScheduleMode = 1U;
    memcpyRet = context_->SetScheduleMode(batchScheduleMode);
    MC2_CHECK_LOG_RET(opName_, memcpyRet);

    return ge::GRAPH_SUCCESS;
}
Mc2Tiling::Mc2Msg &QuantMatmulAllReduceTiling::MutableMc2MsgData()
{
    return quantMatmulAllReduceTilingData_.msg;
}
Mc2Tiling::RCSTiling &QuantMatmulAllReduceTiling::MutableRCSTilingData()
{
    return quantMatmulAllReduceTilingData_.param;
}
AscendC::tiling::TCubeTiling &QuantMatmulAllReduceTiling::MutableTCubeTileTilingData()
{
    return quantMatmulAllReduceTilingData_.tilematmulTiling.matmulTiling;
}
AscendC::tiling::TCubeTiling &QuantMatmulAllReduceTiling::MutableTCubeTailTilingData()
{
    return quantMatmulAllReduceTilingData_.tailmatmulTiling.matmulTiling;
}
ge::graphStatus QuantMatmulAllReduceTiling::DoQuantTiling()
{
    args_.mValue = tileMValue_;
    QuantTilingTransferHelper mmTile(*this, quantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        OP_LOGD(opName_, "Enable SplitK Tiling.");
        auto res = mmTile.DoTiling();
        quantMatmulTPLParam_ = mmTile.GetQuantMatmulTPLParam();
        return res;
    } else {
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
        if (MutableRCSTilingData().tailCnt == 0) {
            quantMatmulTPLParam_ = mmTile.GetQuantMatmulTPLParam();
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        QuantTilingTransferHelper mmTail(*this, quantMatmulAllReduceTilingData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        quantMatmulTPLParam_ = mmTail.GetQuantMatmulTPLParam();
        return res;
    }
}
ge::graphStatus QuantMatmulAllReduceTiling::CheckAxisSize()
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
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    uint64_t x2FirstDim = args_.isBTrans ? n : k;
    uint64_t x2LastDim = args_.isBTrans ? k : n;
    OP_TILING_CHECK(
        x2FirstDim > static_cast<uint64_t>(INT32_MAX) || (x2LastDim > static_cast<uint64_t>(UINT16_MAX)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x2",
                                              (std::to_string(x2FirstDim) + "/" + std::to_string(x2LastDim)).c_str(),
                                              "The value of x2 dimensions exceeds the upper limit"),
        return ge::GRAPH_FAILED);

    return CheckQuantEmptyTensor();
}

ge::graphStatus QuantMatmulAllReduceTiling::CheckDequantScaleType()
{
    auto dequantScaleType = mmrCtxInfo_.dequant_scale->GetDataType();
    auto yType = mmrCtxInfo_.y->GetDataType();
    auto pertokenScaleShape = mmrCtxInfo_.pertoken_scale_shape;
    OP_LOGD(opName_, "dequantScaleType %d, yType %d.", dequantScaleType, yType);
    // 1. y = bf16 时，dequantScale = bf16
    // 2. y = fp16 且 protoken 不存在时， dequantScale = int64、uint64
    // 3. y = fp16 且 protoken 存在时，dequantScale = fp32
    if (yType == ge::DT_BF16) {
        OP_TILING_CHECK(
            dequantScaleType != ge::DT_BF16,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                  std::to_string(static_cast<int32_t>(dequantScaleType)).c_str(),
                                                  "The value of dequantScale dtype must be bf16"),
            return ge::GRAPH_FAILED);
    } else if (pertokenScaleShape == nullptr) {
        OP_TILING_CHECK(
            !(dequantScaleType == ge::DT_UINT64 || dequantScaleType == ge::DT_INT64),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                  std::to_string(static_cast<int32_t>(dequantScaleType)).c_str(),
                                                  "The value of dequantScale dtype must be uint64 or int64"),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            dequantScaleType != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dequantScale",
                                                  std::to_string(static_cast<int32_t>(dequantScaleType)).c_str(),
                                                  "The value of dequantScale dtype must be fp32"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantMatmulAllReduceTiling::CheckInput()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::CheckInput());
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
    OP_TILING_CHECK(x1Type != x2Type,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x2",
                                                          std::to_string(static_cast<int32_t>(x2Type)).c_str(),
                                                          "The value of x2 dtype must be the same as that of x1"),
                    return ge::GRAPH_FAILED);
    // bias数据类型为int32
    if (mmrCtxInfo_.bias_shape != nullptr) {
        auto biasType = mmrCtxInfo_.bias->GetDataType();
        OP_TILING_CHECK(biasType != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "bias",
                                                              std::to_string(static_cast<int32_t>(biasType)).c_str(),
                                                              "The value of bias dtype must be int32"),
                        return ge::GRAPH_FAILED);
    }
    // dequantScale数据类型范围
    MC2_CHECK_LOG_RET(opName_, CheckDequantScaleType());
    // comm_quant_scale不为空时校验数据类型
    if ((mmrCtxInfo_.comm_quant_scale_1_shape != nullptr) && (mmrCtxInfo_.comm_quant_scale_2_shape != nullptr)) {
        auto commQuantScaleType1 = mmrCtxInfo_.comm_quant_scale_1->GetDataType();
        auto commQuantScaleType2 = mmrCtxInfo_.comm_quant_scale_2->GetDataType();
        auto cType = mmrCtxInfo_.y->GetDataType();
        OP_TILING_CHECK((commQuantScaleType1 != cType || commQuantScaleType2 != cType),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "comm_quant_scale",
                            (std::to_string(static_cast<int32_t>(commQuantScaleType1)) + "/" +
                             std::to_string(static_cast<int32_t>(commQuantScaleType2)))
                                .c_str(),
                            "The value of comm_quant_scale dtype must be the same as that of output"),
                        return ge::GRAPH_FAILED);
    }

    return CheckAxisSize();
}
QuantMatmulAllReduceTiling::QuantMatmulAllReduceTiling(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context),
      quantMatmulAllReduceTilingData_(quantMatmulAllReduceTilingDataSelf_)
{}
const gert::Shape QuantTilingTransferHelper::GetX1Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.args_.isATrans) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.kValue), static_cast<int64_t>(tilingProcesser_.args_.mValue)});
    }
    return gert::Shape(
        {static_cast<int64_t>(tilingProcesser_.args_.mValue), static_cast<int64_t>(tilingProcesser_.args_.kValue)});
}
const gert::Shape QuantTilingTransferHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.args_.isBTrans) {
        return gert::Shape(
            {static_cast<int64_t>(tilingProcesser_.args_.nValue), static_cast<int64_t>(tilingProcesser_.args_.kValue)});
    }
    return gert::Shape(
        {static_cast<int64_t>(tilingProcesser_.args_.kValue), static_cast<int64_t>(tilingProcesser_.args_.nValue)});
}
// 使用外部传入的tilingdata和ctxinfo
QuantMatmulAllReduceTiling::QuantMatmulAllReduceTiling(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                       QuantMatmulAllReduceTilingData *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo),
      quantMatmulAllReduceTilingData_(*out)
{}
const gert::Shape &QuantTilingTransferHelper::GetScaleShape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.mmrCtxInfo_.dequant_scale_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "dequant_scale");
        return defaultShape;
    }
    return tilingProcesser_.mmrCtxInfo_.dequant_scale_shape->GetStorageShape();
}

const gert::StorageShape *QuantTilingTransferHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.mmrCtxInfo_.pertoken_scale_shape != nullptr) {
        return &defaultStorageShape;
    }
    return nullptr;
}

const gert::StorageShape *QuantTilingTransferHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return tilingProcesser_.mmrCtxInfo_.bias_shape;
}
ge::graphStatus QuantTilingTransferHelper::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
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
    // optiling::PlatformInfo::GetInstance().intrinsic_fix_pipe_l0c2out = tilingProcesser_.supportL0c2Out_;
    MC2_CHECK_TRUE_RET(tilingProcesser_.opName_, AnalyzeInputs());
    inputParams_.isPerTensor = (tilingProcesser_.quantType_ == Mc2QuantType::PER_TENSOR);
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantTilingTransferHelper::PostTiling()
{
    PrintTilingData();
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

void QuantTilingTransferHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, " transA_ %d transB_ %d, hasBias_ %d", quantBatchMatmulInfo.transA,
            quantBatchMatmulInfo.transB, quantBatchMatmulInfo.hasBias);
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u",
            quantBatchMatmulInfo.mSize, quantBatchMatmulInfo.kSize, quantBatchMatmulInfo.nSize,
            quantBatchMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_, "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld",
            static_cast<int32_t>(quantBatchMatmulInfo.aDtype), static_cast<int32_t>(quantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(quantBatchMatmulInfo.cDtype), static_cast<int32_t>(quantBatchMatmulInfo.biasDtype),
            quantBatchMatmulInfo.outDtype);
    OP_LOGD(tilingProcesser_.opName_,
            "batchA %lu batchA1-A4[%lu:%lu:%lu:%lu];"
            " batchB %lu batchB1-B4[%lu:%lu:%lu:%lu]; batchC %lu; batchBias %lu",
            quantBatchMatmulInfo.batchA, quantBatchMatmulInfo.batchA1, quantBatchMatmulInfo.batchA2,
            quantBatchMatmulInfo.batchA3, quantBatchMatmulInfo.batchA4, quantBatchMatmulInfo.batchB,
            quantBatchMatmulInfo.batchB1, quantBatchMatmulInfo.batchB2, quantBatchMatmulInfo.batchB3,
            quantBatchMatmulInfo.batchB4, quantBatchMatmulInfo.batchC, quantBatchMatmulInfo.batchBias);
    OP_LOGD(tilingProcesser_.opName_, "isPerTensor %d", static_cast<int32_t>(quantBatchMatmulInfo.isPerTensor));
}

QuantMatmulTPLParam QuantTilingTransferHelper::GetQuantMatmulTPLParam()
{
    QuantMatmulTPLParam param;
    param.trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    param.isPertoken = static_cast<uint64_t>(inputParams_.isPertoken);
    return param;
}

QuantTilingTransferHelper::QuantTilingTransferHelper(QuantMatmulAllReduceTiling &quantMatmulAllReduceTiling,
                                                     Mc2QuantBatchMatmulV3TilingData &data)
    : Mc2QuantBatchMatmulV3Tiling(quantMatmulAllReduceTiling.context_, &data),
      tilingProcesser_(quantMatmulAllReduceTiling)
{}

CutResult QuantMatmulAllReduceTiling::GetTilingResult()
{
    CutResult mCutAllreduceRes;
    SocVersion socVersion = SocVersion::SOC910_B;
    SetMCutSocVersion(socVersion);
    const gert::StorageShape *commQuantScaleA = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleB = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleA != nullptr) && (commQuantScaleB != nullptr)) { // low-bit comm
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantAllReduceTiling(args_, args_.rankDim, KernelType::ALL_REDUCE, socVersion);
        quantAllReduceTiling.GetTiling();
        mCutAllreduceRes = quantAllReduceTiling.tilingM_.cutRes;
    } else {
        MMPlusAllReduce allReduceTiling(args_, args_.rankDim, KernelType::ALL_REDUCE, socVersion, isPerBlock_);
        allReduceTiling.GetTiling();
        mCutAllreduceRes = allReduceTiling.tilingM_.cutRes;
    }
    return mCutAllreduceRes;
}

// 注册带SOC版本Tiling的类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, QuantMatmulAllReduceTiling,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 0);
} // namespace optiling
#endif //_QUANT_MATMUL_ALL_REDUCE_TILING_CC_
