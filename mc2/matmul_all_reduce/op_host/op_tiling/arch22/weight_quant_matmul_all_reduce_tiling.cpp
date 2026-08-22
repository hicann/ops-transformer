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
 * \file weight_quant_matmul_all_reduce_tiling.cc
 * \brief
 */
#include "weight_quant_matmul_all_reduce_tiling.h"
#include "common/utils/op_mc2.h"
#include "mc2_log.h"

using namespace Mc2Log;
using namespace Mc2Tiling;
namespace optiling {

ge::graphStatus WeightQuantTilingTransferHelper::GetShapeAttrsInfo()
{
    OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.args_;
    opName_ = tilingProcesser_.opName_;
    matmulInfoPtr_ = std::make_unique<Mc2WeightQuantBatchMatmulInfo>();
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
    OP_TILING_CHECK(
        (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) &&
            (matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_CHANNEL),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "antiQuantType", std::to_string(static_cast<int>(matmulInfoPtr_->antiQuantType)).c_str(),
            "When the weight format is Nz, the value of antiQuantType must be per-channel"),
        return ge::GRAPH_FAILED);
    PrintTilingInputParam(*matmulInfoPtr_);
    return ge::GRAPH_SUCCESS;
}
void WeightQuantTilingTransferHelper::PrintTilingInputParam(Mc2WeightQuantBatchMatmulInfo &weightQuantBatchMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, " transA_ %d transB_ %d, hasBias_ %d, hasAntiQuantOffset_ %d",
            weightQuantBatchMatmulInfo.transA, weightQuantBatchMatmulInfo.transB, weightQuantBatchMatmulInfo.hasBias,
            weightQuantBatchMatmulInfo.hasAntiQuantOffset);
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld groupSize_ %ld",
            weightQuantBatchMatmulInfo.mSize, weightQuantBatchMatmulInfo.kSize, weightQuantBatchMatmulInfo.nSize,
            weightQuantBatchMatmulInfo.groupSize);
    OP_LOGD(tilingProcesser_.opName_, "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d",
            static_cast<int32_t>(weightQuantBatchMatmulInfo.aDtype),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.bDtype),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.cDtype),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.biasDtype));
    OP_LOGD(tilingProcesser_.opName_, "antiQuantType_ %d quantType_ %d bFormat %d",
            static_cast<int32_t>(weightQuantBatchMatmulInfo.antiQuantType),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.quantType),
            static_cast<int32_t>(weightQuantBatchMatmulInfo.bFormat));
}
ge::graphStatus WeightQuantTilingTransferHelper::PostTiling()
{
    tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
    OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
    return ge::GRAPH_SUCCESS;
}

WeightQuantMatmulTPLParam WeightQuantTilingTransferHelper::GetWeightQuantMatmulTPLParam()
{
    Mc2KernelTemplateType templateType = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ ?
                                             Mc2KernelTemplateType::WEIGHT_NZ :
                                             Mc2KernelTemplateType::CUSTOM_ANTIQUANT;
    WeightQuantMatmulTPLParam param;
    param.subAlgorithmCustom = static_cast<uint64_t>(templateType);
    param.hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    param.antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    param.transA = 0; // v220 do not support transA
    param.transB = matmulInfoPtr_->transB;
    return param;
}

bool WeightQuantMatmulAllReduceTiling::IsCapable()
{
    if (isA16W8_ || isA16W4_) {
        OP_LOGI(opName_, "start with weight quant tiling.");
        return true;
    }
    OP_LOGI(opName_, "skip weight quant tiling as dtype not support");
    return false;
}

void WeightQuantMatmulAllReduceTiling::DoEmptyTensorTiling()
{
    MutableTCubeTileTilingData().M = args_.orgMValue;
    MutableTCubeTileTilingData().isBias = args_.isBias;
    MutableTCubeTileTilingData().usedCoreNum = 1;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, CheckA16W8());
    MC2_CHECK_LOG_RET(opName_, CheckInput());
    MC2_CHECK_LOG_RET(opName_, CheckCommModeA2());
    DoRCSTiling();
    DoSplitMTiling();
    if (isKZero_) {
        DoEmptyTensorTiling();
        DoAllReduceTiling(true);
        return ge::GRAPH_SUCCESS;
    }
    MC2_CHECK_LOG_RET(opName_, DoWeightQuantTiling());
    DoAllReduceTiling(true);
    return ge::GRAPH_SUCCESS;
}
uint64_t WeightQuantMatmulAllReduceTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (isKZero_) {
        tilingKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_910B), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            isKZero_, MATMUL_ALLREDUCE_INT8_COMM_F,
            0UL, // ENABLE_L2_CACHE
            0UL, // SHARE_MM
            SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING, SET_NOT_USE_WEIGHT_QUANT_MM_TPL_TILING);
    } else {
        tilingKey = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(ASCEND_910B), static_cast<uint64_t>(MATMUL_ALLREDUCE_MM_TYPE_WEIGHT_QUANT_MATMUL),
            MATMUL_ALLREDUCE_EMPTY_INPUT_F, MATMUL_ALLREDUCE_INT8_COMM_F,
            0UL, // ENABLE_L2_CACHE
            0UL, // SHARE_MM
            SET_NOT_USE_FM_MM_TPL_TILING, SET_NOT_USE_QUANT_MM_TPL_TILING,
            weightQuantMatmulTPLParam_.subAlgorithmCustom, weightQuantMatmulTPLParam_.hasAntiquantOffset,
            weightQuantMatmulTPLParam_.antiquantType, weightQuantMatmulTPLParam_.transA,
            weightQuantMatmulTPLParam_.transB, static_cast<uint64_t>(FORMAT_B_ND));
    }
    OP_LOGI(opName_, " tilingKey %lu", tilingKey);
    return tilingKey;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::GetWorkspaceSize()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::GetWorkspaceSize());
    myWorkSpaceSize_ = myWorkSpaceSize_ + MutableRCSTilingData().biasLen;
    if (isKZero_) {
        myWorkSpaceSize_ = myWorkSpaceSize_ + libApiWorkSpaceSize_;
        OP_LOGD(opName_, " Empty tensor k is 0, set workspace size %lu to context", myWorkSpaceSize_);
    }
    OP_LOGI(opName_, " set max workspace size %lu to context", myWorkSpaceSize_);
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = myWorkSpaceSize_;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::PostTiling()
{
    size_t tilingDataSize = sizeof(WeightQuantMatmulAllReduceTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu and context capacity size: %zu ", tilingDataSize,
            context_->GetRawTilingData()->GetCapacity());
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&weightQuantMatmulAllReduceTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    PrintTilingData();

    context_->SetBlockDim(args_.aicCoreNum);

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    ret = context_->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(opName_, ret);

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus WeightQuantMatmulAllReduceTiling::DoWeightQuantTiling()
{
    args_.mValue = tileMValue_;
    WeightQuantTilingTransferHelper mmTile(*this, weightQuantMatmulAllReduceTilingData_.tilematmulTiling);
    if (args_.enableSplitK) {
        auto res = mmTile.DoTiling();
        weightQuantMatmulTPLParam_ = mmTile.GetWeightQuantMatmulTPLParam();
        return res;
    } else {
        MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
        if (MutableRCSTilingData().tailCnt == 0) {
            weightQuantMatmulTPLParam_ = mmTile.GetWeightQuantMatmulTPLParam();
            return ge::GRAPH_SUCCESS;
        }
        args_.mValue = tailMValue_;
        WeightQuantTilingTransferHelper mmTail(*this, weightQuantMatmulAllReduceTilingData_.tailmatmulTiling);
        auto res = mmTail.DoTiling();
        weightQuantMatmulTPLParam_ = mmTail.GetWeightQuantMatmulTPLParam();
        return res;
    }
}

ge::graphStatus WeightQuantMatmulAllReduceTiling::CheckAxisSize()
{
    const uint64_t m = MatmulAllReduceTilingBase::GetMValue();
    OP_TILING_CHECK(m > static_cast<uint64_t>(INT32_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", std::to_string(m).c_str(),
                                                          "The value of m of x1 must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);
    const uint64_t k = MatmulAllReduceTilingBase::GetKValue();
    OP_TILING_CHECK(k > static_cast<uint64_t>(UINT16_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x1", std::to_string(k).c_str(),
                                                          "The value of k of x1 must not exceed UINT16_MAX"),
                    return ge::GRAPH_FAILED);
    const uint64_t n = MatmulAllReduceTilingBase::GetNValue();
    OP_TILING_CHECK(n > static_cast<uint64_t>(UINT16_MAX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x2", std::to_string(n).c_str(),
                                                          "The value of n of x2 must not exceed UINT16_MAX"),
                    return ge::GRAPH_FAILED);

    return CheckWeightQuantEmptyTensor();
}

ge::graphStatus WeightQuantMatmulAllReduceTiling::CheckInputDtype() const
{
    auto x1Type = mmrCtxInfo_.x1->GetDataType();
    OP_TILING_CHECK(
        !((x1Type == ge::DT_FLOAT16) || (x1Type == ge::DT_BF16)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x1", Ops::Base::ToString(x1Type).c_str(),
                                              "The dtype of x1 must be fp16 or bf16"),
        return ge::GRAPH_FAILED);
    auto x2Type = mmrCtxInfo_.x2->GetDataType();
    OP_TILING_CHECK(
        !((x2Type == ge::DT_INT8) || (x2Type == ge::DT_INT4)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x2", Ops::Base::ToString(x2Type).c_str(),
                                              "The dtype of x2 must be int8 or int4"),
        return ge::GRAPH_FAILED);
    if (mmrCtxInfo_.bias_shape != nullptr) {
        OP_TILING_CHECK(
            x1Type != mmrCtxInfo_.bias->GetDataType(),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x1 and bias",
                (Ops::Base::ToString(x1Type) + " and " + Ops::Base::ToString(mmrCtxInfo_.bias->GetDataType())).c_str(),
                "The dtypes of x1 and bias must be the same"),
            return ge::GRAPH_FAILED);
    }
    // x1,antiquantScale数据类型相同
    auto antiquantScaleType = mmrCtxInfo_.antiquant_scale->GetDataType();
    OP_TILING_CHECK(antiquantScaleType != x1Type,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "antiquantScale and x1",
                        (Ops::Base::ToString(antiquantScaleType) + " and " + Ops::Base::ToString(x1Type)).c_str(),
                        "The dtypes of antiquantScale and x1 must be the same"),
                    return ge::GRAPH_FAILED);
    // antiquantScale和antiquantOffset数据类型相同
    if (mmrCtxInfo_.antiquant_offset_shape != nullptr) {
        auto antiquantOffsetType = mmrCtxInfo_.antiquant_offset->GetDataType();
        OP_TILING_CHECK(
            antiquantOffsetType != antiquantScaleType,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "antiquantScale and antiquantOffset",
                (Ops::Base::ToString(antiquantOffsetType) + " and " + Ops::Base::ToString(antiquantScaleType)).c_str(),
                "The dtypes of antiquantScale and antiquantOffset must be the same"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantMatmulAllReduceTiling::CheckInput()
{
    MC2_CHECK_LOG_RET(opName_, MatmulAllReduceTilingBase::CheckInput());
    const size_t x2DimNum = (static_cast<ge::Format>(ge::GetPrimaryFormat(mmrCtxInfo_.x2->GetStorageFormat())) ==
                                     ge::Format::FORMAT_FRACTAL_NZ ?
                                 4 :
                                 2);
    const size_t actualX2DimNum = mmrCtxInfo_.x2_shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        x2DimNum != actualX2DimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x2", (std::to_string(actualX2DimNum) + "D").c_str(),
                                     (std::to_string(x2DimNum) + "D").c_str()),
        return ge::GRAPH_FAILED);
    MC2_CHECK_LOG_RET(opName_, CheckInputDtype());
    // antiquantgroupsize 校验
    uint64_t kValue = GetKValue();
    if (kValue != 0 && mmrCtxInfo_.antiquantGroupSizePtr != nullptr) {
        const int64_t groupSize = *(mmrCtxInfo_.antiquantGroupSizePtr);
        OP_TILING_CHECK(((groupSize != 0) && (groupSize % ANTIQUANT_GROUP_SIZE_MIN_VALUE != 0 ||
                                              groupSize < ANTIQUANT_GROUP_SIZE_MIN_VALUE ||
                                              groupSize > std::min(static_cast<int32_t>(kValue - 1), INT32_MAX))),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "antiquantGroupSize", std::to_string(groupSize).c_str(),
                            "The value of antiquantGroupSize must be in the range [32, min(k-1, INT_MAX)]"),
                        return ge::GRAPH_FAILED);
    }

    return CheckAxisSize();
}

WeightQuantMatmulAllReduceTiling::WeightQuantMatmulAllReduceTiling(gert::TilingContext *context)
    : MatmulAllReduceTilingBase(context),
      weightQuantMatmulAllReduceTilingData_(weightQuantMatmulAllReduceTilingDataSelf_)
{}
WeightQuantMatmulAllReduceTiling::WeightQuantMatmulAllReduceTiling(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                                                   WeightQuantMatmulAllReduceTilingData *out)
    : MatmulAllReduceTilingBase(context, mmrCtxInfo),
      weightQuantMatmulAllReduceTilingData_(*out)
{}

CutResult WeightQuantMatmulAllReduceTiling::GetTilingResult()
{
    CutResult mCutAllreduceVal;
    SocVersion inputSocVer = SocVersion::SOC910_B;
    SetMCutSocVersion(inputSocVer);
    const gert::StorageShape *commQuantScaleFirstShape = mmrCtxInfo_.comm_quant_scale_1_shape;
    const gert::StorageShape *commQuantScaleSecondShape = mmrCtxInfo_.comm_quant_scale_2_shape;
    if ((commQuantScaleFirstShape != nullptr) && (commQuantScaleSecondShape != nullptr)) { // low-bit comm
        OP_LOGD(opName_, "TileCnt enter comm quant.");
        MMPlusQuantAllReduce quantAllReduceHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVer);
        quantAllReduceHccl.GetTiling();
        mCutAllreduceVal = quantAllReduceHccl.tilingM_.cutRes;
    } else {
        MMPlusAllReduce allReduceHccl(args_, args_.rankDim, KernelType::ALL_REDUCE, inputSocVer, isPerBlock_);
        allReduceHccl.GetTiling();
        mCutAllreduceVal = allReduceHccl.tilingM_.cutRes;
    }
    return mCutAllreduceVal;
}

// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(MatmulAllReduce, WeightQuantMatmulAllReduceTiling,
                                         static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B), 1);
} // namespace optiling
