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
 * \file kc_quant_matmul_allto_all_tiling_base.cpp
 * \brief
 */
#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "kc_quant_matmul_allto_all_tiling_base.h"
#include "matmul_allto_all_fit_balance_tiling.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace MC2Tiling {
gert::StorageShape kcQuantStorageShape = gert::StorageShape();
/**
 * @brief 当前量化过程的准入条件
 * @return true
 */
bool KcQuantMatmulAllToAllTilingBase::IsCapable()
{
    QuantMode mode = MatmulAlltoAllTilingUtil::GetQuantMode(context_, opName_);
    if (mode == QuantMode::KC_QUANT) {
        OP_LOGI(opName_, "Start with KcQuantMatmulAllToAll tiling.");
        return true;
    }
    OP_LOGI(opName_, "Skip KcQuantMatmulAllToAll tiling when not KC_QUANT.");
    return false;
}

/**
 * @brief 校验输入信息是否合规:attr,Dtype,shape等，使用通用校验util中的check方法
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::CheckOpInputInfo()
{
    OP_TILING_CHECK(
        MatmulAlltoAllTilingUtil::CheckAttrsInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "Tiling check Attrs failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckKcTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckKcQuantTensorDataType(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "tiling check Dtype failed in kc quant matmul all to all."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckKcQuantShapeInfo(context_, opName_, MATMUL_ALLTOALL_INDEX_SCHEMA) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check kc quant shape info failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckKcQuantMatrixMulShapes(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check kc quant matrix shape failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验参数的format::是否为私有格式
 *
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param opName: 算子名称
 * @return
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::CheckKcTensorFormat(const gert::TilingContext *context,
                                                                     const char *opName)
{
    OP_TILING_CHECK(MatmulAlltoAllTilingUtil::CheckTensorFormat(context_, opName_) != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName_, "Tiling check format failed."), return ge::GRAPH_FAILED);
    auto x1ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((x1ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x1Scale"),
                    return ge::GRAPH_FAILED);
    ge::Format x1ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1ScaleTensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x1ScaleFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x1Scale", Ops::Base::ToString(x1ScaleFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    auto x2ScaleTensorDesc = context->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    OP_TILING_CHECK((x2ScaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName, "x2Scale"),
                    return ge::GRAPH_FAILED);
    ge::Format x2ScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2ScaleTensorDesc->GetStorageFormat()));
    OP_TILING_CHECK(x2ScaleFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName, "x2Scale", Ops::Base::ToString(x2ScaleFormat).c_str(), "ND"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 根据输入设置tiling参数
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::InitTilingContextParameters()
{
    MC2_CHECK_LOG_RET(
        opName_, MatmulAlltoAllTilingUtil::SetAttrsInfo(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA));
    MC2_CHECK_LOG_RET(opName_, MatmulAlltoAllTilingUtil::SetKcDataTypeInfo(context_, opName_, contextInfo));
    MC2_CHECK_LOG_RET(opName_, MatmulAlltoAllTilingUtil::SetShapeInfo(context_, contextInfo));
    contextInfo.quantMode = QuantMode::KC_QUANT;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl参数；进行通算切分, 获取mm tiling等
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    MC2_CHECK_LOG_RET(opName_, CheckOpInputInfo());
    // 参数校验通过后赋值给全局上下文变量
    MC2_CHECK_LOG_RET(opName_, InitTilingContextParameters());
    // 进行通算切分
    MC2_CHECK_LOG_RET(opName_, TileCommAndCompute());
    // 调用量化Matmul的tiling方法进行切分
    MC2_CHECK_LOG_RET(opName_, DoKcQuantMMTiling());
    // hccl的tiling参数赋值处理
    MC2_CHECK_LOG_RET(opName_, SetHcclTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置hccl的config,进行hccl对应的通信任务设置
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::SetHcclTiling()
{
    OP_TILING_CHECK(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType) ==
                        mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
                    OP_LOGE(opName_, "Cannot find HcclDataType according to ge datatype = %d.",
                            static_cast<int32_t>(contextInfo.args_.geCType)),
                    return ge::GRAPH_FAILED;);
    Mc2CcTilingConfigBuilder allToAllBuilder =
        Mc2CcTilingConfigBuilder::create(contextInfo.group, mc2tiling::AicpuComType::HCCL_CMD_ALLTOALL,
                                         Mc2CcTilingConfigBuilder::AlgConfigType::ALL_TO_ALL);

    // 获取commMode
    uint8_t engineType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA,
                                                        engineType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // reducetype接口附带的数据类型优先于调用通信接口传入的数据类型，因此这里需要设置
    AscendC::Mc2CcTilingConfig allToAllTilingConfig =
        allToAllBuilder
            .withReduceType(opName_, mc2tiling::HcclReduceOp::HCCL_REDUCE_SUM, contextInfo.args_.geCType,
                            contextInfo.args_.geCType)
            .withCommEngine(engineType)
            .build();
    if (!allToAllBuilder.isSuccess()) {
        OP_LOGE(opName_, "Build hccl tiling config failed: %s", allToAllBuilder.errorMsg().c_str());
        return ge::GRAPH_FAILED;
    }
    allToAllTilingConfig.GetTiling(localTilingData_.mc2InitTiling);
    allToAllTilingConfig.GetTiling(localTilingData_.mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 进行通算切分之后单个块的MM Tiling
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::DoKcQuantMMTiling()
{
    // 设置MM切前信息
    mmMvalueLen = inferredInfo.tileM;
    KcQuantMatmulAlltoAllHelper mmTile(*this, localTilingData_.mc2QuantBmmV3TileTilingData, mmMvalueLen);
    MC2_CHECK_LOG_RET(opName_, mmTile.DoTiling());
    if (inferredInfo.tailCnt == 0) {
        return ge::GRAPH_SUCCESS;
    }
    mmMvalueLen = inferredInfo.tailM;
    KcQuantMatmulAlltoAllHelper mmTail(*this, localTilingData_.mc2QuantBmmV3TailTilingData, mmMvalueLen);
    MC2_CHECK_LOG_RET(opName_, mmTail.DoTiling());
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写获取MM index的信息
 * 由于本算子的context和MM不一样，需要重写获取MM
 * index的一些信息，把我们的context传给Matmul，来达到可以调用MM策略的目的。
 * @return ge::graphStatus
 */
const gert::Shape KcQuantMatmulAlltoAllHelper::GetX1Shape(const size_t index)
{
    (void)index;
    return gert::Shape({static_cast<int64_t>(mmLen), static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
}
const gert::Shape KcQuantMatmulAlltoAllHelper::GetX2Shape(const size_t index)
{
    (void)index;
    if (tilingProcesser_.contextInfo.args_.isBTrans) {
        return gert::Shape({static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue),
                            static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue)});
    }
    return gert::Shape({static_cast<int64_t>(tilingProcesser_.contextInfo.args_.kValue),
                        static_cast<int64_t>(tilingProcesser_.contextInfo.args_.nValue)});
}

const gert::Shape &KcQuantMatmulAlltoAllHelper::GetScaleShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_X2_SCALE_INDEX))->GetStorageShape();
}

const gert::StorageShape *KcQuantMatmulAlltoAllHelper::GetOffsetShape(const size_t index)
{
    (void)index;
    return (gert::StorageShape *)nullptr;
}

const gert::StorageShape *KcQuantMatmulAlltoAllHelper::GetPertokenShape(const size_t index)
{
    (void)index;
    kcQuantStorageShape = gert::StorageShape({static_cast<int64_t>(mmLen)}, {static_cast<int64_t>(mmLen)});
    return &kcQuantStorageShape;
}

const gert::StorageShape *KcQuantMatmulAlltoAllHelper::GetBiasShape(const size_t index)
{
    (void)index;
    return context_->GetOptionalInputShape(static_cast<size_t>(INPUT_BIAS_INDEX));
}

ge::graphStatus KcQuantMatmulAlltoAllHelper::GetShapeAttrsInfo()
{
    OP_LOGD(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
    auto &&tilingArgs = tilingProcesser_.contextInfo.args_;
    inputParams_.opName = tilingProcesser_.opName_;
    inputParams_.transA = false;
    inputParams_.transB = tilingArgs.isBTrans;
    inputParams_.hasBias = tilingArgs.isBias;
    inputParams_.libApiWorkSpaceSize = tilingProcesser_.libApiWorkSpaceSize_;
    inputParams_.aDtype = tilingArgs.geAType;
    inputParams_.bDtype = tilingArgs.geBType;
    int64_t yDType = *context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_Y_DTYPE_INDEX);
    auto scaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X2_SCALE_INDEX);
    auto perTokenScaleTensorDesc = context_->GetOptionalInputDesc(INPUT_X1_SCALE_INDEX);
    OP_TILING_CHECK((scaleTensorDesc == nullptr), OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "scale tensor"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((perTokenScaleTensorDesc == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(tilingProcesser_.opName_, "perToken scale tensor"),
                    return ge::GRAPH_FAILED);
    inputParams_.scaleDtype = scaleTensorDesc->GetDataType();
    inputParams_.perTokenScaleDtype = perTokenScaleTensorDesc->GetDataType();
    inputParams_.cDtype = static_cast<ge::DataType>(yDType);
    inputParams_.outDtype = static_cast<int64_t>(yDType);
    OP_LOGD(tilingProcesser_.opName_, "yDType is %ld", inputParams_.outDtype);
    inputParams_.biasDtype = tilingArgs.isBias ? tilingArgs.geBiasType : ge::DT_INT32;
    if (inputParams_.isPerChannel) {
        inputParams_.groupSizeM = 1;
        inputParams_.groupSizeN = 1;
    }
    MC2_CHECK_TRUE_RET(tilingProcesser_.opName_, AnalyzeInputs());
    PrintTilingInputParam(inputParams_);
    return ge::GRAPH_SUCCESS;
}

void KcQuantMatmulAlltoAllHelper::PrintTilingInputParam(Mc2QuantBatchMatmulInfo &quantMatmulInfo)
{
    OP_LOGD(tilingProcesser_.opName_, "mSize_ %ld kSize_ %ld nSize_ %ld libApiWorkSpaceSize %u", quantMatmulInfo.mSize,
            quantMatmulInfo.kSize, quantMatmulInfo.nSize, quantMatmulInfo.libApiWorkSpaceSize);
    OP_LOGD(tilingProcesser_.opName_,
            "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d outDtype %ld"
            " scaleDtype %d perTokenScaleDtype %d",
            static_cast<int32_t>(quantMatmulInfo.aDtype), static_cast<int32_t>(quantMatmulInfo.bDtype),
            static_cast<int32_t>(quantMatmulInfo.cDtype), static_cast<int32_t>(quantMatmulInfo.biasDtype),
            quantMatmulInfo.outDtype, static_cast<int32_t>(quantMatmulInfo.scaleDtype),
            static_cast<int32_t>(quantMatmulInfo.perTokenScaleDtype));
    OP_LOGD(tilingProcesser_.opName_, "Check isPertoken=%d.", static_cast<int32_t>(quantMatmulInfo.isPerChannel));
}

ge::graphStatus KcQuantMatmulAlltoAllHelper::DoLibApiTiling()
{
    MC2_CHECK_LOG_RET(tilingProcesser_.opName_, Mc2AdaptiveSlidingWindowTiling::DoLibApiTiling());
    isBf16Opt_ = false;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 重写友元类PostTiling方法
 * PostTiling主要做的是拷贝tilingdata的活，但是本算子拷贝tilingdata是在大结构体中拷贝，不需要在此处拷贝。
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAlltoAllHelper::PostTiling()
{
    tilingProcesser_.workspaceSize_ = std::max(tilingProcesser_.workspaceSize_, workspaceSize_);
    OP_LOGD(tilingProcesser_.opName_, "set mm workspace size %lu to mc2", tilingProcesser_.workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 构造函数，创建一个KcQuantMatmulAlltoAllHelper对象
 *
 * @param context
 */
KcQuantMatmulAlltoAllHelper::KcQuantMatmulAlltoAllHelper(
    KcQuantMatmulAllToAllTilingBase &kcQuantMatmulAllToAllTilingBase,
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &data, uint64_t &mmMvalueLen)
    : Mc2AdaptiveSlidingWindowTiling(kcQuantMatmulAllToAllTilingBase.context_, &data),
      tilingProcesser_(kcQuantMatmulAllToAllTilingBase),
      mmLen(mmMvalueLen)
{}

/**
 * @brief 打印量化matmul tiling的信息
 *
 * @param opName
 * @param tiling
 */
void KcQuantMatmulAllToAllTilingBase::PrintKcQuantMMV3TilingData(
    const std::string &opName, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    PrintTCubeTilingData(opName, tiling.matmulTiling);
    PrintExtendMatmulTiling(opName, tiling);
}

/**
 * @brief 打印执行过程中的matmul tiling信息
 *
 * @param opName
 * @param tiling
 */
void KcQuantMatmulAllToAllTilingBase::PrintExtendMatmulTiling(const std::string &opName,
                                                              DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams &tiling)
{
    OP_LOGD(opName, "QuantBmmV3Params.batchA=%u.", tiling.params.batchA);
    OP_LOGD(opName, "QuantBmmV3Params.batchB=%u.", tiling.params.batchB);
    OP_LOGD(opName, "QuantBmmV3Params.batchC=%u.", tiling.params.batchC);
    OP_LOGD(opName, "QuantBmmV3Params.batchA1=%u.", tiling.params.batchA1);
    OP_LOGD(opName, "QuantBmmV3Params.batchA2=%u.", tiling.params.batchA2);
    OP_LOGD(opName, "QuantBmmV3Params.batchA3=%u.", tiling.params.batchA3);
    OP_LOGD(opName, "QuantBmmV3Params.batchA4=%u.", tiling.params.batchA4);
    OP_LOGD(opName, "QuantBmmV3Params.batchB1=%u.", tiling.params.batchB1);
    OP_LOGD(opName, "QuantBmmV3Params.batchB2=%u.", tiling.params.batchB2);
    OP_LOGD(opName, "QuantBmmV3Params.batchB3=%u.", tiling.params.batchB3);
    OP_LOGD(opName, "QuantBmmV3Params.batchB4=%u.", tiling.params.batchB4);
    OP_LOGD(opName, "QuantBmmV3Params.batchC1=%u.", tiling.params.batchC1);
    OP_LOGD(opName, "QuantBmmV3Params.batchC2=%u.", tiling.params.batchC2);
    OP_LOGD(opName, "QuantBmmV3Params.batchC3=%u.", tiling.params.batchC3);
    OP_LOGD(opName, "QuantBmmV3Params.batchC4=%u.", tiling.params.batchC4);
    OP_LOGD(opName, "QuantBmmV3Params.singleCoreBatch=%u.", tiling.params.singleCoreBatch);
    OP_LOGD(opName, "QuantBmmV3Params.isPerTensor=%u.", tiling.params.isPerTensor);
    OP_LOGD(opName, "QuantBmmV3Params.isPertoken=%u.", tiling.params.isPertoken);
    OP_LOGD(opName, "QuantBmmV3Params.isDoubleScale=%u.", tiling.params.isDoubleScale);
    OP_LOGD(opName, "QuantBmmV3Params.biasThreeDim=%u.", tiling.params.biasThreeDim);
    OP_LOGD(opName, "QuantBmmV3Params.ubCalcM=%u.", tiling.params.ubCalcM);
    OP_LOGD(opName, "QuantBmmV3Params.ubCalcN=%u.", tiling.params.ubCalcN);
    OP_LOGD(opName, "QuantBmmV3Params.needUbBuffer=%u.", tiling.params.needUbBuffer);
    OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreM=%u.", tiling.params.realSingleCoreM);
    OP_LOGD(opName, "QuantBmmV3Params.realSingleCoreN=%u.", tiling.params.realSingleCoreN);
    OP_LOGD(opName, "QuantBmmV3Params.biasDtype=%u.", tiling.params.biasDtype);
    OP_LOGD(opName, "QuantBmmV3Params.ubSize=%u.", tiling.params.ubSize);
    OP_LOGD(opName, "QuantBmmV3Params.isMClash=%u.", tiling.params.isMClash);
    OP_LOGD(opName, "QuantBmmV3Params.isNClash=%u.", tiling.params.isNClash);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeM=%u.", tiling.params.groupSizeM);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeK=%u.", tiling.params.groupSizeK);
    OP_LOGD(opName, "QuantBmmV3Params.groupSizeN=%u.", tiling.params.groupSizeN);
    OP_LOGD(opName, "TileL2cacheTiling.mTileCntL2=%u.", tiling.tileL2cacheTiling.mTileCntL2);
    OP_LOGD(opName, "TileL2cacheTiling.nTileCntL2=%u.", tiling.tileL2cacheTiling.nTileCntL2);
    OP_LOGD(opName, "TileL2cacheTiling.mTileBlock=%u.", tiling.tileL2cacheTiling.mTileBlock);
    OP_LOGD(opName, "TileL2cacheTiling.nTileBlock=%u.", tiling.tileL2cacheTiling.nTileBlock);
    OP_LOGD(opName, "TileL2cacheTiling.calOrder=%u.", tiling.tileL2cacheTiling.calOrder);
    OP_LOGD(opName, "TileL2cacheTiling.isBasicTiling=%u.", tiling.tileL2cacheTiling.isBasicTiling);
    OP_LOGD(opName, "AdaptiveSlidingWin.mTailTile=%u.", tiling.adaptiveSlidingWin.mTailTile);
    OP_LOGD(opName, "AdaptiveSlidingWin.nTailTile=%u.", tiling.adaptiveSlidingWin.nTailTile);
}

/**
 * @brief 打印tilingInfo信息
 *
 * @param opName
 * @param tilingInfo
 */
void KcQuantMatmulAllToAllTilingBase::PrintKcQuantMatmulAlltoAllTilingInfo(const std::string &opName,
                                                                           MatmulAlltoAllTilingInfo &tilingInfo)
{
    OP_LOGD(opName, "tilingInfo.rankDim: %u", tilingInfo.rankDim);
    OP_LOGD(opName, "tilingInfo.tileM: %u", tilingInfo.tileM);
    OP_LOGD(opName, "tilingInfo.tileCnt: %u", tilingInfo.tileCnt);
    OP_LOGD(opName, "tilingInfo.tailM: %u", tilingInfo.tailM);
    OP_LOGD(opName, "tilingInfo.tailCnt: %u", tilingInfo.tailCnt);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.rankM: %u", tilingInfo.rankM);
    OP_LOGD(opName, "tilingInfo.rankN: %u", tilingInfo.rankN);
    OP_LOGD(opName, "tilingInfo.rankK: %u", tilingInfo.rankK);
    OP_LOGD(opName, "tilingInfo.mmResultLen: %lu", tilingInfo.mmResultLen);
    OP_LOGD(opName, "tilingInfo.permuteLen: %lu", tilingInfo.permuteLen);
    OP_LOGD(opName, "tilingInfo.biasLen: %u", tilingInfo.biasLen);
    OP_LOGD(opName, "tilingInfo.aicCoreNum: %u", tilingInfo.aicCoreNum);
    OP_LOGD(opName, "tilingInfo.hcclDataType: %lu", tilingInfo.hcclDataType);
}

/**
 * @brief 打印传递给kernel的tilingData
 *
 * @param outTilingData tilingData参数
 */
void KcQuantMatmulAllToAllTilingBase::PrintKcQuantMatmulAlltoAllTilingData(QuantMatmulAlltoAllTilingData &outTilingData)
{
    PrintKcQuantMatmulAlltoAllTilingInfo(opName_, outTilingData.quantMatmulAlltoAllTilingInfo);
    PrintKcQuantMMV3TilingData(opName_, outTilingData.mc2QuantBmmV3TileTilingData);
    if (outTilingData.quantMatmulAlltoAllTilingInfo.tailCnt == 0) {
        return;
    }
    OP_LOGD(opName_, "KcQuantMatmulAlltoall has tail");
    PrintKcQuantMMV3TilingData(opName_, outTilingData.mc2QuantBmmV3TailTilingData);
}

/**
 * @brief 保存量化tiling数据到context
 *
 * @return ge::graphStatus
 */
ge::graphStatus KcQuantMatmulAllToAllTilingBase::PostTiling()
{
    context_->SetScheduleMode(1);
    SetTilingInfo(localTilingData_.quantMatmulAlltoAllTilingInfo);
    QuantMatmulAlltoAllTilingData *outTilingData = context_->GetTilingData<QuantMatmulAlltoAllTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tilingBufCap < sizeof(localTilingData_)),
                    OP_LOGE(opName_, "TilingBuffer capacity too small, capacity = %zu, need = %zu.", tilingBufCap,
                            sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(outTilingData, tilingBufCap, &localTilingData_, sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "MatmulAlltoAll postTiling: memcpy_s tiling data failed, ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(QuantMatmulAlltoAllTilingData),
            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(QuantMatmulAlltoAllTilingData));
    context_->SetBlockDim(contextInfo.args_.aicCoreNum);
    PrintKcQuantMatmulAlltoAllTilingData(*outTilingData);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingInfo结构体
 *
 * @param tilingInfo 目标结构体
 */
void KcQuantMatmulAllToAllTilingBase::SetTilingInfo(MatmulAlltoAllTilingInfo &tilingInfo) const
{
    // 基本字段拷贝
    tilingInfo.rankM = contextInfo.args_.mValue;
    tilingInfo.rankN = contextInfo.args_.nValue;
    tilingInfo.rankK = contextInfo.args_.kValue;
    tilingInfo.tileM = inferredInfo.tileM;
    tilingInfo.tileCnt = inferredInfo.tileCnt;
    tilingInfo.tailM = inferredInfo.tailM;
    tilingInfo.tailCnt = inferredInfo.tailCnt;
    tilingInfo.mmResultLen = inferredInfo.mmResultLen;
    tilingInfo.permuteLen = inferredInfo.permuteLen;
    tilingInfo.biasLen = inferredInfo.biasLen;
    tilingInfo.aicCoreNum = contextInfo.args_.aicCoreNum;
    tilingInfo.rankDim = contextInfo.args_.rankDim;
    tilingInfo.hcclDataType =
        (static_cast<uint64_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, contextInfo.args_.geCType))); // hccl数据类型
}

/**
 * @brief 获取对应的tilingKey
 * 使用QUANT_MODE来区分tilingKey,此处的QUANT_MODE指的是x1,x2的QUANT模式组合，以x1为pertoken量化(K)，x2为perchannel量化(C)
 * 为例子，K-C量化就代表一种组合
 *
 * @return uint64_t tilingKey结果
 */
uint64_t KcQuantMatmulAllToAllTilingBase::GetTilingKey() const
{
    // 按照量化组合模式，是否转置，bias数据类型进行展开
    bool x2TransposeFlag = contextInfo.args_.isBTrans ? true : false;
    uint32_t biasDType = DTYPE_BIAS_FP32;

    uint8_t engineType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA,
                                                        engineType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint8_t commMode = (engineType == mc2tiling::A5_CCU_ENGINE) ? COMMMODE_CCU : COMMMODE_AICPU;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(KC_QUANT_MODE, x2TransposeFlag, biasDType, commMode, ADDBIAS_OFF);
    OP_LOGD(opName_,
            "KCQUANTMODE,X2TRANSPOSE,DTYPEBIAS,COMMMODE,ADDBIASSPLITMODE: [%d,%d,%d,%d,0], TilingKey is [%lu].",
            KC_QUANT_MODE, x2TransposeFlag, biasDType, commMode, tilingKey);
    return tilingKey;
}

/**
 * @brief 获取tiling切分结果（arch35覆盖）
 *
 * @return CutResult
 */
CutResult KcQuantMatmulAllToAllTilingBase::GetTilingResult()
{
    uint8_t engineType = 0;
    if (MatmulAlltoAllTilingUtil::GetAndConvertCommMode(context_, opName_, contextInfo, MATMUL_ALLTOALL_INDEX_SCHEMA,
                                                        engineType) != ge::GRAPH_SUCCESS) {
        OP_LOGD(opName_, "GetTilingResult: failed to get commMode, default to AICPU.");
    }

    uint8_t commMode = (engineType == mc2tiling::A5_CCU_ENGINE) ? Mc2Comm::COMM_MODE_CCU : Mc2Comm::COMM_MODE_AICPU;
    return GetArch35TilingResult(contextInfo.args_, KernelType::ALL_TO_ALL, SocVersion::SOC950, npuArch_,
                                 QuantMode::KC_QUANT, commMode);
}

/**
 * @brief 构造函数，创建一个KcQuantMatmulAllToAllTilingBase对象
 *
 * @param context
 */
KcQuantMatmulAllToAllTilingBase::KcQuantMatmulAllToAllTilingBase(gert::TilingContext *context)
    : MatmulAllToAllTilingBase(context)
{}

// 注册tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(MatmulAlltoAll, KcQuantMatmulAllToAllTilingBase,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

} // namespace MC2Tiling
