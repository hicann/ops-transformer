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
 * \file norm_rope_concat_grad_tiling.cpp
 * \brief
 */

#include "norm_rope_concat_grad_tiling.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus NormRopeConcatGradTiling::CheckInput(const InputTensorInfo& x)
{
    if (x.isRequired) {
        OP_CHECK_IF((x.desc == nullptr) || (x.shape == nullptr),
                    OP_LOGE(context_.opName, "dec or tensor or shape of required tensor should not be nullptr!"),
                    return ge::GRAPH_FAILED);
    }
    if (x.desc != nullptr) {
        if (x.type == TensorType::GRADIENT_NORM_TENSOR) {
            OP_CHECK_IF(x.desc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE(context_.opName, "Check dtype Failed, norm mean or rstd dtype should be float!"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(x.desc->GetDataType() != inputDtype_, OP_LOGE(context_.opName, "Check dtype Failed!"),
                        return ge::GRAPH_FAILED);
        }
    }
    if (x.shape != nullptr) {
        auto shape = x.shape->GetStorageShape();
        switch (x.type) {
            case TensorType::INPUT_TENSOR:
            case TensorType::ENCODER_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != INPUT_DIM_NUM,
                            OP_LOGE(context_.opName, "Shape Dims of tensor should be 4"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(BATCH_DIM) != batchSize_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be batchSize_(%lu)",
                                    shape.GetDim(BATCH_DIM), batchSize_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(HEAD_DIM) != headNum_,
                            OP_LOGE(context_.opName, "DIM2(%ld) Value of Shape should be headNum_(%lu)",
                                    shape.GetDim(HEAD_DIM), headNum_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(DIM_DIM) != headDim_,
                            OP_LOGE(context_.opName, "DIM3(%ld) Value of Shape should be headDim_(%lu)",
                                    shape.GetDim(DIM_DIM), headDim_),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::GRADIENT_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != INPUT_DIM_NUM,
                            OP_LOGE(context_.opName, "Shape Dims of tensor should be 4"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(BATCH_DIM) != batchSize_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be batchSize_(%lu)",
                                    shape.GetDim(BATCH_DIM), batchSize_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(
                    shape.GetDim(GRAD_SEQ_DIM) < ropeSeq_,
                    OP_LOGE(context_.opName, "DIM2(%ld) Value of Shape should not be smaller than ropeSeq_(%lu)",
                            shape.GetDim(GRAD_SEQ_DIM), ropeSeq_),
                    return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(GRAD_HEAD_DIM) != headNum_,
                            OP_LOGE(context_.opName, "DIM1(%ld) Value of Shape should be headNum_(%lu)",
                                    shape.GetDim(GRAD_HEAD_DIM), headNum_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(DIM_DIM) != headDim_,
                            OP_LOGE(context_.opName, "DIM3(%ld) Value of Shape should be headDim_(%lu)",
                                    shape.GetDim(DIM_DIM), headDim_),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::NORM_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != 1, OP_LOGE(context_.opName, "Shape Dims of tensor should be 1"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(0) != headDim_, // NOTE: not support cross heads
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be headDim_(%lu)",
                                    shape.GetDim(0), headDim_),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::GRADIENT_NORM_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != INPUT_DIM_NUM,
                            OP_LOGE(context_.opName, "Shape Dims of tensor should be 4"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(BATCH_DIM) != batchSize_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be Seq_(%lu)", shape.GetDim(0),
                                    batchSize_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(HEAD_DIM) != headNum_,
                            OP_LOGE(context_.opName, "DIM2(%ld) Value of Shape should be headNum_(%lu)",
                                    shape.GetDim(2), headNum_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(DIM_DIM) != 1,
                            OP_LOGE(context_.opName, "DIM3(%ld) Value of Shape should be 1", shape.GetDim(DIM_DIM)),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::ROPE_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != ROPE_DIM_NUM,
                            OP_LOGE(context_.opName, "Shape Dims of tensor should be 2"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_SEQ_DIM) > maxSeq_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should smaller than maxSeq_(%lu)",
                                    shape.GetDim(ROPE_SEQ_DIM), maxSeq_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_SEQ_DIM) != ropeSeq_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be ropeSeq_(%lu)",
                                    shape.GetDim(ROPE_SEQ_DIM), ropeSeq_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_DIM_DIM) != headDim_,
                            OP_LOGE(context_.opName, "DIM1(%ld) Value of Shape should be headDim_(%lu)",
                                    shape.GetDim(ROPE_DIM_DIM), headDim_),
                            return ge::GRAPH_FAILED);
                break;
            default:
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::CheckInput()
{
    OP_CHECK_IF((headDim_ % 2 != 0) || (headDim_ > 1024),
                OP_LOGE(context_.opName, "headDim of shape value should be even number and smaller than 1024 "),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < context_.maxInputNum; ++i) {
        if (CheckInput(context_.inputs[i]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(totalQuerySeq_ != querySeq_ + encoderQuerySeq_,
                OP_LOGE(context_.opName, "totalQuerySeq_ != querySeq_ + encoderQuerySeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalKeySeq_ != keySeq_ + encoderKeySeq_,
                OP_LOGE(context_.opName, "totalKeySeq_ != keySeq_ + encoderKeySeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalValueSeq_ != valueSeq_ + encoderValueSeq_,
                OP_LOGE(context_.opName, "totalValueSeq_ != valueSeq_ + encoderValueSeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keySeq_ != valueSeq_, OP_LOGE(context_.opName, "keySeq_ != valueSeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(encoderKeySeq_ != encoderValueSeq_, OP_LOGE(context_.opName, "encoderKeySeq_ != encoderValueSeq_"),
                return ge::GRAPH_FAILED);

    normDim_ = headDim_;
    scale_ = 1 / float(headDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::GetShapeInfo()
{
    size_t queryIndex = static_cast<size_t>(InputIndexBackward::QUERY_INDEX);
    size_t keyIndex = static_cast<size_t>(InputIndexBackward::KEY_INDEX);
    const gert::StorageShape* queryShape = context_.inputs[queryIndex].shape;
    const gert::StorageShape* keyShape = context_.inputs[keyIndex].shape;
    OP_CHECK_IF(queryShape == nullptr || queryShape->GetStorageShape().GetDimNum() != INPUT_DIM_NUM,
                OP_LOGE(context_.opName, "Shape of tensor query should be 4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape == nullptr || keyShape->GetStorageShape().GetDimNum() != INPUT_DIM_NUM,
                OP_LOGE(context_.opName, "Shape of tensor key should be 4"), return ge::GRAPH_FAILED);

    batchSize_ = queryShape->GetStorageShape().GetDim(BATCH_DIM);
    headNum_ = queryShape->GetStorageShape().GetDim(HEAD_DIM);
    headDim_ = queryShape->GetStorageShape().GetDim(DIM_DIM);
    querySeq_ = queryShape->GetStorageShape().GetDim(SEQ_DIM);
    keySeq_ = keyShape->GetStorageShape().GetDim(SEQ_DIM);
    valueSeq_ = keySeq_;

    size_t encoderQueryIndex = static_cast<size_t>(InputIndexBackward::ENCODER_QUERY_INDEX);
    size_t encoderKeyIndex = static_cast<size_t>(InputIndexBackward::ENCODER_KEY_INDEX);
    const gert::StorageShape* encoderQueryShape = context_.inputs[encoderQueryIndex].shape;
    const gert::StorageShape* encoderKeyShape = context_.inputs[encoderKeyIndex].shape;
    if (encoderQueryShape != nullptr && encoderQueryShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderQuerySeq_ = encoderQueryShape->GetStorageShape().GetDim(SEQ_DIM);
    }
    if (encoderKeyShape != nullptr && encoderKeyShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderKeySeq_ = encoderKeyShape->GetStorageShape().GetDim(SEQ_DIM);
    }
    encoderValueSeq_ = encoderKeySeq_;

    size_t ropeSinIndex = static_cast<size_t>(InputIndexBackward::ROPE_SIN_INDEX);
    const gert::StorageShape* ropeSinShape = context_.inputs[ropeSinIndex].shape;
    if (ropeSinShape != nullptr && ropeSinShape->GetStorageShape().GetDimNum() == ROPE_DIM_NUM) {
        ropeSeq_ = ropeSinShape->GetStorageShape().GetDim(ROPE_SEQ_DIM);
        ropeDim_ = ropeSinShape->GetStorageShape().GetDim(ROPE_DIM_DIM);
    }

    totalQuerySeq_ = querySeq_ + encoderQuerySeq_;
    totalKeySeq_ = keySeq_ + encoderKeySeq_;
    totalValueSeq_ = valueSeq_ + encoderValueSeq_;
    maxSeq_ = std::max({totalQuerySeq_, totalKeySeq_, totalValueSeq_});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::ComputeTilingKey()
{
    OP_CHECK_IF(context_.normType == nullptr, OP_LOGE(context_.opName, "normType is nullptr"), return ge::GRAPH_FAILED);
    int64_t normType = *context_.normType;
    OP_CHECK_IF(!IsNormTypeValid(normType), OP_LOGE(context_.opName, "normType is out of range"),
                return ge::GRAPH_FAILED);
    tilingKey_ += normType * NORM_TYPE_TILING_KEY;
    OP_CHECK_IF(context_.normAddedType == nullptr, OP_LOGE(context_.opName, "normAddedType is nullptr"),
                return ge::GRAPH_FAILED);
    int64_t normAddedType = *context_.normAddedType;
    OP_CHECK_IF(!IsNormTypeValid(normAddedType), OP_LOGE(context_.opName, "normAddedType is out of range"),
                return ge::GRAPH_FAILED);
    tilingKey_ += normAddedType * NORM_ADDED_TYPE_TILING_KEY;
    OP_CHECK_IF(context_.ropeType == nullptr, OP_LOGE(context_.opName, "ropeType is nullptr"), return ge::GRAPH_FAILED);
    int64_t ropeType = *context_.ropeType;
    OP_CHECK_IF(!IsRopeTypeValid(ropeType), OP_LOGE(context_.opName, "ropeType is out of range"),
                return ge::GRAPH_FAILED);

    tilingKey_ += ropeType * ROPE_TYPE_TILING_KEY;
    OP_CHECK_IF(context_.concatOrder == nullptr, OP_LOGE(context_.opName, "concatOrder is nullptr"),
                return ge::GRAPH_FAILED);
    int64_t concatOrder = *context_.concatOrder;
    OP_CHECK_IF(!IsConcatOrderValid(concatOrder), OP_LOGE(context_.opName, "concatOrder is out of range"),
                return ge::GRAPH_FAILED);
    tilingKey_ += concatOrder * CONCAT_ORDER_TILING_KEY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::ComputeWorkSpace()
{
    size_t workspaceSize = compileInfo_.sysWorkspace;

    uint64_t extraWorkspace = 0;
    if (*(context_.normType) == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE) ||
        *(context_.normType) == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE_ACROSS_HEADS) ||
        *(context_.normAddedType) == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE) ||
        *(context_.normAddedType) == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE_ACROSS_HEADS)) {
        extraWorkspace +=
            TOTAL_NORM_NUM * usedCore_ * CeilAlign(normDim_, static_cast<uint64_t>(BLOCK_SIZE / sizeof(float)));
        extraWorkspace += MAX_AFFINE_BLOCK_NUM * usedCore_ * std::max(avgHeads_, static_cast<uint64_t>(NUM_TWO)) *
                          NUM_TWO * CeilAlign(normDim_, static_cast<uint64_t>(BLOCK_SIZE / sizeof(float)));
    }
    workspaceSize += static_cast<size_t>(extraWorkspace * sizeof(float));

    workspace_[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::FillTilingData()
{
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_batch(batchSize_);
    tilingData_.set_querySeq(querySeq_);
    tilingData_.set_keySeq(keySeq_);
    tilingData_.set_valueSeq(valueSeq_);
    tilingData_.set_encoderQuerySeq(encoderQuerySeq_);
    tilingData_.set_encoderKeySeq(encoderKeySeq_);
    tilingData_.set_encoderValueSeq(encoderValueSeq_);
    tilingData_.set_totalQuerySeq(totalQuerySeq_);
    tilingData_.set_totalKeySeq(totalKeySeq_);
    tilingData_.set_totalValueSeq(totalValueSeq_);
    tilingData_.set_ropeActualSeq(ropeSeq_);
    tilingData_.set_splitHeadNum(splitHeadNum_);
    tilingData_.set_avgHeads(avgHeads_);
    tilingData_.set_tailHeads(tailHeads_);
    tilingData_.set_normDim(normDim_);
    tilingData_.set_ropeDim(ropeDim_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_headNum(headNum_);
    tilingData_.set_usedCore(usedCore_);
    tilingData_.set_eps(eps_);
    tilingData_.set_scale(scale_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::PrintTilingData(gert::TilingContext* ctx)
{
    OP_LOGD(ctx->GetNodeName(), "tilingKey:           %u.", tilingData_.get_tilingKey());
    OP_LOGD(ctx->GetNodeName(), "batch:               %u.", tilingData_.get_batch());
    OP_LOGD(ctx->GetNodeName(), "querySeq:            %u.", tilingData_.get_querySeq());
    OP_LOGD(ctx->GetNodeName(), "keySeq:              %u.", tilingData_.get_keySeq());
    OP_LOGD(ctx->GetNodeName(), "valueSeq:            %u.", tilingData_.get_valueSeq());
    OP_LOGD(ctx->GetNodeName(), "encoderQuerySeq:     %u.", tilingData_.get_encoderQuerySeq());
    OP_LOGD(ctx->GetNodeName(), "encoderKeySeq:       %u.", tilingData_.get_encoderKeySeq());
    OP_LOGD(ctx->GetNodeName(), "encoderValueSeq:     %u.", tilingData_.get_encoderValueSeq());
    OP_LOGD(ctx->GetNodeName(), "totalQuerySeq:       %u.", tilingData_.get_totalQuerySeq());
    OP_LOGD(ctx->GetNodeName(), "totalKeySeq:         %u.", tilingData_.get_totalKeySeq());
    OP_LOGD(ctx->GetNodeName(), "totalValueSeq:       %u.", tilingData_.get_totalValueSeq());
    OP_LOGD(ctx->GetNodeName(), "ropeActualSeq:       %u.", tilingData_.get_ropeActualSeq());
    OP_LOGD(ctx->GetNodeName(), "headNum:             %u.", tilingData_.get_headNum());
    OP_LOGD(ctx->GetNodeName(), "splitHeadNum:        %u.", tilingData_.get_splitHeadNum());
    OP_LOGD(ctx->GetNodeName(), "avgHeads:            %u.", tilingData_.get_avgHeads());
    OP_LOGD(ctx->GetNodeName(), "tailHeads:           %u.", tilingData_.get_tailHeads());
    OP_LOGD(ctx->GetNodeName(), "normDim:             %u.", tilingData_.get_normDim());
    OP_LOGD(ctx->GetNodeName(), "ropeDim:             %u.", tilingData_.get_ropeDim());
    OP_LOGD(ctx->GetNodeName(), "headDim:             %u.", tilingData_.get_headDim());
    OP_LOGD(ctx->GetNodeName(), "usedCore:            %u.", tilingData_.get_usedCore());
    OP_LOGD(ctx->GetNodeName(), "eps:                 %f.", tilingData_.get_eps());
    OP_LOGD(ctx->GetNodeName(), "scale:               %f.", tilingData_.get_scale());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::ConvertContextOne(gert::TilingContext* ctx, InputTensorInfo& x)
{
    size_t index = x.index;
    bool isRequired = x.isRequired;
    if (isRequired) {
        x.desc = ctx->GetInputDesc(index);
        x.shape = ctx->GetInputShape(index);
    } else {
        x.desc = ctx->GetOptionalInputDesc(index);
        x.shape = ctx->GetOptionalInputShape(index);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::ComputeCoreTilingStrategy()
{
    uint64_t maxSeq = std::max(querySeq_, keySeq_);
    uint64_t maxEncoderSeq = std::max(encoderQuerySeq_, encoderKeySeq_);
    uint64_t maxCoreSeq = std::max(maxSeq, maxEncoderSeq);
    usedCore_ = std::min(maxCoreSeq, compileInfo_.aivNum);
    numBlocks_ = usedCore_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::ComputeUBTilingStrategy()
{
    uint64_t dataTypeSize = inputDtype_ == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    uint64_t alignNum = BLOCK_SIZE / dataTypeSize;
    uint64_t alignedNormDim_ = CeilAlign(normDim_, alignNum);
    uint64_t alignedRopeDim_ = CeilAlign(ropeDim_, alignNum);
    uint64_t normCoef = *(context_.normType) == static_cast<int64_t>(NormType::NONE) &&
                                *(context_.normAddedType) == static_cast<int64_t>(NormType::NONE) ?
                            0 :
                            1;
    uint64_t ropeCoef = *(context_.ropeType) == static_cast<int64_t>(RopeType::NONE) ? 0 : 1;
    uint64_t ropeUsedUbSize = SINGLE_BUFFER * alignedRopeDim_ * 2 * dataTypeSize + alignedRopeDim_ * sizeof(int32_t) +
                              alignedRopeDim_ * 2 * sizeof(float); // mask, sin, cos
    uint64_t normUsedUbSize =
        SINGLE_BUFFER * alignedNormDim_ * dataTypeSize // weight que
        + NUM_TWO * DOUBLE_BUFFER * alignedNormDim_ * sizeof(float) +
        NUM_TWO * alignedNormDim_ * sizeof(float) // gradWeight & gradBias que + gradWeight & gradBias buf
        + SINGLE_BUFFER * alignedNormDim_ * sizeof(float) + alignedNormDim_ * dataTypeSize +
        alignedNormDim_ * sizeof(float) // workspace get&push que, workspace buf
        + MIN_SHARE_BUFFER;             // extra shareCast buf
    uint64_t oneHeadUbSize =
        NUM_THREE * alignedNormDim_ * sizeof(float) +
        NUM_TWO * DOUBLE_BUFFER * alignedNormDim_ *
            dataTypeSize + // grad_ouput & tempND & & shareCast buf + grad_ouput & grad_in que
        normCoef * (NUM_TWO * DOUBLE_BUFFER + NUM_TWO) * sizeof(float) + // mean & rstd buf + mean & rstd que
        normCoef * (NUM_THREE * alignedNormDim_ * sizeof(float) +
                    DOUBLE_BUFFER * alignedNormDim_ * dataTypeSize); // input & share & weight buf + input que

    avgHeads_ =
        (compileInfo_.ubSize - ropeCoef * ropeUsedUbSize - normCoef * normUsedUbSize - RES_UB_BUFFER) / oneHeadUbSize;
    avgHeads_ = std::min(avgHeads_, static_cast<uint64_t>(AVGHEAD_MAX_NUM));
    OP_CHECK_IF(avgHeads_ < 2, OP_LOGE(context_.opName, "input dim is too large, avgHeads_ should larger than 1"),
                return ge::GRAPH_FAILED);
    avgHeads_ = std::min(avgHeads_, headNum_);
    OP_CHECK_IF(avgHeads_ == 0, OP_LOGE(context_.opName, "input dim is too large"), return ge::GRAPH_FAILED);
    splitHeadNum_ = CeilDiv(headNum_, avgHeads_);
    tailHeads_ = headNum_ - (splitHeadNum_ - 1) * avgHeads_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::DoTiling(gert::TilingContext* ctx)
{
    auto platformInfoPtr = ctx->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("NormRopeConcatGradTiling", "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_CHECK_IF(ctx->GetAttrs() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(ctx->GetNodeName(), "attrs got from GE is nullptr"),
                return ge::GRAPH_FAILED);

    auto attrs = ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(ctx->GetNodeName(), "attrs got from GE is nullptr"), return ge::GRAPH_FAILED);

    context_.opName = ctx->GetNodeName();
    for (size_t i = 0; i < context_.requiredInputNum; ++i) {
        context_.inputs[i].index = i;
        context_.inputs[i].isRequired = true;
        context_.inputs[i].type = BACKWARD_INPUT_TYPE[i];
        ConvertContextOne(ctx, context_.inputs[i]);
    }
    inputDtype_ = context_.inputs[0].desc->GetDataType();
    for (size_t i = context_.requiredInputNum; i < context_.maxInputNum; ++i) {
        context_.inputs[i].index = i;
        context_.inputs[i].isRequired = false;
        context_.inputs[i].type = BACKWARD_INPUT_TYPE[i];
        ConvertContextOne(ctx, context_.inputs[i]);
    }

    // common attrs
    size_t normTypeIndex = static_cast<size_t>(AttrIndexBackward::NORM_TYPE_INDEX);
    size_t normAddedTypeIndex = static_cast<size_t>(AttrIndexBackward::NORM_ADDED_TYPE_INDEX);
    size_t ropeTypeIndex = static_cast<size_t>(AttrIndexBackward::ROPE_TYPE_INDEX);
    size_t concatOrderIndex = static_cast<size_t>(AttrIndexBackward::CONCAT_ORDER_INDEX);

    context_.normType = attrs->GetAttrPointer<int64_t>(normTypeIndex);
    context_.normAddedType = attrs->GetAttrPointer<int64_t>(normAddedTypeIndex);
    context_.ropeType = attrs->GetAttrPointer<int64_t>(ropeTypeIndex);
    context_.concatOrder = attrs->GetAttrPointer<int64_t>(concatOrderIndex);
    OP_CHECK_IF(context_.normType == nullptr, OP_LOGE(context_.opName, "normType is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_.normAddedType == nullptr, OP_LOGE(context_.opName, "normAddedType is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_.ropeType == nullptr, OP_LOGE(context_.opName, "ropeType is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_.concatOrder == nullptr, OP_LOGE(context_.opName, "concatOrder is nullptr"),
                return ge::GRAPH_FAILED);

    size_t* workspaceSizes = ctx->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaceSizes == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(ctx->GetNodeName(), "workSpaceSize got from GE is nullptr"),
                return ge::GRAPH_FAILED);
    workspace_ = workspaceSizes;

    if (GetShapeInfo() != ge::GRAPH_SUCCESS || CheckInput() != ge::GRAPH_SUCCESS ||
        ComputeTilingKey() != ge::GRAPH_SUCCESS || ComputeCoreTilingStrategy() != ge::GRAPH_SUCCESS ||
        ComputeUBTilingStrategy() != ge::GRAPH_SUCCESS || ComputeWorkSpace() != ge::GRAPH_SUCCESS ||
        FillTilingData() != ge::GRAPH_SUCCESS || PrintTilingData(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ctx->SetTilingKey(tilingKey_);
    ctx->SetBlockDim(numBlocks_);
    if (tilingKey_ >= BATCH_MODE_MIN_KEY) {
        ctx->SetScheduleMode(1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatGradTiling::SetTiling(gert::TilingContext* ctx)
{
    OP_CHECK_IF(ctx->GetRawTilingData() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(ctx->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return GRAPH_FAILED);
    tilingData_.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingNormRopeConcatGrad(gert::TilingContext* ctx)
{
    OP_CHECK_IF(ctx == nullptr, OPS_REPORT_VECTOR_INNER_ERR("NormRopeConcatGrad", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    NormRopeConcatGradTiling tiling;
    if (tiling.DoTiling(ctx) == ge::SUCCESS) {
        tiling.SetTiling(ctx);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingPrepareForNormRopeConcatGrad(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("TilingPrepareForNormRopeConcatGrad", "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<NormRopeConcatGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE("TilingPrepareForNormRopeConcatGrad", "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    compileInfoPtr->sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NormRopeConcatGrad)
    .Tiling(TilingNormRopeConcatGrad)
    .TilingParse<NormRopeConcatGradCompileInfo>(TilingPrepareForNormRopeConcatGrad);

} // namespace optiling
