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
 * \file norm_rope_concat_tiling.cpp
 * \brief
 */

#include <string>
#include "norm_rope_concat_tiling.h"

using namespace ge;
using namespace nrc;
using namespace AscendC;
namespace optiling {
// 合法输入下 Tiling 失败时，按 EZ1008 正式错误码首报（Arglist: op_name, reason），并输出 PLOG 上下文
inline void ReportTilingFailure(const char* opName, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(opName, "%s\n", reason);
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgvalue = {opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgvalue);
}

// 将 StorageShape 渲染为 "[d0,d1,...]" 形式，供错误码消息的 incorrect_shape 字段使用
static std::string ShapeToString(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        result += (i == 0 ? "" : ",") + std::to_string(shape.GetDim(i));
    }
    return result + "]";
}

static std::string ExpectedShapeToString(const std::vector<int64_t>& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += (i == 0 ? "" : ",") + std::to_string(shape[i]);
    }
    return result + "]";
}

static inline bool IsLayerNormType(int64_t normType)
{
    return normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
           normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE);
}

// 优先使用origin shape（调用方声明的逻辑shape）；未知场景下回退到storage shape
static const gert::Shape& GetCheckedShape(const gert::StorageShape& shape)
{
    const gert::Shape& originShape = shape.GetOriginShape();
    return originShape.GetDimNum() > 0 ? originShape : shape.GetStorageShape();
}

ge::graphStatus NormRopeConcatTiling::CheckInput(const InputTensorInfo& x)
{
    if (x.isRequired) {
        OP_CHECK_IF(x.desc == nullptr, OP_LOGE(context_.opName, "desc is nullptr"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(x.shape == nullptr, OP_LOGE(context_.opName, "shape is nullptr"), return ge::GRAPH_FAILED);
    }
    if (x.desc != nullptr) {
        OP_CHECK_IF(
            x.desc->GetDataType() != inputDtype_,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_.opName, FORWARD_INPUT_NAME[x.index],
                                                  ge::TypeUtils::DataTypeToSerialString(x.desc->GetDataType()).c_str(),
                                                  "the dtype of this input must be the same as query"),
            return ge::GRAPH_FAILED);
    }
    if (x.shape != nullptr) {
        auto shape = x.shape->GetStorageShape();
        switch (x.type) {
            case TensorType::INPUT_TENSOR:
            case TensorType::ENCODER_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != INPUT_DIM_NUM,
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_.opName, FORWARD_INPUT_NAME[x.index],
                                                                     std::to_string(shape.GetDimNum()).c_str(),
                                                                     "the shape dim of this input must be 4"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(BATCH_DIM) != batchSize_,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context_.opName, FORWARD_INPUT_NAME[x.index], ShapeToString(shape).c_str(),
                                "the batch dim of this input must be the same as query"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(HEAD_DIM) != headNum_,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context_.opName, FORWARD_INPUT_NAME[x.index], ShapeToString(shape).c_str(),
                                "the head dim of this input must be the same as query"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(DIM_DIM) != headDim_,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context_.opName, FORWARD_INPUT_NAME[x.index], ShapeToString(shape).c_str(),
                                "the last dim of this input must be the same as query"),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::NORM_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != 1,
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_.opName, FORWARD_INPUT_NAME[x.index],
                                                                     std::to_string(shape.GetDimNum()).c_str(),
                                                                     "the shape dim of this input must be 1"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(0) != headDim_, // NOTE: not support cross heads
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context_.opName, FORWARD_INPUT_NAME[x.index], ShapeToString(shape).c_str(),
                                "the dim of this input must equal the head dim of query"),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::ROPE_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != ROPE_DIM_NUM,
                            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_.opName, FORWARD_INPUT_NAME[x.index],
                                                                     std::to_string(shape.GetDimNum()).c_str(),
                                                                     "the shape dim of this input must be 2"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_SEQ_DIM) > maxSeq_,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context_.opName, FORWARD_INPUT_NAME[x.index], ShapeToString(shape).c_str(),
                                "the seq dim of this input must not exceed the max sequence length"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_DIM_DIM) != headDim_,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                context_.opName, FORWARD_INPUT_NAME[x.index], ShapeToString(shape).c_str(),
                                "the dim of this input must equal the head dim of query"),
                            return ge::GRAPH_FAILED);
                break;
            default:
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::CheckInput()
{
    OP_CHECK_IF((headDim_ % 2 != 0) || (headDim_ > 1024),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "query", std::to_string(headDim_).c_str(),
                                                      "the last dim of query must be even and not greater than 1024"),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < context_.maxInputNum; ++i) {
        if (CheckInput(context_.inputs[i]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(keySeq_ != valueSeq_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_.opName, "key,value",
                    ("key seq=" + std::to_string(keySeq_) + ",value seq=" + std::to_string(valueSeq_)).c_str(),
                    "key and value must have the same seq length"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(encoderKeySeq_ != encoderValueSeq_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_.opName, "encoder_key,encoder_value",
                                                       ("encoder_key seq=" + std::to_string(encoderKeySeq_) +
                                                        ",encoder_value seq=" + std::to_string(encoderValueSeq_))
                                                           .c_str(),
                                                       "encoder_key and encoder_value must have the same seq length"),
                return ge::GRAPH_FAILED);

    normDim_ = headDim_;
    OP_CHECK_IF(context_.eps == nullptr, OP_LOGE(context_.opName, "eps is nullptr"), return ge::GRAPH_FAILED);
    eps_ = *(context_.eps);
    scale_ = 1 / float(headDim_);
    int64_t normType = *context_.normType;
    OP_CHECK_IF(!IsNormTypeValid(normType),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "norm_type", std::to_string(normType).c_str(),
                                                      "the value of norm_type is out of the supported range"),
                return ge::GRAPH_FAILED);
    int64_t normAddedType = *context_.normAddedType;
    OP_CHECK_IF(
        !IsNormTypeValid(normAddedType),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "norm_added_type", std::to_string(normAddedType).c_str(),
                                              "the value of norm_added_type is out of the supported range"),
        return ge::GRAPH_FAILED);
    int64_t ropeType = *context_.ropeType;
    OP_CHECK_IF(!IsRopeTypeValid(ropeType),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "rope_type", std::to_string(ropeType).c_str(),
                                                      "the value of rope_type is out of the supported range"),
                return ge::GRAPH_FAILED);
    int64_t concatOrder = *context_.concatOrder;
    OP_CHECK_IF(
        !IsConcatOrderValid(concatOrder),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_.opName, "concat_order", std::to_string(concatOrder).c_str(),
                                              "the value of concat_order is out of the supported range"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::CheckOutputShape(gert::TilingContext* ctx, size_t index, const char* name,
                                                       const std::vector<int64_t>& expectShape)
{
    const gert::StorageShape* shape = ctx->GetOutputShape(index);
    if (shape == nullptr) {
        OP_LOGW(context_.opName, "[NormRopeConcat] output %s shape is nullptr, expected %s, skip shape check.", name,
                ExpectedShapeToString(expectShape).c_str());
        return ge::GRAPH_SUCCESS;
    }
    // kernel按输入推导出的形状连续写出，使用origin shape与推导值比对
    const gert::Shape& originShape = GetCheckedShape(*shape);
    bool matched = originShape.GetDimNum() == expectShape.size();
    for (size_t i = 0; matched && i < expectShape.size(); ++i) {
        // 动态shape场景下未知维(-1)不参与比较
        if (originShape.GetDim(i) < 0 || expectShape[i] < 0) {
            continue;
        }
        matched = originShape.GetDim(i) == expectShape[i];
    }
    if (!matched) {
        OP_LOGW(context_.opName,
                "[NormRopeConcat] output %s shape %s does not match the shape derived from inputs %s, "
                "kernel will still write by the derived shape.",
                name, ShapeToString(originShape).c_str(), ExpectedShapeToString(expectShape).c_str());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::CheckStatsOutput(gert::TilingContext* ctx, size_t index, const char* name,
                                                       const std::vector<int64_t>& expectShape)
{
    const gert::StorageShape* shape = ctx->GetOutputShape(index);
    // 可选输出未提供时shape为空，此种调用方式保持可用，不参与校验
    if (shape == nullptr || GetCheckedShape(*shape).GetDimNum() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    (void)CheckOutputShape(ctx, index, name, expectShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::CheckOutput(gert::TilingContext* ctx)
{
    const size_t queryOutput = static_cast<size_t>(OutputIndexForward::QUERY_OUTPUT_INDEX);
    const size_t keyOutput = static_cast<size_t>(OutputIndexForward::KEY_OUTPUT_INDEX);
    const size_t valueOutput = static_cast<size_t>(OutputIndexForward::VALUE_OUTPUT_INDEX);
    const std::vector<int64_t> queryOutputShape{batchSize_, headNum_, totalQuerySeq_, headDim_};
    const std::vector<int64_t> keyOutputShape{batchSize_, headNum_, totalKeySeq_, headDim_};
    const std::vector<int64_t> valueOutputShape{batchSize_, headNum_, totalValueSeq_, headDim_};
    (void)CheckOutputShape(ctx, queryOutput, "query_output", queryOutputShape);
    (void)CheckOutputShape(ctx, keyOutput, "key_output", keyOutputShape);
    (void)CheckOutputShape(ctx, valueOutput, "value_output", valueOutputShape);

    const bool isTraining = *(context_.isTraining);
    // LayerNorm且训练模式下kernel才会输出均值/标准差倒数，其余场景不校验，保持已有调用方式不受影响
    if (isTraining && IsLayerNormType(*(context_.normType))) {
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_QUERY_MEAN_INDEX), "norm_query_mean",
                               {batchSize_, querySeq_, headNum_, 1});
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_QUERY_RSTD_INDEX), "norm_query_rstd",
                               {batchSize_, querySeq_, headNum_, 1});
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_KEY_MEAN_INDEX), "norm_key_mean",
                               {batchSize_, keySeq_, headNum_, 1});
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_KEY_RSTD_INDEX), "norm_key_rstd",
                               {batchSize_, keySeq_, headNum_, 1});
    }
    if (isTraining && IsLayerNormType(*(context_.normAddedType))) {
        const std::vector<int64_t> addedQueryStatsShape{batchSize_, encoderQuerySeq_, headNum_, 1};
        const std::vector<int64_t> addedKeyStatsShape{batchSize_, encoderKeySeq_, headNum_, 1};
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_MEAN_INDEX),
                               "norm_added_query_mean", addedQueryStatsShape);
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_ADDED_QUERY_RSTD_INDEX),
                               "norm_added_query_rstd", addedQueryStatsShape);
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_MEAN_INDEX),
                               "norm_added_key_mean", addedKeyStatsShape);
        (void)CheckStatsOutput(ctx, static_cast<size_t>(OutputIndexForward::NORM_ADDED_KEY_RSTD_INDEX),
                               "norm_added_key_rstd", addedKeyStatsShape);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::GetShapeInfo()
{
    size_t queryIndex = static_cast<size_t>(InputIndexForward::QUERY_INDEX);
    size_t keyIndex = static_cast<size_t>(InputIndexForward::KEY_INDEX);
    const gert::StorageShape* queryShape = context_.inputs[queryIndex].shape;
    const gert::StorageShape* keyShape = context_.inputs[keyIndex].shape;
    OP_CHECK_IF(queryShape == nullptr || queryShape->GetStorageShape().GetDimNum() != INPUT_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_.opName, "query",
                    std::to_string(queryShape == nullptr ? 0 : queryShape->GetStorageShape().GetDimNum()).c_str(),
                    "the shape dim of query must be 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape == nullptr || keyShape->GetStorageShape().GetDimNum() != INPUT_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_.opName, "key",
                    std::to_string(keyShape == nullptr ? 0 : keyShape->GetStorageShape().GetDimNum()).c_str(),
                    "the shape dim of key must be 4"),
                return ge::GRAPH_FAILED);

    batchSize_ = queryShape->GetStorageShape().GetDim(BATCH_DIM);
    headNum_ = queryShape->GetStorageShape().GetDim(HEAD_DIM);
    headDim_ = queryShape->GetStorageShape().GetDim(DIM_DIM);
    querySeq_ = queryShape->GetStorageShape().GetDim(SEQ_DIM);
    keySeq_ = keyShape->GetStorageShape().GetDim(SEQ_DIM);
    valueSeq_ = keySeq_;
    size_t valueIndex = static_cast<size_t>(InputIndexForward::VALUE_INDEX);
    const gert::StorageShape* valueShape = context_.inputs[valueIndex].shape;
    if (valueShape != nullptr && valueShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        valueSeq_ = valueShape->GetStorageShape().GetDim(SEQ_DIM);
    }

    size_t encoderQueryIndex = static_cast<size_t>(InputIndexForward::ENCODER_QUERY_INDEX);
    size_t encoderKeyIndex = static_cast<size_t>(InputIndexForward::ENCODER_KEY_INDEX);
    const gert::StorageShape* encoderQueryShape = context_.inputs[encoderQueryIndex].shape;
    const gert::StorageShape* encoderKeyShape = context_.inputs[encoderKeyIndex].shape;
    if (encoderQueryShape != nullptr && encoderQueryShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderQuerySeq_ = encoderQueryShape->GetStorageShape().GetDim(SEQ_DIM);
    }
    if (encoderKeyShape != nullptr && encoderKeyShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderKeySeq_ = encoderKeyShape->GetStorageShape().GetDim(SEQ_DIM);
    }
    encoderValueSeq_ = encoderKeySeq_;
    size_t encoderValueIndex = static_cast<size_t>(InputIndexForward::ENCODER_VALUE_INDEX);
    const gert::StorageShape* encoderValueShape = context_.inputs[encoderValueIndex].shape;
    if (encoderValueShape != nullptr && encoderValueShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderValueSeq_ = encoderValueShape->GetStorageShape().GetDim(SEQ_DIM);
    }

    size_t ropeSinIndex = static_cast<size_t>(InputIndexForward::ROPE_SIN_INDEX);
    const gert::StorageShape* ropeSinShape = context_.inputs[ropeSinIndex].shape;
    if (ropeSinShape != nullptr && ropeSinShape->GetStorageShape().GetDimNum() == ROPE_DIM_NUM) {
        ropeSeq_ = ropeSinShape->GetStorageShape().GetDim(ROPE_SEQ_DIM);
        ropeDim_ = ropeSinShape->GetStorageShape().GetDim(ROPE_DIM_DIM);
    } else if (*context_.ropeType == static_cast<int64_t>(RopeType::NONE)) {
        // ropeType=0时不传ropeSin/ropeCos，ropeDim默认0会导致value等通路搬运0字节，此处按headDim补齐
        ropeDim_ = headDim_;
    }

    totalQuerySeq_ = querySeq_ + encoderQuerySeq_;
    totalKeySeq_ = keySeq_ + encoderKeySeq_;
    totalValueSeq_ = valueSeq_ + encoderValueSeq_;

    maxSeq_ = std::max({totalQuerySeq_, totalKeySeq_, totalValueSeq_});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeTilingKey()
{
    tilingKey_ =
        GET_TPL_TILING_KEY(static_cast<uint64_t>(*context_.normType), static_cast<uint64_t>(*context_.normAddedType),
                           static_cast<uint64_t>(*context_.ropeType), static_cast<uint64_t>(*context_.concatOrder),
                           static_cast<uint64_t>(*context_.isTraining));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeWorkSpace()
{
    size_t workspaceSize = compileInfo_.sysWorkspace;
    workspace_[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::FillTilingData()
{
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

ge::graphStatus NormRopeConcatTiling::PrintTilingData(gert::TilingContext* ctx)
{
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
    OP_LOGD(ctx->GetNodeName(), "splitHeadNum:        %u.", tilingData_.get_splitHeadNum());
    OP_LOGD(ctx->GetNodeName(), "avgHeads:            %u.", tilingData_.get_avgHeads());
    OP_LOGD(ctx->GetNodeName(), "tailHeads:           %u.", tilingData_.get_tailHeads());
    OP_LOGD(ctx->GetNodeName(), "normDim:             %u.", tilingData_.get_normDim());
    OP_LOGD(ctx->GetNodeName(), "ropeDim:             %u.", tilingData_.get_ropeDim());
    OP_LOGD(ctx->GetNodeName(), "headDim:             %u.", tilingData_.get_headDim());
    OP_LOGD(ctx->GetNodeName(), "headNum:             %u.", tilingData_.get_headNum());
    OP_LOGD(ctx->GetNodeName(), "usedCore:            %u.", tilingData_.get_usedCore());
    OP_LOGD(ctx->GetNodeName(), "eps:                 %f.", tilingData_.get_eps());
    OP_LOGD(ctx->GetNodeName(), "scale:               %f.", tilingData_.get_scale());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ConvertContextOne(gert::TilingContext* ctx, InputTensorInfo& x)
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

ge::graphStatus NormRopeConcatTiling::ComputeCoreTilingStrategy()
{
    uint32_t maxSeq = std::max(querySeq_, keySeq_);
    uint32_t maxEncoderSeq = std::max(encoderQuerySeq_, encoderKeySeq_);
    uint32_t maxCoreSeq = std::max(maxSeq, maxEncoderSeq);
    usedCore_ = std::min(maxCoreSeq, compileInfo_.aivNum);

    numBlocks_ = usedCore_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeUBTilingStrategy()
{
    int64_t dataTypeSize = inputDtype_ == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    int64_t alignNum = BLOCK_SIZE / dataTypeSize;
    int64_t alignedNormDim_ = CeilAlign(normDim_, alignNum);
    int64_t alignedRopeDim_ = CeilAlign(ropeDim_, alignNum);
    int64_t normCoef = *(context_.normType) == static_cast<int64_t>(NormType::NONE) &&
                               *(context_.normAddedType) == static_cast<int64_t>(NormType::NONE) ?
                           0 :
                           1;
    int64_t ropeCoef = *(context_.ropeType) == static_cast<int64_t>(RopeType::NONE) ? 0 : 1;
    int64_t trainingCoef = *(context_.isTraining) ? 1 : 0;
    int64_t ropeUsedUbSize = SINGLE_BUFFER * alignedRopeDim_ * 2 * dataTypeSize + alignedRopeDim_ * sizeof(int32_t) +
                             alignedRopeDim_ * 2 * sizeof(float); // mask, sin, cos, max(20480)
    int64_t normUsedUbSize = SINGLE_BUFFER * alignedNormDim_ * 2 * dataTypeSize + alignedNormDim_ * 2 * sizeof(float) +
                             MIN_SHARE_BUFFER; // weight, bias, max(16384)
    int64_t oneHeadUbSize =
        2 * alignedNormDim_ * sizeof(float) + 2 * DOUBLE_BUFFER * alignedNormDim_ * dataTypeSize +
        ropeCoef * ropeUsedUbSize +
        normCoef * (alignedNormDim_ * sizeof(float) + (trainingCoef * 2 * DOUBLE_BUFFER + 1) * sizeof(float));
    avgHeads_ = (compileInfo_.ubSize - ropeCoef * ropeUsedUbSize - normCoef * normUsedUbSize) / oneHeadUbSize;
    avgHeads_ = std::min(avgHeads_, static_cast<int64_t>(AVGHEAD_MAX_NUM));
    avgHeads_ = std::min(avgHeads_, headNum_);
    OP_CHECK_IF(
        avgHeads_ == 0,
        ReportTilingFailure(context_.opName,
                            ("the input dim is too large to be tiled, headDim=" + std::to_string(headDim_)).c_str()),
        return ge::GRAPH_FAILED);
    splitHeadNum_ = CeilDiv(headNum_, avgHeads_);
    tailHeads_ = headNum_ - (splitHeadNum_ - 1) * avgHeads_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::DoTiling(gert::TilingContext* ctx)
{
    auto platformInfoPtr = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto attrs = ctx->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);

    context_.opName = ctx->GetNodeName();
    for (size_t i = 0; i < context_.requiredInputNum; ++i) {
        context_.inputs[i].index = i;
        context_.inputs[i].isRequired = true;
        context_.inputs[i].type = FORWARD_INPUT_TYPE[i];
        ConvertContextOne(ctx, context_.inputs[i]);
    }
    inputDtype_ = context_.inputs[0].desc->GetDataType();
    for (size_t i = context_.requiredInputNum; i < context_.maxInputNum; ++i) {
        context_.inputs[i].index = i;
        context_.inputs[i].isRequired = false;
        context_.inputs[i].type = FORWARD_INPUT_TYPE[i];
        ConvertContextOne(ctx, context_.inputs[i]);
    }
    // attrs
    context_.eps = attrs->GetAttrPointer<float>(static_cast<size_t>(AttrIndexForward::EPS_INDEX));
    context_.isTraining = attrs->GetAttrPointer<bool>(static_cast<size_t>(AttrIndexForward::IS_TRAINING_INDEX));
    context_.normType = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::NORM_TYPE_INDEX));
    context_.normAddedType =
        attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::NORM_ADDED_TYPE_INDEX));
    context_.ropeType = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::ROPE_TYPE_INDEX));
    context_.concatOrder = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::CONCAT_ORDER_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.eps);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.isTraining);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.normType);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.normAddedType);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.ropeType);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.concatOrder);

    size_t* workspaceSizes = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, workspaceSizes);
    workspace_ = workspaceSizes;

    if (GetShapeInfo() != ge::GRAPH_SUCCESS || CheckInput() != ge::GRAPH_SUCCESS ||
        CheckOutput(ctx) != ge::GRAPH_SUCCESS || ComputeTilingKey() != ge::GRAPH_SUCCESS ||
        ComputeCoreTilingStrategy() != ge::GRAPH_SUCCESS || ComputeUBTilingStrategy() != ge::GRAPH_SUCCESS ||
        ComputeWorkSpace() != ge::GRAPH_SUCCESS || FillTilingData() != ge::GRAPH_SUCCESS ||
        PrintTilingData(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ctx->SetTilingKey(tilingKey_);
    ctx->SetBlockDim(numBlocks_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::SetTiling(gert::TilingContext* ctx)
{
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ctx->GetRawTilingData());
    tilingData_.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingNormRopeConcat(gert::TilingContext* ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE("NormRopeConcat", "context is nullptr"), return ge::GRAPH_FAILED);
    NormRopeConcatTiling tiling;
    if (tiling.DoTiling(ctx) == ge::SUCCESS) {
        tiling.SetTiling(ctx);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNormRopeConcat(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<NormRopeConcatCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    compileInfoPtr->sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NormRopeConcat)
    .Tiling(TilingNormRopeConcat)
    .TilingParse<NormRopeConcatCompileInfo>(TilingPrepareForNormRopeConcat);
} // namespace optiling
