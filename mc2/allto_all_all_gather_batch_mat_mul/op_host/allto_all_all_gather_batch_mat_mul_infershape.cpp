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
 * \file allto_all_all_gather_batch_mat_mul_infershape.cpp
 * \brief
 */
#include <algorithm>

#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "common/utils/op_mc2.h"
#include "mc2_moe_utils.h"

using namespace ge;
using namespace Mc2Moe;
namespace {
const char* K_INNER_DEBUG = "MC2: AlltoAllAllGatherBmm Infershape Debug";

// for y2
void DynamicEmptyCheck(const gert::Shape* xShape, const gert::Shape* weightShape, const size_t wDimH,
                       OutShapeInfo& outShapeInfo)
{
    // dynamic
    if (xShape->GetDim(DIM_E) == -1) {
        outShapeInfo.e = -1;
    }
    if (xShape->GetDim(X_DIM_C) == -1) {
        outShapeInfo.c = -1;
    }
    if ((xShape->GetDim(X_DIM_H) == -1) || (weightShape->GetDim(wDimH) == -1)) {
        outShapeInfo.h = -1;
    }

    // empty
    if ((xShape->GetDim(DIM_E) == 0) || (weightShape->GetDim(DIM_E) == 0)) {
        outShapeInfo.e = -1;
    }
    if (xShape->GetDim(X_DIM_C) == 0) {
        outShapeInfo.c = 0;
    }

    return;
}

// 判断是否输出可选输出, 不输出时设置 shape 为 [0, 0, 0]
void OutputFlagCheck(const char* nodeName, const bool flag, const char* outName, OutShapeInfo& outShapeInfo)
{
    if (!flag) {
        outShapeInfo.e = 0;
        outShapeInfo.c = 0;
        outShapeInfo.h = 0;
        OPS_LOG_I(nodeName, "[%s] does not output, infer shape to [0, 0, 0].", outName);
    }
    return;
}

bool ActTypeCheck(const char* nodeName, const int64_t actType, const bool y3Flag)
{
    if (std::find(ops::ACT_TYPE_SUPPORT_VEC.begin(), ops::ACT_TYPE_SUPPORT_VEC.end(), actType) ==
        ops::ACT_TYPE_SUPPORT_VEC.end()) {
        OPS_LOG_E(nodeName, "actType [%ld] is unsupported.", actType);
        return false;
    }

    if (y3Flag &&
        (actType == static_cast<int64_t>(
                        ops::AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE))) {
        OPS_LOG_E(nodeName, "actType should be non-none when need_activation_feature is true, but got actType = none.");
        return false;
    }

    return true;
}

bool CommonCheckTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                            const size_t wDimH)
{
    // 检查每个维度: x dim C >= 1，dim E H M 会在后面限制, 这里不再做校验
    if ((xShape->GetDim(X_DIM_C) < VALUE_C_MIN) && (xShape->GetDim(X_DIM_C) != -1)) {
        OPS_LOG_E(nodeName, "The second dim of x should not < %ld, but got x[1] = %ld.", VALUE_C_MIN,
                  xShape->GetDim(X_DIM_C));
        return false;
    }
    // x[2]、w[wDimH] 是 H 轴 (reduce 轴)，不能为 0
    if ((xShape->GetDim(X_DIM_H) == 0) || (weightShape->GetDim(wDimH) == 0)) {
        OPS_LOG_E(nodeName, "The second dim of w(without transpose) or the last dim of x = 0 is unsupported.");
        return false;
    }
    // x[0]、w[0] 是 E 轴，在 -1 的时候需要相等; 当 x[0] = -1, w[0] 可以为 1
    if ((xShape->GetDim(DIM_E) == -1) || (weightShape->GetDim(DIM_E) == -1)) {
        if (xShape->GetDim(DIM_E) != weightShape->GetDim(DIM_E)) {
            if (!((xShape->GetDim(DIM_E) == -1) && (weightShape->GetDim(DIM_E) == 1))) {
                OPS_LOG_E(nodeName,
                          "The first dim of x should equal the first dim of w when w[0] or x[0] = -1 "
                          "(except for x[0] = -1 and w[0] = 1),"
                          "but got x[0] %ld, w[0] %ld.",
                          xShape->GetDim(DIM_E), weightShape->GetDim(DIM_E));
                return false;
            }
        }
    }
    // x[2]、w[wDimH] 在 -1 的时候需要相等
    if ((xShape->GetDim(X_DIM_H) == -1) || (weightShape->GetDim(wDimH) == -1)) {
        if (xShape->GetDim(X_DIM_H) != weightShape->GetDim(wDimH)) {
            OPS_LOG_E(nodeName,
                      "The last dim of x should equal the second dim of w(without transpose) when "
                      "w[1] or x[2] = -1, but got x[2] = %ld, w[1] = %ld.",
                      xShape->GetDim(X_DIM_H), weightShape->GetDim(wDimH));
            return false;
        }
    }

    return true;
}

// 盘古 2.6T (shard 0) && GPT 2.2T (shard 1)
static bool CheckTensorShapeDimWithoutxShard(const char* nodeName, const gert::Shape* xShape,
                                             const gert::Shape* weightShape, const int64_t epSize, const size_t wDimM)
{
    // x[DIM_E] = E, value E should = [2, 2048]
    if (((xShape->GetDim(DIM_E) < VALUE_E_MIN) || (xShape->GetDim(DIM_E) > VALUE_E_MAX)) &&
        xShape->GetDim(DIM_E) != -1) {
        OPS_LOG_E(nodeName, "Value E should in [%ld, %ld], but got %ld", VALUE_E_MIN, VALUE_E_MAX,
                  xShape->GetDim(DIM_E));
        return false;
    }

    // w[wDimM] = M / Tp, its range should same with H, so it meets M / Tp * H <= 65535 * 65535
    if (((weightShape->GetDim(wDimM) < VALUE_H_MIN) || (weightShape->GetDim(wDimM) > VALUE_H_MAX)) &&
        weightShape->GetDim(wDimM) != -1) {
        OPS_LOG_E(nodeName, "Value M / Tp should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
                  weightShape->GetDim(wDimM));
        return false;
    }

    // x[DIM_E] = E, w[DIM_E] = E / Ep, 所以需要满足 x[DIM_E] = w[DIM_E] * Ep
    if ((xShape->GetDim(DIM_E) != -1) && (weightShape->GetDim(DIM_E) != -1)) {
        if (weightShape->GetDim(DIM_E) * epSize != xShape->GetDim(DIM_E)) {
            OPS_LOG_E(nodeName,
                      "The first dim of w multi epSize should equal the first dim of x,"
                      "but got x[0] = %ld, w[0] = %ld, epSize = %ld",
                      xShape->GetDim(DIM_E), weightShape->GetDim(DIM_E), epSize);
            return false;
        }
    }
    return true;
}

static bool CheckTensorShapeDimWithZeroxShard(const char* nodeName, const gert::Shape* xShape,
                                              const gert::Shape* weightShape, const int64_t tpSize, const size_t wDimH)
{
    // x[X_DIM_H] = H / tp, value H should = [1, 65535]
    if (((xShape->GetDim(X_DIM_H) * tpSize < VALUE_H_MIN) || (xShape->GetDim(X_DIM_H) * tpSize > VALUE_H_MAX)) &&
        (xShape->GetDim(X_DIM_H) != -1)) {
        OPS_LOG_E(nodeName, "Value H should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
                  xShape->GetDim(X_DIM_H) * tpSize);
        return false;
    }

    // x[X_DIM_H] = H / tp, w[wDimH] = H, 所以需要满足 x[X_DIM_H] * Tp = w[wDimH]
    if ((xShape->GetDim(X_DIM_H) * tpSize != weightShape->GetDim(wDimH)) && (xShape->GetDim(X_DIM_H) != -1)) {
        OPS_LOG_E(nodeName,
                  "The last dim of x multi tp should equal the second dim of w, "
                  "but got x[2] = %ld, w[1] %ld.",
                  xShape->GetDim(X_DIM_H), weightShape->GetDim(wDimH));
        return false;
    }
    return true;
}

static bool CheckTensorShapeDimWithOnexShard(const char* nodeName, const gert::Shape* xShape,
                                             const gert::Shape* weightShape, const size_t wDimH)
{
    // x[X_DIM_H] = H, value H should = [1, 65535]
    if (((xShape->GetDim(X_DIM_H) < VALUE_H_MIN) || (xShape->GetDim(X_DIM_H) > VALUE_H_MAX)) &&
        xShape->GetDim(X_DIM_H) != -1) {
        OPS_LOG_E(nodeName, "Value H should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
                  xShape->GetDim(X_DIM_H));
        return false;
    }

    // x[X_DIM_H] = H, w[wDimH] = H, 所以 x[X_DIM_H] 需要等于 w[wDimH]
    if (xShape->GetDim(X_DIM_H) != weightShape->GetDim(wDimH)) {
        OPS_LOG_E(nodeName,
                  "The last dim of x should equal the second dim of w(without transpose), "
                  "but got x[2] = %ld, w[1] %ld.",
                  xShape->GetDim(X_DIM_H), weightShape->GetDim(wDimH));
        return false;
    }
    return true;
}

bool XShardCheckTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                            const int64_t epSize, const int64_t tpSize, const int64_t xShard, const size_t wDimH,
                            const size_t wDimM)
{
    // 检查 shape 维度的范围
    if (!CheckTensorShapeDimWithoutxShard(nodeName, xShape, weightShape, epSize, wDimM)) {
        return false;
    }
    if (xShard == 0) {
        return CheckTensorShapeDimWithZeroxShard(nodeName, xShape, weightShape, tpSize, wDimH);
    } else if (xShard == 1) {
        return CheckTensorShapeDimWithOnexShard(nodeName, xShape, weightShape, wDimH);
    }
    return true;
}

bool CheckBiasShape(const char* nodeName, const gert::Shape* weightShape, const gert::Shape* biasShape,
                    const size_t wDimM)
{
    // 检查 dimNum
    if ((biasShape->GetDimNum() != SUPPORT_DIM_NUM) && (biasShape->GetDimNum() != BIAS_SUPPORT_DIM_NUM)) {
        OPS_LOG_E(nodeName, "Dim of input bias must be the 2 or 3.");
        return false;
    }

    // 检查 shape
    if (biasShape->GetDim(0) != weightShape->GetDim(0)) {
        OPS_LOG_E(nodeName,
                  "The first dim of bias must be equal the first dim of weight, "
                  "but got bias[0] = %ld, w[0] = %ld.",
                  biasShape->GetDim(0), weightShape->GetDim(0));
        return false;
    }

    size_t biasLastDimIdx = 1; // 默认 bias 是二维，所以最后一维的 index 是 1
    if (biasShape->GetDimNum() == SUPPORT_DIM_NUM) {
        if (biasShape->GetDim(1) != 1) { // 三维时候，中间维度为 1
            OPS_LOG_E(nodeName, "The second dim of bias must be 1 when dim num is 3.");
            return false;
        }
        biasLastDimIdx = 2; // 三维时候，bias 的最后一维是 2
    }

    if (biasShape->GetDim(biasLastDimIdx) != weightShape->GetDim(wDimM)) {
        OPS_LOG_E(nodeName,
                  "The last dim of bias must equal the last dim of weight(without transpose), "
                  "but got bias[2] = %ld, w[2] = %ld.",
                  biasShape->GetDim(biasLastDimIdx), weightShape->GetDim(wDimM));
        return false;
    }

    return true;
}

bool CheckTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                      const gert::Shape* biasShape, const int64_t epSize, const int64_t tpSize, const size_t wDimH,
                      const size_t wDimM, const int64_t xShard)
{
    OPS_LOG_D(nodeName, "Begin to print x shape");
    PrintTensorShape(nodeName, xShape, "xShape");
    OPS_LOG_D(nodeName, "Begin to print weight shape");
    PrintTensorShape(nodeName, weightShape, "weightShape");
    if (!DimNumCheck(nodeName, xShape, weightShape)) {
        OPS_LOG_E(nodeName, "Dim num check failed.");
        return false;
    }

    if (!CommonCheckTensorShape(nodeName, xShape, weightShape, wDimH)) {
        OPS_LOG_E(nodeName, "common tensor shape check failed.");
        return false;
    }

    if (xShard == 0 || xShard == 1) {
        if (!XShardCheckTensorShape(nodeName, xShape, weightShape, epSize, tpSize, xShard, wDimH, wDimM)) {
            OPS_LOG_E(nodeName, "xShard = [%ld] tensor shape check failed.", xShard);
            return false;
        }
    } else {
        OPS_LOG_E(nodeName, "x shard type [%ld] is currently unsupported.", xShard);
        return false;
    }

    if (biasShape != nullptr) {
        OPS_LOG_D(nodeName, "Need to check bias shape.");
        OPS_LOG_D(nodeName, "Begin to print bias shape");
        PrintTensorShape(nodeName, biasShape, "biasShape");
        if (!(CheckBiasShape(nodeName, weightShape, biasShape, wDimM))) {
            OPS_LOG_E(nodeName, "Bias shape check failed.");
            return false;
        }
    }

    OPS_LOG_I(nodeName, "Tensor shape check success.");
    return true;
}

bool CheckAttrs(const gert::InferShapeContext* context, int64_t& epSize, int64_t& tpSize, bool& isTransW,
                int64_t& xShard, bool& y2Flag, bool& y3Flag)
{
    const char* nodeName = context->GetNodeName();

    auto attrs = context->GetAttrs();
    if (nodeName != nullptr) {
        OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is null"), return false);
    }

    // get 只有在 index 超出 attr num 的时候才会返回 nullptr
    const char* groupEp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_EP));
    const char* groupTp = attrs->GetStr(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_GROUP_TP));
    const int64_t* tpWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_TP_WORLD_SIZE));
    const int64_t* epWorldSize = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_EP_WORLD_SIZE));
    const bool* isTransWPtr = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_IS_TRANS_W));
    const int64_t* xShardType = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_X_SHARD_TYPE));
    const int64_t* actTypePtr = attrs->GetInt(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_ACT_TYPE));
    const bool* outputY2Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y2_FLAG));
    const bool* outputY3Flag = attrs->GetBool(static_cast<size_t>(ops::AlltoAllAllGatherBmmAttrIdx::K_OUTPUT_Y3_FLAG));

    if ((tpWorldSize == nullptr) || (epWorldSize == nullptr) || (isTransWPtr == nullptr) || (xShardType == nullptr) ||
        (actTypePtr == nullptr) || (outputY2Flag == nullptr) || (outputY3Flag == nullptr)) {
        OPS_LOG_E(nodeName, "attrs index in context is in valid or out of range, attrs got nullptr.");
        return false;
    }

    tpSize = *tpWorldSize;
    epSize = *epWorldSize;
    isTransW = *isTransWPtr;
    xShard = *xShardType;
    y2Flag = *outputY2Flag;
    y3Flag = *outputY3Flag;
    const int64_t actType = *actTypePtr;

    if (!GroupCheck(nodeName, groupEp, groupTp)) {
        OPS_LOG_E(nodeName, "group size check failed");
        return false;
    }

    if (!EpTpSizeCheck(epSize, tpSize)) {
        OPS_LOG_E(nodeName, "rank size error, tpSize [%ld], epSize [%ld]", tpSize, epSize);
        return false;
    }

    if ((xShard != 0) && (xShard != 1)) { // 当前仅支持 gpt2.2T, 值为 0 或 1
        OPS_LOG_E(nodeName, "x shard type [%ld] is invalid.", xShard);
        return false;
    }

    if (!ActTypeCheck(nodeName, actType, y3Flag)) {
        OPS_LOG_E(nodeName, "actType check failed.");
        return false;
    }

    OPS_LOG_I(nodeName,
              "attrs info: groupEp %s, groupTp %s, tpSize %ld, epSize %ld, isTransW %d, xShard %ld, "
              "y2Flag %d, y3Flag %d.",
              groupEp, groupTp, tpSize, epSize, isTransW, xShard, y2Flag, y3Flag);
    return true;
}

void GetY1Y3ShapeInfo(const gert::Shape* xShape, const gert::Shape* weightShape, const int64_t epSize,
                      const int64_t tpSize, const int64_t xShard, const size_t wDimM, OutShapeInfo& outShapeInfo)
{
    if (xShard == 0) {
        outShapeInfo.e = weightShape->GetDim(DIM_E);
        outShapeInfo.c = xShape->GetDim(X_DIM_C) * epSize;
        outShapeInfo.h = weightShape->GetDim(wDimM);
    } else if (xShard == 1) {
        outShapeInfo.e = weightShape->GetDim(DIM_E);
        outShapeInfo.c = xShape->GetDim(X_DIM_C) * epSize * tpSize;
        outShapeInfo.h = weightShape->GetDim(wDimM);
    }
    return;
}

void GetY2ShapeInfo(const gert::Shape* xShape, const gert::Shape* weightShape, const int64_t epSize,
                    const int64_t tpSize, const int64_t xShard, const size_t wDimH, OutShapeInfo& outShapeInfo)
{
    if (xShard == 0) {
        outShapeInfo.e = weightShape->GetDim(DIM_E);
        outShapeInfo.c = xShape->GetDim(X_DIM_C) * epSize;
        outShapeInfo.h = weightShape->GetDim(wDimH);
    } else if (xShard == 1) {
        outShapeInfo.e = weightShape->GetDim(DIM_E);
        outShapeInfo.c = xShape->GetDim(X_DIM_C) * epSize * tpSize;
        outShapeInfo.h = weightShape->GetDim(wDimH);
    }
    return;
}

} // namespace

namespace ops {
static void SetTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                           const int64_t epSize, const int64_t tpSize, const size_t wDimH, const size_t wDimM,
                           const int64_t xShard, gert::Shape* y1Shape, gert::Shape* y2Shape, gert::Shape* y3Shape,
                           bool y2Flag, bool y3Flag)
{
    // y1、y3
    OutShapeInfo outShapeInfo;
    GetY1Y3ShapeInfo(xShape, weightShape, epSize, tpSize, xShard, wDimM, outShapeInfo);
    // 动态 shape 检查
    Mc2Moe::DynamicShapeCheck(xShape, weightShape, wDimM, outShapeInfo);
    Mc2Moe::EmptyShapeCheck(xShape, weightShape, wDimM, outShapeInfo);
    Mc2Moe::SetShape(y1Shape, outShapeInfo);
    // 检查是否输出可选输出 y3，不输出时将各个 dim 的 value 置为 0; 并且 y3 与 y1 shape 一样，不再重复上述检查
    OutputFlagCheck(nodeName, y3Flag, "y3", outShapeInfo);
    Mc2Moe::SetShape(y3Shape, outShapeInfo);

    // y2
    GetY2ShapeInfo(xShape, weightShape, epSize, tpSize, xShard, wDimH, outShapeInfo);
    DynamicEmptyCheck(xShape, weightShape, wDimH, outShapeInfo);
    OutputFlagCheck(nodeName, y2Flag, "y2", outShapeInfo);
    Mc2Moe::SetShape(y2Shape, outShapeInfo);
}

static ge::graphStatus InferShapeAlltoAllAllGatherBmm(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E(K_INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);

    const char* nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter AlltoAllAllGatherBmm infer shape impl.");

    // 检查 shape 是否为空
    const gert::Shape* xShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    OPS_ERR_IF(xShape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "xShape is null."), return ge::GRAPH_FAILED);
    const gert::Shape* weightShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OPS_ERR_IF(weightShape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "weightShape is null."), return ge::GRAPH_FAILED);
    gert::Shape* y1Shape = context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y1));
    OPS_ERR_IF(y1Shape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "y1Shape is null."), return ge::GRAPH_FAILED);
    gert::Shape* y2Shape = context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y2));
    OPS_ERR_IF(y2Shape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "y2Shape is null."), return ge::GRAPH_FAILED);
    gert::Shape* y3Shape = context->GetOutputShape(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y3));
    OPS_ERR_IF(y3Shape == nullptr, OPS_LOG_E(K_INNER_DEBUG, "y3Shape is null."), return ge::GRAPH_FAILED);
    const gert::Shape* biasShape = context->GetOptionalInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));

    // 检查属性
    int64_t epSize = -1;
    int64_t tpSize = -1;
    bool isTransW = false;
    int64_t xShard = -1;
    bool y2Flag = false;
    bool y3Flag = false;

    if (!CheckAttrs(context, epSize, tpSize, isTransW, xShard, y2Flag, y3Flag)) {
        OPS_LOG_E(nodeName, "attrs check failed.");
        return ge::GRAPH_FAILED;
    }

    size_t wDimH = 1; // 设 w = [E, H, M], dimH 指 H 轴, dimM 指 M 轴
    size_t wDimM = 2; // 1 2 分别代表 weight 没有转置时候的维度值, 2: M 轴, 1: H 轴
    // 转置之后，w = [E, M, H], dimH dimM 指代的维度轴不变，但是维度值需交换, 2: H 轴, 1: M 轴
    Mc2Moe::TransDimHMIdx(isTransW, wDimH, wDimM);
    OPS_LOG_I(nodeName, "transpose weight is %d, dim H is %zu, dim M is %zu.", isTransW, wDimH, wDimM);

    // tensor shape 检查
    if (!CheckTensorShape(nodeName, xShape, weightShape, biasShape, epSize, tpSize, wDimH, wDimM, xShard)) {
        OPS_LOG_E(nodeName, "tensor shape check failed.");
        return ge::GRAPH_FAILED;
    }

    SetTensorShape(nodeName, xShape, weightShape, epSize, tpSize, wDimH, wDimM, xShard, y1Shape, y2Shape, y3Shape,
                   y2Flag, y3Flag);
    OPS_LOG_D(nodeName, "Begin to print y1 y2 y3 shape");
    PrintTensorShape(nodeName, y1Shape, "y1Shape");
    PrintTensorShape(nodeName, y2Shape, "y2Shape");
    PrintTensorShape(nodeName, y3Shape, "y3Shape");
    OPS_LOG_I(nodeName, "AlltoAllAllGatherBmm infer shape end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllAllGatherBmm(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E(K_INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);

    const char* nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter AlltoAllAllGatherBmm infer data type impl.");

    const ge::DataType xType = context->GetInputDataType(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    const ge::DataType weightType = context->GetInputDataType(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    const auto biasType = context->GetOptionalInputDataType(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));
    if (!CheckTensorDtype(nodeName, xType, weightType, biasType)) {
        OPS_LOG_E(nodeName, "tensor dtype check failed.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y1), xType);
    context->SetOutputDataType(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y2), xType);
    context->SetOutputDataType(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y3), xType);

    OPS_LOG_D(
        nodeName, "Output 1 dtype set to %u.",
        static_cast<uint32_t>(context->GetOutputDataType(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y1))));
    OPS_LOG_D(
        nodeName, "Output 2 dtype set to %u.",
        static_cast<uint32_t>(context->GetOutputDataType(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y2))));
    OPS_LOG_D(
        nodeName, "Output 3 dtype set to %u.",
        static_cast<uint32_t>(context->GetOutputDataType(static_cast<size_t>(ops::AlltoAllAllGatherBmmOutIdx::K_Y3))));
    OPS_LOG_I(nodeName, "AlltoAllAllGatherBmm infer data type end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllAllGatherBatchMatMul)
    .InferShape(InferShapeAlltoAllAllGatherBmm)
    .InferDataType(InferDataTypeAlltoAllAllGatherBmm);
} // namespace ops
