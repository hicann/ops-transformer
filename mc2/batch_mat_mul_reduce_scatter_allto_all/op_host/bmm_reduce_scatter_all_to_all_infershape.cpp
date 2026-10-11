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
 * \file bmm_reduce_scatter_all_to_all_infershape.cc
 * \brief
 */
#include <algorithm>

#include "register/op_impl_registry.h"
#include "mc2_hcom_topo_info.h"
#include "common/utils/op_mc2.h"
#include "mc2_moe_utils.h"

using namespace ge;
using namespace Mc2Moe;
namespace {
const char* K_INNER_DEBUG = "MC2: BmmReduceScatterAlltoAll Infershape Debug";

bool CommonCheckTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                            const size_t wDimM)
{
    // 检查 < 0 的范围: x dim C >= 1, dim E H M 会在后面拦截
    if ((xShape->GetDim(X_DIM_C) < VALUE_C_MIN) && (xShape->GetDim(X_DIM_C) != -1)) {
        OPS_LOG_E(nodeName, "The second dim of x should not < %ld, but got x[1] = %ld.", VALUE_C_MIN,
                  xShape->GetDim(X_DIM_C));
        return false;
    }
    // x[2]、w[wDimM] 是 M 轴(reduce 轴)，所以需要相等
    if (xShape->GetDim(X_DIM_M) != weightShape->GetDim(wDimM)) {
        OPS_LOG_E(nodeName,
                  "The last dim of x must equal the second dim of w(without transpose), "
                  "but got x[2] = %ld, w[1] = %ld.",
                  xShape->GetDim(X_DIM_M), weightShape->GetDim(wDimM));
        return false;
    }

    // x[2]、w[wDimM] 是 M 轴，不支持为 0
    if (xShape->GetDim(X_DIM_M) == 0) {
        OPS_LOG_E(nodeName,
                  "The last dim of x(x[2]) or the second dim of w(w[1], without transpose) = 0 is unsupported.");
        return false;
    }

    return true;
}

// 盘古 2.6T && GPT 2.2T
bool YShardCheckTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                            const int64_t epSize, const int64_t tpSize, const int64_t yShard, const size_t wDimH,
                            const size_t wDimM)
{
    // 检查 shape 维度的范围
    // value E should = [2, 2048], x[DIM_E] = E / Ep
    if ((xShape->GetDim(DIM_E) * epSize < VALUE_E_MIN || xShape->GetDim(DIM_E) * epSize > VALUE_E_MAX) &&
        xShape->GetDim(DIM_E) != -1) {
        OPS_LOG_E(nodeName, "Value E should in [%ld, %ld], but got %ld", VALUE_E_MIN, VALUE_E_MAX,
                  xShape->GetDim(DIM_E) * epSize);
        return false;
    }
    // w[wDimH] = H, value H should = [1, 65535]
    if (((weightShape->GetDim(wDimH) < VALUE_H_MIN) || (weightShape->GetDim(wDimH) > VALUE_H_MAX)) &&
        weightShape->GetDim(wDimH) != -1) {
        OPS_LOG_E(nodeName, "Value H should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
                  weightShape->GetDim(wDimH));
        return false;
    }
    // w[wDimM] = M / Tp, its range should same with H, so it meets M / Tp * H <= 65535 * 65535
    if (((weightShape->GetDim(wDimM) < VALUE_H_MIN) || (weightShape->GetDim(wDimM) > VALUE_H_MAX)) &&
        weightShape->GetDim(wDimM) != -1) {
        OPS_LOG_E(nodeName, "Value M / tp should in [%ld, %ld], but got %ld", VALUE_H_MIN, VALUE_H_MAX,
                  weightShape->GetDim(wDimM));
        return false;
    }

    // x[0] = E / Ep, w[0] = E / Ep, 所以两者需要相等
    if (xShape->GetDim(DIM_E) != weightShape->GetDim(DIM_E)) {
        OPS_LOG_E(nodeName, "The first dim of x must equal the first dim of w, but got x[0] = %ld, w[0] = %ld.",
                  xShape->GetDim(DIM_E), weightShape->GetDim(DIM_E));
        return false;
    }

    if (yShard == 0) {
        // x[1] = ep * C, 所以 x[1] % Ep 需要等于 0
        if ((xShape->GetDim(X_DIM_C) != -1) && (xShape->GetDim(X_DIM_C) % epSize != 0)) {
            OPS_LOG_E(nodeName,
                      "The second dim of x mod epSize must be 0, "
                      "but got x[1] = %ld, epSize = %ld.",
                      xShape->GetDim(X_DIM_C), epSize);
            return false;
        }

        // w[wDimH] = H, out 应该为 H / Tp, 所以 w[wDimH] % Tp = 0
        if ((weightShape->GetDim(wDimH) != -1) && (weightShape->GetDim(wDimH) % tpSize != 0)) {
            OPS_LOG_E(nodeName, "Input w dim H mod tpSize must be 0.");
            return false;
        }
    } else if (yShard == 1) {
        // x[1] = (c / Tp) * Ep * Tp, 所以 x[1] % (Ep * Tp) 需要等于 0
        if ((xShape->GetDim(X_DIM_C) != -1) && (xShape->GetDim(X_DIM_C) % (epSize * tpSize) != 0)) {
            OPS_LOG_E(nodeName,
                      "The second dim of x mod (epSize * tpSize) must be 0, "
                      "but got x[1] = %ld, epSize * tpSize = %ld.",
                      xShape->GetDim(X_DIM_C), epSize * tpSize);
            return false;
        }
    }

    return true;
}

bool CheckBiasShape(const char* nodeName, const gert::Shape* weightShape, const gert::Shape* biasShape,
                    const int64_t tpSize, const int64_t yShard, const size_t wDimH)
{
    // 检查 dimNum
    if ((biasShape->GetDimNum() != SUPPORT_DIM_NUM) && (biasShape->GetDimNum() != BIAS_SUPPORT_DIM_NUM)) {
        OPS_LOG_E(nodeName, "Dim of input bias must be 2 or 3, but got dim bias %zu.", biasShape->GetDimNum());
        return false;
    }

    // 检查 shape
    if (biasShape->GetDim(0) != weightShape->GetDim(0)) {
        OPS_LOG_E(nodeName,
                  "The first dim of bias must equal the first dim of weight,"
                  "but got bias[0] = %ld, w[0] = %ld.",
                  biasShape->GetDim(0), weightShape->GetDim(0));
        return false;
    }

    size_t biasLastDimValue = 1;                     // 默认 bias 是二维，所以最后一维的 index 是 1
    if (biasShape->GetDimNum() == SUPPORT_DIM_NUM) { // 三维
        if (biasShape->GetDim(1) != 1) {
            OPS_LOG_E(nodeName, "The second dim of bias must be 1 when 3-dim.");
            return false;
        }
        biasLastDimValue = 2; // 三维时候，bias 的最后一维是 2
    }

    if (yShard == 0 && biasShape->GetDim(biasLastDimValue) != -1) {
        // bias [ E / ep, 1 , H / tp ]  weight [ E / ep, M / tp, H ]
        if ((biasShape->GetDim(biasLastDimValue) * tpSize != weightShape->GetDim(wDimH)) &&
            (biasShape->GetDim(biasLastDimValue) != 1 || weightShape->GetDim(wDimH) != -1)) {
            OPS_LOG_E(nodeName,
                      "The last dim of bias (H / tp) multi tp must equal the last dim of weight(without transpose), "
                      "but got bias[2] * tp = %ld, w[2] = %ld.",
                      biasShape->GetDim(biasLastDimValue) * tpSize, weightShape->GetDim(wDimH));
            return false;
        }
    } else {
        // bias [ E / ep, 1 , H ]  weight [ E / ep, M / tp, H ]
        if (biasShape->GetDim(biasLastDimValue) != weightShape->GetDim(wDimH)) {
            OPS_LOG_E(nodeName,
                      "The last dim of bias must equal the last dim of weight(without transpose), "
                      "but got bias[2] = %ld, w[2] = %ld.",
                      biasShape->GetDim(biasLastDimValue), weightShape->GetDim(wDimH));
            return false;
        }
    }

    return true;
}

bool CheckTensorShape(const char* nodeName, const gert::Shape* xShape, const gert::Shape* weightShape,
                      const gert::Shape* biasShape, const int64_t epSize, const int64_t tpSize, const size_t wDimM,
                      const size_t wDimH, const int64_t yShard)
{
    OPS_LOG_D(nodeName, "Begin to print x shape");
    PrintTensorShape(nodeName, xShape, "xShape");
    OPS_LOG_D(nodeName, "Begin to print weight shape");
    PrintTensorShape(nodeName, weightShape, "weightShape");
    if (!DimNumCheck(nodeName, xShape, weightShape)) {
        OPS_LOG_E(nodeName, "Dim num check failed.");
        return false;
    }

    if (!CommonCheckTensorShape(nodeName, xShape, weightShape, wDimM)) {
        OPS_LOG_E(nodeName, "common tensor shape check failed.");
        return false;
    }

    if (yShard == 0 || yShard == 1) {
        if (!YShardCheckTensorShape(nodeName, xShape, weightShape, epSize, tpSize, yShard, wDimH, wDimM)) {
            OPS_LOG_E(nodeName, "yShard = [%ld] tensor shape check failed.", yShard);
            return false;
        }
    } else {
        OPS_LOG_E(nodeName, "y shard type [%ld] is currently unsupported.", yShard);
        return false;
    }

    if (biasShape != nullptr) {
        OPS_LOG_D(nodeName, "need to check bias shape.");
        OPS_LOG_D(nodeName, "Begin to print bias shape");
        PrintTensorShape(nodeName, biasShape, "biasShape");
        if (!(CheckBiasShape(nodeName, weightShape, biasShape, tpSize, yShard, wDimH))) {
            OPS_LOG_E(nodeName, "bias shape check failed.");
            return false;
        }
    }

    OPS_LOG_I(nodeName, "tensor shape check success.");
    return true;
}

bool CheckAttrs(const gert::InferShapeContext* context, int64_t& epSize, int64_t& tpSize, bool& isTransW,
                int64_t& yShard)
{
    const char* nodeName = context->GetNodeName();

    auto attrs = context->GetAttrs();
    if (nodeName != nullptr) {
        OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is null."), return false);
    }

    const char* groupEp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_EP));
    const char* groupTp = attrs->GetStr(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_GROUP_TP));
    const int64_t* epPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_EP_WORLD_SIZE));
    const int64_t* tpPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_TP_WORLD_SIZE));
    const bool* isTransWPtr = attrs->GetBool(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_IS_TRANS_W));
    const int64_t* yPtr = attrs->GetInt(static_cast<size_t>(ops::BmmReduceScatterAlltoAllAttrIdx::K_Y_SHARD_TYPE));

    if ((epPtr == nullptr) || (tpPtr == nullptr) || (isTransWPtr == nullptr) || (yPtr == nullptr)) {
        OPS_LOG_E(nodeName, "attrs index in context is in valid or out of range, attrs got nullptr.");
        return false;
    }

    epSize = *epPtr;
    tpSize = *tpPtr;
    isTransW = *isTransWPtr;
    yShard = *yPtr;

    if (!GroupCheck(nodeName, groupEp, groupTp)) {
        OPS_LOG_E(nodeName, "group size check failed.");
        return false;
    }

    if (!EpTpSizeCheck(epSize, tpSize)) {
        OPS_LOG_E(nodeName, "rank size error, tpSize [%ld], epSize [%ld].", tpSize, epSize);
        return false;
    }

    if ((yShard != 0) && (yShard != 1)) { // 当前支持 gpt2.2T 盘古 2.6T
        OPS_LOG_E(nodeName, "y shard type [%ld] is invalid.", yShard);
        return false;
    }

    OPS_LOG_I(nodeName, "attr info: groupEp %s, groupTp %s, tpSize %ld, epSize %ld, isTransW %d, yShard %ld.", groupEp,
              groupTp, tpSize, epSize, isTransW, yShard);
    return true;
}

void GetShapeInfo(const gert::Shape* xShape, const gert::Shape* weightShape, const int64_t epSize, const int64_t tpSize,
                  const int64_t yShard, const size_t wDimH, OutShapeInfo& outShapeInfo)
{
    if (yShard == 0) {
        outShapeInfo.e = xShape->GetDim(DIM_E) * epSize;
        outShapeInfo.c = xShape->GetDim(X_DIM_C) / epSize;
        outShapeInfo.h = weightShape->GetDim(wDimH) / tpSize;
    } else if (yShard == 1) {
        outShapeInfo.e = xShape->GetDim(DIM_E) * epSize;
        outShapeInfo.c = xShape->GetDim(X_DIM_C) / (epSize * tpSize);
        outShapeInfo.h = weightShape->GetDim(wDimH);
    }
    return;
}

} // namespace

namespace ops {
static ge::graphStatus InferShapeBmmReduceScatterAlltoAll(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E(K_INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);

    const char* nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter BmmReduceScatterAlltoAll infer shape impl.");

    // 检查 shape 是否为空
    const gert::Shape* xShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    OPS_ERR_IF(xShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const gert::Shape* weightShape = context->GetInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    OPS_ERR_IF(weightShape == nullptr, OPS_LOG_E(nodeName, "weightShape is null."), return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(static_cast<size_t>(ops::BmmReduceScatterAlltoAllOutIdx::K_Y));
    OPS_ERR_IF(yShape == nullptr, OPS_LOG_E(nodeName, "yShape is null."), return ge::GRAPH_FAILED);
    const gert::Shape* biasShape = context->GetOptionalInputShape(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));

    // 检查属性
    int64_t epSize = -1;
    int64_t tpSize = -1;
    bool isTransW = false;
    int64_t yShard = -1;
    if (!CheckAttrs(context, epSize, tpSize, isTransW, yShard)) {
        OPS_LOG_E(nodeName, "Attrs check failed.");
        return ge::GRAPH_FAILED;
    }

    size_t wDimM = 1; // 设 w = [E, M, H], dimH 指 H 轴, dimM 指 M 轴
    size_t wDimH = 2; // 1 2 分别代表 weight 没有转置时候的维度值, 2: H 轴, 1: M 轴
    // 如果转置, w = [E, H, M], dimH dimM 指代的维度轴不变, 但是维度值需交换, 2: M 轴, 1: H 轴
    Mc2Moe::TransDimHMIdx(isTransW, wDimH, wDimM);
    OPS_LOG_I(nodeName, "Transpose weight is %d, dim H is %zu, dim M is %zu.", isTransW, wDimH, wDimM);

    // tensor shape 检查
    if (!CheckTensorShape(nodeName, xShape, weightShape, biasShape, epSize, tpSize, wDimM, wDimH, yShard)) {
        OPS_LOG_E(nodeName, "Tensor shape check failed.");
        return ge::GRAPH_FAILED;
    }

    // 根据 y shard 获取 outShapeInfo
    OutShapeInfo outShapeInfo;
    GetShapeInfo(xShape, weightShape, epSize, tpSize, yShard, wDimH, outShapeInfo);

    // 动态 shape 、empty shape 检查，然后 set out shape
    Mc2Moe::DynamicShapeCheck(xShape, weightShape, wDimH, outShapeInfo);
    Mc2Moe::EmptyShapeCheck(xShape, weightShape, wDimH, outShapeInfo);
    Mc2Moe::SetShape(yShape, outShapeInfo);

    OPS_LOG_D(nodeName, "Begin to print y shape");
    PrintTensorShape(nodeName, yShape, "yShape");
    OPS_LOG_I(nodeName, "BmmReduceScatterAlltoAll infer shape end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBmmReduceScatterAlltoAll(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E(K_INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);

    const char* nodeName = context->GetNodeName();
    OPS_LOG_I(nodeName, "Enter BmmReduceScatterAlltoAll infer data type impl.");

    // tensor 类型检查
    const ge::DataType xType = context->GetInputDataType(static_cast<size_t>(ops::MC2MoeInputIdx::K_X));
    const ge::DataType weightType = context->GetInputDataType(static_cast<size_t>(ops::MC2MoeInputIdx::K_WEIGHT));
    const auto biasType = context->GetOptionalInputDataType(static_cast<size_t>(ops::MC2MoeInputIdx::K_BIAS));
    if (!CheckTensorDtype(nodeName, xType, weightType, biasType)) {
        OPS_LOG_E(nodeName, "tensor dtype check failed.");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(static_cast<size_t>(ops::BmmReduceScatterAlltoAllOutIdx::K_Y), xType);

    OPS_LOG_D(nodeName, "Output dtype set to %u.",
              static_cast<uint32_t>(
                  context->GetOutputDataType(static_cast<size_t>(ops::BmmReduceScatterAlltoAllOutIdx::K_Y))));
    OPS_LOG_I(nodeName, "BmmReduceScatterAlltoAll infer data type end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BatchMatMulReduceScatterAlltoAll)
    .InferShape(InferShapeBmmReduceScatterAlltoAll)
    .InferDataType(InferDataTypeBmmReduceScatterAlltoAll);
} // namespace ops
