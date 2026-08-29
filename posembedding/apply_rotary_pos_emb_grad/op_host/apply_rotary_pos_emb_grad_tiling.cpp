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
 * \file apply_rotary_pos_emb_grad_tiling.cpp
 * \brief
 */
#include <cmath>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "apply_rotary_pos_emb_grad_tiling.h"
#include "../op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "../op_kernel/arch35/apply_rotary_pos_emb_grad_dag.h"

namespace {
constexpr int64_t GQ_INDEX = 0;       // grad_query_embed
constexpr int64_t GK_INDEX = 1;       // grad_key_embed
constexpr int64_t COS_INDEX = 2;      // cos
constexpr int64_t SIN_INDEX = 3;      // sin
constexpr int64_t QUERY_INDEX = 4;    // query (optional)
constexpr int64_t KEY_INDEX = 5;      // key (optional)
constexpr int64_t GRAD_Q_INDEX = 0;   // grad_query
constexpr int64_t GRAD_K_INDEX = 1;   // grad_key
constexpr int64_t GRAD_COS_INDEX = 2; // grad_cos
constexpr int64_t GRAD_SIN_INDEX = 3; // grad_sin
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_MODE_COEF = 2;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t INPUT_OUTPUT_NUM = 2; // grad_query + grad_key 双路
constexpr int64_t PARTIAL_TYPE_SIZE = sizeof(float);
constexpr int64_t ATTR_INDEX_ROTARY_MODE = 0; // attr 0: rotary_mode (string)
constexpr int64_t ATTR_INDEX_LAYOUT = 1;      // attr 1: layout (int64_t)
constexpr int64_t ATTR_LAYOUT_BSND = 1;
constexpr int64_t ATTR_LAYOUT_SBND = 2;
constexpr int64_t ATTR_LAYOUT_TND = 4;
constexpr int64_t ATTR_LAYOUT_DEFAULT = ATTR_LAYOUT_BSND;
constexpr int64_t BROADCAST_DIM_SIZE = 1;
constexpr size_t N_AXIS_OFFSET_FROM_END = 2; // N 轴为倒数第二维: 4D(B,S,N,D)/3D(T,N,D)
constexpr uint32_t DCOS_FLAG_OFF = 0;
constexpr uint32_t DCOS_FLAG_ON = 1;
constexpr int32_t SCHEDULE_MODE_BATCH = 1; // batch 模式, 所有核同时启动

const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<std::string> inputNames = {"grad_query_embed", "grad_key_embed", "cos", "sin", "query", "key"};
static const std::vector<std::string> outputNames = {"grad_query", "grad_key", "grad_cos", "grad_sin"};
} // namespace

namespace optiling {
using namespace Ops::Base;

namespace {
bool IsBroadcastPartialFloatTemplate(uint32_t dxTilingKey)
{
    return dxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_BAB) ||
           dxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_AB);
}
} // namespace

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<ApplyRotaryPosEmbGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    vLength_ = Ops::Base::GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckNullptr()
{
    // optional inputs: query and key must both be present or both absent
    bool hasQuery = context_->GetInputShape(QUERY_INDEX) != nullptr;
    bool hasKey = context_->GetInputShape(KEY_INDEX) != nullptr;
    if (hasQuery != hasKey) {
        OP_LOGE(context_, "query and key must both be present or both absent.");
        return ge::GRAPH_FAILED;
    }
    // grad_cos/grad_sin may be nullptr when query/key are absent
    int64_t checkInputIndexRange = hasQuery ? KEY_INDEX : SIN_INDEX;
    int64_t checkOutputIndexRange = hasQuery ? GRAD_SIN_INDEX : GRAD_K_INDEX;

    for (int64_t i = 0; i <= checkInputIndexRange; i++) {
        if ((i == QUERY_INDEX || i == KEY_INDEX) && context_->GetInputShape(i) == nullptr) {
            continue;
        }
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_, "input %ld desc is nullptr.", i), return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_, "input %ld shape is nullptr.", i), return ge::GRAPH_FAILED);
    }

    for (int64_t i = 0; i <= checkOutputIndexRange; i++) {
        auto desc = context_->GetOutputDesc(i);
        if (desc == nullptr) {
            if (i == GRAD_COS_INDEX || i == GRAD_SIN_INDEX) {
                continue;
            }
            OP_LOGE(context_, "output %ld desc is nullptr.", i);
            return ge::GRAPH_FAILED;
        }
        auto shape = context_->GetOutputShape(i);
        if (shape == nullptr) {
            if (i == GRAD_COS_INDEX || i == GRAD_SIN_INDEX) {
                continue;
            }
            OP_LOGE(context_, "output %ld shape is nullptr.", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckInPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of input " + inputNames[idx] +
                                    " can not be an empty tensor or an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[idx].c_str(), shapeMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckOutPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetOutputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            std::string shapeMsg = ToString(shape);
            std::string reasonMsg = "The shape of output " + outputNames[idx] +
                                    " can not be an empty tensor or an invalid tensor with a negative dimension";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputNames[idx].c_str(), shapeMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckShapeAllPositive() const
{
    OP_CHECK_IF(CheckInPutShapeAllPositive(GQ_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "grad_query_embed has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(GK_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "grad_key_embed has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "cos has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInPutShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "sin has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutPutShapeAllPositive(GRAD_Q_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "grad_query has non positive shape."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutPutShapeAllPositive(GRAD_K_INDEX) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "grad_key has non positive shape."), return ge::GRAPH_FAILED);
    if (context_->GetInputShape(QUERY_INDEX) != nullptr) {
        OP_CHECK_IF(CheckInPutShapeAllPositive(QUERY_INDEX) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "query has non positive shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckInPutShapeAllPositive(KEY_INDEX) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "key has non positive shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckOutPutShapeAllPositive(GRAD_COS_INDEX) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "grad_cos has non positive shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckOutPutShapeAllPositive(GRAD_SIN_INDEX) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "grad_sin has non positive shape."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::ValidateBroadcastByLayout(int64_t attrLayout,
                                                                                   const gert::Shape &gqShape,
                                                                                   const gert::Shape &gkShape,
                                                                                   const gert::Shape &cosShape)
{
    isTndLayout_ = (gqShape.GetDimNum() == DIM_NUM_TND);

    // attr=TND 但输入为 4D: 显式报 layout 与 shape 维度不匹配
    if (attrLayout == ATTR_LAYOUT_TND && !isTndLayout_) {
        std::string layoutValStr = std::to_string(attrLayout);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout", layoutValStr.c_str(),
                                  "TND(4) with 4D input shape");
        return ge::GRAPH_FAILED;
    }
    // attr=BSND/SBND 但输入为 3D: layout 与维度数不匹配, BSND/SBND 要求 4D 输入
    if (isTndLayout_ && attrLayout != ATTR_LAYOUT_TND) {
        std::string layoutValStr = std::to_string(attrLayout);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout", layoutValStr.c_str(),
                                  "BSND(1)/SBND(2) with 3D input shape, they require 4D inputs");
        return ge::GRAPH_FAILED;
    }

    // TND: (T, N, D), cos 为 (T, 1, D) → 退化为 B=1 的 BSND
    if (isTndLayout_) {
        int64_t cosT = cosShape.GetDim(DIM_0);
        int64_t cosN = cosShape.GetDim(DIM_1);
        int64_t gqT = gqShape.GetDim(DIM_0);
        if (cosN != BROADCAST_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  "For TND layout, cos N axis must be 1");
            return ge::GRAPH_FAILED;
        }
        if (cosT != gqT) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  "For TND layout, cos T axis must equal grad_query_embed T");
            return ge::GRAPH_FAILED;
        }
        b_ = BROADCAST_DIM_SIZE;
        cosb_ = BROADCAST_DIM_SIZE;
        s_ = gqT;
        nQ_ = gqShape.GetDim(DIM_1);
        nK_ = gkShape.GetDim(DIM_1);
        d_ = gqShape.GetDim(DIM_2);
        // TND 退化为 BSND, 若 shape 完全一致 (B=1, N=1) → NO_BROADCAST (A 模板)
        if (nQ_ == BROADCAST_DIM_SIZE && nK_ == BROADCAST_DIM_SIZE) {
            layout_ = ApplyRopeGradLayout::NO_BROADCAST;
        } else {
            layout_ = ApplyRopeGradLayout::BSND; // TND → BAB template
        }
        return ge::GRAPH_SUCCESS;
    }

    // BSND (attr=1): gq(B, S, N, D), cos(cosB, S, 1, D)
    if (attrLayout == ATTR_LAYOUT_BSND) {
        int64_t gqB = gqShape.GetDim(DIM_0);
        int64_t gqS = gqShape.GetDim(DIM_1);
        int64_t cosB = cosShape.GetDim(DIM_0);
        int64_t cosS = cosShape.GetDim(DIM_1);
        int64_t cosN = cosShape.GetDim(DIM_2);
        if (cosN != BROADCAST_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  "For BSND layout, cos N axis must be 1");
            return ge::GRAPH_FAILED;
        }
        if (cosS != gqS) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  "For BSND layout, cos S axis must equal grad_query_embed S");
            return ge::GRAPH_FAILED;
        }
        if (cosB != BROADCAST_DIM_SIZE && cosB != gqB) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                "For BSND layout, cos B axis must be 1 or equal to grad_query_embed B");
            return ge::GRAPH_FAILED;
        }
        b_ = gqB;
        s_ = gqS;
        cosb_ = cosB;
        nQ_ = gqShape.GetDim(DIM_2);
        nK_ = gkShape.GetDim(DIM_2);
        d_ = gqShape.GetDim(DIM_3);
        // cosB==1 → BSND(BAB), cosB==gqB → SBND(AB)
        // 如果全shape完全一致 → NO_BROADCAST (A)
        // A 模板 MergeDim 按 nQ==nK 合并迭代空间, 若 gk 的 N 轴与 gq 不一致,
        // grad_key 无法全覆盖且 dcos/dsin 的 K 路贡献会读错行, 故 NO_BROADCAST
        // 须同时要求 gkN == gqN, 否则回落 BAB/AB (两者均支持 nQ≠nK)
        if (gqB == cosB && gqS == cosS) {
            // 所有 dim 都相等 (含 gq/gk 的 N 轴) → NO_BROADCAST
            if (cosShape.GetDim(DIM_2) == nQ_ && nK_ == nQ_) {
                layout_ = ApplyRopeGradLayout::NO_BROADCAST;
            } else {
                // cos N=1 ≠ gq N 或 gk N ≠ gq N, so still broadcast
                layout_ = (cosb_ == BROADCAST_DIM_SIZE) ? ApplyRopeGradLayout::BSND : ApplyRopeGradLayout::SBND;
            }
        } else {
            layout_ = (cosb_ == BROADCAST_DIM_SIZE) ? ApplyRopeGradLayout::BSND : ApplyRopeGradLayout::SBND;
        }
        return ge::GRAPH_SUCCESS;
    }

    // SBND (attr=2): gq(S, B, N, D), cos(cosS, cosB, 1, D)
    if (attrLayout == ATTR_LAYOUT_SBND) {
        int64_t gqS = gqShape.GetDim(DIM_0);
        int64_t gqB = gqShape.GetDim(DIM_1);
        int64_t cosS = cosShape.GetDim(DIM_0);
        int64_t cosB = cosShape.GetDim(DIM_1);
        int64_t cosN = cosShape.GetDim(DIM_2);
        nQ_ = gqShape.GetDim(DIM_2);
        nK_ = gkShape.GetDim(DIM_2);
        d_ = gqShape.GetDim(DIM_3);
        // cos 与 gq 所有轴完全一致 (含 gk 的 N 轴) → 无广播, 回落 NO_BROADCAST (A 模板)
        if (gqS == cosS && gqB == cosB && cosN == nQ_ && nK_ == nQ_) {
            b_ = gqB;
            s_ = gqS;
            cosb_ = cosB;
            layout_ = ApplyRopeGradLayout::NO_BROADCAST;
            return ge::GRAPH_SUCCESS;
        }
        if (cosN != BROADCAST_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  "For SBND layout, cos N axis must be 1");
            return ge::GRAPH_FAILED;
        }
        if (cosB != BROADCAST_DIM_SIZE && cosB != gqB) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                "For SBND layout, cos B axis must be 1 or equal to grad_query_embed B");
            return ge::GRAPH_FAILED;
        }
        if (cosS != gqS) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                                  "For SBND layout, cos S axis must equal grad_query_embed S");
            return ge::GRAPH_FAILED;
        }
        b_ = gqB;
        s_ = gqS;
        cosb_ = cosB;
        // SBND → AB template
        layout_ = ApplyRopeGradLayout::SBND;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context_->GetNodeName(), "Unsupported layout value: %ld", attrLayout);
    return ge::GRAPH_FAILED;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckShapeDim() const
{
    auto &gqShape = context_->GetInputShape(GQ_INDEX)->GetStorageShape();
    auto &gkShape = context_->GetInputShape(GK_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &ogqShape = context_->GetOutputShape(GRAD_Q_INDEX)->GetStorageShape();
    auto &ogkShape = context_->GetOutputShape(GRAD_K_INDEX)->GetStorageShape();

    // 必选输入必须3D或4D
    if (gqShape.GetDimNum() != DIM_NUM && gqShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(gqShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_query_embed", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (gkShape.GetDimNum() != DIM_NUM && gkShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(gkShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_key_embed", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDimNum() != DIM_NUM && cosShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(cosShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "cos", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (sinShape.GetDimNum() != DIM_NUM && sinShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(sinShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "sin", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (ogqShape.GetDimNum() != DIM_NUM && ogqShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(ogqShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_query", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (ogkShape.GetDimNum() != DIM_NUM && ogkShape.GetDimNum() != DIM_NUM_TND) {
        std::string dimNumStr = std::to_string(ogkShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_key", dimNumStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }

    // 可选输入维度检查
    if (context_->GetInputShape(QUERY_INDEX) != nullptr) {
        auto &qShape = context_->GetInputShape(QUERY_INDEX)->GetStorageShape();
        auto &kShape = context_->GetInputShape(KEY_INDEX)->GetStorageShape();
        auto &ogcShape = context_->GetOutputShape(GRAD_COS_INDEX)->GetStorageShape();
        auto &ogsShape = context_->GetOutputShape(GRAD_SIN_INDEX)->GetStorageShape();
        if (qShape.GetDimNum() != DIM_NUM && qShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(qShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "query", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
        if (kShape.GetDimNum() != DIM_NUM && kShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(kShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "key", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
        if (ogcShape.GetDimNum() != DIM_NUM && ogcShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(ogcShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_cos", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
        if (ogsShape.GetDimNum() != DIM_NUM && ogsShape.GetDimNum() != DIM_NUM_TND) {
            std::string dimNumStr = std::to_string(ogsShape.GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_sin", dimNumStr.c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckShapeLimit()
{
    auto &gqShape = context_->GetInputShape(GQ_INDEX)->GetStorageShape();
    auto &gkShape = context_->GetInputShape(GK_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &ogqShape = context_->GetOutputShape(GRAD_Q_INDEX)->GetStorageShape();
    auto &ogkShape = context_->GetOutputShape(GRAD_K_INDEX)->GetStorageShape();

    if (cosShape != sinShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(sinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and sin", shapeMsg.c_str(),
                                               "The shapes of input cos and sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (gqShape != ogqShape) {
        std::string shapeMsg = ToString(gqShape) + " and " + ToString(ogqShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "grad_query_embed and grad_query", shapeMsg.c_str(),
            "The shapes of input grad_query_embed and output grad_query should be the same");
        return ge::GRAPH_FAILED;
    }
    if (gkShape != ogkShape) {
        std::string shapeMsg = ToString(gkShape) + " and " + ToString(ogkShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "grad_key_embed and grad_key", shapeMsg.c_str(),
            "The shapes of input grad_key_embed and output grad_key should be the same");
        return ge::GRAPH_FAILED;
    }

    if (context_->GetInputShape(QUERY_INDEX) != nullptr) {
        dCosFlag_ = DCOS_FLAG_ON;
        return CheckOptionalInput();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckOptionalInput() const
{
    auto &gqShape = context_->GetInputShape(GQ_INDEX)->GetStorageShape();
    auto &gkShape = context_->GetInputShape(GK_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &qShape = context_->GetInputShape(QUERY_INDEX)->GetStorageShape();
    auto &kShape = context_->GetInputShape(KEY_INDEX)->GetStorageShape();
    auto &ogcShape = context_->GetOutputShape(GRAD_COS_INDEX)->GetStorageShape();
    auto &ogsShape = context_->GetOutputShape(GRAD_SIN_INDEX)->GetStorageShape();

    if (qShape != gqShape) {
        std::string shapeMsg = ToString(qShape) + " and " + ToString(gqShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "query and grad_query_embed", shapeMsg.c_str(),
            "The shapes of input query and input grad_query_embed should be the same");
        return ge::GRAPH_FAILED;
    }
    if (kShape != gkShape) {
        std::string shapeMsg = ToString(kShape) + " and " + ToString(gkShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "key and grad_key_embed", shapeMsg.c_str(),
                                               "The shapes of input key and input grad_key_embed should be the same");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != ogcShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(ogcShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and grad_cos", shapeMsg.c_str(),
                                               "The shapes of input cos and output grad_cos should be the same");
        return ge::GRAPH_FAILED;
    }
    if (sinShape != ogsShape) {
        std::string shapeMsg = ToString(sinShape) + " and " + ToString(ogsShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "sin and grad_sin", shapeMsg.c_str(),
                                               "The shapes of input sin and output grad_sin should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckShape()
{
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &gqShape = context_->GetInputShape(GQ_INDEX)->GetStorageShape();
    OP_CHECK_IF(CheckShapeDim() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape dim fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeLimit() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "CheckShapeLimit fail."),
                return ge::GRAPH_FAILED);
    // cos 与 gq 必须同为 3D 或 4D
    if (cosShape.GetDimNum() != gqShape.GetDimNum()) {
        std::string shapeMsg = ToString(gqShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_query_embed and cos", shapeMsg.c_str(),
                                               "The dim num of grad_query_embed and cos should be the same");
        return ge::GRAPH_FAILED;
    }
    isTndLayout_ = (gqShape.GetDimNum() == DIM_NUM_TND);
    int64_t gqLastDim = isTndLayout_ ? gqShape.GetDim(DIM_2) : gqShape.GetDim(DIM_3);
    int64_t cosLastDim = isTndLayout_ ? cosShape.GetDim(DIM_2) : cosShape.GetDim(DIM_3);
    if (cosLastDim != gqLastDim) {
        std::string shapeMsg = ToString(gqShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_query_embed and cos", shapeMsg.c_str(),
                                               "The D axis of input grad_query_embed and input cos should be the same");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckRotaryModeShapeRelation(gqLastDim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    // 约束: gradQueryEmbed和gradKeyEmbed除N维度外其它维度必须相同
    auto &gkShape = context_->GetInputShape(GK_INDEX)->GetStorageShape();
    if (gkShape.GetDimNum() != gqShape.GetDimNum()) {
        std::string shapeMsg = ToString(gqShape) + " and " + ToString(gkShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_query_embed and grad_key_embed",
                                               shapeMsg.c_str(),
                                               "The dim num of grad_query_embed and grad_key_embed should be the same");
        return ge::GRAPH_FAILED;
    }
    if (isTndLayout_) {
        // TND: dims (T, N, D) → T and D must be same, N can differ
        if (gkShape.GetDim(DIM_0) != gqShape.GetDim(DIM_0) || gkShape.GetDim(DIM_2) != gqShape.GetDim(DIM_2)) {
            std::string shapeMsg = ToString(gqShape) + " and " + ToString(gkShape);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "grad_query_embed and grad_key_embed", shapeMsg.c_str(),
                "For TND layout, T and D axes of grad_query_embed and grad_key_embed should be the same");
            return ge::GRAPH_FAILED;
        }
    } else {
        // 4D: dims (B, S, N, D) or (S, B, N, D) etc → non-N axes must be same
        for (int64_t d = 0; d < DIM_NUM; ++d) {
            // skip N axis (dim 2 for BSND/SBND)
            if (d == DIM_2)
                continue;
            if (gkShape.GetDim(d) != gqShape.GetDim(d)) {
                std::string shapeMsg = ToString(gqShape) + " and " + ToString(gkShape);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad_query_embed and grad_key_embed",
                                                       shapeMsg.c_str(),
                                                       "For 4D layout, all axes except N (dim 2) of grad_query_embed "
                                                       "and grad_key_embed should be the same");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(GQ_INDEX)->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end()) {
        std::string dtypeStr = ToString(dtype_);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_query_embed", dtypeStr.c_str(),
                                  "FLOAT32, BFLOAT16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }

    int64_t checkInputIndexRange = KEY_INDEX;
    int64_t checkOutputIndexRange = GRAD_SIN_INDEX;
    if (context_->GetInputShape(QUERY_INDEX) == nullptr) {
        checkInputIndexRange = SIN_INDEX;
        checkOutputIndexRange = GRAD_K_INDEX;
    }

    for (int64_t i = GQ_INDEX; i <= checkInputIndexRange; i++) {
        if ((i == QUERY_INDEX || i == KEY_INDEX) && context_->GetInputShape(i) == nullptr) {
            continue;
        }
        auto type = context_->GetInputDesc(i)->GetDataType();
        if (type != dtype_) {
            std::string paramMsg = inputNames[i] + " and grad_query_embed";
            std::string dtypeMsg = ToString(type) + " and " + ToString(dtype_);
            std::string reasonMsg =
                "The dtypes of input " + inputNames[i] + " and input grad_query_embed should be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), dtypeMsg.c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    for (int64_t i = GRAD_Q_INDEX; i <= checkOutputIndexRange; i++) {
        if ((i == GRAD_COS_INDEX || i == GRAD_SIN_INDEX) && context_->GetOutputShape(i) == nullptr) {
            continue;
        }
        auto type = context_->GetOutputDesc(i)->GetDataType();
        if (type != dtype_) {
            std::string paramMsg = outputNames[i] + " and grad_query_embed";
            std::string dtypeMsg = ToString(type) + " and " + ToString(dtype_);
            std::string reasonMsg =
                "The dtypes of output " + outputNames[i] + " and input grad_query_embed should be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(), dtypeMsg.c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platform info is nullptr."), return ge::GRAPH_FAILED);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check dtype and attr fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeAllPositive() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check shape positive fail."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    auto gqShape = context_->GetInputShape(GQ_INDEX)->GetStorageShape();
    if (d > D_LIMIT) {
        std::string shapeMsg = ToString(gqShape);
        std::string reasonMsg = "The D axis can not be greater than " + std::to_string(D_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_query_embed", shapeMsg.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (d % HALF_MODE_COEF != 0) {
        std::string shapeMsg = ToString(gqShape);
        std::string reasonMsg = "The D axis should be divisible by " + std::to_string(HALF_MODE_COEF) +
                                " when the attr rotary_mode is half";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_query_embed", shapeMsg.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    dSplitCoef_ = HALF_MODE_COEF;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // attr 0: rotary_mode (string)
    const char *rotaryModeStr = attrs->GetAttrPointer<char>(ATTR_INDEX_ROTARY_MODE);
    std::string rotaryModeVal = (rotaryModeStr == nullptr) ? "half" : rotaryModeStr;
    if (rotaryModeVal != "half") {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "rotary_mode", rotaryModeVal.c_str(), "half");
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = ApplyRopeGradRotaryMode::HALF;

    // attr 1: layout (int64_t)
    const int64_t *layoutPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_LAYOUT);
    int64_t layoutValue = (layoutPtr == nullptr) ? ATTR_LAYOUT_DEFAULT : (*layoutPtr);
    if (layoutValue != ATTR_LAYOUT_BSND && layoutValue != ATTR_LAYOUT_SBND && layoutValue != ATTR_LAYOUT_TND) {
        std::string layoutValStr = std::to_string(layoutValue);
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout", layoutValStr.c_str(), "1, 2 or 4");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckParam() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "check param fail."), return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(GQ_INDEX)->GetDataType();
    gqShape_ = context_->GetInputShape(GQ_INDEX)->GetStorageShape();
    cosShape_ = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &gkShape = context_->GetInputShape(GK_INDEX)->GetStorageShape();
    OP_CHECK_IF(ValidateBroadcastByLayout(layoutValue, gqShape_, gkShape, cosShape_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ValidateBroadcastByLayout fail."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::GetInputParam(Ops::Base::ReduceOpInputParam &opInput,
                                                                       uint32_t inputIdx, uint32_t axesIdx)
{
    auto inputDesc = context_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    opInput.inputDtype = inputDesc->GetDataType();
    auto dyInput = context_->GetInputTensor(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyInput);
    auto dyInputShape = Ops::Transformer::OpTiling::EnsureNotScalar(dyInput->GetStorageShape());
    size_t shapeSize = dyInputShape.GetDimNum();
    opInput.shape.resize(shapeSize);
    for (size_t i = 0; i < shapeSize; i++) {
        opInput.shape[i] = dyInputShape.GetDim(i);
    }
    inputDesc = context_->GetInputDesc(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto cosInput = context_->GetInputTensor(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosInput);
    auto cosShape = Ops::Transformer::OpTiling::EnsureNotScalar(cosInput->GetStorageShape());
    for (size_t i = 0; i < shapeSize; i++) {
        int64_t dimExtent = static_cast<int64_t>(opInput.shape[i]);
        if (reduceInputFloat_ && i + N_AXIS_OFFSET_FROM_END == shapeSize) {
            dimExtent = std::max(dimExtent, nK_);
        }
        if (cosShape.GetDim(i) == BROADCAST_DIM_SIZE && dimExtent != BROADCAST_DIM_SIZE) {
            opInput.axes.push_back(i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<ApplyRopeGradTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((memset_s(tilingData_, sizeof(ApplyRopeGradTilingData), 0, sizeof(ApplyRopeGradTilingData)) != EOK),
                OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::GetReduceOpCompileInfo(
    Ops::Base::ReduceOpCompileInfo *compileInfo)
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->vectorCoreNum == 0),
                OP_LOGE(context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu",
                        compileInfo->vectorCoreNum),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(Ops::Base::CACHE_BUF_SIZE),
                OP_LOGE(context_->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lld.",
                        compileInfo->ubSize, Ops::Base::CACHE_BUF_SIZE),
                return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;
    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context_);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    compileInfo->vRegSize = Ops::Base::GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::TilingReduce()
{
    if (dCosFlag_ == DCOS_FLAG_OFF) {
        return ge::GRAPH_SUCCESS;
    }
    auto compileInfo = context_->GetCompileInfo<ApplyRotaryPosEmbGradCompileInfo>();
    Ops::Base::ReduceOpCompileInfo opInfo;
    if (compileInfo == nullptr) {
        OP_CHECK_IF((GetReduceOpCompileInfo(&opInfo) == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "GetReduceOpCompileInfo failed"), return ge::GRAPH_FAILED);
    } else {
        opInfo = compileInfo->opInfo;
    }
    Ops::Base::ReduceOpInputParam opInput;
    OP_CHECK_IF((GetInputParam(opInput, GQ_INDEX, COS_INDEX) == ge::GRAPH_FAILED),
                OP_LOGE(context_->GetNodeName(), "ReduceOp get input param failed"), return ge::GRAPH_FAILED);
    ge::graphStatus status;
    if (reduceInputFloat_) {
        opInput.inputDtype = ge::DT_FLOAT;
        opInput.shape[opInput.shape.size() - N_AXIS_OFFSET_FROM_END] = std::max(nQ_, nK_);
    }
    if (dtype_ == ge::DT_FLOAT) {
        status =
            Ops::Base::Tiling4ReduceOp<ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradDag<float, float, float>::OpDag>(
                context_, opInput, key_, &opInfo, &tilingData_->reduceTiling);
    } else if (reduceInputFloat_) {
        status = Ops::Base::Tiling4ReduceOp<
            ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradDag<float, Ops::Base::half, float>::OpDag>(
            context_, opInput, key_, &opInfo, &tilingData_->reduceTiling);
    } else {
        status = Ops::Base::Tiling4ReduceOp<
            ApplyRotaryPosEmbGrad::ApplyRotaryPosEmbGradDag<Ops::Base::half, Ops::Base::half, float>::OpDag>(
            context_, opInput, key_, &opInfo, &tilingData_->reduceTiling);
    }
    OP_CHECK_IF((status == ge::GRAPH_FAILED), OP_LOGE(context_->GetNodeName(), "ReduceOp Tiling failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbGradRegbaseTilingClass::SetTilingKeyBlockDim(uint32_t dxTilingKey)
{
    GEN_REDUCE_TILING_KEY(tilingKey_, key_, dxTilingKey, dCosFlag_);
    OP_LOGI(context_->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, dxTilingKey is: %u, tilingKey is:%lu.", key_.patternID,
            key_.loopARCount, key_.loopInnerARCount, dxTilingKey, tilingKey_);
    int64_t reduceBlockNum = context_->GetBlockDim();
    context_->SetBlockDim(std::max(usedCoreNum_, reduceBlockNum));
    OP_LOGD(context_->GetNodeName(), "reduceBlockNum :%ld, usedCoreNum_ = %ld.\n", reduceBlockNum, usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    auto partialTypeSize = (dCosFlag_ == DCOS_FLAG_ON && IsBroadcastPartialFloatTemplate(dxTilingKey)) ?
                               PARTIAL_TYPE_SIZE :
                               ge::GetSizeByDataType(dtype_);
    auto usrWorkSpaceSize = b_ * s_ * std::max(nQ_, nK_) * d_ * partialTypeSize * INPUT_OUTPUT_NUM;
    if (dCosFlag_ == DCOS_FLAG_ON) {
        context_->SetScheduleMode(SCHEDULE_MODE_BATCH); // batch mode, all cores start simultaneously
        if (dxTilingKey == static_cast<uint32_t>(ApplyRopeGradDxTilingKey::TILING_KEY_A)) {
            constexpr int64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
            workspaces[0] = workspaces[0] + SYS_WORKSPACE_SIZE;
        }
    }
    workspaces[0] = workspaces[0] + usrWorkSpaceSize;
    OP_LOGD(context_->GetNodeName(), "workspaces[0] :%ld, usrWorkSpaceSize = %ld.\n", workspaces[0], usrWorkSpaceSize);
    return ge::GRAPH_SUCCESS;
}

uint64_t ApplyRotaryPosEmbGradRegbaseTilingClass::GetTilingKey() const
{
    return tilingKey_;
}

// =================================================================
// Tiling entry point
// =================================================================

ge::graphStatus Tiling4ApplyRotaryPosEmbGrad(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

template <typename ContextT>
ge::graphStatus TilingPrepare4ReduceOp(ContextT *context, Ops::Base::ReduceOpCompileInfo *compileInfo)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->vectorCoreNum == 0UL),
                OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu",
                        compileInfo->vectorCoreNum),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= Ops::Base::CACHE_BUF_SIZE,
                OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lu.",
                        compileInfo->ubSize, Ops::Base::CACHE_BUF_SIZE),
                return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;
    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
            compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
            compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForApplyRotaryPosEmbGrad(gert::TilingParseContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto compileInfoPtr = context->GetCompiledInfo<ApplyRotaryPosEmbGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compile info is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    compileInfoPtr->numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfoPtr->ubSize = ubSizePlatForm;
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        return TilingPrepare4ReduceOp(context, &compileInfoPtr->opInfo);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyRotaryPosEmbGrad)
    .Tiling(Tiling4ApplyRotaryPosEmbGrad)
    .TilingParse<ApplyRotaryPosEmbGradCompileInfo>(TilingPrepareForApplyRotaryPosEmbGrad);
} // namespace optiling
