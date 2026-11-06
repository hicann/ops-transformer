/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fused_infer_attention_score_tiling_check_arch31.cpp
 * \brief arch31 (KirinX90 / Kirin9030) 专属 Tiling 检查实现
 */

#include "fused_infer_attention_score_tiling_check_arch31.h"
#include "../fused_infer_attention_score_tiling_index.h"
#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;
namespace optiling {

namespace {
constexpr uint32_t ARCH31_D_LIMIT = 512;
constexpr uint32_t ARCH31_D_MIN = 16;
constexpr uint32_t ARCH31_D_FRAC = 16;
constexpr uint32_t ARCH31_SPARSE_MODE_NOMASK = 0;
constexpr uint32_t ARCH31_SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t ARCH31_SPARSE_MODE_LEFTUP_CAUSAL = 2;
constexpr uint32_t ARCH31_SPARSE_MODE_CAUSAL = 3;
constexpr uint32_t ARCH31_SPARSE_MODE_BAND = 4;
constexpr uint32_t ARCH31_SPARSE_MODE_TREE_MASK = 9;
constexpr int64_t ARCH31_PREFIX_LEN = 0;
constexpr uint32_t ARCH31_MASK_S2_ALIGN = 32;
constexpr uint32_t ARCH31_TND_MAX_BATCH = 128;
constexpr int64_t ARCH31_MASK_2048 = 2048;
constexpr int64_t ARCH31_TOKEN_ATTR_DEFAULT = 2147483647;
struct Arch31Dims {
    uint32_t b = 0;
    uint32_t heads = 0;
    uint32_t kvHeads = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t dQ = 0;
    uint32_t dK = 0;
    uint32_t dV = 0;
};

bool ParseLayoutDimsArch31(const gert::Shape& queryShape, const gert::Shape& keyShape, const gert::Shape& valueShape,
                           const std::string& layout, uint32_t headsAttr, uint32_t kvHeadsAttr, Arch31Dims& dims,
                           const int64_t* seqQVals = nullptr, uint32_t seqQNum = 0, const int64_t* seqKvVals = nullptr,
                           uint32_t seqKvNum = 0)
{
    if (layout == "BNSD") {
        if (queryShape.GetDimNum() != 4 || keyShape.GetDimNum() != 4 || valueShape.GetDimNum() != 4) {
            return false;
        }
        dims.b = static_cast<uint32_t>(queryShape.GetDim(0));
        dims.heads = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.s1 = static_cast<uint32_t>(queryShape.GetDim(2));
        dims.dQ = static_cast<uint32_t>(queryShape.GetDim(3));
        dims.kvHeads = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.s2 = static_cast<uint32_t>(keyShape.GetDim(2));
        dims.dK = static_cast<uint32_t>(keyShape.GetDim(3));
        dims.dV = static_cast<uint32_t>(valueShape.GetDim(3));
        return true;
    } else if (layout == "BSND") {
        if (queryShape.GetDimNum() != 4 || keyShape.GetDimNum() != 4 || valueShape.GetDimNum() != 4) {
            return false;
        }
        dims.b = static_cast<uint32_t>(queryShape.GetDim(0));
        dims.s1 = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.heads = static_cast<uint32_t>(queryShape.GetDim(2));
        dims.dQ = static_cast<uint32_t>(queryShape.GetDim(3));
        dims.s2 = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.kvHeads = static_cast<uint32_t>(keyShape.GetDim(2));
        dims.dK = static_cast<uint32_t>(keyShape.GetDim(3));
        dims.dV = static_cast<uint32_t>(valueShape.GetDim(3));
        return true;
    } else if (layout == "BSH") {
        if (queryShape.GetDimNum() != 3 || keyShape.GetDimNum() != 3 || valueShape.GetDimNum() != 3) {
            return false;
        }
        if (headsAttr == 0 || kvHeadsAttr == 0) {
            return false;
        }
        int64_t qFused = queryShape.GetDim(2);
        int64_t kFused = keyShape.GetDim(2);
        int64_t vFused = valueShape.GetDim(2);
        if (qFused % static_cast<int64_t>(headsAttr) != 0 || kFused % static_cast<int64_t>(kvHeadsAttr) != 0 ||
            vFused % static_cast<int64_t>(kvHeadsAttr) != 0) {
            return false;
        }
        dims.b = static_cast<uint32_t>(queryShape.GetDim(0));
        dims.s1 = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.heads = headsAttr;
        dims.dQ = static_cast<uint32_t>(qFused / static_cast<int64_t>(headsAttr));
        dims.s2 = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.kvHeads = kvHeadsAttr;
        dims.dK = static_cast<uint32_t>(kFused / static_cast<int64_t>(kvHeadsAttr));
        dims.dV = static_cast<uint32_t>(vFused / static_cast<int64_t>(kvHeadsAttr));
        return true;
    } else if (layout == "TND") {
        if (queryShape.GetDimNum() != 3 || keyShape.GetDimNum() != 3 || valueShape.GetDimNum() != 3) {
            return false;
        }
        if (seqQVals == nullptr || seqKvVals == nullptr || seqQNum == 0 || seqQNum != seqKvNum) {
            return false;
        }
        dims.b = seqQNum;
        dims.heads = static_cast<uint32_t>(queryShape.GetDim(1));
        dims.dQ = static_cast<uint32_t>(queryShape.GetDim(2));
        dims.kvHeads = static_cast<uint32_t>(keyShape.GetDim(1));
        dims.dK = static_cast<uint32_t>(keyShape.GetDim(2));
        dims.dV = static_cast<uint32_t>(valueShape.GetDim(2));
        uint32_t s1Max = 0;
        uint32_t s2Max = 0;
        for (uint32_t i = 0; i < seqQNum; i++) {
            int64_t qEnd = seqQVals[i];
            int64_t qBegin = (i == 0) ? 0 : seqQVals[i - 1];
            int64_t kvEnd = seqKvVals[i];
            int64_t kvBegin = (i == 0) ? 0 : seqKvVals[i - 1];
            uint32_t s1 = (qEnd > qBegin) ? static_cast<uint32_t>(qEnd - qBegin) : 0;
            uint32_t s2 = (kvEnd > kvBegin) ? static_cast<uint32_t>(kvEnd - kvBegin) : 0;
            s1Max = (s1 > s1Max) ? s1 : s1Max;
            s2Max = (s2 > s2Max) ? s2 : s2Max;
        }
        dims.s1 = s1Max;
        dims.s2 = s2Max;
        return true;
    } else {
        return false;
    }
}

bool IsDecodeSparseBypass(const gert::TilingContext* context, const Arch31Dims& dims, const std::string& inputLayout)
{
    if (inputLayout != "BNSD" && inputLayout != "BSND" && inputLayout != "BSH") {
        return false;
    }
    if (dims.s1 != 1) {
        return false;
    }
    if (dims.dQ != dims.dK || dims.dQ != dims.dV) {
        return false;
    }
    if (context->GetOptionalInputShape(QUERY_ROPE_INDEX) != nullptr ||
        context->GetOptionalInputShape(KEY_ROPE_INDEX) != nullptr) {
        return false;
    }
    return true;
}
} // namespace

ge::graphStatus CheckArch31Inputs(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 1. 检查 query 数据类型（仅FP16）
    auto queryDataType = context->GetInputDesc(0)->GetDataType();
    if (queryDataType != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "query",
                                              std::to_string(static_cast<int>(queryDataType)).c_str(),
                                              "arch31 only supports FLOAT16 query");
        return ge::GRAPH_FAILED;
    }

    // 2. 属性读取
    auto attrs = context->GetAttrs();
    int64_t sparseMode = 0;
    int64_t preTokens = ARCH31_TOKEN_ATTR_DEFAULT;
    int64_t nextTokens = ARCH31_TOKEN_ATTR_DEFAULT;
    std::string layout = "BNSD";
    uint32_t headsAttr = 0;
    uint32_t kvHeadsAttr = 0;

    if (attrs != nullptr) {
        auto sparseAttr = attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX);
        if (sparseAttr != nullptr) {
            sparseMode = *sparseAttr;
        }
        auto preAttr = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
        if (preAttr != nullptr) {
            preTokens = *preAttr;
        }
        auto nextAttr = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
        if (nextAttr != nullptr) {
            nextTokens = *nextAttr;
        }
        auto layoutAttr = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
        if (layoutAttr != nullptr) {
            layout = std::string(layoutAttr);
        }
        auto headsAttrPtr = attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX);
        if (headsAttrPtr != nullptr) {
            headsAttr = static_cast<uint32_t>(*headsAttrPtr);
        }
        auto kvHeadsAttrPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX);
        if (kvHeadsAttrPtr != nullptr) {
            kvHeadsAttr = static_cast<uint32_t>(*kvHeadsAttrPtr);
        }
    }

    // 3. 检查 layout 白名单 {BNSD, BSND, BSH, TND}
    if (layout != "BNSD" && layout != "BSND" && layout != "BSH" && layout != "TND") {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "input_layout", layout.c_str(),
                                               "arch31 supports BNSD, BSND, BSH or TND");
        return ge::GRAPH_FAILED;
    }

    // 4. layout shape 解析
    auto queryShape = context->GetInputShape(0)->GetStorageShape();
    auto keyShape = context->GetInputShape(1)->GetStorageShape();
    auto valueShape = context->GetInputShape(2)->GetStorageShape();
    const int64_t* seqQVals = nullptr;
    const int64_t* seqKvVals = nullptr;
    uint32_t seqQNum = 0;
    uint32_t seqKvNum = 0;
    if (layout == "TND") {
        auto seqQShape = context->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
        auto seqKvShape = context->GetOptionalInputShape(ACTUAL_SEQ_KV_INDEX);
        auto seqQTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
        auto seqKvTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
        if (seqQShape == nullptr || seqKvShape == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "actual_seq_lengths/actual_seq_lengths_kv");
            return ge::GRAPH_FAILED;
        }
        seqQNum = static_cast<uint32_t>(seqQShape->GetStorageShape().GetDim(0));
        seqKvNum = static_cast<uint32_t>(seqKvShape->GetStorageShape().GetDim(0));
        if (seqQTensor != nullptr) {
            seqQVals = seqQTensor->GetData<int64_t>();
        }
        if (seqKvTensor != nullptr) {
            seqKvVals = seqKvTensor->GetData<int64_t>();
        }
        if (seqQVals == nullptr || seqKvVals == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "actual_seq_lengths/actual_seq_lengths_kv");
            return ge::GRAPH_FAILED;
        }
        if (seqQNum > ARCH31_TND_MAX_BATCH || seqQNum != seqKvNum) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "actual_seq_lengths,actual_seq_lengths_kv",
                (std::string("batch=") + std::to_string(seqQNum) + ", kv batch=" + std::to_string(seqKvNum)).c_str(),
                "arch31 TND batch must not exceed " + std::to_string(ARCH31_TND_MAX_BATCH) +
                    " and must match actual_seq_lengths_kv count");
            return ge::GRAPH_FAILED;
        }
    }

    Arch31Dims dims;
    if (!ParseLayoutDimsArch31(queryShape, keyShape, valueShape, layout, headsAttr, kvHeadsAttr, dims, seqQVals,
                               seqQNum, seqKvVals, seqKvNum)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "query,key,value",
            (std::string("query dimNum=") + std::to_string(queryShape.GetDimNum()) + ", key dimNum=" +
             std::to_string(keyShape.GetDimNum()) + ", value dimNum=" + std::to_string(valueShape.GetDimNum()))
                .c_str(),
            "arch31 layout " + layout +
                " expects BNSD/BSND 4D or BSH 3D with num_heads/kv_head_nums and dim2 divisible by heads, or TND 3D");
        return ge::GRAPH_FAILED;
    }

    // 5. 检查 D 值域：16 ≤ D ≤ 512 且为16的倍数
    if (dims.dQ < ARCH31_D_MIN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "D", std::to_string(dims.dQ).c_str(),
                                              "arch31 D must be >= " + std::to_string(ARCH31_D_MIN));
        return ge::GRAPH_FAILED;
    }
    if (dims.dQ % ARCH31_D_FRAC != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "D", std::to_string(dims.dQ).c_str(),
                                              "arch31 D must be a multiple of " + std::to_string(ARCH31_D_FRAC));
        return ge::GRAPH_FAILED;
    }
    if (dims.dQ > ARCH31_D_LIMIT) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "D", std::to_string(dims.dQ).c_str(),
                                              "arch31 D must be <= " + std::to_string(ARCH31_D_LIMIT));
        return ge::GRAPH_FAILED;
    }
    bool decodeGateHit = IsDecodeSparseBypass(context, dims, layout);
    if (decodeGateHit) {
        if (context->GetOptionalInputShape(ATTEN_MASK_INDEX) == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
    }

    // 6. TND sm/mask 校验
    int64_t tndMaskSumSq = 0;
    if (layout == "TND") {
        if (sparseMode != ARCH31_SPARSE_MODE_NOMASK && sparseMode != ARCH31_SPARSE_MODE_TREE_MASK) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparse_mode",
                                                  std::to_string(sparseMode).c_str(),
                                                  "arch31 TND layout sparse_mode must be 0 or 9");
            return ge::GRAPH_FAILED;
        }
        int64_t qT = queryShape.GetDim(0);
        int64_t kT = keyShape.GetDim(0);
        int64_t vT = valueShape.GetDim(0);
        for (uint32_t i = 0; i < seqQNum; i++) {
            if (seqQVals[i] < 0 || seqKvVals[i] < 0) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "actual_seq_lengths,actual_seq_lengths_kv",
                    (std::string("seqQ[") + std::to_string(i) + "]=" + std::to_string(seqQVals[i]) + ", seqKv[" +
                     std::to_string(i) + "]=" + std::to_string(seqKvVals[i]))
                        .c_str(),
                    "arch31 TND actual_seq_lengths values must be non-negative");
                return ge::GRAPH_FAILED;
            }
            if (i > 0 && (seqQVals[i] < seqQVals[i - 1] || seqKvVals[i] < seqKvVals[i - 1])) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "actual_seq_lengths,actual_seq_lengths_kv",
                    (std::string("seqQ[") + std::to_string(i) + "]=" + std::to_string(seqQVals[i]) + ", seqQ[" +
                     std::to_string(i - 1) + "]=" + std::to_string(seqQVals[i - 1]) + ", seqKv[" + std::to_string(i) +
                     "]=" + std::to_string(seqKvVals[i]) + ", seqKv[" + std::to_string(i - 1) +
                     "]=" + std::to_string(seqKvVals[i - 1]))
                        .c_str(),
                    "arch31 TND actual_seq_lengths must be non-decreasing");
                return ge::GRAPH_FAILED;
            }
        }
        if (seqQVals[seqQNum - 1] != qT || seqKvVals[seqKvNum - 1] != kT || seqKvVals[seqKvNum - 1] != vT) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "actual_seq_lengths,actual_seq_lengths_kv",
                (std::string("seqQ_last=") + std::to_string(seqQVals[seqQNum - 1]) + ", query T=" + std::to_string(qT) +
                 ", seqKv_last=" + std::to_string(seqKvVals[seqKvNum - 1]) + ", key T=" + std::to_string(kT) +
                 ", value T=" + std::to_string(vT))
                    .c_str(),
                "arch31 TND actual_seq_lengths last element must equal token count T");
            return ge::GRAPH_FAILED;
        }
        if (sparseMode == ARCH31_SPARSE_MODE_NOMASK && context->GetOptionalInputShape(ATTEN_MASK_INDEX) != nullptr) {
            REPORT_PREDEFINED_ERR_MSG(
                "EZ1009", (std::vector<const char*>{"opName", "reason"}),
                (std::vector<const char*>{context->GetNodeName(),
                                          "arch31 TND layout with sparse_mode=0 does not consume atten_mask"}));
            return ge::GRAPH_FAILED;
        }
        if (sparseMode == ARCH31_SPARSE_MODE_TREE_MASK) {
            for (uint32_t i = 0; i < seqQNum; i++) {
                int64_t s1 = seqQVals[i] - ((i == 0) ? 0 : seqQVals[i - 1]);
                int64_t s2 = seqKvVals[i] - ((i == 0) ? 0 : seqKvVals[i - 1]);
                if (s2 != 0 && s1 > s2) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context->GetNodeName(), "actual_seq_lengths,actual_seq_lengths_kv",
                        (std::string("batch=") + std::to_string(i) + ", Q_S=" + std::to_string(s1) +
                         ", KV_S=" + std::to_string(s2))
                            .c_str(),
                        "arch31 TND sparse_mode=9 requires Q_S<=KV_S per batch");
                    return ge::GRAPH_FAILED;
                }
                tndMaskSumSq += s1 * s1;
            }
        }
    }

    // 7. 检查 sparseMode ∈ {0, 3, 4, 9}
    if (sparseMode != ARCH31_SPARSE_MODE_NOMASK && sparseMode != ARCH31_SPARSE_MODE_ALL_MASK &&
        sparseMode != ARCH31_SPARSE_MODE_LEFTUP_CAUSAL && sparseMode != ARCH31_SPARSE_MODE_CAUSAL &&
        sparseMode != ARCH31_SPARSE_MODE_BAND && sparseMode != ARCH31_SPARSE_MODE_TREE_MASK) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparse_mode", std::to_string(sparseMode).c_str(),
                                              "arch31 sparse_mode must be one of {0,1,2,3,4,9}");
        return ge::GRAPH_FAILED;
    }

    // 8. mask校验
    auto attenMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    if (attenMaskShape != nullptr) {
        auto maskDesc = context->GetOptionalInputDesc(ATTEN_MASK_INDEX);
        ge::DataType maskDtype = (maskDesc != nullptr) ? maskDesc->GetDataType() : ge::DT_UNDEFINED;
        if (maskDtype != ge::DT_BOOL && maskDtype != ge::DT_INT8 && maskDtype != ge::DT_UINT8) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "atten_mask",
                                                  std::to_string(static_cast<int>(maskDtype)).c_str(),
                                                  "arch31 atten_mask dtype must be BOOL, INT8 or UINT8");
            return ge::GRAPH_FAILED;
        }
        auto maskShape = attenMaskShape->GetStorageShape();
        if (sparseMode == ARCH31_SPARSE_MODE_TREE_MASK) {
            if (layout == "TND") {
                if (maskShape.GetDimNum() != 1 || maskShape.GetDim(0) != tndMaskSumSq) {
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "atten_mask",
                        (std::string("dimNum=") + std::to_string(maskShape.GetDimNum()) +
                         ", dim0=" + std::to_string(maskShape.GetDim(0)))
                            .c_str(),
                        "arch31 TND sparse_mode=9 atten_mask must be 1D packed triangular mask with dim0=" +
                            std::to_string(tndMaskSumSq));
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (maskShape.GetDimNum() != 3 || maskShape.GetDim(0) != static_cast<int64_t>(dims.b) ||
                    maskShape.GetDim(1) != static_cast<int64_t>(dims.s1) ||
                    maskShape.GetDim(2) != static_cast<int64_t>(dims.s1)) {
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context->GetNodeName(), "atten_mask",
                        (std::string("dimNum=") + std::to_string(maskShape.GetDimNum()) + ", dims=(" +
                         std::to_string(maskShape.GetDimNum() > 0 ? maskShape.GetDim(0) : -1) + "," +
                         std::to_string(maskShape.GetDimNum() > 1 ? maskShape.GetDim(1) : -1) + "," +
                         std::to_string(maskShape.GetDimNum() > 2 ? maskShape.GetDim(2) : -1) + ")")
                            .c_str(),
                        "arch31 sparse_mode=9 atten_mask must be 3D (B,S1,S1) with B=" + std::to_string(dims.b) +
                            ", S1=" + std::to_string(dims.s1));
                    return ge::GRAPH_FAILED;
                }
            }
        } else if (sparseMode == ARCH31_SPARSE_MODE_LEFTUP_CAUSAL || sparseMode == ARCH31_SPARSE_MODE_CAUSAL ||
                   sparseMode == ARCH31_SPARSE_MODE_BAND) {
            bool shapeOk = false;
            if (maskShape.GetDimNum() == 2) {
                shapeOk = maskShape.GetDim(0) == ARCH31_MASK_2048 && maskShape.GetDim(1) == ARCH31_MASK_2048;
            } else if (maskShape.GetDimNum() == 3) {
                shapeOk = maskShape.GetDim(1) == ARCH31_MASK_2048 && maskShape.GetDim(2) == ARCH31_MASK_2048 &&
                          (maskShape.GetDim(0) == 1 || maskShape.GetDim(0) == static_cast<int64_t>(dims.b));
            } else if (maskShape.GetDimNum() == 4) {
                shapeOk = maskShape.GetDim(1) == 1 && maskShape.GetDim(2) == ARCH31_MASK_2048 &&
                          maskShape.GetDim(3) == ARCH31_MASK_2048 &&
                          (maskShape.GetDim(0) == 1 || maskShape.GetDim(0) == static_cast<int64_t>(dims.b));
            }
            if (!shapeOk) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "atten_mask",
                    (std::string("dimNum=") + std::to_string(maskShape.GetDimNum())).c_str(),
                    "arch31 sparse_mode=" + std::to_string(sparseMode) +
                        " atten_mask shape must be (2048,2048), (1,2048,2048) or (1,1,2048,2048)");
                return ge::GRAPH_FAILED;
            }
        } else {
            bool shapeOk = false;
            if (maskShape.GetDimNum() == 3) {
                shapeOk = maskShape.GetDim(1) == static_cast<int64_t>(dims.s1) &&
                          maskShape.GetDim(2) == static_cast<int64_t>(dims.s2) &&
                          (maskShape.GetDim(0) == static_cast<int64_t>(dims.b) || maskShape.GetDim(0) == 1);
            } else if (maskShape.GetDimNum() == 4) {
                shapeOk = maskShape.GetDim(1) == 1 && maskShape.GetDim(2) == static_cast<int64_t>(dims.s1) &&
                          maskShape.GetDim(3) == static_cast<int64_t>(dims.s2) &&
                          (maskShape.GetDim(0) == static_cast<int64_t>(dims.b) || maskShape.GetDim(0) == 1);
            }
            if (!shapeOk) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "atten_mask",
                    (std::string("dimNum=") + std::to_string(maskShape.GetDimNum())).c_str(),
                    "arch31 atten_mask shape must be (B,S1,S2), (1,S1,S2), (B,1,S1,S2) or (1,1,S1,S2) with S1=" +
                        std::to_string(dims.s1) + ", S2=" + std::to_string(dims.s2) + ", B=" + std::to_string(dims.b));
                return ge::GRAPH_FAILED;
            }
            if (dims.s2 % ARCH31_MASK_S2_ALIGN != 0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "KV_S", std::to_string(dims.s2).c_str(),
                                                      "arch31 KV_S must be a multiple of " +
                                                          std::to_string(ARCH31_MASK_S2_ALIGN) +
                                                          " (attenMask GM row base 32B alignment)");
                return ge::GRAPH_FAILED;
            }
        }
    }

    // 9. sm=1 必传 mask
    if (sparseMode == ARCH31_SPARSE_MODE_ALL_MASK && attenMaskShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "atten_mask");
        return ge::GRAPH_FAILED;
    }
    // sm=2,3,4 必传 mask
    if ((sparseMode == ARCH31_SPARSE_MODE_LEFTUP_CAUSAL || sparseMode == ARCH31_SPARSE_MODE_CAUSAL ||
         sparseMode == ARCH31_SPARSE_MODE_BAND) &&
        attenMaskShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "atten_mask");
        return ge::GRAPH_FAILED;
    }
    // sm=9 必传 mask
    if (sparseMode == ARCH31_SPARSE_MODE_TREE_MASK && attenMaskShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "atten_mask");
        return ge::GRAPH_FAILED;
    }
    if (sparseMode == ARCH31_SPARSE_MODE_TREE_MASK && layout != "TND" && dims.s2 != 0 && dims.s1 > dims.s2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparse_mode", std::to_string(sparseMode).c_str(),
                                              "arch31 sparse_mode=9 requires Q_S<=KV_S, got Q_S=" +
                                                  std::to_string(dims.s1) + " > KV_S=" + std::to_string(dims.s2));
        return ge::GRAPH_FAILED;
    }
    // 10. sm=0 且mask 存在
    if (sparseMode == ARCH31_SPARSE_MODE_NOMASK && attenMaskShape != nullptr) {
        int64_t windowFirst =
            static_cast<int64_t>(dims.s1) - static_cast<int64_t>(dims.s2) - ARCH31_PREFIX_LEN - preTokens;
        if (windowFirst > 0 || nextTokens < 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "pre_token,next_token",
                (std::string("pre_token=") + std::to_string(preTokens) + ", next_token=" + std::to_string(nextTokens) +
                 ", windowFirst=" + std::to_string(windowFirst))
                    .c_str(),
                "arch31 sparse_mode=0 illegal mask window: S1(" + std::to_string(dims.s1) + ")-S2(" +
                    std::to_string(dims.s2) + ")-prefixLen(" + std::to_string(ARCH31_PREFIX_LEN) +
                    ")-pre_token must be <= 0 and next_token must be >= 0");
            return ge::GRAPH_FAILED;
        }
    }

    // 11. sparseMode=4
    if (sparseMode == ARCH31_SPARSE_MODE_BAND) {
        int64_t bandSecond =
            nextTokens + static_cast<int64_t>(dims.s2) + ARCH31_PREFIX_LEN - static_cast<int64_t>(dims.s1);
        if (preTokens < 0 || bandSecond < 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "pre_token,next_token",
                (std::string("pre_token=") + std::to_string(preTokens) + ", next_token=" + std::to_string(nextTokens) +
                 ", bandSecond=" + std::to_string(bandSecond))
                    .c_str(),
                "arch31 sparse_mode=4 illegal band window: pre_token must be >= 0 and next_token+KV_S(" +
                    std::to_string(dims.s2) + ")+prefixLen(" + std::to_string(ARCH31_PREFIX_LEN) + ")-Q_S(" +
                    std::to_string(dims.s1) + ") must be >= 0");
            return ge::GRAPH_FAILED;
        }
    }

    // 12. 非PagedAttention
    auto blockTableInput = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    auto queryPaddingSizeInput = context->GetOptionalInputShape(QUERY_PADDING_SIZE_INDEX);
    auto kvPaddingSizeInput = context->GetOptionalInputShape(KV_PADDING_SIZE_INDEX);
    if (blockTableInput != nullptr || queryPaddingSizeInput != nullptr || kvPaddingSizeInput != nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context->GetNodeName(),
                                      "arch31 does not support PagedAttention (block_table/query_padding_size/"
                                      "kv_padding_size input present)"}));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
