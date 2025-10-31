/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fa_param.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad 参数信息.
 */

#include "fa_param.h"
#include "tests/utils/log.h"

using Tensor = ops::adv::tests::utils::Tensor;

using namespace ops::adv::tests::fa;

FaParam::FaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType)
    : FaParam(pB, pN2, pG, pS1, pS2, pD, pDtype, pLayoutType, pScale, pKeepProb, pPreTokens, pNxtTokens, pInnerPrecise,
              pSparseMode, 1, pPseShapeType, pDropMaskShapeType, pPaddingMaskShapeType, pAttenMaskShapeType,
              pAttenMaskDtype, pPrefixShapeType, {}, {}, {}, {}, {})
{
}

FaParam::FaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType,
                 std::vector<int64_t> pPrefixTensorData, std::vector<int64_t> pActualSeqQLenList,
                 std::vector<int64_t> pActualSeqKvLenList)
    : b(pB), n2(pN2), g(pG), s1(pS1), s2(pS2), d(pD), dtype(pDtype), layoutType(pLayoutType), scale(pScale),
      keepProb(pKeepProb), preTokens(pPreTokens), nxtTokens(pNxtTokens), innerPrecise(pInnerPrecise),
      sparseMode(pSparseMode), pseShapeType(pPseShapeType), dropMaskShapeType(pDropMaskShapeType),
      paddingMaskShapeType(pPaddingMaskShapeType), attenMaskShapeType(pAttenMaskShapeType),
      attenMaskDtype(pAttenMaskDtype), prefixShapeType(pPrefixShapeType),
      prefixTensorData(std::move(pPrefixTensorData)), actualSeqQLenList(std::move(pActualSeqQLenList)),
      actualSeqKVLenList(std::move(pActualSeqKvLenList))
{
}

FaParam::FaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, int64_t pPseType, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType,
                 std::vector<int64_t> pPrefixTensorData, std::vector<int64_t> pActualSeqQLenList,
                 std::vector<int64_t> pActualSeqKvLenList, std::vector<int64_t> pQStartIdxTensorData,
                 std::vector<int64_t> pKVStartIdxTensorData)
    : b(pB), n2(pN2), g(pG), s1(pS1), s2(pS2), d(pD), dtype(pDtype), layoutType(pLayoutType), scale(pScale),
      keepProb(pKeepProb), preTokens(pPreTokens), nxtTokens(pNxtTokens), innerPrecise(pInnerPrecise),
      sparseMode(pSparseMode), pseType(pPseType), pseShapeType(pPseShapeType), dropMaskShapeType(pDropMaskShapeType),
      paddingMaskShapeType(pPaddingMaskShapeType), attenMaskShapeType(pAttenMaskShapeType),
      attenMaskDtype(pAttenMaskDtype), prefixShapeType(pPrefixShapeType),
      prefixTensorData(std::move(pPrefixTensorData)), actualSeqQLenList(std::move(pActualSeqQLenList)),
      actualSeqKVLenList(std::move(pActualSeqKvLenList)), qStartIdxTensorData(std::move(pQStartIdxTensorData)),
      kvStartIdxTensorData(std::move(pKVStartIdxTensorData))
{
}

bool FaParam::Init()
{
    n1 = n2 * g;
    h1 = n1 * d;
    h2 = n2 * d;
    h1_rope = n1 * rope_d;
    h2_rope = n2 * rope_d;

    /* 由于是自动反向用例, 对于 Fas/Fag 均需使用的 Tensor, 以正向视角设置 TensorType */

    std::vector<int64_t> layout1;
    std::vector<int64_t> layout2;
    std::vector<int64_t> layout1_rope;
    std::vector<int64_t> layout2_rope;
    switch (layoutType) {
        case LayoutType::BSH:
            layout = "BSH";
            layout1 = {b, s1, h1};
            layout2 = {b, s2, h2};
            layout1_rope = {b, s1, h1_rope};
            layout2_rope = {b, s2, h2_rope};
            break;
        case LayoutType::SBH:
            layout = "SBH";
            layout1 = {s1, b, h1};
            layout2 = {s2, b, h2};
            layout1_rope = {s1, b, h1_rope};
            layout2_rope = {s2, b, h2_rope};
            break;
        case LayoutType::BNSD:
            layout = "BNSD";
            layout1 = {b, n1, s1, d};
            layout2 = {b, n2, s2, d};
            layout1_rope = {b, n1, s1, rope_d};
            layout2_rope = {b, n2, s2, rope_d};
            break;
        case LayoutType::BSND:
            layout = "BSND";
            layout1 = {b, s1, n1, d};
            layout2 = {b, s2, n2, d};
            layout1_rope = {b, s1, n1, rope_d};
            layout2_rope = {b, s2, n2, rope_d};
            break;
        case LayoutType::TND:
            layout = "TND";

            /* 按 Q, KV 生成累加序列 */
            for (long &it : actualSeqQLenList) {
                auto pre = actualSeqQLenTensorData.empty() ? 0 : actualSeqQLenTensorData.back();
                actualSeqQLenTensorData.push_back(it + pre);
                t1 += it;
            }
            for (long &it : actualSeqKVLenList) {
                auto pre = actualSeqKVLenTensorData.empty() ? 0 : actualSeqKVLenTensorData.back();
                actualSeqKVLenTensorData.push_back(it + pre);
                t2 += it;
            }

            for (auto it1 = actualSeqQLenList.begin(), it2 = actualSeqKVLenList.begin();
                 it1 != actualSeqQLenList.end() && it2 != actualSeqKVLenList.end(); ++it1, ++it2) {
                productS1S2 += *it1 * *it2;
            }
            layout1 = {t1, n1, d};
            layout2 = {t2, n2, d};
            layout1_rope = {t1, n1, rope_d};
            layout2_rope = {t2, n2, rope_d};
            break;
        default:
            LOG_ERR("Unknown LayoutType=%d", static_cast<int32_t>(layoutType));
            return false;
    }
    query = Tensor("query", layout1, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    key = Tensor("key", layout2, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    value = Tensor("value", layout2, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    dy = Tensor("dy", layout1, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    attenRes = Tensor("atten_res", layout1, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    dq = Tensor("dq", layout1, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    dk = Tensor("dk", layout2, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);
    dv = Tensor("dv", layout2, layout.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

    std::string layoutPseStr;
    std::vector<int64_t> layoutPse;
    switch (pseShapeType) {
        case PseShapeType::B_N1_S1_S2:
            layoutPseStr = "B_N1_S1_S2";
            layoutPse = {b, n1, s1, s2};
            break;
        case PseShapeType::B_N1_1_S2:
            layoutPseStr = "B_N1_1_S2";
            layoutPse = {b, n1, 1, s2};
            break;
        case PseShapeType::B_N1_ALIBI_S1_S2:
            layoutPseStr = "B_N1_ALIBI_S1_S2";
            layoutPse = {b, n1, kPseAlibiS1, s2};
            break;
        case PseShapeType::_1_N1_ALIBI_S1_S2:
            layoutPseStr = "1_N1_ALIBI_S1_S2";
            layoutPse = {1, n1, kPseAlibiS1, s2};
            break;
        case PseShapeType::_1_N1_S1_S2:
            layoutPseStr = "1_N1_S1_S2";
            layoutPse = {1, n1, s1, s2};
            break;
        case PseShapeType::S1_S2:
            layoutPseStr = "S1_S2";
            layoutPse = {s1, s2};
            break;
        case PseShapeType::SLOPE_B_N1:
            layoutPseStr = "b_n1";
            layoutPse = {b, n1};
            break;
        case PseShapeType::SLOPE_N1:
            layoutPseStr = "n1";
            layoutPse = {n1};
            break;
        case PseShapeType::TND_1S:
            layoutPseStr = "TND_1S";
            layoutPse = {t2 * n1};
            break;
        case PseShapeType::TND_SS:
            layoutPseStr = "TND_SS";
            layoutPse = {productS1S2 * n1};
            break;
        default:
            layoutPseStr = "None";
            layoutPse = {};
            break;
    }
    if (pseType != 3 && pseType != 2) {
        pse = Tensor("pse", layoutPse, layoutPseStr.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        pse = Tensor("pse", layoutPse, layoutPseStr.c_str(), ge::DT_FLOAT, ge::FORMAT_ND,
                     Tensor::TensorType::OPTIONAL_INPUT);
    }

    dPse = Tensor("dPse", layoutPse, layoutPseStr.c_str(), dtype, ge::FORMAT_ND, Tensor::TensorType::REQUIRED_OUTPUT);

    std::string layoutDropMaskStr;
    std::vector<int64_t> layoutDropMask;
    switch (dropMaskShapeType) {
        case DropMaskShapeType::B_N1_S1_S2DIV8:
            layoutDropMaskStr = "B_N1_S1_S2DIV8/8";
            layoutDropMask = {b, n1, s1, ((s2 + 8 - 1) / 8 * 8) / 8};
            break;
        case DropMaskShapeType::B_N1_S1_S2:
            layoutDropMaskStr = "B_N1_S1_S2";
            layoutDropMask = {b * n1 * s1, s2};
            break;
        default:
            layoutDropMaskStr = "None";
            layoutDropMask = {};
            break;
    }
    dropMask = Tensor("drop_mask", layoutDropMask, layoutDropMaskStr.c_str(), ge::DT_UINT8, ge::FORMAT_ND,
                      Tensor::TensorType::OPTIONAL_INPUT);

    std::string layoutPaddingMaskStr;
    std::vector<int64_t> layoutPaddingMask;
    switch (paddingMaskShapeType) {
        case PaddingMaskShapeType::S1_S2:
            layoutPaddingMaskStr = "S1_S2";
            layoutPaddingMask = {s1, s2};
            break;
        default:
            layoutPaddingMaskStr = "None";
            layoutPaddingMask = {};
            break;
    }
    paddingMask = Tensor("padding_mask", layoutPaddingMask, layoutPaddingMaskStr.c_str(), dtype, ge::FORMAT_ND,
                         Tensor::TensorType::OPTIONAL_INPUT);

    std::string layoutAttenMaskStr;
    std::vector<int64_t> layoutAttenMask;
    switch (attenMaskShapeType) {
        case AttenMaskShapeType::S1_S2:
            layoutAttenMaskStr = "S1_S2";
            layoutAttenMask = {s1, s2};
            break;
        case AttenMaskShapeType::_1_1_S1_S2:
            layoutAttenMaskStr = "1_1_S1_S2";
            layoutAttenMask = {1, 1, s1, s2};
            break;
        case AttenMaskShapeType::B_1_S1_S2:
            layoutAttenMaskStr = "B_1_S1_S2";
            layoutAttenMask = {b, 1, s1, s2};
            break;
        case AttenMaskShapeType::B_N1_S1_S2:
            layoutAttenMaskStr = "B_N1_S1_S2";
            layoutAttenMask = {b, n1, s1, s2};
            break;
        case AttenMaskShapeType::SPARSE:
            layoutAttenMaskStr = "SPARSE";
            layoutAttenMask = {2048, 2048};
            break;
        case AttenMaskShapeType::PREFIXCOMPRESS:
            layoutAttenMaskStr = "PREFIXCOMPRESS";
            layoutAttenMask = {3072, 2048};
            break;
        default:
            layoutAttenMaskStr = "None";
            layoutAttenMask = {};
            break;
    }
    attenMask = Tensor("atten_mask", layoutAttenMask, layoutAttenMaskStr.c_str(), attenMaskDtype, ge::FORMAT_ND,
                       Tensor::TensorType::OPTIONAL_INPUT);

    std::string layoutPrefixStr;
    std::vector<int64_t> layoutPrefix;
    switch (prefixShapeType) {
        case PrefixShapeType::B:
            layoutPrefixStr = "B";
            layoutPrefix = {b};
            break;
        default:
            layoutPrefixStr = "None";
            layoutPrefix = {};
            break;
    }
    prefix = Tensor("prefix", layoutPrefix, layoutPrefixStr.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND,
                    Tensor::TensorType::OPTIONAL_INPUT);

    std::string layoutSoftmaxStr;
    std::vector<int64_t> layoutSoftmax;
    if (layoutType == LayoutType::TND) {
        layoutSoftmaxStr = "T_N1_8";
        layoutSoftmax = {t1, n1, 8};
    } else {
        layoutSoftmaxStr = "B_N1_S1_8";
        layoutSoftmax = {b, n1, s1, 8};
    }
    /* softmaxMax, softmaxSum, softmaxRes 作为正向输出, 设置为 Optional */
    softmaxMax = Tensor("softmax_max", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxSum = Tensor("softmax_sum", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);
    softmaxRes = Tensor("softmax_res", layoutSoftmax, layoutSoftmaxStr.c_str(), ge::DataType::DT_FLOAT, ge::FORMAT_ND,
                        Tensor::TensorType::REQUIRED_OUTPUT);

    if (!actualSeqQLenTensorData.empty()) {
        actualSeqQLen = Tensor("actualSeqQLen", {static_cast<int64_t>(actualSeqQLenTensorData.size())}, "B",
                               ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        actualSeqQLen = Tensor("actualSeqQLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                               Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!actualSeqKVLenTensorData.empty()) {
        actualSeqKvLen = Tensor("actualSeqKvLen", {static_cast<int64_t>(actualSeqKVLenTensorData.size())}, "B",
                                ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        actualSeqKvLen = Tensor("actualSeqKvLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND,
                                Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!qStartIdxTensorData.empty()) {
        qStartIdx =
            Tensor("qStartIdx", {b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        qStartIdx =
            Tensor("qStartIdx", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    }
    if (!kvStartIdxTensorData.empty()) {
        kvStartIdx =
            Tensor("kvStartIdx", {b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    } else {
        kvStartIdx =
            Tensor("kvStartIdx", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    }

    deqScaleQ =
            Tensor("deqScaleQ", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    deqScaleK =
            Tensor("deqScaleK", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    deqScaleV =
            Tensor("deqScaleV", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    deqScaleDy =
            Tensor("deqScaleDy", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);
    deqScaleO =
            Tensor("deqScaleO", {}, "None", ge::DataType::DT_FLOAT, ge::FORMAT_ND, Tensor::TensorType::OPTIONAL_INPUT);

    /**
     * TensorData 初始化
     * 出于性能角度考虑, 此处仅申请 Tiling 阶段必需的 TensorData
     */
    if (!ops::adv::tests::fa::FaParam::InitTensor(prefix, prefixTensorData)) {
        return false;
    }
    if (!ops::adv::tests::fa::FaParam::InitTensor(actualSeqQLen, actualSeqQLenTensorData)) {
        return false;
    }
    if (!ops::adv::tests::fa::FaParam::InitTensor(actualSeqKvLen, actualSeqKVLenTensorData)) {
        return false;
    }
    if (!ops::adv::tests::fa::FaParam::InitTensor(qStartIdx, qStartIdxTensorData)) {
        return false;
    }
    if (!ops::adv::tests::fa::FaParam::InitTensor(kvStartIdx, kvStartIdxTensorData)) {
        return false;
    }
    return true;
}

bool FaParam::IsUnPaddingAttention()
{
    return actualSeqQLen.GetDimNum() != 0 && actualSeqKvLen.GetDimNum() != 0 && !actualSeqQLenTensorData.empty() &&
           !actualSeqKVLenTensorData.empty();
}
