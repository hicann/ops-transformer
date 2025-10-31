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
 * \file fa_param.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad 参数信息.
 */

#pragma once

#include <cstdint>
#include <vector>
#include "graph/types.h"
#include "tests/utils/log.h"
#include "tests/utils/tensor.h"

namespace ops::adv::tests::fa {

class FaParam {
public:
    using Tensor = ops::adv::tests::utils::Tensor;

    /**
     * Pse Alibi 场景下 S1 取值;
     */
    static constexpr int64_t kPseAlibiS1 = 1024;

public:
    enum class PseShapeType {
        NONE,
        B_N1_S1_S2,
        B_N1_1_S2,
        B_N1_ALIBI_S1_S2,
        _1_N1_ALIBI_S1_S2,
        _1_N1_S1_S2,
        S1_S2,
        SLOPE_B_N1,
        SLOPE_N1,
        TND_1S,
        TND_SS
    };

    enum class DropMaskShapeType {
        NONE,
        B_N1_S1_S2DIV8,
        B_N1_S1_S2,
    };

    enum class PaddingMaskShapeType {
        NONE,
        S1_S2,
    };

    enum class AttenMaskShapeType {
        NONE,
        S1_S2,
        _1_1_S1_S2,
        B_1_S1_S2,
        B_N1_S1_S2,
        SPARSE,
        PREFIXCOMPRESS,
    };

    enum class PrefixShapeType {
        NONE,
        B,
    };

    enum class LayoutType {
        BSH,
        SBH,
        BNSD,
        BSND,
        TND,
    };

public:
    /* 设置参数 */
    int64_t b = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t d = 0;
    int64_t rope_d = 0;
    ge::DataType dtype = ge::DataType::DT_UNDEFINED;
    LayoutType layoutType = LayoutType::SBH;
    float scale = 0.0f;
    float keepProb = 0.0f;
    int64_t preTokens = 0;
    int64_t nxtTokens = 0;
    int64_t innerPrecise = 0;
    int64_t sparseMode = 0;
    int64_t pseType = 1;
    PseShapeType pseShapeType = PseShapeType::NONE;
    DropMaskShapeType dropMaskShapeType = DropMaskShapeType::NONE;
    PaddingMaskShapeType paddingMaskShapeType = PaddingMaskShapeType::NONE;
    AttenMaskShapeType attenMaskShapeType = AttenMaskShapeType::NONE;
    ge::DataType attenMaskDtype = ge::DataType::DT_UNDEFINED;
    PrefixShapeType prefixShapeType = PrefixShapeType::NONE;
    std::vector<int64_t> prefixTensorData = {};
    std::vector<int64_t> actualSeqQLenList = {};
    std::vector<int64_t> actualSeqKVLenList = {};
    std::vector<int64_t> qStartIdxTensorData = {};
    std::vector<int64_t> kvStartIdxTensorData = {};

    /* 生成参数 */
    int64_t n1 = 0;
    int64_t h1 = 0;
    int64_t h2 = 0;
    int64_t h1_rope = 0;
    int64_t h2_rope = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    std::string layout;
    int64_t productS1S2 = 0;
    std::vector<int64_t> actualSeqQLenTensorData = {};
    std::vector<int64_t> actualSeqKVLenTensorData = {};

    /* 输入输出 */
    Tensor query, key, value, dy, pse, dropMask, paddingMask, attenMask, prefix, softmaxMax, softmaxSum, softmaxRes,
        attenRes, dq, dk, dv, dPse, actualSeqQLen, actualSeqKvLen, qStartIdx, kvStartIdx, deqScaleQ, deqScaleK,
        deqScaleV, deqScaleDy, deqScaleO, queryRope, keyRope, dqRope, dkRope;

public:
    FaParam() = default;
    FaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
            LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
            int64_t pInnerPrecise, int64_t pSparseMode, PseShapeType pPseShapeType,
            DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
            AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType);
    /**
     * @param pActualSeqQLenList 传入实际 Seq 长度, 内部计算 T1 值
     * @param pActualSeqKvLenList 传入实际 Seq 长度, 内部计算 T2 值
     */
    FaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
            LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
            int64_t pInnerPrecise, int64_t pSparseMode, PseShapeType pPseShapeType,
            DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
            AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType,
            std::vector<int64_t> pPrefixTensorData, std::vector<int64_t> pActualSeqQLenList,
            std::vector<int64_t> pActualSeqKvLenList);

    /**
     * @param pActualSeqQLenList 传入实际 Seq 长度, 内部计算 T1 值
     * @param pActualSeqKvLenList 传入实际 Seq 长度, 内部计算 T2 值
     */
    FaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
            LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
            int64_t pInnerPrecise, int64_t pSparseMode, int64_t pPseType, PseShapeType pPseShapeType,
            DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
            AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType,
            std::vector<int64_t> pPrefixTensorData, std::vector<int64_t> pActualSeqQLenList,
            std::vector<int64_t> pActualSeqKvLenList, std::vector<int64_t> pQStartIdxTensorData,
            std::vector<int64_t> pKVStartIdxTensorData);
    virtual ~FaParam() = default;

    virtual bool Init();

    virtual bool IsUnPaddingAttention();

    template <class T> static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
    {
        if (hostData.empty()) {
            return true;
        }
        int64_t expMinSize = hostData.size() * sizeof(T);
        if (tensor.AllocDevData(0, expMinSize) == nullptr) {
            LOG_ERR("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
            return false;
        }
        return tensor.CopyHostToDevData(hostData);
    }
};

} // namespace ops::adv::tests::fa
