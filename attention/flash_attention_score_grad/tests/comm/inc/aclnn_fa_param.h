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
 * \file aclnn_fa_param.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad Aclnn 参数信息.
 */

#pragma once

#include "fa_case.h"
#include "tests/utils/aclnn_tensor.h"

namespace ops::adv::tests::fa {

class AclnnFaParam : public ops::adv::tests::fa::FaParam {
public:
    using AclnnTensor = ops::adv::tests::utils::AclnnTensor;

public:
    /* 输入输出 */
    AclnnTensor aclnnQuery, aclnnKey, aclnnValue, aclnnDy, aclnnPse, aclnnDropMask, aclnnPaddingMask, aclnnAttenMask,
        aclnnPrefix, aclnnSoftmaxMax, aclnnSoftmaxSum, aclnnSoftmaxRes, aclnnAttenRes, aclnnDq, aclnnDk, aclnnDv,
        aclnnDpse, aclnnActualSeqQLen, aclnnActualSeqKvLen;
    AclnnTensor aclnnQueryRope, aclnnKeyRope;
    AclnnTensor aclnnDqRope, aclnnDkRope;
    aclIntArray *aclnnPrefixIntAry = nullptr;
    aclIntArray *aclnnActualSeqQLenIntAry = nullptr;
    aclIntArray *aclnnActualSeqKvLenIntAry = nullptr;
    aclIntArray *qStartIdxOptionalIntAry = nullptr;
    aclIntArray *kvStartIdxOptionalIntAry = nullptr;

public:
    AclnnFaParam() = default;
    AclnnFaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype,
                 PrefixShapeType pPrefixShapeType);
    AclnnFaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType,
                 std::vector<int64_t> pPrefixTensorData, std::vector<int64_t> pActualSeqQLenList,
                 std::vector<int64_t> pActualSeqKvLenList);
    AclnnFaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, int64_t pPseType, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype,
                 PrefixShapeType pPrefixShapeType);
    AclnnFaParam(int64_t pB, int64_t pN2, int64_t pG, int64_t pS1, int64_t pS2, int64_t pD, ge::DataType pDtype,
                 LayoutType pLayoutType, float pScale, float pKeepProb, int64_t pPreTokens, int64_t pNxtTokens,
                 int64_t pInnerPrecise, int64_t pSparseMode, int64_t pPseType, PseShapeType pPseShapeType,
                 DropMaskShapeType pDropMaskShapeType, PaddingMaskShapeType pPaddingMaskShapeType,
                 AttenMaskShapeType pAttenMaskShapeType, ge::DataType pAttenMaskDtype, PrefixShapeType pPrefixShapeType,
                 std::vector<int64_t> pPrefixTensorData, std::vector<int64_t> pActualSeqQLenList,
                 std::vector<int64_t> pActualSeqKvLenList);
    ~AclnnFaParam();

    bool Init() override;

    bool IsUnPaddingAttention() override;
};

} // namespace ops::adv::tests::fa
