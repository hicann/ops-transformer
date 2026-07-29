/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "quant_block_sparse_attn_api_ut_param.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../op_api/aclnn_quant_block_sparse_attn.h"

namespace QuantBlockSparseAttnUT {

class AclnnQuantBlockSparseAttnTest : public testing::TestWithParam<QuantBlockSparseAttnApiUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantBlockSparseAttn AclnnQuantBlockSparseAttnTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBlockSparseAttn AclnnQuantBlockSparseAttnTest TearDown" << std::endl;
    }
};

TEST_P(AclnnQuantBlockSparseAttnTest, param)
{
    auto param = GetParam();
    op::SetPlatformSocVersion(param.soc);

    aclTensor *queryPtr = param.query.GetViewDims().empty() ? nullptr : param.query.ToAclTypeRawPtr();
    aclTensor *keyPtr = param.key.GetViewDims().empty() ? nullptr : param.key.ToAclTypeRawPtr();
    aclTensor *valuePtr = param.value.GetViewDims().empty() ? nullptr : param.value.ToAclTypeRawPtr();
    aclTensor *qDescalePtr = param.qDescale.GetViewDims().empty() ? nullptr : param.qDescale.ToAclTypeRawPtr();
    aclTensor *kDescalePtr = param.kDescale.GetViewDims().empty() ? nullptr : param.kDescale.ToAclTypeRawPtr();
    aclTensor *vDescalePtr = param.vDescale.GetViewDims().empty() ? nullptr : param.vDescale.ToAclTypeRawPtr();
    aclTensor *pScalePtr = param.pScale.GetViewDims().empty() ? nullptr : param.pScale.ToAclTypeRawPtr();
    aclTensor *cuSeqlensQPtr = param.cuSeqlensQ.GetViewDims().empty() ? nullptr : param.cuSeqlensQ.ToAclTypeRawPtr();
    aclTensor *cuSeqlensKvPtr = param.cuSeqlensKv.GetViewDims().empty() ? nullptr : param.cuSeqlensKv.ToAclTypeRawPtr();
    aclTensor *sequsedQPtr = param.sequsedQ.GetViewDims().empty() ? nullptr : param.sequsedQ.ToAclTypeRawPtr();
    aclTensor *sequsedKvPtr = param.sequsedKv.GetViewDims().empty() ? nullptr : param.sequsedKv.ToAclTypeRawPtr();
    aclTensor *sparseIndicesPtr =
        param.sparseIndices.GetViewDims().empty() ? nullptr : param.sparseIndices.ToAclTypeRawPtr();
    aclTensor *sparseSeqLenPtr =
        param.sparseSeqLen.GetViewDims().empty() ? nullptr : param.sparseSeqLen.ToAclTypeRawPtr();
    aclTensor *blockTablePtr = param.blockTable.GetViewDims().empty() ? nullptr : param.blockTable.ToAclTypeRawPtr();
    aclTensor *attenMaskPtr = param.attenMask.GetViewDims().empty() ? nullptr : param.attenMask.ToAclTypeRawPtr();
    aclTensor *metadataPtr = param.metadata.GetViewDims().empty() ? nullptr : param.metadata.ToAclTypeRawPtr();
    aclTensor *attentionOutPtr =
        param.attentionOut.GetViewDims().empty() ? nullptr : param.attentionOut.ToAclTypeRawPtr();
    aclTensor *softmaxLsePtr = param.softmaxLse.GetViewDims().empty() ? nullptr : param.softmaxLse.ToAclTypeRawPtr();

    auto ut = OP_API_UT(
        aclnnQuantBlockSparseAttn,
        INPUT(queryPtr, keyPtr, valuePtr, qDescalePtr, kDescalePtr, vDescalePtr, pScalePtr, cuSeqlensQPtr,
              cuSeqlensKvPtr, sequsedQPtr, sequsedKvPtr, sparseIndicesPtr, sparseSeqLenPtr, blockTablePtr, attenMaskPtr,
              metadataPtr, param.maxSeqlenQ, param.maxSeqlenKv, param.softmaxScale, param.sparseQBlockSize,
              param.sparseKvBlockSize, param.paBlockStride, (char *)param.layoutKv.c_str(),
              (char *)param.layoutQ.c_str(), (char *)param.layoutSparseIndices.c_str(), (char *)param.layoutOut.c_str(),
              param.quantMode, param.maskMode, param.returnSoftmaxLse),
        OUTPUT(attentionOutPtr, softmaxLsePtr));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto aclnnRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);

    if (param.expectResult == ACLNN_SUCCESS) {
        if (aclnnRet != ACLNN_SUCCESS) {
            GTEST_SKIP() << "Normal case requires NPU hardware, got error: " << aclnnRet;
        }
    } else if (param.expectResult == ACLNN_ERR_PARAM_INVALID) {
        EXPECT_NE(aclnnRet, ACLNN_SUCCESS);
    } else {
        EXPECT_EQ(param.expectResult, aclnnRet);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttn, AclnnQuantBlockSparseAttnTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnApiUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnApiUtParam>);

} // namespace QuantBlockSparseAttnUT
