/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention_grad.h"
#include "../../../op_host/op_api/aclnn_sparse_flash_attention_grad_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor *tensor)
{
    Release(tensor);
}

using AclTensorPtr = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>;

AclTensorPtr MakeTensor(const vector<int64_t> &shape, aclDataType dtype)
{
    return AclTensorPtr(TensorDesc(shape, dtype, ACL_FORMAT_ND).ToAclTypeRawPtr(), DestroyAclTensor);
}
} // namespace

// =================== arch22 回归（无 sink，sinks/dSinks=nullptr）===================
class SparseFlashAttentionGradOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "SparseFlashAttentionGradOpapiUt SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SparseFlashAttentionGradOpapiUt TearDown" << endl;
    }
};

// BSND 正例：value 在，无 sink
TEST_F(SparseFlashAttentionGradOpapiUt, A1_bsnd_fp16_value_present)
{
    auto query = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto key = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto value = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({1, 64, 1, 4}, ACL_INT32);
    auto dOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto out = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto dKeyOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto dValueOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(),
        softmaxSum.get(), nullptr, nullptr, nullptr, nullptr, 0.0883883476, 64, layout, 3, INT64_MAX, INT64_MAX, false,
        dQueryOut.get(), dKeyOut.get(), dValueOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// BSND 正例：value optional null，无 sink
TEST_F(SparseFlashAttentionGradOpapiUt, A2_bsnd_fp16_value_optional_null)
{
    auto query = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto key = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({1, 64, 1, 4}, ACL_INT32);
    auto dOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto out = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto dKeyOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        query.get(), key.get(), nullptr, sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0883883476, 64, layout, 3, INT64_MAX, INT64_MAX, false, dQueryOut.get(),
        dKeyOut.get(), nullptr, nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// 负例：query null
TEST_F(SparseFlashAttentionGradOpapiUt, E1_null_query)
{
    auto key = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto value = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto sparseIndices = MakeTensor({1, 64, 1, 4}, ACL_INT32);
    auto dOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto out = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({1, 8, 64}, ACL_FLOAT);
    auto dQueryOut = MakeTensor({1, 64, 8, 128}, ACL_FLOAT16);
    auto dKeyOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);
    auto dValueOut = MakeTensor({1, 128, 1, 128}, ACL_FLOAT16);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradGetWorkspaceSize(
        nullptr, key.get(), value.get(), sparseIndices.get(), dOut.get(), out.get(), softmaxMax.get(), softmaxSum.get(),
        nullptr, nullptr, nullptr, nullptr, 0.0883883476, 64, layout, 3, INT64_MAX, INT64_MAX, false, dQueryOut.get(),
        dKeyOut.get(), dValueOut.get(), nullptr, nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// =================== arch35（ASCEND950）sink 功能：TND+rope 已验证 shape ====================
// shape 来源：probe/golden 实测可跑通（T1=1, S2=2048, K=2048, D=512, Dr=64, N2=1）
class SparseFlashAttentionGradArch35SinkUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "SparseFlashAttentionGradArch35SinkUt SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SparseFlashAttentionGradArch35SinkUt TearDown" << endl;
    }
};

namespace {
struct TndRopeTensors {
    AclTensorPtr query, key, value, sparseIndices, dOut, out, softmaxMax, softmaxSum;
    AclTensorPtr actSeqQ, actSeqKv, qRope, kRope;
    AclTensorPtr dQ, dK, dV, dQRope, dKRope;
};
TndRopeTensors MakeTndRopeTensors(int64_t n1)
{
    const int64_t t1 = 1, d = 512, n2 = 1, s2 = 2048, dr = 64;
    // 聚合初始化（成员 move 构造，避免 unique_ptr 默认构造/拷贝）
    return TndRopeTensors{
        MakeTensor({t1, n1, d}, ACL_BF16),   // query
        MakeTensor({s2, n2, d}, ACL_BF16),   // key
        MakeTensor({s2, n2, d}, ACL_BF16),   // value
        MakeTensor({t1, n2, s2}, ACL_INT32), // sparseIndices (K=s2=2048)
        MakeTensor({t1, n1, d}, ACL_BF16),   // dOut
        MakeTensor({t1, n1, d}, ACL_BF16),   // out
        MakeTensor({n2, t1, n1}, ACL_FLOAT), // softmaxMax
        MakeTensor({n2, t1, n1}, ACL_FLOAT), // softmaxSum
        MakeTensor({t1}, ACL_INT32),         // actSeqQ (actSeqQLen=[1])
        MakeTensor({t1}, ACL_INT32),         // actSeqKv (shape[1]，data 不影响 tiling)
        MakeTensor({t1, n1, dr}, ACL_BF16),  // qRope
        MakeTensor({s2, n2, dr}, ACL_BF16),  // kRope
        MakeTensor({t1, n1, d}, ACL_BF16),   // dQ
        MakeTensor({s2, n2, d}, ACL_BF16),   // dK
        MakeTensor({s2, n2, d}, ACL_BF16),   // dV
        MakeTensor({t1, n1, dr}, ACL_BF16),  // dQRope
        MakeTensor({s2, n2, dr}, ACL_BF16)   // dKRope
    };
}
} // namespace

// S1：sinks+dSinks 都在 → IS_SINKS=true，tiling 成功
TEST_F(SparseFlashAttentionGradArch35SinkUt, S1_tnd_rope_with_sinks)
{
    constexpr int64_t n1 = 16;
    auto t = MakeTndRopeTensors(n1);
    auto sinks = MakeTensor({n1}, ACL_FLOAT);
    auto dSinks = MakeTensor({n1}, ACL_FLOAT);
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
        t.query.get(), t.key.get(), t.value.get(), t.sparseIndices.get(), t.dOut.get(), t.out.get(), t.softmaxMax.get(),
        t.softmaxSum.get(), sinks.get(), t.actSeqQ.get(), t.actSeqKv.get(), t.qRope.get(), t.kRope.get(), 0.0441941738,
        1, layout, 0, INT64_MAX, INT64_MAX, false, t.dQ.get(), t.dK.get(), t.dV.get(), t.dQRope.get(), t.dKRope.get(),
        dSinks.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// S2：sinks=nullptr → IS_SINKS=false 回归，tiling 成功
TEST_F(SparseFlashAttentionGradArch35SinkUt, S2_tnd_rope_no_sinks)
{
    constexpr int64_t n1 = 16;
    auto t = MakeTndRopeTensors(n1);
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
        t.query.get(), t.key.get(), t.value.get(), t.sparseIndices.get(), t.dOut.get(), t.out.get(), t.softmaxMax.get(),
        t.softmaxSum.get(), nullptr, t.actSeqQ.get(), t.actSeqKv.get(), t.qRope.get(), t.kRope.get(), 0.0441941738, 1,
        layout, 0, INT64_MAX, INT64_MAX, false, t.dQ.get(), t.dK.get(), t.dV.get(), t.dQRope.get(), t.dKRope.get(),
        nullptr, &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// S3：sinks 在 + deterministic=true → deter+sink 路径，tiling 成功
TEST_F(SparseFlashAttentionGradArch35SinkUt, S3_tnd_rope_with_sinks_deter)
{
    constexpr int64_t n1 = 16;
    auto t = MakeTndRopeTensors(n1);
    auto sinks = MakeTensor({n1}, ACL_FLOAT);
    auto dSinks = MakeTensor({n1}, ACL_FLOAT);
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
        t.query.get(), t.key.get(), t.value.get(), t.sparseIndices.get(), t.dOut.get(), t.out.get(), t.softmaxMax.get(),
        t.softmaxSum.get(), sinks.get(), t.actSeqQ.get(), t.actSeqKv.get(), t.qRope.get(), t.kRope.get(), 0.0441941738,
        1, layout, 0, INT64_MAX, INT64_MAX, true, t.dQ.get(), t.dK.get(), t.dV.get(), t.dQRope.get(), t.dKRope.get(),
        dSinks.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// S4：N1=24（halfG=12 非 32B 对齐）+ sinks → 覆盖非对齐 tilingKey，成功
TEST_F(SparseFlashAttentionGradArch35SinkUt, S4_tnd_rope_n1_24_non_aligned)
{
    constexpr int64_t n1 = 24;
    auto t = MakeTndRopeTensors(n1);
    auto sinks = MakeTensor({n1}, ACL_FLOAT);
    auto dSinks = MakeTensor({n1}, ACL_FLOAT);
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
        t.query.get(), t.key.get(), t.value.get(), t.sparseIndices.get(), t.dOut.get(), t.out.get(), t.softmaxMax.get(),
        t.softmaxSum.get(), sinks.get(), t.actSeqQ.get(), t.actSeqKv.get(), t.qRope.get(), t.kRope.get(), 0.0441941738,
        1, layout, 0, INT64_MAX, INT64_MAX, false, t.dQ.get(), t.dK.get(), t.dV.get(), t.dQRope.get(), t.dKRope.get(),
        dSinks.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// E2：sinks 在但 dSinks=nullptr → sink 输出缺失，tiling 应失败（host 校验）
TEST_F(SparseFlashAttentionGradArch35SinkUt, E2_sinks_present_but_dsinks_null)
{
    constexpr int64_t n1 = 16;
    auto t = MakeTndRopeTensors(n1);
    auto sinks = MakeTensor({n1}, ACL_FLOAT);
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclRet = aclnnSparseFlashAttentionGradV2GetWorkspaceSize(
        t.query.get(), t.key.get(), t.value.get(), t.sparseIndices.get(), t.dOut.get(), t.out.get(), t.softmaxMax.get(),
        t.softmaxSum.get(), sinks.get(), t.actSeqQ.get(), t.actSeqKv.get(), t.qRope.get(), t.kRope.get(), 0.0441941738,
        1, layout, 0, INT64_MAX, INT64_MAX, false, t.dQ.get(), t.dK.get(), t.dV.get(), t.dQRope.get(), t.dKRope.get(),
        nullptr, &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}
