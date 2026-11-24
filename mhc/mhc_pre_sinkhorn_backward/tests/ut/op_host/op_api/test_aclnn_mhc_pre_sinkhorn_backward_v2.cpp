/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/op_api/aclnn_mhc_pre_sinkhorn_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class MhcPreSinkhornBackwardV2OpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MhcPreSinkhornBackwardV2OpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPreSinkhornBackwardV2OpapiUt TearDown" << endl;
    }
};

// V2 基础: 无 premix (preMix=nullptr, gradPre=nullptr, gradPreMix=nullptr)
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_basic_4d_no_premix)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// V2 带 premix: preMix/gradPre 成对 + gradPreMix (全部传入)
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_basic_4d_with_premix)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto preMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, gradPre, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// V2 组合校验: preMix 在而 gradPre 缺省 → 161001 (PARAM_NULLPTR)
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_premix_only_no_gradpre)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto preMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, (aclTensor*)nullptr, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// V2 组合校验: gradPre 在而 preMix 缺省 → 161001 (PARAM_NULLPTR)
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_gradpre_only_no_premix)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, (aclTensor*)nullptr, gradPre, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// V2 联动校验: preMix 在但 gradPreMix 缺省 → 161001 (PARAM_NULLPTR)
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_premix_no_gradpremix)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto preMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    // gradPreMix 传 nullptr: preMix 在时应拒绝

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, gradPre, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// V2 联动校验: preMix 缺省但 gradPreMix 传入 → 无害(返回全零), 应 SUCCESS
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_no_premix_with_gradpremix)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// V2 3D TND 格式带 premix
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_basic_3d_tnd_with_premix)
{
    int64_t T = 256;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({T, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({T, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({T, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({T, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, T, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto preMix = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({T, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, gradPre, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// V2 空输入: B=0
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_empty_tensor_4d_B0)
{
    int64_t B = 0;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto preMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, gradPre, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

// V2 shape 不一致: preMix shape 与 x 不匹配 → 161002
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_premix_shape_mismatch)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    // preMix shape 故意与 x 的 [B,S,N] 不匹配
    auto preMix = TensorDesc({B, S + 2, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, gradPre, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// V2 空指针: x 为 nullptr → 161001
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_null_x)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradX = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, (aclTensor*)nullptr, phi, alpha, bias, hPre, hcBeforeNorm,
                              invRms, sumOut, normOut, (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-6, true),
                        OUTPUT(gradX, gradPhi, gradAlpha, gradBias, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// V2 空指针: gradX 输出为 nullptr → 161001
TEST_F(MhcPreSinkhornBackwardV2OpapiUt, aclnn_v2_null_output_gradx)
{
    int64_t B = 2;
    int64_t S = 128;
    int64_t N = 4;
    int64_t C = 256;
    int64_t skIterCount = 20;
    int64_t hcMix = N * N + 2 * N;

    auto gradHin = TensorDesc({B, S, C}, ACL_BF16, ACL_FORMAT_ND);
    auto gradHPost = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradHRes = TensorDesc({B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x = TensorDesc({B, S, N, C}, ACL_BF16, ACL_FORMAT_ND);
    auto phi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hcBeforeNorm = TensorDesc({B, S, hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invRms = TensorDesc({B, S, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({skIterCount * 2, B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto normOut = TensorDesc({skIterCount * 2, B, S, N, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto preMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPre = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gradPhi = TensorDesc({hcMix, N * C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradAlpha = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBias = TensorDesc({hcMix}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradPreMix = TensorDesc({B, S, N}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMhcPreSinkhornBackwardV2,
                        INPUT(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut,
                              normOut, preMix, gradPre, 1e-6, true),
                        OUTPUT((aclTensor*)nullptr, gradPhi, gradAlpha, gradBias, gradPreMix));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
