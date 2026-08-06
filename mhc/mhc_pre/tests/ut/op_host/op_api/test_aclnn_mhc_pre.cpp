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
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../../../../op_host/op_api/aclnn_mhc_pre.h"
#include "../../../../op_host/op_api/aclnn_mhc_pre_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class MhcPreOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        cout << "MhcPreOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "MhcPreOpapiUt TearDown" << endl;
    }
};

namespace {
constexpr int64_t B = 2;
constexpr int64_t S = 4;
constexpr int64_t N = 4;
constexpr int64_t D = 128;
constexpr int64_t PHI_ROWS = N * N + 2 * N;
constexpr int64_t ND = N * D;
constexpr double NORM_EPS = 1e-6;
constexpr double HC_EPS = 1e-6;
constexpr int64_t MHC_PRE_DEFAULT = 0;
constexpr int64_t MHC_PRE_USE_HF32 = 1;
constexpr int64_t MHC_PRE_INVALID_MODE = 2;

using TensorDescPtr = std::unique_ptr<TensorDesc>;

TensorDescPtr MakeTensor(const std::vector<int64_t> &shape, aclDataType dtype, double low, double high)
{
    TensorDescPtr tensor(new TensorDesc(shape, dtype, ACL_FORMAT_ND));
    tensor->ValueRange(low, high);
    return tensor;
}

struct MhcPreTensors {
    TensorDescPtr x;
    TensorDescPtr phi;
    TensorDescPtr alpha;
    TensorDescPtr bias;
    TensorDescPtr gamma;
    TensorDescPtr hIn;
    TensorDescPtr hPost;
    TensorDescPtr hRes;
    TensorDescPtr invRms;
    TensorDescPtr hMix;
    TensorDescPtr hPre;
};

MhcPreTensors MakeValidTensors()
{
    MhcPreTensors tensors;
    tensors.x = MakeTensor({B, S, N, D}, ACL_FLOAT16, -1, 1);
    tensors.phi = MakeTensor({PHI_ROWS, ND}, ACL_FLOAT, -1, 1);
    tensors.alpha = MakeTensor({3}, ACL_FLOAT, 0.5, 1.5);
    tensors.bias = MakeTensor({PHI_ROWS}, ACL_FLOAT, -1, 1);
    tensors.gamma = MakeTensor({N, D}, ACL_FLOAT, -1, 1);
    tensors.hIn = MakeTensor({B, S, D}, ACL_FLOAT16, 0, 0);
    tensors.hPost = MakeTensor({B, S, N}, ACL_FLOAT, 0, 0);
    tensors.hRes = MakeTensor({B, S, N, N}, ACL_FLOAT, 0, 0);
    tensors.invRms = MakeTensor({B, S}, ACL_FLOAT, 0, 0);
    tensors.hMix = MakeTensor({B, S, PHI_ROWS}, ACL_FLOAT, 0, 0);
    tensors.hPre = MakeTensor({B, S, N}, ACL_FLOAT, 0, 0);
    return tensors;
}

using CaseMutator = std::function<void(MhcPreTensors &)>;

struct InvalidCase {
    std::string name;
    std::string detail;
    CaseMutator mutate;
};

aclnnStatus RunMhcPre(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias, TensorDesc &gamma,
                      TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes, TensorDesc &invRms, TensorDesc &hMix,
                      TensorDesc &hPre)
{
    auto ut = OP_API_UT(aclnnMhcPre, INPUT(x, phi, alpha, bias, gamma, NORM_EPS, HC_EPS),
                        OUTPUT(hIn, hPost, hRes, invRms, hMix, hPre));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}


aclnnStatus RunMhcPreV2(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias, TensorDesc &gamma,
                        TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes, TensorDesc &invRms, TensorDesc &hMix,
                        TensorDesc &hPre, int64_t implMode)
{
    auto ut = OP_API_UT(aclnnMhcPreV2, INPUT(x, phi, alpha, bias, gamma, NORM_EPS, HC_EPS, implMode),
                        OUTPUT(hIn, hPost, hRes, invRms, hMix, hPre));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}
aclnnStatus RunMhcPreNoOptional(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias, TensorDesc &gamma,
                                TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes)
{
    auto ut = OP_API_UT(aclnnMhcPre, INPUT(x, phi, alpha, bias, gamma, NORM_EPS, HC_EPS),
                        OUTPUT(hIn, hPost, hRes, nullptr, nullptr, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}

enum class RequiredTensor { X, Phi, Alpha, Bias, HIn, HPost, HRes };

enum class RequiredPointer { None, WorkspaceSize, Executor };

aclnnStatus RunMhcPreMissingRequired(MhcPreTensors &tensors, RequiredTensor missing)
{
    auto x = tensors.x->ToAclType();
    auto phi = tensors.phi->ToAclType();
    auto alpha = tensors.alpha->ToAclType();
    auto bias = tensors.bias->ToAclType();
    auto gamma = tensors.gamma->ToAclType();
    auto hIn = tensors.hIn->ToAclType();
    auto hPost = tensors.hPost->ToAclType();
    auto hRes = tensors.hRes->ToAclType();
    auto getRequired = [missing](RequiredTensor current, auto *tensor) {
        return missing == current ? nullptr : tensor;
    };
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return aclnnMhcPreGetWorkspaceSize(
        getRequired(RequiredTensor::X, x.get()), getRequired(RequiredTensor::Phi, phi.get()),
        getRequired(RequiredTensor::Alpha, alpha.get()), getRequired(RequiredTensor::Bias, bias.get()), gamma.get(),
        NORM_EPS, HC_EPS, getRequired(RequiredTensor::HIn, hIn.get()),
        getRequired(RequiredTensor::HPost, hPost.get()), getRequired(RequiredTensor::HRes, hRes.get()), nullptr,
        nullptr, nullptr, &workspaceSize, &executor);
}

aclnnStatus RunMhcPreMissingRequiredPointer(MhcPreTensors &tensors, RequiredPointer missing)
{
    auto x = tensors.x->ToAclType();
    auto phi = tensors.phi->ToAclType();
    auto alpha = tensors.alpha->ToAclType();
    auto bias = tensors.bias->ToAclType();
    auto gamma = tensors.gamma->ToAclType();
    auto hIn = tensors.hIn->ToAclType();
    auto hPost = tensors.hPost->ToAclType();
    auto hRes = tensors.hRes->ToAclType();
    auto invRms = tensors.invRms->ToAclType();
    auto hMix = tensors.hMix->ToAclType();
    auto hPre = tensors.hPre->ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto workspaceSizePtr = missing == RequiredPointer::WorkspaceSize ? nullptr : &workspaceSize;
    auto executorPtr = missing == RequiredPointer::Executor ? nullptr : &executor;
    return aclnnMhcPreGetWorkspaceSize(x.get(), phi.get(), alpha.get(), bias.get(), gamma.get(), NORM_EPS, HC_EPS,
                                       hIn.get(), hPost.get(), hRes.get(), invRms.get(), hMix.get(), hPre.get(),
                                       workspaceSizePtr, executorPtr);
}

aclnnStatus RunMhcPrePartialOptional(TensorDesc &x, TensorDesc &phi, TensorDesc &alpha, TensorDesc &bias,
                                     TensorDesc &gamma, TensorDesc &hIn, TensorDesc &hPost, TensorDesc &hRes,
                                     TensorDesc &invRms)
{
    auto xAcl = x.ToAclType();
    auto phiAcl = phi.ToAclType();
    auto alphaAcl = alpha.ToAclType();
    auto biasAcl = bias.ToAclType();
    auto gammaAcl = gamma.ToAclType();
    auto hInAcl = hIn.ToAclType();
    auto hPostAcl = hPost.ToAclType();
    auto hResAcl = hRes.ToAclType();
    auto invRmsAcl = invRms.ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    return aclnnMhcPreGetWorkspaceSize(xAcl.get(), phiAcl.get(), alphaAcl.get(), biasAcl.get(), gammaAcl.get(),
                                       NORM_EPS, HC_EPS, hInAcl.get(), hPostAcl.get(), hResAcl.get(), invRmsAcl.get(),
                                       nullptr, nullptr, &workspaceSize, &executor);
}
} // namespace

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_valid_4d_fp16_with_optional_outputs)
{
    auto tensors = MakeValidTensors();

    EXPECT_NE(RunMhcPre(*tensors.x, *tensors.phi, *tensors.alpha, *tensors.bias, *tensors.gamma, *tensors.hIn,
                        *tensors.hPost, *tensors.hRes, *tensors.invRms, *tensors.hMix, *tensors.hPre),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_v2_valid_impl_mode_cases)
{
    for (const auto implMode : {MHC_PRE_DEFAULT, MHC_PRE_USE_HF32}) {
        auto tensors = MakeValidTensors();
        EXPECT_NE(RunMhcPreV2(*tensors.x, *tensors.phi, *tensors.alpha, *tensors.bias, *tensors.gamma, *tensors.hIn,
                              *tensors.hPost, *tensors.hRes, *tensors.invRms, *tensors.hMix, *tensors.hPre, implMode),
                  ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_v2_invalid_impl_mode)
{
    auto tensors = MakeValidTensors();
    EXPECT_EQ(RunMhcPreV2(*tensors.x, *tensors.phi, *tensors.alpha, *tensors.bias, *tensors.gamma, *tensors.hIn,
                          *tensors.hPost, *tensors.hRes, *tensors.invRms, *tensors.hMix, *tensors.hPre,
                          MHC_PRE_INVALID_MODE),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_missing_required_tensor_cases)
{
    const std::vector<std::pair<const char *, RequiredTensor>> cases = {
        {"missing_x", RequiredTensor::X},           {"missing_phi", RequiredTensor::Phi},
        {"missing_alpha", RequiredTensor::Alpha},   {"missing_bias", RequiredTensor::Bias},
        {"missing_hin", RequiredTensor::HIn},       {"missing_hpost", RequiredTensor::HPost},
        {"missing_hres", RequiredTensor::HRes},
    };
    for (const auto &[name, missing] : cases) {
        auto tensors = MakeValidTensors();
        const auto actual = RunMhcPreMissingRequired(tensors, missing);
        const auto result = actual == ACLNN_ERR_PARAM_NULLPTR ? "PASS" : "FAIL";
        cout << "[" << result << "] case=" << name << ", expected=" << ACLNN_ERR_PARAM_NULLPTR
             << ", actual=" << actual << endl;
        EXPECT_EQ(actual, ACLNN_ERR_PARAM_NULLPTR) << name;
    }
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_missing_required_pointer_cases)
{
    auto validTensors = MakeValidTensors();
    ASSERT_EQ(RunMhcPreMissingRequiredPointer(validTensors, RequiredPointer::None), ACLNN_SUCCESS);

    const std::vector<std::pair<const char *, RequiredPointer>> cases = {
        {"missing_workspace_size", RequiredPointer::WorkspaceSize},
        {"missing_executor", RequiredPointer::Executor},
    };
    for (const auto &[name, missing] : cases) {
        auto tensors = MakeValidTensors();
        const auto actual = RunMhcPreMissingRequiredPointer(tensors, missing);
        const auto result = actual == ACLNN_ERR_PARAM_NULLPTR ? "PASS" : "FAIL";
        cout << "[" << result << "] case=" << name << ", expected=" << ACLNN_ERR_PARAM_NULLPTR
             << ", actual=" << actual << endl;
        EXPECT_EQ(actual, ACLNN_ERR_PARAM_NULLPTR) << name;
    }
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_dtype_shape_cases)
{
    std::vector<InvalidCase> cases = {
        {"invalid_x_dtype_float32", "x dtype is float32", [](MhcPreTensors &t) {
             t.x = MakeTensor({B, S, N, D}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_x_shape_rank2", "x shape rank is 2D, expected 3D or 4D", [](MhcPreTensors &t) {
             t.x = MakeTensor({B, D}, ACL_FLOAT16, -1, 1);
         }},
        {"invalid_x_shape_value_n3", "x shape value has n=N-1", [](MhcPreTensors &t) {
             t.x = MakeTensor({B, S, N - 1, D}, ACL_FLOAT16, -1, 1);
         }},
        {"invalid_phi_dtype_float16", "phi dtype is float16", [](MhcPreTensors &t) {
             t.phi = MakeTensor({PHI_ROWS, ND}, ACL_FLOAT16, -1, 1);
         }},
        {"invalid_phi_shape_rank1", "phi shape rank is 1D, expected 2D", [](MhcPreTensors &t) {
             t.phi = MakeTensor({ND}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_phi_shape_value_first_dim", "phi first dim is PHI_ROWS-1", [](MhcPreTensors &t) {
             t.phi = MakeTensor({PHI_ROWS - 1, ND}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_alpha_dtype_float16", "alpha dtype is float16", [](MhcPreTensors &t) {
             t.alpha = MakeTensor({3}, ACL_FLOAT16, 0.5, 1.5);
         }},
        {"invalid_alpha_shape_rank0", "alpha shape rank is 0D, expected 1D", [](MhcPreTensors &t) {
             t.alpha = MakeTensor({}, ACL_FLOAT, 0.5, 1.5);
         }},
        {"invalid_alpha_shape_rank2", "alpha shape rank is 2D, expected 1D", [](MhcPreTensors &t) {
             t.alpha = MakeTensor({1, 3}, ACL_FLOAT, 0.5, 1.5);
         }},
        {"invalid_alpha_shape_value_len4", "alpha shape value is [4], expected [3]", [](MhcPreTensors &t) {
             t.alpha = MakeTensor({4}, ACL_FLOAT, 0.5, 1.5);
         }},
        {"invalid_bias_dtype_float16", "bias dtype is float16", [](MhcPreTensors &t) {
             t.bias = MakeTensor({PHI_ROWS}, ACL_FLOAT16, -1, 1);
         }},
        {"invalid_bias_shape_rank2", "bias shape rank is 2D, expected 1D", [](MhcPreTensors &t) {
             t.bias = MakeTensor({1, PHI_ROWS}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_bias_shape_value_len23", "bias shape value is PHI_ROWS-1", [](MhcPreTensors &t) {
             t.bias = MakeTensor({PHI_ROWS - 1}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_gamma_dtype_float16", "gamma dtype is float16", [](MhcPreTensors &t) {
             t.gamma = MakeTensor({N, D}, ACL_FLOAT16, -1, 1);
         }},
        {"invalid_gamma_shape_rank1", "gamma shape rank is 1D, expected 2D", [](MhcPreTensors &t) {
             t.gamma = MakeTensor({D}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_gamma_shape_value_last_dim", "gamma last dim is D+1", [](MhcPreTensors &t) {
             t.gamma = MakeTensor({N, D + 1}, ACL_FLOAT, -1, 1);
         }},
        {"invalid_hin_dtype_float32", "hIn dtype is float32", [](MhcPreTensors &t) {
             t.hIn = MakeTensor({B, S, D}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hin_shape_rank1", "hIn shape rank is 1D, expected 3D", [](MhcPreTensors &t) {
             t.hIn = MakeTensor({D}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_hin_shape_value_last_dim", "hIn last dim is D+1", [](MhcPreTensors &t) {
             t.hIn = MakeTensor({B, S, D + 1}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_hpost_dtype_float16", "hPost dtype is float16", [](MhcPreTensors &t) {
             t.hPost = MakeTensor({B, S, N}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_hpost_shape_rank1", "hPost shape rank is 1D, expected 3D", [](MhcPreTensors &t) {
             t.hPost = MakeTensor({N}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hpost_shape_value_last_dim", "hPost last dim is N+1", [](MhcPreTensors &t) {
             t.hPost = MakeTensor({B, S, N + 1}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hres_dtype_float16", "hRes dtype is float16", [](MhcPreTensors &t) {
             t.hRes = MakeTensor({B, S, N, N}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_hres_shape_rank2", "hRes shape rank is 2D, expected 4D", [](MhcPreTensors &t) {
             t.hRes = MakeTensor({N, N}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hres_shape_value_last_dim", "hRes last dim is N+1", [](MhcPreTensors &t) {
             t.hRes = MakeTensor({B, S, N, N + 1}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_invrms_dtype_float16", "invRms dtype is float16", [](MhcPreTensors &t) {
             t.invRms = MakeTensor({B, S}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_invrms_shape_rank3", "invRms shape rank is 3D, expected 2D", [](MhcPreTensors &t) {
             t.invRms = MakeTensor({B, S, 1}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_invrms_shape_value_len", "invRms second dim is S+1", [](MhcPreTensors &t) {
             t.invRms = MakeTensor({B, S + 1}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hmix_dtype_float16", "hMix dtype is float16", [](MhcPreTensors &t) {
             t.hMix = MakeTensor({B, S, PHI_ROWS}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_hmix_shape_rank1", "hMix shape rank is 1D, expected 3D", [](MhcPreTensors &t) {
             t.hMix = MakeTensor({PHI_ROWS}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hmix_shape_value_last_dim", "hMix last dim is PHI_ROWS+1", [](MhcPreTensors &t) {
             t.hMix = MakeTensor({B, S, PHI_ROWS + 1}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hpre_dtype_float16", "hPre dtype is float16", [](MhcPreTensors &t) {
             t.hPre = MakeTensor({B, S, N}, ACL_FLOAT16, 0, 0);
         }},
        {"invalid_hpre_shape_rank1", "hPre shape rank is 1D, expected 3D", [](MhcPreTensors &t) {
             t.hPre = MakeTensor({N}, ACL_FLOAT, 0, 0);
         }},
        {"invalid_hpre_shape_value_last_dim", "hPre last dim is N+1", [](MhcPreTensors &t) {
             t.hPre = MakeTensor({B, S, N + 1}, ACL_FLOAT, 0, 0);
         }},
    };

    for (const auto &invalidCase : cases) {
        auto tensors = MakeValidTensors();
        invalidCase.mutate(tensors);
        auto actual = RunMhcPre(*tensors.x, *tensors.phi, *tensors.alpha, *tensors.bias, *tensors.gamma, *tensors.hIn,
                                *tensors.hPost, *tensors.hRes, *tensors.invRms, *tensors.hMix, *tensors.hPre);
        auto result = actual == ACLNN_ERR_PARAM_INVALID ? "PASS" : "FAIL";
        cout << "[" << result << "] case=" << invalidCase.name << ", detail=" << invalidCase.detail
             << ", expected=" << ACLNN_ERR_PARAM_INVALID << ", actual=" << actual << endl;
        EXPECT_EQ(actual, ACLNN_ERR_PARAM_INVALID) << invalidCase.name;
    }
}

TEST_F(MhcPreOpapiUt, aclnn_mhc_pre_invalid_partial_optional_outputs)
{
    auto tensors = MakeValidTensors();

    EXPECT_EQ(RunMhcPrePartialOptional(*tensors.x, *tensors.phi, *tensors.alpha, *tensors.bias, *tensors.gamma,
                                       *tensors.hIn, *tensors.hPost, *tensors.hRes, *tensors.invRms),
              ACLNN_ERR_PARAM_INVALID);
}
