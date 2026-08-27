/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mhc_pre.h"
#include "aclnn_mhc_pre_v2.h"
#include <dlfcn.h>
#include <new>
#include <memory>
#include <string>
#include <unordered_map>
#include <initializer_list>
#include "securec.h"
#include "log/log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mhc_pre.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr const char *ACLNN_OP_NAME = "aclnnMhcPreGetWorkspaceSize";
constexpr size_t DIM_NUM_1 = 1UL;
constexpr size_t DIM_NUM_2 = 2UL;
constexpr size_t DIM_NUM_3 = 3UL;
constexpr size_t DIM_NUM_4 = 4UL;

constexpr int64_t DIM_IDX_0 = 0;
constexpr int64_t DIM_IDX_1 = 1;
constexpr int64_t DIM_IDX_2 = 2;
constexpr int64_t DIM_IDX_3 = 3;

constexpr int64_t N_VALID_VALUES[] = {4, 6, 8};
constexpr int64_t N_VALID_VALUE_A2 = 4;
constexpr int64_t D_ALIGNMENT = 16;
constexpr int64_t D_ALIGNMENT_A2 = 128;
constexpr int64_t ALPHA_DIM_SIZE_3 = 3;
constexpr int64_t ALPHA_DIM_SIZE_2 = 2;
constexpr int64_t PHI_DIM_OFFSET = 2;
constexpr int64_t MHC_PRE_USE_FP32 = 0;
constexpr int64_t MHC_PRE_USE_HF32 = 1;

static int64_t Factorial(int64_t n)
{
    int64_t r = 1;
    for (int64_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

static bool CheckAlphaShape(const aclTensor *alphaTensor);
static bool ValidateNDParams(int64_t n, int64_t d);
static bool ValidateNDParamsA2(int64_t n, int64_t d);
static bool CheckTensorShape(const aclTensor *tensor, std::initializer_list<int64_t> expectedShape, const char *name);

static std::string TensorShapeToString(const aclTensor *tensor)
{
    const auto shape = tensor->GetViewShape();
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        result += (i == 0 ? "" : ", ") + std::to_string(shape.GetDim(i));
    }
    return result + "]";
}

static std::string ExpectedShapeToString(std::initializer_list<int64_t> shape)
{
    std::string result = "[";
    size_t index = 0;
    for (const auto dim : shape) {
        result += (index++ == 0 ? "" : ", ") + std::to_string(dim);
    }
    return result + "]";
}

static bool CheckRequiredTensor(const aclTensor *tensor, const char *name, const char *reason)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, name, "nullptr", reason);
        return false;
    }
    return true;
}

static bool CheckNonEmptyTensor(const aclTensor *tensor, const char *name)
{
    if (tensor->IsEmpty()) {
        const std::string actualShape = TensorShapeToString(tensor);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_OP_NAME, name, actualShape.c_str(), "tensor must not be empty");
        return false;
    }
    return true;
}

static bool CheckTensorRank(const aclTensor *tensor, const char *name, size_t expectedRank)
{
    const size_t actualRank = tensor->GetViewShape().GetDimNum();
    if (actualRank != expectedRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(ACLNN_OP_NAME, name, std::to_string(actualRank).c_str(),
                                     std::to_string(expectedRank).c_str());
        return false;
    }
    return true;
}

static bool CheckTensorDtype(const aclTensor *tensor, const char *name, DataType expectedDtype)
{
    const auto actualDtype = tensor->GetDataType();
    if (actualDtype != expectedDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(ACLNN_OP_NAME, name, op::ToString(actualDtype).GetString(),
                                  op::ToString(expectedDtype).GetString());
        return false;
    }
    return true;
}

struct MhcParamsBase {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *gammaOptional = nullptr;
    double normEps;
    double hcEps;
    int64_t opImplMode = MHC_PRE_USE_FP32;
    aclTensor *hIn = nullptr;
    aclTensor *hPost = nullptr;
    aclTensor *hRes = nullptr;
    aclTensor *invRmsOptional = nullptr;
    aclTensor *hMixOptional = nullptr;
    aclTensor *hPreOptional = nullptr;
    bool hasResi = true;

    // 用于存储转换后的连续tensor（在ConvertDataContiguous中使用）
    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *biasContiguous = nullptr;
    const aclTensor *gammaOptionalContiguous = nullptr;
};

class MhcBuilder {
public:
    static MhcBuilder Create()
    {
        MhcBuilder obj;

        return obj;
    }

    MhcBuilder &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                         const aclTensor *gammaOptional)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        obj_.gammaOptional = gammaOptional;
        return *this;
    }

    MhcBuilder &SetAttr(double normEps, double hcEps, int64_t opImplMode)
    {
        obj_.normEps = normEps;
        obj_.hcEps = hcEps;
        obj_.opImplMode = opImplMode;
        return *this;
    }

    MhcBuilder &SetOutput(aclTensor *hIn, aclTensor *hPost, aclTensor *hRes)
    {
        obj_.hIn = hIn;
        obj_.hPost = hPost;
        obj_.hRes = hRes;
        return *this;
    }

    MhcBuilder &SetOptionalOutput(aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional)
    {
        obj_.invRmsOptional = invRmsOptional;
        obj_.hMixOptional = hMixOptional;
        obj_.hPreOptional = hPreOptional;

        return *this;
    }

    MhcParamsBase Build() const
    {
        return obj_;
    }

private:
    MhcParamsBase obj_;
};

static bool CheckNotNull(const MhcParamsBase &params)
{
    return CheckRequiredTensor(params.x, "x", "required input tensor must not be nullptr") &&
           CheckRequiredTensor(params.phi, "phi", "required input tensor must not be nullptr") &&
           CheckRequiredTensor(params.alpha, "alpha", "required input tensor must not be nullptr") &&
           CheckRequiredTensor(params.bias, "bias", "required input tensor must not be nullptr") &&
           CheckRequiredTensor(params.hIn, "hIn", "required output tensor must not be nullptr") &&
           CheckRequiredTensor(params.hPost, "hPost", "required output tensor must not be nullptr");
}

static bool CheckRequiredHRes(const MhcParamsBase &params)
{
    if (params.alpha->GetViewShape().GetDim(DIM_IDX_0) == ALPHA_DIM_SIZE_3 && params.hRes == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "hRes", "nullptr",
                                              "output tensor is required when alpha shape is [3]");
        return false;
    }
    return true;
}

static bool CheckEmptyTensor(const MhcParamsBase &params)
{
    return CheckNonEmptyTensor(params.x, "x") && CheckNonEmptyTensor(params.phi, "phi") &&
           CheckNonEmptyTensor(params.alpha, "alpha") && CheckNonEmptyTensor(params.bias, "bias");
}

static bool CheckInputDims(const MhcParamsBase &params)
{
    // x dims must be 3 or 4
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum != DIM_NUM_3 && xDimNum != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(ACLNN_OP_NAME, "x", std::to_string(xDimNum).c_str(), "3 or 4");
        return false;
    }
    // phi dims must be 2,  alpha dims must be 1, bias dims must be 1,
    if (!CheckTensorRank(params.phi, "phi", DIM_NUM_2) || !CheckTensorRank(params.alpha, "alpha", DIM_NUM_1) ||
        !CheckTensorRank(params.bias, "bias", DIM_NUM_1)) {
        return false;
    }
    return params.gammaOptional == nullptr || CheckTensorRank(params.gammaOptional, "gammaOptional", DIM_NUM_2);
}

static bool CheckInputDimsA2(const MhcParamsBase &params)
{
    // x dims must be 3 or 4
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum != DIM_NUM_3 && xDimNum != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(ACLNN_OP_NAME, "x", std::to_string(xDimNum).c_str(), "3 or 4");
        return false;
    }

    if (!CheckTensorRank(params.phi, "phi", DIM_NUM_2)) {
        return false;
    }

    if (!CheckTensorRank(params.alpha, "alpha", DIM_NUM_1)) {
        return false;
    }

    if (!CheckTensorRank(params.bias, "bias", DIM_NUM_1)) {
        return false;
    }
    // 如果gamma非空时，dim必须为2
    if (params.gammaOptional != nullptr && !CheckTensorRank(params.gammaOptional, "gammaOptional", DIM_NUM_2)) {
        return false;
    }
    return true;
}

static bool CheckOutputDims(const MhcParamsBase &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    size_t outputDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_3 : DIM_NUM_2;
    if (!CheckTensorRank(params.hIn, "hIn", outputDimNum) || !CheckTensorRank(params.hPost, "hPost", outputDimNum)) {
        return false;
    }
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    return alphaSize != ALPHA_DIM_SIZE_3 || CheckTensorRank(params.hRes, "hRes", xDimNum);
}

static bool CheckOutputDimsA2(const MhcParamsBase &params)
{
    auto xShape = params.x->GetViewShape();
    auto phiShape = params.phi->GetViewShape();
    auto xDimNum = xShape.GetDimNum();
    int64_t n = 0;
    if (xDimNum == DIM_NUM_4) {
        n = xShape.GetDim(2);
    } else if (xDimNum == DIM_NUM_3) {
        n = xShape.GetDim(1);
    }

    size_t outputDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_3 : DIM_NUM_2;
    if (!CheckTensorRank(params.hIn, "hIn", outputDimNum)) {
        return false;
    }

    if (!CheckTensorRank(params.hPost, "hPost", outputDimNum)) {
        return false;
    }

    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    int64_t expectedPhiRows_factorial = Factorial(n) + 2 * n;
    int64_t expectedPhiRows_pow = n * n + 2 * n;
    // alpha is 3, hRes not null,
    // hres shape should be [b,s,n,n] [t,n,n] [b,s, n!] [t, n!]
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape.GetDim(0) == expectedPhiRows_factorial) {
        return CheckTensorRank(params.hRes, "hRes", (xDimNum - 1));
    } else if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape.GetDim(0) == expectedPhiRows_pow) {
        return CheckTensorRank(params.hRes, "hRes", xDimNum);
    }
    return true;
}

static bool CheckOptionalOutputGroup(const MhcParamsBase &params)
{
    bool hasAnyOptional =
        params.invRmsOptional != nullptr || params.hMixOptional != nullptr || params.hPreOptional != nullptr;
    bool hasAllOptional =
        params.invRmsOptional != nullptr && params.hMixOptional != nullptr && params.hPreOptional != nullptr;
    if (hasAnyOptional && !hasAllOptional) {
        const std::string pointerStates =
            std::string("invRmsOptional=") + (params.invRmsOptional == nullptr ? "nullptr" : "non-null") +
            ", hMixOptional=" + (params.hMixOptional == nullptr ? "nullptr" : "non-null") +
            ", hPreOptional=" + (params.hPreOptional == nullptr ? "nullptr" : "non-null");
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_OP_NAME, "invRmsOptional, hMixOptional, hPreOptional",
                                               pointerStates.c_str(),
                                               "all optional outputs must be nullptr or all must be non-null");
        return false;
    }
    return true;
}

static bool CheckOptionalOutputDims(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    size_t outputDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_3 : DIM_NUM_2;
    size_t invRmsDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_2 : DIM_NUM_1;
    return CheckTensorRank(params.invRmsOptional, "invRmsOptional", invRmsDimNum) &&
           CheckTensorRank(params.hMixOptional, "hMixOptional", outputDimNum) &&
           CheckTensorRank(params.hPreOptional, "hPreOptional", outputDimNum);
}

static bool CheckOptionalOutputDimsA2(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    size_t outputDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_3 : DIM_NUM_2;
    size_t invRmsDimNum = xDimNum == DIM_NUM_4 ? DIM_NUM_2 : DIM_NUM_1;
    if (!CheckTensorRank(params.invRmsOptional, "invRmsOptional", invRmsDimNum)) {
        return false;
    }
    if (!CheckTensorRank(params.hMixOptional, "hMixOptional", outputDimNum)) {
        return false;
    }
    if (!CheckTensorRank(params.hPreOptional, "hPreOptional", outputDimNum)) {
        return false;
    }
    return true;
}

static bool CheckOutputAndOptionalDims(const MhcParamsBase &params)
{
    return CheckOutputDims(params) && CheckOptionalOutputGroup(params) && CheckOptionalOutputDims(params);
}

static void GetXShapeInfo(const MhcParamsBase &params, int64_t &n, int64_t &d, int64_t &nD)
{
    auto xShape = params.x->GetViewShape();
    auto xDimNum = xShape.GetDimNum();
    n = xDimNum == DIM_NUM_4 ? xShape.GetDim(DIM_IDX_2) : xShape.GetDim(DIM_IDX_1);
    d = xDimNum == DIM_NUM_4 ? xShape.GetDim(DIM_IDX_3) : xShape.GetDim(DIM_IDX_2);
    nD = n * d;
}

static bool CheckParamShapes(const MhcParamsBase &params, int64_t n, int64_t d, int64_t nD, int64_t expectedParamRows)
{
    if (!ValidateNDParams(n, d) || !CheckTensorShape(params.phi, {expectedParamRows, nD}, "phi") ||
        !CheckTensorShape(params.bias, {expectedParamRows}, "bias")) {
        return false;
    }
    return params.gammaOptional == nullptr || CheckTensorShape(params.gammaOptional, {n, d}, "gammaOptional");
}

static bool CheckParamShapesA2(const MhcParamsBase &params, int64_t n, int64_t d, int64_t nD, int64_t alphaSize,
                               int64_t expectedPhiRows_factorial, int64_t expectedPhiRows_pow, int64_t phiShape_dim0)
{
    // n,d must bigger than 0  || n must be 4,6,8 ||  d must align to 16/128
    if (!ValidateNDParamsA2(n, d)) {
        return false;
    }
    auto biasShape = params.bias->GetViewShape();
    auto phiShape = params.phi->GetViewShape();
    if (alphaSize == ALPHA_DIM_SIZE_2) {
        int64_t expectedPhiRows = 2 * n;
        if (phiShape_dim0 != expectedPhiRows) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor first dim must be %ld, but got %ld", expectedPhiRows,
                    phiShape_dim0);
            return false;
        }
        if (biasShape.GetDim(0) != expectedPhiRows) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dim must be %ld, but got %ld", expectedPhiRows,
                    biasShape.GetDim(0));
            return false;
        }
    } else {
        if (phiShape_dim0 != expectedPhiRows_pow && phiShape_dim0 != expectedPhiRows_factorial) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor first dim must be %ld or %ld, but got %ld",
                    expectedPhiRows_pow, expectedPhiRows_factorial, phiShape_dim0);
            return false;
        }
        if (biasShape.GetDim(0) != expectedPhiRows_pow && biasShape.GetDim(0) != expectedPhiRows_factorial) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dim must be %ld or %ld, but got %ld", expectedPhiRows_pow,
                    expectedPhiRows_factorial, biasShape.GetDim(0));
            return false;
        }
    }
    if (phiShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor second dim must be nD=%ld, but got %ld", nD, phiShape.GetDim(1));
        return false;
    }

    if (params.gammaOptional != nullptr && !CheckTensorShape(params.gammaOptional, {n, d}, "gammaOptional")) {
        return false;
    }
    return true;
}

static bool CheckOutputShape4D(const MhcParamsBase &params, int64_t n, int64_t d, int64_t expectedPhiRows)
{
    auto xShape = params.x->GetViewShape();
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    int64_t b = xShape.GetDim(DIM_IDX_0);
    int64_t s = xShape.GetDim(DIM_IDX_1);
    if (!CheckTensorShape(params.hIn, {b, s, d}, "hIn") || !CheckTensorShape(params.hPost, {b, s, n}, "hPost")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && !CheckTensorShape(params.hRes, {b, s, n, n}, "hRes")) {
        return false;
    }
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    return CheckTensorShape(params.invRmsOptional, {b, s}, "invRmsOptional") &&
           CheckTensorShape(params.hMixOptional, {b, s, expectedPhiRows}, "hMixOptional") &&
           CheckTensorShape(params.hPreOptional, {b, s, n}, "hPreOptional");
}

static bool CheckOutputShape4DA2(const MhcParamsBase &params, int64_t b, int64_t s, int64_t n, int64_t d,
                                 int64_t alphaSize, int64_t expectedPhiRows_factorial, int64_t expectedPhiRows_pow,
                                 int64_t phiShape_dim0)
{
    if (!CheckTensorShape(params.hIn, {b, s, d}, "hIn")) {
        return false;
    }
    if (!CheckTensorShape(params.hPost, {b, s, n}, "hPost")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_pow &&
        !CheckTensorShape(params.hRes, {b, s, n, n}, "hRes")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_factorial &&
        !CheckTensorShape(params.hRes, {b, s, Factorial(n)}, "hRes")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_pow &&
        !CheckTensorShape(params.hMixOptional, {b, s, expectedPhiRows_pow}, "hMixOptional")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_factorial &&
        !CheckTensorShape(params.hMixOptional, {b, s, expectedPhiRows_factorial}, "hMixOptional")) {
        return false;
    }
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    if (!CheckTensorShape(params.invRmsOptional, {b, s}, "invRmsOptional")) {
        return false;
    }
    if (!CheckTensorShape(params.hPreOptional, {b, s, n}, "hPreOptional")) {
        return false;
    }
    return true;
}

static bool CheckOutputShape3D(const MhcParamsBase &params, int64_t n, int64_t d, int64_t expectedPhiRows)
{
    auto xShape = params.x->GetViewShape();
    int64_t alphaSize = params.alpha->GetViewShape().GetDim(0);
    int64_t t = xShape.GetDim(DIM_IDX_0);
    if (!CheckTensorShape(params.hIn, {t, d}, "hIn") || !CheckTensorShape(params.hPost, {t, n}, "hPost")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && !CheckTensorShape(params.hRes, {t, n, n}, "hRes")) {
        return false;
    }
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    return CheckTensorShape(params.invRmsOptional, {t}, "invRmsOptional") &&
           CheckTensorShape(params.hMixOptional, {t, expectedPhiRows}, "hMixOptional") &&
           CheckTensorShape(params.hPreOptional, {t, n}, "hPreOptional");
}

static bool CheckOutputShape3DA2(const MhcParamsBase &params, int64_t t, int64_t n, int64_t d, int64_t alphaSize,
                                 int64_t expectedPhiRows_factorial, int64_t expectedPhiRows_pow, int64_t phiShape_dim0)
{
    if (!CheckTensorShape(params.hIn, {t, d}, "hIn")) {
        return false;
    }
    if (!CheckTensorShape(params.hPost, {t, n}, "hPost")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_pow &&
        !CheckTensorShape(params.hRes, {t, n, n}, "hRes")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_factorial &&
        !CheckTensorShape(params.hRes, {t, Factorial(n)}, "hRes")) {
        return false;
    }

    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_pow &&
        !CheckTensorShape(params.hMixOptional, {t, expectedPhiRows_pow}, "hMixOptional")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_3 && phiShape_dim0 == expectedPhiRows_factorial &&
        !CheckTensorShape(params.hMixOptional, {t, expectedPhiRows_factorial}, "hMixOptional")) {
        return false;
    }
    if (alphaSize == ALPHA_DIM_SIZE_2 && !CheckTensorShape(params.hMixOptional, {t, 2 * n}, "hMixOptional")) {
        return false;
    }
    if (params.invRmsOptional == nullptr) {
        return true;
    }

    if (!CheckTensorShape(params.invRmsOptional, {t}, "invRmsOptional")) {
        return false;
    }

    if (!CheckTensorShape(params.hPreOptional, {t, n}, "hPreOptional")) {
        return false;
    }
    return true;
}

static bool CheckInputOutShape(const MhcParamsBase &params)
{
    int64_t n = 0;
    int64_t d = 0;
    int64_t nD = 0;
    GetXShapeInfo(params, n, d, nD);
    const int64_t alphaSize = params.alpha->GetViewShape().GetDim(DIM_IDX_0);
    const int64_t expectedParamRows = alphaSize == ALPHA_DIM_SIZE_2 ? 2 * n : n * n + 2 * n;
    if (!CheckParamShapes(params, n, d, nD, expectedParamRows)) {
        return false;
    }
    if (params.x->GetViewShape().GetDimNum() == DIM_NUM_4) {
        return CheckOutputShape4D(params, n, d, expectedParamRows);
    }
    return CheckOutputShape3D(params, n, d, expectedParamRows);
}

static bool CheckInputOutShapeA2(const MhcParamsBase &params)
{
    int64_t n = 0;
    int64_t d = 0;
    int64_t nD = 0;
    GetXShapeInfo(params, n, d, nD);
    const int64_t alphaSize = params.alpha->GetViewShape().GetDim(DIM_IDX_0);
    auto xShape = params.x->GetViewShape();
    int64_t b = xShape.GetDim(DIM_IDX_0);
    int64_t s = xShape.GetDim(DIM_IDX_1);

    auto phiShape = params.phi->GetViewShape();
    int64_t expectedPhiRows_factorial = Factorial(n) + 2 * n;
    int64_t expectedPhiRows_pow = n * n + 2 * n;
    int64_t phiShape_dim0 = phiShape.GetDim(0);
    if (!CheckParamShapesA2(params, n, d, nD, alphaSize, expectedPhiRows_factorial, expectedPhiRows_pow,
                            phiShape_dim0)) {
        return false;
    }
    if (xShape.GetDimNum() == DIM_NUM_4) {
        return CheckOutputShape4DA2(params, b, s, n, d, alphaSize, expectedPhiRows_factorial, expectedPhiRows_pow,
                                    phiShape_dim0);
    }

    if (xShape.GetDimNum() == DIM_NUM_3) {
        int64_t t = xShape.GetDim(DIM_IDX_0);
        return CheckOutputShape3DA2(params, t, n, d, alphaSize, expectedPhiRows_factorial, expectedPhiRows_pow,
                                    phiShape_dim0);
    }
    return true;
}

static bool CheckTensorShape(const aclTensor *tensor, std::initializer_list<int64_t> expectedShape, const char *name)
{
    const auto tensorShape = tensor->GetViewShape();
    const size_t actualRank = tensorShape.GetDimNum();
    if (actualRank != expectedShape.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(ACLNN_OP_NAME, name, std::to_string(actualRank).c_str(),
                                     std::to_string(expectedShape.size()).c_str());
        return false;
    }
    size_t dimIndex = 0;
    for (const auto expectedDim : expectedShape) {
        if (tensorShape.GetDim(dimIndex++) != expectedDim) {
            const std::string actualShape = TensorShapeToString(tensor);
            const std::string expectedShapeStr = ExpectedShapeToString(expectedShape);
            OP_LOGE_FOR_INVALID_SHAPE(ACLNN_OP_NAME, name, actualShape.c_str(), expectedShapeStr.c_str());
            return false;
        }
    }
    return true;
}

static bool CheckAlphaShape(const aclTensor *alphaTensor)
{
    auto alphaShape = alphaTensor->GetViewShape();
    int64_t alphaSize = alphaShape.GetDim(0);
    if (alphaSize != ALPHA_DIM_SIZE_2 && alphaSize != ALPHA_DIM_SIZE_3) {
        const std::string actualShape = TensorShapeToString(alphaTensor);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_OP_NAME, "alpha", actualShape.c_str(), "shape must be [2] or [3]");
        return false;
    }
    return true;
}

static bool ValidateNDParams(int64_t n, int64_t d)
{
    if (n <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "n", std::to_string(n).c_str(),
                                              "n in x shape must be positive");
        return false;
    }
    if (d <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "D", std::to_string(d).c_str(),
                                              "D in x shape must be positive");
        return false;
    }
    bool isValidN = false;
    for (size_t i = 0; i < sizeof(N_VALID_VALUES) / sizeof(N_VALID_VALUES[0]); ++i) {
        if (n == N_VALID_VALUES[i]) {
            isValidN = true;
            break;
        }
    }
    if (!isValidN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "n", std::to_string(n).c_str(),
                                              "n in x shape must be 4, 6 or 8");
        return false;
    }
    if (d % D_ALIGNMENT != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "D", std::to_string(d).c_str(),
                                              "D in x shape must be aligned to 16 elements");
        return false;
    }
    return true;
}

static bool ValidateNDParamsA2(int64_t n, int64_t d)
{
    if (n <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "n", std::to_string(n).c_str(),
                                              "n in x shape must be positive");
        return false;
    }
    if (d <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "D", std::to_string(d).c_str(),
                                              "D in x shape must be positive");
        return false;
    }
    // 910 only support n = 4
    if (n != N_VALID_VALUE_A2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "n", std::to_string(n).c_str(), "n in x shape must be 4");
        return false;
    }

    if (d % D_ALIGNMENT_A2 != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "D", std::to_string(d).c_str(),
                                              "D in x shape must be aligned to 128 elements");
        return false;
    }
    return true;
}

static bool CheckInputDtype(const MhcParamsBase &params)
{
    const std::initializer_list<DataType> X_SUPPORT_DTYPE_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};
    auto xDtype = params.x->GetDataType();
    bool xDtypeValid = false;
    for (const auto &dtype : X_SUPPORT_DTYPE_LIST) {
        if (xDtype == dtype) {
            xDtypeValid = true;
            break;
        }
    }
    if (!xDtypeValid) {
        OP_LOGE_FOR_INVALID_DTYPE(ACLNN_OP_NAME, "x", op::ToString(xDtype).GetString(), "BFLOAT16 or FLOAT16");
        return false;
    }
    if (!CheckTensorDtype(params.phi, "phi", DataType::DT_FLOAT)) {
        return false;
    }
    if (!CheckTensorDtype(params.alpha, "alpha", DataType::DT_FLOAT)) {
        return false;
    }
    if (!CheckTensorDtype(params.bias, "bias", DataType::DT_FLOAT)) {
        return false;
    }
    if (params.gammaOptional != nullptr &&
        !CheckTensorDtype(params.gammaOptional, "gammaOptional", DataType::DT_FLOAT)) {
        return false;
    }
    return true;
}

static bool CheckOutputDtype(const MhcParamsBase &params)
{
    if (!CheckTensorDtype(params.hIn, "hIn", params.x->GetDataType())) {
        return false;
    }
    if (!CheckTensorDtype(params.hPost, "hPost", DataType::DT_FLOAT)) {
        return false;
    }
    if (params.hRes != nullptr && !CheckTensorDtype(params.hRes, "hRes", DataType::DT_FLOAT)) {
        return false;
    }
    return true;
}

static bool CheckOptionalOutputDtype(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    if (!CheckTensorDtype(params.invRmsOptional, "invRmsOptional", DataType::DT_FLOAT)) {
        return false;
    }
    if (!CheckTensorDtype(params.hMixOptional, "hMixOptional", DataType::DT_FLOAT)) {
        return false;
    }
    if (!CheckTensorDtype(params.hPreOptional, "hPreOptional", DataType::DT_FLOAT)) {
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const MhcParamsBase &params)
{
    return CheckInputDtype(params) && CheckOutputDtype(params) && CheckOptionalOutputDtype(params);
}

static bool IsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }

    return false;
}

static bool CheckTensorFormat(const aclTensor *tensor, const char *name)
{
    if (IsPrivateFormat(tensor->GetViewFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ACLNN_OP_NAME, name, op::ToString(tensor->GetViewFormat()).GetString(),
                                               "private format is not supported");
        return false;
    }
    return true;
}

static bool CheckInputFormat(const MhcParamsBase &params)
{
    if (!CheckTensorFormat(params.x, "x")) {
        return false;
    }
    if (!CheckTensorFormat(params.phi, "phi")) {
        return false;
    }
    if (!CheckTensorFormat(params.alpha, "alpha")) {
        return false;
    }
    if (!CheckTensorFormat(params.bias, "bias")) {
        return false;
    }
    if (params.gammaOptional != nullptr && !CheckTensorFormat(params.gammaOptional, "gammaOptional")) {
        return false;
    }
    return true;
}

static bool CheckOutputFormat(const MhcParamsBase &params)
{
    if (!CheckTensorFormat(params.hIn, "hIn")) {
        return false;
    }
    if (!CheckTensorFormat(params.hPost, "hPost")) {
        return false;
    }
    if (params.hRes != nullptr && !CheckTensorFormat(params.hRes, "hRes")) {
        return false;
    }
    return true;
}

static bool CheckOptionalOutputFormat(const MhcParamsBase &params)
{
    if (params.invRmsOptional == nullptr) {
        return true;
    }
    if (!CheckTensorFormat(params.invRmsOptional, "invRmsOptional")) {
        return false;
    }
    if (!CheckTensorFormat(params.hMixOptional, "hMixOptional")) {
        return false;
    }
    if (!CheckTensorFormat(params.hPreOptional, "hPreOptional")) {
        return false;
    }
    return true;
}

static bool CheckFormat(const MhcParamsBase &params)
{
    return CheckInputFormat(params) && CheckOutputFormat(params) && CheckOptionalOutputFormat(params);
}

static bool CheckOpImplMode(const MhcParamsBase &params)
{
    if (params.opImplMode != MHC_PRE_USE_FP32 && params.opImplMode != MHC_PRE_USE_HF32) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "opImplMode", std::to_string(params.opImplMode).c_str(),
                                              "must be 0 (MHC_PRE_USE_FP32) or 1 (MHC_PRE_USE_HF32)");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const MhcParamsBase &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        CHECK_RET(CheckInputDimsA2(params), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckInputDims(params), ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckAlphaShape(params.alpha), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRequiredHRes(params), ACLNN_ERR_PARAM_NULLPTR);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        CHECK_RET(CheckOutputDimsA2(params), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckOptionalOutputGroup(params), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckOptionalOutputDimsA2(params), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckInputOutShapeA2(params), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckOutputAndOptionalDims(params), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckInputOutShape(params), ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOpImplMode(params), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ConvertDataContiguous(MhcParamsBase &params, aclOpExecutor *executor)
{
    // Convert input tensors to contiguous tensors.
    params.xContiguous = l0op::Contiguous(params.x, executor);
    CHECK_RET(params.xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.phiContiguous = l0op::Contiguous(params.phi, executor);
    CHECK_RET(params.phiContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.alphaContiguous = l0op::Contiguous(params.alpha, executor);
    CHECK_RET(params.alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.biasContiguous = l0op::Contiguous(params.bias, executor);
    CHECK_RET(params.biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gammaOptional != nullptr) {
        params.gammaOptionalContiguous = l0op::Contiguous(params.gammaOptional, executor);
        CHECK_RET(params.gammaOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CopyRequiredOutputs(const aclTensor *hInOut, const aclTensor *hPostOut, const aclTensor *hResOut,
                                       MhcParamsBase &params, aclOpExecutor *executor)
{
    auto ret0 = l0op::ViewCopy(hInOut, params.hIn, executor);
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret1 = l0op::ViewCopy(hPostOut, params.hPost, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (params.hasResi && params.hRes != nullptr) {
        auto ret2 = l0op::ViewCopy(hResOut, params.hRes, executor);
        CHECK_RET(ret2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CopyOptionalOutputs(const aclTensor *invRmsOut, const aclTensor *hMixOut, const aclTensor *hPreOut,
                                       MhcParamsBase &params, aclOpExecutor *executor)
{
    if (params.invRmsOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto ret3 = l0op::ViewCopy(invRmsOut, params.invRmsOptional, executor);
    CHECK_RET(ret3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret4 = l0op::ViewCopy(hMixOut, params.hMixOptional, executor);
    CHECK_RET(ret4 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret5 = l0op::ViewCopy(hPreOut, params.hPreOptional, executor);
    CHECK_RET(ret5 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus mHCPreCommonProcess(MhcParamsBase &params, aclOpExecutor *executor)
{
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = ConvertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t alphaSize = params.alphaContiguous->GetViewShape().GetDim(0);
    params.hasResi = (alphaSize == ALPHA_DIM_SIZE_3);
    int64_t outFlag = params.invRmsOptional != nullptr ? 1 : 0;
    auto outParams = l0op::MhcPre(params.xContiguous, params.phiContiguous, params.alphaContiguous,
                                  params.biasContiguous, params.gammaOptionalContiguous, outFlag, params.normEps,
                                  params.hcEps, params.opImplMode, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    ret = CopyRequiredOutputs(std::get<0>(outParams), std::get<1>(outParams), std::get<2>(outParams), params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CopyOptionalOutputs(std::get<3>(outParams), std::get<4>(outParams), std::get<5>(outParams), params,
                               executor);
}

static aclnnStatus MhcPreGetWorkspaceSizeCommon(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                                const aclTensor *bias, const aclTensor *gammaOptional, double normEps,
                                                double hcEps, int64_t opImplMode, aclTensor *hIn, aclTensor *hPost,
                                                aclTensor *hRes, aclTensor *invRmsOptional, aclTensor *hMixOptional,
                                                aclTensor *hPreOptional, uint64_t *workspaceSize,
                                                aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "workspaceSize", "nullptr",
                                              "output parameter must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_OP_NAME, "executor", "nullptr",
                                              "output parameter must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();

    MhcParamsBase params = MhcBuilder::Create()
                               .SetInput(x, phi, alpha, bias, gammaOptional)
                               .SetAttr(normEps, hcEps, opImplMode)
                               .SetOutput(hIn, hPost, hRes)
                               .SetOptionalOutput(invRmsOptional, hMixOptional, hPreOptional)
                               .Build();

    auto ret = mHCPreCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                        const aclTensor *bias, const aclTensor *gammaOptional, double normEps,
                                        double hcEps, aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
                                        aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional,
                                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPre, DFX_IN(x, phi, alpha, bias, gammaOptional, normEps, hcEps),
                   DFX_OUT(hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional));
    return MhcPreGetWorkspaceSizeCommon(x, phi, alpha, bias, gammaOptional, normEps, hcEps, MHC_PRE_USE_FP32, hIn,
                                        hPost, hRes, invRmsOptional, hMixOptional, hPreOptional, workspaceSize,
                                        executor);
}

aclnnStatus aclnnMhcPreV2GetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                          const aclTensor *bias, const aclTensor *gammaOptional, double normEps,
                                          double hcEps, int64_t opImplMode, aclTensor *hIn, aclTensor *hPost,
                                          aclTensor *hRes, aclTensor *invRmsOptional, aclTensor *hMixOptional,
                                          aclTensor *hPreOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreV2, DFX_IN(x, phi, alpha, bias, gammaOptional, normEps, hcEps, opImplMode),
                   DFX_OUT(hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional));
    return MhcPreGetWorkspaceSizeCommon(x, phi, alpha, bias, gammaOptional, normEps, hcEps, opImplMode, hIn, hPost,
                                        hRes, invRmsOptional, hMixOptional, hPreOptional, workspaceSize, executor);
}
aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPre);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMhcPreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace
#ifdef __cplusplus
}
#endif
