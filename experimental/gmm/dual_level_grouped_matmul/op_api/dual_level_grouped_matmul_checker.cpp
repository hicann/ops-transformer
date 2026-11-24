// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "dual_level_grouped_matmul_checker.h"
#include "gmm/common/op_api/gmm_tensor_storage_check.h"
#include "opdev/op_executor.h"
#include "log/log.h"
#include "util/math_util.h"
#include <initializer_list>
#include <limits>
#include <string>
using namespace op;
namespace dual_level_gmm {
namespace {
constexpr char API_NAME[] = "aclnnDualLevelGroupedMatmulWeightNz";
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr int64_t MX_SCALE_K = MX_SCALE_PAIR * LEVEL1_GROUP_SIZE;
constexpr int64_t FP4_ELEMENTS_PER_BYTE = 2;
constexpr int64_t MIN_K = FP4_ELEMENTS_PER_BYTE;
constexpr int64_t MATRIX_RANK = 2;
constexpr int64_t EXPERT_WEIGHT_RANK = 3;
constexpr int64_t SPLIT_ITEM_SINGLE_OUTPUT = 2;
constexpr int64_t SPLIT_ITEM_DEFAULT = 3;
constexpr int64_t SPARSE_GROUP_LIST = 2;
constexpr int64_t GROUP_LIST_PAIR_SIZE = 2;
constexpr int64_t WEIGHT_N_AXIS = 2;
constexpr int64_t NZ_ROWS = 16;
constexpr int64_t NZ_COLS = 64;

template <typename Values>
std::string ValuesString(const Values& values)
{
    std::string out = "(";
    for (const auto value : values) {
        if (out.size() != 1) {
            out += ",";
        }
        out += std::to_string(value);
    }
    return out + ")";
}

std::string ShapeString(const Shape& shape)
{
    std::string out = "(";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(shape.GetDim(i));
    }
    return out + ")";
}

void LogTensorMismatch(const char* name, const aclTensor* tensor, const std::string& expected)
{
    const auto descriptor =
        "shape=" + ShapeString(tensor->GetViewShape()) + ", storage=" + ShapeString(tensor->GetStorageShape()) +
        ", stride=" + ValuesString(tensor->GetViewStrides()) + ", offset=" + std::to_string(tensor->GetViewOffset()) +
        ", dtype=" + std::to_string(static_cast<int>(tensor->GetDataType())) +
        ", format=" + std::to_string(static_cast<int>(tensor->GetStorageFormat()));
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, name, descriptor.c_str(), expected.c_str());
}

bool MatchShape(const Shape& shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (auto dim : dims) {
        if (shape.GetDim(i++) != dim) {
            return false;
        }
    }
    return true;
}

bool CheckShape(const aclTensor* tensor, std::initializer_list<int64_t> dims, const char* name)
{
    if (!MatchShape(tensor->GetViewShape(), dims)) {
        LogTensorMismatch(name, tensor, "logical shape=" + ValuesString(dims));
        return false;
    }
    return true;
}

bool MatchStrides(const aclTensor* tensor, std::initializer_list<int64_t> dims)
{
    const auto& strides = tensor->GetViewStrides();
    if (tensor->GetViewOffset() != 0 || strides.size() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (auto stride : dims) {
        if (strides[i++] != stride) {
            return false;
        }
    }
    return true;
}

bool IsDenseNd(const aclTensor* tensor)
{
    if (tensor->GetViewOffset() != 0 || tensor->GetStorageFormat() != Format::FORMAT_ND) {
        return false;
    }
    const auto& shape = tensor->GetViewShape();
    const auto& storage = tensor->GetStorageShape();
    const auto& strides = tensor->GetViewStrides();
    if (shape.GetDimNum() != storage.GetDimNum() || strides.size() != shape.GetDimNum()) {
        return false;
    }
    const bool empty = tensor->IsEmpty();
    int64_t stride = 1;
    for (size_t i = shape.GetDimNum(); i > 0; --i) {
        const auto dim = shape.GetDim(i - 1);
        if (dim < 0 || dim != storage.GetDim(i - 1) ||
            (!empty && (dim == 0 || strides[i - 1] != stride || stride > std::numeric_limits<int64_t>::max() / dim))) {
            return false;
        }
        if (!empty) {
            stride *= dim;
        }
    }
    return true;
}

aclnnStatus CheckNotNull(const DualLevelGroupedMatmulParams& params)
{
    if (params.groupListOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "groupListOptional", "must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    struct NamedList {
        const char* name;
        const aclTensorList* list;
    };
    const NamedList lists[] = {{"x", params.x},
                               {"weight", params.weight},
                               {"scaleOptional", params.scaleOptional},
                               {"perTokenScaleOptional", params.perTokenScaleOptional},
                               {"xLevel0Scale", params.xLevel0Scale},
                               {"weightLevel0Scale", params.weightLevel0Scale},
                               {"out", params.out},
                               {"biasOptional", params.biasOptional}};
    for (const auto& input : lists) {
        if (input.list == nullptr) {
            if (input.name == std::string("biasOptional")) {
                continue;
            }
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, input.name, "must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (input.list->Size() != 1) {
            OP_LOGE_FOR_INVALID_TENSORNUM(API_NAME, input.name, input.list->Size(), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*input.list)[0] == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, input.name,
                                                     "the TensorList element must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

bool CheckStorageBounds(const DualLevelGroupedMatmulParams& params)
{
    using gmm::CheckTensorListStorageBounds;
    // The caller's transposed weight and scale already describe their storage spans.
    return CheckTensorListStorageBounds(params.x, API_NAME, "x") &&
           CheckTensorListStorageBounds(params.weight, API_NAME, "weight") &&
           CheckTensorListStorageBounds(params.biasOptional, API_NAME, "biasOptional") &&
           CheckTensorListStorageBounds(params.scaleOptional, API_NAME, "scaleOptional") &&
           CheckTensorListStorageBounds(params.perTokenScaleOptional, API_NAME, "perTokenScaleOptional") &&
           gmm::CheckTensorStorageBounds(params.groupListOptional, API_NAME, "groupListOptional") &&
           CheckTensorListStorageBounds(params.xLevel0Scale, API_NAME, "xLevel0Scale") &&
           CheckTensorListStorageBounds(params.weightLevel0Scale, API_NAME, "weightLevel0Scale") &&
           CheckTensorListStorageBounds(params.out, API_NAME, "out");
}

aclnnStatus CheckAttributes(const DualLevelGroupedMatmulParams& params)
{
    // With one x and one output tensor, splitItem=2 and 3 use the same layout.
    struct Attribute {
        const char* name;
        int64_t value;
        int64_t low;
        int64_t high;
    };
    const Attribute attributes[] = {{"splitItem", params.splitItem, SPLIT_ITEM_SINGLE_OUTPUT, SPLIT_ITEM_DEFAULT},
                                    {"groupType", params.groupType, 0, 0},
                                    {"groupListType", params.groupListType, 0, SPARSE_GROUP_LIST},
                                    {"level0GroupSize", params.level0GroupSize, LEVEL0_GROUP_SIZE, LEVEL0_GROUP_SIZE},
                                    {"level1GroupSize", params.level1GroupSize, LEVEL1_GROUP_SIZE, LEVEL1_GROUP_SIZE}};
    for (const auto& attr : attributes) {
        if (attr.value < attr.low || attr.value > attr.high) {
            const auto expected = "expected range [" + std::to_string(attr.low) + "," + std::to_string(attr.high) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, attr.name, std::to_string(attr.value).c_str(),
                                                  expected.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.tuningConfigOptional != nullptr &&
        (params.tuningConfigOptional->Size() != 1 || (*params.tuningConfigOptional)[0] != 0)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "tuningConfigOptional", "expected nullptr or {0}");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const DualLevelGroupedMatmulParams& params)
{
    struct TypedInput {
        const char* name;
        const aclTensor* tensor;
        DataType expected;
    };
    const TypedInput inputs[] = {
        {"x", (*params.x)[0], DataType::DT_FLOAT4_E2M1},
        {"weight", (*params.weight)[0], DataType::DT_FLOAT4_E2M1},
        {"xLevel0Scale", (*params.xLevel0Scale)[0], DataType::DT_FLOAT},
        {"weightLevel0Scale", (*params.weightLevel0Scale)[0], DataType::DT_FLOAT},
        {"scaleOptional", (*params.scaleOptional)[0], DataType::DT_FLOAT8_E8M0},
        {"perTokenScaleOptional", (*params.perTokenScaleOptional)[0], DataType::DT_FLOAT8_E8M0},
        {"groupListOptional", params.groupListOptional, DataType::DT_INT64}};
    for (const auto& input : inputs) {
        if (input.tensor->GetDataType() != input.expected) {
            LogTensorMismatch(input.name, input.tensor, "dtype=" + std::to_string(static_cast<int>(input.expected)));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.biasOptional != nullptr && (*params.biasOptional)[0]->GetDataType() != DataType::DT_FLOAT) {
        LogTensorMismatch("biasOptional", (*params.biasOptional)[0], "FP32 bias");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto* output = (*params.out)[0];
    if (output->GetDataType() != DataType::DT_FLOAT16 && output->GetDataType() != DataType::DT_BF16) {
        LogTensorMismatch("out", output, "FP16 or BF16 output");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckWeightStorage(const aclTensor* w, const aclTensor* b1, bool transpose, int64_t e, int64_t k, int64_t n,
                               int64_t mxScaleGroups)
{
    if (b1->GetStorageFormat() != Format::FORMAT_ND ||
        !MatchShape(b1->GetViewShape(), {e, mxScaleGroups, n, MX_SCALE_PAIR})) {
        LogTensorMismatch("scaleOptional", b1, "ND view=(E,ceil(K/64),N,2)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (w->IsEmpty() && (transpose || n % FP4_ELEMENTS_PER_BYTE == 0) && w->GetViewOffset() == 0 &&
        b1->GetViewOffset() == 0) {
        return ACLNN_SUCCESS;
    }
    if (!transpose && (n % FP4_ELEMENTS_PER_BYTE != 0 || !MatchStrides(w, {k * n, n, 1}))) {
        LogTensorMismatch("weight", w, "even N and canonical NZ contiguous/transposed view with zero offset");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t weightRows = transpose ? n : k;
    const int64_t weightCols = transpose ? k : n;
    const std::initializer_list<int64_t> nzShape{e, Ops::Base::CeilDiv(weightCols, NZ_COLS),
                                                 Ops::Base::CeilDiv(weightRows, NZ_ROWS), NZ_ROWS, NZ_COLS};
    if (!MatchShape(w->GetStorageShape(), nzShape)) {
        LogTensorMismatch("weight", w, "NZ storage=" + ValuesString(nzShape));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (transpose) {
        if (!MatchShape(b1->GetStorageShape(), {e, n, mxScaleGroups, MX_SCALE_PAIR}) ||
            !MatchStrides(b1, {n * mxScaleGroups * MX_SCALE_PAIR, MX_SCALE_PAIR, mxScaleGroups * MX_SCALE_PAIR, 1})) {
            LogTensorMismatch("scaleOptional", b1, "transposed scale storage=(E,N,ceil(K/64),2) with zero offset");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (!IsDenseNd(b1)) {
            LogTensorMismatch("scaleOptional", b1, "dense ND storage with zero offset");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus CheckDenseInputs(const DualLevelGroupedMatmulParams& params)
{
    struct NamedTensor {
        const char* name;
        const aclTensor* tensor;
    };
    const NamedTensor inputs[] = {{"x", (*params.x)[0]},
                                  {"xLevel0Scale", (*params.xLevel0Scale)[0]},
                                  {"perTokenScaleOptional", (*params.perTokenScaleOptional)[0]},
                                  {"weightLevel0Scale", (*params.weightLevel0Scale)[0]},
                                  {"groupListOptional", params.groupListOptional},
                                  {"out", (*params.out)[0]}};
    for (const auto& input : inputs) {
        if (!IsDenseNd(input.tensor)) {
            LogTensorMismatch(input.name, input.tensor, "dense ND storage with zero offset");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.biasOptional != nullptr && !IsDenseNd((*params.biasOptional)[0])) {
        LogTensorMismatch("biasOptional", (*params.biasOptional)[0], "dense ND storage with zero offset");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShapeAndFormat(const DualLevelGroupedMatmulParams& params, bool& transposeWeight)
{
    const auto* a = (*params.x)[0];
    const auto* w = (*params.weight)[0];
    const auto* a0 = (*params.xLevel0Scale)[0];
    const auto* b0 = (*params.weightLevel0Scale)[0];
    const auto* a1 = (*params.perTokenScaleOptional)[0];
    const auto* b1 = (*params.scaleOptional)[0];
    const auto* bias = params.biasOptional == nullptr ? nullptr : (*params.biasOptional)[0];
    auto* y = (*params.out)[0];
    const auto& wShape = w->GetViewShape();
    if (wShape.GetDimNum() != EXPERT_WEIGHT_RANK || w->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        LogTensorMismatch("weight", w, "rank=3 and FRACTAL_NZ format");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto e = wShape.GetDim(0);
    const auto k = wShape.GetDim(1);
    const auto n = wShape.GetDim(WEIGHT_N_AXIS);
    const auto& xShape = a->GetViewShape();
    if (xShape.GetDimNum() != MATRIX_RANK) {
        LogTensorMismatch("x", a, "rank=2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto m = xShape.GetDim(0);
    const bool emptyOutput = m == 0 || n == 0;
    if (xShape.GetDim(1) == 0 && !emptyOutput) {
        LogTensorMismatch("x", a, "K may be zero only when M or N is zero");
        return ACLNN_ERR_PARAM_INVALID;
    }
    constexpr int64_t maxDim = std::numeric_limits<int32_t>::max() - 1;
    if (e <= 0 || e > maxDim || k > maxDim || k % FP4_ELEMENTS_PER_BYTE != 0 || n > maxDim ||
        (!emptyOutput && (k <= MIN_K || n <= 1))) {
        LogTensorMismatch("weight", w,
                          "positive E, even K, dimensions at most " + std::to_string(maxDim) +
                              ", and K > 2 and N > 1 for nonempty output");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto mxScaleGroups = Ops::Base::CeilDiv(k, MX_SCALE_K);
    const auto level0Groups = Ops::Base::CeilDiv(k, LEVEL0_GROUP_SIZE);
    if (m > maxDim) {
        LogTensorMismatch("x", a, "0<=M<=" + std::to_string(maxDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckShape(a, {m, k}, "x") && CheckShape(a0, {m, level0Groups}, "xLevel0Scale") &&
                  CheckShape(a1, {m, mxScaleGroups, MX_SCALE_PAIR}, "perTokenScaleOptional") &&
                  CheckShape(b0, {e, level0Groups, n}, "weightLevel0Scale") &&
                  (params.groupListType == SPARSE_GROUP_LIST ?
                       CheckShape(params.groupListOptional, {e, GROUP_LIST_PAIR_SIZE}, "groupListOptional") :
                       CheckShape(params.groupListOptional, {e}, "groupListOptional")) &&
                  CheckShape(y, {m, n}, "out") && (bias == nullptr || CheckShape(bias, {e, n}, "biasOptional")),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDenseInputs(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    transposeWeight = MatchStrides(w, {k * n, 1, k});
    return CheckWeightStorage(w, b1, transposeWeight, e, k, n, mxScaleGroups);
}
} // namespace
aclnnStatus CheckParams(const DualLevelGroupedMatmulParams& params, bool& transposeWeight)
{
    const auto checkNotNullRet = CheckNotNull(params);
    if (checkNotNullRet != ACLNN_SUCCESS) {
        return checkNotNullRet;
    }
    CHECK_RET(CheckStorageBounds(params), ACLNN_ERR_PARAM_INVALID);
    const auto checkAttributesRet = CheckAttributes(params);
    if (checkAttributesRet != ACLNN_SUCCESS) {
        return checkAttributesRet;
    }
    const auto checkDtypeRet = CheckDtype(params);
    if (checkDtypeRet != ACLNN_SUCCESS) {
        return checkDtypeRet;
    }
    return CheckShapeAndFormat(params, transposeWeight);
}
} // namespace dual_level_gmm
