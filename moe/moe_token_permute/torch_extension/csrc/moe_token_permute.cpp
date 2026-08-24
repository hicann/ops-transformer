// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <tuple>

#include <torch/extension.h>
#include "aclnn_common.h"
#if __has_include("aclnnop/aclnn_moe_token_permute_v2.h")
#include "aclnnop/aclnn_moe_token_permute_v2.h"
#elif __has_include("aclnn_moe_token_permute_v2.h")
#include "aclnn_moe_token_permute_v2.h"
#endif

namespace {
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t QUANT_MODE_NONE = -1;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
constexpr int64_t MX_QUANT_BLOCK_SIZE = 32;
constexpr int64_t MXFP4_SCALE_BLOCK_SIZE = 64;
constexpr int64_t PAD_TO_EVEN_FACTOR = 2;

struct MoeTokenPermuteOutput {
    at::Tensor permutedTokens;
    at::Tensor sortedIndices;
    at::Tensor expandedScale;
};

bool IsMxQuantMode(int64_t quantMode)
{
    return quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN ||
           quantMode == QUANT_MODE_MXFP4_E2M1;
}

bool IsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}

int64_t CeilDiv(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

int64_t AlignUp(int64_t value, int64_t align) { return CeilDiv(value, align) * align; }

void CheckMoeTokenPermuteInputs(const at::Tensor &tokens, const at::Tensor &indices)
{
    TORCH_CHECK(tokens.dim() == DIM_TWO, "tokens must be a 2D tensor, but got ", tokens.dim(), " dimensions.");
    TORCH_CHECK(indices.dim() == DIM_ONE || indices.dim() == DIM_TWO, "indices must be a 1D or 2D tensor, but got ",
                indices.dim(), " dimensions.");
}

int64_t GetActualNumOutTokens(const at::Tensor &indices, c10::optional<int64_t> numOutTokens)
{
    int64_t flattenSize = indices.numel();
    int64_t numOutTokensValue = numOutTokens.value_or(0);
    if (numOutTokensValue > 0) {
        return std::min(numOutTokensValue, flattenSize);
    }
    return std::max<int64_t>(numOutTokensValue + flattenSize, 0);
}

int64_t ResolveQuantMode(const at::Tensor &tokens, int64_t quantMode)
{
    int64_t effectiveQuantMode = IsAscend950() ? quantMode : QUANT_MODE_NONE;
    TORCH_CHECK(effectiveQuantMode == QUANT_MODE_NONE || IsMxQuantMode(effectiveQuantMode),
                "quant_mode only supports -1, 2, 3 or 9 on Ascend 950.");
    TORCH_CHECK(!tokens.requires_grad() || effectiveQuantMode == QUANT_MODE_NONE,
                "moe_token_permute does not support autograd when quant_mode is 2, 3 or 9.");
    return effectiveQuantMode;
}

MoeTokenPermuteOutput CreateMoeTokenPermuteOutput(const at::Tensor &tokens, const at::Tensor &indices,
                                                  int64_t actualNumOutTokens, int64_t quantMode)
{
    auto tokenOptions = tokens.options();
    int64_t hiddenSize = tokens.size(1);
    MoeTokenPermuteOutput output;
    auto localDevice = c10::Device(tokens.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    if (quantMode == QUANT_MODE_MXFP8_E5M2) {
        output.permutedTokens = at::empty({actualNumOutTokens, hiddenSize}, tokenOptions.dtype(at::kFloat8_e5m2));
        int64_t scaleCols = AlignUp(CeilDiv(hiddenSize, MX_QUANT_BLOCK_SIZE), PAD_TO_EVEN_FACTOR);
        output.expandedScale = at::empty({actualNumOutTokens, scaleCols}, tokenOptions.dtype(at::kFloat8_e8m0fnu));
    } else if (quantMode == QUANT_MODE_MXFP8_E4M3FN) {
        output.permutedTokens = at::empty({actualNumOutTokens, hiddenSize}, tokenOptions.dtype(at::kFloat8_e4m3fn));
        int64_t scaleCols = AlignUp(CeilDiv(hiddenSize, MX_QUANT_BLOCK_SIZE), PAD_TO_EVEN_FACTOR);
        output.expandedScale = at::empty({actualNumOutTokens, scaleCols}, tokenOptions.dtype(at::kFloat8_e8m0fnu));
    } else if (quantMode == QUANT_MODE_MXFP4_E2M1) {
        TORCH_CHECK(hiddenSize % 2 == 0, "The hidden size must be even when quant_mode is 9.");
        output.permutedTokens = at::empty({actualNumOutTokens, hiddenSize / 2}, tokenOptions.dtype(at::kByte));
        output.expandedScale = at::empty({actualNumOutTokens, CeilDiv(hiddenSize, MXFP4_SCALE_BLOCK_SIZE), 2},
                                         tokenOptions.dtype(at::kByte));
    } else {
        output.permutedTokens = at::empty({actualNumOutTokens, hiddenSize}, tokenOptions);
        output.expandedScale = at::empty({0}, tokenOptions.dtype(at::kFloat));
    }
    output.sortedIndices = at::empty({indices.numel()}, indices.options().dtype(at::kInt));
    return output;
}

TensorWrapper MakeTensorWrapper(const at::Tensor &tensor)
{
    return {tensor, ConvertToAclDataType(tensor.scalar_type())};
}

void FixMxfp4AclDtypes(int64_t quantMode, TensorWrapper &permutedTokensWrapper, TensorWrapper &expandedScaleWrapper)
{
    if (quantMode != QUANT_MODE_MXFP4_E2M1) {
        return;
    }
    permutedTokensWrapper.dtype = aclDataType::ACL_FLOAT4_E2M1;
    expandedScaleWrapper.dtype = aclDataType::ACL_FLOAT8_E8M0;
}
} // namespace

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor> MoeTokenPermute(const at::Tensor &tokens, const at::Tensor &indices,
                                                               c10::optional<int64_t> numOutTokens, bool paddedMode,
                                                               int64_t quantMode)
{
    CheckMoeTokenPermuteInputs(tokens, indices);
    int64_t actualNumOutTokens = GetActualNumOutTokens(indices, numOutTokens);
    int64_t effectiveQuantMode = ResolveQuantMode(tokens, quantMode);
    auto output = CreateMoeTokenPermuteOutput(tokens, indices, actualNumOutTokens, effectiveQuantMode);
    auto permutedTokensWrapper = MakeTensorWrapper(output.permutedTokens);
    auto expandedScaleWrapper = MakeTensorWrapper(output.expandedScale);
    FixMxfp4AclDtypes(effectiveQuantMode, permutedTokensWrapper, expandedScaleWrapper);

    ACLNN_CMD(aclnnMoeTokenPermuteV2, tokens, indices, actualNumOutTokens, paddedMode, effectiveQuantMode,
              permutedTokensWrapper, output.sortedIndices, expandedScaleWrapper);

    return std::make_tuple(std::move(output.permutedTokens), std::move(output.sortedIndices),
                           std::move(output.expandedScale));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe_token_permute", &MoeTokenPermute, "moe_token_permute"); }
} // namespace op_api
