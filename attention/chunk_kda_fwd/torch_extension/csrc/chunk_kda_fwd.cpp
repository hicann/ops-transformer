/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_kda_fwd.cpp
 * \brief Torch bridge for aclnnChunkKdaFwd.
 */

#include <string>
#include <tuple>
#include <vector>

#include <aclnn/acl_meta.h>
#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t MAX_HEADS = 128;
constexpr int64_t MAX_HEAD_DIM = 256;
constexpr int64_t MAX_SEQUENCES = 1024;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

void CheckCuSeqlens(const c10::optional<std::vector<int64_t>>& cuSeqlens, int64_t totalTokens)
{
    if (!cuSeqlens.has_value()) {
        return;
    }
    const auto& cu = cuSeqlens.value();
    TORCH_CHECK(cu.size() >= 2, "chunk_kda_fwd: cu_seqlens must contain at least [0, total_tokens].");
    TORCH_CHECK(cu.front() == 0 && cu.back() == totalTokens,
                "chunk_kda_fwd: cu_seqlens must start at 0 and end at the input sequence length.");
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        TORCH_CHECK(cu[i] <= cu[i + 1], "chunk_kda_fwd: cu_seqlens must be nondecreasing.");
    }
}

int64_t GetTotalChunks(int64_t seqlen, int64_t chunkSize, const c10::optional<std::vector<int64_t>>& cuSeqlens,
                       const c10::optional<std::vector<int64_t>>& chunkIndices)
{
    if (chunkIndices.has_value()) {
        TORCH_CHECK(chunkIndices->size() % 2 == 0,
                    "chunk_kda_fwd: chunk_indices must contain (seq_id, chunk_id) pairs.");
        return static_cast<int64_t>(chunkIndices->size() / 2);
    }
    if (!cuSeqlens.has_value()) {
        return CeilDiv(seqlen, chunkSize);
    }
    int64_t total = 0;
    const auto& cu = cuSeqlens.value();
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        total += CeilDiv(cu[i + 1] - cu[i], chunkSize);
    }
    return total;
}

c10::optional<at::IntArrayRef> ToIntArrayRef(const c10::optional<std::vector<int64_t>>& values)
{
    if (!values.has_value()) {
        return c10::nullopt;
    }
    return at::IntArrayRef(values.value());
}

at::Tensor UndefinedTensor()
{
    return at::Tensor();
}
} // namespace

using ChunkKdaFwdResult = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                     at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

ChunkKdaFwdResult ChunkKdaFwd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                              const at::Tensor& beta, double scale, const c10::optional<at::Tensor>& initialState,
                              bool outputFinalState, const c10::optional<std::vector<int64_t>>& cuSeqlens,
                              const c10::optional<std::vector<int64_t>>& chunkIndices, bool safeGate, double lowerBound,
                              bool useGateInKernel, const c10::optional<at::Tensor>& aLog,
                              const c10::optional<at::Tensor>& dtBias, int64_t chunkSize, bool disableRecompute,
                              bool returnIntermediateStates, bool stateVFirst, const std::string& layout)
{
    TORCH_CHECK(layout == "BSND" || layout == "BNSD" || layout == "TND" || layout == "NTD",
                "chunk_kda_fwd: layout must be one of BSND, BNSD, TND, NTD.");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128, "chunk_kda_fwd: chunk_size must be 64 or 128.");
    const bool rank3 = layout == "TND" || layout == "NTD";
    TORCH_CHECK(q.dim() == (rank3 ? 3 : 4) && k.dim() == q.dim() && v.dim() == q.dim() && g.dim() == q.dim(),
                "chunk_kda_fwd: q/k/v/g ranks must match layout.");
    TORCH_CHECK(beta.dim() == q.dim() - 1, "chunk_kda_fwd: beta rank must be one less than q rank.");
    TORCH_CHECK(q.sizes() == k.sizes(), "chunk_kda_fwd: q and k shapes must match.");

    int64_t batch = rank3 ? 1 : q.size(0);
    int64_t seqlen = 0;
    int64_t qHeads = 0;
    int64_t vHeads = 0;
    if (layout == "TND") {
        seqlen = q.size(0);
        qHeads = q.size(1);
        vHeads = v.size(1);
    } else if (layout == "NTD") {
        qHeads = q.size(0);
        seqlen = q.size(1);
        vHeads = v.size(0);
    } else if (layout == "BNSD") {
        qHeads = q.size(1);
        seqlen = q.size(2);
        vHeads = v.size(1);
    } else {
        seqlen = q.size(1);
        qHeads = q.size(2);
        vHeads = v.size(2);
    }
    const int64_t kDim = q.size(q.dim() - 1);
    const int64_t vDim = v.size(v.dim() - 1);
    TORCH_CHECK(qHeads > 0 && vHeads >= qHeads && vHeads <= MAX_HEADS && vHeads % qHeads == 0,
                "chunk_kda_fwd: heads must satisfy 0 < H <= HV <= 128 and HV % H == 0.");
    TORCH_CHECK(
        kDim >= 16 && kDim <= MAX_HEAD_DIM && kDim % 16 == 0 && vDim >= 16 && vDim <= MAX_HEAD_DIM && vDim % 16 == 0,
        "chunk_kda_fwd: K and V must be multiples of 16 in [16, 256].");
    CheckCuSeqlens(cuSeqlens, seqlen);
    TORCH_CHECK(!cuSeqlens.has_value() || rank3 || batch == 1, "chunk_kda_fwd: rank-4 varlen input requires B=1.");

    const int64_t seqNum = cuSeqlens.has_value() ? static_cast<int64_t>(cuSeqlens->size()) - 1 : batch;
    TORCH_CHECK(seqNum <= MAX_SEQUENCES, "chunk_kda_fwd: at most 1024 sequences are supported.");
    const int64_t totalChunks = GetTotalChunks(seqlen, chunkSize, cuSeqlens, chunkIndices);
    const std::vector<int64_t> attnShape =
        rank3 ? std::vector<int64_t>{seqlen, vHeads, vDim} : std::vector<int64_t>{batch, seqlen, vHeads, vDim};
    const std::vector<int64_t> matrixShape = rank3 ? std::vector<int64_t>{vHeads, seqlen, chunkSize} :
                                                     std::vector<int64_t>{batch, vHeads, seqlen, chunkSize};
    const std::vector<int64_t> kShape =
        rank3 ? std::vector<int64_t>{vHeads, seqlen, kDim} : std::vector<int64_t>{batch, vHeads, seqlen, kDim};
    const std::vector<int64_t> vShape =
        rank3 ? std::vector<int64_t>{vHeads, seqlen, vDim} : std::vector<int64_t>{batch, vHeads, seqlen, vDim};
    const std::vector<int64_t> stateShape = stateVFirst ? std::vector<int64_t>{seqNum, vHeads, vDim, kDim} :
                                                          std::vector<int64_t>{seqNum, vHeads, kDim, vDim};
    std::vector<int64_t> hShape;
    if (rank3) {
        hShape = stateVFirst ? std::vector<int64_t>{totalChunks, vHeads, vDim, kDim} :
                               std::vector<int64_t>{totalChunks, vHeads, kDim, vDim};
    } else {
        hShape = stateVFirst ? std::vector<int64_t>{batch, totalChunks, vHeads, vDim, kDim} :
                               std::vector<int64_t>{batch, totalChunks, vHeads, kDim, vDim};
    }

    at::Tensor attnOut;
    at::Tensor finalState;
    at::Tensor gkOut;
    at::Tensor aqk;
    at::Tensor akk;
    at::Tensor w;
    at::Tensor u;
    at::Tensor qg;
    at::Tensor kg;
    at::Tensor vNew;
    at::Tensor h;
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(q.device()));
        attnOut = at::empty(attnShape, v.options());
        finalState = outputFinalState ? at::empty(stateShape, q.options().dtype(at::kFloat)) : UndefinedTensor();
        gkOut = (!useGateInKernel || disableRecompute) ? at::empty(kShape, q.options().dtype(at::kFloat)) :
                                                         UndefinedTensor();
        aqk = at::empty(matrixShape, q.options());
        akk = at::empty(matrixShape, q.options());
        w = disableRecompute ? at::empty(kShape, q.options()) : UndefinedTensor();
        u = disableRecompute ? at::empty(vShape, v.options()) : UndefinedTensor();
        qg = disableRecompute ? at::empty(kShape, q.options()) : UndefinedTensor();
        kg = disableRecompute ? at::empty(kShape, k.options()) : UndefinedTensor();
        vNew = disableRecompute ? at::empty(vShape, v.options()) : UndefinedTensor();
        h = (disableRecompute || returnIntermediateStates) ? at::empty(hShape, q.options()) : UndefinedTensor();
    }

    const at::Tensor initialStateTensor = initialState.has_value() ? initialState.value() : UndefinedTensor();
    const at::Tensor aLogTensor = aLog.has_value() ? aLog.value() : UndefinedTensor();
    const at::Tensor dtBiasTensor = dtBias.has_value() ? dtBias.value() : UndefinedTensor();
    const auto cuSeqlensRef = ToIntArrayRef(cuSeqlens);
    const auto chunkIndicesRef = ToIntArrayRef(chunkIndices);
    const char* layoutPtr = layout.c_str();
    ACLNN_CMD(aclnnChunkKdaFwd, q, k, v, g, beta, aLogTensor, dtBiasTensor, initialStateTensor, cuSeqlensRef,
              chunkIndicesRef, layoutPtr, scale, chunkSize, safeGate, lowerBound, useGateInKernel, stateVFirst, attnOut,
              finalState, gkOut, aqk, akk, w, u, qg, kg, vNew, h);

    return std::make_tuple(attnOut, finalState, gkOut, aqk, akk, w, u, qg, kg, vNew, h, initialStateTensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_kda_fwd", &ChunkKdaFwd, "ChunkKdaFwd torch wrapper");
}
} // namespace op_api
