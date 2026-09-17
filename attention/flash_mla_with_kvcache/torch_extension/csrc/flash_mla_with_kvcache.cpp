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
 * \file flash_mla_with_kvcache.cpp
 * \brief ACLNN wrapper for flash_mla_with_kvcache / flash_mla_with_kvcache_metadata (DeepSeek MLA, m0070 interface).
 *        Interface: q, k_cache, block_table, cache_seqlens, cu_seqlens_q, seqused_q, attn_mask, metadata
 *        + attrs head_dim_v, softmax_scale, mask_mode, max_seqlen_q, max_seqlen_kv,
 *          layout_q, layout_kv, layout_out, return_softmax_lse
 *        Outputs: attn_out, softmax_lse (optional).
 *        NO v input, NO separate q_rope/k_rope tensors (rope merged into q/k_cache, last dim 576).
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int64_t DIM_ONE = 1;
const int64_t DIM_TWO = 2;
const int64_t DIM_THREE = 3;
const int64_t MAX_DIM_SIZE = 8;

// MLA geometry (m0070 / R1): 每核 16 word，kv head num == 1。
// Metadata buffer worst-case capacity (plan B2/D16): ((aic + aiv) * batch_size * 1 + 1) * 16 int32 words,
// aligned to 4096 int32 elements (same as flash_attn.py)。aic/aiv 核数由 python 侧
// 从硬件获取（torch.npu.get_device_limit）透传，不写死常量。
constexpr int64_t MLA_META_WORDS_PER_CORE = 16;
constexpr int64_t MLA_METADATA_ALIGN_WORDS = 4096;

int64_t CalculateMlaMetadataSizeWords(int64_t batchSize, int64_t aicCoreNum, int64_t aivCoreNum)
{
    // kv==1 upper bound: ((aic + aiv) * batch * 1 + 1) * 16 (int32 words)
    const int64_t words = ((aicCoreNum + aivCoreNum) * batchSize * DIM_ONE + DIM_ONE) * MLA_META_WORDS_PER_CORE;
    return ((words + MLA_METADATA_ALIGN_WORDS - 1) / MLA_METADATA_ALIGN_WORDS) * MLA_METADATA_ALIGN_WORDS;
}

int64_t CalculateMlaMetadataSizeBytes(int64_t batchSize, int64_t aicCoreNum, int64_t aivCoreNum)
{
    return CalculateMlaMetadataSizeWords(batchSize, aicCoreNum, aivCoreNum) * static_cast<int64_t>(sizeof(int32_t));
}

at::Tensor FlashMlaWithKvcacheMetadata(const c10::optional<at::Tensor> &cuSeqlensQ, const at::Tensor &cacheSeqlens,
                                       const c10::optional<at::Tensor> &sequsedQ, int64_t maxSeqlenQ,
                                       int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk,
                                       int64_t headDimV, int64_t maskMode, std::string layoutQ, int64_t aicCoreNum,
                                       int64_t aivCoreNum, const at::Tensor &output)
{
    // Device inference from the seq-lens tensors (sparse_flash_mla precedent): cacheSeqlens 必传, cu/seq 可选.
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else {
        outputDevice = cacheSeqlens.device();
    }

    // Explicit capacity check (B2/D16): batchSize is derived from the required cacheSeqlens length
    // (batch = cacheSeqlens.size(0), same source as the Python side), so the kv==1 worst-case bound is computable.
    int64_t batchSize = cacheSeqlens.size(0);
    int64_t requiredBytes = CalculateMlaMetadataSizeBytes(batchSize, aicCoreNum, aivCoreNum);
    TORCH_CHECK(static_cast<int64_t>(output.nbytes()) >= requiredBytes,
                "The metadata tensor is too small: ", output.nbytes(), " bytes, required at least ", requiredBytes,
                " bytes for batch_size=", batchSize, " (capacity = ((aic ", aicCoreNum, " + aiv ", aivCoreNum,
                ") * batch_size * 1 + 1) * 16 int32 words, "
                "aligned to 4096 int32 elements; kv head num is 1 for MLA). Use the buffer returned by "
                "flash_mla_with_kvcache_metadata.");

    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);

    // ACLNN_CMD's ConvertTypes binds non-const lvalue refs -> hoist layout strings into char* lvalues
    // (direct .c_str()/.data() rvalues fail to compile; same pattern in kernel entry).
    char *layoutQPtr = const_cast<char *>(layoutQ.c_str());

    // Call order must match the schema/def input order (N5): cu_seqlens_q, cache_seqlens, seqused_q first,
    // then int attrs, then layout_q, then the output buffer.
    ACLNN_CMD(aclnnFlashMlaWithKvcacheMetadata, cuSeqlensQVal, cacheSeqlens, sequsedQVal, maxSeqlenQ, maxSeqlenKv,
              numHeadsQ, numHeadsKv, headDimQk, headDimV, maskMode, layoutQPtr, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> FlashMlaWithKvcache(
    const at::Tensor &q, const at::Tensor &kCache, const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &cacheSeqlens, const c10::optional<at::Tensor> &cuSeqlensQ,
    const c10::optional<at::Tensor> &sequsedQ, const c10::optional<at::Tensor> &attnMask,
    const c10::optional<at::Tensor> &metadata, int64_t headDimV, double softmaxScale, int64_t maskMode,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, std::string layoutQ, std::string layoutKv, std::string layoutOut,
    int64_t returnSoftmaxLse, int64_t aicCoreNum, int64_t aivCoreNum)
{
    int64_t tSize = 0;
    int64_t nSize = 0;
    int64_t bSize = 0;
    int64_t sSize = 0;
    at::SmallVector<int64_t, MAX_DIM_SIZE> attentionOutSize;
    at::SmallVector<int64_t, MAX_DIM_SIZE> softmaxOutSize;
    if (layoutQ == "TND") {
        tSize = q.size(0);
        nSize = q.size(1);
    } else if (layoutQ == "BSND") {
        bSize = q.size(0);
        sSize = q.size(1);
        nSize = q.size(2);
    } else {
        bSize = q.size(0);
        nSize = q.size(1);
        sSize = q.size(2);
    }
    // MLA: attn_out last dim = head_dim_v (nope/value width, 512). q/k_cache carry the merged rope in the
    // trailing 64 lanes (last dim 576 = head_dim_v 512 + rope 64), so the output D is NOT q's last dim.
    if (returnSoftmaxLse) {
        if (q.dim() == DIM_THREE) {
            softmaxOutSize = {nSize, tSize};
        } else {
            softmaxOutSize = {bSize, nSize, sSize};
        }
    } else {
        softmaxOutSize = {0};
    }

    at::Tensor attentionOutput{nullptr};
    at::Tensor softmaxLse{nullptr};
    {
        // DeviceGuard must be in scope before allocating the outputs (guidelines 3.1).
        auto localDevice = c10::Device(q.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        if (layoutOut == "TND") {
            attentionOutSize = {tSize, nSize, headDimV};
        } else if (layoutOut == "BNSD") {
            attentionOutSize = {bSize, nSize, sSize, headDimV};
        } else if (layoutOut == "NTD") {
            // TND→NTD 转置输出 (N, T, D)
            attentionOutSize = {nSize, tSize, headDimV};
        } else {
            // BSND（out == layout_q，D7）：输出 (B, S, N, D)
            attentionOutSize = {bSize, sSize, nSize, headDimV};
        }
        softmaxLse = at::empty(softmaxOutSize, q.options().dtype(at::kFloat));
        attentionOutput = at::empty(attentionOutSize, q.options().dtype(q.dtype()));
    }

    if (metadata.has_value() && metadata.value().defined()) {
        // Guard against an undersized metadata buffer supplied by the user (B2): any batch size derived from
        // this tensor can only shrink the requirement, so checking raw word count keeps the check conservative.
        int64_t words = metadata.value().numel();
        TORCH_CHECK(words >= aicCoreNum + aivCoreNum, "The metadata tensor is too small: ", words,
                    " words, expected the buffer produced by "
                    "flash_mla_with_kvcache_metadata (capacity = ((aic + aiv) * batch_size * 1 + 1) * 16 words, kv "
                    "head num == 1).");
    }

    char *layoutQPtr = const_cast<char *>(layoutQ.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKv.c_str());
    char *layoutOutPtr = const_cast<char *>(layoutOut.c_str());

    // Python bool -> C++ int64_t: pybind11 auto-converts return_softmax_lse (True->1, False->0) before here.
    ACLNN_CMD(aclnnFlashMlaWithKvcache, q, kCache, blockTable, cacheSeqlens, cuSeqlensQ, sequsedQ, attnMask, metadata,
              headDimV, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKv, layoutQPtr, layoutKvPtr, layoutOutPtr,
              returnSoftmaxLse, attentionOutput, softmaxLse);

    return std::tuple<at::Tensor, at::Tensor>(attentionOutput, softmaxLse);
}

// Bind the C++ function to the Python module (names must match the schema names exactly).
// torch 公开接口名 = flash_mla_with_kvcache；内部 aclnn/op 名 FlashMlaWithKvcache/flash_mla_with_kvcache。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_mla_with_kvcache", &FlashMlaWithKvcache, "flash_mla_with_kvcache");
    m.def("flash_mla_with_kvcache_metadata", &FlashMlaWithKvcacheMetadata, "flash_mla_with_kvcache_metadata");
}
} // namespace op_api
