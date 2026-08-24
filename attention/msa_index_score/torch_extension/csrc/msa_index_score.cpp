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
 * \file msa_index_score.cpp
 * \brief torch_extension 适配：aclnnMsaIndexScore C++ wrapper。
 *        Atlas A2/A3；key 支持 PA BBND/BNBD 与 TND packed。不支持 Ascend 950 / FP8。
 */

#include <string>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t QUERY_DIM = 3;
constexpr int64_t KEY_TND_DIM = 3;
constexpr int64_t KEY_PA_DIM = 4;
constexpr int64_t BLOCK_TABLE_DIM = 2;
constexpr int64_t SEQ_LEN_Q_DIM = 1;
constexpr int64_t SEQ_LEN_K_DIM = 1;
constexpr int64_t START_LOC_DIM = 1;
constexpr int64_t SCALE_PA_DIM = 3;
constexpr int64_t SCALE_TND_DIM = 2;
constexpr int64_t ATTEN_MASK_DIM = 2;
constexpr int64_t SCORE_STRIDE_ALIGN = 16;
constexpr int64_t SUPPORTED_BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM_ALIGN = 16;
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr int64_t ATTEN_MASK_SIZE = 2048;
constexpr int64_t SPARSE_MODE_DEFAULT = 0;
constexpr int64_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

struct MsaKeyMeta {
    int64_t numKvHeads = 1;
    int64_t keyHeadDim = 0;
    int64_t maxBlocks = 0;
};

std::string ParseLayoutKey(c10::string_view layout_key)
{
    std::string layoutKeyStr(layout_key);
    if (layoutKeyStr.empty()) {
        layoutKeyStr = "BBND";
    }
    TORCH_CHECK(layoutKeyStr == "TND" || layoutKeyStr == "BBND" || layoutKeyStr == "BNBD",
                "layout_key must be TND, BBND or BNBD, got ", layoutKeyStr);
    return layoutKeyStr;
}

void CheckMsaRequiredTensors(const at::Tensor &query, const at::Tensor &key, const at::Tensor &start_loc,
                             const c10::optional<at::Tensor> &actual_seq_qlen,
                             const c10::optional<at::Tensor> &actual_seq_klen)
{
    TORCH_CHECK(query.defined() && query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.defined() && key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(start_loc.defined() && start_loc.numel() > 0, "Tensor start_loc is empty.");
    TORCH_CHECK(actual_seq_qlen.has_value() && actual_seq_qlen.value().defined(),
                "TND query requires actual_seq_qlen.");
    TORCH_CHECK(actual_seq_klen.has_value() && actual_seq_klen.value().defined(), "actual_seq_klen is required.");
}

void CheckMsaLayoutRanks(const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &block_table,
                         const std::string &layoutKeyStr, bool is_tnd, const at::Tensor &cu_q, const at::Tensor &kv_len,
                         const at::Tensor &start_loc)
{
    TORCH_CHECK(query.dim() == QUERY_DIM, "query must be 3D [T, Hq, D], but got dim=", query.dim());
    if (is_tnd) {
        TORCH_CHECK(key.dim() == KEY_TND_DIM, "layout_key=TND requires key [T2,N2,D], but got dim=", key.dim());
        TORCH_CHECK(!block_table.has_value() || !block_table.value().defined(),
                    "layout_key=TND must not pass block_table.");
    } else {
        TORCH_CHECK(key.dim() == KEY_PA_DIM, "layout_key=", layoutKeyStr,
                    " requires key rank 4, but got dim=", key.dim());
        TORCH_CHECK(block_table.has_value() && block_table.value().defined() && block_table.value().numel() > 0,
                    "PageAttention (BBND/BNBD) requires block_table.");
    }
    TORCH_CHECK(cu_q.dim() == SEQ_LEN_Q_DIM, "actual_seq_qlen must be 1D [B+1], but got dim=", cu_q.dim());
    TORCH_CHECK(kv_len.dim() == SEQ_LEN_K_DIM, "actual_seq_klen must be 1D, but got dim=", kv_len.dim());
    TORCH_CHECK(start_loc.dim() == START_LOC_DIM, "start_loc must be 1D [B], but got dim=", start_loc.dim());
}

void CheckMsaSparseAndDtypes(const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &scale,
                             const c10::optional<at::Tensor> &atten_mask, const at::Tensor &cu_q,
                             const at::Tensor &kv_len, const at::Tensor &start_loc, int64_t sparse_mode, bool is_tnd)
{
    TORCH_CHECK(sparse_mode == SPARSE_MODE_DEFAULT || sparse_mode == SPARSE_MODE_RIGHT_DOWN,
                "sparse_mode must be 0 or 3, but got ", sparse_mode);
    if (sparse_mode == SPARSE_MODE_RIGHT_DOWN) {
        TORCH_CHECK(atten_mask.has_value() && atten_mask.value().defined(),
                    "sparse_mode=3 requires atten_mask of shape [2048, 2048].");
        TORCH_CHECK(atten_mask.value().dim() == ATTEN_MASK_DIM && atten_mask.value().size(0) == ATTEN_MASK_SIZE &&
                        atten_mask.value().size(1) == ATTEN_MASK_SIZE,
                    "atten_mask must be [2048, 2048].");
        TORCH_CHECK(atten_mask.value().scalar_type() == at::kChar, "atten_mask dtype must be int8.");
    } else {
        TORCH_CHECK(!atten_mask.has_value() || !atten_mask.value().defined(),
                    "sparse_mode=0 must not pass atten_mask.");
    }

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query dtype must be float16 or bfloat16, but got ", query.scalar_type());
    const bool is_quant = (key.scalar_type() == at::kChar);
    if (is_quant) {
        TORCH_CHECK(query.scalar_type() == at::kHalf, "int8 key currently requires float16 query.");
        TORCH_CHECK(scale.has_value() && scale.value().defined() && scale.value().numel() > 0,
                    "int8 key requires dequant scale.");
        TORCH_CHECK(scale.value().scalar_type() == at::kFloat, "scale dtype must be float32.");
        if (is_tnd) {
            TORCH_CHECK(scale.value().dim() == SCALE_TND_DIM || (scale.value().dim() == 1 && key.size(1) == 1),
                        "TND scale must be [T2, N2] or [T2].");
        } else {
            TORCH_CHECK(scale.value().dim() == SCALE_PA_DIM, "PA scale must be 3D [NP, N_kv, P].");
        }
    } else {
        TORCH_CHECK(key.scalar_type() == query.scalar_type(), "non-quant key dtype must match query.");
        TORCH_CHECK(!scale.has_value() || !scale.value().defined(), "non-quant path must not pass scale.");
    }
    TORCH_CHECK(cu_q.scalar_type() == at::kInt, "actual_seq_qlen dtype must be int32.");
    TORCH_CHECK(kv_len.scalar_type() == at::kInt, "actual_seq_klen dtype must be int32.");
    TORCH_CHECK(start_loc.scalar_type() == at::kInt, "start_loc dtype must be int32.");
}

MsaKeyMeta ResolveTndKeyMeta(const at::Tensor &key, const at::Tensor &cu_q, const at::Tensor &kv_len,
                             const at::Tensor &start_loc, int64_t init_blocks, int64_t local_blocks)
{
    MsaKeyMeta meta;
    meta.numKvHeads = key.size(1);
    meta.keyHeadDim = key.size(DIM_2);
    TORCH_CHECK(cu_q.size(0) == kv_len.size(0), "TND actual_seq_qlen/klen must both be [B+1].");
    TORCH_CHECK(start_loc.size(0) + 1 == kv_len.size(0), "start_loc size must equal batch.");
    TORCH_CHECK(init_blocks >= 0 && local_blocks >= 0, "init_blocks/local_blocks must be >= 0.");
    auto kv_host = kv_len.cpu();
    const int32_t *pref = kv_host.data_ptr<int32_t>();
    const int64_t npref = kv_host.size(0);
    for (int64_t i = 0; i + 1 < npref; ++i) {
        const int64_t kv = static_cast<int64_t>(pref[i + 1]) - static_cast<int64_t>(pref[i]);
        const int64_t blocks = (kv <= 0) ? 0 : ((kv + SUPPORTED_BLOCK_SIZE - 1) / SUPPORTED_BLOCK_SIZE);
        if (blocks > meta.maxBlocks) {
            meta.maxBlocks = blocks;
        }
    }
    TORCH_CHECK(meta.maxBlocks > 0, "TND maxBlocks must be positive.");
    TORCH_CHECK(init_blocks <= meta.maxBlocks && local_blocks <= meta.maxBlocks,
                "init_blocks/local_blocks must be <= maxBlocks.");
    return meta;
}

MsaKeyMeta ResolvePaKeyMeta(const at::Tensor &key, const at::Tensor &bt, const std::string &layoutKeyStr,
                            const at::Tensor &cu_q, const at::Tensor &kv_len, const at::Tensor &start_loc,
                            int64_t init_blocks, int64_t local_blocks)
{
    MsaKeyMeta meta;
    TORCH_CHECK(bt.dim() == BLOCK_TABLE_DIM, "block_table must be 2D [B, MB], but got dim=", bt.dim());
    TORCH_CHECK(bt.scalar_type() == at::kInt, "block_table dtype must be int32.");
    if (layoutKeyStr == "BBND") {
        TORCH_CHECK(key.size(1) == SUPPORTED_BLOCK_SIZE,
                    "layout_key=BBND requires key [NP,P,N2,D] with P=", SUPPORTED_BLOCK_SIZE);
        meta.numKvHeads = key.size(DIM_2);
    } else {
        TORCH_CHECK(key.size(DIM_2) == SUPPORTED_BLOCK_SIZE,
                    "layout_key=BNBD requires key [NP,N2,P,D] with P=", SUPPORTED_BLOCK_SIZE);
        meta.numKvHeads = key.size(1);
    }
    meta.keyHeadDim = key.size(DIM_3);
    TORCH_CHECK(cu_q.size(0) == kv_len.size(0) + 1, "actual_seq_qlen size must be batch+1.");
    TORCH_CHECK(start_loc.size(0) == kv_len.size(0), "start_loc size must equal batch.");
    TORCH_CHECK(bt.size(0) == kv_len.size(0), "block_table batch must equal batch.");
    TORCH_CHECK(init_blocks >= 0 && local_blocks >= 0, "init_blocks/local_blocks must be >= 0.");
    TORCH_CHECK(init_blocks <= bt.size(1) && local_blocks <= bt.size(1),
                "init_blocks/local_blocks must be <= maxBlocks.");
    meta.maxBlocks = bt.size(1);
    return meta;
}

void CheckHeadDim(const at::Tensor &query, int64_t keyHeadDim)
{
    TORCH_CHECK(query.size(DIM_2) == keyHeadDim, "query/key head_dim mismatch.");
    TORCH_CHECK(query.size(DIM_2) > 0 && (query.size(DIM_2) % HEAD_DIM_ALIGN) == 0, "head_dim align error.");
    TORCH_CHECK(query.size(DIM_2) <= MAX_HEAD_DIM, "head_dim too large.");
}

at::Tensor AllocMsaScore(const at::Tensor &query, int64_t maxBlocks)
{
    auto local_device = c10::Device(query.device());
    const c10::OptionalDeviceGuard device_guard(local_device);
    const int64_t total_q = query.size(0);
    const int64_t num_q_heads = query.size(1);
    const int64_t score_stride = ((maxBlocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
    return at::empty({num_q_heads, total_q, score_stride}, query.options().dtype(at::kFloat));
}
} // namespace

at::Tensor msa_index_score(const at::Tensor &query, const at::Tensor &key, const c10::optional<at::Tensor> &block_table,
                           const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &atten_mask,
                           const c10::optional<at::Tensor> &actual_seq_qlen,
                           const c10::optional<at::Tensor> &actual_seq_klen, const at::Tensor &start_loc,
                           c10::string_view layout_key, int64_t sparse_mode, int64_t init_blocks, int64_t local_blocks)
{
    CheckMsaRequiredTensors(query, key, start_loc, actual_seq_qlen, actual_seq_klen);
    const std::string layoutKeyStr = ParseLayoutKey(layout_key);
    const bool is_tnd = (layoutKeyStr == "TND");
    const at::Tensor &cu_q = actual_seq_qlen.value();
    const at::Tensor &kv_len = actual_seq_klen.value();
    CheckMsaLayoutRanks(query, key, block_table, layoutKeyStr, is_tnd, cu_q, kv_len, start_loc);
    CheckMsaSparseAndDtypes(query, key, scale, atten_mask, cu_q, kv_len, start_loc, sparse_mode, is_tnd);
    const MsaKeyMeta meta = is_tnd ? ResolveTndKeyMeta(key, cu_q, kv_len, start_loc, init_blocks, local_blocks) :
                                     ResolvePaKeyMeta(key, block_table.value(), layoutKeyStr, cu_q, kv_len, start_loc,
                                                      init_blocks, local_blocks);
    TORCH_CHECK(meta.numKvHeads == 1, "num_kv_heads currently only supports 1.");
    CheckHeadDim(query, meta.keyHeadDim);
    at::Tensor score = AllocMsaScore(query, meta.maxBlocks);
    ACLNN_CMD(aclnnMsaIndexScore, query, key, block_table, scale, atten_mask, actual_seq_qlen, actual_seq_klen,
              start_loc, layoutKeyStr, sparse_mode, init_blocks, local_blocks, score);
    return score;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("msa_index_score", &msa_index_score, "msa_index_score"); }
} // namespace op_api
