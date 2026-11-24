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
 * \file msa_index_score_v2.cpp
 * \brief torch_extension 适配：aclnnMsaIndexScoreV2 C++ wrapper。
 *        key 支持 PA BBND/BNBD 与 TND / BSND / BNSD。
 *        950 另支持 query/key 同型 HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN；
 *        PA key 允许 dim0 非连续（torch view stride 经 ConvertType 传入）。
 */

#include <cctype>
#include <string>
#include <tuple>
#include <torch/extension.h>
#include "acl/acl.h"
#include "aclnn_common.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

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
constexpr int64_t BLOCK16_SIZE = 16; // BSND/BNSD，以及 BBND/BNBD 页长 16。TND 固定 128
constexpr int64_t BLOCK16_TOPK = 128;
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

bool IsAscend950Device()
{
    const char* soc = aclrtGetSocName();
    if (soc == nullptr) {
        return false;
    }
    const std::string name(soc);
    return name.find("950") != std::string::npos || name.find("910_95") != std::string::npos;
}

std::string NormalizeLayoutName(c10::string_view layout, const char* fallback)
{
    std::string name(layout.empty() ? fallback : layout);
    for (char& ch : name) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return name;
}

std::string ParseLayoutKey(c10::string_view layout_key)
{
    std::string layoutKeyStr = NormalizeLayoutName(layout_key, "BBND");
    TORCH_CHECK(layoutKeyStr == "TND" || layoutKeyStr == "BSND" || layoutKeyStr == "BNSD" || layoutKeyStr == "BBND" ||
                    layoutKeyStr == "BNBD",
                "layout_key must be TND, BSND, BNSD, BBND or BNBD, got ", layoutKeyStr);
    return layoutKeyStr;
}

void CheckMsaRequiredTensors(const at::Tensor& query, const at::Tensor& key, const at::Tensor& start_loc,
                             const c10::optional<at::Tensor>& actual_seq_qlen,
                             const c10::optional<at::Tensor>& actual_seq_klen)
{
    TORCH_CHECK(query.defined(), "Tensor query is not defined.");
    TORCH_CHECK(key.defined(), "Tensor key is not defined.");
    TORCH_CHECK(start_loc.defined() && start_loc.numel() > 0, "Tensor start_loc is empty.");
    TORCH_CHECK(actual_seq_qlen.has_value() && actual_seq_qlen.value().defined(),
                "TND query requires actual_seq_qlen.");
    TORCH_CHECK(actual_seq_klen.has_value() && actual_seq_klen.value().defined(), "actual_seq_klen is required.");
}

void CheckMsaLayoutRanks(const at::Tensor& query, const at::Tensor& key, const c10::optional<at::Tensor>& block_table,
                         const std::string& layoutKeyStr, const std::string& layoutQueryStr, const at::Tensor& cu_q,
                         const at::Tensor& kv_len, const at::Tensor& start_loc)
{
    const bool seq_key = (layoutKeyStr == "TND" || layoutKeyStr == "BSND" || layoutKeyStr == "BNSD");
    if (layoutQueryStr == "TND") {
        TORCH_CHECK(query.dim() == QUERY_DIM, "layout_query=TND requires query [T,N,D], got dim=", query.dim());
    } else {
        TORCH_CHECK(layoutQueryStr == "BSND" || layoutQueryStr == "BNSD",
                    "layout_query must be TND, BSND or BNSD, got ", layoutQueryStr);
        TORCH_CHECK(query.dim() == KEY_PA_DIM, "layout_query=BSND/BNSD requires query rank 4, got dim=", query.dim());
    }
    if (seq_key) {
        const int64_t expect = (layoutKeyStr == "TND") ? KEY_TND_DIM : KEY_PA_DIM;
        TORCH_CHECK(key.dim() == expect, "non-paged key rank mismatch, got dim=", key.dim());
        TORCH_CHECK(!block_table.has_value() || !block_table.value().defined(),
                    "non-paged key must not pass block_table.");
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

void CheckMsaSparseAndDtypes(const at::Tensor& query, const at::Tensor& key, const c10::optional<at::Tensor>& key_scale,
                             const c10::optional<at::Tensor>& atten_mask, const at::Tensor& cu_q,
                             const at::Tensor& kv_len, const at::Tensor& start_loc, int64_t sparse_mode, bool is_tnd)
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

    auto is_hifloat8 = [](at::ScalarType t) {
        const char* name = c10::toString(t);
        return name != nullptr && (std::string(name) == "hifloat8" || std::string(name) == "HiFloat8");
    };
    const bool is_mxfp4 = query.scalar_type() == at::kFloat4_e2m1fn_x2;
    const bool q_ok = query.scalar_type() == at::kChar || query.scalar_type() == at::kHalf ||
                      query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat8_e5m2 ||
                      query.scalar_type() == at::kFloat8_e4m3fn || is_hifloat8(query.scalar_type()) || is_mxfp4;
    TORCH_CHECK(q_ok,
                "query dtype must be int8 / float16 / bfloat16 / hifloat8 / float8_e5m2 / float8_e4m3fn / "
                "float4_e2m1fn_x2, but got ",
                query.scalar_type());
    const bool is_fp8 = query.scalar_type() == at::kFloat8_e5m2 || query.scalar_type() == at::kFloat8_e4m3fn ||
                        is_hifloat8(query.scalar_type());
    const bool is_quant = (key.scalar_type() == at::kChar);
    if (is_mxfp4) {
        TORCH_CHECK(key.scalar_type() == at::kFloat4_e2m1fn_x2, "MXFP4 key dtype must be float4_e2m1fn_x2.");
        TORCH_CHECK(key_scale.has_value() && key_scale.value().defined() && key_scale.value().numel() > 0,
                    "MXFP4 requires key key_scale.");
        TORCH_CHECK(key_scale.value().scalar_type() == at::kFloat8_e8m0fnu,
                    "MXFP4 key key_scale dtype must be float8_e8m0fnu, got ", key_scale.value().scalar_type());
    } else if (is_quant) {
        TORCH_CHECK(query.scalar_type() == at::kChar, "int8 key requires native int8 query; float16/int8 is reserved.");
        TORCH_CHECK(key_scale.has_value() && key_scale.value().defined(), "int8 key requires dequant key_scale.");
        TORCH_CHECK(key_scale.value().scalar_type() == at::kFloat, "key_scale dtype must be float32.");
        if (is_tnd) {
            TORCH_CHECK(key_scale.value().dim() == SCALE_TND_DIM || (key_scale.value().dim() == 1 && key.size(1) == 1),
                        "TND key_scale must be [T2, N2] or [T2].");
        } else {
            TORCH_CHECK(key_scale.value().dim() == SCALE_PA_DIM, "PA key_scale must be 3D [NP, N_kv, P].");
        }
    } else {
        TORCH_CHECK(key.scalar_type() == query.scalar_type(), "non-quant key dtype must match query.");
        TORCH_CHECK(!key_scale.has_value() || !key_scale.value().defined(), "non-quant path must not pass key_scale.");
        TORCH_CHECK(!is_fp8 || (!key_scale.has_value() || !key_scale.value().defined()),
                    "FP8 path must not pass key_scale.");
    }
    TORCH_CHECK(cu_q.scalar_type() == at::kInt, "actual_seq_qlen dtype must be int32.");
    TORCH_CHECK(kv_len.scalar_type() == at::kInt, "actual_seq_klen dtype must be int32.");
    TORCH_CHECK(start_loc.scalar_type() == at::kInt, "start_loc dtype must be int32.");
}

MsaKeyMeta ResolveTndKeyMeta(const at::Tensor& key, const at::Tensor& cu_q, const at::Tensor& kv_len,
                             const at::Tensor& start_loc, int64_t init_blocks, int64_t local_blocks,
                             int64_t logicalBlock, int64_t seqCap, int64_t compressFactor)
{
    MsaKeyMeta meta;
    meta.numKvHeads = key.size(1);
    meta.keyHeadDim = key.size(DIM_2);
    TORCH_CHECK(cu_q.size(0) == kv_len.size(0), "TND actual_seq_qlen/klen must both be [B+1].");
    TORCH_CHECK(start_loc.size(0) + 1 == kv_len.size(0), "start_loc size must equal batch.");
    TORCH_CHECK(init_blocks >= 0 && local_blocks >= 0, "init_blocks/local_blocks must be >= 0.");
    auto kv_host = kv_len.cpu();
    const int32_t* pref = kv_host.data_ptr<int32_t>();
    const int64_t npref = kv_host.size(0);
    for (int64_t i = 0; i + 1 < npref; ++i) {
        const int64_t kv = static_cast<int64_t>(pref[i + 1]) - static_cast<int64_t>(pref[i]);
        TORCH_CHECK(kv >= 0, "actual_seq_klen must be non-decreasing.");
        // S 是压缩行槽。actual_seq_klen 是原始 token 数。
        const int64_t rows = (compressFactor <= 1 || kv <= 0) ? kv : (kv + compressFactor - 1) / compressFactor;
        TORCH_CHECK(seqCap <= 0 || rows <= seqCap, "compressed key rows ", rows, " exceed layout S=", seqCap);
        const int64_t blocks = (kv <= 0) ? 0 : ((kv + logicalBlock - 1) / logicalBlock);
        if (blocks > meta.maxBlocks) {
            meta.maxBlocks = blocks;
        }
    }
    // 全 kv_len=0：按 1 个 dummy block 分配 score 末维 16，与 InferShape 一致。
    if (meta.maxBlocks <= 0) {
        meta.maxBlocks = 1;
    }
    // host 用 16 对齐后的扫描宽度做上界。M-D-8 只有 8 个真实块，local_blocks=9 仍合法。
    const int64_t blockLimit = ((meta.maxBlocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
    TORCH_CHECK(init_blocks <= blockLimit && local_blocks <= blockLimit,
                "init_blocks/local_blocks must be <= aligned block limit ", blockLimit, ".");
    return meta;
}

MsaKeyMeta ResolvePaKeyMeta(const at::Tensor& key, const at::Tensor& bt, const std::string& layoutKeyStr,
                            const at::Tensor& cu_q, const at::Tensor& kv_len, const at::Tensor& start_loc,
                            int64_t init_blocks, int64_t local_blocks, int64_t rowsPerBlock, int64_t poolBlockSize,
                            int64_t compressFactor)
{
    MsaKeyMeta meta;
    TORCH_CHECK(bt.dim() == BLOCK_TABLE_DIM, "block_table must be 2D [B, MB], but got dim=", bt.dim());
    TORCH_CHECK(bt.scalar_type() == at::kInt, "block_table dtype must be int32.");
    const int64_t physicalPage = key.size(layoutKeyStr == "BBND" ? 1 : DIM_2);
    const bool compressedPage =
        poolBlockSize == BLOCK16_SIZE && compressFactor > 1 && physicalPage == SUPPORTED_BLOCK_SIZE / compressFactor;
    if (layoutKeyStr == "BBND") {
        TORCH_CHECK(physicalPage == SUPPORTED_BLOCK_SIZE || physicalPage == BLOCK16_SIZE || compressedPage,
                    "BBND P must be128,16,or128/compress_factor for pool16, got P=", physicalPage);
        meta.numKvHeads = key.size(DIM_2);
        meta.keyHeadDim = key.size(DIM_3);
    } else {
        TORCH_CHECK(physicalPage == SUPPORTED_BLOCK_SIZE || physicalPage == BLOCK16_SIZE || compressedPage,
                    "BNBD P must be128,16,or128/compress_factor for pool16, got P=", physicalPage);
        meta.numKvHeads = key.size(1);
        meta.keyHeadDim = key.size(DIM_3);
    }
    TORCH_CHECK(cu_q.size(0) == kv_len.size(0) + 1, "actual_seq_qlen size must be batch+1.");
    TORCH_CHECK(start_loc.size(0) == kv_len.size(0), "start_loc size must equal batch.");
    TORCH_CHECK(bt.size(0) == kv_len.size(0), "block_table batch must equal batch.");
    TORCH_CHECK(init_blocks >= 0 && local_blocks >= 0, "init_blocks/local_blocks must be >= 0.");
    // 页表宽度是物理页数。页长按 key 行计，score 末维 = 页数 × (页长 / rowsPerBlock)，与 InferShape 一致。
    const int64_t pageLen = (layoutKeyStr == "BBND") ? key.size(1) : key.size(DIM_2);
    TORCH_CHECK(rowsPerBlock > 0 && pageLen % rowsPerBlock == 0, "BBND/BNBD page size(", pageLen,
                ") must be a positive multiple of pool_block_size/compress_factor(", rowsPerBlock, ")");
    const int64_t logicalBlocks = bt.size(1) * (pageLen / rowsPerBlock);
    meta.maxBlocks = logicalBlocks > 0 ? logicalBlocks : 1;
    const int64_t blockLimit = ((meta.maxBlocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
    TORCH_CHECK(init_blocks <= blockLimit && local_blocks <= blockLimit,
                "init_blocks/local_blocks must be <= aligned block limit ", blockLimit, ".");
    return meta;
}

void CheckHeadDim(const at::Tensor& query, int64_t keyHeadDim)
{
    TORCH_CHECK(query.size(-1) == keyHeadDim, "query/key head_dim mismatch.");
    TORCH_CHECK(query.size(-1) > 0 && (query.size(-1) % HEAD_DIM_ALIGN) == 0, "head_dim align error.");
    TORCH_CHECK(query.size(-1) <= MAX_HEAD_DIM, "head_dim too large.");
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> msa_index_score_v2(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& start_loc,
    const c10::optional<at::Tensor>& block_table, const c10::optional<at::Tensor>& query_scale,
    const c10::optional<at::Tensor>& key_scale, const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& actual_seq_qlen, const c10::optional<at::Tensor>& actual_seq_klen,
    c10::string_view layout_key, int64_t sparse_mode, int64_t init_blocks, int64_t local_blocks,
    c10::string_view layout_query, int64_t pool_block_size, int64_t return_choice, int64_t topk_count,
    int64_t compress_factor)
{
    CheckMsaRequiredTensors(query, key, start_loc, actual_seq_qlen, actual_seq_klen);
    if (query.scalar_type() == at::kChar) {
        TORCH_CHECK(query.device().type() == torch_npu::utils::get_npu_device_type(), "query must be an NPU tensor");
        TORCH_CHECK(key.device() == query.device(), "key must be on query's device");
        TORCH_CHECK(start_loc.device() == query.device(), "start_loc must be on query's device");
        const auto check_device = [&query](const c10::optional<at::Tensor>& tensor, const char* name) {
            TORCH_CHECK(!tensor.has_value() || !tensor->defined() || tensor->device() == query.device(), name,
                        " must be on query's device");
        };
        check_device(block_table, "block_table");
        check_device(query_scale, "query_scale");
        check_device(key_scale, "key_scale");
        check_device(atten_mask, "atten_mask");
        TORCH_CHECK(actual_seq_qlen->device().is_cpu() || actual_seq_qlen->device() == query.device(),
                    "actual_seq_qlen must be on CPU or query's device");
        TORCH_CHECK(actual_seq_klen->device().is_cpu() || actual_seq_klen->device() == query.device(),
                    "actual_seq_klen must be on CPU or query's device");
    }
    const std::string layoutKeyStr = ParseLayoutKey(layout_key);
    const std::string layoutQueryStr = NormalizeLayoutName(layout_query, "TND");
    TORCH_CHECK(layoutQueryStr == "TND" || layoutQueryStr == "BSND" || layoutQueryStr == "BNSD",
                "layout_query must be TND, BSND or BNSD, got ", layoutQueryStr);
    const bool seq_key = (layoutKeyStr == "TND" || layoutKeyStr == "BSND" || layoutKeyStr == "BNSD");
    const bool batchKey128 = layoutQueryStr == "TND" && pool_block_size == SUPPORTED_BLOCK_SIZE &&
                             (layoutKeyStr == "BSND" || layoutKeyStr == "BNSD");
    if (seq_key && !batchKey128) {
        TORCH_CHECK(layoutKeyStr == layoutQueryStr,
                    "non-paged key layout must match layout_query, got query=", layoutQueryStr, " key=", layoutKeyStr);
    }
    const at::Tensor& cu_q = actual_seq_qlen.value();
    const at::Tensor& kv_len = actual_seq_klen.value();
    CheckMsaLayoutRanks(query, key, block_table, layoutKeyStr, layoutQueryStr, cu_q, kv_len, start_loc);
    CheckMsaSparseAndDtypes(query, key, key_scale, atten_mask, cu_q, kv_len, start_loc, sparse_mode, seq_key);
    const bool is_mxfp4 = query.scalar_type() == at::kFloat4_e2m1fn_x2;
    if (is_mxfp4) {
        TORCH_CHECK(query_scale.has_value() && query_scale.value().defined() && query_scale.value().numel() > 0,
                    "MXFP4 requires query_scale.");
        TORCH_CHECK(query_scale.value().scalar_type() == at::kFloat8_e8m0fnu,
                    "MXFP4 query_scale dtype must be float8_e8m0fnu, got ", query_scale.value().scalar_type());
    } else if (query.scalar_type() != at::kChar) {
        TORCH_CHECK(!query_scale.has_value() || !query_scale.value().defined(),
                    "query_scale is only supported for MXFP4 and int8 query.");
    }
    // pool_block_size 只允许 16 / 128；BSND/BNSD 固定 16。存储页长（key 的页维度）与打分块长解耦，
    // 但打分块不能超过存储页容量（kernel 按页表取每块前 bs 个 token）。
    const int64_t keyPage = seq_key ? 0 : (layoutKeyStr == "BBND") ? key.size(1) : key.size(DIM_2);
    const int64_t poolBlockSize = pool_block_size;
    TORCH_CHECK(poolBlockSize == SUPPORTED_BLOCK_SIZE || poolBlockSize == BLOCK16_SIZE,
                "pool_block_size must be 16 or 128, got ", poolBlockSize);
    if (layoutKeyStr == "BSND" || layoutKeyStr == "BNSD") {
        TORCH_CHECK(poolBlockSize == BLOCK16_SIZE || batchKey128,
                    "BSND/BNSD requires block16, or block128 with TND query, got ", poolBlockSize);
    } else if (!seq_key) {
        TORCH_CHECK(poolBlockSize <= keyPage * compress_factor, "pool_block_size(", poolBlockSize,
                    ") must not exceed BBND/BNBD page size(", keyPage, ") times compress_factor(", compress_factor,
                    ")");
    }
    // compress_factor：key 已压缩，一个打分块落到 key 上只占 poolBlockSize/compress_factor 行。
    TORCH_CHECK(compress_factor >= 1 && compress_factor <= poolBlockSize && poolBlockSize % compress_factor == 0,
                "compress_factor(", compress_factor, ") must be in [1, pool_block_size(", poolBlockSize,
                ")] and divide it");
    const int64_t rowsPerBlock = poolBlockSize / compress_factor;
    // actual_seq_klen 是原始 token 数，score 列按 pool_block_size 切。
    const int64_t keySeqCap = (layoutKeyStr == "BSND") ? key.size(1) : (layoutKeyStr == "BNSD") ? key.size(2) : 0;
    MsaKeyMeta meta = seq_key ?
                          ResolveTndKeyMeta(key, cu_q, kv_len, start_loc, init_blocks, local_blocks, poolBlockSize,
                                            keySeqCap, compress_factor) :
                          ResolvePaKeyMeta(key, block_table.value(), layoutKeyStr, cu_q, kv_len, start_loc, init_blocks,
                                           local_blocks, rowsPerBlock, poolBlockSize, compress_factor);
    if (batchKey128 && (is_mxfp4 || !kv_len.device().is_cpu())) {
        meta.maxBlocks = std::max<int64_t>(1, (keySeqCap + poolBlockSize - 1) / poolBlockSize);
    }
    if (layoutKeyStr == "BSND") {
        meta.numKvHeads = key.size(2);
        meta.keyHeadDim = key.size(3);
    } else if (layoutKeyStr == "BNSD") {
        meta.numKvHeads = key.size(1);
        meta.keyHeadDim = key.size(3);
    }
    TORCH_CHECK(meta.numKvHeads == 1, "num_kv_heads currently only supports 1.");
    CheckHeadDim(query, meta.keyHeadDim);
    int64_t total_q = query.size(0);
    int64_t num_q_heads = query.size(1);
    if (layoutQueryStr == "BSND" || layoutQueryStr == "BNSD") {
        const int64_t seqCap = (layoutQueryStr == "BSND") ? query.size(1) : query.size(2);
        num_q_heads = (layoutQueryStr == "BSND") ? query.size(2) : query.size(1);
        auto q_host = cu_q.to(at::kCPU);
        const int32_t* qPref = q_host.data_ptr<int32_t>();
        TORCH_CHECK(qPref != nullptr && q_host.size(0) >= 2, "actual_seq_qlen must be [B+1].");
        const int64_t nPref = q_host.size(0);
        for (int64_t i = 0; i + 1 < nPref; ++i) {
            const int64_t qlenB = static_cast<int64_t>(qPref[i + 1]) - static_cast<int64_t>(qPref[i]);
            TORCH_CHECK(qlenB >= 0 && qlenB <= seqCap, "query length ", qlenB, " exceeds layout S=", seqCap);
        }
        total_q = static_cast<int64_t>(qPref[nPref - 1]);
    }
    const bool m3Layout =
        layoutQueryStr == "BSND" || layoutQueryStr == "BNSD" || layoutKeyStr == "BSND" || layoutKeyStr == "BNSD";
    if (m3Layout && poolBlockSize == BLOCK16_SIZE) {
        TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 ||
                        query.scalar_type() == at::kFloat8_e4m3fn || query.scalar_type() == at::kFloat8_e5m2 ||
                        query.scalar_type() == at::kFloat4_e2m1fn_x2,
                    "BSND/BNSD supports float16 / bfloat16 / float8_e4m3fn / float8_e5m2 / MXFP4.");
        TORCH_CHECK(key.scalar_type() == query.scalar_type(), "BSND/BNSD key dtype must match query.");
    }
    // 块长 16（TND 16、BSND/BNSD、页长 16）做 head 间 max，N1 必须整除 128。
    if (poolBlockSize == BLOCK16_SIZE) {
        TORCH_CHECK(num_q_heads > 0 && (128 % num_q_heads) == 0,
                    "pool_block_size 16 requires num_q_heads to divide 128, got ", num_q_heads);
    }
    const int64_t score_stride = ((meta.maxBlocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
    const int64_t score_heads = (poolBlockSize == BLOCK16_SIZE) ? 1 : num_q_heads;
    TORCH_CHECK(return_choice >= 0 && return_choice <= 2, "return_choice must be 0, 1 or 2, got ", return_choice);
    const bool topk = return_choice != 0;
    TORCH_CHECK(!topk || (topk_count >= 1 && topk_count <= 128 && (poolBlockSize != BLOCK16_SIZE || topk_count == 128)),
                "Invalid topk");
    // 块长 16 且 return_choice=0 时用户可见 score 是 FLOAT16。块长 128 仍是 FLOAT。
    const auto scoreDtype = (!topk && poolBlockSize == BLOCK16_SIZE) ? at::kHalf : at::kFloat;
    at::Tensor score = topk ? at::empty({0}, query.options().dtype(at::kFloat)) :
                              at::empty({score_heads, total_q, score_stride}, query.options().dtype(scoreDtype));
    at::Tensor sparse_indices = topk ? at::empty({score_heads, total_q, topk_count}, query.options().dtype(at::kInt)) :
                                       at::empty({0}, query.options().dtype(at::kInt));
    at::Tensor sparse_values = (return_choice == 2) ?
                                   at::empty({score_heads, total_q, topk_count}, query.options().dtype(at::kFloat)) :
                                   at::empty({0}, query.options().dtype(at::kFloat));
    // The ACLNN handler may be queued without owning input tensors. Keep host-length copies alive until completion.
    const bool host_lengths = actual_seq_qlen->device().is_cpu() || actual_seq_klen->device().is_cpu();
    if (host_lengths) {
        const c10::OptionalDeviceGuard length_device_guard(query.device());
        TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None,
                    "CPU sequence lengths are not supported during NPU graph capture");
        (void)c10_npu::getCurrentNPUStream(query.device().index()).query();
    }
    c10::optional<at::Tensor> device_qlen = actual_seq_qlen.value().to(query.device());
    c10::optional<at::Tensor> device_klen = actual_seq_klen.value().to(query.device());
    ACLNN_CMD(aclnnMsaIndexScoreV2, query, key, start_loc, block_table, query_scale, key_scale, atten_mask, device_qlen,
              device_klen, layoutKeyStr, sparse_mode, init_blocks, local_blocks, layoutQueryStr, poolBlockSize,
              return_choice, topk_count, compress_factor, score, sparse_indices, sparse_values);
    if (host_lengths) {
        c10_npu::getCurrentNPUStream(query.device().index()).synchronize();
    }
    return {score, sparse_indices, sparse_values};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("msa_index_score_v2", &msa_index_score_v2, "MsaIndexScore full def interface", py::arg("query"),
          py::arg("key"), py::arg("start_loc"), py::arg("block_table"), py::arg("query_scale"), py::arg("key_scale"),
          py::arg("atten_mask"), py::arg("actual_seq_qlen"), py::arg("actual_seq_klen"), py::arg("layout_key"),
          py::arg("sparse_mode"), py::arg("init_blocks"), py::arg("local_blocks"), py::arg("layout_query"),
          py::arg("pool_block_size") = 128, py::arg("return_choice") = 0, py::arg("topk") = 16,
          py::arg("compress_factor") = 1);
}
} // namespace op_api
