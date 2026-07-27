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
#include <iostream>
#include <string>
#include <vector>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class QuantBlockSparseAttnTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBlockSparseAttnTilingArch35 TearDown" << std::endl;
    }
};

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

struct QuantBlockSparseAttnCompileInfo {};

const TensorDesc EmptyInput({{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND});

// B=1, key blockNum=4, N2=4, blockSize=128, D=128, max_seqlen_kv=256
// T1=256, N1=4, D=128, max_seqlen_q=256 (from query shape)
// qbMax = ceil(256/128) = 2, kbMax = ceil(256/128) = 2
// sparseCount (sparse_indices dim3) = 4
std::vector<TensorDesc> MakeValidInputs(int64_t t = 256, int64_t n1 = 4, int64_t n2 = 4)
{
    int64_t blockNum = 4;
    return {
        {{{t, n1, 128}, {t, n1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query TND
        {{{blockNum, n2, 128, 128}, {blockNum, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // key
        {{{blockNum, n2, 128, 128}, {blockNum, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value
        {{{t, n1}, {t, n1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                           // q_descale TND
        {{{blockNum, n2, 128}, {blockNum, n2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // k_descale
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                 // v_descale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                   // p_scale
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                     // cu_seqlens_q (B+1=2)
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                     // cu_seqlens_kv (B+1=2)
        EmptyInput,                                                    // seqused_q (optional)
        EmptyInput,                                                    // seqused_kv (optional)
        {{{1, n1, 2, 4}, {1, n1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
        {{{1, n1, 2}, {1, n1, 2}}, ge::DT_INT32, ge::FORMAT_ND},       // sparse_seq_len
        {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},               // block_table
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},   // atten_mask
        EmptyInput,                                                    // metadata (optional)
    };
}

std::vector<TensorDesc> MakeValidOutputs(int64_t t = 256, int64_t n1 = 4)
{
    return {
        {{{t, n1, 128}, {t, n1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{n1, t}, {n1, t}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeValidAttrs(int64_t qBlockSize = 128, int64_t kvBlockSize = 128, int64_t maskMode = 3,
                                   int64_t maxSeqlenKv = 256)
{
    return {
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSeqlenKv)},
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(qBlockSize)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvBlockSize)},
        {"paBlockStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>("B_N_Qb_Kb")},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
    };
}

std::vector<OpAttr> MakeAttrsEx(const std::string &layoutQ, int64_t paBlockStride, int64_t maskMode = 3,
                                int64_t maxSeqlenKv = 256, bool returnLse = false, int64_t qBlockSize = 128,
                                int64_t kvBlockSize = 128)
{
    return {
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxSeqlenKv)},
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(qBlockSize)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvBlockSize)},
        {"paBlockStride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(paBlockStride)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BNSD")},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQ)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>("B_N_Qb_Kb")},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maskMode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)},
    };
}

gert::TilingContextPara BuildTilingPara(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                                        const std::vector<OpAttr> &attrs)
{
    static QuantBlockSparseAttnCompileInfo compileInfo;
    return gert::TilingContextPara("QuantBlockSparseAttn", inputs, outputs, attrs, &compileInfo, "Ascend950", 64,
                                   262144, 65536);
}

void ExpectTilingResult(const gert::TilingContextPara &para, bool expectSuccess)
{
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(para, tilingInfo);
    EXPECT_EQ(ok, expectSuccess);
    if (expectSuccess) {
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    }
}

std::vector<TensorDesc> MakeNTDInputs(int64_t n1 = 4, int64_t t = 256, int64_t n2 = 4)
{
    int64_t blockNum = 4;
    return {
        {{{n1, t, 128}, {n1, t, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                         // query NTD
        {{{blockNum, n2, 128, 128}, {blockNum, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // key
        {{{blockNum, n2, 128, 128}, {blockNum, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value
        {{{n1, t}, {n1, t}}, ge::DT_FLOAT, ge::FORMAT_ND},                                           // q_descale NTD
        {{{blockNum, n2, 128}, {blockNum, n2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // k_descale
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                 // v_descale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                   // p_scale
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                                                   // cu_seqlens_q
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                                                   // cu_seqlens_kv
        EmptyInput,                                                                                  // seqused_q
        EmptyInput,                                                                                  // seqused_kv
        {{{1, n1, 2, 4}, {1, n1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},                               // sparse_indices
        {{{1, n1, 2}, {1, n1, 2}}, ge::DT_INT32, ge::FORMAT_ND},                                     // sparse_seq_len
        {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},                                             // block_table
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask
        EmptyInput,                                                                                  // metadata
    };
}

std::vector<TensorDesc> MakeBSNDInputs(int64_t b = 4, int64_t s = 256, int64_t n1 = 4, int64_t n2 = 4)
{
    int64_t blockNum = 4;
    int64_t qb = (s + 127) / 128;
    return {
        {{{b, s, n1, 128}, {b, s, n1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},                   // query BSND (4D)
        {{{blockNum, n2, 128, 128}, {blockNum, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // key
        {{{blockNum, n2, 128, 128}, {blockNum, n2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value
        {{{n1}, {n1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                 // q_descale
        {{{blockNum, n2, 128}, {blockNum, n2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // k_descale
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                 // v_descale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                                   // p_scale
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                                                   // cu_seqlens_q
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                                                   // cu_seqlens_kv
        EmptyInput,                                                                                  // seqused_q
        EmptyInput,                                                                                  // seqused_kv
        {{{b, n1, qb, 4}, {b, n1, qb, 4}}, ge::DT_INT32, ge::FORMAT_ND},                             // sparse_indices
        {{{b, n1, qb}, {b, n1, qb}}, ge::DT_INT32, ge::FORMAT_ND},                                   // sparse_seq_len
        {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},                                             // block_table
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},                                 // atten_mask
        EmptyInput,                                                                                  // metadata
    };
}

// 1D combined KV storage: key/value 为 1D storage 切片，k_descale 为 1D，v_descale 为 (N2,)
// pa_block_stride = N2 * blockSize * D + N2 * blockSize * Dv + N2 * blockSize * 4
// 当 N2=4, blockSize=128, D=Dv=128 时: 4*128*128 + 4*128*128 + 4*128*4 = 65536 + 65536 + 2048 = 131072
// keyStorageSize = blockNum * pa_block_stride = 4 * 131072 = 524288
std::vector<TensorDesc> Make1DKVInputs(int64_t n1 = 4, int64_t n2 = 4, int64_t keyStorageSize = 524288)
{
    return {
        {{{256, n1, 128}, {256, n1, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},     // query TND
        {{{keyStorageSize}, {keyStorageSize}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // key 1D
        {{{keyStorageSize}, {keyStorageSize}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, // value 1D
        {{{256, n1}, {256, n1}}, ge::DT_FLOAT, ge::FORMAT_ND},                       // q_descale TND
        {{{keyStorageSize}, {keyStorageSize}}, ge::DT_FLOAT, ge::FORMAT_ND},         // k_descale 1D
        {{{n2}, {n2}}, ge::DT_FLOAT, ge::FORMAT_ND},                                 // v_descale
        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                   // p_scale
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cu_seqlens_q
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                                   // cu_seqlens_kv
        EmptyInput,                                                                  // seqused_q
        EmptyInput,                                                                  // seqused_kv
        {{{1, n1, 2, 4}, {1, n1, 2, 4}}, ge::DT_INT32, ge::FORMAT_ND},               // sparse_indices
        {{{1, n1, 2}, {1, n1, 2}}, ge::DT_INT32, ge::FORMAT_ND},                     // sparse_seq_len
        {{{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND},                             // block_table
        {{{2048, 2048}, {2048, 2048}}, ge::DT_UINT8, ge::FORMAT_ND},                 // atten_mask
        EmptyInput,                                                                  // metadata
    };
}
} // namespace

// ===================== 正向用例：基础功能 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_basic)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_gqa)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(256, 8, 2), MakeValidOutputs(256, 8), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_mask_mode_0)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(128, 128, 0)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_mask_mode_0_without_atten_mask)
{
    auto inputs = MakeValidInputs();
    inputs[14] = EmptyInput;

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs(128, 128, 0)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_ntd_layout)
{
    ExpectTilingResult(BuildTilingPara(MakeNTDInputs(), MakeValidOutputs(), MakeAttrsEx("NTD", 0)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_with_block_table)
{
    auto inputs = MakeValidInputs();
    inputs[13] = TensorDesc({{1, 8}, {1, 8}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_return_softmax_lse)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeAttrsEx("TND", 0, 3, 256, true)),
                       true);
}

// ===================== 负向用例：属性与基础校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_mask_mode_3_without_atten_mask)
{
    auto inputs = MakeValidInputs();
    inputs[14] = EmptyInput;

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs(128, 128, 3)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_query_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_block_size)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(64, 128, 3)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_mask_mode)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(128, 128, 5)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_mask_mode_1)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(128, 128, 1)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_bsnd_layout)
{
    ExpectTilingResult(BuildTilingPara(MakeBSNDInputs(), MakeValidOutputs(), MakeAttrsEx("BSND", 0)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_kv_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_hifloat8_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{256, 4, 128}, {256, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND);
    inputs[1] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_query_layout_mismatch)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeAttrsEx("BSND", 0)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_key_3d)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 128, 128}, {4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 128, 128}, {4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_sparse_indices_3d)
{
    auto inputs = MakeValidInputs();
    inputs[11] = TensorDesc({{1, 4, 4}, {1, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND);
    inputs[12] = TensorDesc({{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_atten_mask_3d)
{
    auto inputs = MakeValidInputs();
    inputs[14] = TensorDesc({{256, 256, 1}, {256, 256, 1}}, ge::DT_UINT8, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_dsize)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{256, 4, 64}, {256, 4, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_n1_n2_mismatch_4d)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 3, 128, 128}, {4, 3, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 3, 128, 128}, {4, 3, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_block_table_b)
{
    auto inputs = MakeValidInputs();
    inputs[13] = TensorDesc({{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

// ===================== 负向用例：dtype 校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_q_descale_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[3] = TensorDesc({{256, 4}, {256, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_k_descale_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[4] = TensorDesc({{4, 4, 128}, {4, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_v_descale_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[5] = TensorDesc({{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_p_scale_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[6] = TensorDesc({{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_sparse_indices_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[11] = TensorDesc({{1, 4, 2, 4}, {1, 4, 2, 4}}, ge::DT_INT64, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_sparse_seq_len_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[12] = TensorDesc({{1, 4, 2}, {1, 4, 2}}, ge::DT_INT64, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_atten_mask_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[14] = TensorDesc({{2048, 2048}, {2048, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_block_table_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[13] = TensorDesc({{1, 8}, {1, 8}}, ge::DT_INT64, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_cu_seqlens_q_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[7] = TensorDesc({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_cu_seqlens_kv_dtype)
{
    auto inputs = MakeValidInputs();
    inputs[8] = TensorDesc({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_attention_out_dtype)
{
    auto outputs = MakeValidOutputs();
    outputs[0] = TensorDesc({{256, 4, 128}, {256, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), outputs, MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_softmax_lse_dtype)
{
    auto outputs = MakeValidOutputs();
    outputs[1] = TensorDesc({{4, 256}, {4, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), outputs, MakeValidAttrs()), false);
}

// ===================== 负向用例：必传输入存在性校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_missing_block_table)
{
    auto inputs = MakeValidInputs();
    inputs[13] = EmptyInput;
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_missing_cu_seqlens_q)
{
    auto inputs = MakeValidInputs();
    inputs[7] = EmptyInput;
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_missing_cu_seqlens_kv)
{
    auto inputs = MakeValidInputs();
    inputs[8] = EmptyInput;
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

// ===================== 负向用例：key/value shape 校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_value_shape_mismatch_blocknum)
{
    auto inputs = MakeValidInputs();
    inputs[2] = TensorDesc({{8, 4, 128, 128}, {8, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_value_shape_mismatch_n2)
{
    auto inputs = MakeValidInputs();
    inputs[2] = TensorDesc({{4, 3, 128, 128}, {4, 3, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_value_shape_mismatch_dv)
{
    auto inputs = MakeValidInputs();
    inputs[2] = TensorDesc({{4, 4, 128, 64}, {4, 4, 128, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_key_shape_mismatch_blocksize)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 4, 64, 128}, {4, 4, 64, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 4, 64, 128}, {4, 4, 64, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_key_shape_mismatch_headdim)
{
    auto inputs = MakeValidInputs();
    inputs[1] = TensorDesc({{4, 4, 128, 64}, {4, 4, 128, 64}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    inputs[2] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

// ===================== 负向用例：量化参数组 shape 校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_q_descale_shape_1d)
{
    auto inputs = MakeValidInputs();
    inputs[3] = TensorDesc({{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_q_descale_shape_mismatch_t1)
{
    auto inputs = MakeValidInputs();
    inputs[3] = TensorDesc({{128, 4}, {128, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_k_descale_shape_1d)
{
    auto inputs = MakeValidInputs();
    inputs[4] = TensorDesc({{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_v_descale_shape_mismatch_n2)
{
    auto inputs = MakeValidInputs();
    inputs[5] = TensorDesc({{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_p_scale_shape_not_scalar)
{
    auto inputs = MakeValidInputs();
    inputs[6] = TensorDesc({{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

// ===================== 负向用例：ActualSeqLen shape 校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_cu_seqlens_q_shape)
{
    auto inputs = MakeValidInputs();
    inputs[7] = TensorDesc({{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_cu_seqlens_kv_shape)
{
    auto inputs = MakeValidInputs();
    inputs[8] = TensorDesc({{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_seqused_q_shape)
{
    auto inputs = MakeValidInputs();
    inputs[9] = TensorDesc({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_seqused_kv_shape)
{
    auto inputs = MakeValidInputs();
    inputs[10] = TensorDesc({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

// ===================== 负向用例：atten_mask 维度值校验 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_atten_mask_dim_value)
{
    auto inputs = MakeValidInputs();
    inputs[14] = TensorDesc({{1024, 1024}, {1024, 1024}}, ge::DT_UINT8, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_invalid_atten_mask_dim_value_256)
{
    auto inputs = MakeValidInputs();
    inputs[14] = TensorDesc({{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

// ===================== 正向用例：seqused 可选传入 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_with_seqused)
{
    auto inputs = MakeValidInputs();
    inputs[9] = TensorDesc({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    inputs[10] = TensorDesc({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), true);
}

// ===================== 1D combined KV storage 用例 =====================

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_combined_kv)
{
    ExpectTilingResult(BuildTilingPara(Make1DKVInputs(), MakeValidOutputs(), MakeAttrsEx("TND", 131072)), true);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_pa_block_stride_zero)
{
    ExpectTilingResult(BuildTilingPara(Make1DKVInputs(), MakeValidOutputs(), MakeAttrsEx("TND", 0)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_invalid_storage_size)
{
    ExpectTilingResult(BuildTilingPara(Make1DKVInputs(4, 4, 524289), MakeValidOutputs(), MakeAttrsEx("TND", 131072)),
                       false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_n1_n2_mismatch)
{
    auto inputs = Make1DKVInputs();
    inputs[5] = TensorDesc({{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeAttrsEx("TND", 131072)), false);
}

TEST_F(QuantBlockSparseAttnTilingArch35, tiling_1d_kv_value_not_1d)
{
    auto inputs = Make1DKVInputs();
    inputs[2] = TensorDesc({{4, 4, 128, 128}, {4, 4, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeAttrsEx("TND", 131072)), false);
}
