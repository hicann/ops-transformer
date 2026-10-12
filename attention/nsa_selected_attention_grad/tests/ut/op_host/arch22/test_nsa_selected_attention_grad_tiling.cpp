/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/nsa_selected_attention_grad_tiling_common.h"

namespace {
constexpr int64_t kBlockCountBasic = 16;
constexpr int64_t kBlockSizeBasic = 64;
constexpr int64_t kBlockCountGeneral = 32;

constexpr int64_t kBatch = 1;
constexpr int64_t kT1 = 2;
constexpr int64_t kT2 = 2048;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kDhead = 192;
constexpr int64_t kDhead2 = 128;

// Basic template (priority 1) tilingKey:
//   sparse_mode == 0 (no mask): 10
//   sparse_mode == 2 (with mask): 11
constexpr uint64_t kTilingKeyBasicNoMask = 10UL;
constexpr uint64_t kTilingKeyBasicWithMask = 11UL;

// General template (priority 10) tilingKey:
//   sparse_mode == 0 (no mask, deterministic=0): 0
//   sparse_mode == 2 (with mask, deterministic=0): 1
constexpr uint64_t kTilingKeyGeneralNoMask = 0UL;

// Prefix-sum of actual_seq_q_len / actual_seq_kv_len for batch=1 setting.
int64_t kActualSeqQLen[1] = {kT1};
int64_t kActualSeqKVLen[1] = {kT2};

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

inline optiling::nsa::NsaSelectedAttentionGradCompileInfo MakeCompileInfo()
{
    optiling::nsa::NsaSelectedAttentionGradCompileInfo info{};
    info.aivNum = 1U;
    info.aicNum = 1U;
    info.ubSize = 196608ULL;
    info.l1Size = 524288ULL;
    info.l0aSize = 65536ULL;
    info.l0bSize = 65536ULL;
    info.l0cSize = 131072ULL;
    info.l2CacheSize = 33554432ULL;
    info.coreNum = 1;
    return info;
}

// 11 inputs: q, k, v, attentionOut, attentionOutGrad, softmaxMax, softmaxSum,
//            topkIndices, actualSeqQLen(opt), actualSeqKVLen(opt), attenMsk(opt)
std::vector<TensorDesc> MakeBaseInputs(ge::DataType dtype, int64_t blockCount, bool withMask, int64_t d = kDhead,
                                       int64_t d2 = kDhead2, int64_t n1 = kN1, int64_t n2 = kN2, int64_t t1 = kT1,
                                       int64_t t2 = kT2)
{
    std::vector<TensorDesc> inputs = {
        {{{t1, n1, d}, {t1, n1, d}}, dtype, ge::FORMAT_ND},                          // query
        {{{t2, n2, d}, {t2, n2, d}}, dtype, ge::FORMAT_ND},                          // key
        {{{t2, n2, d2}, {t2, n2, d2}}, dtype, ge::FORMAT_ND},                        // value
        {{{t1, n1, d2}, {t1, n1, d2}}, dtype, ge::FORMAT_ND},                        // attentionOut
        {{{t1, n1, d2}, {t1, n1, d2}}, dtype, ge::FORMAT_ND},                        // attentionOutGrad
        {{{t1, n1, 8}, {t1, n1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // softmaxMax (DT_FLOAT)
        {{{t1, n1, 8}, {t1, n1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                   // softmaxSum (DT_FLOAT)
        {{{t1, n2, blockCount}, {t1, n2, blockCount}}, ge::DT_INT32, ge::FORMAT_ND}, // topkIndices
        {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND, true, static_cast<void*>(kActualSeqQLen)}, // actualSeqQLen
        {{{kBatch}, {kBatch}},
         ge::DT_INT64,
         ge::FORMAT_ND,
         true,
         static_cast<void*>(kActualSeqKVLen)}, // actualSeqKVLen
    };
    if (withMask) {
        inputs.push_back({{{kBlockSizeBasic, kBlockSizeBasic}, {kBlockSizeBasic, kBlockSizeBasic}},
                          ge::DT_BOOL,
                          ge::FORMAT_ND}); // attenMsk
    } else {
        inputs.push_back({{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND}); // attenMsk empty
    }
    return inputs;
}

std::vector<TensorDesc> MakeBaseOutputs(ge::DataType dtype, int64_t d = kDhead, int64_t d2 = kDhead2, int64_t n1 = kN1,
                                        int64_t n2 = kN2, int64_t t1 = kT1, int64_t t2 = kT2)
{
    return {
        {{{t1, n1, d}, {t1, n1, d}}, dtype, ge::FORMAT_ND},
        {{{t2, n2, d}, {t2, n2, d}}, dtype, ge::FORMAT_ND},
        {{{t2, n2, d2}, {t2, n2, d2}}, dtype, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeAttrs(int64_t blockCount, int64_t blockSize = kBlockSizeBasic, int64_t sparseMode = 0,
                              const std::string& layout = "TND", int64_t headNum = kN1, float scaleValue = 0.088388f)
{
    return {
        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
        {"selectedBlockCount", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockCount)},
        {"selectedBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"headNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headNum)},
        {"inputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
    };
}
} // namespace

class NsaSelectedAttentionGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaSelectedAttentionGradTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "NsaSelectedAttentionGradTiling TearDown" << std::endl;
    }
};

// A1: positive - basic template, fp16, sparse_mode=0 (no mask).
//     selected_block_count=16, selected_block_size=64, d=192, d2=128 -> basic IsCapable hits.
TEST_F(NsaSelectedAttentionGradTiling, A1_basic_fp16_no_mask)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttentionGrad", MakeBaseInputs(ge::DT_FLOAT16, kBlockCountBasic, /*withMask=*/false),
        MakeBaseOutputs(ge::DT_FLOAT16), MakeAttrs(kBlockCountBasic, kBlockSizeBasic, /*sparseMode=*/0), &compileInfo,
        "Ascend910B",
        /*coreNum=*/1,
        /*ubSize=*/196608ULL,
        /*tilingDataSize=*/8192);

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyBasicNoMask);
}

// A2: positive - basic template, bf16, sparse_mode=2 (with mask).
TEST_F(NsaSelectedAttentionGradTiling, A2_basic_bf16_with_mask)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttentionGrad", MakeBaseInputs(ge::DT_BF16, kBlockCountBasic, /*withMask=*/true),
        MakeBaseOutputs(ge::DT_BF16), MakeAttrs(kBlockCountBasic, kBlockSizeBasic, /*sparseMode=*/2), &compileInfo,
        "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyBasicWithMask);
}

// A3: positive - general template, fp16, sparse_mode=0.
//     selected_block_count=32 != 16, so basic IsCapable=false, general is used.
TEST_F(NsaSelectedAttentionGradTiling, A3_general_fp16_no_mask)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttentionGrad", MakeBaseInputs(ge::DT_FLOAT16, kBlockCountGeneral, /*withMask=*/false),
        MakeBaseOutputs(ge::DT_FLOAT16), MakeAttrs(kBlockCountGeneral, kBlockSizeBasic, /*sparseMode=*/0), &compileInfo,
        "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyGeneralNoMask);
}

// E1: negative - invalid sparse_mode=1.
TEST_F(NsaSelectedAttentionGradTiling, E1_invalid_sparse_mode)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttentionGrad", MakeBaseInputs(ge::DT_FLOAT16, kBlockCountBasic, /*withMask=*/false),
        MakeBaseOutputs(ge::DT_FLOAT16), MakeAttrs(kBlockCountBasic, kBlockSizeBasic, /*sparseMode=*/1), &compileInfo,
        "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2: negative - invalid layout "BSH".
TEST_F(NsaSelectedAttentionGradTiling, E2_invalid_layout_BSH)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttentionGrad", MakeBaseInputs(ge::DT_FLOAT16, kBlockCountBasic, /*withMask=*/false),
        MakeBaseOutputs(ge::DT_FLOAT16), MakeAttrs(kBlockCountBasic, kBlockSizeBasic, /*sparseMode=*/0, "BSH"),
        &compileInfo, "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E3: negative - q.head_dim mismatches k.head_dim.
//     query D=128 but key D=192 -> CheckBaseShapeInfo fails.
TEST_F(NsaSelectedAttentionGradTiling, E3_qkv_head_dim_mismatch)
{
    auto compileInfo = MakeCompileInfo();
    std::vector<TensorDesc> inputs = {
        {{{kT1, kN1, 128}, {kT1, kN1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},                         // query d=128
        {{{kT2, kN2, kDhead}, {kT2, kN2, kDhead}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key d=192
        {{{kT2, kN2, kDhead2}, {kT2, kN2, kDhead2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // value d2=128
        {{{kT1, kN1, kDhead2}, {kT1, kN1, kDhead2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // attentionOut
        {{{kT1, kN1, kDhead2}, {kT1, kN1, kDhead2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // attentionOutGrad
        {{{kT1, kN1, 8}, {kT1, kN1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // softmaxMax
        {{{kT1, kN1, 8}, {kT1, kN1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // softmaxSum
        {{{kT1, kN2, kBlockCountBasic}, {kT1, kN2, kBlockCountBasic}}, ge::DT_INT32, ge::FORMAT_ND}, // topkIndices
        {{{kBatch}, {kBatch}}, ge::DT_INT64, ge::FORMAT_ND, true, static_cast<void*>(kActualSeqQLen)}, // actualSeqQLen
        {{{kBatch}, {kBatch}},
         ge::DT_INT64,
         ge::FORMAT_ND,
         true,
         static_cast<void*>(kActualSeqKVLen)},  // actualSeqKVLen
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND}, // attenMsk empty
    };

    gert::TilingContextPara para("NsaSelectedAttentionGrad", inputs, MakeBaseOutputs(ge::DT_FLOAT16),
                                 MakeAttrs(kBlockCountBasic, kBlockSizeBasic, /*sparseMode=*/0), &compileInfo,
                                 "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E4: negative - missing actualSeqQLen / actualSeqKVLen optional tensors in TND.
//     Tiling reads them mandatorily for TND, so it must FAIL.
TEST_F(NsaSelectedAttentionGradTiling, E4_missing_actual_seq_lens)
{
    auto compileInfo = MakeCompileInfo();
    std::vector<TensorDesc> inputs = {
        {{{kT1, kN1, kDhead}, {kT1, kN1, kDhead}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // query
        {{{kT2, kN2, kDhead}, {kT2, kN2, kDhead}}, ge::DT_FLOAT16, ge::FORMAT_ND},                   // key
        {{{kT2, kN2, kDhead2}, {kT2, kN2, kDhead2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // value
        {{{kT1, kN1, kDhead2}, {kT1, kN1, kDhead2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // attentionOut
        {{{kT1, kN1, kDhead2}, {kT1, kN1, kDhead2}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // attentionOutGrad
        {{{kT1, kN1, 8}, {kT1, kN1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // softmaxMax
        {{{kT1, kN1, 8}, {kT1, kN1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},                               // softmaxSum
        {{{kT1, kN2, kBlockCountBasic}, {kT1, kN2, kBlockCountBasic}}, ge::DT_INT32, ge::FORMAT_ND}, // topkIndices
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // actualSeqQLen empty
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // actualSeqKVLen empty
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},  // attenMsk empty
    };

    gert::TilingContextPara para("NsaSelectedAttentionGrad", inputs, MakeBaseOutputs(ge::DT_FLOAT16),
                                 MakeAttrs(kBlockCountBasic, kBlockSizeBasic, /*sparseMode=*/0), &compileInfo,
                                 "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E5: negative - invalid selected_block_count (out of [1, 128] range).
//     basic IsCapable=false (block_count!=16) so general kicks in and rejects 200.
TEST_F(NsaSelectedAttentionGradTiling, E5_invalid_block_count_too_large)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara para(
        "NsaSelectedAttentionGrad", MakeBaseInputs(ge::DT_FLOAT16, /*blockCount=*/200, /*withMask=*/false),
        MakeBaseOutputs(ge::DT_FLOAT16), MakeAttrs(/*blockCount=*/200, kBlockSizeBasic, /*sparseMode=*/0), &compileInfo,
        "Ascend910B", 1, 196608ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
