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
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using AnyValue = Ops::Transformer::AnyValue;
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// TND 布局下 actual_seq_lengths 为累加和：单 batch、query 1 个 token、KV 有效长度 512。
int32_t g_actualSeqQuery[] = {1};
int32_t g_actualSeqKv[] = {512};

// query [T=1, N1=8, D=576]；KV 为 PA_BSND [blockNum=4, blockSize=128, N2=1, 386]，
// slot 386B = 256B 打包 nibble + 128B bfloat16 rope + 2B float16 归一化系数。
std::vector<TensorDesc> BaseInputs()
{
    return {{{{1, 8, 576}, {1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1, 386}, {4, 128, 1, 386}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 128, 1, 386}, {4, 128, 1, 386}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 256}, {1, 1, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // key_dequant_scale：COMBINE 模式下不消费
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // value_dequant_scale
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, g_actualSeqQuery},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, g_actualSeqKv}};
}

std::vector<TensorDesc> BaseOutputs()
{
    return {{{{1, 8, 512}, {1, 8, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}};
}

// 属性基线取 host 侧唯一合法值；负向用例在此基础上单点改写，以定位到具体校验项。
std::vector<OpAttr> BaseAttrs()
{
    return {{"scale_value", AnyValue::CreateFrom<float>(0.044194173f)},
            {"key_quant_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"value_quant_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"sparse_block_size", AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", AnyValue::CreateFrom<int64_t>(64)},
            {"return_softmax_lse", AnyValue::CreateFrom<bool>(false)}};
}

// 单点改写某个属性
std::vector<OpAttr> AttrsWith(const std::string &name, const AnyValue &value)
{
    auto attrs = BaseAttrs();
    for (auto &attr : attrs) {
        if (attr.attrName_ == name) {
            attr.attr_ = value;
        }
    }
    return attrs;
}
} // namespace

class TurboQuantSparseFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TurboQuantSparseFlashAttentionTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TurboQuantSparseFlashAttentionTiling TearDown" << std::endl; }
};

struct TurboQuantSparseFlashAttentionCompileInfo {};

// ---------------- 正向 ----------------

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_base)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(), BaseAttrs(),
                                 &compileInfo, "Ascend910B", 64, 262144, 16384);
    // 期望值由本用例实跑采集，用于锁定 tiling 结果不被后续改动无声改变。
    int64_t expectTilingKey = 1158;
    std::string expectTilingData = "2199023255553 1 128 4294967300 4410436851802832897 4294967304 3 1 "
                                   "256 386 274877907456 3 3 128 0 0 40532396646334464 262144 64 "
                                   "39582418608128 2199023255560 ";
    std::vector<size_t> expectWorkspaces = {184090624};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ---------------- 负向：属性取值 ----------------

// tile_size 未被 kernel 消费，slot 布局按 128 写死，非 128 不产生对应语义，须直接拒绝。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_tile_size_64)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("tile_size", AnyValue::CreateFrom<int64_t>(64)), &compileInfo, "Ascend910B",
                                 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_rope_head_dim_32)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("rope_head_dim", AnyValue::CreateFrom<int64_t>(32)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_key_quant_mode_1)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("key_quant_mode", AnyValue::CreateFrom<int64_t>(1)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_attention_mode_0)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("attention_mode", AnyValue::CreateFrom<int64_t>(0)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_quant_scale_repo_mode_0)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("quant_scale_repo_mode", AnyValue::CreateFrom<int64_t>(0)), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------- 负向：layout ----------------

// query 仅支持 TND：launcher 强制 query 为 3D，输出固定按 TND 构造。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_layout_query_bsnd)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("layout_query", AnyValue::CreateFrom<std::string>("BSND")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// KV 仅支持 PA_BSND：非 PA_BSND 的 KV 要求与 query 同 layout，而 query 已限定为 TND。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_layout_kv_bsnd)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>("BSND")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_layout_kv_tnd)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                 AttrsWith("layout_kv", AnyValue::CreateFrom<std::string>("TND")), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---------------- 负向：dtype 与 shape ----------------

// 算子仅声明支持 BFLOAT16，FP16 已从支持列表移除。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_query_fp16)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[0].dtype_ = ge::DT_FLOAT16;
    auto outputs = BaseOutputs();
    outputs[0].dtype_ = ge::DT_FLOAT16;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_actual_seq_kv_int64)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[8].dtype_ = ge::DT_INT64;
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// query 的 head dim 写死为 576（512 latent + 64 rope），kernel 的 MM1 K 宽度按此固定。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_query_head_dim_640)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[0].shape_ = {{1, 8, 640}, {1, 8, 640}};
    auto outputs = BaseOutputs();
    outputs[0].shape_ = {{1, 8, 576}, {1, 8, 576}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, outputs, BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// MLA 场景下 K 与 V 为同一份 latent，value 的 shape 必须与 key 完全一致。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_value_shape_mismatch)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[2].shape_ = {{2, 128, 1, 386}, {2, 128, 1, 386}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// block_table 的第二维（每 batch 的最大 block 数）须大于 0：PA_BSND 下
// s2Size = block_table.dim1 * block_size，该校验保证 s2Size 恒大于 0，
// 因此 tiling 无需再为 s2Size == 0 兜底。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_block_table_dim1_zero)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[6].shape_ = {{1, 0}, {1, 0}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// block_table 首维须等于 batch size。
TEST_F(TurboQuantSparseFlashAttentionTiling, TurboQuantSparseFlashAttention_910b_block_table_batch_mismatch)
{
    TurboQuantSparseFlashAttentionCompileInfo compileInfo;
    auto inputs = BaseInputs();
    inputs[6].shape_ = {{2, 4}, {2, 4}};
    gert::TilingContextPara para("TurboQuantSparseFlashAttention", inputs, BaseOutputs(), BaseAttrs(), &compileInfo,
                                 "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
