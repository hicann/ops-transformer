/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rain_fusion_attention_tiling.cpp
 * \brief UT for RainFusionAttention tiling. Each negative case targets exactly one OP_LOGE report branch.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "../../../op_host/rain_fusion_attention_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {

constexpr uint64_t SKIP_TILING_KEY_CHECK = UINT64_MAX;

// 静态占位数据：用于shape为{0}但需要非空数据指针的用例
static int64_t g_dummyData = 0;

struct RfaTilingCase {
    std::string qLayout = "TND";
    std::string kvLayout = "TND";
    int64_t numKvHeads = 2;
    int64_t maskType = 0;
    float scaleValue = 0.1f;
    int64_t innerPrecise = 1;
    int64_t blockSize = 128;
    ge::DataType dtype = ge::DT_FLOAT16;

    int64_t t = 4;   // TND格式 query 总token数
    int64_t kvT = 4; // TND格式 kv 总token数
    int64_t b = 2;   // BNSD格式 batch
    int64_t s = 2;   // BNSD格式 query seqlen
    int64_t kvS = 2; // BNSD格式 kv seqlen
    int64_t n = 8;   // query heads
    int64_t kvN = 2; // kv heads
    int64_t d = 128; // head dim

    bool provideQSeqlen = true;
    bool provideKvSeqlen = true;
    std::vector<int64_t> qSeqlen = {2, 2};
    std::vector<int64_t> kvSeqlen = {2, 2};
    std::vector<int64_t> blockShape = {128, 128};
    int64_t maxKvBlockNum = 4;
};

gert::TilingContextPara BuildPara(const RfaTilingCase& c, optiling::RainFusionAttentionCompileInfo& compileInfo)
{
    bool isBnsd = (c.qLayout == "BNSD");
    const std::vector<int64_t>& qSeqlen = c.qSeqlen;
    const std::vector<int64_t>& kvSeqlen = c.kvSeqlen;

    // 0 query
    gert::TilingContextPara::TensorDescription queryDesc({{}, {}}, c.dtype, ge::FORMAT_ND);
    if (isBnsd) {
        queryDesc = gert::TilingContextPara::TensorDescription({{c.b, c.n, c.s, c.d}, {c.b, c.n, c.s, c.d}}, c.dtype,
                                                               ge::FORMAT_ND);
    } else {
        queryDesc =
            gert::TilingContextPara::TensorDescription({{c.t, c.n, c.d}, {c.t, c.n, c.d}}, c.dtype, ge::FORMAT_ND);
    }
    // 1 key / 2 value
    gert::TilingContextPara::TensorDescription keyDesc({{}, {}}, c.dtype, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription valueDesc({{}, {}}, c.dtype, ge::FORMAT_ND);
    if (isBnsd) {
        keyDesc = gert::TilingContextPara::TensorDescription({{c.b, c.kvN, c.kvS, c.d}, {c.b, c.kvN, c.kvS, c.d}},
                                                             c.dtype, ge::FORMAT_ND);
        valueDesc = gert::TilingContextPara::TensorDescription({{c.b, c.kvN, c.kvS, c.d}, {c.b, c.kvN, c.kvS, c.d}},
                                                               c.dtype, ge::FORMAT_ND);
    } else {
        keyDesc = gert::TilingContextPara::TensorDescription({{c.kvT, c.kvN, c.d}, {c.kvT, c.kvN, c.d}}, c.dtype,
                                                             ge::FORMAT_ND);
        valueDesc = gert::TilingContextPara::TensorDescription({{c.kvT, c.kvN, c.d}, {c.kvT, c.kvN, c.d}}, c.dtype,
                                                               ge::FORMAT_ND);
    }
    // 3 selectIdx / 4 selectNumIdx
    gert::TilingContextPara::TensorDescription selectIdxDesc({{c.b, c.n, c.maxKvBlockNum}, {c.b, c.n, c.maxKvBlockNum}},
                                                             ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription selectNumIdxDesc({{c.b, c.n}, {c.b, c.n}}, ge::DT_INT64, ge::FORMAT_ND);
    // 5 blockShape（必选，需要数据）
    gert::TilingContextPara::TensorDescription blockShapeDesc({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                              (void*)c.blockShape.data());
    // 6 attenMask（缺省）
    gert::TilingContextPara::TensorDescription attenMaskDesc({{}, {}}, c.dtype, ge::FORMAT_ND);
    // 7 actualSeqLengths / 8 actualSeqLengthsKv（可选，带数据）
    gert::TilingContextPara::TensorDescription qSeqlenDesc({{}, {}}, ge::DT_INT64, ge::FORMAT_ND);
    if (c.provideQSeqlen) {
        if (qSeqlen.empty()) {
            qSeqlenDesc = gert::TilingContextPara::TensorDescription({{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                     (void*)&g_dummyData);
        } else {
            qSeqlenDesc =
                gert::TilingContextPara::TensorDescription({{(int64_t)qSeqlen.size()}, {(int64_t)qSeqlen.size()}},
                                                           ge::DT_INT64, ge::FORMAT_ND, true, (void*)qSeqlen.data());
        }
    }
    gert::TilingContextPara::TensorDescription kvSeqlenDesc({{}, {}}, ge::DT_INT64, ge::FORMAT_ND);
    if (c.provideKvSeqlen) {
        if (kvSeqlen.empty()) {
            kvSeqlenDesc = gert::TilingContextPara::TensorDescription({{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                      (void*)&g_dummyData);
        } else {
            kvSeqlenDesc =
                gert::TilingContextPara::TensorDescription({{(int64_t)kvSeqlen.size()}, {(int64_t)kvSeqlen.size()}},
                                                           ge::DT_INT64, ge::FORMAT_ND, true, (void*)kvSeqlen.data());
        }
    }
    // 9 blockTable（缺省）
    gert::TilingContextPara::TensorDescription blockTableDesc({{}, {}}, ge::DT_INT32, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        queryDesc,      keyDesc,       valueDesc,   selectIdxDesc, selectNumIdxDesc,
        blockShapeDesc, attenMaskDesc, qSeqlenDesc, kvSeqlenDesc,  blockTableDesc};

    // 输出：0 attentionOut / 1 softmaxLse（缺省）
    gert::TilingContextPara::TensorDescription attentionOutDesc({{}, {}}, c.dtype, ge::FORMAT_ND);
    if (isBnsd) {
        attentionOutDesc = gert::TilingContextPara::TensorDescription({{c.b, c.n, c.s, c.d}, {c.b, c.n, c.s, c.d}},
                                                                      c.dtype, ge::FORMAT_ND);
    } else {
        attentionOutDesc =
            gert::TilingContextPara::TensorDescription({{c.t, c.n, c.d}, {c.t, c.n, c.d}}, c.dtype, ge::FORMAT_ND);
    }
    gert::TilingContextPara::TensorDescription softmaxLseDesc({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {attentionOutDesc, softmaxLseDesc};

    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.qLayout)},
        {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.kvLayout)},
        {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.numKvHeads)},
        {"maskType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.maskType)},
        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(c.scaleValue)},
        {"innerPrecise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.innerPrecise)},
        {"blockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.blockSize)}};

    return gert::TilingContextPara("RainFusionAttention", inputs, outputs, attrs, &compileInfo);
}

void RunCase(const RfaTilingCase& c, ge::graphStatus expectStatus)
{
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    ExecuteTestCase(para, expectStatus, SKIP_TILING_KEY_CHECK);
}
} // namespace

class RainFusionAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- RainFusionAttentionTiling UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- RainFusionAttentionTiling UT TearDown ---" << std::endl;
    }
};

// ============================ 成功用例 ============================

// TND格式成功用例：覆盖正常tiling流程与tilingKey生成
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_success)
{
    RfaTilingCase c;
    RunCase(c, ge::GRAPH_SUCCESS);
}

// BNSD格式成功用例：无actualSeqLengths时使用uniform seqlen
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_success)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_SUCCESS);
}

// ============================ 负向用例：算子逻辑报错 ============================

// 对应 ParseQInputLayout：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("qInputLayout", ...)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_q_layout_invalid)
{
    RfaTilingCase c;
    c.qLayout = "BSHD";
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ParseKvInputLayout：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("kvInputLayout", ...)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_kv_layout_invalid)
{
    RfaTilingCase c;
    c.kvLayout = "BSHD";
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessQueryShape（TND）：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", "TND format must have 3 dimensions")
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_query_dims_invalid)
{
    RfaTilingCase c;
    c.d = 128; // 保持其余合法
    // TND下query给4维
    c.qLayout = "TND";
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // 手工改query为4维：直接构造4维query
    para.inputTensorDesc_[0] = gert::TilingContextPara::TensorDescription({{c.b, c.n, c.s, c.d}, {c.b, c.n, c.s, c.d}},
                                                                          c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ProcessQueryShape（BNSD）：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", "BNSD format must have 4
// dimensions")
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_query_dims_invalid)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // 手工改query为3维
    para.inputTensorDesc_[0] =
        gert::TilingContextPara::TensorDescription({{c.t, c.n, c.d}, {c.t, c.n, c.d}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ProcessQueryShape：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", "N dimension (numHeads) ...")
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_numheads_zero)
{
    RfaTilingCase c;
    c.n = 0;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessQSeqLengths（TND）：OP_LOGE_WITH_INVALID_INPUT("actualSeqLengths")
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_seqlen_absent)
{
    RfaTilingCase c;
    c.provideQSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessQSeqLengths（TND）：OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON("actualSeqLengths", batch==0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_seqlen_empty)
{
    RfaTilingCase c;
    c.qSeqlen = {};
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessKvSeqLengthsTND：OP_LOGE_WITH_INVALID_INPUT("actualSeqLengthsKv")
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_kvseqlen_absent)
{
    RfaTilingCase c;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessKvSeqLengthsTND：OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON("actualSeqLengthsKv", batch不匹配)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_kvseqlen_batch_mismatch)
{
    RfaTilingCase c;
    c.kvSeqlen = {2, 2, 2}; // size 3 != batch 2
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessQSeqLengths（BNSD uniform分支）：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", S维度必须大于0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_maxqseqlen_zero)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.s = 0;
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateBNSDQSeqlen：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("actualSeqLengths", 超过maxQSeqlen)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_qseqlen_exceed)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.b = 1;
    c.s = 2;
    c.kvS = 2;
    c.qSeqlen = {4}; // 4 > maxQSeqlen(2)
    c.provideQSeqlen = true;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 GetKvSeqlenFromShape：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("key", KV S维度必须大于0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_kv_s_zero)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.kvS = 0;
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateBNSDQSeqlen：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", 提供数组时S维度必须大于0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_maxqseqlen_zero_with_array)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.b = 1;
    c.s = 0;         // maxQSeqlen_为0
    c.qSeqlen = {1}; // 提供actualSeqLengths且batch匹配
    c.provideQSeqlen = true;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateBNSDKvSeqlen：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("actualSeqLengthsKv", 超过maxKvSeqlen)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_kvseqlen_exceed)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.b = 1;
    c.kvSeqlen = {8}; // 8 > maxKvSeqlen(2)
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateConfiguration：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", D维度必须大于0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_embedding_size_zero)
{
    RfaTilingCase c;
    c.d = 0;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateConfiguration：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("query", embeddingSize必须为64或128)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_embedding_size_invalid)
{
    RfaTilingCase c;
    c.d = 96;
    RunCase(c, ge::GRAPH_FAILED);
}

// issue 6677：headDim=32 非法（仅支持 64/128）
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_embedding_size_32_invalid)
{
    RfaTilingCase c;
    c.d = 32;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateConfiguration：OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("blockShape", 必须大于0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_blockshape_nonpositive)
{
    RfaTilingCase c;
    c.blockShape = {0, 128};
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateConfiguration：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("blockShape", blockShapeY必须能被128整除)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_blockshape_not_divisible)
{
    RfaTilingCase c;
    c.blockShape = {128, 100};
    RunCase(c, ge::GRAPH_FAILED);
}

// issue 6677：blockShapeY=64 不能被 128 整除
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_blockshape_y64_not_divisible)
{
    RfaTilingCase c;
    c.blockShape = {128, 64};
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ProcessBlockShape：OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON("blockShape", 元素数少于2)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_blockshape_elem_num_invalid)
{
    RfaTilingCase c;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // 手工改blockShape为单元素，图模式下首段校验不生效，tiling需自行拦截
    para.inputTensorDesc_[5] = gert::TilingContextPara::TensorDescription({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                          (void*)c.blockShape.data());
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// BF16 当前仅支持 float32 softmax，innerPrecise=1 会被归一化为 0
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bf16_innerprecise_normalized)
{
    RfaTilingCase c;
    c.dtype = ge::DT_BF16;
    c.innerPrecise = 1;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 9000000030022222ULL);
}

// 对应 CheckKvCacheLayout：OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("qInputLayout, kvInputLayout", 必须一致)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_layout_mismatch)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "TND";
    c.provideQSeqlen = false; // BNSD的Q用uniform seqlen
    c.kvSeqlen = {2, 2};      // TND的KV需要数组且batch匹配（batch来自BNSD query的B=2）
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateTNDFormat：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("key", TND格式KV必须3维)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_key_dims_invalid)
{
    RfaTilingCase c;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // 手工改key为4维
    para.inputTensorDesc_[1] = gert::TilingContextPara::TensorDescription(
        {{c.b, c.kvN, c.kvS, c.d}, {c.b, c.kvN, c.kvS, c.d}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ValidateTNDFormat：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("numKeyValueHeads", 与key的N维不匹配)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_kv_heads_mismatch)
{
    RfaTilingCase c;
    c.kvN = 4; // attr numKeyValueHeads=2
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateTNDFormat：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("key", 必须匹配embeddingSize)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_key_headdim_mismatch)
{
    RfaTilingCase c;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // key/value的D改为64，query仍为128
    para.inputTensorDesc_[1] =
        gert::TilingContextPara::TensorDescription({{c.kvT, c.kvN, 64}, {c.kvT, c.kvN, 64}}, c.dtype, ge::FORMAT_ND);
    para.inputTensorDesc_[2] =
        gert::TilingContextPara::TensorDescription({{c.kvT, c.kvN, 64}, {c.kvT, c.kvN, 64}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ValidateBNSDFormat：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("key", BNSD格式KV必须4维)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_key_dims_invalid)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // 手工改key为3维
    para.inputTensorDesc_[1] =
        gert::TilingContextPara::TensorDescription({{c.kvT, c.kvN, c.d}, {c.kvT, c.kvN, c.d}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ValidateBNSDFormat：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("key", batch必须匹配query的B)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_key_batch_mismatch)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // key/value的B改为3，query B=2
    para.inputTensorDesc_[1] = gert::TilingContextPara::TensorDescription(
        {{3, c.kvN, c.kvS, c.d}, {3, c.kvN, c.kvS, c.d}}, c.dtype, ge::FORMAT_ND);
    para.inputTensorDesc_[2] = gert::TilingContextPara::TensorDescription(
        {{3, c.kvN, c.kvS, c.d}, {3, c.kvN, c.kvS, c.d}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ValidateBNSDFormat：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("key", 必须匹配embeddingSize)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_key_headdim_mismatch)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // key/value的D改为64，query仍为128
    para.inputTensorDesc_[1] = gert::TilingContextPara::TensorDescription(
        {{c.b, c.kvN, c.kvS, 64}, {c.b, c.kvN, c.kvS, 64}}, c.dtype, ge::FORMAT_ND);
    para.inputTensorDesc_[2] = gert::TilingContextPara::TensorDescription(
        {{c.b, c.kvN, c.kvS, 64}, {c.b, c.kvN, c.kvS, 64}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ValidateBNSDFormat：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("numKeyValueHeads", BNSD下与key的N维不匹配)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_kv_heads_mismatch)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.kvN = 4; // attr numKeyValueHeads=2
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateBNSDFormat：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("value", BNSD格式Value必须4维)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_value_dims_invalid)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // 手工改value为3维
    para.inputTensorDesc_[2] =
        gert::TilingContextPara::TensorDescription({{c.kvT, c.kvN, c.d}, {c.kvT, c.kvN, c.d}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 ValidateBNSDFormat：OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("key and value", Value shape必须与Key一致)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_bnsd_value_shape_mismatch)
{
    RfaTilingCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.provideQSeqlen = false;
    c.provideKvSeqlen = false;
    optiling::RainFusionAttentionCompileInfo compileInfo;
    auto para = BuildPara(c, compileInfo);
    // value的D改为64，与key不一致
    para.inputTensorDesc_[2] = gert::TilingContextPara::TensorDescription(
        {{c.b, c.kvN, c.kvS, 64}, {c.b, c.kvN, c.kvS, 64}}, c.dtype, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED, SKIP_TILING_KEY_CHECK);
}

// 对应 CalculateTaskSplit：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("numKeyValueHeads", 必须大于0)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_numkvheads_zero)
{
    RfaTilingCase c;
    c.numKvHeads = 0;
    c.kvN = 0; // key的N也置0以通过ValidateTNDFormat
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateTNDSeqlenSum：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("actualSeqLengths", sum(qseqlen)必须等于T)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_qseqlen_sum_mismatch)
{
    RfaTilingCase c;
    c.qSeqlen = {3, 2}; // sum=5 != T=4
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 ValidateTNDSeqlenSum：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("actualSeqLengthsKv", sum(kvseqlen)必须等于KV T)
TEST_F(RainFusionAttentionTiling, rain_fusion_attention_tnd_kvseqlen_sum_mismatch)
{
    RfaTilingCase c;
    c.kvSeqlen = {3, 2}; // sum=5 != KV T=4
    RunCase(c, ge::GRAPH_FAILED);
}
