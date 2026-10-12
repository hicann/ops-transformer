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
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/qkv_rms_norm_rope_cache_tiling.h"

using namespace std;
using namespace ge;

int batch_size;
int seq_len;
int Nqkv;
int Nq;
int Nk;
int Nv;
int dim;
int block_num;
int block_size;

class QkvRmsNormRopeCacheTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QkvRmsNormRopeCacheTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QkvRmsNormRopeCacheTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

template <typename T>
static string to_string(void* buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quantA)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quant_AS)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quantB)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 11898;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quantC)
{
    batch_size = 16;
    seq_len = 3;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 11898;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_k_quant)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_v_quant)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_no_quant)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quant_small)
{
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_k_quant_small)
{
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_v_quant_small)
{
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_fp16_pa_nz_no_quant_small)
{
    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 16, block_size, 16}, {block_num, Nv * dim / 16, block_size, 16}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_is_None_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_dimSize_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size + 2, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkvDim_32_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 129;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_gamma_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_gamma_dims_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, dim}, {1, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_gamma_dimOne_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{72}, {72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_cos_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_cos_dims_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1 * seq_len, dim}, {1 * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_cos_dtype_diff_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_sin_is_None_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_sin_dims_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_sin_dtype_diff_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_index_is_None_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_index_dims_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len + 2}, {batch_size * seq_len + 2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_index_dtype_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qout_is_None_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qout_dtype_diff_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qout_dims_wrong)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim + 2}, {batch_size * seq_len, Nq * dim + 2}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 16, block_size, 16}, {block_num, Nk * dim / 16, block_size, 16}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kcache_blocksize_32_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 129;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_dtype_diff_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num + 1, Nv * dim / 32, block_size, 32}, {block_num + 1, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vcache_dims_blocksize_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size + 1, 32}, {block_num, Nv * dim / 32, block_size + 1, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kscale_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kscale_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_kscale_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{10, dim}, {10, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_is_None_int8_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_is_notNone_fp16_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_dtype_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_vscale_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_size_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_size_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, -2};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_qkv_size_dims_relation_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 2;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_is_None_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_dims_wrong)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq + 2, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_shape_wrong1)
{
    // dlog_setlevel(0, 0, 0);
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 3;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq + 2, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            // input info
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}

TEST_F(QkvRmsNormRopeCacheTiling, test_QkvRmsNormRopeCache_headNums_shape_wrong2)
{
    batch_size = 72;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 3;
    Nv = 3;
    dim = 128;
    block_num = 72;
    block_size = 128;

    string cache_mode("PA_NZ");
    vector<int64_t> qkv_size = {batch_size, seq_len, Nqkv, dim};
    vector<int64_t> head_nums = {Nq + 2, Nk, Nv};

    optiling::QkvRmsNormRopeCacheCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "QkvRmsNormRopeCache",
        {
            {{{batch_size * seq_len, Nqkv * dim}, {batch_size * seq_len, Nqkv * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{dim}, {dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len, dim}, {batch_size * seq_len, dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{batch_size * seq_len}, {batch_size * seq_len}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{batch_size * seq_len, Nq * dim}, {batch_size * seq_len, Nq * dim}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{block_num, Nk * dim / 32, block_size, 32}, {block_num, Nk * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{block_num, Nv * dim / 32, block_size, 32}, {block_num, Nv * dim / 32, block_size, 32}},
             ge::DT_INT8,
             ge::FORMAT_ND},
            {{{Nk, dim}, {Nk, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{Nv, dim}, {Nv, dim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"qkv_size", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(qkv_size)},
            {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(head_nums)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-06)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(cache_mode)},
            {"is_output_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 3;
    string expectTilingData = "";
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData);
}
