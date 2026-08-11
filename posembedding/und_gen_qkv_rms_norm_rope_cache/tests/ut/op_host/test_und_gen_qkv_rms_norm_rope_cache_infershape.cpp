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
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;
constexpr size_t OUTPUT_IDX_Q = 0;
constexpr size_t OUTPUT_IDX_K_CACHE = 1;
constexpr size_t OUTPUT_IDX_V_CACHE = 2;

class UndGenQkvRmsNormRopeCacheInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UndGenQkvRmsNormRopeCacheInferShape TearDown" << std::endl;
    }
};

gert::InfershapeContextPara BuildInferShapeContext(int64_t undLen, int64_t genLen, int64_t numHeadQ, int64_t numHeadK,
                                                   int64_t numHeadV, int64_t headDim = HEAD_DIM,
                                                   int64_t maxPos = 4096)
{
    const int64_t total = undLen + genLen;
    const int64_t numHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (total + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    return gert::InfershapeContextPara(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, headDim}, {undLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, headDim}, {maxPos, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, headDim}, {blockNum, BLOCK_SIZE, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, headDim}, {blockNum, BLOCK_SIZE, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, total}, {MROPE_AXIS_NUM, total}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{genLen, numHead, headDim}, {genLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{total}, {total}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16, 16})},
        });
}

// 纯 prefill 形态：4 个可选输入一个都不实例化（实例数为 0），只传 8 个必选输入
gert::InfershapeContextPara BuildInferShapeContextWithoutOptional(int64_t undLen, int64_t numHeadQ, int64_t numHeadK,
                                                                  int64_t numHeadV, int64_t headDim = HEAD_DIM,
                                                                  int64_t maxPos = 4096)
{
    const int64_t numHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (undLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    return gert::InfershapeContextPara(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, headDim}, {undLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, headDim}, {maxPos, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, headDim}, {blockNum, BLOCK_SIZE, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, headDim}, {blockNum, BLOCK_SIZE, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{undLen}, {undLen}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, undLen}, {MROPE_AXIS_NUM, undLen}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16, 16})},
        },
        // gen_qkv/gen_weights_q/gen_weights_k/cat_indices 的实例数为 0，即调用方未传
        {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, {1, 1, 1});
}

// 只实例化 cat_indices 一个可选输入，gen 那三个都不传：
// 用来钉住"按 IR 下标取可选输入"这件事——此时 cat_indices 的 IR 下标是 11，实例化下标却是 8
gert::InfershapeContextPara BuildInferShapeContextOnlyCatIndices(int64_t undLen, int64_t catLen, int64_t numHeadQ,
                                                                 int64_t numHeadK, int64_t numHeadV,
                                                                 int64_t headDim = HEAD_DIM, int64_t maxPos = 4096)
{
    const int64_t numHead = numHeadQ + numHeadK + numHeadV;
    const int64_t blockNum = (undLen + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    return gert::InfershapeContextPara(
        "UndGenQkvRmsNormRopeCache",
        {
            {{{undLen, numHead, headDim}, {undLen, numHead, headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{headDim}, {headDim}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{maxPos, headDim}, {maxPos, headDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadK, headDim}, {blockNum, BLOCK_SIZE, numHeadK, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{blockNum, BLOCK_SIZE, numHeadV, headDim}, {blockNum, BLOCK_SIZE, numHeadV, headDim}},
             ge::DT_BF16,
             ge::FORMAT_ND},
            {{{undLen}, {undLen}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{MROPE_AXIS_NUM, undLen}, {MROPE_AXIS_NUM, undLen}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{catLen}, {catLen}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_heads_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadQ)},
            {"num_heads_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadK)},
            {"num_heads_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numHeadV)},
            {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(NORM_EPS)},
            {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 16, 16})},
        },
        {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1}, {1, 1, 1});
}

// 造一个指定 rank 的 shape，各维取值相同
gert::StorageShape MakeStorageShape(size_t rank, int64_t dimValue)
{
    gert::StorageShape shape;
    for (size_t i = 0; i < rank; ++i) {
        shape.MutableOriginShape().AppendDim(dimValue);
        shape.MutableStorageShape().AppendDim(dimValue);
    }
    return shape;
}

gert::StorageShape UnknownRankShape()
{
    return {{-2}, {-2}};
}

void CheckInferDataType()
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("UndGenQkvRmsNormRopeCache")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType bf16Dtype = ge::DT_BF16;
    ge::DataType floatDtype = ge::DT_FLOAT;
    ge::DataType int64Dtype = ge::DT_INT64;
    ge::DataType qOutDtype = ge::DT_FLOAT;
    ge::DataType kCacheOutDtype = ge::DT_FLOAT;
    ge::DataType vCacheOutDtype = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(12)
                             .NodeIoNum(12, 3)
                             .NodeInputTd(0, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, floatDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, int64Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, int64Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(8, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(9, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(10, bf16Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(11, int64Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&bf16Dtype, &bf16Dtype, &bf16Dtype, &floatDtype, &bf16Dtype, &bf16Dtype,
                                              &int64Dtype, &int64Dtype, &bf16Dtype, &bf16Dtype, &bf16Dtype,
                                              &int64Dtype})
                             .OutputDataTypes({&qOutDtype, &kCacheOutDtype, &vCacheOutDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_Q), bf16Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_K_CACHE), bf16Dtype);
    EXPECT_EQ(context->GetOutputDataType(OUTPUT_IDX_V_CACHE), bf16Dtype);
}
} // namespace

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_success_h8_1_1)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    // total = 8, blockNum = 1 + 1 = 2
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
    CheckInferDataType();
}

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_success_h16_2_2)
{
    auto para = BuildInferShapeContext(80, 16, 16, 2, 2);
    // total = 96, blockNum = 1 + 1 = 2
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {96, 16, 128},
        {2, 128, 2, 128},
        {2, 128, 2, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infershape 只保留推导前提的校验（und_qkv/gen_qkv 必须是 3D），
// dtype、维度值、跨输入一致性、支持范围的校验统一在 tiling 侧，见 test_*_tiling.cpp
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_und_qkv_not_3d)
{
    auto para = BuildInferShapeContext(8, 0, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{8, 1280}, {8, 1280}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_gen_qkv_not_3d)
{
    auto para = BuildInferShapeContext(8, 4, 8, 1, 1);
    para.inputTensorDesc_[8] = {{{4, 1280}, {4, 1280}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// rank 不是 -2 就是确定的：任一输入 rank 与 IR 定义不符都直接判失败，
// 不能因为"还能从别的输入推出同一个维度"就把它跳过去
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_cos_sin_cache_not_2d)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[3] = {{{4096, 128, 1}, {4096, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_k_cache_not_4d)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[4] = {{{2, 128, 128}, {2, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// 可选输入同样要查：cat_indices 传了就必须是 1D
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_fail_when_cat_indices_not_1d)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[11] = {{{2, 4}, {2, 4}}, ge::DT_INT64, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// 三个输出的 rank 与头数恒定（头数来自属性），T、D、Bn、Bs 能从任一输入推出就推，
// 全都推不出才置 -1。以下用例逐条钉住各维度的取值链路。

// und_qkv 为未知 rank（-2）：T 退到 positions 的 [3, T]，D 退到 cos_sin_cache 的 [max_pos, D]
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_unknown_rank_und_qkv)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// gen_qkv 为未知 rank：und_len + gen_len 不成立，T 退到 positions；D 仍从 und_qkv 尾维取
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_unknown_rank_gen_qkv)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[8] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// D 的取值链路：und_qkv 与 cos_sin_cache 都是 -2 时，退到 und_weights_q 的 [D]
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_head_dim_falls_back_to_weights)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    para.inputTensorDesc_[3] = {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// T 的取值链路：und_qkv 与 positions 都是 -2 时，退到 slot_mapping 的 [T]
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_total_falls_back_to_slot_mapping)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    para.inputTensorDesc_[7] = {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 12 个输入全是 -2：T、D、Bn、Bs 都推不出来，只剩属性给出的头数
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_unknown_rank_all_inputs)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    for (auto& desc : para.inputTensorDesc_) {
        desc.shape_ = {{-2}, {-2}};
    }
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, -1},
        {-1, -1, 1, -1},
        {-1, -1, 1, -1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 4 个可选输入全部不传（实例数为 0）：T 退化为 und_len 单段，不再做加法
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_without_optional_inputs)
{
    auto para = BuildInferShapeContextWithoutOptional(5, 8, 1, 1);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 可选输入不传、且 8 个必选输入全是 -2：取值链路走到未实例化的可选输入时必须跳过，
// 不能把"没传"当成一个可用来源，最终 T 与 D 都推不出来
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_without_optional_inputs_all_unknown_rank)
{
    auto para = BuildInferShapeContextWithoutOptional(5, 8, 1, 1);
    for (auto& desc : para.inputTensorDesc_) {
        desc.shape_ = {{-2}, {-2}};
    }
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, -1},
        {-1, -1, 1, -1},
        {-1, -1, 1, -1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 可选输入部分实例化：只传 cat_indices，且把 und_qkv/positions/slot_mapping 都置为 -2，
// 使 cat_indices 成为 T 的唯一来源。它的 IR 下标是 11、实例化下标是 8，
// 若用 GetInputShape(11) 取会落空，本用例即失败。
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_total_from_only_instantiated_cat_indices)
{
    auto para = BuildInferShapeContextOnlyCatIndices(5, 7, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-2}, {-2}}, ge::DT_BF16, ge::FORMAT_ND};
    para.inputTensorDesc_[6] = {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND};
    para.inputTensorDesc_[7] = {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {7, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank 已知但按 token 组织的维度全为动态 shape（-1）：T 必须是 -1。
// 修复前无条件做 und_len + gen_len，会算出 (-1) + 3 = 2 并推出貌似合法的 q。
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_dynamic_dim_no_total_source)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{-1, 10, 128}, {-1, 10, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    para.inputTensorDesc_[6] = {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND};
    para.inputTensorDesc_[7] = {{{3, -1}, {3, -1}}, ge::DT_INT64, ge::FORMAT_ND};
    para.inputTensorDesc_[11] = {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// gen_len 为动态 shape（-1）而 und_len 已知：加法同样不成立，且无其他 T 来源时为 -1
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_dynamic_dim_gen_len)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[6] = {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND};
    para.inputTensorDesc_[7] = {{{3, -1}, {3, -1}}, ge::DT_INT64, ge::FORMAT_ND};
    para.inputTensorDesc_[8] = {{{-1, 10, 128}, {-1, 10, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    para.inputTensorDesc_[11] = {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 来源的 rank 合法、但目标维本身是动态 shape（-1）时也要继续往下找：
// und_qkv 的 D 维为 -1，D 应退到 cos_sin_cache
TEST_F(UndGenQkvRmsNormRopeCacheInferShape, infershape_skip_source_whose_dim_is_dynamic)
{
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    para.inputTensorDesc_[0] = {{{5, 10, -1}, {5, 10, -1}}, ge::DT_BF16, ge::FORMAT_ND};
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        {2, 128, 1, 128},
        {2, 128, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

namespace {
struct RankCase {
    size_t inputIdx;
    size_t expectDimNum;
    const char* name;
};

struct HeadDimSourceCase {
    size_t inputIdx;
    const char* name;
};

// D 出现在这 9 个输入里，与 infershape 中 HEAD_DIM_SOURCES 一一对应
constexpr size_t HEAD_DIM_INPUT_IDXS[] = {0, 1, 2, 3, 4, 5, 8, 9, 10};
constexpr size_t IDX_K_CACHE = 4;
constexpr size_t IDX_V_CACHE = 5;
} // namespace

class UndGenQkvRmsNormRopeCacheInferShapeRank : public testing::TestWithParam<RankCase> {};

// 12 个输入逐个验：rank 与 IR 定义不符必须失败。
// 这是在钉 INPUT_SPECS 这张表本身的录入正确性，表里任一条 expectDimNum 写错都会被抓到。
TEST_P(UndGenQkvRmsNormRopeCacheInferShapeRank, infershape_fail_when_rank_mismatch)
{
    const auto& param = GetParam();
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    auto dtype = para.inputTensorDesc_[param.inputIdx].dtype_;
    // 比预期多一维
    para.inputTensorDesc_[param.inputIdx] = {MakeStorageShape(param.expectDimNum + 1, 1), dtype, ge::FORMAT_ND};
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

INSTANTIATE_TEST_SUITE_P(AllInputs, UndGenQkvRmsNormRopeCacheInferShapeRank,
                         testing::Values(RankCase{0, 3, "und_qkv"}, RankCase{1, 1, "und_weights_q"},
                                         RankCase{2, 1, "und_weights_k"}, RankCase{3, 2, "cos_sin_cache"},
                                         RankCase{4, 4, "k_cache"}, RankCase{5, 4, "v_cache"},
                                         RankCase{6, 1, "slot_mapping"}, RankCase{7, 2, "positions"},
                                         RankCase{8, 3, "gen_qkv"}, RankCase{9, 1, "gen_weights_q"},
                                         RankCase{10, 1, "gen_weights_k"}, RankCase{11, 1, "cat_indices"}),
                         [](const testing::TestParamInfo<RankCase>& info) { return std::string(info.param.name); });

class UndGenQkvRmsNormRopeCacheInferShapeHeadDim : public testing::TestWithParam<HeadDimSourceCase> {};

// 9 个带 D 的输入逐个验：把其余 8 个都置为 -2，D 必须仍能从剩下的这一个取到。
// 这是在钉 HEAD_DIM_SOURCES 里每一条的 dimIdx 写得对不对。
// 其中 v_cache 那条同时覆盖了 Bn/Bs 在 k_cache 不可用时退到 v_cache 的路径。
TEST_P(UndGenQkvRmsNormRopeCacheInferShapeHeadDim, infershape_head_dim_from_single_source)
{
    const auto& param = GetParam();
    auto para = BuildInferShapeContext(5, 3, 8, 1, 1);
    for (size_t idx : HEAD_DIM_INPUT_IDXS) {
        if (idx != param.inputIdx) {
            para.inputTensorDesc_[idx].shape_ = UnknownRankShape();
        }
    }
    // T 恒由 positions 给出；Bn/Bs 只有在保留 k_cache 或 v_cache 时才可知
    const bool cacheKnown = (param.inputIdx == IDX_K_CACHE) || (param.inputIdx == IDX_V_CACHE);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8, 128},
        cacheKnown ? std::vector<int64_t>{2, 128, 1, 128} : std::vector<int64_t>{-1, -1, 1, 128},
        cacheKnown ? std::vector<int64_t>{2, 128, 1, 128} : std::vector<int64_t>{-1, -1, 1, 128},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(AllHeadDimSources, UndGenQkvRmsNormRopeCacheInferShapeHeadDim,
                         testing::Values(HeadDimSourceCase{0, "und_qkv"}, HeadDimSourceCase{1, "und_weights_q"},
                                         HeadDimSourceCase{2, "und_weights_k"}, HeadDimSourceCase{3, "cos_sin_cache"},
                                         HeadDimSourceCase{4, "k_cache"}, HeadDimSourceCase{5, "v_cache"},
                                         HeadDimSourceCase{8, "gen_qkv"}, HeadDimSourceCase{9, "gen_weights_q"},
                                         HeadDimSourceCase{10, "gen_weights_k"}),
                         [](const testing::TestParamInfo<HeadDimSourceCase>& info) {
                             return std::string(info.param.name);
                         });
