/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/swin_transformer_ln_qkv_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
using namespace std;
class SwinTransformerLnQkvQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwinTransformerLnQkvQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwinTransformerLnQkvQuantTiling TearDown" << std::endl;
    }
};

TEST_F(SwinTransformerLnQkvQuantTiling, swin_transformer_ln_qkv_quant_tiling_1)
{
    optiling::SwinTransformerLnQkvQuantCompileInfo compileInfo = {8};
    gert::TilingContextPara tilingContextPara(
        "SwinTransformerLnQkvQuant",
        {
            {{{1, 6272, 96}, {1, 6272, 96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{288, 96}, {288, 96}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{128, 3, 49, 32}, {128, 3, 49, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"seq_length", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(0.00001)},
            {"ori_height", Ops::Transformer::AnyValue::CreateFrom<int64_t>(56)},
            {"ori_weight", Ops::Transformer::AnyValue::CreateFrom<int64_t>(112)},
            {"h_win_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"w_win_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"weight_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);
    int64_t expectTilingKey = 100000UL;
    string expectTilingData =
        "602112 0 997257046392832 412316860928 3974362538702799136 26938034880513 12884901984 30064771079 34359738400 "
        "420906795024 481036337248 4294967392 1 420906795008 0 412316860514 288 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "420906795009 412316860704 420906795104 412316860704 1099511627888 4294967392 4294967298 4294967298 0 "
        "215504279044096 114688 4294967297 4294967297 4294967297 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1048576};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SwinTransformerLnQkvQuantTiling, swin_transformer_ln_qkv_quant_tiling_subm_loop)
{
    optiling::SwinTransformerLnQkvQuantCompileInfo compileInfo = {8};
    gert::TilingContextPara tilingContextPara(
        "SwinTransformerLnQkvQuant",
        {
            {{{3, 6272, 640}, {3, 6272, 640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640}, {640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640}, {640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1920, 640}, {1920, 640}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1920}, {1920}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{640}, {640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{640}, {640}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1920}, {1920}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{156, 20, 56, 32}, {156, 20, 56, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"seq_length", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(0.00001)},
            {"ori_height", Ops::Transformer::AnyValue::CreateFrom<int64_t>(56)},
            {"ori_weight", Ops::Transformer::AnyValue::CreateFrom<int64_t>(112)},
            {"h_win_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"w_win_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
            {"weight_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);
    int64_t expectTilingKey = 100000UL;
    string expectTilingData =
        "12042240 0 964271697559552 2748779069952 3974362538702800768 26938034880515 12884902528 30064771079 "
        "34359738400 68719476752 68719477376 17179869824 1 1262720385024 0 2748779069489 640 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 210453397505 2748779070080 210453398144 2748779070080 1099511627840 21474836608 4294967298 4294967297 "
        "4294967296 285873023221760 65536 4294967297 4294967297 4294967301 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {1048576};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
