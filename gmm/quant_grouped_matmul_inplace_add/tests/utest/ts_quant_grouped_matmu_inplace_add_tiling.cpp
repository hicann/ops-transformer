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
 * \file ts_quant_grouped_matmul_inplace_add_tiling.cpp
 * \brief QuantGroupedMatmulInplaceAdd tiling用例.
 */

#include "ts_quant_grouped_matmul_inplace_add.h"

using qgmmiaTestParam::Ts_QuantGroupedMatmulInplaceAdd_WithParam_Ascend910_9591;
namespace {
const auto Tc_QuantGroupedMatmulInplaceAdd_Tiling_Case = ::testing::Values(
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_mxfp8_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, 4, 32)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_FLOAT8_E5M2),
               GenTensorList("scale2", {{12, 128, 2}}, ge::DataType::DT_FLOAT8_E8M0),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {12, 96, 2}, ge::DataType::DT_FLOAT8_E8M0),
              GenTensor("group_list", {4}, ge::DataType::DT_INT64), {1, 1, 1, 1}, 0, 0),
        0),
    // pertensor-pertensor count 模式: groupListType=1, 各 group 中 m 的数量
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_pertensor_pertensor_CountMode", true, "",
        OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey, 32)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{4}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 128, 128, 128}, 1, 0),
        0),
    // pertensor-pertensor 单分组 g=1
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_pertensor_pertensor_SingleGroup", true, "",
        OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey, 32)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{1}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{1, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {1}, ge::DataType::DT_FLOAT), GenTensor("group_list", {1}, ge::DataType::DT_INT64),
              {512}, 0, 0),
        0),
    // pertensor-pertensor 多分组 g=8
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_pertensor_pertensor_MultiGroup", true, "",
        OpInfo(ControlInfo(true, false), ExpectInfo(true, ExpectInfo::kInvalidTilingKey, 32)),
        Param({GenTensorList("x1", {{1024, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{1024, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{8, 1}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{8, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {8, 1}, ge::DataType::DT_FLOAT), GenTensor("group_list", {8}, ge::DataType::DT_INT64),
              {128, 256, 384, 512, 640, 768, 896, 1024}, 0, 0),
        0),
    // 失败用例: y dtype 非 DT_FLOAT, CheckDtype 失败
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_Invalid_YDtype", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{4}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT16)},
              GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 256, 384, 512}, 0, 0),
        0),
    // 失败用例: HIFLOAT8 下 scale2 dtype 非 DT_FLOAT, CheckDtype 失败
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_Invalid_Scale2Dtype", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{4}}, ge::DataType::DT_FLOAT16),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 256, 384, 512}, 0, 0),
        0),
    // 失败用例: HIFLOAT8 下 scale1 (perTokenScale) dtype 非 DT_FLOAT, CheckDtype 失败
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_Invalid_Scale1Dtype", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{4}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {4}, ge::DataType::DT_FLOAT16), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 256, 384, 512}, 0, 0),
        0),
    // 失败用例: scale1 group 数不匹配 (g+1)
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_Invalid_Scale1GroupMismatch", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{4}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {5}, ge::DataType::DT_FLOAT), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 256, 384, 512}, 0, 0),
        0),
    // 失败用例: scale1 形如 (g, 2), 第二维不为 1, CheckShapeForHif8Quant 失败
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_Invalid_Scale1TrailingDim", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{4}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {4, 2}, ge::DataType::DT_FLOAT), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 256, 384, 512}, 0, 0),
        0),
    // 失败用例: scale2 group 数不匹配 (g+1)
    QuantGroupedMatmulInplaceAddCase(
        "QuantGroupedQuantInplaceAddMM_Invalid_Scale2GroupMismatch", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        Param({GenTensorList("x1", {{512, 96}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("x2", {{512, 128}}, ge::DataType::DT_HIFLOAT8),
               GenTensorList("scale2", {{5}}, ge::DataType::DT_FLOAT),
               GenTensorList("y", {{4, 96, 128}}, ge::DataType::DT_FLOAT)},
              GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("group_list", {4}, ge::DataType::DT_INT64),
              {128, 256, 384, 512}, 0, 0),
        0));

INSTANTIATE_TEST_SUITE_P(QuantGroupedMatmulInplaceAdd_David, Ts_QuantGroupedMatmulInplaceAdd_WithParam_Ascend910_9591,
                         Tc_QuantGroupedMatmulInplaceAdd_Tiling_Case);
TEST_P(Ts_QuantGroupedMatmulInplaceAdd_WithParam_Ascend910_9591, Tc_Tiling_QGmmInplaceAdd)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

} // namespace
