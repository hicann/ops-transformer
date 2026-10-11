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
 * \file ts_aclnn_quant_grouped_matmul_inplace_add.cpp
 * \brief QGMMInplaceAdd ACLNN 测试用例.
 */

#include "ts_aclnn_quant_grouped_matmul_inplace_add.h"

using qgmmInplaceAddTestParam::Ts_Aclnn_QGMMInplaceAdd_WithParam_Ascend910_9591;
namespace {

TEST_P(Ts_Aclnn_QGMMInplaceAdd_WithParam_Ascend910_9591, Tc_Tiling_QGmmInplaceAdd)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_QGmm_Aclnn_David_Case = ::testing::Values(
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_mxfp8_Case0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam({GenTensor("x1", {512, 96}, ge::DataType::DT_FLOAT8_E5M2),
                                  GenTensor("x2", {512, 128}, ge::DataType::DT_FLOAT8_E5M2),
                                  GenTensor("scale2", {12, 128, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
                                  GenTensor("scale1", {12, 96, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
                                 {128, 256, 300, 512}, 0, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_mxfp8_Case1", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam({GenTensor("x1", {512, 96}, ge::DataType::DT_FLOAT8_E5M2),
                                  GenTensor("x2", {512, 128}, ge::DataType::DT_FLOAT8_E4M3FN),
                                  GenTensor("scale2", {12, 128, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
                                  GenTensor("scale1", {12, 96, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
                                 {128, 256, 318, 512}, 0, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_mxfp8_Case2", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam({GenTensor("x1", {512, 128}, ge::DataType::DT_FLOAT8_E4M3FN),
                                  GenTensor("x2", {512, 96}, ge::DataType::DT_FLOAT8_E5M2),
                                  GenTensor("scale2", {12, 96, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("y", {4, 128, 96}, ge::DataType::DT_FLOAT),
                                  GenTensor("scale1", {12, 128, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
                                 {128, 256, 400, 512}, 0, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_mxfp8_Case3", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam({GenTensor("x1", {512, 96}, ge::DataType::DT_FLOAT8_E4M3FN),
                                  GenTensor("x2", {512, 128}, ge::DataType::DT_FLOAT8_E4M3FN),
                                  GenTensor("scale2", {12, 128, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
                                  GenTensor("scale1", {12, 96, 2}, ge::DataType::DT_FLOAT8_E8M0),
                                  GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
                                 {128, 256, 300, 512}, 0, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_mxfp8_Case4", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 128}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_K_C_error_Case5", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 128}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4, 96}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_y_shape_error_Case6", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 128}, ge::DataType::DT_FLOAT),
             GenTensor("y", {5, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_scale2_shape_error_Case7", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 52}, ge::DataType::DT_FLOAT), GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
             GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_group_limit_error_Case", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam({GenTensor("x1", {1025, 96}, ge::DataType::DT_HIFLOAT8),
                                  GenTensor("x2", {1025, 128}, ge::DataType::DT_HIFLOAT8),
                                  GenTensor("scale2", {1025, 128}, ge::DataType::DT_FLOAT),
                                  GenTensor("y", {1025, 96, 128}, ge::DataType::DT_FLOAT),
                                  GenTensor("scale1", {1025}, ge::DataType::DT_FLOAT),
                                  GenTensor("groupList", {1025}, ge::DataType::DT_INT64)},
                                 std::vector<int64_t>(1025, 1), 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_M0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 0}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 128}, ge::DataType::DT_FLOAT), GenTensor("y", {4, 0, 128}, ge::DataType::DT_FLOAT),
             GenTensor("scale1", {4, 0}, ge::DataType::DT_FLOAT), GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_N0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 0}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4, 0}, ge::DataType::DT_FLOAT),
             GenTensor("y", {5, 96, 0}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_K0", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(
            ControlInfo(true, false),
            ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                       ExpectInfo::kInvalidTilingBlockDim)), /* ExpectSuccess, ExpectTilingKey, ExpectTilingBlockDim */
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {0, 96}, ge::DataType::DT_HIFLOAT8), GenTensor("x2", {0, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 128}, ge::DataType::DT_FLOAT),
             GenTensor("y", {5, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {0, 0, 0, 0}, 1, 0),
        0),
    // 失败用例: x1 dim != 2 (3D), CheckShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_X1_Dim_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {2, 256, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: x2 dim != 2 (3D), CheckShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_X2_Dim_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {2, 256, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4}, ge::DataType::DT_FLOAT), GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
             GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: groupList dim != 1 (2D), CheckShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_GroupList_Dim_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4, 1}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: y dim != 3 (2D), CheckShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Y_Dim_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: x1.k != x2.k, CheckShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_K_Mismatch", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {256, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: y dtype 非 FLOAT, CheckDtype 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Y_Dtype_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT16), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: groupList dtype 非 INT64, CheckDtype 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_GroupList_Dtype_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT32)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: 混合 dtype (x1=HIFLOAT8, x2=FLOAT8_E4M3FN), CheckDtype 不支持组合
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Mixed_Dtype_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_FLOAT8_E4M3FN),
             GenTensor("scale2", {4}, ge::DataType::DT_FLOAT), GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
             GenTensor("scale1", {4}, ge::DataType::DT_FLOAT), GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: scale1 dim=3, CheckHif8QuantParamsShape 失败 (HIFLOAT8 模式下 perToken 必须 1D 或 2D)
    AclnnQGMMInplaceAddCase("AclnnQGMMInplaceAddCase_Scale1_Dim_Invalid", true, "",
                            OpInfo(ControlInfo(true, false), ExpectInfo(false, ExpectInfo::kInvalidTilingKey,
                                                                        ExpectInfo::kInvalidTilingBlockDim)),
                            AclnnQGMMInplaceAddParam({GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
                                                      GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
                                                      GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
                                                      GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
                                                      GenTensor("scale1", {4, 1, 1}, ge::DataType::DT_FLOAT),
                                                      GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
                                                     {128, 128, 128, 128}, 1, 0),
                            0),
    // 失败用例: scale1 firstDim != groupNum, CheckHif8QuantParamsShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Scale1_GroupMismatch", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {5}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: scale2 firstDim != groupNum (1D), CheckHif8QuantParamsShape 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Scale2_GroupMismatch", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {5}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: HIFLOAT8 下 scale1 dtype 非 FLOAT, CheckHif8QuantParams 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Scale1_Dtype_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT16),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 失败用例: HIFLOAT8 下 scale2 dtype 非 FLOAT, CheckHif8QuantParams 失败
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_Scale2_Dtype_Invalid", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(false, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT16),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 正常用例: HIFLOAT8 T-T pertensor, scale1/scale2 均为 (g,), groupList cumsum
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_TT_Pertensor_1D", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 256, 384, 512}, 0, 0),
        0),
    // 正常用例: HIFLOAT8 T-T pertensor, scale1/scale2 均为 (g, 1)
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_TT_Pertensor_2D_TrailingOne", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
             GenTensor("scale2", {4, 1}, ge::DataType::DT_FLOAT), GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
             GenTensor("scale1", {4, 1}, ge::DataType::DT_FLOAT), GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 256, 384, 512}, 0, 0),
        0),
    // 正常用例: HIFLOAT8 T-C, scale2 形如 (g, n) per-channel, scale1 (g,)
    AclnnQGMMInplaceAddCase("AclnnQGMMInplaceAddCase_TC_PerChannel", true, "",
                            OpInfo(ControlInfo(true, false),
                                   ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
                            AclnnQGMMInplaceAddParam({GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
                                                      GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8),
                                                      GenTensor("scale2", {4, 128}, ge::DataType::DT_FLOAT),
                                                      GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT),
                                                      GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
                                                      GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
                                                     {128, 256, 384, 512}, 0, 0),
                            0),
    // 正常用例: HIFLOAT8 T-T pertensor, count 模式 (groupListType=1)
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_TT_CountMode", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {4}, ge::DataType::DT_FLOAT),
             GenTensor("y", {4, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {4}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {4}, ge::DataType::DT_INT64)},
            {128, 128, 128, 128}, 1, 0),
        0),
    // 正常用例: HIFLOAT8 T-T pertensor, 单分组 g=1
    AclnnQGMMInplaceAddCase(
        "AclnnQGMMInplaceAddCase_TT_SingleGroup", true, "",
        OpInfo(ControlInfo(true, false),
               ExpectInfo(true, ExpectInfo::kInvalidTilingKey, ExpectInfo::kInvalidTilingBlockDim)),
        AclnnQGMMInplaceAddParam(
            {GenTensor("x1", {512, 96}, ge::DataType::DT_HIFLOAT8),
             GenTensor("x2", {512, 128}, ge::DataType::DT_HIFLOAT8), GenTensor("scale2", {1}, ge::DataType::DT_FLOAT),
             GenTensor("y", {1, 96, 128}, ge::DataType::DT_FLOAT), GenTensor("scale1", {1}, ge::DataType::DT_FLOAT),
             GenTensor("groupList", {1}, ge::DataType::DT_INT64)},
            {512}, 0, 0),
        0));
INSTANTIATE_TEST_SUITE_P(QuantGroupedMatmulInplaceAdd, Ts_Aclnn_QGMMInplaceAdd_WithParam_Ascend910_9591,
                         Tc_QGmm_Aclnn_David_Case);
} // namespace
