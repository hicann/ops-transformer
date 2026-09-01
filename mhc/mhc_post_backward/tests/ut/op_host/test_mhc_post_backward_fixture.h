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
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/mhc_post_backward_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

constexpr char kSocVersion950[] = "Ascend950";
// ==================== 枚举定义 ====================

// 错误类型枚举
enum class ErrorType {
    NONE, // 正常用例
    // ======================== 空指针/空tensor ===========================
    GRAD_OUTPUT_NULLPTR,
    GRAD_OUTPUT_NULL_DESC,
    X_NULLPTR,
    X_NULL_DESC,
    H_RES_NULLPTR,
    H_RES_NULL_DESC,
    H_OUT_NULLPTR,
    H_OUT_NULL_DESC,
    H_POST_NULLPTR,
    H_POST_NULL_DESC,
    GRAD_X_NULLPTR,
    GRAD_X_NULL_DESC,
    GRAD_H_RES_NULLPTR,
    GRAD_H_RES_NULL_DESC,
    GRAD_H_OUT_NULLPTR,
    GRAD_H_OUT_NULL_DESC,
    GRAD_H_POST_NULLPTR,
    GRAD_H_POST_NULL_DESC,
    // ======================== 异常 shape ===========================
    GRAD_OUTPUT_2D,
    GRAD_OUTPUT_5D,
    X_SHAPE_DIFF,
    H_RES_SHAPE_DIFF_BS,
    H_RES_SHAPE_DIFF_T,
    H_RES_SHAPE_DIFF_NN,
    H_OUT_SHAPE_NOT_BSD,
    H_OUT_SHAPE_NOT_TD,
    H_POST_SHAPE_NOT_BSN,
    H_POST_SHAPE_NOT_TN,
    GRAD_X_SHAPE_DIFF_GRAD_OUTPUT,
    GRAD_H_RES_SHAPE_DIFF_H_RES,
    GRAD_H_OUT_SHAPE_DIFF_H_OUT,
    GRAD_H_POST_SHAPE_DIFF_H_POST,
    // ======================== 异常 dtype ===========================
    GRAD_OUTPUT_DTYPE_FLOAT,
    X_DTYPE_DIFF_GRAD_OUTPUT,
    H_OUT_DTYPE_DIFF_GRAD_OUTPUT,
    GRAD_X_DTYPE_DIFF_GRAD_OUTPUT,
    GRAD_H_OUT_DTYPE_DIFF_GRAD_OUTPUT,
    H_RES_DTYPE_NOT_FP32,
    H_POST_DTYPE_NOT_FP32,
    GRAD_H_RES_DTYPE_NOT_FP32,
    GRAD_H_POST_DTYPE_NOT_FP32,
    // ======================== 不支持的 shape ===========================
    BS_GT_512K,
    T_GT_512K,
    N_NOT_4_6_8,
    D_GT_24K
};

// ==================== 测试夹具类 ====================

class MhcPostBackwardTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPostBackwardTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPostBackwardTiling TearDown" << std::endl;
    }

    /**
     * @brief 构造TND模式(3D)的TilingContext
     */
    static gert::TilingContextPara BuildTilingContextTND(uint32_t T, uint32_t n, uint32_t D, ge::DataType dtype,
                                                         const std::string &socVersion)
    {
        static optiling::MhcPostBackwardCompileInfo compileInfo = {48, 48};
        return gert::TilingContextPara("MhcPostBackward",
                                       {
                                           {{{T, n, D}, {T, n, D}}, dtype, ge::FORMAT_ND},        // grad_output
                                           {{{T, n, D}, {T, n, D}}, dtype, ge::FORMAT_ND},        // x
                                           {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND}, // h_res
                                           {{{T, D}, {T, D}}, dtype, ge::FORMAT_ND},              // h_out
                                           {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}        // h_post
                                       },
                                       {
                                           {{{T, n, D}, {T, n, D}}, dtype, ge::FORMAT_ND},        // grad_x
                                           {{{T, n, n}, {T, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grad_h_res
                                           {{{T, D}, {T, D}}, dtype, ge::FORMAT_ND},              // grad_h_out
                                           {{{T, n}, {T, n}}, ge::DT_FLOAT, ge::FORMAT_ND}        // grad_h_post
                                       },
                                       {}, &compileInfo, socVersion);
    }

    /**
     * @brief 构造BSND模式(4D)的TilingContext
     */
    static gert::TilingContextPara BuildTilingContextBSND(uint32_t B, uint32_t S, uint32_t n, uint32_t D,
                                                          ge::DataType dtype, const std::string &socVersion)
    {
        static optiling::MhcPostBackwardCompileInfo compileInfo = {48, 48};
        return gert::TilingContextPara("MhcPostBackward",
                                       {
                                           {{{B, S, n, D}, {B, S, n, D}}, dtype, ge::FORMAT_ND},        // grad_output
                                           {{{B, S, n, D}, {B, S, n, D}}, dtype, ge::FORMAT_ND},        // x
                                           {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND}, // h_res
                                           {{{B, S, D}, {B, S, D}}, dtype, ge::FORMAT_ND},              // h_out
                                           {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}        // h_post
                                       },
                                       {
                                           {{{B, S, n, D}, {B, S, n, D}}, dtype, ge::FORMAT_ND},        // grad_x
                                           {{{B, S, n, n}, {B, S, n, n}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grad_h_res
                                           {{{B, S, D}, {B, S, D}}, dtype, ge::FORMAT_ND},              // grad_h_out
                                           {{{B, S, n}, {B, S, n}}, ge::DT_FLOAT, ge::FORMAT_ND}        // grad_h_post
                                       },
                                       {}, &compileInfo, socVersion);
    }

    /**
     * @brief 根据错误类型修改TilingContext
     */
    static void ApplyErrorModifications(gert::TilingContextPara &context, ErrorType errorType, bool isTshape,
                                        uint32_t n, uint32_t D, uint32_t T, uint32_t B, uint32_t S)
    {
        switch (errorType) {
            case ErrorType::NONE:
                break;
            // 空指针/空tensor
            case ErrorType::GRAD_OUTPUT_NULL_DESC:
                context.inputTensorDesc_[0].shape_ = {{}, {}};
                break;
            case ErrorType::X_NULL_DESC:
                context.inputTensorDesc_[1].shape_ = {{}, {}};
                break;
            case ErrorType::H_RES_NULL_DESC:
                context.inputTensorDesc_[2].shape_ = {{}, {}};
                break;
            case ErrorType::H_OUT_NULL_DESC:
                context.inputTensorDesc_[3].shape_ = {{}, {}};
                break;
            case ErrorType::H_POST_NULL_DESC:
                context.inputTensorDesc_[4].shape_ = {{}, {}};
                break;

            case ErrorType::GRAD_X_NULL_DESC:
                context.outputTensorDesc_[0].shape_ = {{}, {}};
                break;
            case ErrorType::GRAD_H_RES_NULL_DESC:
                context.outputTensorDesc_[1].shape_ = {{}, {}};
                break;
            case ErrorType::GRAD_H_OUT_NULL_DESC:
                context.outputTensorDesc_[2].shape_ = {{}, {}};
                break;
            case ErrorType::GRAD_H_POST_NULL_DESC:
                context.outputTensorDesc_[3].shape_ = {{}, {}};
                break;

            // 异常 shape
            case ErrorType::GRAD_OUTPUT_2D:
                context.inputTensorDesc_[0].shape_ = {{n, n}, {n, n}};
                break;
            case ErrorType::GRAD_OUTPUT_5D:
                context.inputTensorDesc_[0].shape_ = {{B, S, n, n, n}, {B, S, n, n, n}};
                break;
            case ErrorType::X_SHAPE_DIFF:
                if (isTshape) {
                    context.inputTensorDesc_[1].shape_ = {{T, n, D + 1}, {T, n, D + 1}};
                } else {
                    context.inputTensorDesc_[1].shape_ = {{B, S, n, D + 1}, {B, S, n, D + 1}};
                }
                break;
            case ErrorType::H_RES_SHAPE_DIFF_BS:
            case ErrorType::H_RES_SHAPE_DIFF_T:
                if (isTshape) {
                    context.inputTensorDesc_[2].shape_ = {{T + 2, n, n}, {T + 2, n, n}};
                } else {
                    context.inputTensorDesc_[2].shape_ = {{B + 2, S, n, n}, {B + 2, S, n, n}};
                }
                break;
            case ErrorType::H_RES_SHAPE_DIFF_NN:
                if (isTshape) {
                    context.inputTensorDesc_[2].shape_ = {{T, n + 2, n}, {T, n + 2, n}};
                } else {
                    context.inputTensorDesc_[2].shape_ = {{B, S, n + 2, n}, {B, S, n + 2, n}};
                }
                break;
            case ErrorType::H_OUT_SHAPE_NOT_BSD:
            case ErrorType::H_OUT_SHAPE_NOT_TD:
                if (isTshape) {
                    context.inputTensorDesc_[3].shape_ = {{T, D + 4}, {T, D + 4}};
                } else {
                    context.inputTensorDesc_[3].shape_ = {{B, S, D + 4}, {B, S, D + 4}};
                }
                break;
            case ErrorType::H_POST_SHAPE_NOT_BSN:
            case ErrorType::H_POST_SHAPE_NOT_TN:
                if (isTshape) {
                    context.inputTensorDesc_[4].shape_ = {{T, n + 4}, {T, n + 4}};
                } else {
                    context.inputTensorDesc_[4].shape_ = {{B, S, n + 4}, {B, S, n + 4}};
                }
                break;
            case ErrorType::GRAD_X_SHAPE_DIFF_GRAD_OUTPUT:
                if (isTshape) {
                    context.outputTensorDesc_[0].shape_ = {{T, n, D + 4}, {T, n, D + 4}};
                } else {
                    context.outputTensorDesc_[0].shape_ = {{B, S, n, D + 4}, {B, S, n, D + 4}};
                }
                break;
            case ErrorType::GRAD_H_RES_SHAPE_DIFF_H_RES:
                if (isTshape) {
                    context.outputTensorDesc_[1].shape_ = {{T, n, n + 4}, {T, n, n + 4}};
                } else {
                    context.outputTensorDesc_[1].shape_ = {{B, S, n, n + 4}, {B, S, n, n + 4}};
                }
                break;
            case ErrorType::GRAD_H_OUT_SHAPE_DIFF_H_OUT:
                if (isTshape) {
                    context.outputTensorDesc_[2].shape_ = {{T, D + 4}, {T, D + 4}};
                } else {
                    context.outputTensorDesc_[2].shape_ = {{B, S, D + 4}, {B, S, D + 4}};
                }
                break;
            case ErrorType::GRAD_H_POST_SHAPE_DIFF_H_POST:
                if (isTshape) {
                    context.outputTensorDesc_[3].shape_ = {{T, n + 4}, {T, n + 4}};
                } else {
                    context.outputTensorDesc_[3].shape_ = {{B, S, n + 4}, {B, S, n + 4}};
                }
                break;

            // 异常 dtype
            case ErrorType::GRAD_OUTPUT_DTYPE_FLOAT:
                context.inputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
                break;
            case ErrorType::X_DTYPE_DIFF_GRAD_OUTPUT:
                context.inputTensorDesc_[1].dtype_ = ge::DT_FLOAT;
                break;
            case ErrorType::H_RES_DTYPE_NOT_FP32:
                context.inputTensorDesc_[2].dtype_ = ge::DT_FLOAT16;
                break;
            case ErrorType::H_OUT_DTYPE_DIFF_GRAD_OUTPUT:
                context.inputTensorDesc_[3].dtype_ = ge::DT_FLOAT;
                break;
            case ErrorType::H_POST_DTYPE_NOT_FP32:
                context.inputTensorDesc_[4].dtype_ = ge::DT_FLOAT16;
                break;
            case ErrorType::GRAD_X_DTYPE_DIFF_GRAD_OUTPUT:
                context.outputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
                break;
            case ErrorType::GRAD_H_RES_DTYPE_NOT_FP32:
                context.outputTensorDesc_[1].dtype_ = ge::DT_FLOAT16;
                break;
            case ErrorType::GRAD_H_OUT_DTYPE_DIFF_GRAD_OUTPUT:
                context.outputTensorDesc_[2].dtype_ = ge::DT_FLOAT;
                break;
            case ErrorType::GRAD_H_POST_DTYPE_NOT_FP32:
                context.outputTensorDesc_[3].dtype_ = ge::DT_FLOAT16;
                break;

            // 不支持的 shape
            case ErrorType::BS_GT_512K:
            case ErrorType::T_GT_512K:
            case ErrorType::N_NOT_4_6_8:
            case ErrorType::D_GT_24K:
                break;

            default:
                break;
        }
    }

    /**
     * @brief TNN模式测试夹具
     */
    static void TestMhcPostBackwardTND(const string &caseName, uint32_t T, uint32_t n, uint32_t D, ge::DataType dtype,
                                       const std::string &socVersion, ErrorType errorType = ErrorType::NONE)
    {
        (void)caseName;
        gert::TilingContextPara tilingContextPara = BuildTilingContextTND(T, n, D, dtype, socVersion);

        if (errorType != ErrorType::NONE) {
            ApplyErrorModifications(tilingContextPara, errorType, true, n, D, T, 0, 0);
        }

        int64_t expectTilingKey = 0;
        string expectTilingDataStr = "";
        ge::graphStatus expectStatus = (errorType == ErrorType::NONE) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        std::vector<size_t> expectWorkspaces =
            (errorType == ErrorType::NONE) ? std::vector<size_t>{16777216} : std::vector<size_t>{0};

        ExecuteTestCase(tilingContextPara, expectStatus, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                        TilingData2Str<int64_t>);
    }

    /**
     * @brief BSNN模式测试夹具
     */
    static void TestMhcPostBackwardBSND(const string &caseName, uint32_t B, uint32_t S, uint32_t n, uint32_t D,
                                        ge::DataType dtype, const std::string &socVersion,
                                        ErrorType errorType = ErrorType::NONE)
    {
        (void)caseName;
        gert::TilingContextPara tilingContextPara = BuildTilingContextBSND(B, S, n, D, dtype, socVersion);

        if (errorType != ErrorType::NONE) {
            ApplyErrorModifications(tilingContextPara, errorType, false, n, D, 0, B, S);
        }

        int64_t expectTilingKey = 0;
        string expectTilingDataStr = "";
        ge::graphStatus expectStatus = (errorType == ErrorType::NONE) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        std::vector<size_t> expectWorkspaces =
            (errorType == ErrorType::NONE) ? std::vector<size_t>{16777216} : std::vector<size_t>{0};

        ExecuteTestCase(tilingContextPara, expectStatus, expectTilingKey, expectTilingDataStr, expectWorkspaces, 0,
                        TilingData2Str<int64_t>);
    }

    template <typename T>
    static string TilingData2Str(void *buf, size_t size)
    {
        string result;
        const T *data = reinterpret_cast<const T *>(buf);
        size_t len = size / sizeof(T);
        for (size_t i = 0; i < len; i++) {
            result += std::to_string(data[i]);
            result += " ";
        }
        return result;
    }
};
