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
 * \file test_ai_infra_mhc_grad_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "test_mhc_post_backward_fixture.h"

// ======================== 带fixture的写法 ===========================
// ======================== 基础用例 网络shape ===========================
TEST_F(MhcPostBackwardTilingTest, case_normal_network_b1_s1k_n4_d5120_bf16)
{
    TestMhcPostBackwardBSND("case_normal_network_b1_s1k_n4_d5120_bf16", 1, 1024, 4, 5120, ge::DT_BF16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_network_b1_s2k_n4_d2560_bf16)
{
    TestMhcPostBackwardBSND("case_normal_network_b1_s2k_n4_d2560_bf16", 1, 2048, 4, 2560, ge::DT_BF16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_network_b1_s4k_n4_d2560_bf16)
{
    TestMhcPostBackwardBSND("case_normal_network_b1_s4k_n4_d2560_bf16", 1, 4096, 4, 2560, ge::DT_BF16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_network_b1_s1k_n4_d5120_fp16)
{
    TestMhcPostBackwardBSND("case_normal_network_b1_s1k_n4_d5120_fp16", 1, 1024, 4, 5120, ge::DT_FLOAT16,
                            kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_network_b1_s2k_n4_d2560_fp16)
{
    TestMhcPostBackwardBSND("case_normal_network_b1_s2k_n4_d2560_fp16", 1, 2048, 4, 2560, ge::DT_FLOAT16,
                            kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_network_b1_s4k_n4_d2560_fp16)
{
    TestMhcPostBackwardBSND("case_normal_network_b1_s4k_n4_d2560_fp16", 1, 4096, 4, 2560, ge::DT_FLOAT16,
                            kSocVersion950);
}

// ======================== 典型泛化 TND ===========================
TEST_F(MhcPostBackwardTilingTest, case_normal_typical_t1k_n4_d5120_bf16)
{
    TestMhcPostBackwardTND("case_normal_typical_t1k_n4_d5120_bf16", 1024, 4, 5120, ge::DT_BF16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_typical_t2k_n4_d2560_bf16)
{
    TestMhcPostBackwardTND("case_normal_typical_t2k_n4_d2560_bf16", 2048, 4, 2560, ge::DT_BF16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_typical_t4k_n4_d2560_bf16)
{
    TestMhcPostBackwardTND("case_normal_typical_t4k_n4_d2560_bf16", 4096, 4, 2560, ge::DT_BF16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_typical_t1k_n4_d5120_fp16)
{
    TestMhcPostBackwardTND("case_normal_typical_t1k_n4_d5120_fp16", 1024, 4, 5120, ge::DT_FLOAT16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_typical_t2k_n4_d2560_fp16)
{
    TestMhcPostBackwardTND("case_normal_typical_t2k_n4_d2560_fp16", 2048, 4, 2560, ge::DT_FLOAT16, kSocVersion950);
}

TEST_F(MhcPostBackwardTilingTest, case_normal_typical_t4k_n4_d2560_fp16)
{
    TestMhcPostBackwardTND("case_normal_typical_t4k_n4_d2560_fp16", 4096, 4, 2560, ge::DT_FLOAT16, kSocVersion950);
}

// ======================== 异常用例 ===========================
// ======================== 空指针/空tensor ===========================
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_output_null_desc)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_grad_output_null_desc", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::GRAD_OUTPUT_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_x_null_desc)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_x_null_desc", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::X_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_res_null_desc)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_h_res_null_desc", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_RES_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_out_null_desc)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_h_out_null_desc", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::H_OUT_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_post_null_desc)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_h_post_null_desc", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_POST_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_x_null_desc)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_x_null_desc", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_X_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_h_res_null_desc)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_h_res_null_desc", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_H_RES_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_h_out_null_desc)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_h_out_null_desc", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_H_OUT_NULL_DESC);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_h_post_null_desc)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_h_post_null_desc", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_H_POST_NULL_DESC);
}

// ======================== 异常 shape ===========================
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_output_2d)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_output_2d", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_OUTPUT_2D);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_output_5d)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_grad_output_5d", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::GRAD_OUTPUT_5D);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_x_diff_grad_output)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_x_diff_grad_output", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::X_SHAPE_DIFF);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_res_diff_bs)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_h_res_diff_bs", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::H_RES_SHAPE_DIFF_BS);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_res_diff_t)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_h_res_diff_t", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_RES_SHAPE_DIFF_T);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_res_diff_nn)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_h_res_diff_nn", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_RES_SHAPE_DIFF_NN);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_out_not_bsd)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_h_out_not_bsd", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::H_OUT_SHAPE_NOT_BSD);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_out_not_td)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_h_out_not_td", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_OUT_SHAPE_NOT_TD);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_post_not_bsn)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_h_post_not_bsn", 1, 512, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::H_POST_SHAPE_NOT_BSN);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_h_out_not_tn)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_h_out_not_tn", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_POST_SHAPE_NOT_TN);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_x_diff_grad_output)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_x_diff_grad_output", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_X_SHAPE_DIFF_GRAD_OUTPUT);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_h_res_diff_h_res)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_grad_h_res_diff_h_res", 2, 128, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::GRAD_H_RES_SHAPE_DIFF_H_RES);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_h_out_diff_h_out)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_grad_h_out_diff_h_out", 2, 128, 4, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::GRAD_H_OUT_SHAPE_DIFF_H_OUT);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_grad_h_post_diff_h_post)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_grad_h_post_diff_h_post", 128, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_H_POST_SHAPE_DIFF_H_POST);
}

// ======================== 异常 dtype ===========================
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_grad_output)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_grad_output", 512, 4, 394, ge::DT_FLOAT, kSocVersion950,
                           ErrorType::GRAD_OUTPUT_DTYPE_FLOAT);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_x_diff_grad_output)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_x_diff_grad_output", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::X_DTYPE_DIFF_GRAD_OUTPUT);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_h_out_diff_grad_output)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_h_out_diff_grad_output", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_OUT_DTYPE_DIFF_GRAD_OUTPUT);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_grad_x_diff_grad_output)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_grad_x_diff_grad_output", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_X_DTYPE_DIFF_GRAD_OUTPUT);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_grad_h_out_diff_grad_output)
{
    TestMhcPostBackwardBSND("case_error_wrong_dtype_grad_h_out_diff_grad_output", 2, 512, 4, 394, ge::DT_FLOAT16,
                            kSocVersion950, ErrorType::GRAD_H_OUT_DTYPE_DIFF_GRAD_OUTPUT);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_h_res_not_fp32)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_h_res_not_fp32", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_RES_DTYPE_NOT_FP32);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_h_post_not_fp32)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_h_post_not_fp32", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::H_POST_DTYPE_NOT_FP32);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_grad_h_res_not_fp32)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_grad_h_res_not_fp32", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_H_RES_DTYPE_NOT_FP32);
}

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_dtype_grad_h_post_not_fp32)
{
    TestMhcPostBackwardTND("case_error_wrong_dtype_grad_h_post_not_fp32", 512, 4, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::GRAD_H_POST_DTYPE_NOT_FP32);
}

// ======================== 不支持的 shape ===========================

TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_tnd_n_not_4_6_8)
{
    TestMhcPostBackwardTND("case_error_wrong_shape_tnd_n_not_4_6_8", 128, 5, 394, ge::DT_BF16, kSocVersion950,
                           ErrorType::N_NOT_4_6_8);
}
TEST_F(MhcPostBackwardTilingTest, case_error_wrong_shape_bsnd_n_not_4_6_8)
{
    TestMhcPostBackwardBSND("case_error_wrong_shape_bsnd_n_not_4_6_8", 1, 512, 7, 394, ge::DT_BF16, kSocVersion950,
                            ErrorType::N_NOT_4_6_8);
}
