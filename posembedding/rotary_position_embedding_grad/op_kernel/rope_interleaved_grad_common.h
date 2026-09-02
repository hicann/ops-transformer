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
 * \file rope_interleaved_grad_common.h
 * \brief
 */
#ifndef ROPE_INTERLEAVED_GRAD_COMMON_H
#define ROPE_INTERLEAVED_GRAD_COMMON_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t LOG_FP32_SIZE = 2;
constexpr int32_t FP32_DIVIDE_FP16 = 2;
constexpr int32_t LOG_BLOCK_FP32_NUM = 3;
constexpr int32_t BLOCK_FP32_NUM = 8;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t MINI_HEADIM_NUM = 32;
constexpr int32_t MASK_FP32 = 64;
constexpr int64_t MASK_INT32 = 64;
constexpr int32_t MASK_FP16 = 128;

#endif // ROTATE_INTERLEAVED_GRAD_COMMON_H
