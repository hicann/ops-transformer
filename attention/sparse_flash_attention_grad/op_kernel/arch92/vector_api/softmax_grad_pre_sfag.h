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
 * \file softmax_grad_pre_sfag.h
 * \brief D=512 softmaxGrad VF。pre 主流程在 sparse_flash_attention_grad_pre_regbase.h。
 */
#ifndef SOFTMAX_GRAD_PRE_SFAG_H
#define SOFTMAX_GRAD_PRE_SFAG_H
#include "./vf_softmax_grad_front_cast_sfag.h"

namespace SfagPreApi {
using namespace AscendC;
constexpr uint32_t SFMG_UB_ROW_STRIDE = 512;
} // namespace SfagPreApi

#endif // SOFTMAX_GRAD_PRE_SFAG_H
