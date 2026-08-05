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
 * \file flash_attn_common_def.h
 * \brief
 */

#ifndef FLASH_ATTN_UTILS_H_
#define FLASH_ATTN_UTILS_H_

#include "kernel_operator_list_tensor_intf.h"

using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;


namespace fa_base_vector {

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

}

static constexpr uint32_t FA_BYTE_BLOCK = 32;
static constexpr uint32_t FA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);

#endif  // FLASH_ATTN_COMMON_DEF_H_