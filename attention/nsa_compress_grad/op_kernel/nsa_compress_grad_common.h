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
 * \file compress_grad_common.h
 * \brief compress_grad head file
 */

#ifndef NSA_COMPRESS_GRAD_COMMON_H
#define NSA_COMPRESS_GRAD_COMMON_H

#include <cstdint>
#include <kernel_operator.h>

namespace NsaCompressGrad {

using namespace AscendC;

__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t CompressBlkNum(int64_t inputLen, int64_t blkSize, int64_t blkStride)
{
    if (inputLen < blkSize) {
        return 0;
    }
    if (blkStride != 0) {
        return (inputLen - blkSize) / blkStride + 1;
    }
    return 0;
}

} // namespace NsaCompressGrad
#endif
