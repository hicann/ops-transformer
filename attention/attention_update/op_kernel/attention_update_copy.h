/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_UPDATE_COPY_H_
#define ATTENTION_UPDATE_COPY_H_
#include "kernel_operator.h"

namespace AttentionUpdateCopy {
template <typename T>
__aicore__ inline void CopyRows(const AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
                                uint64_t firstRow, uint64_t rows, uint64_t d, uint64_t rowStride, uint64_t dAlign)
{
    constexpr uint64_t MAX_DMA_ROWS = 4095;
    constexpr uint64_t MAX_DMA_GAP = 0xffffffffULL;
    const uint64_t gapBytes = (rowStride - d) * sizeof(T);
    uint64_t done = 0;
    while (done < rows) {
        uint64_t batch = rows - done;
        if (batch > MAX_DMA_ROWS) {
            batch = MAX_DMA_ROWS;
        }
        // A large stride is valid; one-row copies avoid narrowing the DMA gap.
        if (gapBytes > MAX_DMA_GAP) {
            batch = 1;
        }
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(batch), static_cast<uint32_t>(d * sizeof(T)),
                                          static_cast<uint32_t>(batch > 1 ? gapBytes : 0), 0, 0};
        AscendC::DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst[done * dAlign], src[(firstRow + done) * rowStride], params, padding);
        done += batch;
    }
}
} // namespace AttentionUpdateCopy
#endif
