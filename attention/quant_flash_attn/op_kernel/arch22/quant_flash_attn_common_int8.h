/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_COMMON_INT8_H_
#define QUANT_FLASH_ATTN_COMMON_INT8_H_
#include "kernel_operator.h"
#include "quant_flash_attn_tiling_data_int8.h"
#include "quant_flash_attn_workload_int8.h"
namespace BaseApi {
constexpr uint32_t QFA_INT8_DMA_BLOCK_BYTES = 32U;
constexpr float QFA_FP16_MAX_FINITE = 65504.0f;

// Dispatch policies preserve the tuned GS1/S1G traversal and bounded scale cache.
enum class Int8GqaPath {
    BNSD,
    BSND,
    TND
};
__aicore__ inline uint64_t QfaRow(uint32_t layout, uint32_t batch, uint32_t head, uint32_t token, uint32_t heads,
                                  uint32_t seq, uint32_t start)
{
    if (layout == Int8GqaTnd::QFA_LAYOUT_BNSD)
        return (static_cast<uint64_t>(batch) * heads + head) * seq + token;
    if (layout == Int8GqaTnd::QFA_LAYOUT_BSND)
        return (static_cast<uint64_t>(batch) * seq + token) * heads + head;
    return (static_cast<uint64_t>(start) + token) * heads + head;
}
template <Int8GqaPath PATH>
struct QuantFlashAttnLayoutInt8 {
    __aicore__ static inline bool IsSg(uint32_t layout)
    {
        if constexpr (PATH == Int8GqaPath::BNSD)
            return false;
        if constexpr (PATH == Int8GqaPath::BSND)
            return layout == Int8GqaTnd::QFA_LAYOUT_BSND;
        return layout != Int8GqaTnd::QFA_LAYOUT_BNSD;
    }

    __aicore__ static inline bool QfaUseKScaleCache(const QuantFlashAttnTilingDataInt8& t)
    {
        if constexpr (PATH == Int8GqaPath::BNSD) {
            return false;
        } else {
            return t.layoutQ != Int8GqaTnd::QFA_LAYOUT_BNSD && t.layoutKv != Int8GqaTnd::QFA_LAYOUT_BNSD &&
                   Int8GqaTnd::QfaUseKScaleCache(t.seqLenQ, t.seqLenKv, t.headDim,
                                                 static_cast<uint64_t>(t.groupSize) * t.seqLenQ, t.blockM, t.blockN);
        }
    }
    __aicore__ static inline uint32_t QfaQueryToken(const QuantFlashAttnTilingDataInt8& t, uint32_t row)
    {
        return IsSg(t.layoutQ) ? row / t.groupSize : row % t.seqLenQ;
    }
    __aicore__ static inline uint32_t QfaQueryHead(const QuantFlashAttnTilingDataInt8& t, uint32_t kv, uint32_t row)
    {
        return kv * t.groupSize + (IsSg(t.layoutQ) ? row % t.groupSize : row / t.seqLenQ);
    }
    __aicore__ static inline uint32_t QfaQueryRun(const QuantFlashAttnTilingDataInt8& t, uint32_t row,
                                                  uint32_t remaining)
    {
        const uint32_t run = IsSg(t.layoutQ) ? t.groupSize - row % t.groupSize : t.seqLenQ - row % t.seqLenQ;
        return run < remaining ? run : remaining;
    }
    __aicore__ static inline void QfaCopySgQuery(const AscendC::LocalTensor<int8_t>& dst,
                                                 const AscendC::GlobalTensor<int8_t>& src,
                                                 const QuantFlashAttnTilingDataInt8& t, uint32_t batch, uint32_t kv,
                                                 uint32_t start, uint32_t rows, uint32_t alignedRows)
    {
        AscendC::Nd2NzParams p(1U, 0U, t.headDim, 0U, t.headDim, alignedRows, 1U, 0U);

        const uint64_t initial = QfaRow(t.layoutQ, batch, QfaQueryHead(t, kv, start), QfaQueryToken(t, start),
                                        t.numHeadsQ, t.seqLenQ, t.qTokenStart) *
                                 t.headDim;
        if (t.numHeadsKv == 1U || t.groupSize == 1U) {
            p.nValue = rows;
            const uint64_t stride = t.numHeadsKv == 1U ? t.headDim : static_cast<uint64_t>(t.numHeadsQ) * t.headDim;
            if (stride > UINT16_MAX) {
                p.nValue = 1U;
                p.srcDValue = t.headDim;
                for (uint32_t row = 0; row < rows; ++row)
                    AscendC::DataCopy(dst[row * QFA_INT8_DMA_BLOCK_BYTES], src[initial + row * stride], p);
            } else {
                p.srcDValue = stride;
                AscendC::DataCopy(dst, src[initial], p);
            }
            return;
        }
        uint32_t done = QfaQueryRun(t, start, rows);
        p.nValue = done;
        AscendC::DataCopy(dst, src[initial], p);
        const uint64_t tokenStride = static_cast<uint64_t>(t.numHeadsQ) * t.headDim;
        uint64_t offset = initial + tokenStride - (start % t.groupSize) * t.headDim;
        const uint32_t groups = (rows - done) / t.groupSize;
        if (groups > 0U) {
            p.nValue = t.groupSize;
            if (tokenStride <= UINT16_MAX) {
                p.ndNum = groups;
                p.srcNdMatrixStride = tokenStride;
                p.dstNzMatrixStride = t.groupSize * QFA_INT8_DMA_BLOCK_BYTES;
                AscendC::DataCopy(dst[done * QFA_INT8_DMA_BLOCK_BYTES], src[offset], p);
            } else {
                for (uint32_t i = 0; i < groups; ++i)
                    AscendC::DataCopy(dst[(done + i * t.groupSize) * QFA_INT8_DMA_BLOCK_BYTES],
                                      src[offset + i * tokenStride], p);
            }
            done += groups * t.groupSize;
            offset += groups * tokenStride;
        }
        if (done < rows) {
            p.ndNum = 1;
            p.srcNdMatrixStride = 0;
            p.dstNzMatrixStride = 0;
            p.nValue = rows - done;
            AscendC::DataCopy(dst[done * QFA_INT8_DMA_BLOCK_BYTES], src[offset], p);
        }
    }
};
// Complete a local pipeline dependency without allocating a new event.
template <AscendC::HardEvent EVENT>
__aicore__ inline void QfaSyncEvent(int32_t id)
{
    AscendC::SetFlag<EVENT>(id);
    AscendC::WaitFlag<EVENT>(id);
}

// P8 is in [0, 127], but signed INT8 V can be -128. Therefore
// |sum_j P8_j * V8_j * vDescale / pScale|
//     <= tileN * 127 * 128 * |vDescale / pScale|.
// The Fixpipe output is FP16; divide by the smallest power of two that
// bounds that worst case, then compensate once at final normalization.
__aicore__ inline float PvFixpipeCompensation(float vDescale, float pScale, uint32_t tileN)
{
    constexpr float maxQuantizedProbability = 127.0f;
    constexpr float maxValueMagnitude = 128.0f;
    const float scaledDescale = vDescale * (maxQuantizedProbability / pScale);
    const float magnitude = scaledDescale < 0.0f ? -scaledDescale : scaledDescale;
    // AICore rejects unsigned-to-float scalar casts. Keep each bound in FP32 as N * max |V8|.
    constexpr uint32_t wideKvTileTokens = 768U;
    constexpr float maxProductsN256 = 256.0f * maxValueMagnitude;
    constexpr float maxProductsN512 = 512.0f * maxValueMagnitude;
    constexpr float maxProductsN768 = 768.0f * maxValueMagnitude;
    constexpr float maxProductsN1024 = 1024.0f * maxValueMagnitude;
    float worstTileProducts = maxProductsN1024;
    if (tileN <= Int8GqaTnd::QFA_MEDIUM_KV_TILE_TOKENS) {
        worstTileProducts = maxProductsN256;
    } else if (tileN <= Int8GqaTnd::QFA_DEFAULT_KV_TILE_TOKENS) {
        worstTileProducts = maxProductsN512;
    } else if (tileN <= wideKvTileTokens) {
        worstTileProducts = maxProductsN768;
    }
    const float safeScale = QFA_FP16_MAX_FINITE / worstTileProducts;
    float compensation = 1.0f;
    // A 30-step cap covers all useful FP16-output scales. Larger descales can already overflow the final FP16
    // output for ordinary V inputs.
    constexpr uint32_t maxCompensationDoublings = 30U;
    for (uint32_t i = 0; i < maxCompensationDoublings && magnitude > safeScale * compensation; ++i) {
        compensation *= 2.0f;
    }
    return compensation;
}

// FixpipeParamsV220.deqScalar is uint64_t. DEQF16 interprets its low 32 bits as the FP32 scale, so keep v_descale
// in FP32 until this encoding.
__aicore__ inline uint64_t PvFixpipeDeqBits(float vDescale, float pScale, uint32_t tileN)
{
    const float compensation = PvFixpipeCompensation(vDescale, pScale, tileN);
    float scale = (vDescale / pScale) / compensation;
    return static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(&scale));
}

} // namespace BaseApi
#endif
