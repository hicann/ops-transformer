/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_cube.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_vec.h"

namespace QkvRmsNormRopeCacheWithKScale {

template <uint32_t QKV_LAYOUT, uint32_t Q_OUT_LAYOUT, uint32_t ROPE_MODE, uint32_t K_QUANT_MODE, uint32_t Q_QUANT_MODE>
class QkvRmsNormRopeCacheWithKScaleBasicBlock {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        if ASCEND_IS_AIC {
            InitAic(tensors, tilingData);
        } else {
            InitAiv(tensors, tilingData);
        }
    }

    __aicore__ inline void PrepareBeforeLoop(uint64_t coreTokenBegin, uint64_t coreTokenEnd)
    {
        if ASCEND_IS_AIC {
            cube_.PrepareRotationBeforeLoop();
            for (uint32_t i = 0U; i < A_ROT_L1_BUFFER_NUM; ++i) {
                SetAicMte1ToAivMte3AConsumed();
            }
        } else {
            vec_.PrepareBeforeLoop(coreTokenBegin, coreTokenEnd);
            if constexpr (IS_MROPE) {
                for (uint32_t i = 0U; i < MROPE_K_ROTATION_BUFFER_NUM; ++i) {
                    SetAivVToAicFixMropeKConsumed();
                }
            } else {
                SetAivMte3ToAicFixOutputConsumed();
            }
        }
    }

    __aicore__ inline void ComputeTile(TileParam &tile, const TileParam &lastTile, uint64_t tileCount, bool isLastTile)
    {
        if ASCEND_IS_AIC {
            ComputeTileAic(tile, isLastTile);
        } else {
            ComputeTileAiv(tile, lastTile, tileCount, isLastTile);
        }
    }

    __aicore__ inline void End(const TileParam &lastTile, uint64_t tileCount)
    {
        if ASCEND_IS_AIC {
            EndAic();
        } else {
            EndAiv(lastTile, tileCount);
        }
    }

private:
    __aicore__ inline void ResetBufferState()
    {
        aRotL1BufferUseId_ = 0U;
        outputBufferUseId_ = 0U;
        kRotationBufferUseId_ = 0U;
    }

    __aicore__ inline void InitAic(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        ResetBufferState();
        aRotL1Pool_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_A_ROT_L1_POOL_OFFSET,
                                              QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS);
        if constexpr (IS_MROPE) {
            kRotationDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_OFFSET,
                                                    QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_FLOAT_ELEMENTS);
        } else {
            outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        }
        cube_.Init(tensors, tilingData);
        cube_.InitIntraCoreEvents();
    }

    __aicore__ inline void InitAiv(const GlobalTensors &tensors,
                                   const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        ResetBufferState();
        aRotL1Pool_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_A_ROT_L1_POOL_OFFSET,
                                              QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS);
        if constexpr (IS_MROPE) {
            outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
            kRotationDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_OFFSET,
                                                    QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_FLOAT_ELEMENTS);
        } else {
            outputDbPoolUb_ = LocalTensor<float>(TPosition::LCM, QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET,
                                                 QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        }
        vec_.Init(tensors, tilingData);
        vec_.InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTileAic(const TileParam &tile, bool isLastTile)
    {
        const uint32_t aRotL1BufferId = static_cast<uint32_t>(aRotL1BufferUseId_ % A_ROT_L1_BUFFER_NUM);
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        uint32_t outputBufferId;
        if constexpr (IS_MROPE) {
            outputBufferId = static_cast<uint32_t>(kRotationBufferUseId_ % MROPE_K_ROTATION_BUFFER_NUM);
            WaitAivVToAicFixMropeKConsumed();
        } else {
            // AIV uses the output pool once for the current preprocess and once
            // for the previous tile's postprocess; count logical uses, not tiles.
            outputBufferUseId_ += isLastTile ? 1U : ROPE_OUTPUT_BUFFER_USES_PER_TILE;
            outputBufferId = static_cast<uint32_t>(outputBufferUseId_ % ROPE_OUTPUT_BUFFER_NUM);
            WaitAivMte3ToAicFixOutputConsumed();
        }

        WaitAivMte3ToAicMte1AReady();
        if constexpr (IS_MROPE) {
            cube_.ComputeTile(
                tile, aRotL1Nz,
                kRotationDbPoolUb_[outputBufferId * QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS]);
        } else {
            cube_.ComputeTile(tile, aRotL1Nz,
                              outputDbPoolUb_[outputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS]);
        }
        SetAicMte1ToAivMte3AConsumed();
        SetAicFixToAivOutputReady();
        if constexpr (IS_MROPE) {
            // The output-ready flag is ordered after the K Fixpipe and before
            // the independent Q L0C-to-GM Fixpipe.
            cube_.FixpipeQToGm(tile);
            ++kRotationBufferUseId_;
        }
        ++aRotL1BufferUseId_;
    }

    __aicore__ inline void ComputeTileAiv(TileParam &tile, const TileParam &lastTile, uint64_t tileCount,
                                          bool isLastTile)
    {
        const uint32_t aRotL1BufferId = static_cast<uint32_t>(aRotL1BufferUseId_ % A_ROT_L1_BUFFER_NUM);
        const LocalTensor<bfloat16_t> aRotL1Nz =
            aRotL1Pool_[aRotL1BufferId * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        if constexpr (IS_MROPE) {
            const uint32_t outputBufferId = static_cast<uint32_t>(outputBufferUseId_ % MROPE_OUTPUT_BUFFER_NUM);

            const LocalTensor<float> outputUb =
                outputDbPoolUb_[outputBufferId * QKV_K_SCALE_MROPE_COMPACT_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
            WaitAicMte1ToAivMte3AConsumed();
            vec_.ComputeTile(tile, aRotL1Nz, outputUb, outputBufferId);
            SetAivMte3ToAicMte1AReady();

            if (tileCount > 0U) {
                const uint32_t kRotationBufferId =
                    static_cast<uint32_t>(kRotationBufferUseId_ % MROPE_K_ROTATION_BUFFER_NUM);
                WaitAicFixToAivOutputReady();
                vec_.PostprocessMropeK(
                    lastTile, kRotationDbPoolUb_[kRotationBufferId *
                                                 QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS]);
                SetAivVToAicFixMropeKConsumed();
                ++kRotationBufferUseId_;
            }
            ++outputBufferUseId_;
        } else {
            const uint32_t outputBufferId = static_cast<uint32_t>(outputBufferUseId_ % ROPE_OUTPUT_BUFFER_NUM);
            WaitAicMte1ToAivMte3AConsumed();
            vec_.ComputeTile(tile, aRotL1Nz,
                             outputDbPoolUb_[outputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                             outputBufferId);
            ++outputBufferUseId_;
            if (isLastTile && tileCount > 0U) {
                SetAivMte3ToAicFixOutputConsumed();
            }
            SetAivMte3ToAicMte1AReady();
            if (tileCount > 0U) {
                const uint32_t lastOutputBufferId = static_cast<uint32_t>(outputBufferUseId_ % ROPE_OUTPUT_BUFFER_NUM);
                WaitAicFixToAivOutputReady();
                vec_.PostprocessRopeQk(
                    lastTile, outputDbPoolUb_[lastOutputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                    lastOutputBufferId);
                if (!isLastTile) {
                    SetAivMte3ToAicFixOutputConsumed();
                }
                ++outputBufferUseId_;
            }
        }
        ++aRotL1BufferUseId_;
    }

    __aicore__ inline void EndAic()
    {
        if constexpr (IS_MROPE) {
            for (uint32_t i = 0U; i < MROPE_K_ROTATION_BUFFER_NUM; ++i) {
                WaitAivVToAicFixMropeKConsumed();
            }
        } else {
            WaitAivMte3ToAicFixOutputConsumed();
        }
        cube_.EndIntraCoreEvents();
    }

    __aicore__ inline void EndAiv(const TileParam &lastTile, uint64_t tileCount)
    {
        if constexpr (IS_MROPE) {
            if (tileCount > 0U) {
                const uint32_t kRotationBufferId =
                    static_cast<uint32_t>(kRotationBufferUseId_ % MROPE_K_ROTATION_BUFFER_NUM);
                WaitAicFixToAivOutputReady();
                vec_.PostprocessMropeK(
                    lastTile, kRotationDbPoolUb_[kRotationBufferId *
                                                 QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS]);
                SetAivVToAicFixMropeKConsumed();
                ++kRotationBufferUseId_;
            }
        } else if (tileCount > 0U) {
            const uint32_t lastOutputBufferId = static_cast<uint32_t>(outputBufferUseId_ % ROPE_OUTPUT_BUFFER_NUM);
            WaitAicFixToAivOutputReady();
            vec_.PostprocessRopeQk(lastTile,
                                   outputDbPoolUb_[lastOutputBufferId * QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS],
                                   lastOutputBufferId);
            SetAivMte3ToAicFixOutputConsumed();
            ++outputBufferUseId_;
        }
        for (uint32_t i = 0U; i < A_ROT_L1_BUFFER_NUM; ++i) {
            WaitAicMte1ToAivMte3AConsumed();
        }
        vec_.EndIntraCoreEvents();
    }

    __aicore__ inline void SetAivMte3ToAicMte1AReady()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_A_READY);
    }

    __aicore__ inline void WaitAivMte3ToAicMte1AReady()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_READY +
                                                                                QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_READY);
    }

    __aicore__ inline void SetAicMte1ToAivMte3AConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_CONSUMED +
                                                                               QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void WaitAicMte1ToAivMte3AConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_A_CONSUMED);
    }

    __aicore__ inline void SetAicFixToAivOutputReady()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_FIX_OUTPUT_READY +
                                                                              QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_FIX_OUTPUT_READY);
    }

    __aicore__ inline void WaitAicFixToAivOutputReady()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_FIX_OUTPUT_READY);
    }

    __aicore__ inline void SetAivMte3ToAicFixOutputConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(SYNC_FIX_OUTPUT_CONSUMED);
    }

    __aicore__ inline void WaitAivMte3ToAicFixOutputConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_FIX_OUTPUT_CONSUMED +
                                                                               QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_FIX_OUTPUT_CONSUMED);
    }

    __aicore__ inline void SetAivVToAicFixMropeKConsumed()
    {
        AscendC::CrossCoreSetFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_V>(SYNC_MROPE_K_CONSUMED);
    }

    __aicore__ inline void WaitAivVToAicFixMropeKConsumed()
    {
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_CONSUMED +
                                                                               QKV_K_SCALE_AIV1_FLAG_OFFSET);
        AscendC::CrossCoreWaitFlag<QKV_K_SCALE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(SYNC_MROPE_K_CONSUMED);
    }

    static constexpr bool IS_MROPE = ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE;
    static constexpr uint32_t A_ROT_L1_BUFFER_NUM =
        QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS / QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS;
    static constexpr uint32_t ROPE_OUTPUT_BUFFER_NUM =
        QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS / QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS;
    static constexpr uint32_t ROPE_OUTPUT_BUFFER_USES_PER_TILE = 2U;
    static constexpr uint32_t MROPE_OUTPUT_BUFFER_NUM = QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_FLOAT_ELEMENTS /
                                                        QKV_K_SCALE_MROPE_COMPACT_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS;
    static constexpr uint32_t MROPE_K_ROTATION_BUFFER_NUM =
        QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_FLOAT_ELEMENTS /
        QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS;

    uint64_t aRotL1BufferUseId_;
    uint64_t outputBufferUseId_;
    uint64_t kRotationBufferUseId_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<float> outputDbPoolUb_;
    LocalTensor<float> kRotationDbPoolUb_;
    QkvRmsNormRopeCacheWithKScaleCube<ROPE_MODE> cube_;
    QkvRmsNormRopeCacheWithKScaleVec<QKV_LAYOUT, Q_OUT_LAYOUT, ROPE_MODE, K_QUANT_MODE, Q_QUANT_MODE> vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
