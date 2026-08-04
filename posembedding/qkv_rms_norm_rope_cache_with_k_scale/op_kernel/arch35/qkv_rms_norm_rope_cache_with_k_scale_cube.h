/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"

namespace QkvRmsNormRopeCacheWithKScale {

constexpr FixpipeConfig QKV_K_SCALE_FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
constexpr FixpipeConfig QKV_K_SCALE_FIXPIPE_ROW_MAJOR_GM = {CO2Layout::ROW_MAJOR, false};
constexpr uint8_t QKV_K_SCALE_FIXPIPE_DUAL_DST_SPLIT_M = 1;

template <uint32_t ROPE_MODE>
class QkvRmsNormRopeCacheWithKScaleCube {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        qHeadNum_ = tilingData->qHeadNum;
        kvHeadNum_ = tilingData->kvHeadNum;
        l0cBufferUseId_ = 0U;
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < AIC_L0A_BUFFER_NUM; ++bufferId) {
            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
        }
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < AIC_L0A_BUFFER_NUM; ++bufferId) {
            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
        }
    }

    __aicore__ inline void PrepareRotationBeforeLoop()
    {
        CopyRotationGmToL1Nz();
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
        LoadRotationToL0();
        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_ROTATION_READY));
        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_ROTATION_READY));
    }

    __aicore__ inline void ComputeTile(const TileParam &tile, const LocalTensor<bfloat16_t> &aRotL1Nz,
                                       LocalTensor<float> qkAfterCubeUb)
    {
        const uint32_t l0cBufferId = static_cast<uint32_t>(l0cBufferUseId_ % AIC_L0C_BUFFER_NUM);
        const uint64_t qRows = tile.cubeTokenSize * qHeadNum_;
        const uint64_t kRows = tile.cubeTokenSize * kvHeadNum_;
        const uint64_t qRowsPerAiv = tile.cubeHalfTokenSize * qHeadNum_;
        const uint64_t kL0cOffset = AlignUp(qRows, QKV_K_SCALE_NZ_C0) * QKV_K_SCALE_HEAD_DIM_D128;
        ComputePreprocessedQkToL0C(aRotL1Nz, qRows, kRows, l0cBufferId, kL0cOffset);
        if constexpr (!IS_MROPE) {
            FixpipeSegmentToAivUb(qRows, 0U, l0cBufferId, 0U, qkAfterCubeUb);
        }
        if constexpr (IS_MROPE) {
            FixpipeSegmentToAivUb(kRows, 0U, l0cBufferId, kL0cOffset, qkAfterCubeUb);
        } else {
            FixpipeSegmentToAivUb(kRows, qRowsPerAiv, l0cBufferId, kL0cOffset, qkAfterCubeUb);
        }
        ++l0cBufferUseId_;
    }

    __aicore__ inline void FixpipeQToGm(const TileParam &tile)
    {
        if constexpr (!IS_MROPE) {
            return;
        }

        const uint32_t l0cBufferId =
            static_cast<uint32_t>((l0cBufferUseId_ + AIC_L0C_BUFFER_NUM - 1U) % AIC_L0C_BUFFER_NUM);
        const uint64_t qRows = tile.cubeTokenSize * qHeadNum_;
        const uint64_t validQRows = tile.tokenSize * qHeadNum_;
        const uint64_t qOutBaseRow = tile.tokenOffset * qHeadNum_;
        FixpipeQSegmentToGm(validQRows, qOutBaseRow, qRows, l0cBufferId);
    }

private:
    static constexpr bool IS_MROPE = ROPE_MODE == QKV_K_SCALE_ROPE_MODE_MROPE;
    static constexpr uint32_t AIC_L0A_BUFFER_NUM = 2U;
    static constexpr uint32_t AIC_L0C_BUFFER_NUM = 2U;
    static constexpr uint32_t AIC_M_L0_TILE = 128U;
    static constexpr uint32_t AIC_L0A_BUFFER_ELEMENTS = 64U * QKV_K_SCALE_KIB / sizeof(bfloat16_t);
    static constexpr uint32_t AIC_L0B_BUFFER_ELEMENTS = 64U * QKV_K_SCALE_KIB / sizeof(bfloat16_t);
    static constexpr uint32_t AIC_L0C_BUFFER_ELEMENTS = 256U * QKV_K_SCALE_KIB / sizeof(float);
    static constexpr uint32_t AIC_L0A_ONE_BUFFER_ELEMENTS = AIC_L0A_BUFFER_ELEMENTS / AIC_L0A_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_ONE_BUFFER_ELEMENTS = AIC_L0C_BUFFER_ELEMENTS / AIC_L0C_BUFFER_NUM;
    static constexpr uint8_t AIC_UNIT_FLAG_CHECK_ONLY = 0b10;
    static constexpr uint8_t AIC_UNIT_FLAG_UPDATE = 0b11;
    static constexpr uint32_t EVT_MTE2_TO_MTE1_ROTATION_READY = 5U;
    static constexpr uint32_t EVT_MTE1_TO_MMAD_ROTATION_READY = 5U;
    static constexpr uint32_t EVT_MTE1_TO_MMAD_L0_BASE = 3U;
    static constexpr uint32_t EVT_MMAD_TO_MTE1_L0_BASE = 3U;

    __aicore__ inline void CopyRotationGmToL1Nz()
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1U;
        nd2nzParams.nValue = QKV_K_SCALE_HEAD_DIM_D128;
        nd2nzParams.dValue = QKV_K_SCALE_HEAD_DIM_D128;
        nd2nzParams.srcNdMatrixStride = 0U;
        nd2nzParams.srcDValue = QKV_K_SCALE_HEAD_DIM_D128;
        nd2nzParams.dstNzC0Stride = QKV_K_SCALE_HEAD_DIM_D128;
        nd2nzParams.dstNzNStride = 1U;
        nd2nzParams.dstNzMatrixStride = 0U;
        DataCopy(rotationL1Nz_, rotationGm_, nd2nzParams);
    }

    __aicore__ inline void LoadRotationToL0()
    {
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = 0U;
        loadDataParams.mStep = static_cast<uint16_t>(CeilDiv(QKV_K_SCALE_HEAD_DIM_D128, QKV_K_SCALE_NZ_C0));
        loadDataParams.kStep =
            static_cast<uint16_t>(QKV_K_SCALE_HEAD_DIM_D128 / (QKV_K_SCALE_BLOCK_BYTES / sizeof(bfloat16_t)));
        loadDataParams.srcStride = static_cast<int32_t>(CeilDiv(QKV_K_SCALE_HEAD_DIM_D128, QKV_K_SCALE_NZ_C0));
        loadDataParams.dstStride = static_cast<uint16_t>(CeilDiv(QKV_K_SCALE_HEAD_DIM_D128, QKV_K_SCALE_NZ_C0));
        loadDataParams.ifTranspose = true;
        LoadData(l0b_, rotationL1Nz_, loadDataParams);
    }

    __aicore__ inline void ComputePreprocessedQkToL0C(const LocalTensor<bfloat16_t> &aRotL1Nz, uint64_t qRows,
                                                      uint64_t kRows, uint32_t bufferId, uint64_t kL0cOffset)
    {
        if (qRows == 0U && kRows == 0U) {
            return;
        }

        const LocalTensor<bfloat16_t> kSegmentL1Nz = aRotL1Nz[NzMatrixElements(qRows)];
        uint32_t l0aBufferUseId = 0U;
        if constexpr (IS_MROPE) {
            // unitFlag pairs MMAD and Fixpipe in issue order; M-RoPE consumes K before Q.
            ComputeSegmentToL0C(kRows, kSegmentL1Nz, bufferId, kL0cOffset, l0aBufferUseId);
            ComputeSegmentToL0C(qRows, aRotL1Nz, bufferId, 0U, l0aBufferUseId);
        } else {
            ComputeSegmentToL0C(qRows, aRotL1Nz, bufferId, 0U, l0aBufferUseId);
            ComputeSegmentToL0C(kRows, kSegmentL1Nz, bufferId, kL0cOffset, l0aBufferUseId);
        }
    }

    __aicore__ inline void ComputeSegmentToL0C(uint64_t segmentRows, const LocalTensor<bfloat16_t> &segmentL1Nz,
                                               uint32_t l0cBufferId, uint64_t l0cOffset, uint32_t &l0aBufferUseId)
    {
        if (segmentRows == 0U) {
            return;
        }

        // Keep M intact so L0C keeps one continuous segment layout for the single L0C->UB Fixpipe.
        // Splitting by M would make each MMAD tile use its own M stride in L0C, so the segment is non-contiguous.
        const uint64_t alignedRows = AlignUp(segmentRows, QKV_K_SCALE_NZ_C0);
        const uint64_t maxDByL0a = (AIC_L0A_ONE_BUFFER_ELEMENTS / alignedRows / QKV_K_SCALE_NZ_C0) * QKV_K_SCALE_NZ_C0;
        const uint64_t dTile = MinU64(QKV_K_SCALE_HEAD_DIM_D128, maxDByL0a);
        for (uint64_t dStart = 0U; dStart < QKV_K_SCALE_HEAD_DIM_D128; dStart += dTile) {
            const uint64_t dSize = MinU64(dTile, QKV_K_SCALE_HEAD_DIM_D128 - dStart);
            const bool isFirstD = (dStart == 0U);
            const bool isLastD = (dStart + dSize >= QKV_K_SCALE_HEAD_DIM_D128);
            const uint32_t l0aBufferId = l0aBufferUseId % AIC_L0A_BUFFER_NUM;

            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + l0aBufferId));
            LoadAToL0(segmentRows, segmentRows, dStart, dSize, l0aBufferId, segmentL1Nz);
            SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + l0aBufferId));
            WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + l0aBufferId));
            ComputeMmad(segmentRows, dSize, isFirstD, isLastD, l0aBufferId, l0cBufferId,
                        l0b_[dStart * QKV_K_SCALE_HEAD_DIM_D128], l0cOffset);
            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + l0aBufferId));
            ++l0aBufferUseId;
        }
    }

    __aicore__ inline void LoadAToL0(uint64_t segmentRows, uint64_t mSize, uint64_t dStart, uint64_t dSize,
                                     uint32_t bufferId, const LocalTensor<bfloat16_t> &segmentL1Nz)
    {
        if (mSize == 0U) {
            return;
        }

        const uint64_t rowSizeAligned = AlignUp(segmentRows, QKV_K_SCALE_NZ_C0);
        const uint64_t mSizeAligned = AlignUp(mSize, QKV_K_SCALE_NZ_C0);
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = static_cast<uint16_t>(dStart / QKV_K_SCALE_NZ_C0);
        loadDataParams.mStep = static_cast<uint16_t>(CeilDiv(mSize, QKV_K_SCALE_NZ_C0));
        loadDataParams.kStep = static_cast<uint16_t>(dSize / (QKV_K_SCALE_BLOCK_BYTES / sizeof(bfloat16_t)));
        loadDataParams.srcStride = static_cast<int32_t>(CeilDiv(rowSizeAligned, QKV_K_SCALE_NZ_C0));
        loadDataParams.dstStride = static_cast<uint16_t>(CeilDiv(mSizeAligned, QKV_K_SCALE_NZ_C0));
        loadDataParams.ifTranspose = false;
        LoadData(l0a_[bufferId * AIC_L0A_ONE_BUFFER_ELEMENTS], segmentL1Nz, loadDataParams);
    }

    __aicore__ inline void ComputeMmad(uint64_t mSize, uint64_t dSize, bool initC, bool updateUnitFlag,
                                       uint32_t l0aBufferId, uint32_t l0cBufferId,
                                       const LocalTensor<bfloat16_t> &rotationL0bChunk, uint64_t l0cOffset)
    {
        if (mSize == 0U) {
            return;
        }

        MmadParams params;
        params.m = static_cast<uint16_t>(AlignUp(mSize, QKV_K_SCALE_NZ_C0));
        params.n = static_cast<uint16_t>(QKV_K_SCALE_HEAD_DIM_D128);
        params.k = static_cast<uint16_t>(dSize);
        params.unitFlag = updateUnitFlag ? AIC_UNIT_FLAG_UPDATE : AIC_UNIT_FLAG_CHECK_ONLY;
        params.cmatrixInitVal = initC;
        params.cmatrixSource = false;
        Mmad(l0c_[l0cBufferId * AIC_L0C_ONE_BUFFER_ELEMENTS + l0cOffset],
             l0a_[l0aBufferId * AIC_L0A_ONE_BUFFER_ELEMENTS], rotationL0bChunk, params);
    }

    __aicore__ inline void FixpipeSegmentToAivUb(uint64_t segmentRows, uint64_t dstBaseRow, uint32_t bufferId,
                                                 uint64_t l0cBaseOffset, const LocalTensor<float> &qkAfterCubeUb)
    {
        if (segmentRows == 0U) {
            return;
        }

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = static_cast<uint16_t>(QKV_K_SCALE_HEAD_DIM_D128);
        fixpipeParams.mSize = static_cast<uint16_t>(AlignUp(segmentRows, QKV_K_SCALE_MIX_AIV_PER_AIC));
        fixpipeParams.srcStride = static_cast<uint16_t>(AlignUp(fixpipeParams.mSize, QKV_K_SCALE_NZ_C0));
        fixpipeParams.dstStride = QKV_K_SCALE_HEAD_DIM_D128;
        fixpipeParams.dualDstCtl = QKV_K_SCALE_FIXPIPE_DUAL_DST_SPLIT_M;
        fixpipeParams.subBlockId = false;
        fixpipeParams.unitFlag = AIC_UNIT_FLAG_UPDATE;
        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = 0U;
        fixpipeParams.params.dstNdStride = 0U;

        Fixpipe<float, float, QKV_K_SCALE_FIXPIPE_ROW_MAJOR_UB>(
            qkAfterCubeUb[dstBaseRow * QKV_K_SCALE_HEAD_DIM_D128],
            l0c_[bufferId * AIC_L0C_ONE_BUFFER_ELEMENTS + l0cBaseOffset], fixpipeParams);
    }

    __aicore__ inline void FixpipeQSegmentToGm(uint64_t validRows, uint64_t dstBaseRow, uint64_t sourceRows,
                                               uint32_t bufferId)
    {
        if (validRows == 0U) {
            return;
        }

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = static_cast<uint16_t>(QKV_K_SCALE_HEAD_DIM_D128);
        fixpipeParams.mSize = static_cast<uint16_t>(validRows);
        fixpipeParams.srcStride = static_cast<uint16_t>(AlignUp(sourceRows, QKV_K_SCALE_NZ_C0));
        fixpipeParams.dstStride = QKV_K_SCALE_HEAD_DIM_D128;
        fixpipeParams.quantPre = QuantMode_t::F322BF16;
        fixpipeParams.dualDstCtl = 0U;
        fixpipeParams.subBlockId = false;
        fixpipeParams.unitFlag = AIC_UNIT_FLAG_UPDATE;
        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = 0U;
        fixpipeParams.params.dstNdStride = 0U;

        Fixpipe<bfloat16_t, float, QKV_K_SCALE_FIXPIPE_ROW_MAJOR_GM>(
            qOutBf16Gm_[dstBaseRow * QKV_K_SCALE_HEAD_DIM_D128], l0c_[bufferId * AIC_L0C_ONE_BUFFER_ELEMENTS],
            fixpipeParams);
    }

    __aicore__ inline void BindLocalTensors()
    {
        rotationL1Nz_ = LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_K_SCALE_ROTATION_L1_OFFSET,
                                                QKV_K_SCALE_ROTATION_ONE_L1_ELEMENTS);
        l0a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, AIC_L0A_BUFFER_ELEMENTS);
        l0b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, AIC_L0B_BUFFER_ELEMENTS);
        l0c_ = LocalTensor<float>(TPosition::CO1, 0, AIC_L0C_BUFFER_ELEMENTS);
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        rotationGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.rotation);
        if constexpr (IS_MROPE) {
            qOutBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qOut);
        }
    }

    uint64_t l0cBufferUseId_;
    uint64_t qHeadNum_;
    uint64_t kvHeadNum_;
    GlobalTensor<bfloat16_t> rotationGm_;
    GlobalTensor<bfloat16_t> qOutBf16Gm_;
    LocalTensor<bfloat16_t> rotationL1Nz_;
    LocalTensor<bfloat16_t> l0a_;
    LocalTensor<bfloat16_t> l0b_;
    LocalTensor<float> l0c_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
