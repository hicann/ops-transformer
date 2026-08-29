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
 * \file chunk_gated_delta_rule_stage3_arch35.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE3_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_STAGE3_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_matmul_basic.h"
#include "vf/chunk_gated_delta_rule_stage3_vf.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;

struct StageThreeParams {
    GlobalTensor<bfloat16_t> qkt; // (Nv, Sp, Dk)
    GlobalTensor<float> gCumExp;  // (Nv, Sp)
    GlobalTensor<bfloat16_t> vInner;
    GlobalTensor<float> maskTensor;
    GM_ADDR ws;
    GlobalTensor<bfloat16_t> attnOut;
    TPipe *pipe;
    ChunkGroup *cg;
    float scale;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
};

template <bool gOptional>
class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        coreId_ = GetBlockIdx();
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        chunkNum_ = (seqLength_ + chunkSize_ - 1) / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        uint64_t workSpaceOffset = 0;
        tmpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            initParams->ws + workSpaceOffset + coreNum_ * chunkSize_ * chunkSize_ * sizeof(float)));
        if ASCEND_IS_AIC {
            mmBasic_.Init();
            return;
        }
        coreId_ /= AIC_AIV_1_1;
        if (GetSubBlockIdx() == 1) {
            return;
        }
        uint64_t inQueueSize =
            static_cast<uint64_t>(chunkSize_) * AscendC::Std::max((int64_t)chunkSize_, paddedDv_) * sizeof(bfloat16_t);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inQueueSize);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE,
                          chunkSize_ > paddedDv_ ? chunkSize_ * chunkSize_ * sizeof(float) :
                                                   chunkSize_ * paddedDv_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(tmpBuff_, (STAGE3_BUFFER_COUNT * chunkSize_ * chunkSize_ * sizeof(float)));
        uint64_t buffOffset = 0;
        uint64_t tmpOffset = chunkSize_ * chunkSize_;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        maskBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);

        // 搬入mask
        DataCopyExtParams inParams{static_cast<uint16_t>(chunkSize_), static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                                   0, 0, 0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(maskBuffer_, sTP_->maskTensor, inParams, copyPadParams);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
    }

    __aicore__ inline void Process()
    {
        int64_t totalChunks = Nv_ * chunkNum_; // Nv Nc 融合
        int64_t chunksPerCore = (totalChunks + coreNum_ - 1) / coreNum_;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        int64_t startChunk = coreId_ * chunksPerCore;
        int64_t endChunk = startChunk + chunksPerCore > totalChunks ? totalChunks : startChunk + chunksPerCore;
        for (int64_t idx = startChunk; idx < endChunk; idx++) {
            int64_t nvId = idx / chunkNum_;
            int64_t chunkId = idx % chunkNum_;
            int64_t chunkPos = chunkId * chunkSize_;                                 // 当前chunk起始位置
            curChunkSize_ = (chunkId == chunkNum_ - 1) ? lastChunkSize : chunkSize_; // 尾块
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    CalMaskedQKT(tmpGM_[coreId_ * chunkSize_ * chunkSize_], nvId, chunkPos);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                CrossCoreWaitFlag(0x3);
            }

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(0x4);
                mmBasic_.Execute<false, false, true, bfloat16_t>(tmpGM_[coreId_ * chunkSize_ * chunkSize_],
                                                                 sTP_->vInner[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                                                                 sTP_->attnOut[nvId * Dv_ + chunkPos * Nv_ * Dv_],
                                                                 curChunkSize_, Dv_, curChunkSize_, 0, 0, Nv_ * Dv_);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x3);
            }
        }
        if ASCEND_IS_AIC {
            mmBasic_.End();
        }
    }

    __aicore__ inline void CalMaskedQKT(GlobalTensor<bfloat16_t> outGM, int nvId, int chunkPos)
    {
        // gated 路径先将长度为 curChunkSize_ 的 g 向量搬入独立 TBuf；随后 qkt 的 MTE2
        // 搬运与 DeQue 保证此前的 g 搬运也已完成，使两个输入可由同一次 VF_CALL 读取。
        if constexpr (gOptional) {
            DataCopyExtParams gInParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curChunkSize_ * sizeof(float)),
                                        0, 0, 0};
            DataCopyPadExtParams<float> gPadParams{false, 0, 0, 0};
            DataCopyPad(tmpBuffer1_, sTP_->gCumExp[nvId * Sp_ + chunkPos], gInParams, gPadParams);
        }

        // qkt
        AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_], curChunkSize_, curChunkSize_);
        auto qkt = inQueue_.DeQue<bfloat16_t>();
        auto scale_qkt = outQueue_.AllocTensor<bfloat16_t>();
        auto maskAddr = reinterpret_cast<__ubuf__ float *>(maskBuffer_.GetPhyAddr());
        auto qktAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(qkt.GetPhyAddr());
        auto scaleQktAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(scale_qkt.GetPhyAddr());
        if constexpr (gOptional) {
            auto gAddr = reinterpret_cast<__ubuf__ float *>(tmpBuffer1_.GetPhyAddr());
            AscendC::VF_CALL<ComputeMaskedQktGatedVF>(scaleQktAddr, qktAddr, gAddr, maskAddr, sTP_->scale,
                                                      static_cast<uint16_t>(curChunkSize_),
                                                      static_cast<uint16_t>(chunkSize_));
        } else {
            AscendC::VF_CALL<ComputeMaskedQktNoGateVF>(scaleQktAddr, qktAddr, maskAddr, sTP_->scale,
                                                       static_cast<uint16_t>(curChunkSize_),
                                                       static_cast<uint16_t>(chunkSize_));
        }
        outQueue_.EnQue(scale_qkt);
        AlignedCopyOut(outGM, curChunkSize_, curChunkSize_);
        inQueue_.FreeTensor(qkt);
    }

    template <typename inType>
    __aicore__ inline void AlignedCopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        // 非对齐拷入会自动对齐, 然后离散拷入UB
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType));
        DataCopyExtParams inParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0),
                                   static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(inType) / BLOCK_SIZE), 0};
        DataCopyPadExtParams<inType> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void AlignedCopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(outType)) * (BLOCK_SIZE / sizeof(outType));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(outType) / BLOCK_SIZE);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        DataCopyPad(tmpGM, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageThreeParams *sTP_;
    TPipe *pipe_;
    CGDRMatmulBasic mmBasic_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<bfloat16_t> tmpGM_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> tmpBuffer2_;
    LocalTensor<float> maskBuffer_;
    int64_t seqLength_;
    int64_t Sp_;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
    int64_t paddedDv_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int32_t curChunkSize_;
    int32_t chunkSize_;
    int32_t coreId_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE3_ARCH35_H
