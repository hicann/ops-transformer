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
 * \file quant_block_sparse_attn_mx_block_cube.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 cube 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_

#include "kernel_operator.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "common/CopyInL1.h"
#include "common/FixpipeOut.h"
#include "common/buffer_manager.h"
#include "common/buffers_policy.h"
#include "common/matmul.h"
#include "common/util_regbase_mx.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_base_matmul;
using namespace regbasemx;

namespace BaseApi {
template <typename inputType, typename mmType, typename outputType, QBSALayout layout, QBSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxBlockCube {
public:
    using INPUT_T = inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = mmType;
    using MxBmm2Buf = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    static constexpr bool USE_DN = useDn;
    static constexpr bool IS_PA = isPa;
    static constexpr QBSALayout LAYOUT = layout;
    static constexpr QBSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == QBSALayout::TND && KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MX cube currently only supports TND query and PA_BNBD KV");
    static_assert(M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MX cube currently only supports S1=128, S2=512, D=128 and DV=128");
    // 每 32 个 fp8 数据元素对应一个 e8m0 scale。
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    // V scale 为 [blockNum,N,blockSize/64,DV,2]。
    static constexpr uint32_t MX_TOKEN_GROUP = 64U;
    static constexpr uint32_t MX_SCALE_LAST_DIM = 2U;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void Init(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *qScale, __gm__ uint8_t *kScale, __gm__ uint8_t *vScale)
    {
        if ASCEND_IS_AIC {
            tPipe_ = pipe;
            l1BufferManager_ = l1BuffMgr;
            queryGm_.SetGlobalBuffer((__gm__ INPUT_T *)query);
            keyGm_.SetGlobalBuffer((__gm__ INPUT_T *)key);
            valueGm_.SetGlobalBuffer((__gm__ INPUT_T *)value);
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
            qScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)qScale);
            kScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)kScale);
            vScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)vScale);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIC {
            static_assert(USE_DN, "MXFullQuantMode cube currently implements DN copy/load path; "
                                  "use_dn is kept for future non-DN extension");
            // data 与 scale 连续存放，LoadData 在 L1->L0 时完成随路反量化。
            constexpr uint32_t qSize = M_BASE * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t qScaleSize = M_BASE * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t kvSize = S2_SPLIT * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t kvScaleSize = S2_SPLIT * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t valueScaleSize =
                (S2_SPLIT / MX_TOKEN_GROUP) * DV_BASE * MX_SCALE_LAST_DIM * sizeof(SCALE_T);

            // Q 双 buffer；K 四 buffer；V 双 buffer。
            l1QBuffers_.Init(*l1BufferManager_, qSize + qScaleSize);
            l1KVBuffers_.Init(*l1BufferManager_, kvSize + kvScaleSize);
            l1VBuffers_.Init(*l1BufferManager_, kvSize + valueScaleSize);

            // L0A/L0B 双 buffer；L0C 双 buffer。C1 最大输出为 [256,128] fp32，需要 128KB。
            l0aBufferManager_.Init(tPipe_, 64U * 1024U);
            l0bBufferManager_.Init(tPipe_, 64U * 1024U);
            l0cBufferManager_.Init(tPipe_, 256U * 1024U);
            mmL0ABuffers_.Init(l0aBufferManager_, 32U * 1024U);
            mmL0BBuffers_.Init(l0bBufferManager_, 32U * 1024U);
            mmL0CBuffers_.Init(l0cBufferManager_, 128U * 1024U);
        }
    }

    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        IterateBmm1Impl<0U, true, true>(outputBuf, runInfo, constInfo);
    }

    __aicore__ inline void IterateBmm1ReuseQ(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf0,
                                             Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf1,
                                             MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        // Both 256-column C1 subLoops share Q/QScale. Keep Q resident in L0B across the paired MMADs.
        IterateBmm1Impl<0U, true, false>(outputBuf0, runInfo, constInfo);
        IterateBmm1Impl<1U, false, true>(outputBuf1, runInfo, constInfo);
    }

private:
    template <uint32_t SUB_LOOP, bool LOAD_Q_TO_L0B, bool RELEASE_Q_L0B>
    __aicore__ inline void IterateBmm1Impl(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                           MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        static_assert(SUB_LOOP < 2U, "MX C1 only supports subLoop 0/1");
        // C1: K[256,D] x Q^T[D,128]；K 由多个 sparse block 拼接。
        const uint32_t s2CalcSize = GetSubLoopS2Size<SUB_LOOP>(runInfo);

        Buffer<BufferType::L1> qBuf;
        const uint64_t qScaleOffset = static_cast<uint64_t>(AlignUp32(runInfo.actMSize)) * constInfo.dSize;
        if constexpr (LOAD_Q_TO_L0B) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                // Q 用 ND2NZ，QScale 用 DN2NZ；后续 S2 loop 复用。
                qBuf = l1QBuffers_.Get();
                qBuf.Wait<HardEvent::MTE1_MTE2>();
                LocalTensor<INPUT_T> qTensor = qBuf.GetTensor<INPUT_T>();
                CopyToL1Nd2Nz<INPUT_T>(qTensor, queryGm_[runInfo.queryOffset], runInfo.actMSize, constInfo.dSize,
                                       constInfo.n2GD);

                LocalTensor<SCALE_T> qScaleTensor = qBuf.GetTensor<SCALE_T>(qScaleOffset);
                MxCopyScaleToL1Dn2Nz<SCALE_T>(qScaleTensor, qScaleGm_[runInfo.queryScaleOffset],
                                              constInfo.queryScaleDSize * constInfo.scaleLastDim, runInfo.actMSize,
                                              constInfo.qScaleN1D);
                qBuf.Set<HardEvent::MTE2_MTE1>();
            } else {
                qBuf = l1QBuffers_.GetPre();
            }
        } else {
            // Paired subLoop1 reuses both the current L1 Q buffer and its already loaded L0B tile.
            qBuf = l1QBuffers_.GetPre();
        }

        Buffer<BufferType::L1> kBuf = l1KVBuffers_.Get();
        kBuf.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<INPUT_T> kTensor = kBuf.GetTensor<INPUT_T>();
        const uint64_t kScaleOffset = static_cast<uint64_t>(AlignUp32(s2CalcSize)) * constInfo.dSize;
        LocalTensor<SCALE_T> kScaleTensor = kBuf.GetTensor<SCALE_T>(kScaleOffset);
        // K/KScale 共用 sparse overlap 计算，分别发起 PA data 与 scale 搬运。
        CopyKeyAndScaleToL1(kTensor, kScaleTensor, runInfo, constInfo, s2CalcSize, SUB_LOOP);
        kBuf.Set<HardEvent::MTE2_MTE1>();

        kBuf.Wait<HardEvent::MTE2_MTE1>();
        if constexpr (LOAD_Q_TO_L0B) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                qBuf.Wait<HardEvent::MTE2_MTE1>();
            }
        }

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers_.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();
        // MxMatmulFull 同时消费 fp8 data 与 e8m0 scale。
        MMParam param = MxMakeMMParam(s2CalcSize, runInfo.actMSize, constInfo.dSize, false, true);
        if constexpr (LOAD_Q_TO_L0B && RELEASE_Q_L0B) {
            MxMatmulFull<INPUT_T, INPUT_T, MM_T, S2_SPLIT, M_BASE, D_BASE, ABLayout::MK, ABLayout::KN,
                         BuffersPolicyDB<BufferType::L0A>, BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T,
                         mx_fp8_e4m3_t, mx_fp8_e4m3_t>(kTensor, qBuf.GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_,
                                                       mm1ResL0C.GetTensor<MM_T>(), param, kScaleTensor,
                                                       qBuf.GetTensor<SCALE_T>(qScaleOffset));
        } else {
            MxMatmulFullReuseB<LOAD_Q_TO_L0B, RELEASE_Q_L0B, INPUT_T, INPUT_T, MM_T, S2_SPLIT, M_BASE, D_BASE,
                               ABLayout::MK, ABLayout::KN, BuffersPolicyDB<BufferType::L0A>,
                               BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
                kTensor, qBuf.GetTensor<INPUT_T>(), mmL0ABuffers_, mmL0BBuffers_, mm1ResL0C.GetTensor<MM_T>(), param,
                kScaleTensor, qBuf.GetTensor<SCALE_T>(qScaleOffset));
        }

        if constexpr (RELEASE_Q_L0B) {
            if (unlikely(runInfo.isLastS2Loop)) {
                qBuf.Set<HardEvent::MTE1_MTE2>();
            }
        }
        kBuf.Set<HardEvent::MTE1_MTE2>();
        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        FixpipeMm1(outputBuf.GetTensor<MM_T>(), mm1ResL0C.GetTensor<MM_T>(), runInfo, s2CalcSize);
        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

public:
    __aicore__ inline void IterateBmm2(MxBmm2Buf &outputBuf,
                                       BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo)
    {
        // C2: P[128,512] x V[512,DV]，P/V 均携带 e8m0 scale。
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> pBuf = inputBuf.GetCube();
        pBuf.WaitCrossCore();
        constexpr uint64_t pScaleOffset = static_cast<uint64_t>(M_BASE) * S2_BASE;
        if (runInfo.actSingleLoopS2Size > S2_SPLIT) {
            // tail PScale 填 e8m0(1.0)。
            constexpr uint32_t int16Bytes = 2U;
            LocalTensor<int16_t> pScalePadTensor =
                pBuf.GetTensor<int16_t>(pScaleOffset / int16Bytes + M_BASE * S2_SPLIT / MX_SCALE_GROUP / int16Bytes);
            InitConstValue(pScalePadTensor, {1, M_BASE * S2_SPLIT / MX_SCALE_GROUP * sizeof(SCALE_T) / 32U, 0, 0x7f7f});
        }
        LocalTensor<SCALE_T> pScaleTensor = pBuf.GetTensor<SCALE_T>(pScaleOffset);

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers_.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>();
        constexpr uint32_t baseK = S2_SPLIT;
        constexpr uint64_t l1BaseKOffset = static_cast<uint64_t>(baseK) * M_BASE;
        constexpr uint64_t l1ScaleOffset = l1BaseKOffset / MX_SCALE_GROUP;
        const uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1U) / baseK;
        for (uint32_t kIdx = 0U; kIdx < kLoops; ++kIdx) {
            const uint32_t realK = (kIdx == kLoops - 1U) ? runInfo.actSingleLoopS2Size - kIdx * baseK : baseK;
            Buffer<BufferType::L1> vBuf = l1VBuffers_.Get();
            vBuf.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<INPUT_T> vTensor = vBuf.GetTensor<INPUT_T>();
            // V 使用 ND2NZ，tail 补零到 64-token 对齐。
            InitValuePadding(vTensor, realK, constInfo.dSizeV);
            CopyValueToL1(vTensor, runInfo, constInfo, realK, kIdx);

            const uint64_t vScaleOffset = static_cast<uint64_t>(AlignUp64(realK)) * constInfo.dSizeV;
            LocalTensor<SCALE_T> vScaleTensor = vBuf.GetTensor<SCALE_T>(vScaleOffset);
            // VScale 按 64-token group 寻址。
            CopyValueScaleToL1(vScaleTensor, runInfo, constInfo, realK, kIdx);
            vBuf.Set<HardEvent::MTE2_MTE1>();
            vBuf.Wait<HardEvent::MTE2_MTE1>();

            MMParam param =
                MxMakeMMParam(M_BASE, constInfo.dSizeV, AlignUp64(realK), true, false, kIdx == 0U, kIdx == 0U);
            // P 转置，V 保持 KN。
            MxMatmulFull<INPUT_T, INPUT_T, MM_T, M_BASE, DV_BASE, baseK, ABLayout::MK, ABLayout::KN,
                         BuffersPolicyDB<BufferType::L0A>, BuffersPolicyDB<BufferType::L0B>, SCALE_T, SCALE_T,
                         mx_fp8_e4m3_t, mx_fp8_e4m3_t>(pBuf.GetTensor<INPUT_T>()[kIdx * l1BaseKOffset], vTensor,
                                                       mmL0ABuffers_, mmL0BBuffers_, mm2ResL0C.GetTensor<MM_T>(), param,
                                                       pScaleTensor[kIdx * l1ScaleOffset], vScaleTensor);
            vBuf.Set<HardEvent::MTE1_MTE2>();
        }

        mm2ResL0C.Set<HardEvent::M_FIX>();
        mm2ResL0C.Wait<HardEvent::M_FIX>();
        outputBuf.WaitCrossCore();
        FixpipeMm2(outputBuf.GetTensor<MM_T>(), mm2ResL0C.GetTensor<MM_T>(), runInfo, constInfo);
        mm2ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) const
    {
        return (value + 31U) >> 5 << 5;
    }

    __aicore__ inline uint32_t AlignUp64(uint32_t value) const
    {
        return (value + 63U) >> 6 << 6;
    }

    template <uint32_t SUB_LOOP>
    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo &runInfo) const
    {
        static_assert(SUB_LOOP < 2U, "MX C1 only supports subLoop 0/1");
        if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            if constexpr (SUB_LOOP == 0U) {
                return runInfo.actSingleLoopS2Size;
            }
            return 0U;
        }
        if constexpr (SUB_LOOP == 0U) {
            return S2_SPLIT;
        }
        return runInfo.actSingleLoopS2Size - S2_SPLIT;
    }

    __aicore__ inline uint32_t GetBlockOverlapStart(uint32_t tileStart, uint32_t blockStart) const
    {
        return tileStart > blockStart ? tileStart : blockStart;
    }

    __aicore__ inline uint32_t GetBlockOverlapEnd(uint32_t tileEnd, uint32_t blockEnd) const
    {
        return tileEnd < blockEnd ? tileEnd : blockEnd;
    }

    __aicore__ inline void CopyKeyAndScaleToL1(LocalTensor<INPUT_T> &keyDstTensor,
                                               LocalTensor<SCALE_T> &keyScaleDstTensor, const MxRunInfo &runInfo,
                                               const MxConstInfo &constInfo, uint32_t s2CalcSize, uint32_t subLoop)
    {
        // PA BNBD K 与 KScale 使用相同的 sparse 顺序和 token overlap。
        const uint32_t tileStart = subLoop * S2_SPLIT;
        const uint32_t tileEnd = tileStart + s2CalcSize;
        const uint32_t copyRowNumAlign = AlignUp32(s2CalcSize);
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);

        Position startPos;
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape keyShape;
        keyShape.blockSize = constInfo.paBlockSize;
        keyShape.headNum = constInfo.n2Size;
        keyShape.headDim = constInfo.dSize;
        keyShape.actHeadDim = constInfo.dSize;
        keyShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        keyShape.copyRowNumAlign = copyRowNumAlign;
        keyShape.pageStride = constInfo.paBlockStride;
        keyShape.rowStride = constInfo.dSize;

        // KScale GM 为 [physicalBlock,kvHeads,blockSize,D/64,2]，单行使用 DN2NZ。
        PAShape keyScaleShape;
        keyScaleShape.blockSize = constInfo.paBlockSize;
        keyScaleShape.headNum = constInfo.n2Size;
        keyScaleShape.headDim = constInfo.keyScaleDSize * constInfo.scaleLastDim;
        keyScaleShape.actHeadDim = keyScaleShape.headDim;
        keyScaleShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        keyScaleShape.copyRowNumAlign = copyRowNumAlign;
        keyScaleShape.pageStride = constInfo.n2Size * constInfo.kScaleN2D;
        keyScaleShape.rowStride = keyScaleShape.headDim;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] + copyStartInBlock;
            keyShape.copyRowNum = copyRowNum;
            LocalTensor<INPUT_T> keyBlockDstTensor = keyDstTensor[(overlapStart - tileStart) * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(keyBlockDstTensor, keyGm_, blockTableGm_, KVLAYOUT::BNBD, keyShape, startPos);

            keyScaleShape.copyRowNum = copyRowNum;
            LocalTensor<SCALE_T> keyScaleBlockDstTensor =
                keyScaleDstTensor[(overlapStart - tileStart) * keyScaleShape.actHeadDim];
            MxGmFlatScaleCopyInToL1PAForDN<SCALE_T>(keyScaleBlockDstTensor, kScaleGm_, blockTableGm_, keyScaleShape,
                                                    startPos);
        }
    }

    __aicore__ inline void CopyValueToL1(LocalTensor<INPUT_T> &dstTensor, const MxRunInfo &runInfo,
                                         const MxConstInfo &constInfo, uint32_t realK, uint32_t kIdx)
    {
        // V 使用与 K 相同的 PA 拼接顺序。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);

        Position startPos;
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape valueShape;
        valueShape.blockSize = constInfo.paBlockSize;
        valueShape.headNum = constInfo.n2Size;
        valueShape.headDim = constInfo.dSizeV;
        valueShape.actHeadDim = constInfo.dSizeV;
        valueShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        valueShape.copyRowNumAlign = AlignUp64(realK);
        valueShape.pageStride = constInfo.paBlockStride;
        valueShape.rowStride = constInfo.dSizeV;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] + copyStartInBlock;
            valueShape.copyRowNum = copyRowNum;
            LocalTensor<INPUT_T> valueDstTensor = dstTensor[(overlapStart - tileStart) * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(valueDstTensor, valueGm_, blockTableGm_, KVLAYOUT::BNBD, valueShape, startPos);
        }
    }

    __aicore__ inline void CopyValueScaleToL1(LocalTensor<SCALE_T> &dstTensor, const MxRunInfo &runInfo,
                                              const MxConstInfo &constInfo, uint32_t realK, uint32_t kIdx)
    {
        // VScale GM 为 [physicalBlock, kvHeads, blockSize/64, DV, 2]，offset 单位为 64-token group。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        const uint32_t totalGroupCount = AlignUp64(realK) / MX_TOKEN_GROUP;
        const uint32_t scaleBlockSize = constInfo.paBlockSize / MX_TOKEN_GROUP;
        constexpr uint32_t blockElementCnt = 32U / sizeof(SCALE_T);
        uint32_t copiedGroupCount = 0U;

        Position startPos;
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape valueScaleShape;
        valueScaleShape.blockSize = scaleBlockSize;
        valueScaleShape.headNum = constInfo.n2Size;
        valueScaleShape.headDim = constInfo.valueScaleDSize;
        valueScaleShape.actHeadDim = constInfo.valueScaleDSize;
        valueScaleShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        valueScaleShape.copyRowNumAlign = totalGroupCount;
        valueScaleShape.pageStride = constInfo.n2Size * constInfo.valueScaleN2D;
        valueScaleShape.rowStride = valueScaleShape.headDim;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;
            const uint32_t groupStartInBlock = copyStartInBlock / MX_TOKEN_GROUP;
            const uint32_t groupEndInBlock = (copyStartInBlock + copyRowNum + MX_TOKEN_GROUP - 1U) / MX_TOKEN_GROUP;
            const uint32_t copyGroupNum = groupEndInBlock - groupStartInBlock;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] / MX_TOKEN_GROUP + groupStartInBlock;
            valueScaleShape.copyRowNum = copyGroupNum;
            LocalTensor<SCALE_T> valueScaleDstTensor = dstTensor[copiedGroupCount * blockElementCnt];
            MxGmFlatScaleCopyInToL1PAForND<SCALE_T>(valueScaleDstTensor, vScaleGm_, blockTableGm_, valueScaleShape,
                                                    startPos);
            copiedGroupCount += copyGroupNum;
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      const MxRunInfo &runInfo, uint32_t s2RealSize)
    {
        // C1 写为 VF 使用的 row-major UB layout。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = AlignUp32(runInfo.actMSize);
        fixpipeParams.mSize = (s2RealSize + 7U) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15U) >> 4) << 4U;
        fixpipeParams.dstStride = 64U;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm2(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      const MxRunInfo &, const MxConstInfo &constInfo)
    {
        // C2 以 fp32 写入 UB，供 V2 update。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (constInfo.dSizeV + 7U) >> 3 << 3;
        fixpipeParams.mSize = M_BASE;
        fixpipeParams.srcStride = ((M_BASE + 15U) >> 4) << 4U;
        fixpipeParams.dstStride = (DV_BASE + 15U) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void InitValuePadding(const LocalTensor<INPUT_T> &valueL1, uint32_t realK, uint32_t curN)
    {
        // V tail 补零到 64-token 对齐。
        const uint32_t curPadK = AlignUp64(realK);
        if (likely(curPadK == realK)) {
            return;
        }
        InitConstValueParams<half> initConstValueParams;
        initConstValueParams.repeatTimes = AlignUp64(curN) >> 5;
        initConstValueParams.blockNum = curPadK - realK;
        initConstValueParams.dstGap = realK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[realK << 4U], initConstValueParams);
    }

    TPipe *tPipe_ = nullptr;
    BufferManager<BufferType::L1> *l1BufferManager_ = nullptr;
    GlobalTensor<INPUT_T> queryGm_;
    GlobalTensor<INPUT_T> keyGm_;
    GlobalTensor<INPUT_T> valueGm_;
    GlobalTensor<SCALE_T> qScaleGm_;
    GlobalTensor<SCALE_T> kScaleGm_;
    GlobalTensor<SCALE_T> vScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;
    BufferManager<BufferType::L0C> l0cBufferManager_;
    BuffersPolicyDB<BufferType::L1> l1QBuffers_;
    BuffersPolicy4buff<BufferType::L1> l1KVBuffers_;
    BuffersPolicyDB<BufferType::L1> l1VBuffers_;
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers_;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers_;
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers_;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
