/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_KERNEL_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"
#include "mhc_pre_sinkhorn_backward_key_arch35.h"
#include "mhc_pre_sinkhorn_backward_mm_arch35.h"

namespace MhcPreSinkhornBackwardNs {
using namespace AscendC;

constexpr uint32_t SYNC_MODE = 4;
constexpr uint32_t SECOND_AIV_FLAG_OFFSET = 16;
constexpr uint32_t FLAG_GQ_READY = 8;
constexpr uint32_t FLAG_INPUTS_RELEASED = 9;
constexpr uint32_t FLAG_MM1_READY = 10;
constexpr uint32_t FLAG_X_READY = 11;
constexpr uint32_t FLAG_MM2_READY = 12;
constexpr uint32_t FLAG_MM1_RELEASED = 13;
constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L1_GQ_OFFSET = 0;
constexpr uint32_t ZERO_TILE = K_TILE_VALUE;
constexpr uint32_t ROOT_POOL_BUFFER_COUNT = STAGE_POOL_BUFFER_COUNT;

template <typename T, SplitMode MODE, bool DETERMINISTIC>
class MhcPreSinkhornBackwardKernel {
    MhcBwdShape shape_;
    // Tiling parameters derived from N at runtime.
    uint32_t gqTileElements_ = 0;
    uint32_t l1GqWaveElems_ = 0;
    uint32_t workspaceRowsGrOffset_ = 0;
    uint32_t workspaceRowsHpreOffset_ = 0;
    uint32_t workspaceRowsElements_ = 0;
    uint32_t prepareOutGrOffset_ = 0;
    uint32_t prepareOutElements_ = 0;
    uint32_t l1XOffset_ = 0;
    constexpr static uint32_t MM1_SLOT_ELEMENTS = ROWS_PER_AIV * K_TILE_VALUE;

    __aicore__ inline void InitDerivedShape()
    {
        gqTileElements_ = ROWS_PER_AIV * shape_.hAlign;
        l1GqWaveElems_ = MM_TILE_M * shape_.hAlign;
        workspaceRowsGrOffset_ = gqTileElements_;
        workspaceRowsHpreOffset_ = workspaceRowsGrOffset_ + ROWS_PER_AIV;
        workspaceRowsElements_ = workspaceRowsHpreOffset_ + ROWS_PER_AIV * shape_.n;
        prepareOutGrOffset_ = shape_.hAlign * PREPARE_BATCH;
        prepareOutElements_ = prepareOutGrOffset_ + FLOAT_C0;
        l1XOffset_ = L1_BYTES / sizeof(float) - 2 * MM_TILE_M * K_TILE_VALUE;
    }

public:
    __aicore__ inline void Init(GM_ADDR gradHin, GM_ADDR gradHPost, GM_ADDR gradHRes, GM_ADDR x, GM_ADDR phi,
                                GM_ADDR alpha, GM_ADDR bias, GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha,
                                GM_ADDR gradBias, GM_ADDR userWorkspace,
                                const MhcPreSinkhornBackwardArch35TilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        shape_.Init(tiling->n);
        InitDerivedShape();
        phiGm_.SetGlobalBuffer((__gm__ float*)phi);
        // Keep phi in L2 to reduce GM accesses during repeated reads.
        phiGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_PERSISTENT);
        gradPhiGm_.SetGlobalBuffer((__gm__ float*)gradPhi);
        gradAlphaGm_.SetGlobalBuffer((__gm__ float*)gradAlpha);
        gradBiasGm_.SetGlobalBuffer((__gm__ float*)gradBias);
        if constexpr (MODE == SplitMode::MK_SPLIT) {
            workspaceGm_.SetGlobalBuffer((__gm__ float*)userWorkspace);
        }

        pipe_->InitBuffer(l1Buf_, L1_BYTES);
        l1_ = l1Buf_.Get<float>();

        // The phi B1 ping-pong region limits GQ cache waves; derive capacity from the phi region offset in tiling data.
        l1GqWaves_ = tiling_->l1BOffsetElems / l1GqWaveElems_;
        cacheWaves_ = tiling_->maxMWave < l1GqWaves_ ? tiling_->maxMWave : l1GqWaves_;
        uint32_t grElements = ROWS_PER_AIV * (MODE == SplitMode::MK_SPLIT ? cacheWaves_ : 1);
        uint32_t hPreCacheRows = MODE == SplitMode::MK_SPLIT ? grElements : 0;
        uint32_t rootPoolBytes = STAGE_POOL_BYTES + shape_.paramElements * sizeof(float) +
                                 gqTileElements_ * sizeof(float) + grElements * sizeof(float) +
                                 shape_.scalarReduceAligned * sizeof(float) +
                                 shape_.scalarReduceAligned * sizeof(float) +
                                 MM1_SLOT_ELEMENTS * sizeof(float) * (MODE == SplitMode::MK_SPLIT ? 2 : 1) +
                                 hPreCacheRows * shape_.n * sizeof(float);
        if constexpr (MODE == SplitMode::M_SPLIT) {
            rootPoolBytes += shape_.mm2RowsPerAiv * K_TILE_VALUE * sizeof(float);
        }
        // TBufPool rounds each buffer up to 32 bytes; round the pool size the same way to keep allocations within
        // bounds.
        rootPoolBytes =
            (rootPoolBytes + FLOAT_C0 * sizeof(float) - 1) / (FLOAT_C0 * sizeof(float)) * (FLOAT_C0 * sizeof(float));
        pipe_->InitBufPool(rootPool_, rootPoolBytes);
        rootPool_.InitBuffer(mm1Buf_, MM1_SLOT_ELEMENTS * sizeof(float) * (MODE == SplitMode::MK_SPLIT ? 2 : 1));
        mm1Local_ = mm1Buf_.Get<float>();
        if constexpr (MODE == SplitMode::M_SPLIT) {
            rootPool_.InitBuffer(mm2Buf_, shape_.mm2RowsPerAiv * K_TILE_VALUE * sizeof(float));
            mm2Local_ = mm2Buf_.Get<float>();
        }

        if ASCEND_IS_AIC {
            mm_.Init(tiling_);
        } else {
            rootPool_.InitBuffer(gqBuf_, gqTileElements_ * sizeof(float));
            rootPool_.InitBuffer(grBuf_, grElements * sizeof(float));
            rootPool_.InitBuffer(rowPartBuf_, shape_.scalarReduceAligned * sizeof(float));
            rootPool_.InitBuffer(rowPartCallBuf_, shape_.scalarReduceAligned * sizeof(float));
            gqNd_ = gqBuf_.Get<float>();
            grLocal_ = grBuf_.Get<float>();
            rowPart_ = rowPartBuf_.Get<float>();
            rowPartCall_ = rowPartCallBuf_.Get<float>();
            vector_.InitShape(tiling_->n);
            vector_.InitPermanentBuffers(rootPool_, hPreCacheRows);
            rootPool_.InitBufPool(stageStorage_, STAGE_POOL_BYTES);
            vector_.Init(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut,
                         gradX, gradPhi, gradAlpha, gradBias, tiling_);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            vector_.LoadConstants();
            if constexpr (MODE == SplitMode::M_SPLIT) {
                InitClearStage();
                ClearReductionOutputs();
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            if (GetBlockIdx() >= static_cast<uint32_t>(tiling_->usedAic)) {
                // AIC cores that do not compute must also reclaim the buffer tokens set by Init.
                mm_.End();
                return;
            }
        }
        if constexpr (MODE == SplitMode::M_SPLIT) {
            ProcessMSplit();
        } else {
            ProcessMkSplit();
        }
        if ASCEND_IS_AIV {
            vector_.ReleaseConstants();
        } else {
            mm_.End();
        }
    }

private:
    __aicore__ inline void ResetStagePool()
    {
        stagePool_.Reset();
        // Reset clears the address range in CANN 9.2; rebind the reserved UB region.
        rootPool_.InitBufPool(stagePool_, STAGE_POOL_BYTES, stageStorage_);
    }

    __aicore__ inline void InitClearStage()
    {
        ResetStagePool();
        stagePool_.InitBuffer(clearOutQue_, 1, ZERO_TILE * sizeof(float));
    }

    __aicore__ inline void InitPrepareStage()
    {
        ResetStagePool();
        vector_.InitPrepareStage(stagePool_, MODE == SplitMode::MK_SPLIT ? PREPARE_BATCH : 1);
        if constexpr (MODE == SplitMode::MK_SPLIT) {
            stagePool_.InitBuffer(prepareOutQue_, 1, prepareOutElements_ * sizeof(float));
        }
    }

    __aicore__ inline void InitMComputeStage()
    {
        ResetStagePool();
        vector_.InitGradXStage(stagePool_);
        stagePool_.InitBuffer(gqL1OutQue_, 1, gqTileElements_ * sizeof(float));
        stagePool_.InitBuffer(scalarInQue_, 1, (FLOAT_C0 + shape_.hValue) * sizeof(float));
        stagePool_.InitBuffer(scalarOutQue_, 1, (FLOAT_C0 + shape_.hValue) * sizeof(float));
        stagePool_.InitBuffer(phiInQue_, 1, K_TILE_VALUE * sizeof(float));
        stagePool_.InitBuffer(phiOutQue_, 1, K_TILE_VALUE * sizeof(float));
    }

    __aicore__ inline void InitMkScalarReduceStage()
    {
        ResetStagePool();
        // Separate input segments as [alpha rows aligned to 8][bias rows aligned to AlignU32(hcMix,8)] to avoid
        // cross-segment Gather.
        stagePool_.InitBuffer(scalarInQue_, 1,
                              tiling_->usedPrepAiv * (FLOAT_C0 + AlignU32(shape_.hValue, FLOAT_C0)) * sizeof(float) +
                                  SMALL_BUFFER_SLACK * sizeof(float));
        stagePool_.InitBuffer(scalarOutQue_, 1, (FLOAT_C0 + shape_.hValue) * sizeof(float));
    }

    __aicore__ inline void InitMkComputeStage()
    {
        ResetStagePool();
        vector_.InitGradXStage(stagePool_, true);
        stagePool_.InitBuffer(gqL1OutQue_, 1, gqTileElements_ * sizeof(float));
        stagePool_.InitBuffer(workspaceRowsInQue_, 1, workspaceRowsElements_ * sizeof(float));
    }

    __aicore__ inline void ClearReductionOutputs()
    {
        uint32_t vecId = GetBlockIdx();
        uint32_t usedAiv = tiling_->usedAic * 2;
        LocalTensor<float> zero = clearOutQue_.AllocTensor<float>();
        AscendC::Duplicate(zero, 0.0f, ZERO_TILE);
        clearOutQue_.EnQue(zero);
        zero = clearOutQue_.DeQue<float>();
        int64_t phiElements = tiling_->hcMix * tiling_->k;
        for (int64_t begin = vecId * ZERO_TILE; begin < phiElements; begin += usedAiv * ZERO_TILE) {
            uint32_t count = phiElements - begin < ZERO_TILE ? phiElements - begin : ZERO_TILE;
            CopyUbToGm(gradPhiGm_[begin], zero, count);
        }
        if (vecId == 0) {
            CopyUbToGm(gradAlphaGm_, zero, 3);
            if (shape_.hValue < 16) {
                SyncMte3Writes();
            }
            CopyUbToGm(gradBiasGm_, zero, shape_.hValue);
        }
        clearOutQue_.FreeTensor(zero);
    }

    // Wait for MTE3 completion between small GM writes of alpha and bias.
    __aicore__ inline void WriteScalarOutputs(const LocalTensor<float>& output, uint32_t alphaCount, uint32_t biasCount)
    {
        CopyUbToGm(gradAlphaGm_, output, alphaCount);
        if (shape_.hValue < 16) {
            // Synchronize with the previous write when bias occupies fewer than 60 bytes.
            SyncMte3Writes();
        }
        CopyUbToGm(gradBiasGm_, output[FLOAT_C0], biasCount);
    }

    __aicore__ inline uint32_t ValidRows(int64_t rowBase) const
    {
        if (rowBase >= tiling_->bs) {
            return 0;
        }
        return tiling_->bs - rowBase < ROWS_PER_AIV ? tiling_->bs - rowBase : ROWS_PER_AIV;
    }

    // Round valid GQ rows up to CUBE_M to skip MM2 computation on padded rows.
    __aicore__ inline uint32_t AicWaveKRows(int64_t waveRowBegin) const
    {
        int64_t remain = tiling_->bs - waveRowBegin;
        if (remain <= 0) {
            return ROWS_PER_AIV;
        }
        if (remain > MM_TILE_M) {
            remain = MM_TILE_M;
        }
        return AlignU32(static_cast<uint32_t>(remain), CUBE_M);
    }

    __aicore__ inline void PrepareLocalRows(int64_t rowBase, uint32_t validRows)
    {
        // Empty waves also participate in reduction and must contribute zero.
        AscendC::Duplicate(rowPart_, 0.0f, shape_.scalarReduceElements);
        for (uint32_t localRow = 0; localRow < validRows; ++localRow) {
            vector_.PrepareRow(rowBase + localRow, localRow, gqNd_, grLocal_, rowPartCall_);
            VFAdd(rowPart_, rowPart_, rowPartCall_, shape_.scalarReduceElements);
        }
    }

    __aicore__ inline void LoadLocalRowsFromWorkspace(int64_t rowBase, uint32_t validRows, uint32_t cacheOffset = 0)
    {
        LocalTensor<float> input = workspaceRowsInQue_.AllocTensor<float>();
        uint64_t gqOffset = tiling_->offGQ / sizeof(float);
        uint64_t grOffset = tiling_->offGR / sizeof(float);
        if (validRows != 0) {
            // rightPadding is limited to 32 bytes; pad to 8 elements and use dstStride to skip other unread columns.
            uint32_t hValueAligned = AlignU32(shape_.hValue, FLOAT_C0);
            uint32_t zeroPad = hValueAligned - shape_.hValue;
            uint32_t gapBlocks = (shape_.hAlign - hValueAligned) / FLOAT_C0;
            DataCopyExtParams copy = {static_cast<uint16_t>(validRows),
                                      static_cast<uint32_t>(shape_.hValue * sizeof(float)), 0, gapBlocks, 0};
            DataCopyPadExtParams<float> pad = {true, 0, static_cast<uint8_t>(zeroPad), 0};
            DataCopyPad(input, workspaceGm_[gqOffset + rowBase * shape_.hValue], copy, pad);
            CopyGmToUb(workspaceGm_[grOffset + rowBase], input[workspaceRowsGrOffset_], validRows);
            vector_.LoadHPre(input[workspaceRowsHpreOffset_], rowBase, validRows);
        }
        workspaceRowsInQue_.EnQue(input);
        input = workspaceRowsInQue_.DeQue<float>();
        if (validRows != 0) {
            DataCopy(gqNd_, input, validRows * shape_.hAlign);
            DataCopy(grLocal_[cacheOffset], input[workspaceRowsGrOffset_], AlignU32(validRows, FLOAT_C0));
            vector_.CacheHPre(input[workspaceRowsHpreOffset_], cacheOffset, validRows);
        }
        workspaceRowsInQue_.FreeTensor(input);
    }

    __aicore__ inline void StageGqToL1(uint32_t subBlock, uint32_t validRows, uint32_t slot = 0, bool notify = true)
    {
        LocalTensor<float> gqNz = gqL1OutQue_.AllocTensor<float>();
        // MM2 transposes GQ in pairs of C0 blocks, including the zero-padded columns.
        if (validRows == 0) {
            AscendC::Duplicate(gqNz, 0.0f, gqTileElements_);
        } else if (validRows == ROWS_PER_AIV) {
            TransNdToNz<false, true>(gqNz, gqNd_, ROWS_PER_AIV, shape_.hAlign, shape_.hAlign, ROWS_PER_AIV,
                                     shape_.hValue);
        } else {
            TransNdToNz<true, true>(gqNz, gqNd_, ROWS_PER_AIV, shape_.hAlign, shape_.hAlign, validRows, shape_.hValue);
        }
        gqL1OutQue_.EnQue(gqNz);
        gqNz = gqL1OutQue_.DeQue<float>();
        DataCopyParams copy;
        copy.blockCount = shape_.hAlign / FLOAT_C0;
        copy.blockLen = ROWS_PER_AIV;
        copy.srcStride = 0;
        copy.dstStride = ROWS_PER_AIV;
        DataCopy(l1_[L1_GQ_OFFSET + slot * MM_TILE_M * shape_.hAlign + subBlock * ROWS_PER_AIV * FLOAT_C0], gqNz, copy);
        if (notify) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_GQ_READY);
        }
        gqL1OutQue_.FreeTensor(gqNz);
    }

    __aicore__ inline void StageXToL1(uint32_t subBlock, bool release = true, uint32_t slot = 0)
    {
        LocalTensor<float> xStage = vector_.DeQueXStage(slot);
        LocalTensor<float> xNz = xStage[X_STAGE_NZ_OFFSET];
        DataCopyParams copy;
        copy.blockCount = K_TILE_VALUE / FLOAT_C0;
        copy.blockLen = ROWS_PER_AIV;
        copy.srcStride = 0;
        copy.dstStride = ROWS_PER_AIV;
        DataCopy(l1_[l1XOffset_ + slot * MM_TILE_M * K_TILE_VALUE + subBlock * ROWS_PER_AIV * FLOAT_C0], xNz, copy);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_X_READY);
        if (release) {
            vector_.FreeXStage(xStage);
        }
    }

    __aicore__ inline void ReduceScalarMSplit()
    {
        uint32_t vecId = GetBlockIdx();
        for (uint32_t owner = 0; owner < tiling_->reduceOwnerCount * 2; ++owner) {
            if (vecId == owner) {
                LocalTensor<float> oldValue = scalarInQue_.AllocTensor<float>();
                CopyGmToUb(gradAlphaGm_, oldValue, 3);
                CopyGmToUb(gradBiasGm_, oldValue[FLOAT_C0], shape_.hValue);
                scalarInQue_.EnQue(oldValue);
                oldValue = scalarInQue_.DeQue<float>();
                LocalTensor<float> output = scalarOutQue_.AllocTensor<float>();
                VFAdd(output, oldValue, rowPart_, 3);
                VFAdd(output[FLOAT_C0], oldValue[FLOAT_C0], rowPart_[3], shape_.hValue);
                scalarInQue_.FreeTensor(oldValue);
                scalarOutQue_.EnQue(output);
                output = scalarOutQue_.DeQue<float>();
                WriteScalarOutputs(output, 3, shape_.hValue);
                scalarOutQue_.FreeTensor(output);
            }
            SyncAll<true>();
        }
    }

    __aicore__ inline void ReducePhiMSplit(uint32_t kBegin, uint32_t kValid, uint32_t logicalBlock, uint32_t subBlock)
    {
        uint32_t hBegin = subBlock * shape_.mm2RowsPerAiv;
        for (uint32_t owner = 0; owner < tiling_->reduceOwnerCount; ++owner) {
            if (logicalBlock == owner) {
                for (uint32_t hLocal = 0; hLocal < shape_.mm2RowsPerAiv; ++hLocal) {
                    // The final zero-padded row for odd hcMix only balances dualDst output and is not written to
                    // gradPhi.
                    if (hBegin + hLocal >= static_cast<uint32_t>(tiling_->hcMix)) {
                        continue;
                    }
                    LocalTensor<float> oldValue = phiInQue_.AllocTensor<float>();
                    CopyGmToUb(gradPhiGm_[(hBegin + hLocal) * tiling_->k + kBegin], oldValue, kValid);
                    phiInQue_.EnQue(oldValue);
                    oldValue = phiInQue_.DeQue<float>();
                    LocalTensor<float> output = phiOutQue_.AllocTensor<float>();
                    VFAdd(output, oldValue, mm2Local_[hLocal * K_TILE_VALUE], kValid);
                    phiInQue_.FreeTensor(oldValue);
                    phiOutQue_.EnQue(output);
                    output = phiOutQue_.DeQue<float>();
                    CopyUbToGm(gradPhiGm_[(hBegin + hLocal) * tiling_->k + kBegin], output, kValid);
                    phiOutQue_.FreeTensor(output);
                }
            }
            SyncAll<true>();
        }
    }

    __aicore__ inline void ProcessMSplit()
    {
        uint32_t block = GetBlockIdx();
        uint32_t logicalBlock = block;
        uint32_t subBlock = 0;
        if ASCEND_IS_AIV {
            logicalBlock = block / 2;
            subBlock = block % 2;
        }
        int64_t blockRowBegin = logicalBlock * tiling_->mBlockRows;
        for (uint32_t wave = 0; wave < tiling_->maxMWave; ++wave) {
            int64_t rowBase = blockRowBegin + wave * MM_TILE_M + subBlock * ROWS_PER_AIV;
            uint32_t validRows = ValidRows(rowBase);
            if ASCEND_IS_AIV {
                InitPrepareStage();
                PrepareLocalRows(rowBase, validRows);
                InitMComputeStage();
                StageGqToL1(subBlock, validRows);
                ReduceScalarMSplit();
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_GQ_READY);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_GQ_READY + SECOND_AIV_FLAG_OFFSET);
            }

            for (uint32_t kBegin = 0; kBegin < static_cast<uint32_t>(tiling_->k); kBegin += K_TILE_VALUE) {
                uint32_t kValid = tiling_->k - kBegin < K_TILE_VALUE ? tiling_->k - kBegin : K_TILE_VALUE;
                if ASCEND_IS_AIC {
                    LocalTensor<float> phiL1 =
                        mm_.PreloadPhi(phiGm_, l1_[tiling_->l1BOffsetElems], tiling_->k, kBegin, kValid);
                    mm_.ComputeMm1(l1_[L1_GQ_OFFSET], phiL1, mm1Local_, kValid);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_READY);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_READY + SECOND_AIV_FLAG_OFFSET);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_X_READY);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_X_READY + SECOND_AIV_FLAG_OFFSET);
                    mm_.ComputeMm2(l1_[L1_GQ_OFFSET], l1_[l1XOffset_], kValid, true, true,
                                   AicWaveKRows(blockRowBegin + wave * MM_TILE_M));
                    mm_.CopyMm2ToAiv(mm2Local_, kValid);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM2_READY);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM2_READY + SECOND_AIV_FLAG_OFFSET);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MM1_READY);
                    vector_.ComputeGradXAndStageX(rowBase, validRows, kBegin, kValid, mm1Local_, grLocal_);
                    StageXToL1(subBlock);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MM2_READY);
                    ReducePhiMSplit(kBegin, kValid, logicalBlock, subBlock);
                }
            }
        }
    }

    __aicore__ inline void PrepareMkWorkspace()
    {
        if ASCEND_IS_AIV {
            InitPrepareStage();
            uint32_t vecId = GetBlockIdx();
            if (vecId >= tiling_->usedPrepAiv) {
                return;
            }
            AscendC::Duplicate(rowPart_, 0.0f, shape_.scalarReduceElements);
            uint64_t gqOffset = tiling_->offGQ / sizeof(float);
            uint64_t grOffset = tiling_->offGR / sizeof(float);
            for (int64_t row = vecId * PREPARE_BATCH; row < tiling_->bs; row += tiling_->usedPrepAiv * PREPARE_BATCH) {
                uint32_t rows = tiling_->bs - row < PREPARE_BATCH ? tiling_->bs - row : PREPARE_BATCH;
                LocalTensor<float> output = prepareOutQue_.AllocTensor<float>();
                vector_.PrepareRows(row, rows, output, output[prepareOutGrOffset_], rowPartCall_);
                VFAdd(rowPart_, rowPart_, rowPartCall_, shape_.scalarReduceElements);
                prepareOutQue_.EnQue(output);
                output = prepareOutQue_.DeQue<float>();
                DataCopyExtParams copy = {static_cast<uint16_t>(rows),
                                          static_cast<uint32_t>(shape_.hValue * sizeof(float)),
                                          (shape_.hAlign - shape_.hValue) / FLOAT_C0, 0, 0};
                DataCopyPad(workspaceGm_[gqOffset + row * shape_.hValue], output, copy);
                CopyUbToGm(workspaceGm_[grOffset + row], output[prepareOutGrOffset_], rows);
                prepareOutQue_.FreeTensor(output);
            }
            uint64_t rowPartOffset = tiling_->offRowPart / sizeof(float) + vecId * shape_.scalarReduceElements;
            LocalTensor<float> output = prepareOutQue_.AllocTensor<float>();
            DataCopy(output, rowPart_, shape_.scalarReduceAligned);
            prepareOutQue_.EnQue(output);
            output = prepareOutQue_.DeQue<float>();
            CopyUbToGm(workspaceGm_[rowPartOffset], output, shape_.scalarReduceElements);
            prepareOutQue_.FreeTensor(output);
        }
    }

    __aicore__ inline void ReduceMkScalars()
    {
        if ASCEND_IS_AIV {
            InitMkScalarReduceStage();
            if (GetBlockIdx() == 0) {
                LocalTensor<float> output = scalarOutQue_.AllocTensor<float>();
                uint64_t base = tiling_->offRowPart / sizeof(float);
                LocalTensor<float> partial = scalarInQue_.AllocTensor<float>();
                uint32_t biasStride = AlignU32(shape_.hValue, FLOAT_C0);
                // rowPart packs [alpha(3)][bias(hcMix)]; copy each segment into 8-aligned rows before reduction.
                DataCopyPadExtParams<float> alphaPad = {true, 0, FLOAT_C0 - 3, 0};
                DataCopyExtParams alphaCopy = {static_cast<uint16_t>(tiling_->usedPrepAiv), 3 * sizeof(float),
                                               static_cast<uint32_t>(shape_.hValue * sizeof(float)), 0, 0};
                DataCopyPad(partial, workspaceGm_[base], alphaCopy, alphaPad);
                DataCopyPadExtParams<float> biasPad = {true, 0, static_cast<uint8_t>(biasStride - shape_.hValue), 0};
                DataCopyExtParams biasCopy = {static_cast<uint16_t>(tiling_->usedPrepAiv),
                                              static_cast<uint32_t>(shape_.hValue * sizeof(float)), 3 * sizeof(float),
                                              0, 0};
                DataCopyPad(partial[tiling_->usedPrepAiv * FLOAT_C0], workspaceGm_[base + 3], biasCopy, biasPad);
                scalarInQue_.EnQue(partial);
                partial = scalarInQue_.DeQue<float>();
                VFSumRows(output, partial, tiling_->usedPrepAiv, 3, FLOAT_C0);
                VFSumRows(output[FLOAT_C0], partial[tiling_->usedPrepAiv * FLOAT_C0], tiling_->usedPrepAiv,
                          shape_.hValue, biasStride);
                scalarInQue_.FreeTensor(partial);
                scalarOutQue_.EnQue(output);
                output = scalarOutQue_.DeQue<float>();
                WriteScalarOutputs(output, 3, shape_.hValue);
                scalarOutQue_.FreeTensor(output);
            }
        }
    }

    __aicore__ inline void LoadMkCache(int64_t blockRowBegin, uint32_t subBlock, uint32_t firstWave, uint32_t waveCount)
    {
        if ASCEND_IS_AIV {
            InitMkComputeStage();
            for (uint32_t wave = 0; wave < waveCount; ++wave) {
                int64_t rowBase = blockRowBegin + (firstWave + wave) * MM_TILE_M + subBlock * ROWS_PER_AIV;
                LoadLocalRowsFromWorkspace(rowBase, ValidRows(rowBase), wave * ROWS_PER_AIV);
                StageGqToL1(subBlock, ValidRows(rowBase), wave, false);
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_GQ_READY);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_GQ_READY);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_GQ_READY + SECOND_AIV_FLAG_OFFSET);
        }
    }

    __aicore__ inline void ComputeMkWaves(int64_t blockRowBegin, uint32_t subBlock, uint32_t kBegin, uint32_t kValid,
                                          uint32_t firstWave, uint32_t waveCount, const LocalTensor<float>& phiL1)
    {
        for (uint32_t wave = 0; wave < waveCount; ++wave) {
            uint32_t waveSlot = wave & 1;
            uint32_t globalWave = firstWave + wave;
            int64_t rowBase = blockRowBegin + globalWave * MM_TILE_M + subBlock * ROWS_PER_AIV;
            uint32_t validRows = ValidRows(rowBase);
            if ASCEND_IS_AIV {
                if (wave >= 2) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_INPUTS_RELEASED);
                }
                vector_.template ComputeGradXAndStageX<true>(rowBase, validRows, kBegin, kValid, mm1Local_,
                                                             grLocal_[wave * ROWS_PER_AIV], wave * ROWS_PER_AIV);
                StageXToL1(subBlock, false, waveSlot);
            } else {
                if (wave >= 2) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_RELEASED);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_RELEASED + SECOND_AIV_FLAG_OFFSET);
                }
                mm_.ComputeMm1(l1_[L1_GQ_OFFSET + wave * MM_TILE_M * shape_.hAlign], phiL1,
                               mm1Local_[waveSlot * MM1_SLOT_ELEMENTS], kValid, globalWave + 1 == tiling_->maxMWave,
                               globalWave == 0);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_READY);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_READY + SECOND_AIV_FLAG_OFFSET);
            }
            if ASCEND_IS_AIV {
                if (wave != 0) {
                    uint32_t previousSlot = (wave - 1) & 1;
                    int64_t previousRow = rowBase - MM_TILE_M;
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MM1_READY);
                    vector_.FinishGradX(previousRow, ValidRows(previousRow), kBegin, kValid,
                                        mm1Local_[previousSlot * MM1_SLOT_ELEMENTS], previousSlot);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_MM1_RELEASED);
                }
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_X_READY);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_X_READY + SECOND_AIV_FLAG_OFFSET);
                mm_.ComputeMm2(l1_[L1_GQ_OFFSET + wave * MM_TILE_M * shape_.hAlign],
                               l1_[l1XOffset_ + waveSlot * MM_TILE_M * K_TILE_VALUE], kValid, globalWave == 0,
                               globalWave + 1 == tiling_->maxMWave,
                               AicWaveKRows(blockRowBegin + globalWave * MM_TILE_M));
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_INPUTS_RELEASED);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_INPUTS_RELEASED + SECOND_AIV_FLAG_OFFSET);
            }
        }
        if ASCEND_IS_AIV {
            uint32_t lastWave = waveCount - 1;
            uint32_t lastSlot = lastWave & 1;
            int64_t lastRow = blockRowBegin + (firstWave + lastWave) * MM_TILE_M + subBlock * ROWS_PER_AIV;
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MM1_READY);
            vector_.FinishGradX(lastRow, ValidRows(lastRow), kBegin, kValid, mm1Local_[lastSlot * MM1_SLOT_ELEMENTS],
                                lastSlot);
            CrossCoreSetFlag<SYNC_MODE, PIPE_V>(FLAG_MM1_RELEASED);
            for (uint32_t drain = 0; drain < 2 && drain < waveCount; ++drain) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_INPUTS_RELEASED);
            }
        } else {
            for (uint32_t drain = 0; drain < 2 && drain < waveCount; ++drain) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_RELEASED);
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_MM1_RELEASED + SECOND_AIV_FLAG_OFFSET);
            }
        }
    }

    __aicore__ inline void ProcessMkSplit()
    {
        PrepareMkWorkspace();
        SyncAll<false>();
        ReduceMkScalars();

        uint32_t block = GetBlockIdx();
        uint32_t lane = block;
        uint32_t subBlock = 0;
        if ASCEND_IS_AIV {
            lane = block / 2;
            subBlock = block % 2;
        }
        uint32_t mBlock = lane / tiling_->kBlockCount;
        uint32_t kBlock = lane % tiling_->kBlockCount;
        int64_t blockRowBegin = mBlock * tiling_->mBlockRows;
        const bool cachedM = tiling_->maxMWave <= l1GqWaves_;
        if (cachedM) {
            LoadMkCache(blockRowBegin, subBlock, 0, cacheWaves_);
        }
        // Use phi prefetched into the B1 ping-pong buffer at the end of the previous kTile, or load it here initially.
        LocalTensor<float> phiL1;
        bool phiPrefetched = false;
        for (uint32_t kTile = 0; kTile < tiling_->gradPhiLoopK; ++kTile) {
            uint32_t kBegin = kBlock * tiling_->kBlockSize + kTile * K_TILE_VALUE;
            uint32_t kValid =
                kBegin >= tiling_->k ? 0 : (tiling_->k - kBegin < K_TILE_VALUE ? tiling_->k - kBegin : K_TILE_VALUE);
            uint32_t slot = kTile;
            if ASCEND_IS_AIV {
                if (cachedM) {
                    InitMkComputeStage();
                }
            }
            if (kValid != 0) {
                if ASCEND_IS_AIC {
                    if (!phiPrefetched) {
                        phiL1 = mm_.PreloadPhi(phiGm_, l1_[tiling_->l1BOffsetElems], tiling_->k, kBegin, kValid);
                    }
                    phiPrefetched = false;
                }
                for (uint32_t firstWave = 0; firstWave < tiling_->maxMWave; firstWave += cacheWaves_) {
                    uint32_t remainingWaves = tiling_->maxMWave - firstWave;
                    uint32_t waveCount = remainingWaves < cacheWaves_ ? remainingWaves : cacheWaves_;
                    if (!cachedM) {
                        LoadMkCache(blockRowBegin, subBlock, firstWave, waveCount);
                    }
                    ComputeMkWaves(blockRowBegin, subBlock, kBegin, kValid, firstWave, waveCount, phiL1);
                }
                if ASCEND_IS_AIC {
                    // Prefetch phi for the next kTile into the other B1 slot, overlapping the final MM2 wave and output
                    // transfer.
                    uint32_t nextKBegin = kBegin + K_TILE_VALUE;
                    if (kTile + 1 < tiling_->gradPhiLoopK && nextKBegin < tiling_->k) {
                        uint32_t nextKValid =
                            tiling_->k - nextKBegin < K_TILE_VALUE ? tiling_->k - nextKBegin : K_TILE_VALUE;
                        phiL1 =
                            mm_.PreloadPhi(phiGm_, l1_[tiling_->l1BOffsetElems], tiling_->k, nextKBegin, nextKValid);
                        phiPrefetched = true;
                    }
                    // Write a single mBlock directly to GM; otherwise write per-mBlock workspace partitions and reduce
                    // in a fixed order.
                    if (tiling_->mBlockCount == 1) {
                        mm_.CopyMm2ToGm(gradPhiGm_[kBegin], kValid, tiling_->k);
                    } else {
                        mm_.CopyMm2ToGm(workspaceGm_[GradPhiPartOffset(kBlock, mBlock, slot)], kValid, K_TILE_VALUE);
                    }
                }
            }
        }
        if (tiling_->mBlockCount > 1) {
            SyncAll<false>();
            for (uint32_t reduceTile = 0; reduceTile < tiling_->gradPhiLoopK; ++reduceTile) {
                uint32_t reduceBegin = kBlock * tiling_->kBlockSize + reduceTile * K_TILE_VALUE;
                uint32_t reduceValid =
                    reduceBegin >= tiling_->k ?
                        0 :
                        (tiling_->k - reduceBegin < K_TILE_VALUE ? tiling_->k - reduceBegin : K_TILE_VALUE);
                if (reduceValid != 0) {
                    ReduceMkPhi(kBlock, reduceBegin, reduceValid, reduceTile, mBlock, subBlock);
                }
            }
            SyncAll<false>();
        }
    }

    __aicore__ inline void InitMkPhiReduceStage()
    {
        ResetStagePool();
        uint32_t elements = CeilDivU32(shape_.hValue, tiling_->mBlockCount * 2) * K_TILE_VALUE;
        stagePool_.InitBuffer(phiInQue_, 1, tiling_->mBlockCount * elements * sizeof(float));
        stagePool_.InitBuffer(phiOutQue_, 1, elements * sizeof(float));
    }

    __aicore__ inline uint64_t GradPhiPartOffset(uint32_t kBlock, uint32_t mBlock, uint32_t slot) const
    {
        return tiling_->offGradPhiPart / sizeof(float) +
               ((static_cast<uint64_t>(kBlock) * tiling_->mBlockCount + mBlock) * tiling_->gradPhiLoopK + slot) *
                   shape_.hValue * K_TILE_VALUE;
    }

    __aicore__ inline void ReduceMkPhi(uint32_t kBlock, uint32_t kBegin, uint32_t kValid, uint32_t slot,
                                       uint32_t mBlock, uint32_t subBlock)
    {
        if ASCEND_IS_AIV {
            InitMkPhiReduceStage();
            uint32_t hValue = shape_.hValue;
            uint32_t rowsPerReducer = CeilDivU32(hValue, tiling_->mBlockCount * 2);
            uint32_t hBegin = (mBlock * 2 + subBlock) * rowsPerReducer;
            if (hBegin >= hValue) {
                return;
            }
            uint32_t rows = hValue - hBegin < rowsPerReducer ? hValue - hBegin : rowsPerReducer;
            uint32_t elements = rows * K_TILE_VALUE;
            LocalTensor<float> output = phiOutQue_.AllocTensor<float>();
            uint64_t part = GradPhiPartOffset(kBlock, 0, slot) + hBegin * K_TILE_VALUE;
            LocalTensor<float> partial = phiInQue_.AllocTensor<float>();
            DataCopyExtParams copy = {
                static_cast<uint16_t>(tiling_->mBlockCount), static_cast<uint32_t>(elements * sizeof(float)),
                static_cast<uint32_t>((tiling_->gradPhiLoopK * shape_.hValue * K_TILE_VALUE - elements) *
                                      sizeof(float)),
                0, 0};
            DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
            DataCopyPad(partial, workspaceGm_[part], copy, pad);
            phiInQue_.EnQue(partial);
            partial = phiInQue_.DeQue<float>();
            VFSumRows(output, partial, static_cast<uint16_t>(tiling_->mBlockCount), elements, elements);
            phiInQue_.FreeTensor(partial);
            phiOutQue_.EnQue(output);
            output = phiOutQue_.DeQue<float>();
            DataCopyExtParams outCopy = {static_cast<uint16_t>(rows), static_cast<uint32_t>(kValid * sizeof(float)),
                                         (K_TILE_VALUE - kValid) / FLOAT_C0,
                                         static_cast<uint32_t>((tiling_->k - kValid) * sizeof(float)), 0};
            DataCopyPad(gradPhiGm_[hBegin * tiling_->k + kBegin], output, outCopy);
            phiOutQue_.FreeTensor(output);
        }
    }

    const MhcPreSinkhornBackwardArch35TilingData* tiling_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t cacheWaves_ = 0;
    uint32_t l1GqWaves_ = 0;
    GlobalTensor<float> phiGm_;
    GlobalTensor<float> gradPhiGm_;
    GlobalTensor<float> gradAlphaGm_;
    GlobalTensor<float> gradBiasGm_;
    GlobalTensor<float> workspaceGm_;
    TBuf<TPosition::A1> l1Buf_;
    TBufPool<QuePosition::VECCALC, ROOT_POOL_BUFFER_COUNT> rootPool_;
    BackwardStagePool stageStorage_;
    BackwardStagePool stagePool_;
    TBuf<TPosition::VECCALC> mm1Buf_;
    TBuf<TPosition::VECCALC> mm2Buf_;
    TBuf<TPosition::VECCALC> gqBuf_;
    TBuf<TPosition::VECCALC> grBuf_;
    TBuf<TPosition::VECCALC> rowPartBuf_;
    TBuf<TPosition::VECCALC> rowPartCallBuf_;
    LocalTensor<float> l1_;
    LocalTensor<float> mm1Local_;
    LocalTensor<float> mm2Local_;
    LocalTensor<float> gqNd_;
    LocalTensor<float> grLocal_;
    LocalTensor<float> rowPart_;
    LocalTensor<float> rowPartCall_;
    MhcPreSinkhornBackwardMm mm_;
    MhcPreSinkhornBackwardVector<T> vector_;

    TQue<QuePosition::VECIN, 1> workspaceRowsInQue_;
    TQue<QuePosition::VECIN, 1> scalarInQue_;
    TQue<QuePosition::VECIN, 1> phiInQue_;
    TQue<QuePosition::VECOUT, 1> clearOutQue_;
    TQue<QuePosition::VECOUT, 1> prepareOutQue_;
    TQue<QuePosition::VECOUT, 1> gqL1OutQue_;
    TQue<QuePosition::VECOUT, 1> scalarOutQue_;
    TQue<QuePosition::VECOUT, 1> phiOutQue_;
};
} // namespace MhcPreSinkhornBackwardNs

#endif // MHC_PRE_SINKHORN_BACKWARD_KERNEL_ARCH35_H
