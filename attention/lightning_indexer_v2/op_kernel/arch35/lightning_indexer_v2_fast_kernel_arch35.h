/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH35_H
#define LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH35_H

#include <type_traits>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lightning_indexer_v2_common_arch35.h"
#include "lightning_indexer_v2_fast_common_arch35.h"
#include "lightning_indexer_v2_fast_service_cube_arch35.h"
#include "lightning_indexer_v2_fast_service_vector_arch35.h"

namespace LIV2Kernel {

template <typename LIT>
class LightningIndexerV2FastKernel {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using W_T = float;
    using OUT_T = typename LIT::outputType;

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
                                const LIV2TilingData* tiling, TPipe* pipe, __gm__ uint8_t* actualSeqLengthsQ,
                                __gm__ uint8_t* actualSeqLengthsK, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* usedKeyLengths, __gm__ uint8_t* metadata)
    {
        fastMetadataGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
        InitParams(tiling);
        returnValue_ = tiling->returnValue;
        sequencePlan_.Init(tiling, actualSeqLengthsQ, actualSeqLengthsK, usedKeyLengths);
        if (constInfo_.blockIdx >= constInfo_.blockNum) {
            return;
        }
        InitBuffers(pipe);
        if ASCEND_IS_AIC {
            queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T*>(query));
            keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ K_T*>(key));
            cubeService_.InitPageAttention(tiling, blockTable);
        } else {
            weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T*>(weights));
            indiceOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_T*>(sparseIndices));
            if (returnValue_) {
                valueOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sparseValues));
            }
            vectorService_.InitParams(constInfo_);
            vectorService_.InitWorkspace(workspace);
        }
    }

    __aicore__ inline void Process()
    {
        if (constInfo_.blockIdx >= constInfo_.blockNum) {
            return;
        }
        // Metadata assigns a contiguous interval of global Q8 tiles. Intersect it
        // with each batch, then bind batch-local GM addresses and tile bounds.
        uint32_t batchTileStart = 0U;
        for (uint32_t batch = 0; batch < sequencePlan_.GetBatchSize(); ++batch) {
            if (batchTileStart >= taskTileEnd_) {
                break;
            }
            const auto sequence = sequencePlan_.GetSequence(batch);
            const uint32_t queryTiles = (sequence.querySize + LIV2Fast::kTileS1 - 1U) / LIV2Fast::kTileS1;
            const uint32_t batchTileEnd = batchTileStart + queryTiles;
            const uint32_t start = max(taskTileStart_, batchTileStart);
            const uint32_t end = min(taskTileEnd_, batchTileEnd);
            const uint32_t localStart = start - batchTileStart;
            const uint32_t localEnd = end - batchTileStart;
            batchTileStart = batchTileEnd;
            if (start >= end) {
                continue;
            }
            constInfo_.firstQueryTile = localStart;
            constInfo_.queryTileEnd = localEnd;
            constInfo_.s1Size = sequence.querySize;
            constInfo_.s2Size = sequence.keySize;
            if ASCEND_IS_AIC {
                cubeService_.InitParams(constInfo_);
                cubeService_.SetBatchIndex(batch);
                cubeService_.InitMm1GlobalTensor(
                    queryGm_[sequence.queryOffset * constInfo_.headCount * LIV2Fast::kHeadDim],
                    keyGm_[sequence.keyOffset * LIV2Fast::kHeadDim]);
            } else {
                vectorService_.InitParams(constInfo_);
                vectorService_.InitVecInputTensor(
                    weightsGm_[sequence.queryOffset * constInfo_.headCount],
                    indiceOutGm_[sequence.queryOffset * constInfo_.outputTopK],
                    returnValue_ ? valueOutGm_.GetPhyAddr(sequence.queryOffset * constInfo_.outputTopK) : nullptr);
            }
            ProcessSequence();
        }
    }

private:
    __aicore__ inline void ProcessSequence()
    {
        if ASCEND_IS_AIC {
            cubeService_.Process();
        } else {
            vectorService_.Process();
        }
    }

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        if ASCEND_IS_AIC {
            cubeService_.InitBuffers(pipe);
        } else {
            vectorService_.InitBuffers(pipe);
        }
    }

    __aicore__ inline void InitParams(const LIV2TilingData* tiling)
    {
        constInfo_.s1Size = tiling->s1Size;
        constInfo_.s2Size = tiling->s2Size;
        constInfo_.sparseMode = tiling->maskMode;
        constInfo_.outputTopK = tiling->topk;
        constInfo_.headCount = tiling->gSize;
        constInfo_.blockNum = fastMetadataGm_.GetValue(optiling::LI_V2_FAST_CORE_NUM_INDEX);
        const uint32_t physicalBlock = GetBlockIdx();
        if ASCEND_IS_AIC {
            constInfo_.blockIdx = physicalBlock;
        } else {
            constInfo_.blockIdx = physicalBlock / LIV2Fast::kAivPerAic;
            constInfo_.subBlockIdx = physicalBlock % LIV2Fast::kAivPerAic;
        }
        if (constInfo_.blockIdx < constInfo_.blockNum) {
            taskTileStart_ = fastMetadataGm_.GetValue(optiling::LI_V2_FAST_TILE_START_INDEX + constInfo_.blockIdx);
            taskTileEnd_ = fastMetadataGm_.GetValue(optiling::LI_V2_FAST_TILE_END_INDEX + constInfo_.blockIdx);
        }
    }

    LightningIndexerV2FastSequencePlan<LIT> sequencePlan_;
    LIV2Fast::FastConstInfo constInfo_{};
    uint32_t taskTileStart_{0U};
    uint32_t taskTileEnd_{0U};
    GlobalTensor<uint32_t> fastMetadataGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<W_T> weightsGm_;
    GlobalTensor<OUT_T> indiceOutGm_;
    GlobalTensor<float> valueOutGm_;
    bool returnValue_{false};
    LightningIndexerV2FastServiceCube<LIT> cubeService_;
    LightningIndexerV2FastServiceVector<LIT> vectorService_;
};

template <typename LIT>
__aicore__ inline void RunLightningIndexerV2FastPath(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights, __gm__ uint8_t* sparseIndices,
    __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace, const LIV2TilingData* tiling, TPipe* pipe,
    __gm__ uint8_t* actualSeqLengthsQ = nullptr, __gm__ uint8_t* actualSeqLengthsK = nullptr,
    __gm__ uint8_t* blockTable = nullptr, __gm__ uint8_t* usedKeyLengths = nullptr, __gm__ uint8_t* metadata = nullptr)
{
    if constexpr ((std::is_same_v<typename LIT::queryType, bfloat16_t> ||
                   std::is_same_v<typename LIT::queryType, half>) &&
                  std::is_same_v<typename LIT::keyType, typename LIT::queryType> && LIT::weightsTypeFlag) {
        LightningIndexerV2FastKernel<LIT> kernel;
        kernel.Init(query, key, weights, sparseIndices, sparseValues, workspace, tiling, pipe, actualSeqLengthsQ,
                    actualSeqLengthsK, blockTable, usedKeyLengths, metadata);
        kernel.Process();
    }
}

} // namespace LIV2Kernel
#endif // LIGHTNING_INDEXER_V2_FAST_KERNEL_ARCH35_H
