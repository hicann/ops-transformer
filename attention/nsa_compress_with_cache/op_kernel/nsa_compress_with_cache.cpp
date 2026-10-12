/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
using namespace AscendC;

namespace NsaCompressWithCacheBase {
constexpr uint32_t BROADCAST_DIM_NUM_1 = 1;
constexpr uint32_t BROADCAST_DIM_SIZE_2 = 2;
constexpr uint32_t MAX_BATCH_SIZE = 2048;
constexpr uint32_t FP32_NUM_PER_UB_BLOCK = 8;
constexpr uint32_t FP32_NUM_PER_UB_REPEAT = 64;
constexpr uint32_t FP16_NUM_PER_UB_BLOCK = 16;
constexpr uint32_t NUM_2 = 2;

#define INVOKE_NCWC_GENERAL_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<__VA_ARGS__> op; \
        op.Init(input, weight, slotMapping, output_cache_ref, act_seq_len, block_table, &tiling_data); \
        op.Process(); \
    } while (0)

template <typename T>
class KernelNsaCompressWithCacheBase {
public:
    __aicore__ inline KernelNsaCompressWithCacheBase() {}

protected:
    __aicore__ inline void InitBase(GM_ADDR kv_cache, GM_ADDR weight, GM_ADDR slot_mapping, GM_ADDR compress_kv_cache,
                                    GM_ADDR act_seq_len, GM_ADDR block_table,
                                    const NsaCompressWithCacheTilingData* __restrict tilingData)
    {
        this->tilingData = tilingData;
        this->blockIdx = AscendC::GetBlockIdx();
        this->coreStartHeadIdx = this->tilingData->coreStartHeadIdxList[this->blockIdx];
        this->repeatNum = (tilingData->coreStartHeadIdxList[this->blockIdx + 1] - this->coreStartHeadIdx) /
                          tilingData->headsNumPerCore;
        this->startBatchIdx = tilingData->coreStartBatchIdxList[this->blockIdx];
        this->endBatchIdx = tilingData->coreStartBatchIdxList[this->blockIdx + 1];

        kvCalcBufSize =
            (tilingData->tokenNumPerTile + tilingData->tokenRepeatCount) * tilingData->compressKvCacheSizePerCore;
        this->kvCacheGm.SetGlobalBuffer((__gm__ T*)kv_cache, this->tilingData->kvCacheSize);
        this->weightGm.SetGlobalBuffer((__gm__ T*)weight, this->tilingData->weightSize);
        this->slotMappingGm.SetGlobalBuffer((__gm__ int32_t*)slot_mapping, this->tilingData->batchSize);
        this->compressKvCacheGm.SetGlobalBuffer((__gm__ T*)compress_kv_cache, this->tilingData->compressKvCacheSize);
        this->actSeqLenGm.SetGlobalBuffer((__gm__ int64_t*)act_seq_len, this->tilingData->batchSize);
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t*)block_table, this->tilingData->blockTableSize);
        uint32_t inQueueSize = tilingData->kvCacheSizePerCore > tilingData->weightSize ?
                                   tilingData->kvCacheSizePerCore :
                                   tilingData->weightSize;
        this->pipe.InitBuffer(this->inQueue, this->tilingData->bufferNum, inQueueSize * sizeof(T));
        this->pipe.InitBuffer(this->outQueue, this->tilingData->bufferNum,
                              this->tilingData->compressKvCacheSizePerCore * sizeof(T));
        this->pipe.InitBuffer(this->kvCalcBuf, this->tilingData->bufferNum * kvCalcBufSize * sizeof(float));
    }

    __aicore__ inline void InitActSeqLen()
    {
        uint32_t idx = coreStartHeadIdx / tilingData->headNum;
        compressBatchIdxList[idx++] = startBatchIdx;
        for (uint32_t i = startBatchIdx + 1; i < endBatchIdx; i++) {
            int64_t curSeqLen = actSeqLenGm.GetValue(i);
            if (curSeqLen >= tilingData->compressBlockSize &&
                (curSeqLen - tilingData->compressBlockSize) % tilingData->compressStride == 0) {
                compressBatchIdxList[idx++] = i;
            }
        }
        compressBatchIdxList[idx] = endBatchIdx;
    }

    __aicore__ inline void InitWeight()
    {
        AscendC::LocalTensor<T> weightLocal = inQueue.AllocTensor<T>();
        AscendC::DataCopy(weightLocal, weightGm, tilingData->weightSize);
        inQueue.EnQue<T>(weightLocal);
        weightLocal = inQueue.DeQue<T>();

        AscendC::LocalTensor<float> castWeightCalcLocal = castWeightBuf.template Get<float>();
        AscendC::Cast(castWeightCalcLocal, weightLocal, AscendC::RoundMode::CAST_NONE, tilingData->weightSize);
        pipe_barrier(PIPE_ALL);
        inQueue.FreeTensor(weightLocal);
    }

    __aicore__ inline void InitOffset(uint32_t idx)
    {
        pingPongIdx = idx % this->tilingData->bufferNum;
        uint32_t totalHeadIdx = coreStartHeadIdx + idx * tilingData->headsNumPerCore;
        uint32_t compressBatchIdx = totalHeadIdx / tilingData->headNum;
        headIdx = totalHeadIdx - compressBatchIdx * tilingData->headNum;
        batchIdx = compressBatchIdxList[compressBatchIdx];
        int64_t seqLen = actSeqLenGm.GetValue(batchIdx);
        seqLenStartidx = seqLen - tilingData->compressBlockSize;
        int32_t slotMappingIdx = slotMappingGm.GetValue(batchIdx);
        outputOffset = static_cast<uint64_t>(slotMappingIdx) * tilingData->headDim * tilingData->headNum +
                       static_cast<uint64_t>(headIdx) * tilingData->headDim;
    }

    __aicore__ inline void CopyKvCache(uint32_t tileIdx)
    {
        pageBlockUnitOffset =
            static_cast<uint64_t>(tilingData->pageBlockSize) * tilingData->headNum * tilingData->headDim;
        AscendC::LocalTensor<T> kvCacheLocal = inQueue.AllocTensor<T>();
        uint32_t curTokenIdx = seqLenStartidx + tileIdx * this->tilingData->tokenNumPerTile;
        uint64_t curBlockPageIdx = curTokenIdx / tilingData->pageBlockSize;
        uint64_t curPageOffset =
            (curTokenIdx - curBlockPageIdx * tilingData->pageBlockSize) * tilingData->headNum * tilingData->headDim;

        curBlockPageIdx = blockTableGm.GetValue(batchIdx * tilingData->pageNumPerBatch + curBlockPageIdx);
        uint64_t curGMOffset = curBlockPageIdx * pageBlockUnitOffset + curPageOffset + headIdx * tilingData->headDim;

        AscendC::DataCopyParams repeatParams;
        repeatParams.blockLen = tilingData->headDim * tilingData->headsNumPerCore / FP16_NUM_PER_UB_BLOCK;
        repeatParams.srcStride = tilingData->headDim * tilingData->headsNumPerCore *
                                 (tilingData->coresNumPerCompress - 1) / FP16_NUM_PER_UB_BLOCK;
        repeatParams.dstStride = 0;
        repeatParams.blockCount = tilingData->tokenNumPerTile;

        AscendC::DataCopy(kvCacheLocal, kvCacheGm[curGMOffset], repeatParams);
        inQueue.EnQue(kvCacheLocal);
    }

    __aicore__ inline void ReduceBlock(AscendC::LocalTensor<float> tensor, uint32_t reduceSize,
                                       uint32_t alignReduceSize)
    {
        // align step: align reduce size to power of two, such as 15->8, 16->8
        uint32_t addVectorNum = reduceSize - alignReduceSize;
        AscendC::Add(tensor, tensor, tensor[alignReduceSize * tilingData->compressKvCacheSizePerCore],
                     addVectorNum * tilingData->compressKvCacheSizePerCore);
        PipeBarrier<PIPE_V>();
        // reduce step: reduce compressBlockSize to 1
        while (alignReduceSize > 1) {
            alignReduceSize = alignReduceSize >> 1;
            AscendC::Add(tensor, tensor, tensor[alignReduceSize * tilingData->compressKvCacheSizePerCore],
                         alignReduceSize * tilingData->compressKvCacheSizePerCore);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeSubTile(uint32_t repeatIdx)
    {
        AscendC::LocalTensor<T> kvCacheLocal = inQueue.DeQue<T>();
        // cast fp16 to fp32
        AscendC::LocalTensor<float> castKvCacheLocal = kvCalcBuf.template GetWithOffset<float>(
            tilingData->kvCacheSizePerCore,
            (tilingData->compressKvCacheSizePerCore * repeatIdx + kvCalcBufSize * pingPongIdx) * sizeof(float));
        AscendC::Cast(castKvCacheLocal, kvCacheLocal, AscendC::RoundMode::CAST_NONE, tilingData->kvCacheSizePerCore);
        PipeBarrier<PIPE_V>();
        inQueue.FreeTensor(kvCacheLocal);

        AscendC::LocalTensor<float> broadcastWeightLocal =
            broadcastWeightBuf.template Get<float>()[this->tilingData->weightSizePerCore * this->pingPongIdx];

        // elem-wise mul & reduce
        AscendC::Mul(castKvCacheLocal, castKvCacheLocal, broadcastWeightLocal, tilingData->kvCacheSizePerCore);
        PipeBarrier<PIPE_V>();
        ReduceBlock(castKvCacheLocal, this->tilingData->tokenNumPerTile, this->tilingData->tokenNumPerTile / NUM_2);
    }

    __aicore__ inline void ComputeReduce()
    {
        AscendC::LocalTensor<float> castKvCacheLocal = kvCalcBuf.template Get<float>()[kvCalcBufSize * pingPongIdx];

        // reduce
        ReduceBlock(castKvCacheLocal, this->tilingData->tokenRepeatCount, tilingData->alignReduceSize);

        // cast fp32 to fp16
        AscendC::LocalTensor<T> compressKvCacheLocal = outQueue.AllocTensor<T>();
        AscendC::Cast(compressKvCacheLocal, castKvCacheLocal, AscendC::RoundMode::CAST_RINT,
                      tilingData->compressKvCacheSizePerCore);
        PipeBarrier<PIPE_V>();
        outQueue.EnQue<T>(compressKvCacheLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> compressKvCacheLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(compressKvCacheGm[outputOffset], compressKvCacheLocal,
                          tilingData->compressKvCacheSizePerCore);
        outQueue.FreeTensor(compressKvCacheLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<T> kvCacheGm;
    AscendC::GlobalTensor<T> weightGm;
    AscendC::GlobalTensor<T> compressKvCacheGm;
    AscendC::GlobalTensor<int32_t> blockTableGm;
    AscendC::GlobalTensor<int64_t> actSeqLenGm;
    AscendC::GlobalTensor<int32_t> slotMappingGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC> kvCalcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castWeightBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastWeightBuf;

    uint32_t broadcastWeightOffset;
    uint32_t broadcastWeightLen;
    uint32_t castWeightOffset;
    uint32_t castWeightLen;
    uint32_t firstBroadcastWeightLen;
    uint32_t firstBroadcastWeightOffset;
    uint32_t tileWeightLen;
    uint32_t tileWeighttOffset;
    uint32_t seqLenStartidx;
    uint32_t startBatchIdx;
    uint32_t endBatchIdx;
    uint32_t pingPongIdx;
    uint32_t kvCalcBufSize;

    uint32_t blockIdx;
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t coreStartHeadIdx;
    uint32_t coreBatchStartIdx;
    uint32_t repeatNum;
    uint32_t pageNum;
    uint32_t firstPageIdx;
    uint32_t firstPageTokenNum;
    uint32_t firstPageOffset;
    uint32_t lastPageIdx;
    uint32_t lastPageTokenNum;
    uint32_t lastPageOffset;
    uint64_t pageBlockUnitOffset;
    uint32_t pageBlockSize;
    uint32_t compressBatchSize;
    uint64_t outputOffset;
    uint32_t compressBatchIdxList[MAX_BATCH_SIZE];
    const NsaCompressWithCacheTilingData* __restrict tilingData;
}; // namespace NsaCompressWithCacheBasetemplate<typenameT>class KernelNsaCompressWithCacheBase

template <typename T>
class KernelNsaCompressWithCacheDoubleBroadcast : public KernelNsaCompressWithCacheBase<T> {
public:
    __aicore__ inline KernelNsaCompressWithCacheDoubleBroadcast() {}
    __aicore__ inline void Init(GM_ADDR kv_cache, GM_ADDR weight, GM_ADDR slot_mapping, GM_ADDR compress_kv_cache,
                                GM_ADDR act_seq_len, GM_ADDR block_table,
                                const NsaCompressWithCacheTilingData* __restrict tilingData)
    {
        this->InitBase(kv_cache, weight, slot_mapping, compress_kv_cache, act_seq_len, block_table, tilingData);

        this->castWeightLen = this->tilingData->weightSize;
        this->broadcastWeightLen = tilingData->compressKvCacheSizePerCore * this->tilingData->tokenNumPerTile;

        this->firstBroadCastSize = FP32_NUM_PER_UB_BLOCK / this->tilingData->headsNumPerCore;
        this->weightShape[0] = this->tilingData->weightSize;
        this->weightShape[1] = 1;
        this->firstBroadCastWeightShape[0] = this->tilingData->weightSize;
        this->firstBroadCastWeightShape[1] = this->firstBroadCastSize;

        this->tiledweightShape[0] =
            this->tilingData->tokenNumPerTile * this->tilingData->headsNumPerCore * this->firstBroadCastSize;
        this->tiledweightShape[1] = 1;
        this->finalBroadCastWeightShape[0] = tiledweightShape[0];
        this->finalBroadCastWeightShape[1] = this->tilingData->headDim / this->firstBroadCastSize;

        this->firstBroadcastWeightLen = this->tilingData->weightSize * this->firstBroadCastSize;
        this->tileWeightLen =
            this->tilingData->tokenNumPerTile * this->tilingData->headsNumPerCore * this->firstBroadCastSize;

        this->pipe.InitBuffer(this->castWeightBuf, (this->castWeightLen) * sizeof(float));
        this->pipe.InitBuffer(this->firstBroadCastWeightBuf,
                              this->tilingData->bufferNum * (this->firstBroadcastWeightLen) * sizeof(float));
        this->pipe.InitBuffer(this->tileWeightBuf, this->tilingData->bufferNum * (this->tileWeightLen) * sizeof(float));
        this->pipe.InitBuffer(this->broadcastWeightBuf,
                              this->tilingData->bufferNum * (this->broadcastWeightLen) * sizeof(float));

        this->InitActSeqLen();
    }
    __aicore__ inline void Process()
    {
        this->InitWeight();
        for (uint32_t idx = 0; idx < this->repeatNum; idx++) {
            this->InitOffset(idx);
            for (uint32_t tile_idx = 0; tile_idx < this->tilingData->tokenRepeatCount; tile_idx++) {
                this->CopyKvCache(tile_idx);
                InitWeightDoubleBroadCast(tile_idx);
                this->ComputeSubTile(tile_idx);
            }
            this->ComputeReduce();
            this->CopyOut();
        }
    }

protected:
    __aicore__ inline void InitWeightDoubleBroadCast(uint32_t tile_idx)
    {
        // 取全量weight
        AscendC::LocalTensor<float> castWeightCalcLocal = this->castWeightBuf.template Get<float>();
        // 第一次广播
        AscendC::LocalTensor<float> firstBroadcastWeightLocal =
            this->firstBroadCastWeightBuf.template Get<float>()[this->firstBroadcastWeightLen * this->pingPongIdx];
        AscendC::LocalTensor<uint8_t> tempLocal = this->broadcastWeightBuf.template Get<uint8_t>();
        AscendC::BroadCast<float, BROADCAST_DIM_SIZE_2, BROADCAST_DIM_NUM_1>(
            firstBroadcastWeightLocal, castWeightCalcLocal, this->firstBroadCastWeightShape, this->weightShape,
            tempLocal);
        PipeBarrier<PIPE_V>();
        // 跳读数据，切分headNum
        AscendC::LocalTensor<float> tileWeightCalcLocal =
            tileWeightBuf.template Get<float>()[this->tileWeightLen * this->pingPongIdx];
        AscendC::LocalTensor<float> broadcastWeightLocal =
            this->broadcastWeightBuf.template Get<float>()[this->broadcastWeightLen * this->pingPongIdx];
        AscendC::Copy(tileWeightCalcLocal,
                      firstBroadcastWeightLocal[tile_idx * this->tilingData->tokenNumPerTile *
                                                    this->tilingData->headNum * this->firstBroadCastSize +
                                                this->headIdx * this->firstBroadCastSize],
                      FP32_NUM_PER_UB_REPEAT, this->tilingData->tokenNumPerTile / FP32_NUM_PER_UB_BLOCK,
                      {1, static_cast<uint16_t>(this->tilingData->coresNumPerCompress),
                       static_cast<uint16_t>(FP32_NUM_PER_UB_BLOCK),
                       static_cast<uint16_t>(FP32_NUM_PER_UB_BLOCK * this->tilingData->coresNumPerCompress)});
        PipeBarrier<PIPE_V>();
        // 二次广播
        AscendC::BroadCast<float, BROADCAST_DIM_SIZE_2, BROADCAST_DIM_NUM_1>(
            broadcastWeightLocal, tileWeightCalcLocal, this->finalBroadCastWeightShape, this->tiledweightShape);
        PipeBarrier<PIPE_V>();
    }

    uint32_t firstBroadCastSize;
    uint32_t weightShape[BROADCAST_DIM_SIZE_2];
    uint32_t firstBroadCastWeightShape[BROADCAST_DIM_SIZE_2];
    uint32_t tiledweightShape[BROADCAST_DIM_SIZE_2];
    uint32_t finalBroadCastWeightShape[BROADCAST_DIM_SIZE_2];
    AscendC::TBuf<AscendC::TPosition::VECCALC> firstBroadCastWeightBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tileWeightBuf;
};

template <typename T>
class KernelNsaCompressWithCacheTileBroadcast : public KernelNsaCompressWithCacheBase<T> {
public:
    __aicore__ inline KernelNsaCompressWithCacheTileBroadcast() {}
    __aicore__ inline void Init(GM_ADDR kv_cache, GM_ADDR weight, GM_ADDR slot_mapping, GM_ADDR compress_kv_cache,
                                GM_ADDR act_seq_len, GM_ADDR block_table,
                                const NsaCompressWithCacheTilingData* __restrict tilingData)
    {
        this->InitBase(kv_cache, weight, slot_mapping, compress_kv_cache, act_seq_len, block_table, tilingData);

        this->tiledweightShape[0] = this->tilingData->tokenNumPerTile * this->tilingData->headsNumPerCore;
        this->tiledweightShape[1] = 1;
        this->finalBroadCastWeightShape[0] = this->tiledweightShape[0];
        this->finalBroadCastWeightShape[1] = this->tilingData->headDim;

        this->tileWeightLen = this->tiledweightShape[0];
        this->pipe.InitBuffer(this->castWeightBuf, this->tilingData->weightSize * sizeof(float));
        this->pipe.InitBuffer(this->broadcastWeightBuf,
                              this->tilingData->bufferNum * this->tilingData->weightSizePerCore * sizeof(float));
        this->pipe.InitBuffer(this->tileWeightBuf, this->tilingData->bufferNum * this->tileWeightLen * sizeof(float));
        this->InitActSeqLen();
    }

    __aicore__ inline void Process()
    {
        this->InitWeight();
        for (uint32_t idx = 0; idx < this->repeatNum; idx++) {
            this->InitOffset(idx);
            for (uint32_t tile_idx = 0; tile_idx < this->tilingData->tokenRepeatCount; tile_idx++) {
                this->CopyKvCache(tile_idx);
                InitWeightTileBroadCast(tile_idx);
                this->ComputeSubTile(tile_idx);
            }
            this->ComputeReduce();
            this->CopyOut();
        }
    }

protected:
    __aicore__ inline void InitWeightTileBroadCast(uint32_t tile_idx)
    {
        // 取全量weight
        AscendC::LocalTensor<float> castWeightCalcLocal = this->castWeightBuf.template Get<float>();
        // 跳读数据，切分headNum
        AscendC::LocalTensor<float> tileWeightCalcLocal =
            this->tileWeightBuf.template Get<float>()[this->tileWeightLen * this->pingPongIdx];
        AscendC::Copy(tileWeightCalcLocal,
                      castWeightCalcLocal[tile_idx * this->tilingData->tokenNumPerTile * this->tilingData->headNum +
                                          this->headIdx],
                      FP32_NUM_PER_UB_REPEAT, this->tilingData->tokenNumPerTile / FP32_NUM_PER_UB_BLOCK,
                      {1, static_cast<uint16_t>(this->tilingData->coresNumPerCompress),
                       static_cast<uint16_t>(FP32_NUM_PER_UB_BLOCK),
                       static_cast<uint16_t>(FP32_NUM_PER_UB_BLOCK * this->tilingData->coresNumPerCompress)});
        PipeBarrier<PIPE_V>();
        // 二次广播
        AscendC::LocalTensor<float> broadcastWeightLocal =
            this->broadcastWeightBuf.template Get<float>()[this->tilingData->weightSizePerCore * this->pingPongIdx];
        AscendC::BroadCast<float, BROADCAST_DIM_SIZE_2, BROADCAST_DIM_NUM_1>(
            broadcastWeightLocal, tileWeightCalcLocal, this->finalBroadCastWeightShape, this->tiledweightShape);
        PipeBarrier<PIPE_V>();
    }

    uint32_t tiledweightShape[BROADCAST_DIM_SIZE_2];
    uint32_t finalBroadCastWeightShape[BROADCAST_DIM_SIZE_2];
    AscendC::TBuf<AscendC::TPosition::VECCALC> tileWeightBuf;
};

template <typename T>
class KernelNsaCompressWithCacheFullBroadcast : public KernelNsaCompressWithCacheBase<T> {
public:
    __aicore__ inline KernelNsaCompressWithCacheFullBroadcast() {}
    __aicore__ inline void Init(GM_ADDR kv_cache, GM_ADDR weight, GM_ADDR slot_mapping, GM_ADDR compress_kv_cache,
                                GM_ADDR act_seq_len, GM_ADDR block_table,
                                const NsaCompressWithCacheTilingData* __restrict tilingData)
    {
        this->InitBase(kv_cache, weight, slot_mapping, compress_kv_cache, act_seq_len, block_table, tilingData);

        this->weightShape[0] = this->tilingData->tokenNumPerTile * this->tilingData->headsNumPerCore;
        this->weightShape[1] = 1;
        this->BroadCastWeightShape[0] = this->weightShape[0];
        this->BroadCastWeightShape[1] = this->tilingData->headDim;

        this->tileWeightLen = this->weightShape[0];

        this->pipe.InitBuffer(this->castWeightBuf,
                              this->tilingData->compressBlockSize * this->tilingData->headNum * sizeof(float));
        this->pipe.InitBuffer(this->broadcastWeightBuf,
                              this->tilingData->bufferNum * this->tilingData->weightSizePerCore * sizeof(float));

        this->InitActSeqLen();
    }

    __aicore__ inline void Process()
    {
        this->InitWeight();
        for (uint32_t idx = 0; idx < this->repeatNum; idx++) {
            this->InitOffset(idx);
            for (uint32_t tile_idx = 0; tile_idx < this->tilingData->tokenRepeatCount; tile_idx++) {
                this->CopyKvCache(tile_idx);
                InitWeightFullBroadCast(tile_idx);
                this->ComputeSubTile(tile_idx);
            }
            this->ComputeReduce();
            this->CopyOut();
        }
    }

protected:
    __aicore__ inline void InitWeightFullBroadCast(uint32_t tile_idx)
    {
        AscendC::LocalTensor<float> castWeightCalcLocal = this->castWeightBuf.template Get<float>();
        AscendC::LocalTensor<float> broadcastWeightLocal =
            this->broadcastWeightBuf.template Get<float>()[this->tilingData->weightSizePerCore * this->pingPongIdx];
        AscendC::BroadCast<float, BROADCAST_DIM_SIZE_2, BROADCAST_DIM_NUM_1>(
            broadcastWeightLocal,
            castWeightCalcLocal[tile_idx * this->tilingData->tokenNumPerTile * this->tilingData->headNum],
            this->BroadCastWeightShape, this->weightShape);
        PipeBarrier<PIPE_V>();
    }

    uint32_t weightShape[BROADCAST_DIM_SIZE_2];
    uint32_t BroadCastWeightShape[BROADCAST_DIM_SIZE_2];
};
} // namespace NsaCompressWithCacheBase

extern "C" __global__ __aicore__ void nsa_compress_with_cache(GM_ADDR input, GM_ADDR weight, GM_ADDR slotMapping,
                                                              GM_ADDR output_cache, GM_ADDR act_seq_len,
                                                              GM_ADDR block_table, GM_ADDR output_cache_ref,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.isEmpty) {
        return;
    }
#if (ORIG_DTYPE_INPUT == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        INVOKE_NCWC_GENERAL_OP_IMPL(NsaCompressWithCacheBase::KernelNsaCompressWithCacheDoubleBroadcast, half);
    } else if (TILING_KEY_IS(1)) {
        INVOKE_NCWC_GENERAL_OP_IMPL(NsaCompressWithCacheBase::KernelNsaCompressWithCacheTileBroadcast, half);
    } else if (TILING_KEY_IS(2)) {
        INVOKE_NCWC_GENERAL_OP_IMPL(NsaCompressWithCacheBase::KernelNsaCompressWithCacheFullBroadcast, half);
    }
#endif
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && (ORIG_DTYPE_INPUT == DT_BF16)
    if (TILING_KEY_IS(0)) {
        INVOKE_NCWC_GENERAL_OP_IMPL(NsaCompressWithCacheBase::KernelNsaCompressWithCacheDoubleBroadcast, bfloat16_t);
    } else if (TILING_KEY_IS(1)) {
        INVOKE_NCWC_GENERAL_OP_IMPL(NsaCompressWithCacheBase::KernelNsaCompressWithCacheTileBroadcast, bfloat16_t);
    } else if (TILING_KEY_IS(2)) {
        INVOKE_NCWC_GENERAL_OP_IMPL(NsaCompressWithCacheBase::KernelNsaCompressWithCacheFullBroadcast, bfloat16_t);
    }
#endif
}
