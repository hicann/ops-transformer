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
 * \file bsa_vector_service.h
 * \brief
 */
#ifndef BSA_VECTOR_SERVICE_H
#define BSA_VECTOR_SERVICE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"
#include "bsa_vec_pool_service.h"
#include "bsa_vec_sm_service.h"
#include "bsa_vec_post_pool_service.h"

template <typename BSAT>
class BSAVectorService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;
    using POOL_OUT_T = half;
    using SFTMAX_OUT_T = half;

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;

    __aicore__ inline BSAVectorService(){};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGM(GlobalTensor<POOL_OUT_T> &qCmpGm, GlobalTensor<POOL_OUT_T> &kCmpGm,
                                  GlobalTensor<half> &attnScorFp16eGm, GlobalTensor<T> &ScoreFp32Gm,
                                  GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
                                  GlobalTensor<int64_t> &actualBlockLenQGm, GlobalTensor<int64_t> &actualBlockLenKVGm,
                                  GlobalTensor<int64_t> &actualSeqLensQGm, GlobalTensor<int64_t> &actualSeqLensKVGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void PoolingSingleQBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t qBlockIdx,
                                               GlobalTensor<IN_T> &queryGm, GlobalTensor<int64_t> &actualBlockLenQGm,
                                               GlobalTensor<POOL_OUT_T> &qCmpGm, uint64_t seqPrefixSumQ,
                                               uint64_t blockPrefixSumQ);
    __aicore__ inline void PoolingSingleKBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t kBlockIdx,
                                               GlobalTensor<IN_T> &keyGm, GlobalTensor<int64_t> &actualBlockLenKVGm,
                                               GlobalTensor<POOL_OUT_T> &kCmpGm, uint64_t seqPrefixSumKV,
                                               uint64_t blockPrefixSumKV);
    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                       uint32_t kChunkSize, uint32_t validYBlocks);
    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                    uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx,
                                                    uint32_t validYBlocks);
    __aicore__ inline void PostPoolRange(uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                         uint32_t validYBlocks);
    __aicore__ inline void InitPostPoolBuffers();
    __aicore__ inline void InitPostPoolGM(GlobalTensor<half> &attnScoreGm, GlobalTensor<half> &pooledScoreGm);
    __aicore__ inline LocalTensor<half> GetPostPoolScalarScratch();

private:
    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    TBuf<> uBuf_;

    // vector op
    BSAVecPoolService<BSAT> poolOP;
    BSAVecSmService<BSAT> softmaxOP;
    BSAVecPostPoolService<BSAT> postPoolOP;
};

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitParams(
    const BSAConstInfo &constInfo, const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    poolOP.InitParams(constInfo, tilingData);
    softmaxOP.InitParams(constInfo, tilingData);
    postPoolOP.InitParams(constInfo, tilingData);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(uBuf_, BSAConstInfo::BUFFER_SIZE_BYTE_192K);
    poolOP.InitBuffers(&uBuf_);
    softmaxOP.InitBuffers(&uBuf_);
    if (constInfo.usePostBlockShape) {
        postPoolOP.InitBuffers(&uBuf_);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitGM(
    GlobalTensor<POOL_OUT_T> &qCmpGm, GlobalTensor<POOL_OUT_T> &kCmpGm, GlobalTensor<half> &attnScorFp16eGm,
    GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenQGm, GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<int64_t> &actualSeqLensQGm, GlobalTensor<int64_t> &actualSeqLensKVGm)
{
    poolOP.InitGM(qCmpGm, kCmpGm, queryGm, keyGm, actualBlockLenQGm, actualBlockLenKVGm, actualSeqLensQGm,
                  actualSeqLensKVGm);
    softmaxOP.InitGM(ScoreFp32Gm, attnScorFp16eGm);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::AllocEventID()
{
    poolOP.AllocEventID();
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::FreeEventID()
{
    poolOP.FreeEventID();
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleQBlock(uint32_t batchIdx, uint32_t headIdx,
                                                                   uint32_t qBlockIdx, GlobalTensor<IN_T> &queryGm,
                                                                   GlobalTensor<int64_t> &actualBlockLenQGm,
                                                                   GlobalTensor<POOL_OUT_T> &qCmpGm,
                                                                   uint64_t seqPrefixSumQ, uint64_t blockPrefixSumQ)
{
    poolOP.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx, queryGm, actualBlockLenQGm, qCmpGm, seqPrefixSumQ,
                               blockPrefixSumQ);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleKBlock(uint32_t batchIdx, uint32_t headIdx,
                                                                   uint32_t kBlockIdx, GlobalTensor<IN_T> &keyGm,
                                                                   GlobalTensor<int64_t> &actualBlockLenKVGm,
                                                                   GlobalTensor<POOL_OUT_T> &kCmpGm,
                                                                   uint64_t seqPrefixSumKV, uint64_t blockPrefixSumKV)
{
    poolOP.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx, keyGm, actualBlockLenKVGm, kCmpGm, seqPrefixSumKV,
                               blockPrefixSumKV);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                           uint32_t kChunkStart, uint32_t kChunkSize,
                                                                           uint32_t validYBlocks)
{
    softmaxOP.OnlineSoftmaxFirstPassChunk(qChunkStart, qChunkSize, kChunkStart, kChunkSize, validYBlocks);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                        uint32_t kChunkStart, uint32_t kChunkSize,
                                                                        uint32_t batchIdx, uint32_t headIdx,
                                                                        uint32_t validYBlocks)
{
    softmaxOP.SoftmaxSecondPassAndCast(qChunkStart, qChunkSize, kChunkStart, kChunkSize, batchIdx, headIdx,
                                       validYBlocks);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitPostPoolBuffers()
{
    postPoolOP.InitBuffers(&uBuf_);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitPostPoolGM(GlobalTensor<half> &attnScoreGm,
                                                              GlobalTensor<half> &pooledScoreGm)
{
    postPoolOP.InitPostPoolGM(attnScoreGm, pooledScoreGm);
}

template <typename BSAT>
__aicore__ inline LocalTensor<half> BSAVectorService<BSAT>::GetPostPoolScalarScratch()
{
    return postPoolOP.GetScalarScratch();
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PostPoolRange(uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                                             uint32_t validYBlocks)
{
    postPoolOP.PostPoolRange(batchIdx, headIdx, validXBlocks, validYBlocks);
}

#endif // BSA_VECTOR_SERVICE_H
