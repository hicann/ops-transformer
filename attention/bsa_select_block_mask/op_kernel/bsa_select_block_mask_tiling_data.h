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
 * \file bsa_select_block_mask_tiling_data.h
 * \brief
 */
#ifndef __BSA_SELECT_BLOCK_MASK_TILING_DATA_H__
#define __BSA_SELECT_BLOCK_MASK_TILING_DATA_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t BLOCK_SHAPE_INPUT_INDEX = 2;
constexpr uint32_t POST_BLOCK_SHAPE_INPUT_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_LENS_Q_INPUT_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_LENS_KV_INPUT_INDEX = 5;
constexpr uint32_t ACTUAL_BLOCK_LEN_Q_INPUT_INDEX = 6;
constexpr uint32_t ACTUAL_BLOCK_LEN_KV_INPUT_INDEX = 7;

constexpr uint32_t MASK_OUTPUT_INDEX = 0;

constexpr uint32_t Q_LAYOUT_ATTR_INDEX = 0;
constexpr uint32_t KV_LAYOUT_ATTR_INDEX = 1;
constexpr uint32_t NUM_KV_HEADS_ATTR_INDEX = 2;
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 3;
constexpr uint32_t SPARSITY_ATTR_INDEX = 4;

constexpr uint32_t MAX_CORE_NUM = 48;
constexpr uint32_t AIC_AIV_RATIO = 2;
constexpr uint32_t D_SIZE = 128;
constexpr uint32_t CUBE_BASE_BLOCK = 128;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t VEC_ALIGN_SIZE = 32;
constexpr uint32_t FLOAT_DATA_BLOCK_NUM = 8;
constexpr uint32_t FLOAT_REPEAT_NUM = 64;
constexpr uint32_t RADIX_BITS_PER_ROUND = 2;
constexpr uint32_t RADIX_NUM_BINS = 4;
constexpr uint32_t RADIX_ROUNDS_FP16 = 8;
constexpr uint16_t XOR_OP_VALUE_16 = 0x8000;
constexpr uint32_t HALF_SAFE_MAX = 2048;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint32_t BYTE_SIZE = 8;

enum class LayoutType : uint8_t {
    LAYOUT_TND = 0,
    LAYOUT_BNSD = 1,
    LAYOUT_NONE
};

enum class SparsityMode : uint8_t {
    TOP_K = 0,
    BOTTOM_K = 1
};

class BSABaseParams {
public:
    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t dSize;
    uint32_t rsvd0;
    uint64_t blockShapeX;
    uint64_t blockShapeY;
    uint32_t xBlocks;
    uint32_t yBlocks;
    float scaleValue;
    float sparsity;
    uint64_t topKValue;
    uint8_t sparsityMode;
    uint8_t queryLayout;
    uint8_t kvLayout;
    uint8_t useActualBlockLenQ;
    uint8_t useActualBlockLenK;
    uint8_t useActualSeqLenQ;
    uint8_t useActualSeqLenK;
    uint16_t qChunkSize;
    uint16_t kChunkSize;
    uint8_t usePostBlockShape;
    uint16_t postBlockShapeX;
    uint16_t postBlockShapeY;
    uint32_t postXBlocks;
    uint32_t postYBlocks;

    uint32_t get_batchSize() const
    {
        return batchSize;
    }
    void set_batchSize(uint32_t v)
    {
        this->batchSize = v;
    }
    uint32_t get_numHeads() const
    {
        return numHeads;
    }
    void set_numHeads(uint32_t v)
    {
        this->numHeads = v;
    }
    uint32_t get_maxQSeqlen() const
    {
        return maxQSeqlen;
    }
    void set_maxQSeqlen(uint32_t v)
    {
        this->maxQSeqlen = v;
    }
    uint32_t get_maxKvSeqlen() const
    {
        return maxKvSeqlen;
    }
    void set_maxKvSeqlen(uint32_t v)
    {
        this->maxKvSeqlen = v;
    }
    uint32_t get_dSize() const
    {
        return dSize;
    }
    void set_dSize(uint32_t v)
    {
        this->dSize = v;
    }
    uint64_t get_blockShapeX() const
    {
        return blockShapeX;
    }
    void set_blockShapeX(uint64_t v)
    {
        this->blockShapeX = v;
    }
    uint64_t get_blockShapeY() const
    {
        return blockShapeY;
    }
    void set_blockShapeY(uint64_t v)
    {
        this->blockShapeY = v;
    }
    uint32_t get_xBlocks() const
    {
        return xBlocks;
    }
    void set_xBlocks(uint32_t v)
    {
        this->xBlocks = v;
    }
    uint32_t get_yBlocks() const
    {
        return yBlocks;
    }
    void set_yBlocks(uint32_t v)
    {
        this->yBlocks = v;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    void set_scaleValue(float v)
    {
        this->scaleValue = v;
    }
    float get_sparsity() const
    {
        return sparsity;
    }
    void set_sparsity(float v)
    {
        this->sparsity = v;
    }
    uint64_t get_topKValue() const
    {
        return topKValue;
    }
    void set_topKValue(uint64_t v)
    {
        this->topKValue = v;
    }
    uint8_t get_sparsityMode() const
    {
        return sparsityMode;
    }
    void set_sparsityMode(uint8_t v)
    {
        this->sparsityMode = v;
    }
    uint8_t get_queryLayout() const
    {
        return queryLayout;
    }
    void set_queryLayout(uint8_t v)
    {
        this->queryLayout = v;
    }
    uint8_t get_kvLayout() const
    {
        return kvLayout;
    }
    void set_kvLayout(uint8_t v)
    {
        this->kvLayout = v;
    }
    uint8_t get_useActualBlockLenQ() const
    {
        return useActualBlockLenQ;
    }
    void set_useActualBlockLenQ(uint8_t v)
    {
        this->useActualBlockLenQ = v;
    }
    uint8_t get_useActualBlockLenK() const
    {
        return useActualBlockLenK;
    }
    void set_useActualBlockLenK(uint8_t v)
    {
        this->useActualBlockLenK = v;
    }
    uint8_t get_useActualSeqLenQ() const
    {
        return useActualSeqLenQ;
    }
    void set_useActualSeqLenQ(uint8_t v)
    {
        this->useActualSeqLenQ = v;
    }
    uint8_t get_useActualSeqLenK() const
    {
        return useActualSeqLenK;
    }
    void set_useActualSeqLenK(uint8_t v)
    {
        this->useActualSeqLenK = v;
    }
    uint16_t get_qChunkSize() const
    {
        return qChunkSize;
    }
    void set_qChunkSize(uint16_t v)
    {
        this->qChunkSize = v;
    }
    uint16_t get_kChunkSize() const
    {
        return kChunkSize;
    }
    void set_kChunkSize(uint16_t v)
    {
        this->kChunkSize = v;
    }
    uint8_t get_usePostBlockShape() const
    {
        return usePostBlockShape;
    }
    void set_usePostBlockShape(uint8_t v)
    {
        this->usePostBlockShape = v;
    }
    uint16_t get_postBlockShapeX() const
    {
        return postBlockShapeX;
    }
    void set_postBlockShapeX(uint16_t v)
    {
        this->postBlockShapeX = v;
    }
    uint16_t get_postBlockShapeY() const
    {
        return postBlockShapeY;
    }
    void set_postBlockShapeY(uint16_t v)
    {
        this->postBlockShapeY = v;
    }
    uint32_t get_postXBlocks() const
    {
        return postXBlocks;
    }
    void set_postXBlocks(uint32_t v)
    {
        this->postXBlocks = v;
    }
    uint32_t get_postYBlocks() const
    {
        return postYBlocks;
    }
    void set_postYBlocks(uint32_t v)
    {
        this->postYBlocks = v;
    }
};

class BSAMultiCoreParams {
public:
    uint32_t coreNum;
    uint32_t activeCoreNum;
    uint32_t rowsPerCore;
    uint32_t extraCores;
    uint32_t totalRows;
    uint32_t yBlocksPerCore;
    uint32_t extraYCores;
    uint32_t rsvd0;
    uint32_t activeYVecCoreNum;

    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    void set_coreNum(uint32_t v)
    {
        this->coreNum = v;
    }
    uint32_t get_activeCoreNum() const
    {
        return activeCoreNum;
    }
    void set_activeCoreNum(uint32_t v)
    {
        this->activeCoreNum = v;
    }
    uint32_t get_rowsPerCore() const
    {
        return rowsPerCore;
    }
    void set_rowsPerCore(uint32_t v)
    {
        this->rowsPerCore = v;
    }
    uint32_t get_extraCores() const
    {
        return extraCores;
    }
    void set_extraCores(uint32_t v)
    {
        this->extraCores = v;
    }
    uint32_t get_totalRows() const
    {
        return totalRows;
    }
    void set_totalRows(uint32_t v)
    {
        this->totalRows = v;
    }
    uint32_t get_yBlocksPerCore() const
    {
        return yBlocksPerCore;
    }
    void set_yBlocksPerCore(uint32_t v)
    {
        this->yBlocksPerCore = v;
    }
    uint32_t get_extraYCores() const
    {
        return extraYCores;
    }
    void set_extraYCores(uint32_t v)
    {
        this->extraYCores = v;
    }
    uint32_t get_activeYCoreNum() const
    {
        return activeYVecCoreNum;
    }
    void set_activeYCoreNum(uint32_t v)
    {
        this->activeYVecCoreNum = v;
    }
};

class BSAOutputParams {
public:
    uint64_t qCmpSize;
    uint64_t kCmpSize;
    uint64_t attnScoreSize;
    uint64_t topkWorkspaceSize;
    uint64_t softmaxTmpSize;
    uint64_t totalWorkspaceSize;
    uint64_t pooledScoreSize;
    uint64_t pooledScoreOffset;

    uint64_t get_qCmpSize() const
    {
        return qCmpSize;
    }
    void set_qCmpSize(uint64_t v)
    {
        this->qCmpSize = v;
    }
    uint64_t get_kCmpSize() const
    {
        return kCmpSize;
    }
    void set_kCmpSize(uint64_t v)
    {
        this->kCmpSize = v;
    }
    uint64_t get_attnScoreSize() const
    {
        return attnScoreSize;
    }
    void set_attnScoreSize(uint64_t v)
    {
        this->attnScoreSize = v;
    }
    uint64_t get_topkWorkspaceSize() const
    {
        return topkWorkspaceSize;
    }
    void set_topkWorkspaceSize(uint64_t v)
    {
        this->topkWorkspaceSize = v;
    }
    uint64_t get_softmaxTmpSize() const
    {
        return softmaxTmpSize;
    }
    void set_softmaxTmpSize(uint64_t v)
    {
        this->softmaxTmpSize = v;
    }
    uint64_t get_totalWorkspaceSize() const
    {
        return totalWorkspaceSize;
    }
    void set_totalWorkspaceSize(uint64_t v)
    {
        this->totalWorkspaceSize = v;
    }
    uint64_t get_pooledScoreSize() const
    {
        return pooledScoreSize;
    }
    void set_pooledScoreSize(uint64_t v)
    {
        this->pooledScoreSize = v;
    }
    uint64_t get_pooledScoreOffset() const
    {
        return pooledScoreOffset;
    }
    void set_pooledScoreOffset(uint64_t v)
    {
        this->pooledScoreOffset = v;
    }
};

class BSASelectBlockMaskTilingData {
public:
    BSABaseParams baseParams;
    BSAMultiCoreParams multiCoreParams;
    BSAOutputParams outputParams;
};

template <typename T>
inline T BSAAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

} // namespace optiling
#endif
