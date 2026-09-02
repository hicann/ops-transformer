/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_grad_tiling.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_GRAD_TILING_H_
#define LIGHTNING_INDEXER_GRAD_TILING_H_
#include <stdint.h>

namespace optiling {
// -----------算子TilingData定义---------------
class LIGTilingData {
public:
    uint32_t batch;
    uint32_t seqlenQ;
    uint32_t seqlenK;
    uint32_t topK;
    uint32_t headNumQ;
    uint32_t headNumK;
    uint32_t groupNum;
    uint32_t headDim;
    uint32_t usedCoreNum;
    int64_t dkSize;
    int64_t dkCoreSize;
    int64_t dkWorkSpaceOffset;
    int64_t dkCoreWorkspaceOffset;
    int64_t keyGatherWorkspaceOffset;
    int64_t reluInWorkspaceOffset;
    int64_t reluGradWorkspaceOffset;
    int64_t scatterAddWorkspaceOffset;
    uint64_t sparseMode;
    bool deterministic;

    // ========================
    // Getter & Setter 方法
    // ========================
    uint32_t get_batch() const
    {
        return batch;
    }
    void set_batch(uint32_t batch)
    {
        this->batch = batch;
    }

    uint32_t get_seqlenQ() const
    {
        return seqlenQ;
    }
    void set_seqlenQ(uint32_t seqlenQ)
    {
        this->seqlenQ = seqlenQ;
    }

    uint32_t get_seqlenK() const
    {
        return seqlenK;
    }
    void set_seqlenK(uint32_t seqlenK)
    {
        this->seqlenK = seqlenK;
    }

    uint32_t get_topK() const
    {
        return topK;
    }
    void set_topK(uint32_t topK)
    {
        this->topK = topK;
    }

    uint32_t get_headNumQ() const
    {
        return headNumQ;
    }
    void set_headNumQ(uint32_t headNumQ)
    {
        this->headNumQ = headNumQ;
    }

    uint32_t get_headNumK() const
    {
        return headNumK;
    }
    void set_headNumK(uint32_t headNumK)
    {
        this->headNumK = headNumK;
    }

    uint32_t get_groupNum() const
    {
        return groupNum;
    }
    void set_groupNum(uint32_t groupNum)
    {
        this->groupNum = groupNum;
    }

    float get_headDim() const
    {
        return headDim;
    }
    void set_headDim(float headDim)
    {
        this->headDim = headDim;
    }

    float get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    void set_usedCoreNum(float usedCoreNum)
    {
        this->usedCoreNum = usedCoreNum;
    }

    int64_t get_dkSize() const
    {
        return dkSize;
    }
    void set_dkSize(int64_t dkSize)
    {
        this->dkSize = dkSize;
    }

    int64_t get_dkCoreSize() const
    {
        return dkCoreSize;
    }
    void set_dkCoreSize(int64_t dkCoreSize)
    {
        this->dkCoreSize = dkCoreSize;
    }

    int64_t get_dkWorkSpaceOffset() const
    {
        return dkWorkSpaceOffset;
    }
    void set_dkWorkSpaceOffset(int64_t dkWorkSpaceOffset)
    {
        this->dkWorkSpaceOffset = dkWorkSpaceOffset;
    }

    int64_t get_dkCoreWorkspaceOffset() const
    {
        return dkCoreWorkspaceOffset;
    }
    void set_dkCoreWorkspaceOffset(int64_t dkCoreWorkspaceOffset)
    {
        this->dkCoreWorkspaceOffset = dkCoreWorkspaceOffset;
    }

    int64_t get_keyGatherWorkspaceOffset() const
    {
        return keyGatherWorkspaceOffset;
    }
    void set_keyGatherWorkspaceOffset(int64_t keyGatherWorkspaceOffset)
    {
        this->keyGatherWorkspaceOffset = keyGatherWorkspaceOffset;
    }

    int64_t get_reluInWorkspaceOffset() const
    {
        return reluInWorkspaceOffset;
    }
    void set_reluInWorkspaceOffset(int64_t reluInWorkspaceOffset)
    {
        this->reluInWorkspaceOffset = reluInWorkspaceOffset;
    }

    int64_t get_reluGradWorkspaceOffset() const
    {
        return reluGradWorkspaceOffset;
    }
    void set_reluGradWorkspaceOffset(int64_t reluGradWorkspaceOffset)
    {
        this->reluGradWorkspaceOffset = reluGradWorkspaceOffset;
    }

    int64_t get_scatterAddWorkspaceOffset() const
    {
        return scatterAddWorkspaceOffset;
    }
    void set_scatterAddWorkspaceOffset(int64_t scatterAddWorkspaceOffset)
    {
        this->scatterAddWorkspaceOffset = scatterAddWorkspaceOffset;
    }

    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    void set_sparseMode(uint32_t sparseMode)
    {
        this->sparseMode = sparseMode;
    }

    bool get_determinstic() const
    {
        return deterministic;
    }
    void set_deterministic(bool deterministic)
    {
        this->deterministic = deterministic;
    }

    void reset()
    {
        set_batch(0);
        set_seqlenQ(0);
        set_seqlenK(0);
        set_topK(0);
        set_headNumQ(0);
        set_dkCoreSize(0);
        set_scatterAddWorkspaceOffset(0);
        set_reluGradWorkspaceOffset(0);
        set_reluInWorkspaceOffset(0);
        set_keyGatherWorkspaceOffset(0);
        set_dkWorkSpaceOffset(0);
        set_dkCoreWorkspaceOffset(0);
        set_dkSize(0);
        set_usedCoreNum(0);
        set_headDim(0);
        set_groupNum(0);
        set_headNumK(0);
        set_deterministic(false);
    }
};
} // namespace optiling
#endif // LIGHTNING_INDEXER_GRAD_TILING_H_
