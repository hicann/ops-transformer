/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_CACHE_ARCH92_H
#define SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_CACHE_ARCH92_H

#include "sparse_lightning_indexer_grad_kl_loss_tiling_regbase_arch92.h"

namespace SligKlLoss {

using optiling::GetSligDKeyYBase;
using optiling::SLIG_DKEY_HALF_ROWS;
using optiling::SLIG_DKEY_MIN_SPLIT_ROWS;

struct SligTilingCacheBase {
    optiling::SLIGradKLLossBaseParamsRegbase baseParams;
    int64_t totalSize;
    uint32_t syKTotalSize;
    uint32_t pKTotalSize;
};

template <bool IS_DETER>
struct SligTilingCache : SligTilingCacheBase {};

template <>
struct SligTilingCache<false> : SligTilingCacheBase {
    int64_t taskBegin;
    int64_t taskEnd;
};

static_assert(sizeof(SligTilingCache<true>) == 88, "Unexpected deterministic tiling cache layout");
static_assert(sizeof(SligTilingCache<false>) == 104, "Unexpected normal tiling cache layout");

// Deterministic kernels retain the complete local tiling object and its original pointer consumers.
template <bool IS_DETER>
struct SligTilingTraits {
    using KernelTiling = SligTilingCache<false>;
    using VectorTiling = int64_t;
};

template <>
struct SligTilingTraits<true> {
    using KernelTiling = optiling::SparseLightningIndexerGradKLLossRegBaseTilingData;
    using VectorTiling = const KernelTiling*;
};

// Preserve the tiling pointer's address space when reading the fields used by this core.
template <bool IS_DETER, typename TilingPointer>
__aicore__ inline SligTilingCache<IS_DETER> LoadSligTilingCache(TilingPointer tiling, uint32_t aicIdx)
{
    SligTilingCache<IS_DETER> cache;
    cache.baseParams.bSize = tiling->baseParams.bSize;
    cache.baseParams.n2Size = tiling->baseParams.n2Size;
    cache.baseParams.gSizeQuery = tiling->baseParams.gSizeQuery;
    cache.baseParams.gSizeQueryIndex = tiling->baseParams.gSizeQueryIndex;
    cache.baseParams.s1Size = tiling->baseParams.s1Size;
    cache.baseParams.s2Size = tiling->baseParams.s2Size;
    cache.baseParams.dSizeQuery = tiling->baseParams.dSizeQuery;
    cache.baseParams.dSizeQueryIndex = tiling->baseParams.dSizeQueryIndex;
    cache.baseParams.kSize = tiling->baseParams.kSize;
    cache.baseParams.sparseMode = tiling->baseParams.sparseMode;
    cache.baseParams.pairedDKeyMinRows = tiling->baseParams.pairedDKeyMinRows;
    cache.baseParams.scaleValue = tiling->baseParams.scaleValue;
    cache.baseParams.t1Size = tiling->baseParams.t1Size;
    cache.baseParams.t2Size = tiling->baseParams.t2Size;
    cache.baseParams.totalAivNum = tiling->baseParams.totalAivNum;
    cache.baseParams.layoutType = tiling->baseParams.layoutType;
    cache.totalSize = tiling->multiCoreParams.totalSize;
    cache.syKTotalSize = tiling->multiCoreParams.syKTotalSize;
    cache.pKTotalSize = tiling->multiCoreParams.pKTotalSize;
    if constexpr (!IS_DETER) {
        cache.taskBegin = tiling->multiCoreParams.bS1Index[aicIdx];
        // Preserve the wire table boundary, including launches with fewer than 80 cores.
        cache.taskEnd = aicIdx + 1 < optiling::MAX_CORE_NUM_REGBASE ? tiling->multiCoreParams.bS1Index[aicIdx + 1] :
                                                                      cache.totalSize;
    }
    return cache;
}

} // namespace SligKlLoss
#endif // SPARSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_TILING_CACHE_ARCH92_H
