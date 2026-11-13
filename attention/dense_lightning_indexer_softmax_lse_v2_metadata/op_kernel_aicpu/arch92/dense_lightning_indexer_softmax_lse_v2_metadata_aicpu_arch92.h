/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file dense_lightning_indexer_softmax_lse_v2_metadata_aicpu_arch92.h
\brief arch92 (Ascend960) specific implementations for AICPU metadata kernel.
*/
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_ARCH92_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_ARCH92_H
#include "../dense_lightning_indexer_softmax_lse_v2_metadata_aicpu.h"
#include "../../../dense_lightning_indexer_softmax_lse_v2/op_kernel/dense_lightning_indexer_softmax_lse_v2_metadata.h"

#include <algorithm>
#include <vector>

using namespace optiling;

namespace aicpu {

inline bool DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::BuildMetadataArch92()
{
    int64_t totalNum = CalcTotalSize();
    if (totalNum < 0) {
        KERNEL_LOG_ERROR("Total valid sequence should be >= 0, but got %ld", totalNum);
        return false;
    }
    int64_t N = aicCoreNum_;
    if (N <= 0) {
        KERNEL_LOG_ERROR("aic_core_num should be positive, but got %ld", N);
        return false;
    }

    int64_t forecoreNum = 0;
    int64_t tailCoreNum = 0;
    int64_t bS1PerCore = 0;
    int64_t bS1PerTailCore = 0;

    if (totalNum < N) {
        forecoreNum = totalNum;
        tailCoreNum = 0;
        bS1PerCore = 1;
        bS1PerTailCore = 0;
    } else {
        int64_t base = totalNum / N;
        int64_t rem = totalNum % N;
        if (rem == 0) {
            forecoreNum = N;
            tailCoreNum = 0;
            bS1PerCore = base;
            bS1PerTailCore = 0;
        } else {
            forecoreNum = rem;
            bS1PerCore = base + 1;
            tailCoreNum = N - rem;
            bS1PerTailCore = base;
        }
    }

    auto* metadataData = reinterpret_cast<DLI_METADATA_T*>(metadata_->GetData());
    const int64_t metadataSize = metadata_->GetTensorShape()->GetDimSize(0);
    if (metadataSize < static_cast<int64_t>(DLI_METADATA_SIZE)) {
        KERNEL_LOG_ERROR("metadata output size should be >= %u, but got %ld", DLI_METADATA_SIZE, metadataSize);
        return false;
    }
    std::fill_n(metadataData, metadataSize, static_cast<DLI_METADATA_T>(0));
    auto* metadata = reinterpret_cast<detail::DenseLISoftmaxLseV2MetaData*>(metadataData);
    metadata->forecore_num = static_cast<int32_t>(forecoreNum);
    metadata->tail_core_num = static_cast<int32_t>(tailCoreNum);
    metadata->b_s1_per_core = static_cast<int32_t>(bS1PerCore);
    metadata->b_s1_per_tail_core = static_cast<int32_t>(bS1PerTailCore);

    // Size-aware scheduling table: only when the caller allocated the expanded
    // buffer (BSND with batch_size*max_seqlen_q bound, or TND with the packed q
    // row count) and the core count fits.
    if (metadataSize < static_cast<int64_t>(DLI_SCHED_ITEM_IDX) || N > static_cast<int64_t>(DLI_SCHED_CORE_NUM)) {
        return true; // legacy-only buffer or unsupported core count
    }
    const int64_t tableSlots =
        (metadataSize - static_cast<int64_t>(DLI_SCHED_ITEM_IDX)) / static_cast<int64_t>(DLI_SCHED_ENTRY_STRIDE);
    if (totalNum > tableSlots) {
        return true; // item bound exceeded (table capped at DLI_SCHED_ITEM_MAX_NUM)
    }

    const int32_t* seqQ =
        IsTensorValid(seqUsedQuery_) ? reinterpret_cast<const int32_t*>(seqUsedQuery_->GetData()) : nullptr;
    const int32_t* seqK =
        IsTensorValid(seqUsedKey_) ? reinterpret_cast<const int32_t*>(seqUsedKey_->GetData()) : nullptr;
    const int32_t* cuQ =
        IsTensorValid(cuSeqLensQuery_) ? reinterpret_cast<const int32_t*>(cuSeqLensQuery_->GetData()) : nullptr;
    const int32_t* cuK =
        IsTensorValid(cuSeqLensKey_) ? reinterpret_cast<const int32_t*>(cuSeqLensKey_->GetData()) : nullptr;
    const int32_t* resK =
        IsTensorValid(cmpResidualKey_) ? reinterpret_cast<const int32_t*>(cmpResidualKey_->GetData()) : nullptr;

    // Offset precomputation needs layout ground truth: TND requires the
    // cumulative-seqlens tensors (packed offsets); BSND requires max_seqlen_q.
    if (layoutType_ == DliLayout::TND) {
        if (cuQ == nullptr || cuK == nullptr) {
            return true; // cannot map packed offsets; legacy only
        }
    } else if (s1Size_ <= 0) {
        return true; // BSND without max_seqlen_q: cannot compute offsets
    }

    // Row-block scheduling (arch92): group consecutive valid q rows of the
    // same batch into blocks of R rows. One cube pass (a qk tile of R*n1
    // heads x 128 k) then covers R rows, amortizing k loads, acc->vec
    // handshakes, q loads and per-block scalar setup over R rows.
    struct SchedBlock {
        int32_t qOffset;
        int32_t kBase;
        int32_t rows;
        int32_t vkMax;
        int32_t vkParam;
        int32_t qHeadOffset;
        int64_t cost;
    };
    // Constraints mirrored from the arch92 kernel / tiling:
    //  - the qk tile is [TN=128, TKV=128]: R * num_heads_q <= 128
    //  - each row's reduce_sum region is tiles(vk_max) * 128 fp32 in UB and
    //    the cache holds MAX_UB_S2_TILES = 237 tiles: R * ceil(vk_max/128) <= 237
    //  - long-s2 (total k rows > 237*128) runs the GM workspace path: R = 1
    constexpr int64_t kQkTileRows = 128;
    constexpr int64_t kS2TileElems = 128;
    constexpr int64_t kMaxUbS2Tiles = 237;
    int64_t totalK = s2Size_;
    if (layoutType_ == DliLayout::TND && cuK != nullptr) {
        totalK = static_cast<int64_t>(cuK[bSize_]);
    }
    const bool longS2 = totalK > kMaxUbS2Tiles * kS2TileElems;
    const int64_t rHeadCap = numHeadsQ_ > 0 ? kQkTileRows / numHeadsQ_ : 1;
    const int64_t rUbCap =
        longS2 ? 1 : 2; // adaptive: R=2 (parallelism + amortization); long-s2 GM workspace path requires R=1

    std::vector<SchedBlock> blocks;
    blocks.reserve(static_cast<size_t>(totalNum));
    for (int64_t b = 0; b < bSize_; ++b) {
        int64_t qUsed = 0;
        if (seqQ != nullptr) {
            qUsed = static_cast<int64_t>(seqQ[b]);
        } else if (layoutType_ == DliLayout::TND && cuQ != nullptr) {
            qUsed = static_cast<int64_t>(cuQ[b + 1]) - static_cast<int64_t>(cuQ[b]);
        } else {
            qUsed = s1Size_;
        }
        int64_t kUsed = 0;
        if (seqK != nullptr) {
            kUsed = static_cast<int64_t>(seqK[b]);
        } else if (layoutType_ == DliLayout::TND && cuK != nullptr) {
            kUsed = static_cast<int64_t>(cuK[b + 1]) - static_cast<int64_t>(cuK[b]);
        } else {
            kUsed = s2Size_;
        }
        int64_t residual = (maskMode_ == 3 && cmpRatio_ > 1 && resK != nullptr) ? static_cast<int64_t>(resK[b]) : 0;
        int64_t qkResidual = qUsed - (kUsed * cmpRatio_ + residual);
        const int64_t qBase = (layoutType_ == DliLayout::TND) ? static_cast<int64_t>(cuQ[b]) : b * s1Size_;
        const int64_t kBase = (layoutType_ == DliLayout::TND) ? static_cast<int64_t>(cuK[b]) : b * s2Size_;
        // Per-row valid k bound, mirroring the kernel strided path:
        //   mask 3: vk(s1) = max(s1 + 1 - qkResidual, 0) / cmpRatio
        //   else:   vk(s1) = kUsed
        auto vkOf = [&](int64_t s1I) -> int64_t {
            if (maskMode_ == 3) {
                const int64_t num = s1I + 1 - qkResidual;
                return num > 0 ? num / cmpRatio_ : 0;
            }
            return kUsed;
        };
        if (maskMode_ != 3 && kUsed <= 0) {
            continue; // no k work at all for this batch
        }
        int64_t validStart = 0;
        if (maskMode_ == 3) {
            // first s1 with vk >= 1: s1 + 1 - qkResidual >= cmpRatio
            validStart = qkResidual + cmpRatio_ - 1;
            if (validStart < 0) {
                validStart = 0;
            }
        }
        const int64_t validRows = qUsed - validStart;
        if (validRows <= 0) {
            continue;
        }
        int64_t r = rUbCap;
        if (r > rHeadCap) {
            r = rHeadCap;
        }
        const int64_t tilesLast = (vkOf(qUsed - 1) + kS2TileElems - 1) / kS2TileElems;
        if (tilesLast > 0) {
            const int64_t rTilesCap = kMaxUbS2Tiles / tilesLast;
            if (r > rTilesCap) {
                r = rTilesCap;
            }
        }
        if (r < 1) {
            r = 1;
        }
        for (int64_t start = validStart; start < qUsed; start += r) {
            int64_t rows = qUsed - start;
            if (rows > r) {
                rows = r;
            }
            const int64_t vkMax = vkOf(start + rows - 1);
            const int64_t vkParam = (maskMode_ == 3) ? (start + 1 - qkResidual) : kUsed;
            int64_t cost = 0;
            for (int64_t rr = 0; rr < rows; ++rr) {
                cost += vkOf(start + rr);
            }
            blocks.push_back({static_cast<int32_t>(qBase + start), static_cast<int32_t>(kBase),
                              static_cast<int32_t>(rows), static_cast<int32_t>(vkMax), static_cast<int32_t>(vkParam),
                              static_cast<int32_t>((qBase + start) * numHeadsQ_), cost});
        }
    }

    if (blocks.empty()) {
        return true; // nothing valid to schedule; keep the legacy flag unset
    }

    // LPT (longest processing time first): sort by cost desc, assign to the
    // least-loaded core. Deterministic: stable sort + lowest core index on ties.
    std::stable_sort(blocks.begin(), blocks.end(),
                     [](const SchedBlock& lhs, const SchedBlock& rhs) { return lhs.cost > rhs.cost; });
    std::vector<int64_t> coreLoad(static_cast<size_t>(N), 0);
    std::vector<std::vector<SchedBlock>> perCore(static_cast<size_t>(N));
    for (const auto& blk : blocks) {
        int64_t best = 0;
        for (int64_t c = 1; c < N; ++c) {
            if (coreLoad[static_cast<size_t>(c)] < coreLoad[static_cast<size_t>(best)]) {
                best = c;
            }
        }
        perCore[static_cast<size_t>(best)].push_back(blk);
        coreLoad[static_cast<size_t>(best)] += blk.cost;
    }

    metadataData[DLI_SCHED_FLAG_IDX] = 1;
    int64_t offset = 0;
    for (int64_t c = 0; c < N; ++c) {
        metadataData[DLI_SCHED_CNT_IDX + c] = static_cast<DLI_METADATA_T>(perCore[static_cast<size_t>(c)].size());
        metadataData[DLI_SCHED_OFF_IDX + c] = static_cast<DLI_METADATA_T>(offset);
        for (const auto& blk : perCore[static_cast<size_t>(c)]) {
            const int64_t entry = static_cast<int64_t>(DLI_SCHED_ITEM_IDX) + DLI_SCHED_ENTRY_STRIDE * offset;
            metadataData[entry] = blk.qOffset;
            metadataData[entry + 1] = blk.kBase;
            metadataData[entry + 2] = blk.rows;
            metadataData[entry + 3] = blk.vkMax;
            metadataData[entry + 4] = blk.vkParam;
            metadataData[entry + 5] = blk.qHeadOffset;
            metadataData[entry + 6] = 0;
            metadataData[entry + 7] = 0;
            ++offset;
        }
    }
    return true;
}

} // namespace aicpu

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_ARCH92_H
