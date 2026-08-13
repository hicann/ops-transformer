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
 * \\file dense_lightning_indexer_softmax_lse_v2_metadata_aicpu_arch35.h
 * \brief arch35 (Ascend950) specific implementations for AICPU metadata kernel.
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_ARCH35_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_ARCH35_H

#include "../dense_lightning_indexer_softmax_lse_v2_metadata_aicpu.h"
#include "../../../dense_lightning_indexer_softmax_lse_v2/op_kernel/dense_lightning_indexer_softmax_lse_v2_metadata.h"

#include <algorithm>

using namespace optiling;

namespace aicpu {

inline int64_t DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::CalcTotalSize() const
{
    if (IsTensorValid(seqUsedQuery_)) {
        auto *data = reinterpret_cast<const int32_t *>(seqUsedQuery_->GetData());
        int64_t total = 0;
        for (int64_t i = 0; i < bSize_; ++i) {
            total += static_cast<int64_t>(data[i]);
        }
        return total;
    }
    if (layoutType_ == DliLayout::TND) {
        if (IsTensorValid(cuSeqLensQuery_)) {
            auto *data = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
            return static_cast<int64_t>(data[bSize_]);
        }
        return 0;
    }
    return bSize_ * s1Size_;
}

inline bool DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::BuildMetadata()
{
    int64_t totalNum = CalcTotalSize();
    if (totalNum < 0) {
        KERNEL_LOG_ERROR("Total valid sequence should be >= 0, but got %ld", totalNum);
        return false;
    }

    int64_t N = aicCoreNum_;
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

    auto *metadataData = reinterpret_cast<DLI_METADATA_T *>(metadata_->GetData());
    std::fill_n(metadataData, DLI_METADATA_SIZE, static_cast<DLI_METADATA_T>(0));
    auto *metadata = reinterpret_cast<detail::DenseLISoftmaxLseV2MetaData *>(metadataData);
    metadata->forecore_num = static_cast<int32_t>(forecoreNum);
    metadata->tail_core_num = static_cast<int32_t>(tailCoreNum);
    metadata->b_s1_per_core = static_cast<int32_t>(bS1PerCore);
    metadata->b_s1_per_tail_core = static_cast<int32_t>(bS1PerTailCore);
    return true;
}

} // namespace aicpu

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_ARCH35_H
