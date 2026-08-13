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
 * \file sparse_flash_mla_softmax_l1_norm_metadata_aicpu_arch35.h
 * \brief arch35 (Ascend950) specific implementations for AICPU metadata kernel.
 */
#ifndef SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_AICPU_ARCH35_H
#define SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_AICPU_ARCH35_H

#include "../sparse_flash_mla_softmax_l1_norm_metadata_aicpu.h"
#include "../../../sparse_flash_mla_softmax_l1_norm/op_kernel/sparse_flash_mla_softmax_l1_norm_metadata.h"

#include <algorithm>

using namespace optiling;

namespace aicpu {

template <typename T>
inline T CeilDiv(T x, T y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

inline int64_t SparseFlashMlaSoftmaxL1NormMetadataCpuKernel::CalcTotalSize() const
{
    if (IsTensorValid(seqUsedQuery_)) {
        auto *data = reinterpret_cast<const int32_t *>(seqUsedQuery_->GetData());
        int64_t total = 0;
        for (int64_t i = 0; i < bSize_; ++i) {
            total += static_cast<int64_t>(data[i]);
        }
        return total;
    }
    if (layoutType_ == SmlaLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
        auto *data = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
        return static_cast<int64_t>(data[bSize_]);
    }
    return bSize_ * s1Size_;
}

inline bool SparseFlashMlaSoftmaxL1NormMetadataCpuKernel::BuildMetadata()
{
    int64_t totalNum = CalcTotalSize();
    if (totalNum < 0) {
        KERNEL_LOG_ERROR("Total valid sequence should be >= 0, but got %ld", totalNum);
        return false;
    }

    int64_t totalCoreNum = std::min<int64_t>(std::min(totalNum, aicCoreNum_), SMLA_METADATA_MAX_CORE_NUM);
    int64_t formerCoreProcessNum = CeilDiv(totalNum, totalCoreNum);
    int64_t remainCoreProcessNum = totalNum / totalCoreNum;
    int64_t remainder = totalNum % totalCoreNum;
    int64_t remainCoreNum = (remainder == 0) ? 0 : (totalCoreNum - remainder);

    auto *metadataData = reinterpret_cast<SMLA_METADATA_T *>(metadata_->GetData());
    std::fill_n(metadataData, SMLA_METADATA_SIZE, static_cast<SMLA_METADATA_T>(0));
    auto *metadata = reinterpret_cast<detail::SmlaSoftmaxL1NormMetaData *>(metadataData);
    metadata->totalNum = static_cast<int32_t>(totalNum);
    metadata->formerCoreProcessNum = static_cast<int32_t>(formerCoreProcessNum);
    metadata->remainCoreProcessNum = static_cast<int32_t>(remainCoreProcessNum);
    metadata->remainCoreNum = static_cast<int32_t>(remainCoreNum);
    metadata->totalCoreNum = static_cast<int32_t>(totalCoreNum);
    return true;
}

} // namespace aicpu

#endif // SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_AICPU_ARCH35_H
