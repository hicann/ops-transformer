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
 * \\file dense_lightning_indexer_softmax_lse_v2_metadata_aicpu.h
 * \brief
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_H

#include <cstdint>
#include <string>
#include <type_traits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"

namespace aicpu {

enum class DliSparseMode : int32_t {
    NO_MASK = 0,
    RIGHT_DOWN_CAUSAL = 3,
};

enum class DliLayout {
    BSND = 0,
    TND = 1,
};

enum class ValidSocVersion {
    ASCEND910 = 0,
    ASCEND950
};

inline bool IsTensorValid(Tensor *tensor)
{
    return tensor != nullptr && tensor->GetData() != nullptr && tensor->GetTensorShape() != nullptr;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type GetAttrValue(CpuKernelContext &ctx,
                                                                               const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = static_cast<T>(attr->GetInt());
    return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
        return false;
    }
    value = attr->GetString();
    return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type GetAttrValueOpt(CpuKernelContext &ctx,
                                                                                  const std::string &name, T &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = static_cast<T>(attr->GetInt());
    }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, std::string &value)
{
    auto attr = ctx.GetAttr(name);
    if (attr != nullptr) {
        value = attr->GetString();
    }
}

class DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel : public CpuKernel {
public:
    DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel() = default;
    ~DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool CheckTensorValues();
    ValidSocVersion ProcessSocVersion();
    bool BuildMetadata();
    int64_t CalcTotalSize() const;

    Tensor *cuSeqLensQuery_ = nullptr;
    Tensor *cuSeqLensKey_ = nullptr;
    Tensor *seqUsedQuery_ = nullptr;
    Tensor *seqUsedKey_ = nullptr;
    Tensor *cmpResidualKey_ = nullptr;
    Tensor *metadata_ = nullptr;

    int64_t bSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t numHeadsQ_ = 0;
    int64_t numHeadsK_ = 0;
    int64_t headDim_ = 0;
    int64_t maskMode_ = 0;
    int64_t cmpRatio_ = 1;
    int64_t aicCoreNum_ = 0;
    std::string socVersion_;
    std::string layout_ = "BSND";
    std::string layoutK_ = "BSND";
    DliLayout layoutType_ = DliLayout::BSND;
};

} // namespace aicpu

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_AICPU_H
