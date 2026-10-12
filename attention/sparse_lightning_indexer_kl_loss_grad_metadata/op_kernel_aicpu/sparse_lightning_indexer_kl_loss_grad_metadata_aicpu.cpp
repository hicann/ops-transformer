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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.cpp
 * \brief Registration entry for AICPU kernel with runtime arch selection by soc_version
 */

#include "sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"
#include "arch22/sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch22.h"
#include "arch35/sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch35.h"

namespace aicpu {

class SparseLightningIndexerKLLossGradMetadataCpuKernel : public CpuKernel {
public:
    SparseLightningIndexerKLLossGradMetadataCpuKernel() = default;
    ~SparseLightningIndexerKLLossGradMetadataCpuKernel() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override
    {
        std::string socVersion;
        if (!GetAttrValue(ctx, "soc_version", socVersion)) {
            KERNEL_LOG_ERROR("Get soc_version failed!");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (socVersion.find("Ascend910") != std::string::npos) {
            return arch22Kernel_.Compute(ctx);
        } else {
            return arch35Kernel_.Compute(ctx);
        }
    }

private:
    SparseLightningIndexerKLLossGradMetadataCpuKernelArch22 arch22Kernel_;
    SparseLightningIndexerKLLossGradMetadataCpuKernelArch35 arch35Kernel_;
};

static const char* kernelType = "SparseLightningIndexerKLLossGradMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseLightningIndexerKLLossGradMetadataCpuKernel);

} // namespace aicpu
