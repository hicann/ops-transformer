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
 * \file moe_distribute_combine.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

/**
 * @brief Warpper for moe_distribute_combine
 */
at::Tensor NpuMoeDistributeCombine(
    const at::Tensor &context, const at::Tensor &expandX, const at::Tensor &expertIds,
    const at::Tensor &assistInfoForCombine, const at::Tensor &epSendCounts, const at::Tensor &expertScales,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize,
    const c10::optional<at::Tensor> &tpSendCounts, const c10::optional<at::Tensor> &xActiveMask,
    const c10::optional<at::Tensor> &expandScales, const c10::optional<at::Tensor> &sharedExpertX,
    const c10::optional<at::Tensor> &elasticInfo, const c10::optional<at::Tensor> &oriX,
    const c10::optional<at::Tensor> &constExpertAlpha1, const c10::optional<at::Tensor> &constExpertAlpha2,
    const c10::optional<at::Tensor> &constExpertV, const c10::optional<at::Tensor> &performanceInfo,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, std::string commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum)
{
    TORCH_CHECK((expandX.dim() == DIM_TWO) && (expertIds.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((expandX.scalar_type() == at::kBFloat16) || (expandX.scalar_type() == at::kHalf) ||
                    (expandX.scalar_type() == at::kInt),
                "dtype of expand_x should be BFloat16, Float16 or Int, but got " +
                    std::string(c10::toString(expandX.scalar_type())));
    TORCH_CHECK(expertIds.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expertIds.scalar_type())));
    auto expandXSize = expandX.sizes();
    auto expertIdsSize = expertIds.sizes();

    int64_t bs = expertIdsSize[0];
    int64_t h = expandXSize[1];
    int64_t globalBsReal = (globalBs == 0) ? (bs * epWorldSize) : globalBs;

    at::Tensor output;
    {
        auto localDevice = c10::Device(expandX.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        if (expandX.scalar_type() != at::kInt) {
            output = at::empty({bs, h}, expandX.options().dtype(expandX.scalar_type()));
        } else {
            output = at::empty({bs, h}, expandX.options().dtype(at::kBFloat16));
        }
    }
    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    int64_t outDtype = 0;
    int64_t groupListType = 0;

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlgStr.c_str());

    ACLNN_CMD(aclnnMoeDistributeCombineV5, context, expandX, expertIds, assistInfoForCombine, epSendCounts,
              expertScales, tpSendCounts, xActiveMask, nulltensor, nulltensor, nulltensor, expandScales, sharedExpertX,
              elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, epWorldSize,
              epRankId, moeExpertNum, cclBufferSize, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBsReal, outDtype, commQuantMode, groupListType, commAlgPtr, zeroExpertNum,
              copyExpertNum, constExpertNum, output);

    return output;
}
// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_distribute_combine", &NpuMoeDistributeCombine, "moe_distribute_combine");
}
} // namespace op_api
