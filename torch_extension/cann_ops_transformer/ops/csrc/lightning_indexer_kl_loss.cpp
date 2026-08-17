/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor lightning_indexer_kl_loss(const at::Tensor &targetScore, const at::Tensor &indexProbs, float eps,
                                     const std::string &weightType)
{
    TORCH_CHECK((eps > 0), "eps should be greater than 0, current is: ", eps);
    TORCH_CHECK((weightType == "logits" || weightType == "probs"),
                "weightType must be 'logits' or 'probs', current is: ", weightType);
    TORCH_CHECK((targetScore.scalar_type() == indexProbs.scalar_type()),
                "targetScore and indexProbs should have the same dtype.");
    TORCH_CHECK((targetScore.dim() >= 2 && targetScore.dim() <= 3),
                "targetScore.dim() should be in [2, 3], current is: ", targetScore.dim());
    TORCH_CHECK((indexProbs.dim() >= 2 && indexProbs.dim() <= 3),
                "indexProbs.dim() should be in [2, 3], current is: ", indexProbs.dim());

    at::Tensor loss{nullptr};
    {
        auto local_device = c10::Device(targetScore.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        // 输出为标量，分配一个单元素 tensor
        loss = at::empty({}, targetScore.options());
    }

    char *weightTypePtr = const_cast<char *>(weightType.c_str());
    ACLNN_CMD(aclnnLightningIndexerKLLoss, targetScore, indexProbs, eps, weightTypePtr, loss);

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer_kl_loss", &lightning_indexer_kl_loss, "lightning_indexer_kl_loss", py::arg("target_score"),
          py::arg("index_probs"), py::arg("eps"), py::arg("weight_type"));
}

} // namespace op_api
