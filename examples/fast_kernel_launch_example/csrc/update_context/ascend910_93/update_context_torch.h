/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UPDATE_CONTEXT_TORCH_H
#define UPDATE_CONTEXT_TORCH_H
#include <torch/extension.h>
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_mc2.h"
#include "hccl/hccl_res_expt.h"

struct Mc2ContextStru {
    uint64_t epRankId;
    uint64_t kfcContextAddr;
    uint64_t epHcclBuffer[1024];
};

std::tuple<at::Tensor, int64_t> update_context(const at::Tensor& x, c10::string_view group_ep, int64_t ep_world_size);

#endif
