/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_POLICY_HPP
#define DISPATCH_POLICY_HPP

#include "../../attn_infra/base_defs.hpp"
#include "../../attn_infra/arch/arch.hpp"

namespace NpuArch::Gemm {

template <bool ASYNC_ = false>
struct MmadAtlasA5Base {
    using ArchTag = Arch::AtlasA5;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA5 = MmadAtlasA5Base<false>;

struct MmadAtlasA5SplitKvQK : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
};

struct MmadAtlasA5SplitKvPV : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
};

}  // namespace NpuArch::Gemm

#endif  // GEMM_DISPATCH_POLICY_HPP
