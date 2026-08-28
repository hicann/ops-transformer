/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_DISPATCH_POLICY_HPP
#define EPILOGUE_DISPATCH_POLICY_HPP

#include "../../attn_infra/base_defs.hpp"
#include "../../attn_infra/arch/arch.hpp"

namespace NpuArch::Epilogue {

// Prefill Phase1 online softmax (bf16 S / fp32 S specializations)
struct EpilogueOnlineSoftmaxBsa {
    using ArchTag = Arch::AtlasA5;
};

// Prefill Phase2 rescale-O combine
struct EpilogueRescaleOSplitKvArch35 {
    using ArchTag = Arch::AtlasA5;
};

} // namespace NpuArch::Epilogue

#endif // EPILOGUE_DISPATCH_POLICY_HPP
