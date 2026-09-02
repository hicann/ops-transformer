/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_MSA_SPLIT_KV_EPILOGUE_DISPATCH_POLICY_HPP
#define EPILOGUE_MSA_SPLIT_KV_EPILOGUE_DISPATCH_POLICY_HPP

#include "../../attn_infra/msa_split_kv_base_defs.hpp"
#include "../../attn_infra/arch/msa_split_kv_arch.hpp"

namespace NpuArch::Epilogue {

enum class LseMode {
    NONE = 0,
    OUT_ONLY = 1
};

// Prefill Phase1 online softmax (bf16 S / fp32 S specializations)
struct EpilogueOnlineSoftmaxBsa {
    using ArchTag = Arch::AtlasA5;
};

// Prefill Phase2 rescale-O combine
struct EpilogueRescaleOSplitKvArch35 {
    using ArchTag = Arch::AtlasA5;
};

// For AtlasA2 prefill Phase1 online softmax (mirror A5, ArchTag = AtlasA2)
struct EpilogueOnlineSoftmaxPrefillA2 {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2 prefill Phase2 rescale-O combine (mirror A5, ArchTag = AtlasA2)
struct EpilogueRescaleOPrefillA2 {
    using ArchTag = Arch::AtlasA2;
};

} // namespace NpuArch::Epilogue

#endif // EPILOGUE_MSA_SPLIT_KV_EPILOGUE_DISPATCH_POLICY_HPP
