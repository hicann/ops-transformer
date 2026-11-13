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

#include "../../attn_infra/gbsa_base_defs.hpp"
#include "../../attn_infra/arch/gbsa_arch.hpp"

namespace NpuArch::Gemm {

// Block Mmad Policies

template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

template <bool ASYNC_ = false>
struct MmadAtlasA5Base {
    using ArchTag = Arch::AtlasA5;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadAtlasA2Base<false>;
using MmadAtlasA5 = MmadAtlasA5Base<false>;

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false, bool HAS_RESIDUAL_BLOCK_ = false,
          bool PRESERVE_SPARSE_ORDER_ = false>
struct MmadAtlasA2SFAIQK : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool HAS_RESIDUAL_BLOCK = HAS_RESIDUAL_BLOCK_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false, bool HAS_RESIDUAL_BLOCK_ = false,
          bool PRESERVE_SPARSE_ORDER_ = false>
struct MmadAtlasA2SFAIPV : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool HAS_RESIDUAL_BLOCK = HAS_RESIDUAL_BLOCK_;
};

// feat/lch/gbsa A5 QK template (ND: Q * K^T). Keep this free of union flags so
// the base path stays mergeable with upstream A5 work.
template <bool PAGED_CACHE_FLAG_ = false, bool HAS_RESIDUAL_BLOCK_ = false, bool PRECOMPUTE_SPARSE_ = false>
struct MmadAtlasA5BsaQK : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool HAS_RESIDUAL_BLOCK = HAS_RESIDUAL_BLOCK_;
    static constexpr bool PRECOMPUTE_SPARSE = PRECOMPUTE_SPARSE_;
};

// Union (maskType=3) QK: K * Q^T = S^T. Independent of MmadAtlasA5BsaQK so
// union can evolve without touching the lch A5 ND template.
template <bool PAGED_CACHE_FLAG_ = false, bool HAS_RESIDUAL_BLOCK_ = false, bool PRECOMPUTE_SPARSE_ = false>
struct MmadAtlasA5UnionQK : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool HAS_RESIDUAL_BLOCK = HAS_RESIDUAL_BLOCK_;
    static constexpr bool PRECOMPUTE_SPARSE = PRECOMPUTE_SPARSE_;
};

template <bool PAGED_CACHE_FLAG_ = false, bool HAS_RESIDUAL_BLOCK_ = false, bool PRECOMPUTE_SPARSE_ = false>
struct MmadAtlasA5BsaPV : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
    static constexpr bool HAS_RESIDUAL_BLOCK = HAS_RESIDUAL_BLOCK_;
    static constexpr bool PRECOMPUTE_SPARSE = PRECOMPUTE_SPARSE_;
};

// MXFP8 owns separate QK/PV implementations. Paging is resolved by the
// packed descriptor producer before these blocks consume the sparse records.
struct MmadAtlasA5GbsaQKMxfp8 : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
};

struct MmadAtlasA5GbsaPVMxfp8 : public MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
};

} // namespace NpuArch::Gemm

#endif // GEMM_DISPATCH_POLICY_HPP
