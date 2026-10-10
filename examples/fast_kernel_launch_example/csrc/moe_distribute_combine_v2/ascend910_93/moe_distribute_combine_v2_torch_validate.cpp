/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_combine_v2_torch_validate.h"
#include <torch/torch.h>

namespace ascend_ops {
namespace MoeDistributeCombineV2 {

static void InitValidateContext(MoeDistributeCombineV2ValidateParams& params)
{
    params.maxEpWorldSize = ValidationConstants::MAX_EP_WORLD_SIZE;
    params.maxMoeExpertNum = ValidationConstants::MOE_EXPERT_MAX_NUM;

    TORCH_CHECK(params.zero_expert_num <= INT32_MAX,
                "zero_expert_num exceeds INT32_MAX: zero=", params.zero_expert_num);

    TORCH_CHECK(params.copy_expert_num <= INT32_MAX,
                "copy_expert_num exceeds INT32_MAX: copy=", params.copy_expert_num);

    TORCH_CHECK(params.const_expert_num <= INT32_MAX,
                "const_expert_num exceeds INT32_MAX: const=", params.const_expert_num);

    params.zeroComputeExpertNum = params.zero_expert_num + params.copy_expert_num + params.const_expert_num;

    TORCH_CHECK(params.ep_world_size > 0, "ep_world_size must be greater than 0, but got ", params.ep_world_size);

    TORCH_CHECK(params.shared_expert_num >= 0, "shared_expert_num must be >= 0, but got ", params.shared_expert_num);

    TORCH_CHECK(params.shared_expert_rank_num >= 0, "shared_expert_rank_num must be >= 0, but got ",
                params.shared_expert_rank_num);

    params.moeExpertNumDivisor = params.ep_world_size - params.shared_expert_rank_num;

    TORCH_CHECK(params.moeExpertNumDivisor > 0,
                "moeExpertNumDivisor (ep_world_size - shared_expert_rank_num) must be > 0, "
                "but ep_world_size = ",
                params.ep_world_size, ", shared_expert_rank_num = ", params.shared_expert_rank_num,
                ", resulting divisor = ", params.moeExpertNumDivisor);

    TORCH_CHECK(params.moe_expert_num > 0, "moe_expert_num must be > 0, but got ", params.moe_expert_num);

    params.localMoeExpertNum = params.moe_expert_num / params.moeExpertNumDivisor;

    TORCH_CHECK(params.localMoeExpertNum > 0, "localMoeExpertNum must be > 0, but got ", params.localMoeExpertNum);

    params.isNoShared = (params.shared_expert_num == 0) && (params.shared_expert_rank_num == 0);

    params.isValidShared = false;
    if (params.shared_expert_num > 0) {
        TORCH_CHECK(params.shared_expert_num != 0,
                    "shared_expert_num is 0, division by zero in isValidShared calculation");
        TORCH_CHECK((params.shared_expert_rank_num / params.shared_expert_num) > 0,
                    "shared_expert_rank_num / shared_expert_num must be > 0");
        TORCH_CHECK((params.shared_expert_rank_num % params.shared_expert_num) == 0,
                    "shared_expert_rank_num must be divisible by shared_expert_num");
        params.isValidShared = true;
    }
}

static void ValidateEpAttrs(const MoeDistributeCombineV2ValidateParams& params)
{
    TORCH_CHECK((params.ep_world_size >= ValidationConstants::MIN_EP_WORLD_SIZE) &&
                    (params.ep_world_size <= params.maxEpWorldSize),
                "ep_world_size invalid: support [", ValidationConstants::MIN_EP_WORLD_SIZE, ", ", params.maxEpWorldSize,
                "], got: ", params.ep_world_size);

    TORCH_CHECK((params.ep_rank_id >= 0) && (params.ep_rank_id < params.ep_world_size),
                "ep_rank_id invalid: support [0, ep_world_size), ep_world_size=", params.ep_world_size,
                ", ep_rank_id=", params.ep_rank_id);
}

static void ValidateSharedExpertAttrs(const MoeDistributeCombineV2ValidateParams& params)
{
    // 验证三种有效的共享专家配置场景
    TORCH_CHECK(params.isNoShared || params.isValidShared,
                "shared_expert_num/rank_num invalid configs. Valid: (0,0), or (n,m) where m%n==0. ",
                "Got: shared_expert_num=", params.shared_expert_num,
                ", shared_expert_rank_num=", params.shared_expert_rank_num);

    // 验证shared_expert_rank_num范围
    TORCH_CHECK((params.shared_expert_rank_num >= 0) && (params.shared_expert_rank_num < params.ep_world_size),
                "shared_expert_rank_num invalid: [0, ep_world_size), ep_world_size=", params.ep_world_size,
                ", rank_num=", params.shared_expert_rank_num);
}

static void ValidateMoeExpertAttrs(const MoeDistributeCombineV2ValidateParams& params)
{
    // 验证moe_expert_num范围
    TORCH_CHECK((params.moe_expert_num > 0) && (params.moe_expert_num <= params.maxMoeExpertNum),
                "moe_expert_num invalid: (0, ", params.maxMoeExpertNum, "], got: ", params.moe_expert_num);

    // 验证整除关系
    TORCH_CHECK(params.moe_expert_num % params.moeExpertNumDivisor == 0,
                "moe_expert_num divisible check failed: num=", params.moe_expert_num,
                ", divisor=", params.moeExpertNumDivisor);

    // 验证localMoeExpertNum * ep_world_size不超过限制
    TORCH_CHECK(params.localMoeExpertNum * params.ep_world_size <= ValidationConstants::LOCAL_EXPERT_MAX_SIZE,
                "localMoeExpertNum * ep_world_size exceeds ", ValidationConstants::LOCAL_EXPERT_MAX_SIZE, ": got ",
                params.localMoeExpertNum * params.ep_world_size);
}

static void ValidateQuantMode(const MoeDistributeCombineV2ValidateParams& params)
{
    TORCH_CHECK((params.comm_quant_mode >= static_cast<int64_t>(CommQuantMode::NON_QUANT)) &&
                    (params.comm_quant_mode <= static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANT)),
                "comm_quant_mode invalid: [", static_cast<int64_t>(CommQuantMode::NON_QUANT), ", ",
                static_cast<int64_t>(CommQuantMode::MXFP8_E4M3_QUANT), "], got: ", params.comm_quant_mode);
}

static void ValidateCommonAttrs(const MoeDistributeCombineV2ValidateParams& params)
{
    TORCH_CHECK(params.expert_shard_type == 0,
                "expert_shard_type invalid: only 0 supported, got: ", params.expert_shard_type);

    TORCH_CHECK(params.global_bs >= 0, "global_bs invalid: >= 0 required, got: ", params.global_bs);

    TORCH_CHECK(params.zero_expert_num >= 0, "zero_expert_num invalid: >= 0 required, got: ", params.zero_expert_num);

    TORCH_CHECK(params.copy_expert_num >= 0, "copy_expert_num invalid: >= 0 required, got: ", params.copy_expert_num);

    TORCH_CHECK(params.const_expert_num >= 0,
                "const_expert_num invalid: >= 0 required, got: ", params.const_expert_num);

    // 验证总专家数量不超过INT32_MAX
    TORCH_CHECK((params.zeroComputeExpertNum + params.moe_expert_num) <= INT32_MAX,
                "expert count exceeds INT32_MAX: zero=", params.zero_expert_num, ", copy=", params.copy_expert_num,
                ", const=", params.const_expert_num, ", moe=", params.moe_expert_num);
}

static void ValidateGlobalBs(const MoeDistributeCombineV2ValidateParams& params, int64_t bs, bool hasActiveMask)
{
    TORCH_CHECK(!((params.global_bs != 0) &&
                  ((params.global_bs < params.ep_world_size * bs) || (params.global_bs % params.ep_world_size != 0))),
                "global_bs invalid: only support 0 or max_bs * ep_world_size, "
                "but got global_bs=",
                params.global_bs, ", bs=", bs, ", ep_world_size=", params.ep_world_size);

    if (hasActiveMask) {
        TORCH_CHECK(params.global_bs <= params.ep_world_size * bs,
                    "Variable bs on different rank cannot work when isActiveMask=true, "
                    "global_bs=",
                    params.global_bs, ", bs=", bs, ", ep_world_size=", params.ep_world_size);
    }
}

static void ValidateRequiredTensorDims(const at::Tensor& expand_x, const at::Tensor& expert_ids,
                                       const at::Tensor& assist_info_for_combine, const at::Tensor& ep_send_counts,
                                       const at::Tensor& expert_scales)
{
    TORCH_CHECK(expand_x.dim() == 2, "expand_x dim invalid: 2D required, got ", expand_x.dim());

    TORCH_CHECK(expert_ids.dim() == 2, "expert_ids dim invalid: 2D required, got ", expert_ids.dim());

    TORCH_CHECK(assist_info_for_combine.dim() == 1, "assist_info dim invalid: 1D required, got ",
                assist_info_for_combine.dim());

    TORCH_CHECK(ep_send_counts.dim() == 1, "ep_send_counts dim invalid: 1D required, got ", ep_send_counts.dim());

    TORCH_CHECK(expert_scales.dim() == 2, "expert_scales dim invalid: 2D required, got ", expert_scales.dim());
}

static void ValidateDimValues(const at::Tensor& expand_x, const at::Tensor& expert_ids,
                              const MoeDistributeCombineV2ValidateParams& params, int64_t& bs, int64_t& h, int64_t& k)
{
    bs = expert_ids.size(0);
    h = expand_x.size(1);
    k = expert_ids.size(1);

    TORCH_CHECK((h >= ValidationConstants::H_MIN) && (h <= ValidationConstants::H_MAX), "h invalid: [",
                ValidationConstants::H_MIN, ", ", ValidationConstants::H_MAX, "], got: ", h);

    TORCH_CHECK((bs > 0) && (bs <= ValidationConstants::BS_UPPER_BOUND), "bs invalid: (0, ",
                ValidationConstants::BS_UPPER_BOUND, "], got: ", bs);

    TORCH_CHECK((k > 0) && (k <= ValidationConstants::K_MAX), "k invalid: (0, ", ValidationConstants::K_MAX,
                "], got: ", k);

    TORCH_CHECK(k <= params.moe_expert_num + params.zeroComputeExpertNum, "k exceeds expert_num limit: k=", k,
                ", moe=", params.moe_expert_num, ", zero_compute=", params.zeroComputeExpertNum);
}

static void ValidateOptionalDims(const c10::optional<at::Tensor>& x_active_mask,
                                 const c10::optional<at::Tensor>& shared_expert_x,
                                 const c10::optional<at::Tensor>& ori_x, const c10::optional<at::Tensor>& const_alpha1,
                                 const c10::optional<at::Tensor>& const_alpha2,
                                 const c10::optional<at::Tensor>& const_v,
                                 const c10::optional<at::Tensor>& performance_info, int64_t bs, int64_t h, int64_t k,
                                 const MoeDistributeCombineV2ValidateParams& params)
{
    // x_active_mask验证
    if (x_active_mask.has_value()) {
        const auto& mask = x_active_mask.value();
        TORCH_CHECK((mask.dim() == 1) || (mask.dim() == 2), "x_active_mask dim invalid: 1D or 2D, got ", mask.dim());
        TORCH_CHECK(mask.size(0) == bs, "x_active_mask dim0 mismatch: expected ", bs, ", got ", mask.size(0));
        if (mask.dim() == 2) {
            TORCH_CHECK(mask.size(1) == k, "x_active_mask dim1 mismatch: expected ", k, ", got ", mask.size(1));
        }
    }

    // shared_expert_x验证
    if (shared_expert_x.has_value()) {
        const auto& sxp = shared_expert_x.value();
        TORCH_CHECK((sxp.dim() == 2) || (sxp.dim() == 3), "shared_expert_x dim invalid: 2D or 3D, got ", sxp.dim());
    }

    // ori_x验证
    if (ori_x.has_value()) {
        const auto& ox = ori_x.value();
        TORCH_CHECK(ox.dim() == 2, "ori_x dim invalid: 2D, got ", ox.dim());
        TORCH_CHECK(ox.size(0) == bs, "ori_x dim0 mismatch: ", bs, ", got ", ox.size(0));
        TORCH_CHECK(ox.size(1) == h, "ori_x dim1 mismatch: ", h, ", got ", ox.size(1));
    }

    // const_expert_alpha_1/2/v验证
    if (const_alpha1.has_value()) {
        TORCH_CHECK(const_alpha1.value().dim() == 2, "const_expert_alpha_1 dim invalid: 2D, got ",
                    const_alpha1.value().dim());
    }
    if (const_alpha2.has_value()) {
        TORCH_CHECK(const_alpha2.value().dim() == 2, "const_expert_alpha_2 dim invalid: 2D, got ",
                    const_alpha2.value().dim());
    }
    if (const_v.has_value()) {
        TORCH_CHECK(const_v.value().dim() == 2, "const_expert_v dim invalid: 2D, got ", const_v.value().dim());
    }

    // performance_info验证
    if (performance_info.has_value()) {
        const auto& perf = performance_info.value();
        TORCH_CHECK(perf.dim() == 1, "performance_info dim invalid: 1D, got ", perf.dim());
        TORCH_CHECK(perf.size(0) == params.ep_world_size, "performance_info size mismatch: ", params.ep_world_size,
                    ", got ", perf.size(0));
    }
}

static void ValidateRequiredDataTypes(const at::Tensor& expand_x, const at::Tensor& expert_ids,
                                      const at::Tensor& assist_info_for_combine, const at::Tensor& ep_send_counts,
                                      const at::Tensor& expert_scales)
{
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf),
                "expand_x dtype invalid: BF16 or FP16 required, got ", c10::toString(expand_x.scalar_type()));

    TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "expert_ids dtype invalid: Int32 required, got ",
                c10::toString(expert_ids.scalar_type()));

    TORCH_CHECK(assist_info_for_combine.scalar_type() == at::kInt, "assist_info dtype invalid: Int32 required, got ",
                c10::toString(assist_info_for_combine.scalar_type()));

    TORCH_CHECK(ep_send_counts.scalar_type() == at::kInt, "ep_send_counts dtype invalid: Int32 required, got ",
                c10::toString(ep_send_counts.scalar_type()));

    TORCH_CHECK(expert_scales.scalar_type() == at::kFloat, "expert_scales dtype invalid: Float required, got ",
                c10::toString(expert_scales.scalar_type()));
}

static void ValidateOptionalDataTypes(const c10::optional<at::Tensor>& x_active_mask,
                                      const c10::optional<at::Tensor>& shared_expert_x,
                                      const c10::optional<at::Tensor>& ori_x,
                                      const c10::optional<at::Tensor>& const_alpha1,
                                      const c10::optional<at::Tensor>& const_alpha2,
                                      const c10::optional<at::Tensor>& const_v,
                                      const c10::optional<at::Tensor>& performance_info, const at::Tensor& expand_x)
{
    if (x_active_mask.has_value()) {
        const auto& mask = x_active_mask.value();
        TORCH_CHECK(mask.scalar_type() == at::kBool, "x_active_mask dtype invalid: Bool required, got ",
                    c10::toString(mask.scalar_type()));
    }

    if (shared_expert_x.has_value()) {
        const auto& sxp = shared_expert_x.value();
        TORCH_CHECK(sxp.scalar_type() == expand_x.scalar_type(), "shared_expert_x dtype mismatch: expected ",
                    c10::toString(expand_x.scalar_type()), ", got ", c10::toString(sxp.scalar_type()));
    }

    if (ori_x.has_value()) {
        const auto& ox = ori_x.value();
        TORCH_CHECK(ox.scalar_type() == expand_x.scalar_type(),
                    "ori_x dtype mismatch: ", c10::toString(expand_x.scalar_type()), ", got ",
                    c10::toString(ox.scalar_type()));
    }

    if (const_alpha1.has_value()) {
        const auto& ca1 = const_alpha1.value();
        TORCH_CHECK(ca1.scalar_type() == expand_x.scalar_type(),
                    "const_expert_alpha_1 dtype mismatch: ", c10::toString(expand_x.scalar_type()));
    }

    if (const_alpha2.has_value()) {
        const auto& ca2 = const_alpha2.value();
        TORCH_CHECK(ca2.scalar_type() == expand_x.scalar_type(),
                    "const_expert_alpha_2 dtype mismatch: ", c10::toString(expand_x.scalar_type()));
    }

    if (const_v.has_value()) {
        const auto& cv = const_v.value();
        TORCH_CHECK(cv.scalar_type() == expand_x.scalar_type(),
                    "const_expert_v dtype mismatch: ", c10::toString(expand_x.scalar_type()));
    }

    if (performance_info.has_value()) {
        const auto& perf = performance_info.value();
        TORCH_CHECK(perf.scalar_type() == at::kLong, "performance_info dtype invalid: Int64 required, got ",
                    c10::toString(perf.scalar_type()));
    }
}

bool ValidateMoeDistributeCombineV2Input(
    const at::Tensor& expand_x, const at::Tensor& expert_ids, const at::Tensor& assist_info_for_combine,
    const at::Tensor& ep_send_counts, const at::Tensor& expert_scales, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& shared_expert_x, const c10::optional<at::Tensor>& ori_x,
    const c10::optional<at::Tensor>& const_expert_alpha_1, const c10::optional<at::Tensor>& const_expert_alpha_2,
    const c10::optional<at::Tensor>& const_expert_v, const c10::optional<at::Tensor>& performance_info,
    MoeDistributeCombineV2ValidateParams& params)
{
    // 初始化和属性验证
    InitValidateContext(params);
    ValidateEpAttrs(params);
    ValidateSharedExpertAttrs(params);
    ValidateMoeExpertAttrs(params);
    ValidateQuantMode(params);
    ValidateCommonAttrs(params);

    // 张量维度验证
    int64_t bs, h, k;
    ValidateRequiredTensorDims(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales);
    ValidateDimValues(expand_x, expert_ids, params, bs, h, k);
    ValidateOptionalDims(x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2,
                         const_expert_v, performance_info, bs, h, k, params);

    // global_bs验证
    ValidateGlobalBs(params, bs, x_active_mask.has_value());

    // 数据类型验证
    ValidateRequiredDataTypes(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales);
    ValidateOptionalDataTypes(x_active_mask, shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2,
                              const_expert_v, performance_info, expand_x);

    return true;
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops
