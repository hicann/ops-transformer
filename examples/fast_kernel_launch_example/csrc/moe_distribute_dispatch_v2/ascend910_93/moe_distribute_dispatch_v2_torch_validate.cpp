/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_distribute_dispatch_v2_torch_validate.h"

namespace ascend_ops {
namespace MoeDistributeDispatchV2 {

static void InitValidateContext(MoeDistributeDispatchV2ValidateParams& params)
{
    params.maxEpWorldSize = ValidationConstants::MAX_EP_WORLD_SIZE;
    params.maxMoeExpertNum = ValidationConstants::MOE_EXPERT_MAX_NUM;
    params.maxH = ValidationConstants::H_MAX;
    params.maxBs = ValidationConstants::BS_UPPER_BOUND;

    params.isFullmeshV2 = (params.comm_alg == "fullmesh_v2");
    if (params.isFullmeshV2) {
        params.maxBs = ValidationConstants::FULLMESH_BS_UPPER_BOUND;
    }

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

    params.maxK = params.isFullmeshV2 ? ValidationConstants::FULLMESH_K_MAX : ValidationConstants::K_MAX;
    params.isScales = false;
    params.bs = 0;
    params.h = 0;
    params.k = 0;
}

static void ValidateEpAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.ep_world_size >= ValidationConstants::MIN_EP_WORLD_SIZE) &&
                    (params.ep_world_size <= params.maxEpWorldSize),
                "ep_world_size is invalid, only support [", ValidationConstants::MIN_EP_WORLD_SIZE, ", ",
                params.maxEpWorldSize, "], but got ", "ep_world_size: ", params.ep_world_size);

    TORCH_CHECK((params.ep_rank_id >= 0) && (params.ep_rank_id < params.ep_world_size),
                "ep_rank_id is invalid, only support [0, ep_world_size), but got",
                " ep_world_size: ", params.ep_world_size, ", ep_rank_id: ", params.ep_rank_id);
}

static void ValidateSharedExpertAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK(params.isNoShared || params.isValidShared,
                "shared_expert_num/rank_num invalid configs. Valid: (0,0), or (n,m) where m%n==0. ",
                "Got: shared_expert_num=", params.shared_expert_num,
                ", shared_expert_rank_num=", params.shared_expert_rank_num);

    TORCH_CHECK((params.shared_expert_rank_num >= 0) && (params.shared_expert_rank_num < params.ep_world_size),
                "shared_expert_rank_num is invalid, only support [0, ep_world_size), but got",
                " ep_world_size: ", params.ep_world_size, ", shared_expert_rank_num: ", params.shared_expert_rank_num);
}

static void ValidateMoeExpertAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.moe_expert_num > 0) && (params.moe_expert_num <= params.maxMoeExpertNum),
                "moe_expert_num is invalid, only support (0, ", params.maxMoeExpertNum, "], but got ",
                params.moe_expert_num);

    TORCH_CHECK(params.moe_expert_num % params.moeExpertNumDivisor == 0,
                "moe_expert_num should be divisible by (ep_world_size - shared_expert_rank_num), "
                "but moe_expert_num = ",
                params.moe_expert_num, ", ep_world_size = ", params.ep_world_size,
                ", shared_expert_rank_num = ", params.shared_expert_rank_num,
                ", divisor = ", params.moeExpertNumDivisor);

    TORCH_CHECK(params.localMoeExpertNum * params.ep_world_size <= ValidationConstants::LOCAL_EXPERT_MAX_SIZE,
                "localMoeExpertNum * ep_world_size must be less than or equal to ",
                ValidationConstants::LOCAL_EXPERT_MAX_SIZE,
                ", but localMoeExpertNum * ep_world_size = ", params.localMoeExpertNum * params.ep_world_size,
                ", localMoeExpertNum = ", params.localMoeExpertNum, ", ep_world_size = ", params.ep_world_size);
}

static void ValidateQuantModeAndScales(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.quant_mode == static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
                    (params.quant_mode == static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
                "quant_mode is invalid, only support 0 (NON_QUANT) or 2 (PERTOKEN_DYNAMIC_QUANT), but got ",
                params.quant_mode);
}

static void ValidateCommonAttrs(const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((params.expert_token_nums_type == 0) || (params.expert_token_nums_type == 1),
                "The expert_token_nums_type should be 0 or 1.");

    TORCH_CHECK(params.global_bs >= 0, "global_bs is invalid, only support >= 0, but got ", params.global_bs);

    TORCH_CHECK(params.zero_expert_num >= 0, "zero_expert_num is invalid, only support >= 0, but got ",
                params.zero_expert_num);

    TORCH_CHECK(params.copy_expert_num >= 0, "copy_expert_num is invalid, only support >= 0, but got ",
                params.copy_expert_num);

    TORCH_CHECK(params.const_expert_num >= 0, "const_expert_num is invalid, only support >= 0, but got ",
                params.const_expert_num);

    TORCH_CHECK((params.zeroComputeExpertNum + params.moe_expert_num) <= INT32_MAX,
                "zero_expert_num + copy_expert_num + const_expert_num + moe_expert_num exceeds INT32_MAX. ",
                "zero_expert_num: ", params.zero_expert_num, ", copy_expert_num: ", params.copy_expert_num,
                ", const_expert_num: ", params.const_expert_num, ", moe_expert_num: ", params.moe_expert_num);
}

static void ValidateGlobalBs(const MoeDistributeDispatchV2ValidateParams& params, int64_t bs, bool hasActiveMask)
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

static void ValidateInputTensorDim(const at::Tensor& x, const at::Tensor& expert_ids,
                                   MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2-dimension, but got ", x.dim(), " dimension");

    TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must be 2-dimension, but got ", expert_ids.dim(), " dimension");

    params.bs = x.size(0);
    params.h = x.size(1);
    params.k = expert_ids.size(1);

    TORCH_CHECK((params.h >= ValidationConstants::H_MIN) && (params.h <= params.maxH),
                "h (hidden size) is invalid, only support [", ValidationConstants::H_MIN, ", ", params.maxH,
                "], but got ", params.h);

    TORCH_CHECK((params.bs > 0) && (params.bs <= params.maxBs), "bs (batch size) is invalid, only support (0, ",
                params.maxBs, "], but got ", params.bs);

    TORCH_CHECK((params.k > 0) && (params.k <= params.maxK), "k (top-k) is invalid, only support (0, ", params.maxK,
                "], but got ", params.k);

    TORCH_CHECK(params.k <= params.moe_expert_num + params.zeroComputeExpertNum,
                "k (top-k) is invalid, should not exceed moe_expert_num + zero_expert_num + copy_expert_num + "
                "const_expert_num. ",
                "k: ", params.k, ", moe_expert_num: ", params.moe_expert_num,
                ", zero_expert_num: ", params.zero_expert_num, ", copy_expert_num: ", params.copy_expert_num,
                ", const_expert_num: ", params.const_expert_num);
}

static void ValidateOptionalTensorDim(const c10::optional<at::Tensor>& scales,
                                      const c10::optional<at::Tensor>& x_active_mask,
                                      const c10::optional<at::Tensor>& expert_scales,
                                      const c10::optional<at::Tensor>& performance_info,
                                      MoeDistributeDispatchV2ValidateParams& params)
{
    params.isScales = scales.has_value();

    if (x_active_mask.has_value()) {
        const auto& mask = x_active_mask.value();
        TORCH_CHECK((mask.dim() == 1) || (mask.dim() == 2),
                    "x_active_mask must be 1-dimension or 2-dimension, but got ", mask.dim(), " dimension");
        TORCH_CHECK(mask.size(0) == params.bs, "x_active_mask's dim0 not equal to bs, x_active_mask: ", mask.size(0),
                    ", bs: ", params.bs);
        if (mask.dim() == 2) {
            TORCH_CHECK(mask.size(1) == params.k, "x_active_mask's dim1 not equal to k, x_active_mask: ", mask.size(1),
                        ", k: ", params.k);
        }
    }

    if (expert_scales.has_value()) {
        const auto& exp_scales = expert_scales.value();
        TORCH_CHECK(exp_scales.dim() == 2, "expert_scales must be 2-dimension, but got ", exp_scales.dim(),
                    " dimension");
    }

    if (performance_info.has_value()) {
        const auto& perf = performance_info.value();
        TORCH_CHECK(perf.dim() == 1, "performance_info must be 1-dimension, but got ", perf.dim(), " dimension");
        TORCH_CHECK(perf.size(0) == params.ep_world_size,
                    "performance_info's dim0 not equal to ep_world_size, performance_info: ", perf.size(0),
                    ", ep_world_size: ", params.ep_world_size);
    }
}

static void ValidateInputDataType(const at::Tensor& x, const at::Tensor& expert_ids,
                                  const MoeDistributeDispatchV2ValidateParams& params)
{
    TORCH_CHECK((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf),
                "dtype of x should be BFloat16 or Float16, but got ", c10::toString(x.scalar_type()));

    TORCH_CHECK(expert_ids.scalar_type() == at::kInt, "dtype of expert_ids should be Int, but got ",
                c10::toString(expert_ids.scalar_type()));
}

static void ValidateOptionalTensorDataType(const c10::optional<at::Tensor>& scales,
                                           const c10::optional<at::Tensor>& x_active_mask,
                                           const c10::optional<at::Tensor>& expert_scales,
                                           const c10::optional<at::Tensor>& performance_info)
{
    if (scales.has_value()) {
        const auto& scales_tensor = scales.value();
        TORCH_CHECK(scales_tensor.scalar_type() == at::kFloat,
                    "scales datatype is invalid, datatype should be Float, but is ",
                    c10::toString(scales_tensor.scalar_type()));
    }

    if (x_active_mask.has_value()) {
        const auto& mask = x_active_mask.value();
        TORCH_CHECK(mask.scalar_type() == at::kBool,
                    "x_active_mask datatype is invalid, datatype should be Bool, but is ",
                    c10::toString(mask.scalar_type()));
    }

    if (expert_scales.has_value()) {
        const auto& exp_scales = expert_scales.value();
        TORCH_CHECK(exp_scales.scalar_type() == at::kFloat,
                    "expert_scales datatype is invalid, datatype should be Float, but is ",
                    c10::toString(exp_scales.scalar_type()));
    }

    if (performance_info.has_value()) {
        const auto& perf = performance_info.value();
        TORCH_CHECK(perf.scalar_type() == at::kLong,
                    "performance_info datatype is invalid, datatype should be Long, but is ",
                    c10::toString(perf.scalar_type()));
    }
}

bool ValidateMoeDistributeDispatchV2Input(const at::Tensor& x, const at::Tensor& expert_ids,
                                          const c10::optional<at::Tensor>& scales,
                                          const c10::optional<at::Tensor>& x_active_mask,
                                          const c10::optional<at::Tensor>& expert_scales,
                                          const c10::optional<at::Tensor>& performance_info,
                                          MoeDistributeDispatchV2ValidateParams& params)
{
    // 初始化和属性验证
    InitValidateContext(params);
    ValidateEpAttrs(params);
    ValidateSharedExpertAttrs(params);
    ValidateMoeExpertAttrs(params);
    ValidateQuantModeAndScales(params);
    ValidateCommonAttrs(params);

    // 张量维度验证
    ValidateInputTensorDim(x, expert_ids, params);
    ValidateOptionalTensorDim(scales, x_active_mask, expert_scales, performance_info, params);

    // global_bs验证
    ValidateGlobalBs(params, expert_ids.size(0), x_active_mask.has_value());

    // 数据类型验证
    ValidateInputDataType(x, expert_ids, params);
    ValidateOptionalTensorDataType(scales, x_active_mask, expert_scales, performance_info);

    return true;
}

} // namespace MoeDistributeDispatchV2
} // namespace ascend_ops
