/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_TORCH_VALIDATE_H
#define MOE_DISTRIBUTE_COMBINE_V2_TORCH_VALIDATE_H

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ascend_ops {
namespace MoeDistributeCombineV2 {

/**
 * @brief 通信量化模式枚举
 */
enum class CommQuantMode : int32_t {
    NON_QUANT = 0,
    INT12_QUANT = 1,
    INT8_QUANT = 2,
    MXFP8_E5M2_QUANT = 3,
    MXFP8_E4M3_QUANT = 4
};

/**
 * @brief 验证参数常量结构体
 *
 * 包含所有用于输入验证的常量值，便于统一管理和调整
 */
struct ValidationConstants {
    static constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    static constexpr int64_t MAX_EP_WORLD_SIZE = 768;
    static constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
    static constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
    static constexpr int64_t K_MAX = 16;
    static constexpr int64_t H_MIN = 1024;
    static constexpr int64_t H_MAX = 8192;
    static constexpr int64_t BS_UPPER_BOUND = 512;
};

/**
 * @brief MOE分步组合验证参数结构体
 *
 * 用于验证MOE分布式组合操作的输入参数合法性
 */
struct MoeDistributeCombineV2ValidateParams {
    // 必需属性参数
    int64_t ep_world_size;
    int64_t ep_rank_id;
    int64_t moe_expert_num;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t global_bs;
    int64_t comm_quant_mode;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;

    // 计算出的中间变量
    int64_t maxEpWorldSize;
    int64_t maxMoeExpertNum;
    int64_t zeroComputeExpertNum;
    int64_t moeExpertNumDivisor;
    int64_t localMoeExpertNum;

    // 共享专家相关标志
    bool isNoShared;
    bool isValidShared;
};

/**
 * @brief 验证MOE分布式组合V2输入参数
 *
 * @param expand_x 扩展输入张量 [bs*k, h]
 * @param expert_ids 专家ID张量 [bs, k]
 * @param assist_info_for_combine 组合辅助信息张量 [bs*k]
 * @param ep_send_counts EP发送计数张量 [ep_world_size]
 * @param expert_scales 专家缩放张量 [bs, k]
 * @param x_active_mask 可选: 激活掩码张量
 * @param shared_expert_x 可选: 共享专家输入张量
 * @param ori_x 可选: 原始输入张量 [bs, h]
 * @param const_expert_alpha_1 可选: 常数专家alpha1张量
 * @param const_expert_alpha_2 可选: 常数专家alpha2张量
 * @param const_expert_v 可选: 常数专家V张量
 * @param performance_info 可选: 性能信息张量 [ep_world_size]
 * @param params 验证参结构体
 * @return 验证通过返回true，失败则抛出TorchCheckError异常
 */
bool ValidateMoeDistributeCombineV2Input(
    const at::Tensor& expand_x, const at::Tensor& expert_ids, const at::Tensor& assist_info_for_combine,
    const at::Tensor& ep_send_counts, const at::Tensor& expert_scales, const c10::optional<at::Tensor>& x_active_mask,
    const c10::optional<at::Tensor>& shared_expert_x, const c10::optional<at::Tensor>& ori_x,
    const c10::optional<at::Tensor>& const_expert_alpha_1, const c10::optional<at::Tensor>& const_expert_alpha_2,
    const c10::optional<at::Tensor>& const_expert_v, const c10::optional<at::Tensor>& performance_info,
    MoeDistributeCombineV2ValidateParams& params);

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops

#endif // MOE_DISTRIBUTE_COMBINE_V2_TORCH_VALIDATE_H
