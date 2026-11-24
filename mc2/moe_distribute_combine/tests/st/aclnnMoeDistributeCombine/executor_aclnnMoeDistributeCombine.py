# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""aclnnMoeDistributeCombine: shared A2/A3 reference and native target entry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from moe_distribute_combine_test_utils import reference, init_arguments, call, outputs

TARGET = "aclnnMoeDistributeCombine"
KIND = "combine"
VARIANT = 1
SIGNATURE = "aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize( const aclTensor* expandX, const aclTensor* expertIds, const aclTensor* expandIdx, const aclTensor* epSendCounts, const aclTensor* expertScales, const aclTensor* tpSendCounts, const aclTensor* xActiveMask, const aclTensor* activationScale, const aclTensor* weightScale, const aclTensor* groupList, const aclTensor* expandScales, const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype, int64_t commQuantMode, int64_t groupListType, aclTensor* x, uint64_t* workspaceSize, aclOpExecutor** executor)"


def generate_topk(num_tokens, num_experts, topk, seed, rank, balanced):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261119 + int(seed) * 1000003 + int(rank) * 1009)
    if balanced:
        experts = torch.randperm(num_experts, generator=generator, device="cpu")
        rows = torch.randperm(num_tokens, generator=generator, device="cpu")
        slots = (
            torch.arange(num_tokens * topk, device="cpu").reshape(num_tokens, topk)
            % num_experts
        )
        return experts[slots][rows].to(torch.int32)
    weights = torch.ones(num_experts, dtype=torch.float32, device="cpu")
    return torch.multinomial(
        weights.expand(num_tokens, -1), topk, replacement=False, generator=generator
    ).to(torch.int32)


def make_inputs(data, options, rank, world, case_id):
    x = data["x"].clone()
    bs, topk = x.shape[0], int(data["topk"])
    balanced = options["input_distribution"] == "moe_balanced"
    if balanced:
        if x.is_floating_point():
            x = x.round()
        x[:, 0] = rank
        x[:, 1] = torch.arange(bs, device=x.device)
        x[:, -1] = 127
    else:
        x[:, 0] = rank + torch.arange(bs, device=x.device) / 16
    expert_ids = generate_topk(bs, world, topk, case_id, rank, balanced).to(x.device)
    weights = torch.arange(1, topk + 1, device=x.device, dtype=torch.float32)
    weights = weights.expand(bs, topk).contiguous() / 4
    return x, expert_ids, weights


@register("moe_distribute_combine_reference")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        return reference(
            input_data,
            self.task_result,
            self.name == "cpu",
            make_inputs,
            generate_topk,
            KIND,
        )


@register("moe_distribute_combine_aclnn")
class AclnnExecutor(AclnnBaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def init_by_input_data(self, input_data):
        return init_arguments(
            self, input_data, make_inputs, generate_topk, KIND, VARIANT, TARGET
        )

    def __call__(self):
        call(self)

    def after_call(self, output_packages):
        return outputs(self)
