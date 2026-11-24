# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A3 EPLB fixed and historical random cases through one ACLNN interface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ctypes

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from moe_update_expert_test_options import OPTION_NAMES, OPTIONS

SIGNATURE = """aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(
    const aclTensor* expertIds, const aclTensor* eplbTable,
    const aclTensor* expertScalesOptional, const aclTensor* pruningThresholdOptional,
    const aclTensor* activeMaskOptional, int64_t localRankId, int64_t worldSize,
    int64_t balanceMode, aclTensor* balancedExpertIds, aclTensor* balancedActiveMask,
    uint64_t* workspaceSize, aclOpExecutor** executor);"""
INPUT_NAMES = (
    "expertIds",
    "eplbTable",
    "expertScalesOptional",
    "pruningThresholdOptional",
    "activeMaskOptional",
    "localRankId",
    "worldSize",
    "balanceMode",
)
OPTIONAL_NAMES = (
    "expertScalesOptional",
    "pruningThresholdOptional",
    "activeMaskOptional",
)


def fixed_inputs(data, info, case_id):
    ids = torch.arange(16).reshape(8, 2).remainder(4).to(data["expertIds"].dtype)
    values = {
        "expertIds": ids,
        "eplbTable": torch.tensor(
            [[2, 0, 3], [1, 1, 0], [2, 4, 2], [1, 5, 0]], dtype=torch.int32
        ),
        "expertScalesOptional": torch.tensor(
            [[0.75, 0.25]] * 8, dtype=data["expertScalesOptional"].dtype
        ),
        "pruningThresholdOptional": torch.tensor([0.1, 0.4]),
        "activeMaskOptional": torch.arange(8) < 6,
        "localRankId": info.rank,
        "worldSize": info.world_size,
        "balanceMode": data["balanceMode"],
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261119 + int(case_id) * 1000003 + int(info.rank) * 1009)
    rows = torch.randperm(ids.shape[0], generator=generator, device="cpu")
    for name in ("expertIds", "expertScalesOptional", "activeMaskOptional"):
        values[name] = values[name].index_select(0, rows)
    for name in OPTIONAL_NAMES:
        if data[name].numel() == 0:
            values[name] = torch.empty(0, dtype=data[name].dtype)
    return values


def random_inputs(data, case_id):
    # The random draw order preserves the historical case-ID seed exactly.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(case_id))
    world = int(data["worldSize"])
    width = torch.randint(2, world + 2, (1,), generator=generator).item()
    rank = torch.randint(0, world, (1,), generator=generator).item()
    bs, topk = data["expertIds"].shape
    experts = data["eplbTable"].shape[0]
    counts = torch.randint(1, width, (experts, 1), generator=generator)
    physical = torch.zeros(experts, width - 1, dtype=torch.int32)
    ranks = torch.arange(world)
    for expert in range(experts):
        shuffled = ranks[torch.randperm(world, generator=generator)]
        count = int(counts[expert, 0])
        physical[expert, :count] = shuffled[:count]
    table = torch.cat([counts, physical], dim=1).to(torch.int32)
    ids = torch.randint(0, experts, (bs, topk), generator=generator).to(
        data["expertIds"].dtype
    )
    scales = torch.empty(0, dtype=data["expertScalesOptional"].dtype)
    threshold = torch.empty(0, dtype=torch.float32)
    mask = torch.empty(0, dtype=torch.bool)
    if data["expertScalesOptional"].numel():
        scales = torch.sort(
            torch.rand(bs, topk, generator=generator) * 0.25, dim=1, descending=True
        ).values
        threshold = (
            torch.rand(
                tuple(data["pruningThresholdOptional"].shape), generator=generator
            )
            * 0.15
        )
        num_true = torch.randint(0, bs + 1, (1,), generator=generator).item()
        mask = torch.cat(
            [
                torch.ones(num_true, dtype=torch.bool),
                torch.zeros(bs - num_true, dtype=torch.bool),
            ]
        )
        scales = scales.to(data["expertScalesOptional"].dtype)
        if data["activeMaskOptional"].numel() == 0:
            mask = torch.empty(0, dtype=torch.bool)
    return {
        "expertIds": ids,
        "eplbTable": table,
        "expertScalesOptional": scales,
        "pruningThresholdOptional": threshold,
        "activeMaskOptional": mask,
        "localRankId": rank,
        "worldSize": world,
        "balanceMode": data["balanceMode"],
    }


def case_inputs(kwargs, task_result):
    data = dict(kwargs)
    options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
    profile = options["input_distribution"]
    if profile == "eplb_fixed":
        data = fixed_inputs(
            data, task_result.dist_task_info, task_result.case_config.id
        )
    elif profile == "eplb_random":
        data = random_inputs(data, task_result.case_config.id)
    return {name: data[name] for name in INPUT_NAMES}, profile


def reference(data, compare_mask):
    ids = data["expertIds"]
    table = data["eplbTable"]
    counts = table[ids.long(), 0].long()
    if data["balanceMode"] == 0:
        columns = (
            data["localRankId"] // ((data["worldSize"] + counts - 1) // counts) + 1
        )
    else:
        columns = torch.arange(ids.shape[0], device=ids.device)[:, None] % counts + 1
    balanced = table[ids.long(), columns].to(ids.dtype)
    if not compare_mask:
        return [balanced]
    mask = torch.ones_like(ids, dtype=torch.bool)
    if data["expertScalesOptional"].numel():
        scales = data["expertScalesOptional"].float()
        mask = (
            scales >= scales.sum(dim=1, keepdim=True) * data["pruningThresholdOptional"]
        )
        if data["activeMaskOptional"].numel():
            mask &= data["activeMaskOptional"][:, None]
    return [balanced, mask]


@register("moe_update_expert_reference")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        data, profile = case_inputs(input_data.kwargs, self.task_result)
        # Historical random cases use CPU for both references; fixed cases keep
        # their independent NPU small-op benchmark.
        device = (
            "cpu"
            if self.name == "cpu" or profile == "eplb_random"
            else f"npu:{self.task_result.dist_task_info.rank}"
        )
        data = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in data.items()
        }
        compare_mask = (
            profile != "eplb_random" or data["expertScalesOptional"].numel() != 0
        )
        return reference(data, compare_mask)


@register("moe_update_expert_aclnn")
class AclnnExecutor(AclnnBaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def init_by_input_data(self, input_data):
        data, profile = case_inputs(input_data.kwargs, self.task_result)
        input_data.kwargs = data
        args, outputs = super().init_by_input_data(input_data)
        for index, name in enumerate(OPTIONAL_NAMES, start=2):
            if data[name].numel() == 0:
                args[index] = ctypes.POINTER(AclTensor)()
        # The ABI always writes both output tensors. Historical non-pruning
        # cases compare only IDs, so the extra mask is not an accuracy output.
        if profile == "eplb_random" and data["expertScalesOptional"].numel() == 0:
            args.append(
                self.torch_tensor_to_acl(
                    torch.zeros_like(data["expertIds"], dtype=torch.bool)
                )
            )
        return args, outputs

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(output) for output in output_packages]
