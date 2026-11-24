# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A3 DistributeBarrier 仅作功能性冒烟验证。

算子没有数值输出；ATK 的 equal 比较仅检查调用前后 xRef 内容保持不变，
结合调用成功完成判定功能性结果，不涉及数值计算精度验证。
当前用例未专门验证跨 rank 等待、提前返回及屏障前后依赖次序。
"""

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("a3_barrier_reference")
class BarrierReference(BaseApi):
    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnDistributeBarrierGetWorkspaceSize("
            "aclTensor* xRef, const char* group, int64_t worldSize, "
            "uint64_t* workspaceSize, aclOpExecutor** executor)"
        )

    def __call__(self, input_data, with_output=False):
        dist.barrier()
        return [input_data.kwargs["xRef"].clone()]


@register("a3_barrier_aclnn")
class BarrierAclnn(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        info = self.task_result.dist_task_info
        self.x = input_data.kwargs["xRef"].to(f"npu:{info.rank}")
        group = _get_default_group()
        input_data.kwargs = {
            "xRef": self.x,
            "group": group._get_backend(torch.device("npu")).get_hccl_comm_name(
                info.rank
            ),
            "worldSize": info.world_size,
        }
        args, outputs = super().init_by_input_data(input_data)
        self.x_pack = args[0]
        # xRef is an input dependency; the ABI has no additional output tensor.
        return args[:3], outputs

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        # 将输入 xRef 作为 ATK 比较对象，仅检查内容保持；它不是算子数值输出。
        return [self.acl_tensor_to_torch(self.x_pack)]
