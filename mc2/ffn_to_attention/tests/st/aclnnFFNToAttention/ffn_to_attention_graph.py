# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Graph capability is retained separately from the default ACLNN task.
import torch
import torchair
import torch_npu
import logging
from typing import Tuple


def compile_model(model, graph_type, is_full_core=True):
    """图模式编译模型"""
    if graph_type == 0:
        logging.debug("graph_type=0，测试场景：单算子模式")
        return model
    from torchair.configs.compiler_config import CompilerConfig

    if is_full_core:
        compiler_config = None
    else:
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = "12|24"
        print("当前图模式仅使用: 12 aic核, 24 aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
    torch._dynamo.reset()
    if graph_type == 1:
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:
        print("graph_type=2, 测试场景: ACLNN入图模式, 动态shape+二进制场景")
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)
    return compiled_model


class FFN_TO_ATTENTION_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        session_ids: torch.Tensor,
        micro_batch_ids: torch.Tensor,
        token_ids: torch.Tensor,
        expert_offsets: torch.Tensor,
        actual_token_num: torch.Tensor,
        attn_rank_table: torch.Tensor,
        *,
        group: str = "",
        world_size=0,
        token_info_table_shape: Tuple[int, int, int] = (0, 0, 0),
        token_data_shape: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        token_info_table_shape_list = [
            int(token_info_table_shape[0]),
            int(token_info_table_shape[1]),
            int(token_info_table_shape[2]),
        ]
        token_data_shape_list = [
            int(token_data_shape[0]),
            int(token_data_shape[1]),
            int(token_data_shape[2]),
            int(token_data_shape[3]),
        ]
        output = torch_npu.npu_ffn_to_attention(
            x=x,
            session_ids=session_ids,
            micro_batch_ids=micro_batch_ids,
            token_ids=token_ids,
            expert_offsets=expert_offsets,
            actual_token_num=actual_token_num,
            attn_rank_table=attn_rank_table,
            group=group,
            world_size=world_size,
            token_info_table_shape=token_info_table_shape_list,
            token_data_shape=token_data_shape_list,
        )
        return output
