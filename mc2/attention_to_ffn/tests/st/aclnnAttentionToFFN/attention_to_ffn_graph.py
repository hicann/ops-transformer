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
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)
    return compiled_model


class ATTENTION_TO_FFN_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        session_id,
        micro_batch_id,
        layer_id,
        expert_ids,
        expert_rank_table,
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        attn_token_info_table_shape,
        moe_expert_num,
        scales,
        active_mask,
        quant_mode,
        sync_flag,
    ):
        ffn_token_info_table_shape_list = [
            int(ffn_token_info_table_shape[0]),
            int(ffn_token_info_table_shape[1]),
            int(ffn_token_info_table_shape[2]),
        ]
        ffn_token_data_shape_list = [
            int(ffn_token_data_shape[0]),
            int(ffn_token_data_shape[1]),
            int(ffn_token_data_shape[2]),
            int(ffn_token_data_shape[3]),
            int(ffn_token_data_shape[4]),
        ]
        attn_token_info_table_shape_list = [
            int(attn_token_info_table_shape[0]),
            int(attn_token_info_table_shape[1]),
            int(attn_token_info_table_shape[2]),
        ]
        output = torch_npu.npu_attention_to_ffn(
            x=x,
            session_id=session_id,
            micro_batch_id=micro_batch_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_rank_table=expert_rank_table,
            group=group,
            world_size=world_size,
            ffn_token_info_table_shape=ffn_token_info_table_shape_list,
            ffn_token_data_shape=ffn_token_data_shape_list,
            attn_token_info_table_shape=attn_token_info_table_shape_list,
            moe_expert_num=moe_expert_num,
            scales=scales,
            active_mask=active_mask,
            quant_mode=quant_mode,
            sync_flag=sync_flag,
        )
        return output
