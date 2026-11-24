# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A3 send/receive window references and direct ACLNN target."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import torch
import torch.distributed as dist
import torch_npu
import torchair
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

SIGNATURE = """aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* sessionIds,
    const aclTensor* microBatchIds,
    const aclTensor* tokenIds,
    const aclTensor* expertOffsets,
    const aclTensor* actualTokenNum,
    const aclTensor* attnRankTable,
    const char* group,
    int64_t worldSize,
    const aclIntArray* tokenInfoTableShape,
    const aclIntArray* tokenDataShape,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);"""
INPUT_NAMES = (
    "x",
    "sessionIds",
    "microBatchIds",
    "tokenIds",
    "expertOffsets",
    "actualTokenNum",
    "attnRankTable",
    "group",
    "worldSize",
    "tokenInfoTableShape",
    "tokenDataShape",
)


@register("ffn_to_attention_reference")
class Reference(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.ep_hcomm_info = None
        self.ep_group = None
        self.compiled_model = None

    def init_by_input_data(self, input_data: InputDataset):
        # ATK encodes this optional tensor as an empty descriptor in the CSV.
        if (
            input_data.kwargs["attnRankTable"] is not None
            and input_data.kwargs["attnRankTable"].numel() == 0
        ):
            input_data.kwargs["attnRankTable"] = None
        rank_id = self.dist_task_info.rank
        world_size = input_data.kwargs["worldSize"]
        attn_rank_table = input_data.kwargs["attnRankTable"]
        micro, bs, slots, _ = input_data.kwargs["tokenDataShape"]
        attn_rank_num = (
            int(attn_rank_table.shape[0])
            if attn_rank_table is not None
            else input_data.kwargs["x"].shape[0] // (micro * bs * slots)
        )
        self.attn_rank_num = attn_rank_num
        ffn_rank_num = world_size - attn_rank_num
        if input_data.kwargs["bsList"].shape[0] >= ffn_rank_num:
            bs_list = input_data.kwargs["bsList"][:ffn_rank_num]
        else:
            bs_list = input_data.kwargs["bsList"][0].repeat(ffn_rank_num)[:ffn_rank_num]
        total_token_num = input_data.kwargs["x"].shape[0]
        tokens_per_rank = total_token_num // ffn_rank_num
        remainder = total_token_num % ffn_rank_num
        start_idx = 0
        end_idx = total_token_num + 1
        if attn_rank_num <= rank_id < world_size:
            new_rankid = rank_id - attn_rank_num
            if new_rankid < remainder:
                start_idx = (tokens_per_rank + 1) * new_rankid
                end_idx = start_idx + tokens_per_rank + 1
                bs_list[new_rankid] = tokens_per_rank + 1
                input_data.kwargs["actualTokenNum"][0] = tokens_per_rank + 1
            else:
                start_idx = (tokens_per_rank + 1) * remainder + tokens_per_rank * (
                    new_rankid - remainder
                )
                end_idx = start_idx + tokens_per_rank
                bs_list[new_rankid] = tokens_per_rank
                input_data.kwargs["actualTokenNum"][0] = tokens_per_rank
        input_data.kwargs["bsList"] = bs_list
        global_bs = total_token_num
        micro_batch_num = input_data.kwargs["tokenDataShape"][0]
        bs = input_data.kwargs["tokenDataShape"][1]
        expert_num_per_token = input_data.kwargs["tokenDataShape"][2]
        device = input_data.kwargs["x"].device
        session_repeat = int(total_token_num / attn_rank_num)
        input_data.kwargs["sessionIds"] = torch.repeat_interleave(
            torch.arange(attn_rank_num, dtype=torch.int32), repeats=session_repeat
        ).to(device)
        input_data.kwargs["tokenIds"] = (
            torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32), repeats=expert_num_per_token
            )
            .repeat(micro_batch_num * attn_rank_num)
            .to(device)
        )
        expert_repeats = int(total_token_num / expert_num_per_token)
        input_data.kwargs["expertOffsets"] = (
            torch.arange(expert_num_per_token, dtype=torch.int32)
            .repeat(expert_repeats)
            .to(device)
        )
        x_world = input_data.kwargs["x"][:global_bs]
        session_ids_world = input_data.kwargs["sessionIds"][:global_bs]
        micro_batch_ids_world = input_data.kwargs["microBatchIds"][:global_bs]
        token_ids_world = input_data.kwargs["tokenIds"][:global_bs]
        expert_offsets_world = input_data.kwargs["expertOffsets"][:global_bs]
        input_data.kwargs["x_world"] = x_world
        input_data.kwargs["sessionIds_world"] = session_ids_world
        input_data.kwargs["microBatchIds_world"] = micro_batch_ids_world
        input_data.kwargs["tokenIds_world"] = token_ids_world
        input_data.kwargs["expertOffsets_world"] = expert_offsets_world
        if attn_rank_table is not None:
            for i in range(attn_rank_num):
                input_data.kwargs["attnRankTable"][i] = i
        input_data.kwargs["x"] = x_world[start_idx:end_idx]
        input_data.kwargs["sessionIds"] = session_ids_world[start_idx:end_idx]
        input_data.kwargs["microBatchIds"] = micro_batch_ids_world[start_idx:end_idx]
        input_data.kwargs["tokenIds"] = token_ids_world[start_idx:end_idx]
        input_data.kwargs["expertOffsets"] = expert_offsets_world[start_idx:end_idx]

    def init_hccl_comm(self, rank_id, world_size):
        group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
        return (
            group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id),
            group,
        )

    def destroy_hccl_comm(self, ep_group):
        dist.destroy_process_group(ep_group)
        self.ep_group = None
        self.ep_hcomm_info = None

    def ffn2attn_get_npu(
        self,
        ep_group,
        micro_batch_num: int,
        bs: int,
        expert_num_per_token: int,
        h: int,
        model_dtype,
    ):
        window_addr = (
            ep_group._get_backend(torch.device("npu"))._get_window_mem().data_ptr()
        )
        npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
            [micro_batch_num, bs, expert_num_per_token], torch.int32, [window_addr]
        )
        attn_window_states = npu_state_tensor[0]
        attn_token_size = attn_window_states.element_size() * attn_window_states.numel()
        attn_token_area_size = (attn_token_size + 511) // 512 * 512
        token_size = h
        tensor_dtype = model_dtype
        attn_data_tensors = torchair.llm_datadist.create_npu_tensors(
            [micro_batch_num, bs, expert_num_per_token, token_size],
            tensor_dtype,
            [window_addr + attn_token_area_size],
        )
        attn_data_tensor = attn_data_tensors[0]
        return (attn_window_states.clone(), attn_data_tensor.clone())

    def ffn2attn_golden(
        self,
        attn_rank_num: int,
        micro_batch_num: int,
        bs: int,
        expert_num_per_token: int,
        h: int,
        input_data: InputDataset,
    ):
        token_info = torch.zeros(
            attn_rank_num, micro_batch_num, bs, expert_num_per_token
        )
        token_data = torch.zeros(
            attn_rank_num,
            micro_batch_num,
            bs,
            expert_num_per_token,
            h,
            dtype=input_data.kwargs["x_world"].dtype,
        )
        total_token_num = input_data.kwargs["x_world"].shape[0]
        for token_id in range(total_token_num):
            current_attention_rank = input_data.kwargs["sessionIds_world"][
                token_id
            ].cpu()
            current_micro_batch_id = input_data.kwargs["microBatchIds_world"][
                token_id
            ].cpu()
            current_batch_id = input_data.kwargs["tokenIds_world"][token_id].cpu()
            current_expert_offset = input_data.kwargs["expertOffsets_world"][
                token_id
            ].cpu()
            token_i = input_data.kwargs["x_world"][token_id].cpu()
            token_info[
                current_attention_rank,
                current_micro_batch_id,
                current_batch_id,
                current_expert_offset,
            ] = 1
            token_data[
                current_attention_rank,
                current_micro_batch_id,
                current_batch_id,
                current_expert_offset,
                :,
            ] = token_i
        return (token_info, token_data)

    def npu_benchmark(self, input_data):
        """Independent HCCL broadcast + NPU index_copy reference."""
        d = input_data.kwargs
        rank = self.dist_task_info.rank
        world = int(d["worldSize"])
        a = self.attn_rank_num
        micro, bs, slots, h = map(int, d["tokenDataShape"])
        device = torch.device("npu", torch_npu.npu.current_device())
        state = torch.zeros(micro * bs * slots, dtype=torch.int32, device=device)
        data = torch.zeros((micro * bs * slots, h), dtype=d["x"].dtype, device=device)
        for source in range(a, world):
            count = torch.tensor(
                [d["x"].shape[0] if rank == source else 0],
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(count, source)
            size = int(count.item())
            x = (
                d["x"].to(device).contiguous()
                if rank == source
                else torch.empty((size, h), dtype=d["x"].dtype, device=device)
            )
            if rank == source:
                meta = torch.stack(
                    [
                        d[key].to(device=device, dtype=torch.int32)
                        for key in (
                            "sessionIds",
                            "microBatchIds",
                            "tokenIds",
                            "expertOffsets",
                        )
                    ],
                    dim=1,
                ).contiguous()
            else:
                meta = torch.empty((size, 4), dtype=torch.int32, device=device)
            if size:
                dist.broadcast(x, source)
                dist.broadcast(meta, source)
                if rank < a:
                    dest = (
                        d["attnRankTable"]
                        .to(device=device, dtype=torch.int64)
                        .index_select(0, meta[:, 0].long())
                        if d["attnRankTable"] is not None
                        else meta[:, 0].long()
                    )
                    selected = torch.nonzero(dest == rank).reshape(-1)
                    selected_meta = meta.index_select(0, selected).long()
                    offsets = (
                        selected_meta[:, 1] * bs + selected_meta[:, 2]
                    ) * slots + selected_meta[:, 3]
                    data.index_copy_(0, offsets, x.index_select(0, selected))
                    state.index_fill_(0, offsets, 1)
        torch_npu.npu.synchronize()
        if rank < a:
            return (state.reshape(micro, bs, slots), data.reshape(micro, bs, slots, h))
        return (
            torch.zeros((1, 1, 1), device=device),
            torch.zeros((1, 1, 1, 1024), dtype=d["x"].dtype, device=device),
        )

    def __call__(self, input_data, with_output=False):
        data = input_data.kwargs
        rank = self.dist_task_info.rank
        attn_count = self.attn_rank_num
        micro, bs, slots, hidden = map(int, data["tokenDataShape"])
        if self.name == "cpu":
            state, values = self.ffn2attn_golden(
                attn_count, micro, bs, slots, hidden, input_data
            )
            if rank < attn_count:
                return (state[rank], values[rank])
            return (
                torch.zeros(1, 1, 1),
                torch.zeros(1, 1, 1, 1024, dtype=data["x"].dtype),
            )
        if self.dist_task_info.is_bm:
            return self.npu_benchmark(input_data)
        if self.ep_group is None:
            self.ep_hcomm_info, self.ep_group = self.init_hccl_comm(
                rank, data["worldSize"]
            )
        group = self.ep_group
        sender = rank >= attn_count
        peers = (
            list(range(attn_count))
            if sender
            else list(range(attn_count, data["worldSize"]))
        )
        group._get_backend(torch.device("npu"))._window_register_and_exchange(
            200 * 1024 * 1024, peers
        )
        if sender:
            if data["graphType"] == 1:
                if self.compiled_model is None:
                    from ffn_to_attention_graph import (
                        FFN_TO_ATTENTION_Model,
                        compile_model,
                    )

                    self.compiled_model = compile_model(
                        FFN_TO_ATTENTION_Model(), graph_type=1
                    )
                self.compiled_model(
                    x=data["x"].npu(),
                    session_ids=data["sessionIds"].int().npu(),
                    micro_batch_ids=data["microBatchIds"].int().npu(),
                    token_ids=data["tokenIds"].int().npu(),
                    expert_offsets=data["expertOffsets"].int().npu(),
                    actual_token_num=data["actualTokenNum"].int().npu(),
                    attn_rank_table=data["attnRankTable"].int().npu()
                    if data["attnRankTable"] is not None
                    else None,
                    group=self.ep_hcomm_info,
                    world_size=data["worldSize"],
                    token_info_table_shape=data["tokenInfoTableShape"],
                    token_data_shape=data["tokenDataShape"],
                )
            else:
                self.launch()
            result = (
                torch.zeros(1, 1, 1),
                torch.zeros(1, 1, 1, 1024, dtype=data["x"].dtype),
            )
            torch_npu.npu.synchronize()
        dist.barrier()
        if not sender:
            result = self.ffn2attn_get_npu(
                group, micro, bs, slots, hidden, data["x"].dtype
            )
            torch_npu.npu.synchronize()
        dist.barrier()
        self.destroy_hccl_comm(group)
        return result


@register("ffn_to_attention_aclnn")
class AclnnExecutor(AclnnBaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def init_by_input_data(self, input_data):
        self.reference = Reference(self.task_result)
        self.reference.name = "target"
        self.input_data = copy.copy(input_data)
        self.input_data.kwargs = dict(input_data.kwargs)
        self.reference.init_by_input_data(self.input_data)
        info = self.task_result.dist_task_info
        self.reference.ep_hcomm_info, self.reference.ep_group = (
            self.reference.init_hccl_comm(info.rank, info.world_size)
        )
        values = dict(self.input_data.kwargs)
        values["group"] = self.reference.ep_hcomm_info
        native_input = copy.copy(input_data)
        native_input.kwargs = {name: values[name] for name in INPUT_NAMES}
        args, outputs = super().init_by_input_data(native_input)
        self.reference.launch = self.launch
        return (args[: len(INPUT_NAMES)], outputs)

    def launch(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def __call__(self):
        self.result = self.reference(self.input_data)

    def after_call(self, output_packages):
        return self.result
