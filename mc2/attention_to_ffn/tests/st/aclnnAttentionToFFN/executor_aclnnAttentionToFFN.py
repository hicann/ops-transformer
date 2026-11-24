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
import random
import torch
import torch.distributed as dist
import torch_npu
import torchair
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

SIGNATURE = """aclnnStatus aclnnAttentionToFFNGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* sessionId,
    const aclTensor* microBatchId,
    const aclTensor* layerId,
    const aclTensor* expertIds,
    const aclTensor* expertRankTable,
    const aclTensor* scales,
    const aclTensor* activeMask,
    const char* group,
    int64_t worldSize,
    const aclIntArray* ffnTokenInfoTableShape,
    const aclIntArray* ffnTokenDataShape,
    const aclIntArray* attnTokenInfoTableShape,
    int64_t moeExpertNum,
    int64_t quantMode,
    int64_t syncFlag,
    int64_t ffnStartRankId,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);"""
INPUT_NAMES = (
    "x",
    "sessionId",
    "microBatchId",
    "layerId",
    "expertIds",
    "expertRankTable",
    "scales",
    "activeMask",
    "group",
    "worldSize",
    "ffnTokenInfoTableShape",
    "ffnTokenDataShape",
    "attnTokenInfoTableShape",
    "moeExpertNum",
    "quantMode",
    "syncFlag",
    "ffnStartRankId",
)


@register("attention_to_ffn_reference")
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
            input_data.kwargs["activeMask"] is not None
            and input_data.kwargs["activeMask"].numel() == 0
        ):
            input_data.kwargs["activeMask"] = None
        torch.manual_seed(0)
        random.seed(0)

        def gen_expert_rank_table(
            moe_rank_ffn_num,
            moe_expert_per_rank,
            moe_expert_num,
            shared_expert_num,
            rank_num_per_shared_expert,
            layer_num,
            max_len,
        ):
            data = []
            valid_rank_id = [i for i in range(moe_rank_ffn_num)] * moe_expert_per_rank
            valid_exp_id = list(range(moe_expert_num))
            cards = {}
            for i in range(moe_rank_ffn_num):
                start = i * moe_expert_per_rank
                end = start + moe_expert_per_rank
                cards[i] = valid_exp_id[start:end]
            for l in range(layer_num):
                layer_e1 = []
                for e in range(moe_expert_num):
                    cards_h = copy.deepcopy(cards)
                    valid_rank_id_h = copy.deepcopy(valid_rank_id)
                    i = random.randint(1, moe_rank_ffn_num)
                    item = [i]
                    for index_i in range(i):
                        num_rank_id = random.choice(valid_rank_id_h)
                        item.append(num_rank_id)
                        valid_rank_id_h.remove(num_rank_id)
                        num_exp_id = random.choice(cards_h[num_rank_id])
                        item.append(num_exp_id)
                        cards_h[num_rank_id].remove(num_exp_id)
                    layer_e1.append(item)
                for i in range(shared_expert_num):
                    item = [rank_num_per_shared_expert]
                    for j in range(rank_num_per_shared_expert):
                        item.append(
                            moe_rank_ffn_num + i * rank_num_per_shared_expert + j
                        )
                        item.append(moe_expert_num + i)
                    layer_e1.append(item)
                data.append(layer_e1)
            padded_data = []
            for layer in data:
                padded_layer = []
                for item in layer:
                    item_padded = item + [0] * (max_len - len(item))
                    padded_layer.append(item_padded)
                padded_data.append(padded_layer)
            tensor_data = torch.tensor(padded_data)
            return tensor_data

        def gen_active_mask(is_mask, attn_rank_num, bs, x):
            min_real_token_num = 0 if is_mask else 1
            active_mask_list = []
            for _ in range(attn_rank_num):
                real_token_num = (
                    torch.randint(min_real_token_num, bs + 1, (1,)).item()
                    if is_mask
                    else bs
                )
                active_mask = torch.cat(
                    (torch.ones(real_token_num), torch.zeros(bs - real_token_num))
                ).to(torch.bool)
                active_mask = active_mask.unsqueeze(0)
                active_mask = active_mask.repeat(x, 1)
                active_mask_list.append(active_mask)
            active_mask_list = torch.stack(active_mask_list)
            return active_mask_list

        rank_id = self.dist_task_info.rank
        x = input_data.kwargs["x"].shape[0]
        l = input_data.kwargs["expertRankTable"].shape[0]
        m = input_data.kwargs["expertRankTable"].shape[2]
        k = input_data.kwargs["expertIds"].shape[2]
        h = input_data.kwargs["x"].shape[2]
        world_size = input_data.kwargs["worldSize"]
        attn_rank_num = input_data.kwargs["ffnTokenInfoTableShape"][0]
        ffn_rank_num = world_size - attn_rank_num
        bs = input_data.kwargs["x"].shape[1]
        localmoe_expert = input_data.kwargs["localmoeExpert"]
        moe_expert_num = input_data.kwargs["moeExpertNum"]
        moe_rank_num = int(moe_expert_num / localmoe_expert)
        shared_rank_num = ffn_rank_num - moe_rank_num
        shared_expert_num = input_data.kwargs["sharedExpertNum"]
        rank_num_per_shared_expert = int(shared_rank_num / shared_expert_num)
        attn_id = 0
        if ffn_rank_num <= rank_id < world_size:
            attn_id = rank_id - ffn_rank_num
            input_data.kwargs["sessionId"] = torch.tensor([attn_id], dtype=torch.int32)
        input_data.kwargs["bsList"][0] = bs
        if input_data.kwargs["bsList"].shape[0] >= attn_rank_num:
            bs_list = input_data.kwargs["bsList"][:attn_rank_num]
        else:
            bs_list = input_data.kwargs["bsList"][0].repeat(attn_rank_num)[
                :attn_rank_num
            ]
        input_data.kwargs["bsList"] = bs_list
        x_world = input_data.kwargs["x_world"].to(input_data.kwargs["x"].dtype)
        ep_expert_ids = [
            torch.randperm(moe_expert_num, dtype=torch.int32)[:k]
            for _ in range(bs * attn_rank_num)
        ]
        expert_ids_world = torch.cat(ep_expert_ids).view([attn_rank_num, 1, bs, k])
        quant_mode = input_data.kwargs["quantMode"]
        is_attn2ffn_scales = input_data.kwargs["is_attn2ffn_scales"]
        is_quant = quant_mode > 0
        if is_quant and is_attn2ffn_scales:
            input_data.kwargs["scales"] = torch.empty(
                size=[l, shared_expert_num + moe_expert_num, h], dtype=torch.float32
            ).uniform_(-1, 1)
        else:
            input_data.kwargs["scales"] = None
        is_mask = input_data.kwargs["is_mask"]
        active_mask_optional_world = gen_active_mask(is_mask, attn_rank_num, bs, x)
        input_data.kwargs["x_world"] = x_world
        input_data.kwargs["expertIds_world"] = expert_ids_world
        input_data.kwargs["activeMaskOptional_world"] = active_mask_optional_world
        input_data.kwargs["x"] = x_world[attn_id]
        input_data.kwargs["expertIds"] = expert_ids_world[attn_id]
        input_data.kwargs["activeMask"] = (
            active_mask_optional_world[attn_id]
            if input_data.kwargs["activeMask"] is not None
            else None
        )
        input_data.kwargs["expertRankTable"] = gen_expert_rank_table(
            moe_rank_num,
            localmoe_expert,
            moe_expert_num,
            shared_expert_num,
            rank_num_per_shared_expert,
            l,
            m,
        )

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

    def attn2ffn_get_npu(
        self,
        ep_group,
        attn_rank_num: int,
        micro_batch_num: int,
        bs: int,
        expert_num_per_token: int,
        token_size: int,
        h: int,
        model_dtype,
    ):
        window_addr = (
            ep_group._get_backend(torch.device("npu"))._get_window_mem().data_ptr()
        )
        npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
            [attn_rank_num, micro_batch_num, 1 + 1 + bs * expert_num_per_token],
            torch.int32,
            [window_addr],
        )
        ffn_window_states = npu_state_tensor[0]
        ffn_token_size = ffn_window_states.element_size() * ffn_window_states.numel()
        ffn_token_area_size = (ffn_token_size + 511) // 512 * 512
        ffn_data_tensors = torchair.llm_datadist.create_npu_tensors(
            [attn_rank_num, micro_batch_num, bs, expert_num_per_token, token_size],
            model_dtype,
            [window_addr + ffn_token_area_size],
        )
        ffn_data_tensor = ffn_data_tensors[0]
        ffn_data_tensor = ffn_data_tensor[:, :, :, :, :h]
        return (ffn_window_states.clone(), ffn_data_tensor.clone())

    def attn2ffn_golden(
        self,
        ffn_rank_num: int,
        attn_rank_num: int,
        micro_batch_num: int,
        bs: int,
        expert_num_per_token: int,
        input_data: InputDataset,
    ):
        torch.manual_seed(0)
        random.seed(0)
        x = input_data.kwargs["x"].shape[0]
        k = input_data.kwargs["expertIds"].shape[2]
        h = input_data.kwargs["x"].shape[2]
        token_size = h
        quant_mode = input_data.kwargs["quantMode"]
        is_attn2ffn_scales = input_data.kwargs["is_attn2ffn_scales"]
        is_quant = quant_mode > 0
        shared_expert_num = input_data.kwargs["sharedExpertNum"]
        moe_expert_num = input_data.kwargs["moeExpertNum"]
        ffn_token_state = torch.zeros(
            ffn_rank_num,
            attn_rank_num,
            micro_batch_num,
            1 + 1 + bs * expert_num_per_token,
        )
        ffn_token_data = torch.zeros(
            ffn_rank_num,
            attn_rank_num,
            micro_batch_num,
            bs,
            expert_num_per_token,
            token_size,
            dtype=torch.int8 if is_quant else input_data.kwargs["x"].dtype,
        )
        token_data_mask = torch.zeros(
            ffn_rank_num,
            attn_rank_num,
            micro_batch_num,
            bs,
            expert_num_per_token,
            token_size,
            dtype=torch.bool,
        )
        attn_token_state = torch.zeros(
            attn_rank_num, micro_batch_num, bs, expert_num_per_token
        )
        x_world = input_data.kwargs["x_world"]
        expert_ids_world = input_data.kwargs["expertIds_world"]
        scales_optional = input_data.kwargs["scales"]
        active_mask_optional_world = input_data.kwargs["activeMaskOptional_world"]
        expert_rank_table = input_data.kwargs["expertRankTable"]
        sync_flag = input_data.kwargs["syncFlag"]

        def load_balance(session_id, layer_id, expert_id):
            count = expert_rank_table[layer_id, expert_id, 0]
            idx = session_id % count
            dst_rank_id = expert_rank_table[layer_id, expert_id, 1 + idx * 2]
            dst_exp_id = expert_rank_table[layer_id, expert_id, 2 + idx * 2]
            return (dst_rank_id, dst_exp_id)

        def quant_process(y, scale):
            y = y.to(torch.float32)
            if scale is not None:
                y = y * scale
            y_max, _ = y.abs().max(dim=-1)
            y_max[y_max == 0] = 1.0
            dynamic_scales = torch.tensor(127.0, dtype=torch.float32) / y_max
            y = (y * dynamic_scales.view(-1, 1)).round().to(torch.int8)
            return y

        def att2ffn(row_idx, x_i, expert_id, attn_id):
            scales_i = None
            default_layer_id = 0
            default_micro_batch_id = 0
            for topk_id, expert_i in enumerate(expert_id):
                rank_id, exp_id = load_balance(attn_id, default_layer_id, expert_i)
                if is_quant and is_attn2ffn_scales:
                    scales_i = scales_optional[default_layer_id, expert_i]
                ffn_token_data[
                    rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h
                ] = (
                    x_i
                    if not is_quant
                    else quant_process(
                        x_i.to("cpu"),
                        scales_i.to("cpu") if scales_i is not None else None,
                    )
                )
                token_data_mask[
                    rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h
                ] = torch.ones(h, dtype=torch.bool)
                ffn_token_state[rank_id, attn_id, default_micro_batch_id, 0] = 1
                ffn_token_state[rank_id, attn_id, default_micro_batch_id, 1] = (
                    default_layer_id
                )
                ffn_token_state[
                    rank_id,
                    attn_id,
                    default_micro_batch_id,
                    row_idx * expert_num_per_token + topk_id + 2,
                ] = exp_id.float()
            for i in range(shared_expert_num):
                shared_rank_id, shared_exp_id = load_balance(
                    attn_id, default_layer_id, moe_expert_num + i
                )
                ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, 0] = 1
                ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, 1] = (
                    default_layer_id
                )
                ffn_token_state[
                    shared_rank_id,
                    attn_id,
                    default_micro_batch_id,
                    row_idx * expert_num_per_token + k + i + 2,
                ] = shared_exp_id.float()
                if is_quant and is_attn2ffn_scales:
                    scales_i = scales_optional[default_layer_id, moe_expert_num + i]
                ffn_token_data[
                    shared_rank_id, attn_id, default_micro_batch_id, row_idx, k + i, :h
                ] = (
                    x_i
                    if not is_quant
                    else quant_process(
                        x_i.to("cpu"),
                        scales_i.to("cpu") if scales_i is not None else None,
                    )
                )
                token_data_mask[
                    shared_rank_id, attn_id, default_micro_batch_id, row_idx, k + i, :h
                ] = torch.ones(h, dtype=torch.bool)

        for attn_id in range(attn_rank_num):
            for token_id in range(bs):
                if active_mask_optional_world[attn_id, x - 1, token_id]:
                    att2ffn(
                        token_id,
                        x_world[attn_id, x - 1, token_id],
                        expert_ids_world[attn_id, x - 1, token_id],
                        attn_id,
                    )
                else:
                    attn_token_state[attn_id, x - 1, token_id, :] = torch.ones(
                        expert_num_per_token
                    )
        if sync_flag == 0:
            ffn_token_state[:, :, :, 0] = 1
        return (
            ffn_token_data,
            token_data_mask,
            ffn_token_state.to(torch.int32),
            attn_token_state.to(torch.int32),
        )

    def npu_benchmark(self, input_data):
        """Independent NPU broadcast + gather/quantize/index_copy reference."""
        d = input_data.kwargs
        rank = self.dist_task_info.rank
        world = int(d["worldSize"])
        a, micro, bs, slots, _ = map(int, d["ffnTokenDataShape"])
        f = world - a
        h = d["x"].shape[-1]
        k = d["expertIds"].shape[-1]
        device = torch.device("npu", torch_npu.npu.current_device())
        dtype = torch.int8 if d["quantMode"] == 2 else d["x"].dtype
        state = torch.zeros(
            (a, micro, 2 + bs * slots), dtype=torch.int32, device=device
        )
        data = torch.zeros(
            (a * micro * bs * slots, h),
            dtype=torch.int32 if d["quantMode"] == 2 else dtype,
            device=device,
        )
        mask_self = None
        table = d["expertRankTable"].cpu()
        scales = d["scales"]
        if scales is not None:
            scales = scales.to(device=device, dtype=torch.float32)
        for source in range(f, world):
            attn = source - f
            x = (
                d["x"].to(device).contiguous()
                if rank == source
                else torch.empty((1, bs, h), dtype=d["x"].dtype, device=device)
            )
            ids = (
                d["expertIds"].to(device=device, dtype=torch.int32).contiguous()
                if rank == source
                else torch.empty((1, bs, k), dtype=torch.int32, device=device)
            )
            mask = (
                d["activeMaskOptional_world"][source - f]
                .to(device=device, dtype=torch.int32)
                .contiguous()
                if rank == source
                else torch.empty((1, bs), dtype=torch.int32, device=device)
            )
            dist.broadcast(x, source)
            dist.broadcast(ids, source)
            dist.broadcast(mask, source)
            if rank == source:
                mask_self = mask
            if rank >= f:
                continue
            ids_cpu, active = (ids.cpu(), mask.cpu())
            rows, positions, experts, destinations = ([], [], [], [])
            for token in range(bs):
                if not active[0, token]:
                    continue
                selected = ids_cpu[0, token].tolist() + list(
                    range(
                        int(d["moeExpertNum"]),
                        int(d["moeExpertNum"]) + int(d["sharedExpertNum"]),
                    )
                )
                for slot, expert in enumerate(selected):
                    entry = table[0, expert]
                    replica = attn % int(entry[0])
                    if int(entry[1 + 2 * replica]) == rank:
                        rows.append(token)
                        positions.append(token * slots + slot)
                        experts.append(expert)
                        destinations.append(int(entry[2 + 2 * replica]))
            if rows:
                row_idx = torch.tensor(rows, dtype=torch.int64, device=device)
                pos_idx = torch.tensor(positions, dtype=torch.int64, device=device)
                values = x[0].index_select(0, row_idx)
                if d["quantMode"] == 2:
                    values = values.float()
                    if scales is not None:
                        exp_idx = torch.tensor(
                            experts, dtype=torch.int64, device=device
                        )
                        values = values * scales[0].index_select(0, exp_idx)
                    peak = values.abs().amax(dim=-1, keepdim=True)
                    peak = torch.where(peak == 0, torch.ones_like(peak), peak)
                    values = (values * (127.0 / peak)).round().to(torch.int32)
                data.index_copy_(0, pos_idx + attn * micro * bs * slots, values)
                state[attn, 0, 2:].index_copy_(
                    0,
                    pos_idx,
                    torch.tensor(destinations, dtype=torch.int32, device=device),
                )
                state[attn, 0, 0] = 1
            if int(d["syncFlag"]) == 0:
                state[attn, :, 0] = 1
        torch_npu.npu.synchronize()
        if rank < f:
            return (state, data.reshape(a, micro, bs, slots, h).to(dtype))
        attn_state = (
            (1 - mask_self).reshape(1, bs, 1).expand(micro, bs, slots).contiguous()
        )
        return (attn_state, torch.zeros((micro, bs, slots), device=device))

    def __call__(self, input_data, with_output=False):
        data = input_data.kwargs
        rank = self.dist_task_info.rank
        attn_count, micro, bs, slots, token_size = map(int, data["ffnTokenDataShape"])
        ffn_count = data["worldSize"] - attn_count
        if self.name == "cpu":
            values, mask, ffn_state, attn_state = self.attn2ffn_golden(
                ffn_count, attn_count, micro, bs, slots, input_data
            )
            if rank < ffn_count:
                return (ffn_state[rank], values[rank])
            return (attn_state[rank - ffn_count], torch.zeros(micro, bs, slots))
        if self.dist_task_info.is_bm:
            return self.npu_benchmark(input_data)
        if self.ep_group is None:
            self.ep_hcomm_info, self.ep_group = self.init_hccl_comm(
                rank, data["worldSize"]
            )
        group = self.ep_group
        sender = rank >= ffn_count
        peers = (
            list(range(ffn_count))
            if sender
            else list(range(ffn_count, data["worldSize"]))
        )
        backend = group._get_backend(torch.device("npu"))
        backend._window_register_and_exchange(200 * 1024 * 1024, peers)
        if sender:
            if data["graphType"] == 1:
                if self.compiled_model is None:
                    from attention_to_ffn_graph import (
                        ATTENTION_TO_FFN_Model,
                        compile_model,
                    )

                    self.compiled_model = compile_model(
                        ATTENTION_TO_FFN_Model(), graph_type=1
                    )
                self.compiled_model(
                    x=data["x"].npu(),
                    session_id=data["sessionId"].int().npu(),
                    micro_batch_id=data["microBatchId"].int().npu(),
                    layer_id=data["layerId"].int().npu(),
                    expert_ids=data["expertIds"].int().npu(),
                    expert_rank_table=data["expertRankTable"].int().npu(),
                    group=self.ep_hcomm_info,
                    world_size=data["worldSize"],
                    ffn_token_info_table_shape=data["ffnTokenInfoTableShape"],
                    ffn_token_data_shape=data["ffnTokenDataShape"],
                    attn_token_info_table_shape=data["attnTokenInfoTableShape"],
                    moe_expert_num=data["moeExpertNum"],
                    scales=data["scales"].float().npu()
                    if data["scales"] is not None
                    else None,
                    active_mask=data["activeMask"].bool().npu()
                    if data["activeMask"] is not None
                    else None,
                    quant_mode=data["quantMode"],
                    sync_flag=data["syncFlag"],
                )
            else:
                self.launch()
            state = torch.zeros(micro, bs, slots, dtype=torch.int32)
            if data["is_mask"]:
                address = backend._get_window_mem().data_ptr()
                state = torchair.llm_datadist.create_npu_tensors(
                    [micro, bs, slots], torch.int32, [address]
                )[0].clone()
            result = (state, torch.zeros(micro, bs, slots))
            torch_npu.npu.synchronize()
        dist.barrier()
        if not sender:
            dtype = torch.int8 if data["quantMode"] else data["x"].dtype
            result = self.attn2ffn_get_npu(
                group,
                attn_count,
                micro,
                bs,
                slots,
                token_size,
                data["x"].shape[-1],
                dtype,
            )
            torch_npu.npu.synchronize()
        dist.barrier()
        self.destroy_hccl_comm(group)
        return result


@register("attention_to_ffn_aclnn")
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
