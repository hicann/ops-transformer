# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mega KDA 的递推、记录发布、门控和输出投影阶段。"""

from __future__ import annotations

from cannbotdsl import MemLoc, PIPE, Tensor, dtypes, from_tensor, make_layout
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vexp,
    vload,
    vload_broadcast,
    vload_unpack,
    vmadd,
    vmul,
    vmuls,
    vneg,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    global_sync_all,
    onchip_rewind,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import ceil_div, idx2crd, tile_slice
from typing import NamedTuple

if __package__:
    from .mega_recurrent_kda_common import (
        BETA_TILE,
        HIDDEN_SIZE,
        OPROJ_KC,
        OPROJ_KS,
        PROJECTION_KC,
        PROJECTION_KS,
        ReplayPool,
        S4_SEQUENCE_LENGTH,
        S8_SEQUENCE_LENGTH,
        Schedule,
        TILE,
        TP16_HEADS,
        TP4_HEADS,
        TP8_HEADS,
        VL,
        _ProjectionOperations,
        _checkpoint_head,
        _copy_weight_with_policy,
        replay_head,
    )
else:
    from mega_recurrent_kda_common import (
        BETA_TILE,
        HIDDEN_SIZE,
        OPROJ_KC,
        OPROJ_KS,
        PROJECTION_KC,
        PROJECTION_KS,
        ReplayPool,
        S4_SEQUENCE_LENGTH,
        S8_SEQUENCE_LENGTH,
        Schedule,
        TILE,
        TP16_HEADS,
        TP4_HEADS,
        TP8_HEADS,
        VL,
        _ProjectionOperations,
        _checkpoint_head,
        _copy_weight_with_policy,
        replay_head,
    )


# 递推展开、容量及预取。
D_SEGMENTS = TILE // VL
KDA_STATE_KEY_UNROLL = 4
KDA_STATE_UPDATE_UNROLL = 2
RECURRENT_HEAD_CAPACITY = 24
S8_CHECKPOINT_PREFETCH_TOKEN = S8_SEQUENCE_LENGTH - 2


@jit
def _recurrent_item_index(item, batch_size: int, num_heads: int, nb_group_batch: int):
    """在每个完整的 OProj 发布块内，先按 head、再按 batch 排列任务。

    最后一个块使用实际 batch 数，填充任务不会读取或覆盖其他请求的状态。
    发布计数仍采用任务的顺序编号。
    """
    group_tasks = nb_group_batch * num_heads
    group_start = item // group_tasks * nb_group_batch
    group_batch = batch_size - group_start
    if group_batch > nb_group_batch:
        group_batch = nb_group_batch
    offset = item % group_tasks
    batch = group_start + offset % group_batch
    head = offset // group_batch
    return batch * num_heads + head


@jit
def prefetch_recurrent_item(
    recurrent_stage,
    output_gate_stage,
    q_workspace_gm,
    k_workspace_gm,
    v_workspace_gm,
    decay_workspace_gm,
    output_gate_workspace_gm,
    beta_workspace_gm,
    item,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    row_block: int,
    nb_group_batch: int = 1,
):
    """为一个完整的 batch/head 任务发起 token 0 的输入读取。"""
    batch_index, value_head = idx2crd(
        _recurrent_item_index(item, batch_size, num_heads, nb_group_batch),
        [batch_size, num_heads],
    )
    base = batch_index * seq_len
    recurrent_stage.prefetch_inputs(
        tile_slice(k_workspace_gm, (1, TILE), (base, value_head)),
        tile_slice(q_workspace_gm, (1, TILE), (base, value_head)),
        tile_slice(decay_workspace_gm, (1, TILE), (base, value_head)),
        tile_slice(v_workspace_gm, (1, row_block), (base, value_head)),
        tile_slice(beta_workspace_gm, (1, 1), (base, value_head)),
    )
    output_gate_stage.issue_gate(
        tile_slice(output_gate_workspace_gm, (1, row_block), (base, value_head))
    )


@jit
def prefetch_checkpoint_state(
    recurrent_stage,
    initial_state_gm,
    replay_k_gm,
    replay_decay_gm,
    item,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    row_block: int,
    nb_group_batch: int = 1,
):
    """读取一个 batch/head 任务的只读检查点。"""
    batch_index, value_head = idx2crd(
        _recurrent_item_index(item, batch_size, num_heads, nb_group_batch),
        [batch_size, num_heads],
    )
    recurrent_stage.load_state(
        tile_slice(
            _checkpoint_head(initial_state_gm, batch_index, value_head),
            (row_block, TILE),
            (0, 0),
        )
    )


@jit
def run_recurrent_wave(
    recurrent_stage,
    output_gate_stage,
    q_workspace_gm,
    k_workspace_gm,
    v_workspace_gm,
    decay_workspace_gm,
    output_gate_workspace_gm,
    beta_workspace_gm,
    initial_state_gm,
    replay_u_gm,
    replay_k_gm,
    replay_decay_gm,
    item_start,
    item_end,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    row_block: int,
    scale_value: float,
    lower_bound: float,
    rms_norm_eps: float,
    defer_final_snapshot: bool = False,
    first_inputs_prefetched: bool = False,
    first_state_prefetched: bool = False,
    next_item_stride: int = 0,
    nb_group_batch: int = 1,
    checkpoint_prefetched: bool = False,
):
    """按 NB 顺序执行任务，并对 token 和状态预取沿用同一顺序。"""
    for item in range(item_start, item_end):
        batch_index, value_head = idx2crd(
            _recurrent_item_index(item, batch_size, num_heads, nb_group_batch),
            [batch_size, num_heads],
        )
        base = batch_index * seq_len
        if not const_expr(first_inputs_prefetched):
            prefetch_recurrent_item(
                recurrent_stage,
                output_gate_stage,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                beta_workspace_gm,
                item,
                batch_size,
                seq_len,
                num_heads,
                row_block,
                nb_group_batch=nb_group_batch,
            )
        if const_expr(checkpoint_prefetched):
            recurrent_stage.state = recurrent_stage.state_ring.next_read()
        elif not const_expr(first_state_prefetched):
            recurrent_stage.load_state(
                tile_slice(
                    _checkpoint_head(initial_state_gm, batch_index, value_head),
                    (row_block, TILE),
                    (0, 0),
                )
            )
        for token in range(seq_len):
            recurrent_stage.normalize_qk(scale_value)
            recurrent_stage.publish_replay_k(
                tile_slice(
                    replay_head(replay_k_gm, batch_index, value_head),
                    (1, TILE),
                    (token, 0),
                )
            )
            recurrent_stage.activate_g_beta(lower_bound, value_head)
            recurrent_stage.publish_replay_decay(
                tile_slice(
                    replay_head(replay_decay_gm, batch_index, value_head),
                    (1, TILE),
                    (token, 0),
                )
            )
            if token + 1 < seq_len:
                next_row = base + token + 1
                recurrent_stage.prefetch_inputs(
                    tile_slice(k_workspace_gm, (1, TILE), (next_row, value_head)),
                    tile_slice(q_workspace_gm, (1, TILE), (next_row, value_head)),
                    tile_slice(decay_workspace_gm, (1, TILE), (next_row, value_head)),
                    tile_slice(v_workspace_gm, (1, row_block), (next_row, value_head)),
                    tile_slice(beta_workspace_gm, (1, 1), (next_row, value_head)),
                )
                output_gate_stage.issue_gate(
                    tile_slice(
                        output_gate_workspace_gm, (1, row_block), (next_row, value_head)
                    )
                )
            if const_expr(checkpoint_prefetched):
                if token == seq_len - 2 and next_item_stride > 0:
                    next_item = item + next_item_stride
                    if next_item < batch_size * num_heads:
                        next_batch, next_head = idx2crd(
                            _recurrent_item_index(
                                next_item, batch_size, num_heads, nb_group_batch
                            ),
                            [batch_size, num_heads],
                        )
                        next_state = recurrent_stage.state_ring.next_write()
                        mem_copy(
                            next_state,
                            tile_slice(
                                _checkpoint_head(
                                    initial_state_gm, next_batch, next_head
                                ),
                                (row_block, TILE),
                                (0, 0),
                            ),
                            options=copy_options(
                                l2_cache_ctl=recurrent_stage.load_policy
                            ),
                        )
            recurrent_stage.compute_delta()
            recurrent_stage.publish_replay_u(
                tile_slice(
                    replay_head(replay_u_gm, batch_index, value_head),
                    (1, row_block),
                    (token, 0),
                )
            )
            recurrent_stage.update_state()
            output_gate_stage.apply_and_store(
                tile_slice(q_workspace_gm, (1, row_block), (base + token, value_head)),
                rms_norm_eps,
                recurrent_stage.out,
            )
            if next_item_stride > 0:
                if token + 1 == seq_len:
                    next_item = item + next_item_stride
                    if next_item < batch_size * num_heads:
                        prefetch_recurrent_item(
                            recurrent_stage,
                            output_gate_stage,
                            q_workspace_gm,
                            k_workspace_gm,
                            v_workspace_gm,
                            decay_workspace_gm,
                            output_gate_workspace_gm,
                            beta_workspace_gm,
                            next_item,
                            batch_size,
                            seq_len,
                            num_heads,
                            row_block,
                            nb_group_batch=nb_group_batch,
                        )
                        if not const_expr(checkpoint_prefetched):
                            prefetch_checkpoint_state(
                                recurrent_stage,
                                initial_state_gm,
                                replay_k_gm,
                                replay_decay_gm,
                                next_item,
                                batch_size,
                                seq_len,
                                num_heads,
                                row_block,
                                nb_group_batch=nb_group_batch,
                            )


def publish_vector_grid_to_cube(vector_flag: int, cube_flag: int, ack_flag: int):
    """复用 AIV 标志前先确认槽位接收完成，然后并行执行投影。"""
    vec_sync_block_arrive(PIPE.MTE3, vector_flag, mode=2)
    cube_sync_block_wait(PIPE.S, vector_flag, mode=2)
    cube_sync_block_arrive(PIPE.MTE3, ack_flag, mode=2)
    vec_sync_block_wait(PIPE.S, ack_flag, mode=2)
    cube_sync_block_arrive(PIPE.FIXPIPE, cube_flag, mode=0)
    cube_sync_block_wait(PIPE.S, cube_flag, mode=0)


class _RecurrentMath:
    """使用调度提供的状态、向量和记录缓冲区执行递推算术。"""

    @jit
    def normalize_qk(self, scale_value: float):
        raw_query = self.raw_query.next_read()
        raw_key = self.raw_key.next_read()
        replay_k = self.replay_k.next_write()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            query_pre_bf16 = vload_unpack(
                raw_query, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            query_pre = vcast(query_pre_bf16, dtypes.float32, mask=full)
            query_post_bf16 = vload_unpack(
                raw_query, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            query_post = vcast(query_post_bf16, dtypes.float32, mask=full)
            key_pre_bf16 = vload_unpack(raw_key, 0, unpack_mode=UnpackMode.B16_TO_B32)
            key_pre = vcast(key_pre_bf16, dtypes.float32, mask=full)
            key_post_bf16 = vload_unpack(raw_key, VL, unpack_mode=UnpackMode.B16_TO_B32)
            key_post = vcast(key_post_bf16, dtypes.float32, mask=full)
            query_pre_square = vmul(query_pre, query_pre, mask=full)
            query_pre_sum = vreduce_sum(query_pre_square, mask=full)
            query_post_square = vmul(query_post, query_post, mask=full)
            query_post_sum = vreduce_sum(query_post_square, mask=full)
            key_pre_square = vmul(key_pre, key_pre, mask=full)
            key_pre_sum = vreduce_sum(key_pre_square, mask=full)
            key_post_square = vmul(key_post, key_post, mask=full)
            key_post_sum = vreduce_sum(key_post_square, mask=full)
            lane0, _ = update_mask(1, elem_bits=32)
            query_sum = vadd(query_pre_sum, query_post_sum, mask=lane0)
            key_sum = vadd(key_pre_sum, key_post_sum, mask=lane0)
            query_sum_eps = vadds(query_sum, 1e-6, mask=lane0)
            query_norm = vsqrt(query_sum_eps, mask=lane0)
            key_sum_eps = vadds(key_sum, 1e-6, mask=lane0)
            key_norm = vsqrt(key_sum_eps, mask=lane0)
            full, _ = update_mask(VL, elem_bits=32)
            query_denom = vdup(query_norm, mask=full)
            key_denom = vdup(key_norm, mask=full)
            query_pre = vdiv(query_pre, query_denom, mask=full)
            query_pre = vmuls(query_pre, scale_value, mask=full)
            vstore(self.query, 0, query_pre, full)
            query_post = vdiv(query_post, query_denom, mask=full)
            query_post = vmuls(query_post, scale_value, mask=full)
            vstore(self.query, VL, query_post, full)
            key_pre = vdiv(key_pre, key_denom, mask=full)
            vstore(self.key, 0, key_pre, full)
            key_pre_fp16 = vcast(key_pre, dtypes.float16, mask=full)
            vstore_pack(replay_k, 0, key_pre_fp16, full, pack_mode=PackMode.B32_TO_B16)
            key_post = vdiv(key_post, key_denom, mask=full)
            vstore(self.key, VL, key_post, full)
            key_post_fp16 = vcast(key_post, dtypes.float16, mask=full)
            vstore_pack(
                replay_k, VL, key_post_fp16, full, pack_mode=PackMode.B32_TO_B16
            )

    @jit
    def activate_g_beta(self, lower_bound: float, value_head):
        raw_g = self.raw_g.next_read()
        raw_beta = self.raw_beta.next_read()
        value_slot = self.value.next_read()
        replay_decay = self.replay_decay.next_write()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32)
            alpha = vload_broadcast(self.a_log, value_head)
            alpha = vexp(alpha, mask=full)
            negative_alpha = vneg(alpha, mask=full)
            gate_offset = value_head * TILE
            raw_gate_pre_bf16 = vload_unpack(
                raw_g, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            raw_gate_pre = vcast(raw_gate_pre_bf16, dtypes.float32, mask=full)
            raw_gate_post_bf16 = vload_unpack(
                raw_g, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            raw_gate_post = vcast(raw_gate_post_bf16, dtypes.float32, mask=full)
            gate_bias_pre = vload(self.dt_bias, gate_offset)
            gate_pre = vadd(raw_gate_pre, gate_bias_pre, mask=full)
            gate_pre = vmul(negative_alpha, gate_pre, mask=full)
            gate_bias_post = vload(self.dt_bias, gate_offset + VL)
            gate_post = vadd(raw_gate_post, gate_bias_post, mask=full)
            gate_post = vmul(negative_alpha, gate_post, mask=full)
            activated_pre = vexp(gate_pre, mask=full)
            activated_pre = vadds(activated_pre, 1.0, mask=full)
            activated_pre = vdiv(one, activated_pre, mask=full)
            activated_pre = vmuls(activated_pre, lower_bound, mask=full)
            activated_post = vexp(gate_post, mask=full)
            activated_post = vadds(activated_post, 1.0, mask=full)
            activated_post = vdiv(one, activated_post, mask=full)
            activated_post = vmuls(activated_post, lower_bound, mask=full)
            decay_exp_pre = vexp(activated_pre, mask=full)
            vstore(self.decay_exp, 0, decay_exp_pre, full)
            vstore(replay_decay, 0, decay_exp_pre, full)
            decay_exp_post = vexp(activated_post, mask=full)
            vstore(self.decay_exp, VL, decay_exp_post, full)
            vstore(replay_decay, VL, decay_exp_post, full)

            beta_logit_bf16 = vload_unpack(
                raw_beta, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            beta_logit = vcast(beta_logit_bf16, dtypes.float32, mask=full)
            beta = vneg(beta_logit, mask=full)
            beta = vexp(beta, mask=full)
            beta = vadds(beta, 1.0, mask=full)
            beta = vdiv(one, beta, mask=full)
            vstore(self.ub_beta, 0, beta, full)
            value_bf16 = vload_unpack(value_slot, 0, unpack_mode=UnpackMode.B16_TO_B32)
            value = vcast(value_bf16, dtypes.float32, mask=full)
            vstore(self.ub_value, 0, value, full)
            value_post_bf16 = vload_unpack(
                value_slot, VL, unpack_mode=UnpackMode.B16_TO_B32
            )
            value_post = vcast(value_post_bf16, dtypes.float32, mask=full)
            vstore(self.ub_value, VL, value_post, full)

    @jit
    def compute_delta(self):
        replay_u = self.replay_u.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            decay_key_pre = vmul(decay_pre, key_pre, mask=mask)
            decay_key_post = vmul(decay_post, key_post, mask=mask)
            for dv in dsl_range(0, self.row_block, 1, unroll=self.state_key_unroll):
                offset = dv * TILE
                state_key_pre = vload(self.state, offset)
                state_key_post = vload(self.state, offset + VL)
                state_key_post = vmul(state_key_post, decay_key_post, mask=mask)
                state_key = vmadd(
                    state_key_pre, decay_key_pre, state_key_post, mask=mask
                )
                state_key_sum = vreduce_sum(state_key, mask=mask)
                vstore_first(self.state_key_sums, dv, state_key_sum)
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            beta = vload_broadcast(self.ub_beta, 0)
            value_pre = vload(self.ub_value, 0)
            state_key_sum_pre = vload(self.state_key_sums, 0)
            residual_pre = vsub(value_pre, state_key_sum_pre, mask=mask)
            delta_pre = vmul(residual_pre, beta, mask=mask)
            vstore(self.delta_row, 0, delta_pre, mask)
            vstore(replay_u, 0, delta_pre, mask)
            value_post = vload(self.ub_value, VL)
            state_key_sum_post = vload(self.state_key_sums, VL)
            residual_post = vsub(value_post, state_key_sum_post, mask=mask)
            delta_post = vmul(residual_post, beta, mask=mask)
            vstore(self.delta_row, VL, delta_post, mask)
            vstore(replay_u, VL, delta_post, mask)

    @jit
    def update_state(self):
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            decay_pre = vload(self.decay_exp, 0)
            decay_post = vload(self.decay_exp, VL)
            key_pre = vload(self.key, 0)
            key_post = vload(self.key, VL)
            query_pre = vload(self.query, 0)
            query_post = vload(self.query, VL)
            for dv in dsl_range(0, self.row_block, 1, unroll=self.state_update_unroll):
                offset = dv * TILE
                delta = vload_broadcast(self.delta_row, dv)
                state_pre = vload(self.state, offset)
                update_pre = vmul(delta, key_pre, mask=mask)
                state_pre = vmadd(state_pre, decay_pre, update_pre, mask=mask)
                state_post = vload(self.state, offset + VL)
                update_post = vmul(delta, key_post, mask=mask)
                state_post = vmadd(state_post, decay_post, update_post, mask=mask)
                vstore(self.state, offset, state_pre, mask)
                vstore(self.state, offset + VL, state_post, mask)
                output_post = vmul(state_post, query_post, mask=mask)
                output = vmadd(state_pre, query_pre, output_post, mask=mask)
                output = vreduce_sum(output, mask=mask)
                vstore_first(self.out, dv, output)


class RecurrentKDAStage(_RecurrentMath):
    """为一个完整的 value head 执行带回放记录的 delta 规则递推。"""

    def __init__(
        self,
        row_block: int,
        num_heads: int,
        state_key_unroll: int = KDA_STATE_KEY_UNROLL,
        state_update_unroll: int = KDA_STATE_UPDATE_UNROLL,
        state_buffer_depth: int = 1,
    ):
        self.row_block = row_block
        self.num_heads = int(num_heads)
        self.state_key_unroll = int(state_key_unroll)
        self.state_update_unroll = int(state_update_unroll)
        self.state_ring = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.row_block, TILE)),
            depth=state_buffer_depth,
        )
        self.state = self.state_ring.next_write()
        self.key = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)))
        self.query = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)))
        self.decay_exp = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, TILE)))
        self.ub_value = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.row_block))
        )
        self.ub_beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VL)))
        self.out = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, self.row_block)))
        self.value = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, self.row_block)), depth=2
        )
        self.state_key_sums = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.row_block))
        )
        self.delta_row = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.row_block))
        )
        # PA 状态接口以 FP32 输出修正量 U，以 FP16 输出归一化 K。
        # 本阶段的递推计算和寄存器中保留的中间状态仍使用 FP32。
        self.replay_u = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, self.row_block)), depth=1
        )
        self.replay_k = RingBuffer(
            MemLoc.UB, dtypes.float16, make_layout((1, TILE)), depth=1
        )
        self.replay_decay = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, TILE)), depth=1
        )
        self.raw_query = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, TILE)), depth=2
        )
        self.raw_key = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, TILE)), depth=2
        )
        self.raw_g = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, TILE)), depth=2
        )
        self.raw_beta = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, VL)), depth=2
        )
        self.dt_bias = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.num_heads, TILE))
        )
        self.a_log = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.num_heads,)))

    def load_gate_params(self, a_log_gm, dt_bias_gm):
        mem_copy(self.a_log, a_log_gm)
        mem_copy(self.dt_bias, dt_bias_gm)

    def load_state(self, state_gm):
        mem_copy(self.state, state_gm, options=copy_options(l2_cache_ctl=2))

    def prefetch_inputs(
        self, raw_key_gm, raw_query_gm, raw_g_gm, value_gm, raw_beta_gm
    ):
        mem_copy(self.raw_query.next_write(), raw_query_gm)
        mem_copy(self.raw_key.next_write(), raw_key_gm)
        mem_copy(self.raw_g.next_write(), raw_g_gm)
        mem_copy(tile_slice(self.raw_beta.next_write(), (1, 1), (0, 0)), raw_beta_gm)
        mem_copy(self.value.next_write(), value_gm)

    @jit
    def publish_replay_k(self, replay_k_gm):
        """发布归一化生成的 FP16 K 记录。"""
        mem_copy(replay_k_gm, self.replay_k.next_read())

    @jit
    def publish_replay_decay(self, replay_decay_gm):
        """发布门控激活生成的 FP32 decay 记录。"""
        mem_copy(replay_decay_gm, self.replay_decay.next_read())

    @jit
    def publish_replay_u(self, replay_u_gm):
        """发布 delta 计算生成的 FP32 修正量。"""
        mem_copy(replay_u_gm, self.replay_u.next_read())


class OutputGateStage:
    """为一个完整的 value head 执行 RMSNorm 和 sigmoid 输出门控。"""

    def __init__(self):
        self.gamma_fp32 = Buffer(MemLoc.UB, dtypes.float32, make_layout((TILE,)))
        self.gate_input = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, TILE)), depth=2
        )
        self.output = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, TILE)), depth=2
        )

    @jit
    def load_gamma(self, output_norm_weight_gm: Tensor):
        mem_copy(self.gamma_fp32, output_norm_weight_gm)

    def issue_gate(self, gate_gm):
        mem_copy(self.gate_input.next_write(), gate_gm)

    @jit
    def apply_and_store(self, output_gm, rms_norm_eps: float, recurrent_output):
        """执行 FP32 RMSNorm 和 sigmoid，并发布门控后的 BF16 结果。"""
        gate_input_slot = self.gate_input.next_read()
        output_slot = self.output.next_write()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32)
            raw_pre = vload(recurrent_output, 0)
            raw_post = vload(recurrent_output, VL)
            sum_pre = vreduce_sum(vmul(raw_pre, raw_pre, mask=full), mask=full)
            sum_post = vreduce_sum(vmul(raw_post, raw_post, mask=full), mask=full)
            total_sum = vadd(
                vdup(sum_pre, mask=full), vdup(sum_post, mask=full), mask=full
            )
            mean_square = vmuls(total_sum, 1.0 / TILE, mask=full)
            mean_square = vadds(mean_square, rms_norm_eps, mask=full)
            root_mean_square = vsqrt(mean_square, mask=full)
            inverse_rms = vdiv(one, root_mean_square, mask=full)
            for segment in range(D_SEGMENTS):
                offset = segment * VL
                raw = vload(recurrent_output, offset)
                gamma = vload(self.gamma_fp32, offset)
                normalized = vmul(raw, inverse_rms, mask=full)
                normalized = vmul(normalized, gamma, mask=full)
                gate_bf16 = vload_unpack(
                    gate_input_slot, offset, unpack_mode=UnpackMode.B16_TO_B32
                )
                gate = vcast(gate_bf16, dtypes.float32, mask=full)
                activated_gate = vneg(gate, mask=full)
                activated_gate = vexp(activated_gate, mask=full)
                activated_gate = vadds(activated_gate, 1.0, mask=full)
                activated_gate = vdiv(one, activated_gate, mask=full)
                final_output = vmul(normalized, activated_gate, mask=full)
                final_bf16 = vcast(final_output, dtypes.bfloat16, mask=full)
                vstore_pack(
                    output_slot, offset, final_bf16, full, pack_mode=PackMode.B32_TO_B16
                )
        mem_copy(output_gm, output_slot)


class OutputProjectionStage:
    """共用输出投影，分块映射和缓冲区大小在编译期确定。"""

    def __init__(
        self,
        tile_width: int = 128,
        row_count: int = TILE,
        input_l2_cache_ctl: int = 0,
        weight_l2_cache_ctl: int = 0,
        store_l2_cache_ctl: int = 0,
        tiles_per_core: int = 2,
        *,
        kc: int = OPROJ_KC,
        resident_input: bool = False,
        resident_weight: bool = False,
        tile_stride: int = 1,
        l0c_depth: int = 1,
        row_capacity: int = 64,
        num_heads: int,
        allow_tail: bool = False,
    ):
        self.allow_tail = bool(allow_tail)
        self.num_heads = int(num_heads)
        self.tiles_per_core = int(tiles_per_core)
        self.tile_width = int(tile_width)
        self.row_count = row_count
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.store_l2_cache_ctl = int(store_l2_cache_ctl)
        self.kc = int(kc)
        self.resident_input = resident_input
        self.resident_weight = resident_weight
        self.tile_stride = int(tile_stride)
        self.row_capacity = int(row_capacity)
        self.ks = OPROJ_KS
        if self.resident_weight:
            self.resident_weight_rows = self.tile_width * self.tiles_per_core
            self.b_l1 = Buffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.resident_weight_rows, self.num_heads * TILE)),
            )
        else:
            self.b_l1 = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.tile_width, self.kc)),
                depth=2,
            )
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout(
                (
                    self.row_capacity
                    * (self.num_heads * TILE if self.resident_input else self.kc),
                )
            ),
            depth=1 if self.resident_input else 2,
            data_format="nd",
        )
        self.a_l1_shape = (
            self.row_count,
            self.num_heads * TILE if self.resident_input else self.kc,
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((self.row_capacity * self.ks,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (self.row_count, self.ks)
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((self.tile_width, self.ks)),
            depth=2,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((self.row_capacity * self.tile_width,)),
            depth=l0c_depth,
            data_format="nd",
        )
        self.l0c_shape = (self.row_count, self.tile_width)

    @jit
    def prefetch_weight(self, weight_gm: Tensor):
        """将当前 AIC 负责的完整输出宽度、K768 的 NZ 权重分片保留在 L1。"""
        if const_expr(self.resident_weight):
            resident_shards = ceil_div(weight_gm.shape[0], self.resident_weight_rows)
            if get_block_idx() < resident_shards:
                mem_copy(
                    self.b_l1,
                    tile_slice(
                        weight_gm,
                        (self.resident_weight_rows, self.num_heads * TILE),
                        (get_block_idx(), 0),
                    ),
                    options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
                )

    @jit
    def project(
        self,
        out_gm: Tensor,
        gated_workspace_gm: Tensor,
        weight_gm: Tensor,
        k_tiles: int,
        row_tile: int,
    ):
        """计算分配给各 AIC 的输出分块。"""
        tile_count = ceil_div(out_gm.shape[1], self.tile_width)
        wave_start = row_tile * self.row_count
        wave_end = wave_start + self.row_count
        if const_expr(self.allow_tail):
            if wave_end > out_gm.shape[0]:
                wave_end = out_gm.shape[0]
        valid = wave_end - wave_start
        gated_wave = gated_workspace_gm[wave_start:wave_end, None]
        out_wave = out_gm[wave_start:wave_end, None]
        if const_expr(self.resident_input):
            resident_a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                resident_a,
                tile_slice(
                    gated_wave, (self.row_capacity, self.num_heads * TILE), (0, 0)
                ),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
        first_tile = get_block_idx() * self.tiles_per_core
        for local_tile in range(self.tiles_per_core):
            if const_expr(self.resident_input):
                output_tile = get_block_idx() * self.tiles_per_core + local_tile
            elif const_expr(self.tile_stride != 1):
                output_tile = get_block_idx() + local_tile * self.tile_stride
            else:
                output_tile = first_tile + local_tile
            if output_tile < tile_count:
                accumulator = from_tensor(
                    self.l0c.next_write(),
                    layout=make_layout(self.l0c_shape),
                    data_format="nz",
                )
                for kc_idx in range(k_tiles * TILE // self.kc):
                    if const_expr(self.resident_input):
                        a_panel = tile_slice(
                            resident_a, (self.row_capacity, self.kc), (0, kc_idx)
                        )
                    else:
                        a_panel = from_tensor(
                            self.a_l1.next_write(),
                            layout=make_layout(self.a_l1_shape),
                            data_format="nz",
                        )
                        if const_expr(self.allow_tail):
                            mem_copy(
                                a_panel[0:valid, None],
                                gated_wave[
                                    None, kc_idx * self.kc : (kc_idx + 1) * self.kc
                                ],
                                options=copy_options(
                                    l2_cache_ctl=self.input_l2_cache_ctl
                                ),
                            )
                        else:
                            mem_copy(
                                a_panel,
                                tile_slice(
                                    gated_wave,
                                    (self.row_capacity, self.kc),
                                    (0, kc_idx),
                                ),
                                options=copy_options(
                                    l2_cache_ctl=self.input_l2_cache_ctl
                                ),
                            )
                    if const_expr(self.resident_weight):
                        b_panel = tile_slice(
                            self.b_l1, (self.tile_width, self.kc), (local_tile, kc_idx)
                        )
                    else:
                        b_panel = self.b_l1.next_write()
                        mem_copy(
                            b_panel,
                            tile_slice(
                                weight_gm,
                                (self.tile_width, self.kc),
                                (output_tile, kc_idx),
                            ),
                            options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
                        )
                    for ks_idx in range(self.kc // self.ks):
                        l0a_slot = from_tensor(
                            self.l0a.next_write(),
                            layout=make_layout(self.l0a_shape),
                            data_format="nz",
                        )
                        l0a_slot_source_6 = tile_slice(
                            a_panel, (self.row_capacity, self.ks), (0, ks_idx)
                        )
                        l0a_slot = l0a_slot[tuple(map(slice, l0a_slot_source_6.shape))]
                        mem_copy(l0a_slot, l0a_slot_source_6)
                        l0b_slot = self.l0b.next_write()
                        l0b_slot_source_15 = tile_slice(
                            b_panel, (self.tile_width, self.ks), (0, ks_idx)
                        )
                        l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_15.shape))]
                        mem_copy(l0b_slot, l0b_slot_source_15)
                        matmul(
                            accumulator,
                            l0a_slot,
                            l0b_slot,
                            init=kc_idx == 0 and ks_idx == 0,
                        )
                if const_expr(self.allow_tail):
                    if valid < self.row_count:
                        for row in range(valid // 32):
                            mem_copy(
                                tile_slice(
                                    out_wave, (32, self.tile_width), (row, output_tile)
                                ),
                                tile_slice(
                                    accumulator, (32, self.tile_width), (row, 0)
                                ),
                                options=copy_options(
                                    l2_cache_ctl=self.store_l2_cache_ctl
                                ),
                            )
                    else:
                        mem_copy(
                            tile_slice(
                                out_wave,
                                (self.row_capacity, self.tile_width),
                                (0, output_tile),
                            ),
                            accumulator,
                            options=copy_options(l2_cache_ctl=self.store_l2_cache_ctl),
                        )
                else:
                    mem_copy(
                        tile_slice(
                            out_wave,
                            (self.row_capacity, self.tile_width),
                            (0, output_tile),
                        ),
                        accumulator,
                        options=copy_options(l2_cache_ctl=self.store_l2_cache_ctl),
                    )


class RecurrentStage(RecurrentKDAStage):
    """保持运行中的状态为 FP32，并沿用对应调度的检查点加载策略。"""

    def __init__(self, row_block, num_heads, config, state_buffer_depth: int = 1):
        super().__init__(
            row_block,
            num_heads,
            state_key_unroll=4,
            state_update_unroll=config.state_update_unroll,
            state_buffer_depth=state_buffer_depth,
        )
        self.store_policy = config.state_store_l2_cache_ctl
        self.load_policy = config.state_load_l2_cache_ctl

    def load_state(self, state_gm):
        mem_copy(
            self.state, state_gm, options=copy_options(l2_cache_ctl=self.load_policy)
        )


class TP4HeadOutputStage:
    """每个 AIC 为至多四个列分块保留输出部分和。"""

    def __init__(self, rows, heads):
        self.rows = rows
        self.heads = heads
        self.panel_width = 256
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((rows, self.panel_width)), depth=2
        )
        self.a_l1_shape = (rows, self.panel_width)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((256, self.panel_width)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((rows, OPROJ_KS)), depth=2
        )
        self.l0a_shape = (rows, OPROJ_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((256, OPROJ_KS)), depth=2
        )
        self.c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()
        self.c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()
        self.c2 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()
        self.c3 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()

    @jit
    def _accumulate_tile(self, tensors, accumulator, column, head_range):
        out, gated, weight = tensors
        first_head, last_head = head_range
        if column < HIDDEN_SIZE // 256:
            if get_block_num() <= 24:
                for head in range(first_head, last_head, 2):
                    a = from_tensor(
                        self.a_l1.next_write(),
                        layout=make_layout(self.a_l1_shape),
                        data_format="nz",
                    )
                    b = self.b_l1.next_write()
                    mem_copy(a, tile_slice(gated, (self.rows, 256), (0, head // 2)))
                    mem_copy(b, tile_slice(weight, (256, 256), (column, head // 2)))
                    for ks in range(4):
                        la = from_tensor(
                            self.l0a.next_write(),
                            layout=make_layout(self.l0a_shape),
                            data_format="nz",
                        )
                        lb = self.l0b.next_write()
                        mem_copy(la, tile_slice(a, (self.rows, OPROJ_KS), (0, ks)))
                        mem_copy(lb, tile_slice(b, (256, OPROJ_KS), (0, ks)))
                        matmul(accumulator, la, lb, init=head == 0 and ks == 0)
            else:
                for head in range(first_head, last_head):
                    a = from_tensor(
                        self.a_l1.next_write(),
                        layout=make_layout(self.a_l1_shape),
                        data_format="nz",
                    )
                    b = self.b_l1.next_write()
                    mem_copy(
                        tile_slice(a, (self.rows, TILE), (0, 0)),
                        tile_slice(gated, (self.rows, TILE), (0, head)),
                    )
                    mem_copy(
                        tile_slice(b, (256, TILE), (0, 0)),
                        tile_slice(weight, (256, TILE), (column, head)),
                    )
                    for ks in range(TILE // OPROJ_KS):
                        la = from_tensor(
                            self.l0a.next_write(),
                            layout=make_layout(self.l0a_shape),
                            data_format="nz",
                        )
                        lb = self.l0b.next_write()
                        mem_copy(la, tile_slice(a, (self.rows, OPROJ_KS), (0, ks)))
                        mem_copy(lb, tile_slice(b, (256, OPROJ_KS), (0, ks)))
                        matmul(accumulator, la, lb, init=head == 0 and ks == 0)
            if last_head == self.heads:
                mem_copy(tile_slice(out, (self.rows, 256), (0, column)), accumulator)

    @jit
    def accumulate(self, out, gated, weight, first_head, last_head):
        tensors = (out, gated, weight)
        head_range = (first_head, last_head)
        column = get_block_idx()
        stride = get_block_num()
        self._accumulate_tile(tensors, self.c0, column, head_range)
        self._accumulate_tile(tensors, self.c1, column + stride, head_range)
        self._accumulate_tile(tensors, self.c2, column + 2 * stride, head_range)
        self._accumulate_tile(tensors, self.c3, column + 3 * stride, head_range)


class TP4DynamicRecurrentStage(RecurrentStage):
    def __init__(self):
        super().__init__(128, TP4_HEADS, Schedule())

    @jit
    def load_gate_params(self, a_log_gm, dt_bias_gm):
        n = a_log_gm.shape[0]
        mem_copy(self.a_log[0:n,], a_log_gm)
        mem_copy(self.dt_bias[0:n, None], dt_bias_gm)


class DynamicOutputStage(OutputProjectionStage):
    @jit
    def project(self, out_gm, gated_workspace_gm, weight_gm, k_tiles, row_tile):
        tile_count = ceil_div(out_gm.shape[1], self.tile_width)
        wave_start = row_tile * self.row_count
        valid = out_gm.shape[0] - wave_start
        if valid > self.row_count:
            valid = self.row_count
        gated_wave = gated_workspace_gm[wave_start : wave_start + valid, None]
        out_wave = out_gm[wave_start : wave_start + valid, None]
        if const_expr(self.resident_input):
            resident_a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                resident_a[0:valid, None], gated_wave[None, 0 : self.num_heads * TILE]
            )
        for output_tile in range(get_block_idx(), tile_count, get_block_num()):
            accumulator = from_tensor(
                self.l0c.next_write(),
                layout=make_layout(self.l0c_shape),
                data_format="nz",
            )
            for kc_idx in range(k_tiles * TILE // self.kc):
                if const_expr(self.resident_input):
                    a_panel = tile_slice(
                        resident_a, (self.row_capacity, self.kc), (0, kc_idx)
                    )
                else:
                    a_panel = from_tensor(
                        self.a_l1.next_write(),
                        layout=make_layout(self.a_l1_shape),
                        data_format="nz",
                    )
                    mem_copy(
                        a_panel[0:valid, None],
                        gated_wave[None, kc_idx * self.kc : (kc_idx + 1) * self.kc],
                        options=copy_options(),
                    )
                if const_expr(self.resident_weight):
                    b_panel = tile_slice(
                        self.b_l1, (self.tile_width, self.kc), (0, kc_idx)
                    )
                else:
                    b_panel = self.b_l1.next_write()
                    mem_copy(
                        b_panel,
                        tile_slice(
                            weight_gm, (self.tile_width, self.kc), (output_tile, kc_idx)
                        ),
                    )
                for ks in range(self.kc // self.ks):
                    a = from_tensor(
                        self.l0a.next_write(),
                        layout=make_layout(self.l0a_shape),
                        data_format="nz",
                    )
                    b = self.l0b.next_write()
                    mem_copy(
                        a, tile_slice(a_panel, (self.row_capacity, self.ks), (0, ks))
                    )
                    mem_copy(
                        b, tile_slice(b_panel, (self.tile_width, self.ks), (0, ks))
                    )
                    matmul(accumulator, a, b, init=kc_idx == 0 and ks == 0)
            if valid < self.row_count:
                for row in range(valid // 32):
                    mem_copy(
                        tile_slice(out_wave, (32, self.tile_width), (row, output_tile)),
                        tile_slice(accumulator, (32, self.tile_width), (row, 0)),
                    )
            else:
                mem_copy(
                    tile_slice(
                        out_wave, (self.row_capacity, self.tile_width), (0, output_tile)
                    ),
                    accumulator,
                    options=copy_options(),
                )
            cube_sync_all()


class S4OutputProjectionStage(OutputProjectionStage):
    """输出投影仅处理有效行，在所有支持的核数配额下覆盖全部输出分块。"""

    def __init__(self, rows, heads, resident=False):
        super().__init__(
            256,
            rows,
            kc=384 if resident else 256,
            row_capacity=64 if resident else 128,
            num_heads=heads,
            resident_weight=resident,
            tiles_per_core=1,
        )

    @jit
    def prefetch_weight(self, weight):
        if const_expr(self.resident_weight):
            if get_block_idx() < HIDDEN_SIZE // self.tile_width:
                _copy_weight_with_policy(
                    self.b_l1,
                    tile_slice(
                        weight,
                        (self.tile_width, self.num_heads * TILE),
                        (get_block_idx(), 0),
                    ),
                    self.weight_l2_cache_ctl,
                )

    @jit
    def project(self, out, gated, weight, heads, chunk):
        start = chunk * self.row_count
        wave = gated[start : start + self.row_count, None]
        for column in range(
            get_block_idx(), HIDDEN_SIZE // self.tile_width, get_block_num()
        ):
            accumulator = from_tensor(
                self.l0c.next_write(),
                layout=make_layout(self.l0c_shape),
                data_format="nz",
            )
            for kc in range(heads * TILE // self.kc):
                left = from_tensor(
                    self.a_l1.next_write(),
                    layout=make_layout(self.a_l1_shape),
                    data_format="nz",
                )
                mem_copy(left, wave[None, kc * self.kc : (kc + 1) * self.kc])
                if const_expr(self.resident_weight):
                    right = tile_slice(self.b_l1, (self.tile_width, self.kc), (0, kc))
                else:
                    right = self.b_l1.next_write()
                    _copy_weight_with_policy(
                        right,
                        tile_slice(weight, (self.tile_width, self.kc), (column, kc)),
                        self.weight_l2_cache_ctl,
                    )
                for ks in range(self.kc // self.ks):
                    a = from_tensor(
                        self.l0a.next_write(),
                        layout=make_layout(self.l0a_shape),
                        data_format="nz",
                    )
                    b = self.l0b.next_write()
                    mem_copy(
                        a,
                        tile_slice(left, (self.row_capacity, self.ks), (0, ks))[
                            0 : self.row_count, None
                        ],
                    )
                    mem_copy(b, tile_slice(right, (self.tile_width, self.ks), (0, ks)))
                    matmul(accumulator, a, b, init=kc == 0 and ks == 0)
            mem_copy(
                out[
                    start : start + self.row_count,
                    column * self.tile_width : (column + 1) * self.tile_width,
                ],
                accumulator,
            )
        cube_sync_all()


class TP4HeadPanelBackend:
    """使小 batch 的 NB 递推与输出 K 数据面板的累加重叠执行。"""

    def __init__(self, heads):
        self.heads = heads
        self.rows = 32
        self.config = Schedule()

    @jit
    def backend(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
    ):
        batch = replay_decay.indices.shape[0]
        stride = 2 * get_block_num()
        total = batch * self.heads
        recurrent = RecurrentStage(128, self.heads, self.config)
        output_gate = OutputGateStage()
        output = TP4HeadOutputStage(self.rows, self.heads)
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = 2 * get_block_idx() + get_subblock_id()
        for slot in range(ceil_div(total, stride)):
            nb_item = slot * stride + first
            if nb_item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    nb_item,
                    nb_item + 1,
                    batch,
                    8,
                    self.heads,
                    128,
                    scale,
                    lower,
                    eps,
                    nb_group_batch=batch,
                )
            ready = (slot + 1) * stride
            if ready > total:
                ready = total
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            output.accumulate(out, q, wo, slot * stride // batch, ready // batch)
            # 跨 head 组发布结果时，保留 Cube 累加器中的中间结果。
        onchip_rewind(reset_sync_id=True)


class TP4BackendSchedule:
    def __init__(self, heads):
        self.heads = heads

    @jit
    def project_template(
        self, out, q, wo, n, chunk, width, rows, heads, resident_input, resident_weight
    ):
        # 声明当前分支的资源前，先完成上一个 Cube 阶段。
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        output = DynamicOutputStage(
            width,
            rows,
            kc=384 if resident_weight else 256,
            row_capacity=rows,
            allow_tail=True,
            num_heads=heads,
            resident_input=resident_input,
            resident_weight=resident_weight,
            tiles_per_core=1,
        )
        if const_expr(resident_weight):
            if chunk == 0:
                output.prefetch_weight(wo)
        output.project(out, q, wo, n, chunk)
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)

    @jit
    def project_output(self, out, q, wo, n, chunk):
        if q.shape[0] == 32:
            self.project_template(
                out, q, wo, n, chunk, 256, 32, TP4_HEADS, False, False
            )
        elif q.shape[0] == 64:
            self.project_template(
                out, q, wo, n, chunk, 256, 64, TP4_HEADS, False, False
            )
        else:
            self.project_template(
                out, q, wo, n, chunk, 256, 128, TP4_HEADS, False, False
            )
        # 所有模板分支都必须在共用的数据块循环回边前完成。
        cube_sync_all()

    @jit
    def backend(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
        beta_weight,
    ):
        n = self.heads
        core_count = get_block_num()
        stride = 2 * core_count
        total = replay_decay.indices.shape[0] * n
        slots = ceil_div(total, stride)
        recurrent = TP4DynamicRecurrentStage()
        output_gate = OutputGateStage()
        publish_rows = 128
        if q.shape[0] == 32:
            publish_rows = 32
        elif q.shape[0] == 64:
            publish_rows = 64

        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        if first < total:
            prefetch_recurrent_item(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                first,
                replay_decay.indices.shape[0],
                8,
                n,
                128,
                nb_group_batch=publish_rows // 8,
            )
            prefetch_checkpoint_state(
                recurrent,
                initial,
                replay_k,
                replay_decay,
                first,
                replay_decay.indices.shape[0],
                8,
                n,
                128,
                nb_group_batch=publish_rows // 8,
            )
        for slot in range(slots):
            item = slot * stride + first
            if item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    item,
                    item + 1,
                    replay_decay.indices.shape[0],
                    8,
                    n,
                    128,
                    scale,
                    lower,
                    eps,
                    defer_final_snapshot=True,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=stride,
                    nb_group_batch=publish_rows // 8,
                )
            before = slot * stride
            after = (slot + 1) * stride if (slot + 1) * stride < total else total
            jobs = publish_rows // 8 * n
            ready_before = before // jobs
            ready_after = after // jobs
            if after == total:
                ready_after = ceil_div(after, jobs)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for chunk in range(ready_before, ready_after):
                self.project_output(out, q, wo, n, chunk)
            cube_sync_all()
            onchip_rewind(reset_sync_id=True)


class TP4S4BackendSchedule:
    """在一次内核启动中融合 FP32 U 递推与仅处理有效行的输出投影。"""

    def __init__(self, heads):
        self.heads = heads

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
        rows,
        policy,
    ):
        recurrent = RecurrentStage(128, self.heads, Schedule())
        output_gate = OutputGateStage()
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        stride = 2 * get_block_num()
        for item in range(first, replay_u.indices.shape[0] * self.heads, stride):
            run_recurrent_wave(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                item,
                item + 1,
                replay_u.indices.shape[0],
                4,
                self.heads,
                128,
                scale,
                lower,
                eps,
                defer_final_snapshot=True,
            )
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.project_output(out, q, wo, rows, policy)
        global_sync_all()

    @jit
    def project_output(self, out, gated, weight, rows, policy):
        stage = S4OutputProjectionStage(rows, self.heads)
        stage.weight_l2_cache_ctl = policy
        for chunk in range(out.shape[0] // rows):
            stage.project(out, gated, weight, self.heads, chunk)


class _RawRecurrentContext(NamedTuple):
    q: Tensor
    k: Tensor
    v: Tensor
    decay: Tensor
    beta: Tensor
    initial: Tensor
    replay_u: ReplayPool
    replay_k: ReplayPool
    replay_decay: ReplayPool
    scale: float
    lower: float


@jit
def _run_raw_recurrent_item(
    recurrent,
    context,
    batch,
    head,
    raw,
    item,
    *,
    checkpoint_prefetched: bool = False,
    next_batch=0,
    next_head=0,
    next_valid=False,
):
    (q, k, v, decay, beta, initial, replay_u, replay_k, replay_decay, scale, lower) = (
        context
    )
    base = batch * 8
    if const_expr(checkpoint_prefetched):
        recurrent.state = recurrent.state_ring.next_read()
    else:
        recurrent.load_state(
            tile_slice(_checkpoint_head(initial, batch, head), (128, 128), (0, 0))
        )
    recurrent.prefetch_inputs(
        tile_slice(k, (1, 128), (base, head)),
        tile_slice(q, (1, 128), (base, head)),
        tile_slice(decay, (1, 128), (base, head)),
        tile_slice(v, (1, 128), (base, head)),
        tile_slice(beta, (1, 1), (base, head)),
    )
    for token in range(8):
        recurrent.normalize_qk(scale)
        recurrent.publish_replay_k(
            tile_slice(replay_head(replay_k, batch, head), (1, 128), (token, 0))
        )
        recurrent.activate_g_beta(lower, head)
        recurrent.publish_replay_decay(
            tile_slice(replay_head(replay_decay, batch, head), (1, 128), (token, 0))
        )
        if token + 1 < 8:
            recurrent.prefetch_inputs(
                tile_slice(k, (1, 128), (base + token + 1, head)),
                tile_slice(q, (1, 128), (base + token + 1, head)),
                tile_slice(decay, (1, 128), (base + token + 1, head)),
                tile_slice(v, (1, 128), (base + token + 1, head)),
                tile_slice(beta, (1, 1), (base + token + 1, head)),
            )
        if const_expr(checkpoint_prefetched):
            # token 7 的输入在 MTE2 队列中位于此次搬运之前。
            # 最后两个 token 使用当前槽位时，可以预取并填充另一个检查点槽位。
            if token == S8_CHECKPOINT_PREFETCH_TOKEN and next_valid:
                next_state = recurrent.state_ring.next_write()
                mem_copy(
                    next_state,
                    tile_slice(
                        _checkpoint_head(initial, next_batch, next_head),
                        (128, 128),
                        (0, 0),
                    ),
                    options=copy_options(l2_cache_ctl=recurrent.load_policy),
                )
        recurrent.compute_delta()
        recurrent.publish_replay_u(
            tile_slice(replay_head(replay_u, batch, head), (1, 128), (token, 0))
        )
        recurrent.update_state()
        mem_copy(tile_slice(raw, (1, 128), (item // 64 * 8 + token, 0)), recurrent.out)


class _DelayedGateProjection(_ProjectionOperations):
    """执行完整门控投影，不分配仅用于类型转换的 Vector 缓冲区。"""

    def __init__(
        self,
        row_count: int = TILE,
        input_l2_cache_ctl: int = 0,
        weight_l2_cache_ctl: int = 0,
        tile_width: int = TILE,
        row_capacity: int = 128,
    ):
        self.row_count = row_count
        self.row_capacity = int(row_capacity)
        self.tile_width = int(tile_width)
        self.row_block = self.row_count // 2
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((self.row_capacity * PROJECTION_KC,)),
            depth=2,
            data_format="nd",
        )
        self.a_l1_shape = (self.row_count, PROJECTION_KC)
        if self.tile_width != BETA_TILE:
            self.b_l1 = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.tile_width, PROJECTION_KC)),
                depth=2,
            )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((self.row_capacity * PROJECTION_KS,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (self.row_count, PROJECTION_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((self.tile_width, PROJECTION_KS)),
            depth=2,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((self.row_capacity * self.tile_width,)),
            depth=2,
            data_format="nd",
        )
        self.l0c_shape = (self.row_count, self.tile_width)


class _RawGateContext(NamedTuple):
    q: Tensor
    gate: Tensor
    raw: Tensor
    eps: float


@jit
def _store_raw_gated_item(output_gate, context, batch, head, item):
    q, gate, raw, eps = context
    output_gate.issue_gate(tile_slice(gate, (1, 128), (batch * 8, head)))
    for token in range(8):
        if token + 1 < 8:
            output_gate.issue_gate(
                tile_slice(gate, (1, 128), (batch * 8 + token + 1, head))
            )
        output_gate.apply_and_store(
            tile_slice(q, (1, 128), (batch * 8 + token, head)),
            eps,
            tile_slice(raw, (1, 128), (item // 64 * 8 + token, 0)),
        )


@jit
def run_s4_raw_item(recurrent, context, batch, head, raw, item):
    """保留 FP32 原始输出和 FP32 U，检查点始终只读。"""
    q, k, v, decay, beta, initial, replay_u, replay_k, replay_decay, scale, lower = (
        context
    )
    recurrent.load_state(_checkpoint_head(initial, batch, head))
    base = batch * 4
    recurrent.prefetch_inputs(
        tile_slice(k, (1, 128), (base, head)),
        tile_slice(q, (1, 128), (base, head)),
        tile_slice(decay, (1, 128), (base, head)),
        tile_slice(v, (1, 128), (base, head)),
        tile_slice(beta, (1, 1), (base, head)),
    )
    for token in range(4):
        recurrent.normalize_qk(scale)
        recurrent.publish_replay_k(
            tile_slice(replay_head(replay_k, batch, head), (1, 128), (token, 0))
        )
        recurrent.activate_g_beta(lower, head)
        recurrent.publish_replay_decay(
            tile_slice(replay_head(replay_decay, batch, head), (1, 128), (token, 0))
        )
        if token + 1 < 4:
            recurrent.prefetch_inputs(
                tile_slice(k, (1, 128), (base + token + 1, head)),
                tile_slice(q, (1, 128), (base + token + 1, head)),
                tile_slice(decay, (1, 128), (base + token + 1, head)),
                tile_slice(v, (1, 128), (base + token + 1, head)),
                tile_slice(beta, (1, 1), (base + token + 1, head)),
            )
        recurrent.compute_delta()
        recurrent.publish_replay_u(
            tile_slice(replay_head(replay_u, batch, head), (1, 128), (token, 0))
        )
        recurrent.update_state()
        mem_copy(tile_slice(raw, (1, 128), (item // 64 * 4 + token, 0)), recurrent.out)


class SequenceRecurrentStage:
    """从只读检查点回放任意正长度的序列。"""

    def __init__(self, heads):
        self.heads = int(heads)
        self.recurrent = RecurrentKDAStage(128, self.heads)
        self.output_gate = OutputGateStage()

    @jit
    def replay(
        self,
        out,
        convout,
        decay,
        beta,
        z,
        gamma,
        alog,
        bias,
        checkpoint,
        indices,
        cu,
        replay_u,
        replay_k,
        replay_decay,
        checkpoint_stride0,
        scale,
        lower,
        eps,
    ):
        self.recurrent.load_gate_params(alog, bias)
        self.output_gate.load_gamma(gamma)
        for item in range(
            get_block_idx(), indices.shape[0] * self.heads, get_block_num()
        ):
            batch = item // self.heads
            head = item % self.heads
            slot = dtypes.int64(indices[batch])
            start = dtypes.int64(cu[batch])
            end = dtypes.int64(cu[batch + 1])
            state_begin = slot * checkpoint_stride0 + head * 128 * 128
            self.recurrent.load_state(
                checkpoint[state_begin : state_begin + 128 * 128,].view(128, 128)
            )
            for token in range(start, end):
                self.recurrent.prefetch_inputs(
                    tile_slice(convout, (1, 128), (token, self.heads + head)),
                    tile_slice(convout, (1, 128), (token, head)),
                    tile_slice(decay, (1, 128), (token, head)),
                    tile_slice(convout, (1, 128), (token, 2 * self.heads + head)),
                    tile_slice(beta, (1, 1), (token, head)),
                )
                self.output_gate.issue_gate(tile_slice(z, (1, 128), (token, head)))
                self.recurrent.normalize_qk(scale)
                self.recurrent.publish_replay_k(
                    tile_slice(
                        replay_head(replay_k, batch, head), (1, 128), (token - start, 0)
                    )
                )
                self.recurrent.activate_g_beta(lower, head)
                self.recurrent.publish_replay_decay(
                    tile_slice(
                        replay_head(replay_decay, batch, head),
                        (1, 128),
                        (token - start, 0),
                    )
                )
                self.recurrent.compute_delta()
                self.recurrent.publish_replay_u(
                    tile_slice(
                        replay_head(replay_u, batch, head), (1, 128), (token - start, 0)
                    )
                )
                self.recurrent.update_state()
                self.output_gate.apply_and_store(
                    tile_slice(out, (1, 128), (token, head)), eps, self.recurrent.out
                )


class TP4TPScheduleBackendOperations:
    @jit
    def _configure_recurrent(self, _recurrent_stage):
        """静态调度使用计算阶段构造函数中设置的策略。"""
        return

    @jit
    def backend(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
    ):
        total = replay_decay.indices.shape[0] * self.num_heads
        slots = ceil_div(total, 64)
        recurrent = RecurrentStage(
            128, self.num_heads, self.config, state_buffer_depth=1
        )
        self._configure_recurrent(recurrent)
        output_gate = OutputGateStage()
        resident = False
        kc = self.config.output_kc or (384 if self.publish_rows <= 64 else 256)
        if self.publish_rows == 256 and resident:
            kc = 128
        output = OutputProjectionStage(
            256,
            self.publish_rows,
            kc=kc,
            tiles_per_core=1,
            resident_weight=resident,
            row_capacity=self.publish_rows,
            num_heads=self.num_heads,
            allow_tail=self.config.ragged_output,
        )
        output.prefetch_weight(wo)
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        if first < total:
            prefetch_recurrent_item(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                first,
                replay_decay.indices.shape[0],
                8,
                self.num_heads,
                128,
                nb_group_batch=self.publish_rows // 8,
            )
            prefetch_checkpoint_state(
                recurrent,
                initial,
                replay_k,
                replay_decay,
                first,
                replay_decay.indices.shape[0],
                8,
                self.num_heads,
                128,
                nb_group_batch=self.publish_rows // 8,
            )
        for slot in range(slots):
            item = slot * 64 + first
            if item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    item,
                    item + 1,
                    replay_decay.indices.shape[0],
                    8,
                    self.num_heads,
                    128,
                    scale,
                    lower,
                    eps,
                    defer_final_snapshot=True,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=64,
                    nb_group_batch=self.publish_rows // 8,
                    checkpoint_prefetched=False,
                )
            before = slot * 64
            after = (slot + 1) * 64 if (slot + 1) * 64 < total else total
            jobs = self.publish_rows // 8 * self.num_heads
            ready_before = before // jobs
            ready_after = after // jobs
            if const_expr(self.config.ragged_output):
                if after == total:
                    ready_after = ceil_div(after, jobs)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for chunk in range(ready_before, ready_after):
                output.project(out, q, wo, self.num_heads, chunk)
            onchip_rewind(reset_sync_id=True)


class TP8HeadOutputStage:
    """每个 AIC 为至多四个列分块保留输出部分和。"""

    def __init__(self, rows, heads):
        self.rows = rows
        self.heads = heads
        self.panel_width = TILE
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((rows, self.panel_width)), depth=2
        )
        self.a_l1_shape = (rows, self.panel_width)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((256, self.panel_width)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((rows, OPROJ_KS)), depth=2
        )
        self.l0a_shape = (rows, OPROJ_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((256, OPROJ_KS)), depth=2
        )
        self.c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()
        self.c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()
        self.c2 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()
        self.c3 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 256)), depth=1
        ).next_write()

    @jit
    def _accumulate_tile(self, tensors, accumulator, column, head_range):
        out, gated, weight = tensors
        first_head, last_head = head_range
        if column < HIDDEN_SIZE // 256:
            for head in range(first_head, last_head):
                a = from_tensor(
                    self.a_l1.next_write(),
                    layout=make_layout(self.a_l1_shape),
                    data_format="nz",
                )
                b = self.b_l1.next_write()
                mem_copy(
                    tile_slice(a, (self.rows, TILE), (0, 0)),
                    tile_slice(gated, (self.rows, TILE), (0, head)),
                )
                mem_copy(
                    tile_slice(b, (256, TILE), (0, 0)),
                    tile_slice(weight, (256, TILE), (column, head)),
                )
                for ks in range(TILE // OPROJ_KS):
                    la = from_tensor(
                        self.l0a.next_write(),
                        layout=make_layout(self.l0a_shape),
                        data_format="nz",
                    )
                    lb = self.l0b.next_write()
                    mem_copy(la, tile_slice(a, (self.rows, OPROJ_KS), (0, ks)))
                    mem_copy(lb, tile_slice(b, (256, OPROJ_KS), (0, ks)))
                    matmul(accumulator, la, lb, init=head == 0 and ks == 0)
            if last_head == self.heads:
                mem_copy(tile_slice(out, (self.rows, 256), (0, column)), accumulator)

    @jit
    def accumulate(self, out, gated, weight, first_head, last_head):
        tensors = (out, gated, weight)
        head_range = (first_head, last_head)
        column = get_block_idx()
        stride = get_block_num()
        self._accumulate_tile(tensors, self.c0, column, head_range)
        self._accumulate_tile(tensors, self.c1, column + stride, head_range)
        self._accumulate_tile(tensors, self.c2, column + 2 * stride, head_range)
        self._accumulate_tile(tensors, self.c3, column + 3 * stride, head_range)


class TP8DynamicRecurrentStage(RecurrentStage):
    def __init__(self):
        super().__init__(128, RECURRENT_HEAD_CAPACITY, Schedule())

    @jit
    def load_gate_params(self, a_log_gm, dt_bias_gm):
        n = a_log_gm.shape[0]
        mem_copy(self.a_log[0:n,], a_log_gm)
        mem_copy(self.dt_bias[0:n, None], dt_bias_gm)


class TP8HeadPanelBackend:
    """使小 batch 的 NB 递推与输出 K 数据面板的累加重叠执行。"""

    def __init__(self, heads):
        self.heads = heads
        self.rows = 64
        self.config = Schedule()

    @jit
    def backend(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
    ):
        batch = replay_decay.indices.shape[0]
        stride = 2 * get_block_num()
        total = batch * self.heads
        recurrent = RecurrentStage(128, self.heads, self.config)
        output_gate = OutputGateStage()
        output = TP8HeadOutputStage(self.rows, self.heads)
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = 2 * get_block_idx() + get_subblock_id()
        for slot in range(ceil_div(total, stride)):
            nb_item = slot * stride + first
            if nb_item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    nb_item,
                    nb_item + 1,
                    batch,
                    8,
                    self.heads,
                    128,
                    scale,
                    lower,
                    eps,
                    nb_group_batch=batch,
                )
            ready = (slot + 1) * stride
            if ready > total:
                ready = total
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            output.accumulate(out, q, wo, slot * stride // batch, ready // batch)
            # 跨 head 组发布结果时，保留 Cube 累加器中的中间结果。
        onchip_rewind(reset_sync_id=True)


class TP8BackendSchedule:
    def __init__(self, heads):
        self.heads = heads

    @jit
    def project_template(
        self, out, q, wo, n, chunk, width, rows, heads, resident_input, resident_weight
    ):
        # 声明当前分支的资源前，先完成上一个 Cube 阶段。
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        output = DynamicOutputStage(
            width,
            rows,
            kc=384 if resident_weight else 256,
            row_capacity=rows,
            allow_tail=True,
            num_heads=heads,
            resident_input=resident_input,
            resident_weight=resident_weight,
            tiles_per_core=1,
        )
        if const_expr(resident_weight):
            if chunk == 0:
                output.prefetch_weight(wo)
        output.project(out, q, wo, n, chunk)
        cube_sync_all()
        onchip_rewind(reset_sync_id=True)

    @jit
    def project_output(self, out, q, wo, n, chunk):
        if q.shape[0] == 32:
            self.project_template(
                out, q, wo, n, chunk, 256, 32, TP8_HEADS, False, False
            )
        elif q.shape[0] == 64:
            self.project_template(
                out, q, wo, n, chunk, 256, 64, TP8_HEADS, False, False
            )
        else:
            self.project_template(
                out, q, wo, n, chunk, 128, 128, TP8_HEADS, True, False
            )
        # 所有模板分支都必须在共用的数据块循环回边前完成。
        cube_sync_all()

    @jit
    def backend(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
        beta_weight,
    ):
        n = self.heads
        core_count = get_block_num()
        stride = 2 * core_count
        total = replay_decay.indices.shape[0] * n
        slots = ceil_div(total, stride)
        recurrent = TP8DynamicRecurrentStage()
        output_gate = OutputGateStage()
        publish_rows = 128
        if q.shape[0] == 32:
            publish_rows = 32
        elif q.shape[0] == 64:
            publish_rows = 64

        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        if first < total:
            prefetch_recurrent_item(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                first,
                replay_decay.indices.shape[0],
                8,
                n,
                128,
                nb_group_batch=publish_rows // 8,
            )
            prefetch_checkpoint_state(
                recurrent,
                initial,
                replay_k,
                replay_decay,
                first,
                replay_decay.indices.shape[0],
                8,
                n,
                128,
                nb_group_batch=publish_rows // 8,
            )
        for slot in range(slots):
            item = slot * stride + first
            if item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    item,
                    item + 1,
                    replay_decay.indices.shape[0],
                    8,
                    n,
                    128,
                    scale,
                    lower,
                    eps,
                    defer_final_snapshot=True,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=stride,
                    nb_group_batch=publish_rows // 8,
                )
            before = slot * stride
            after = (slot + 1) * stride if (slot + 1) * stride < total else total
            jobs = publish_rows // 8 * n
            ready_before = before // jobs
            ready_after = after // jobs
            if after == total:
                ready_after = ceil_div(after, jobs)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for chunk in range(ready_before, ready_after):
                self.project_output(out, q, wo, n, chunk)
            cube_sync_all()
            onchip_rewind(reset_sync_id=True)


class S4PipelinedBackendSchedule(TP4S4BackendSchedule):
    """按完整行块发布输出投影，并预取下一项递推状态。"""

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
        rows,
        policy,
    ):
        recurrent = RecurrentStage(128, self.heads, Schedule(), state_buffer_depth=2)
        output_gate = OutputGateStage()
        stage = S4OutputProjectionStage(rows, self.heads)
        stage.weight_l2_cache_ctl = policy
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        stride = 2 * get_block_num()
        batch = replay_u.indices.shape[0]
        total = batch * self.heads
        if first < total:
            prefetch_recurrent_item(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                first,
                batch,
                4,
                self.heads,
                128,
            )
            prefetch_checkpoint_state(
                recurrent,
                initial,
                replay_k,
                replay_decay,
                first,
                batch,
                4,
                self.heads,
                128,
            )
        # 每个完整发布块对应一组连续 batch，AIV 可在 Cube 投影时递推下一组。
        chunk_items = rows // S4_SEQUENCE_LENGTH * self.heads
        for slot in range(ceil_div(total, stride)):
            item = slot * stride + first
            if item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    item,
                    item + 1,
                    batch,
                    4,
                    self.heads,
                    128,
                    scale,
                    lower,
                    eps,
                    defer_final_snapshot=True,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=stride,
                    checkpoint_prefetched=True,
                )
            ready_before = slot * stride // chunk_items
            ready_after = (
                (slot + 1) * stride if (slot + 1) * stride < total else total
            ) // chunk_items
            if ready_after > ready_before:
                publish_vector_grid_to_cube(3, 4, ack_flag=5)
                for chunk in range(ready_before, ready_after):
                    stage.project(out, q, wo, self.heads, chunk)
                cube_sync_all()
        global_sync_all()


class _HeadStreamOutput(OutputProjectionStage):
    def __init__(self):
        super().__init__(
            224,
            256,
            kc=256,
            tiles_per_core=1,
            resident_weight=False,
            row_capacity=256,
            num_heads=TP8_HEADS,
        )
        self.accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )

    @jit
    def prefetch_panel(self, weight, panel):
        mem_copy(
            self.b_l1.next_write(),
            tile_slice(weight, (224, 256), (get_block_idx(), panel)),
            options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
        )

    @jit
    def accumulate_panel(self, q, panel, chunk):
        a = from_tensor(
            self.a_l1.next_write(),
            layout=make_layout(self.a_l1_shape),
            data_format="nz",
        )
        mem_copy(
            a,
            tile_slice(q, (256, 256), (chunk, panel)),
            options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
        )
        b = self.b_l1.next_read()
        for ks in range(4):
            la = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            src_a = tile_slice(a, (256, 64), (0, ks))
            la = la[tuple(map(slice, src_a.shape))]
            mem_copy(la, src_a)
            lb = self.l0b.next_write()
            src_b = tile_slice(b, (224, 64), (0, ks))
            lb = lb[tuple(map(slice, src_b.shape))]
            mem_copy(lb, src_b)
            matmul(self.accumulator, la, lb, init=panel == 0 and ks == 0)

    @jit
    def finish(self, out, chunk):
        mem_copy(
            tile_slice(out, (256, 224), (chunk, get_block_idx())),
            self.accumulator,
            options=copy_options(l2_cache_ctl=self.store_l2_cache_ctl),
        )


@jit
def _head_stream_persistent_raw_round(recurrent, context, raw, first, row, chunk):
    item = first + row * 64
    batch = chunk * 32 + item % 32
    head = item // 32
    next_item = item + 64
    next_batch = chunk * 32 + next_item % 32
    next_head = next_item // 32
    next_valid = next_item < 384
    if next_item >= 384:
        # 每个完整窗口消耗每个 AIV 的六个状态槽位。处理最后一个任务时，
        # 发布下一窗口的首个槽位，以保持两个环形缓冲阶段的推进顺序。
        next_batch = (chunk + 1) * 32 + first % 32
        next_head = first // 32
        next_valid = next_batch < context.initial.indices.shape[0]
    _run_raw_recurrent_item(
        recurrent,
        context,
        batch,
        head,
        raw,
        item,
        checkpoint_prefetched=True,
        next_batch=next_batch,
        next_head=next_head,
        next_valid=next_valid,
    )


class TP8TPScheduleBackendOperations:
    @jit
    def _configure_recurrent(self, _recurrent_stage):
        """静态调度使用计算阶段构造函数中设置的策略。"""
        return

    @jit
    def backend(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
    ):
        total = replay_decay.indices.shape[0] * self.num_heads
        slots = ceil_div(total, 64)
        recurrent = RecurrentStage(
            128, self.num_heads, self.config, state_buffer_depth=2
        )
        self._configure_recurrent(recurrent)
        output_gate = OutputGateStage()
        resident = False
        kc = self.config.output_kc or (384 if self.publish_rows <= 64 else 256)
        if self.publish_rows == 256 and resident:
            kc = 128
        output = OutputProjectionStage(
            256,
            self.publish_rows,
            kc=kc,
            tiles_per_core=1,
            resident_weight=resident,
            row_capacity=self.publish_rows,
            num_heads=self.num_heads,
            allow_tail=self.config.ragged_output,
        )
        output.prefetch_weight(wo)
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        if first < total:
            prefetch_recurrent_item(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                first,
                replay_decay.indices.shape[0],
                8,
                self.num_heads,
                128,
                nb_group_batch=self.publish_rows // 8,
            )
            prefetch_checkpoint_state(
                recurrent,
                initial,
                replay_k,
                replay_decay,
                first,
                replay_decay.indices.shape[0],
                8,
                self.num_heads,
                128,
                nb_group_batch=self.publish_rows // 8,
            )
        for slot in range(slots):
            item = slot * 64 + first
            if item < total:
                run_recurrent_wave(
                    recurrent,
                    output_gate,
                    q,
                    k,
                    v,
                    initial.accurate_decay,
                    gate,
                    beta,
                    initial,
                    replay_u,
                    replay_k,
                    replay_decay,
                    item,
                    item + 1,
                    replay_decay.indices.shape[0],
                    8,
                    self.num_heads,
                    128,
                    scale,
                    lower,
                    eps,
                    defer_final_snapshot=True,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=64,
                    nb_group_batch=self.publish_rows // 8,
                    checkpoint_prefetched=True,
                )
            before = slot * 64
            after = (slot + 1) * 64 if (slot + 1) * 64 < total else total
            jobs = self.publish_rows // 8 * self.num_heads
            ready_before = before // jobs
            ready_after = after // jobs
            if const_expr(self.config.ragged_output):
                if after == total:
                    ready_after = ceil_div(after, jobs)
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for chunk in range(ready_before, ready_after):
                output.project(out, q, wo, self.num_heads, chunk)
            onchip_rewind(reset_sync_id=True)


def publish_batch_group_vector_grid_to_cube(vector_flag: int, cube_flag: int):
    """将按序完成的 AIV MTE3 写入发布给全部 AIC，无需反向广播。"""
    vec_sync_block_arrive(PIPE.MTE3, vector_flag, mode=2)
    cube_sync_block_wait(PIPE.S, vector_flag, mode=2)
    cube_sync_block_arrive(PIPE.FIXPIPE, cube_flag, mode=0)
    cube_sync_block_wait(PIPE.S, cube_flag, mode=0)


class _TiledOutputProjectionStage:
    """共用输出投影，分块映射和缓冲区大小在编译期确定。"""

    def __init__(
        self,
        tile_width: int = 128,
        row_count: int = TILE,
        input_l2_cache_ctl: int = 0,
        weight_l2_cache_ctl: int = 0,
        store_l2_cache_ctl: int = 0,
        tiles_per_core: int = 2,
        *,
        kc: int = OPROJ_KC,
        resident_input: bool = False,
        resident_weight: bool = False,
        tile_stride: int = 1,
        l0c_depth: int = 1,
        row_capacity: int = 64,
    ):
        self.tiles_per_core = int(tiles_per_core)
        self.tile_width = int(tile_width)
        self.row_count = row_count
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.store_l2_cache_ctl = int(store_l2_cache_ctl)
        self.kc = int(kc)
        self.resident_input = resident_input
        self.resident_weight = resident_weight
        self.tile_stride = int(tile_stride)
        self.row_capacity = int(row_capacity)
        self.ks = OPROJ_KS
        if self.resident_weight:
            self.resident_weight_rows = self.tile_width * self.tiles_per_core
            self.b_l1 = Buffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.resident_weight_rows, TP16_HEADS * TILE)),
            )
        else:
            self.b_l1 = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.tile_width, self.kc)),
                depth=2,
            )
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout(
                (self.row_capacity * (768 if self.resident_input else self.kc),)
            ),
            depth=1 if self.resident_input else 2,
            data_format="nd",
        )
        self.a_l1_shape = (self.row_count, 768 if self.resident_input else self.kc)
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((self.row_capacity * self.ks,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (self.row_count, self.ks)
        self.l0b = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((self.tile_width, self.ks)),
            depth=2,
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((self.row_capacity * self.tile_width,)),
            depth=l0c_depth,
            data_format="nd",
        )
        self.l0c_shape = (self.row_count, self.tile_width)

    @jit
    def prefetch_weight(self, weight_gm: Tensor):
        """将当前 AIC 负责的完整输出宽度、K768 的 NZ 权重分片保留在 L1。"""
        if const_expr(self.resident_weight):
            resident_shards = ceil_div(weight_gm.shape[0], self.resident_weight_rows)
            if get_block_idx() < resident_shards:
                mem_copy(
                    self.b_l1,
                    tile_slice(
                        weight_gm,
                        (self.resident_weight_rows, TP16_HEADS * TILE),
                        (get_block_idx(), 0),
                    ),
                    options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
                )

    @jit
    def project(
        self,
        out_gm: Tensor,
        gated_workspace_gm: Tensor,
        weight_gm: Tensor,
        k_tiles: int,
        row_tile: int,
    ):
        """计算分配给各 AIC 的输出分块。"""
        tile_count = ceil_div(out_gm.shape[1], self.tile_width)
        wave_start = row_tile * self.row_count
        wave_end = wave_start + self.row_count
        gated_wave = gated_workspace_gm[wave_start:wave_end, None]
        out_wave = out_gm[wave_start:wave_end, None]
        if const_expr(self.resident_input):
            resident_a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                resident_a,
                tile_slice(gated_wave, (self.row_capacity, 768), (0, 0)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
        first_tile = get_block_idx() * self.tiles_per_core
        for local_tile in range(self.tiles_per_core):
            if const_expr(self.resident_input):
                output_tile = get_block_idx() * self.tiles_per_core + local_tile
            elif const_expr(self.tile_stride != 1):
                output_tile = get_block_idx() + local_tile * self.tile_stride
            else:
                output_tile = first_tile + local_tile
            if output_tile < tile_count:
                accumulator = from_tensor(
                    self.l0c.next_write(),
                    layout=make_layout(self.l0c_shape),
                    data_format="nz",
                )
                for kc_idx in range(k_tiles * TILE // self.kc):
                    if const_expr(self.resident_input):
                        a_panel = tile_slice(
                            resident_a, (self.row_capacity, self.kc), (0, kc_idx)
                        )
                    else:
                        a_panel = from_tensor(
                            self.a_l1.next_write(),
                            layout=make_layout(self.a_l1_shape),
                            data_format="nz",
                        )
                        mem_copy(
                            a_panel,
                            tile_slice(
                                gated_wave, (self.row_capacity, self.kc), (0, kc_idx)
                            ),
                            options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
                        )
                    if const_expr(self.resident_weight):
                        b_panel = tile_slice(
                            self.b_l1, (self.tile_width, self.kc), (local_tile, kc_idx)
                        )
                    else:
                        b_panel = self.b_l1.next_write()
                        mem_copy(
                            b_panel,
                            tile_slice(
                                weight_gm,
                                (self.tile_width, self.kc),
                                (output_tile, kc_idx),
                            ),
                            options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
                        )
                    for ks_idx in range(self.kc // self.ks):
                        l0a_slot = from_tensor(
                            self.l0a.next_write(),
                            layout=make_layout(self.l0a_shape),
                            data_format="nz",
                        )
                        l0a_slot_source_5 = tile_slice(
                            a_panel, (self.row_capacity, self.ks), (0, ks_idx)
                        )
                        l0a_slot = l0a_slot[tuple(map(slice, l0a_slot_source_5.shape))]
                        mem_copy(l0a_slot, l0a_slot_source_5)
                        l0b_slot = self.l0b.next_write()
                        l0b_slot_source_14 = tile_slice(
                            b_panel, (self.tile_width, self.ks), (0, ks_idx)
                        )
                        l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_14.shape))]
                        mem_copy(l0b_slot, l0b_slot_source_14)
                        matmul(
                            accumulator,
                            l0a_slot,
                            l0b_slot,
                            init=kc_idx == 0 and ks_idx == 0,
                        )
                mem_copy(
                    tile_slice(
                        out_wave, (self.row_capacity, self.tile_width), (0, output_tile)
                    ),
                    accumulator,
                    options=copy_options(l2_cache_ctl=self.store_l2_cache_ctl),
                )


class S4TP16Output:
    """AIC 数量至少为 28 且容量允许时，跨 S4 行组将各 OProj 权重分片保留在 L1。"""

    @jit
    def execute(self, out, gated, weight, rows, policy):
        if get_block_num() >= 28:
            # 将驻留 A 的容量限制为 M64，为完整的 N256xK768 权重分片
            # 留出空间，保持在原有 TP16 的 L1 容量预算内。
            self.project(out, gated, weight, rows if rows <= 64 else 64, policy, True)
        else:
            self.project(out, gated, weight, rows, policy, False)

    @jit
    def project(self, out, gated, weight, rows, policy, resident):
        stage = S4OutputProjectionStage(rows, TP16_HEADS, resident)
        stage.weight_l2_cache_ctl = policy
        stage.prefetch_weight(weight)
        for chunk in range(out.shape[0] // rows):
            stage.project(out, gated, weight, TP16_HEADS, chunk)


class TP16S4BackendSchedule:
    """在一次内核启动中融合 FP32 U 递推与仅处理有效行的输出投影。"""

    def __init__(self, heads):
        self.heads = heads
        self.tp16_output = S4TP16Output()

    @jit
    def execute(
        self,
        out,
        q,
        k,
        v,
        decay,
        gate,
        beta,
        gamma,
        wo,
        alog,
        bias,
        initial,
        replay_u,
        replay_k,
        replay_decay,
        scale,
        lower,
        eps,
        rows,
        policy,
    ):
        recurrent = RecurrentStage(128, self.heads, Schedule())
        output_gate = OutputGateStage()
        recurrent.load_gate_params(alog, bias)
        output_gate.load_gamma(gamma)
        first = get_block_idx() * 2 + get_subblock_id()
        stride = 2 * get_block_num()
        for item in range(first, replay_u.indices.shape[0] * self.heads, stride):
            run_recurrent_wave(
                recurrent,
                output_gate,
                q,
                k,
                v,
                initial.accurate_decay,
                gate,
                beta,
                initial,
                replay_u,
                replay_k,
                replay_decay,
                item,
                item + 1,
                replay_u.indices.shape[0],
                4,
                self.heads,
                128,
                scale,
                lower,
                eps,
                defer_final_snapshot=True,
            )
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.project_output(out, q, wo, rows, policy)
        global_sync_all()

    @jit
    def project_output(self, out, gated, weight, rows, policy):
        self.tp16_output.execute(out, gated, weight, rows, policy)


class GridScheduleBackendOperations:
    @jit
    def _run_backend_stream(
        self,
        out_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        fa_workspace_gm: Tensor,
        norm_exchange_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        output_norm_weight_gm: Tensor,
        o_proj_weight_gm: Tensor,
        a_log_gm: Tensor,
        dt_bias_gm: Tensor,
        initial_state_gm: Tensor,
        replay_u_gm: ReplayPool,
        replay_k_gm: ReplayPool,
        replay_decay_gm: ReplayPool,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        scale_value: float,
        lower_bound: float,
        rms_norm_eps: float,
    ):
        batch_size = replay_decay_gm.indices.shape[0]
        total = batch_size * TP16_HEADS
        slot_capacity = 2 * get_block_num()
        slot_count = ceil_div(total, slot_capacity)
        chunk_rows = 8 * self.seq_len
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        recurrent_stage = RecurrentKDAStage(
            TILE,
            self.num_heads,
            state_key_unroll=KDA_STATE_KEY_UNROLL,
            state_update_unroll=KDA_STATE_UPDATE_UNROLL,
        )
        output_gate_stage = OutputGateStage()
        if const_expr(self.block_num == 8):
            output_projection_stage = _TiledOutputProjectionStage(
                self.output_tile_width,
                chunk_rows,
                kc=384,
                tiles_per_core=self.output_tiles_per_core,
                row_capacity=64,
                resident_input=True,
                resident_weight=False,
                tile_stride=1,
                l0c_depth=2,
            )
        elif const_expr(self.block_num == 16 or self.block_num == 24):
            output_projection_stage = _TiledOutputProjectionStage(
                self.output_tile_width,
                chunk_rows,
                kc=384,
                tiles_per_core=self.output_tiles_per_core,
                row_capacity=64,
                resident_input=False,
                resident_weight=False,
                tile_stride=self.block_num,
                l0c_depth=2,
            )
        else:
            output_projection_stage = _TiledOutputProjectionStage(
                self.output_tile_width,
                chunk_rows,
                kc=384,
                tiles_per_core=self.output_tiles_per_core,
                row_capacity=64,
                resident_input=False,
                resident_weight=True,
                tile_stride=1,
                l0c_depth=1,
            )
        output_projection_stage.prefetch_weight(o_proj_weight_gm)
        recurrent_stage.load_gate_params(a_log_gm, dt_bias_gm)
        output_gate_stage.load_gamma(output_norm_weight_gm)
        first_item = block_idx * 2 + subblock_idx
        if first_item < total:
            prefetch_recurrent_item(
                recurrent_stage,
                output_gate_stage,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                initial_state_gm.accurate_decay,
                output_gate_workspace_gm,
                beta_workspace_gm,
                first_item,
                batch_size,
                self.seq_len,
                TP16_HEADS,
                TILE,
                nb_group_batch=chunk_rows // self.seq_len,
            )
            prefetch_checkpoint_state(
                recurrent_stage,
                initial_state_gm,
                replay_k_gm,
                replay_decay_gm,
                first_item,
                batch_size,
                self.seq_len,
                TP16_HEADS,
                TILE,
                nb_group_batch=chunk_rows // self.seq_len,
            )
        for slot in range(slot_count):
            item = slot * slot_capacity + block_idx * 2 + subblock_idx
            if item < total:
                run_recurrent_wave(
                    recurrent_stage,
                    output_gate_stage,
                    q_workspace_gm,
                    k_workspace_gm,
                    v_workspace_gm,
                    initial_state_gm.accurate_decay,
                    output_gate_workspace_gm,
                    beta_workspace_gm,
                    initial_state_gm,
                    replay_u_gm,
                    replay_k_gm,
                    replay_decay_gm,
                    item,
                    item + 1,
                    batch_size,
                    self.seq_len,
                    TP16_HEADS,
                    TILE,
                    scale_value,
                    lower_bound,
                    rms_norm_eps,
                    defer_final_snapshot=True,
                    first_inputs_prefetched=True,
                    first_state_prefetched=True,
                    next_item_stride=2 * self.block_num,
                    nb_group_batch=chunk_rows // self.seq_len,
                )
            completed_before = (
                slot * slot_capacity if slot * slot_capacity < total else total
            )
            completed_after = (
                (slot + 1) * slot_capacity
                if (slot + 1) * slot_capacity < total
                else total
            )
            ready_before = completed_before // 48
            ready_after = completed_after // 48
            publish_vector_grid_to_cube(3, 4, ack_flag=5)
            for chunk in range(ready_before, ready_after):
                output_projection_stage.project(
                    out_gm,
                    q_workspace_gm,
                    o_proj_weight_gm,
                    k_tiles=TP16_HEADS,
                    row_tile=chunk,
                )
            onchip_rewind(reset_sync_id=True)
