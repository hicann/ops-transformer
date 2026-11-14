# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mega KDA 的投影、卷积与辅助投影阶段。"""

from __future__ import annotations

import cannbotdsl
import torch
from cannbotdsl import (
    MemLoc,
    PIPE,
    Tensor,
    TensorSpec,
    dtypes,
    from_tensor,
    make_layout,
)
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vabs,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdups,
    veq,
    vexp,
    vle,
    vload,
    vload_broadcast,
    vload_unpack,
    vmadd,
    vmem_bar,
    vmul,
    vmuls,
    vneg,
    vselect,
    vstore,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    global_sync_all,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import ceil_div, make_tiler, tile_slice

if __package__:
    from .mega_recurrent_kda_common import (
        BATCH_TILE,
        BETA_TILE,
        CONV_KERNEL,
        HIDDEN_SIZE,
        PROJECTION_KC,
        PROJECTION_KS,
        QKV_PANEL_COUNT,
        ReplayPool,
        S8_SEQUENCE_LENGTH,
        Schedule,
        TILE,
        TP16_HEADS,
        VL,
        _Core32ProjectionOperations,
        _DynamicProjectionOperations,
        _Projection384Operations,
        _ProjectionOperations,
        _conv_slot,
        _copy_weight_with_policy,
    )
else:
    from mega_recurrent_kda_common import (
        BATCH_TILE,
        BETA_TILE,
        CONV_KERNEL,
        HIDDEN_SIZE,
        PROJECTION_KC,
        PROJECTION_KS,
        QKV_PANEL_COUNT,
        ReplayPool,
        S8_SEQUENCE_LENGTH,
        Schedule,
        TILE,
        TP16_HEADS,
        VL,
        _Core32ProjectionOperations,
        _DynamicProjectionOperations,
        _Projection384Operations,
        _ProjectionOperations,
        _conv_slot,
        _copy_weight_with_policy,
    )


# 卷积搬运同步与 32 AIC 投影。
CONV_RAW_V_TO_MTE3_EVENT = 2
CORE_NUM = 32


class ProjectionStage(_ProjectionOperations):
    """对全部 BSH 行执行 Cube 投影，并由配对 AIV 发布结果。"""

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
        self.projection_half = (
            RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((self.row_capacity // 2 * self.tile_width,)),
                depth=1,
            )
            .next_write()
            .view(
                (self.row_block, self.tile_width),
            )
        )
        self.projection_bf16 = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((self.row_capacity // 2 * self.tile_width,)),
        ).view(self.row_block, self.tile_width)
        self.projection_fp32: object


class _ResidentN384ProjectionStage:
    """采用 M128xN384 投影，为三个输出复用同一 hidden 数据面板。"""

    def __init__(self, input_l2_cache_ctl: int = 0, weight_l2_cache_ctl: int = 0):
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((TILE, PROJECTION_KC)), depth=2
        )
        self.a_l1_shape = (TILE, PROJECTION_KC)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((3 * TILE, PROJECTION_KC)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((TILE, PROJECTION_KS)), depth=2
        )
        self.l0a_shape = (TILE, PROJECTION_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((TILE, PROJECTION_KS)), depth=2
        )
        self.l0c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.l0c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.l0c2 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.projection_half0 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((TILE // 2, TILE)),
            depth=1,
        ).next_write()
        self.projection_half1 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((TILE // 2, TILE)),
            depth=1,
        ).next_write()
        self.projection_half2 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((TILE // 2, TILE)),
            depth=1,
        ).next_write()
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((TILE // 2, TILE))
        )
        self.projection_fp32: object

    @jit
    def matmul_hidden3(self, hidden_gm, weight_gm, m_group, n_macro, k_tiles):
        for kc_idx in range(k_tiles * TILE // PROJECTION_KC):
            a_l1_slot = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                a_l1_slot,
                tile_slice(hidden_gm, (TILE, PROJECTION_KC), (m_group, kc_idx)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            b_l1_slot = self.b_l1.next_write()
            mem_copy(
                b_l1_slot,
                tile_slice(weight_gm, (3 * TILE, PROJECTION_KC), (n_macro, kc_idx)),
                options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
            )
            for ks_idx in range(PROJECTION_KC // PROJECTION_KS):
                l0a_slot = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                l0a_slot_source_2 = tile_slice(
                    a_l1_slot, (TILE, PROJECTION_KS), (0, ks_idx)
                )
                l0a_slot = l0a_slot[tuple(map(slice, l0a_slot_source_2.shape))]
                mem_copy(l0a_slot, l0a_slot_source_2)
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_5 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (0, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_5.shape))]
                mem_copy(l0b_slot, l0b_slot_source_5)
                matmul(self.l0c0, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half0, self.l0c0.partition(dim=0))
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_6 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (1, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_6.shape))]
                mem_copy(l0b_slot, l0b_slot_source_6)
                matmul(self.l0c1, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half1, self.l0c1.partition(dim=0))
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_7 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (2, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_7.shape))]
                mem_copy(l0b_slot, l0b_slot_source_7)
                matmul(self.l0c2, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half2, self.l0c2.partition(dim=0))

    @jit
    def acquire_result0(self):
        self.projection_fp32 = self.projection_half0

    @jit
    def acquire_result1(self):
        self.projection_fp32 = self.projection_half1

    @jit
    def acquire_result2(self):
        self.projection_fp32 = self.projection_half2

    @jit
    def cast_result(self):
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range(TILE // 2 * TILE // VL):
                offset = segment * VL
                value = vcast(
                    vload(self.projection_fp32, offset), dtypes.bfloat16, mask=full
                )
                vstore_pack(
                    self.projection_bf16,
                    offset,
                    value,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @staticmethod
    def publish(workspace_gm, m_group, output_head, subblock_idx, source):
        whole_tile = tile_slice(workspace_gm, (TILE, TILE), (m_group, output_head))
        out_half = tile_slice(
            whole_tile,
            make_tiler((TILE // 2, TILE), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, source)


class LowRankDecayStage:
    """将 f_a 保留在 L1，随后计算六个依赖它的 f_b 分块。"""

    def __init__(self, projection_stage: ProjectionStage, num_heads: int):
        self.projection_stage = projection_stage
        self.num_heads = int(num_heads)
        self.fa_l1 = from_tensor(
            RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.projection_stage.row_capacity * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.projection_stage.row_count, TILE)),
            data_format="nz",
        )
        self.fb_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=1
        ).next_write()

    @jit
    def project(
        self,
        decay_workspace_gm: Tensor,
        hidden_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        k_tiles,
        subblock_idx,
    ):
        """计算 f_b(f_a(hidden))，中间结果不经 GM 中转。"""
        self.projection_stage.matmul_hidden(hidden_gm, fa_weight_gm, 0, k_tiles)
        fa_result = from_tensor(
            self.projection_stage.l0c.next_read(),
            layout=make_layout(self.projection_stage.l0c_shape),
            data_format="nz",
        )
        mem_copy(self.fa_l1, fa_result)
        for output_head in range(self.num_heads):
            mem_copy(
                self.fb_l1,
                tile_slice(fb_weight_gm, (TILE, TILE), (output_head, 0)),
                options=copy_options(
                    l2_cache_ctl=self.projection_stage.weight_l2_cache_ctl
                ),
            )
            self.projection_stage.matmul_resident(self.fa_l1, self.fb_l1)
            self.projection_stage.publish_result(
                tile_slice(
                    decay_workspace_gm,
                    (self.projection_stage.row_capacity, TILE),
                    (0, output_head),
                )
            )
            vec_sync_all()


class ConvSiluStage:
    """在同一 AIV 流水线中融合 K=4 的逐通道因果 Conv1D 与 SiLU。"""

    def __init__(self, seq_len: int, row_block: int, row_capacity: int = 64):
        self.seq_len = seq_len
        self.row_block = row_block
        self.accurate_raw = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((seq_len, TILE))
        )
        self.window = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((CONV_KERNEL, TILE))
        )
        self.weight = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((3 * CONV_KERNEL, TILE))
        )
        self.output_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((row_capacity * TILE,))
        ).view(self.row_block, TILE)
        self.raw_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((row_capacity * TILE,))
        ).view(self.row_block, TILE)
        self.weight_ch = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((CONV_KERNEL, TILE)), depth=1
        ).next_write()

    @jit
    def install_history(self, history_ub):
        mem_copy(tile_slice(self.window, (3, TILE), (0, 0)), history_ub)
        vec_sync_all()

    @jit
    def load_weight(self, weight_gm):
        mem_copy(self.weight_ch, weight_gm)
        mem_copy(tile_slice(self.weight, (CONV_KERNEL, TILE), (0, 0)), self.weight_ch)
        vec_sync_all()

    @jit
    def load_accurate_sequence(self, pool, batch, head):
        mem_copy(
            self.accurate_raw,
            tile_slice(pool.accurate_raw, (self.seq_len, TILE), (batch, head)),
        )

    @jit
    def convolve_sequence(self, local_batch: int, raw_projection, weight_ub):
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range(TILE // VL):
                off = segment * VL
                x0_bf16 = vload_unpack(
                    self.window, off, unpack_mode=UnpackMode.B16_TO_B32
                )
                x0 = vcast(x0_bf16, dtypes.float32, mask=full)
                x1_bf16 = vload_unpack(
                    self.window, TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                x1 = vcast(x1_bf16, dtypes.float32, mask=full)
                x2_bf16 = vload_unpack(
                    self.window, 2 * TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                x2 = vcast(x2_bf16, dtypes.float32, mask=full)
                w0_bf16 = vload_unpack(
                    weight_ub, off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w0 = vcast(w0_bf16, dtypes.float32, mask=full)
                w1_bf16 = vload_unpack(
                    weight_ub, TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w1 = vcast(w1_bf16, dtypes.float32, mask=full)
                w2_bf16 = vload_unpack(
                    weight_ub, 2 * TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w2 = vcast(w2_bf16, dtypes.float32, mask=full)
                w3_bf16 = vload_unpack(
                    weight_ub, 3 * TILE + off, unpack_mode=UnpackMode.B16_TO_B32
                )
                w3 = vcast(w3_bf16, dtypes.float32, mask=full)
                for token in range(self.seq_len):
                    local_row = local_batch * self.seq_len + token
                    raw_offset = local_row * TILE
                    x3 = vload(raw_projection, raw_offset + off)
                    accurate = vload(self.accurate_raw, token * TILE + off)
                    matches = veq(x3, accurate, mask=full)
                    x3 = vselect(x3, accurate, cond_mask=matches)
                    x3_bf16 = vcast(x3, dtypes.bfloat16, mask=full)
                    product1 = vmul(x1, w1, mask=full)
                    product01 = vmadd(x0, w0, product1, mask=full)
                    product3 = vmul(x3, w3, mask=full)
                    product23 = vmadd(x2, w2, product3, mask=full)
                    acc = vadd(product01, product23, mask=full)
                    negative_acc = vneg(acc, mask=full)
                    exp_value = vexp(negative_acc, mask=full)
                    denominator = vadds(exp_value, 1.0, mask=full)
                    activated = vdiv(acc, denominator, mask=full)
                    # 小参数采用补偿计算，其余范围保留原 SiLU 结果。
                    regular_activated = activated
                    hi = vdups(0.0, dtypes.float32)
                    lo = vdups(0.0, dtypes.float32)
                    product = vmul(x0, w0, mask=full)
                    negative_product = vneg(product, mask=full)
                    product_error = vmadd(x0, w0, negative_product, mask=full)
                    total = vadd(hi, product, mask=full)
                    recovered = vsub(total, hi, mask=full)
                    prior = vsub(total, recovered, mask=full)
                    prior_error = vsub(hi, prior, mask=full)
                    new_error = vsub(product, recovered, mask=full)
                    sum_error = vadd(prior_error, new_error, mask=full)
                    term_error = vadd(sum_error, product_error, mask=full)
                    lo = vadd(lo, term_error, mask=full)
                    hi = total
                    product = vmul(x1, w1, mask=full)
                    negative_product = vneg(product, mask=full)
                    product_error = vmadd(x1, w1, negative_product, mask=full)
                    total = vadd(hi, product, mask=full)
                    recovered = vsub(total, hi, mask=full)
                    prior = vsub(total, recovered, mask=full)
                    prior_error = vsub(hi, prior, mask=full)
                    new_error = vsub(product, recovered, mask=full)
                    sum_error = vadd(prior_error, new_error, mask=full)
                    term_error = vadd(sum_error, product_error, mask=full)
                    lo = vadd(lo, term_error, mask=full)
                    hi = total
                    product = vmul(x2, w2, mask=full)
                    negative_product = vneg(product, mask=full)
                    product_error = vmadd(x2, w2, negative_product, mask=full)
                    total = vadd(hi, product, mask=full)
                    recovered = vsub(total, hi, mask=full)
                    prior = vsub(total, recovered, mask=full)
                    prior_error = vsub(hi, prior, mask=full)
                    new_error = vsub(product, recovered, mask=full)
                    sum_error = vadd(prior_error, new_error, mask=full)
                    term_error = vadd(sum_error, product_error, mask=full)
                    lo = vadd(lo, term_error, mask=full)
                    hi = total
                    product = vmul(x3, w3, mask=full)
                    negative_product = vneg(product, mask=full)
                    product_error = vmadd(x3, w3, negative_product, mask=full)
                    total = vadd(hi, product, mask=full)
                    recovered = vsub(total, hi, mask=full)
                    prior = vsub(total, recovered, mask=full)
                    prior_error = vsub(hi, prior, mask=full)
                    new_error = vsub(product, recovered, mask=full)
                    sum_error = vadd(prior_error, new_error, mask=full)
                    term_error = vadd(sum_error, product_error, mask=full)
                    lo = vadd(lo, term_error, mask=full)
                    hi = total
                    total = vadd(hi, lo, mask=full)
                    recovered = vsub(total, hi, mask=full)
                    prior = vsub(total, recovered, mask=full)
                    prior_error = vsub(hi, prior, mask=full)
                    new_error = vsub(lo, recovered, mask=full)
                    lo = vadd(prior_error, new_error, mask=full)
                    hi = total
                    absolute_hi = vabs(hi, mask=full)
                    small_limit = vdups(0.0078125, dtypes.float32)
                    small = vle(absolute_hi, small_limit, mask=full)
                    zero = vdups(0.0, dtypes.float32)
                    hi = vselect(hi, zero, cond_mask=small)
                    lo = vselect(lo, zero, cond_mask=small)
                    # SiLU = x/2 + x^2/4 - x^4/48 + x^6/480 + O(x^8).
                    # 保留保护位至 FP32 舍入，再转换为 BF16。
                    square = vmul(hi, hi, mask=full)
                    negative_square = vneg(square, mask=full)
                    square_error = vmadd(hi, hi, negative_square, mask=full)
                    half = vmuls(hi, 0.5, mask=full)
                    quarter_square = vmuls(square, 0.25, mask=full)
                    major = vadd(half, quarter_square, mask=full)
                    recovered = vsub(major, half, mask=full)
                    prior = vsub(major, recovered, mask=full)
                    prior_error = vsub(half, prior, mask=full)
                    new_error = vsub(quarter_square, recovered, mask=full)
                    major_error = vadd(prior_error, new_error, mask=full)
                    quartic = vmul(square, square, mask=full)
                    sixth = vmul(quartic, square, mask=full)
                    fourth_term = vmuls(quartic, -0.020833333333333332, mask=full)
                    sixth_term = vmuls(sixth, 0.0020833333333333333, mask=full)
                    tail = vadd(fourth_term, sixth_term, mask=full)
                    half_hi = vmuls(hi, 0.5, mask=full)
                    derivative_main = vadds(half_hi, 0.5, mask=full)
                    cubic = vmul(hi, square, mask=full)
                    derivative_tail = vmuls(cubic, 0.08333333333333333, mask=full)
                    derivative = vsub(derivative_main, derivative_tail, mask=full)
                    square_correction = vmuls(square_error, 0.25, mask=full)
                    low_correction = vmul(derivative, lo, mask=full)
                    correction = vadd(square_correction, low_correction, mask=full)
                    correction = vadd(correction, tail, mask=full)
                    correction = vadd(major_error, correction, mask=full)
                    accurate_activated = vadd(major, correction, mask=full)
                    activated = vselect(
                        accurate_activated, regular_activated, cond_mask=small
                    )
                    activated_bf16 = vcast(activated, dtypes.bfloat16, mask=full)
                    output_offset = raw_offset + off
                    vstore_pack(
                        self.output_bf16,
                        output_offset,
                        activated_bf16,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vstore_pack(
                        self.raw_bf16,
                        output_offset,
                        x3_bf16,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    x0 = x1
                    x1 = x2
                    x2 = vcast(x3_bf16, dtypes.float32, mask=full)
                final_x0 = vcast(x0, dtypes.bfloat16, mask=full)
                final_x1 = vcast(x1, dtypes.bfloat16, mask=full)
                final_x2 = vcast(x2, dtypes.bfloat16, mask=full)
                vstore_pack(
                    self.window, off, final_x0, full, pack_mode=PackMode.B32_TO_B16
                )
                vstore_pack(
                    self.window,
                    TILE + off,
                    final_x1,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    self.window,
                    2 * TILE + off,
                    final_x2,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
            vmem_bar("vst_vld")

    @jit
    def zero_sequence(self, local_batch: int):
        with vf(mode="simd"):
            full_bf16, _ = update_mask(TILE, elem_bits=16)
            zero = vdups(0.0, dtypes.bfloat16, mask=full_bf16)
            for token in range(self.seq_len):
                local_row = local_batch * self.seq_len + token
                vstore(self.output_bf16, local_row * TILE, zero, full_bf16)
            vmem_bar("vst_vld")


class _BaseProjection384(_Projection384Operations):
    """采用 MxN384 投影，为三个输出复用同一 hidden 数据面板。"""

    def __init__(self, rows=128, input_l2_cache_ctl=0, weight_l2_cache_ctl=0):
        self.rows = int(rows)
        self.allow_tail = False
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((self.rows, PROJECTION_KC)), depth=2
        )
        self.a_l1_shape = (self.rows, PROJECTION_KC)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((3 * TILE, PROJECTION_KC)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((self.rows, PROJECTION_KS)),
            depth=2,
        )
        self.l0a_shape = (self.rows, PROJECTION_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((TILE, PROJECTION_KS)), depth=2
        )
        self.l0c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((self.rows, TILE)), depth=1
        ).next_write()
        self.l0c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((self.rows, TILE)), depth=1
        ).next_write()
        self.l0c2 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((self.rows, TILE)), depth=1
        ).next_write()
        self.projection_half0 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.rows // 2, TILE)),
            depth=1,
        ).next_write()
        self.projection_half1 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.rows // 2, TILE)),
            depth=1,
        ).next_write()
        self.projection_half2 = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.rows // 2, TILE)),
            depth=1,
        ).next_write()
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((self.rows // 2, TILE))
        )
        self.projection_fp32: object


class _BaseProjection256(_Projection384Operations):
    """使用两个 N128 累加器，并考虑 M256 下的 L0C 容量边界。"""

    def __init__(self, rows, allow_tail=False):
        self.rows = int(rows)
        self.allow_tail = bool(allow_tail)
        self.input_l2_cache_ctl = 0
        self.weight_l2_cache_ctl = 4
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((rows, 256)), depth=2
        )
        self.a_l1_shape = (rows, 256)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((256, 256)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((rows, 64)), depth=2
        )
        self.l0a_shape = (rows, 64)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((128, 64)), depth=2
        )
        self.l0c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 128)), depth=1
        ).next_write()
        self.l0c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, 128)), depth=1
        ).next_write()
        self.half = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((rows // 2, 128)), depth=1
        )
        self.half_shape = (rows // 2, 128)
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((rows // 2, 128))
        )

    @jit
    def matmul_hidden2(self, x, w, group, macro):
        for kc in range(28):
            a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            b = self.b_l1.next_write()
            if const_expr(self.allow_tail):
                valid = x.shape[0] - group * self.rows
                if valid > self.rows:
                    valid = self.rows
                mem_copy(
                    a[0:valid, None],
                    x[
                        group * self.rows : group * self.rows + valid,
                        kc * 256 : (kc + 1) * 256,
                    ],
                    options=copy_options(l2_cache_ctl=0),
                )
            else:
                mem_copy(
                    a,
                    tile_slice(x, (self.rows, 256), (group, kc)),
                    options=copy_options(l2_cache_ctl=0),
                )
            mem_copy(
                b,
                tile_slice(w, (256, 256), (macro, kc)),
                options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
            )
            for ks in range(4):
                l0a = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                l0a_source = tile_slice(a, (self.rows, 64), (0, ks))
                l0a = l0a[tuple(map(slice, l0a_source.shape))]
                mem_copy(l0a, l0a_source)
                l0b = self.l0b.next_write()
                l0b_source = tile_slice(b, (128, 64), (0, ks))
                l0b = l0b[tuple(map(slice, l0b_source.shape))]
                mem_copy(l0b, l0b_source)
                matmul(self.l0c0, l0a, l0b, init=kc == 0 and ks == 0)
                l0b = self.l0b.next_write()
                l0b_source_2 = tile_slice(b, (128, 64), (1, ks))
                l0b = l0b[tuple(map(slice, l0b_source_2.shape))]
                mem_copy(l0b, l0b_source_2)
                matmul(self.l0c1, l0a, l0b, init=kc == 0 and ks == 0)

    @jit
    def acquire_result0(self):
        self.projection_fp32 = self.half.next_write().view(self.half_shape)
        mem_copy(self.projection_fp32, self.l0c0.partition(dim=0))

    @jit
    def acquire_result1(self):
        self.projection_fp32 = self.half.next_write().view(self.half_shape)
        mem_copy(self.projection_fp32, self.l0c1.partition(dim=0))


class BetaProjectionStage(_ProjectionOperations):
    def __init__(
        self,
        rows,
        input_l2_cache_ctl=0,
        weight_l2_cache_ctl=0,
        tile_width=16,
        row_capacity=128,
    ):
        self.row_count = int(rows)
        self.row_capacity = int(row_capacity)
        self.tile_width = int(tile_width)
        self.row_block = self.row_count // 2
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((rows, 128)), depth=2
        )
        self.a_l1_shape = (rows, 128)
        self.l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((rows, 64)), depth=2
        )
        self.l0a_shape = (rows, 64)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((tile_width, 64)), depth=2
        )
        self.l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((rows, tile_width)), depth=2
        )
        self.l0c_shape = (rows, tile_width)

    @jit
    def matmul_hidden(self, hidden_gm, weight_gm, output_tile, k_tiles, k_begin=0):
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        beta_l1 = Buffer(
            MemLoc.L1,
            weight_gm.dtype,
            weight_gm.layout,
            physical_layout=weight_gm.physical_layout,
        )
        mem_copy(
            beta_l1,
            weight_gm,
            options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
        )
        for kc_idx in range(k_tiles):
            a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                a,
                tile_slice(hidden_gm, (self.row_capacity, 128), (0, kc_idx + k_begin)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            for ks_idx in range(2):
                l0a = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                l0b = self.l0b.next_write()
                mem_copy(l0a, tile_slice(a, (self.row_capacity, 64), (0, ks_idx)))
                mem_copy(
                    l0b,
                    tile_slice(
                        beta_l1,
                        (self.tile_width, 64),
                        (0, (kc_idx + k_begin) * 2 + ks_idx),
                    ),
                )
                matmul(accumulator, l0a, l0b, init=kc_idx == 0 and ks_idx == 0)


class FaResident(BetaProjectionStage):
    @jit
    def matmul_hidden(self, hidden_gm, weight_gm, output_tile, k_tiles, k_begin=0):
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        wb = Buffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((self.tile_width, HIDDEN_SIZE))
        )
        mem_copy(wb, weight_gm, options=copy_options(l2_cache_ctl=0))
        for kc in range(k_tiles):
            a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(a, tile_slice(hidden_gm, (self.row_count, 128), (0, kc + k_begin)))
            for ks in range(2):
                la = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                lb = self.l0b.next_write()
                mem_copy(la, tile_slice(a, (self.row_count, 64), (0, ks)))
                mem_copy(
                    lb,
                    tile_slice(wb, (self.tile_width, 64), (0, (kc + k_begin) * 2 + ks)),
                )
                matmul(accumulator, la, lb, init=kc == 0 and ks == 0)


class _Core32Projection384(_Core32ProjectionOperations):
    """逻辑行数为 M32/M64/M128，片上容量固定为 M128。"""

    def __init__(self, rows=128, input_l2_cache_ctl=0, weight_l2_cache_ctl=0):
        self.rows = rows
        self.allow_tail = False
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((128 * PROJECTION_KC,)),
            depth=2,
            data_format="nd",
        )
        self.a_l1_shape = (self.rows, PROJECTION_KC)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((3 * TILE, PROJECTION_KC)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((128 * PROJECTION_KS,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (self.rows, PROJECTION_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((TILE, PROJECTION_KS)), depth=2
        )
        self.l0c0 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.rows, TILE)),
            data_format="nz",
        )
        self.l0c1 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.rows, TILE)),
            data_format="nz",
        )
        self.l0c2 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.rows, TILE)),
            data_format="nz",
        )
        self.projection_half0 = (
            RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout(((128 // 2) * TILE,)), depth=1
            )
            .next_write()
            .view((self.rows // 2, TILE))
        )
        self.projection_half1 = (
            RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout(((128 // 2) * TILE,)), depth=1
            )
            .next_write()
            .view((self.rows // 2, TILE))
        )
        self.projection_half2 = (
            RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout(((128 // 2) * TILE,)), depth=1
            )
            .next_write()
            .view((self.rows // 2, TILE))
        )
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((128 // 2 * TILE,))
        ).view(self.rows // 2, TILE)
        self.projection_fp32: object


class _Core32Projection256(_Core32ProjectionOperations):
    """逻辑行数为 M32/M64/M128/M256，容量固定为 M256，并对尾部使用掩码。"""

    def __init__(self, rows, allow_tail=False):
        self.rows = rows
        self.allow_tail = bool(allow_tail)
        self.input_l2_cache_ctl = 0
        self.weight_l2_cache_ctl = 4
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((256 * 256,)),
            depth=2,
            data_format="nd",
        )
        self.a_l1_shape = (rows, 256)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((256, 256)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((256 * 64,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (rows, 64)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((128, 64)), depth=2
        )
        self.l0c0 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((256 * 128,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((rows, 128)),
            data_format="nz",
        )
        self.l0c1 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((256 * 128,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((rows, 128)),
            data_format="nz",
        )
        self.half = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout(((256 // 2) * 128,)), depth=1
        )
        self.half_shape = (rows // 2, 128)
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((256 // 2 * 128,))
        ).view(rows // 2, 128)

    @jit
    def acquire_result0(self):
        self.projection_fp32 = self.half.next_write().view(self.half_shape)
        mem_copy(self.projection_fp32, self.l0c0.partition(dim=0))

    @jit
    def acquire_result1(self):
        self.projection_fp32 = self.half.next_write().view(self.half_shape)
        mem_copy(self.projection_fp32, self.l0c1.partition(dim=0))


class DynamicBetaStage(BetaProjectionStage):
    @jit
    def matmul_hidden(self, hidden_gm, weight_gm, output_tile, k_tiles, k_begin=0):
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        beta_l1 = Buffer(
            MemLoc.L1,
            weight_gm.dtype,
            weight_gm.layout,
            physical_layout=weight_gm.physical_layout,
        )
        mem_copy(
            beta_l1,
            weight_gm,
            options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
        )
        for kc_idx in range(k_tiles):
            a = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                a,
                tile_slice(hidden_gm, (self.row_capacity, 128), (0, kc_idx + k_begin)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            for ks_idx in range(2):
                l0a = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                l0b = self.l0b.next_write()
                mem_copy(l0a, tile_slice(a, (self.row_capacity, 64), (0, ks_idx)))
                mem_copy(
                    l0b,
                    tile_slice(
                        beta_l1,
                        (self.tile_width, 64),
                        (0, (kc_idx + k_begin) * 2 + ks_idx),
                    ),
                )
                matmul(accumulator, l0a, l0b, init=kc_idx == 0 and ks_idx == 0)


class N384ProjectionStage(_DynamicProjectionOperations):
    def __init__(self, rows=128, input_l2_cache_ctl=0, weight_l2_cache_ctl=0):
        self.rows = rows
        self.allow_tail = False
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((128 * PROJECTION_KC,)),
            depth=2,
            data_format="nd",
        )
        self.a_l1_shape = (self.rows, PROJECTION_KC)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((3 * TILE, PROJECTION_KC)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((128 * PROJECTION_KS,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (self.rows, PROJECTION_KS)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((TILE, PROJECTION_KS)), depth=2
        )
        self.l0c0 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.rows, TILE)),
            data_format="nz",
        )
        self.l0c1 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.rows, TILE)),
            data_format="nz",
        )
        self.l0c2 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * TILE,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((self.rows, TILE)),
            data_format="nz",
        )
        self.projection_half0 = (
            RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout(((128 // 2) * TILE,)), depth=1
            )
            .next_write()
            .view((self.rows // 2, TILE))
        )
        self.projection_half1 = (
            RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout(((128 // 2) * TILE,)), depth=1
            )
            .next_write()
            .view((self.rows // 2, TILE))
        )
        self.projection_half2 = (
            RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout(((128 // 2) * TILE,)), depth=1
            )
            .next_write()
            .view((self.rows // 2, TILE))
        )
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((128 // 2 * TILE,))
        ).view(self.rows // 2, TILE)
        self.projection_fp32: object


class N256ProjectionStage(_DynamicProjectionOperations):
    def __init__(self, rows, allow_tail=False):
        self.rows = rows
        self.allow_tail = bool(allow_tail)
        self.input_l2_cache_ctl = 0
        self.weight_l2_cache_ctl = 4
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((256 * 256,)),
            depth=2,
            data_format="nd",
        )
        self.a_l1_shape = (rows, 256)
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((256, 256)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((256 * 64,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (rows, 64)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((128, 64)), depth=2
        )
        self.l0c0 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((256 * 128,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((rows, 128)),
            data_format="nz",
        )
        self.l0c1 = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((256 * 128,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((rows, 128)),
            data_format="nz",
        )
        self.half = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout(((256 // 2) * 128,)), depth=1
        )
        self.half_shape = (rows // 2, 128)
        self.projection_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((256 // 2 * 128,))
        ).view(rows // 2, 128)

    @jit
    def acquire_result0(self):
        self.projection_fp32 = self.half.next_write().view(self.half_shape)
        mem_copy(self.projection_fp32, self.l0c0.partition(dim=0))

    @jit
    def acquire_result1(self):
        self.projection_fp32 = self.half.next_write().view(self.half_shape)
        mem_copy(self.projection_fp32, self.l0c1.partition(dim=0))


class S4ProjectionStage(ProjectionStage):
    """采用 N128 投影，逻辑行数为 M16～M128，片上容量固定。

    仅对逻辑有效行执行读取、累加和发布。较大的分配空间表示编译器容量，
    不表示对 hidden 输入填充，也不会扩大 GM 搬运范围。
    """

    def __init__(self, rows, width=128):
        super().__init__(rows, 0, 0, tile_width=width, row_capacity=128)

    @jit
    def matmul_hidden(self, hidden_gm, weight_gm, output_tile, k_tiles):
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        for kc in range(k_tiles * TILE // PROJECTION_KC):
            left = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                left,
                tile_slice(
                    hidden_gm,
                    make_tiler((self.row_count, PROJECTION_KC), alignment=(16, 16)),
                    (0, kc),
                ),
            )
            right = self.b_l1.next_write()
            _copy_weight_with_policy(
                right,
                tile_slice(
                    weight_gm, (self.tile_width, PROJECTION_KC), (output_tile, kc)
                ),
                self.weight_l2_cache_ctl,
            )
            self.accumulate_panel(accumulator, left, right, 0, kc == 0)

    @jit
    def accumulate_panel(self, accumulator, left, right, first_k, init):
        for ks in range(PROJECTION_KC // PROJECTION_KS):
            a = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            b = self.l0b.next_write()
            mem_copy(
                a,
                tile_slice(left, (self.row_capacity, PROJECTION_KS), (0, ks))[
                    0 : self.row_count, None
                ],
            )
            mem_copy(
                b,
                tile_slice(right, (self.tile_width, PROJECTION_KS), (0, first_k + ks)),
            )
            matmul(accumulator, a, b, init=init and ks == 0)

    @jit
    def matmul_resident(self, resident_l1, weight_l1):
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        for ks in range(TILE // PROJECTION_KS):
            a = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            b = self.l0b.next_write()
            mem_copy(
                a,
                tile_slice(resident_l1, (self.row_capacity, PROJECTION_KS), (0, ks))[
                    0 : self.row_count, None
                ],
            )
            mem_copy(b, tile_slice(weight_l1, (TILE, PROJECTION_KS), (0, ks)))
            matmul(accumulator, a, b, init=ks == 0)


class S4BetaProjectionStage(S4ProjectionStage):
    """保留完整的 NZ beta 分配及其 head 维度的物理填充。"""

    def __init__(self, rows, width):
        # K128 数据面板使完整的 N32 NZ 权重和两个 A 槽位
        # 能放入与现有 DynamicBetaStage 相同的 L1 容量预算。
        self.row_count = rows
        self.row_capacity = 128
        self.tile_width = width
        self.row_block = rows // 2
        self.weight_l2_cache_ctl = 0
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((128 * 128,)),
            depth=2,
            data_format="nd",
        )
        self.a_l1_shape = (rows, 128)
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((128 * 64,)),
            depth=2,
            data_format="nd",
        )
        self.l0a_shape = (rows, 64)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((width, 64)), depth=2
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((128 * width,)),
            depth=2,
            data_format="nd",
        )
        self.l0c_shape = (rows, width)

    @jit
    def matmul_hidden(self, hidden_gm, weight_gm, output_tile, k_tiles):
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        resident = Buffer(
            MemLoc.L1,
            weight_gm.dtype,
            weight_gm.layout,
            physical_layout=weight_gm.physical_layout,
        )
        _copy_weight_with_policy(resident, weight_gm, self.weight_l2_cache_ctl)
        for kc in range(k_tiles):
            left = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                left,
                tile_slice(
                    hidden_gm,
                    make_tiler((self.row_count, 128), alignment=(16, 16)),
                    (0, kc),
                ),
            )
            for ks in range(2):
                a = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                b = self.l0b.next_write()
                mem_copy(
                    a,
                    tile_slice(left, (self.row_capacity, 64), (0, ks))[
                        0 : self.row_count, None
                    ],
                )
                mem_copy(
                    b, tile_slice(resident, (self.tile_width, 64), (0, kc * 2 + ks))
                )
                matmul(accumulator, a, b, init=kc == 0 and ks == 0)


class S4LowRankStage(LowRankDecayStage):
    """融合 Fa/Fb，严格限定逻辑行范围，并保留 Fa 到 BF16 的舍入。"""

    @jit
    def project(self, decay, x, wa, wb, k_tiles, subblock):
        stage = self.projection_stage
        stage.matmul_hidden(x, wa, 0, k_tiles)
        result = from_tensor(
            stage.l0c.next_read(), layout=make_layout(stage.l0c_shape), data_format="nz"
        )
        mem_copy(self.fa_l1, result)
        for head in range(self.num_heads):
            _copy_weight_with_policy(
                self.fb_l1,
                tile_slice(wb, (TILE, TILE), (head, 0)),
                stage.weight_l2_cache_ctl,
            )
            stage.matmul_resident(self.fa_l1, self.fb_l1)
            stage.publish_result(
                decay[0 : stage.row_count, head * TILE : (head + 1) * TILE]
            )
            vec_sync_all()


class TP4S4AuxiliarySchedule:
    """计算 Beta 并融合 Fa/Fb，避免 beta 重排和 Fa 经 GM 中转。"""

    def __init__(self, heads):
        self.heads = heads
        self.beta_width = 32

    @jit
    def beta_panel(self, beta, x, weight, group, rows, policy):
        onchip_rewind(reset_sync_id=True)
        stage = S4BetaProjectionStage(rows, self.beta_width)
        stage.weight_l2_cache_ctl = policy
        hidden = tile_slice(
            x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
        )
        target = tile_slice(
            beta, make_tiler((rows, self.beta_width), alignment=(16, 16)), (group, 0)
        )
        stage.project_to_gm(hidden, weight, target, 0, 56)
        vec_sync_all()

    @jit
    def decay_panel(self, decay, x, wa, wb, group, rows, policy):
        onchip_rewind(reset_sync_id=True)
        stage = S4ProjectionStage(rows)
        stage.weight_l2_cache_ctl = policy
        low_rank = S4LowRankStage(stage, self.heads)
        low_rank.project(
            tile_slice(
                decay,
                make_tiler((rows, self.heads * TILE), alignment=(16, 16)),
                (group, 0),
            ),
            tile_slice(
                x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
            ),
            wa,
            wb,
            56,
            get_subblock_id(),
        )
        vec_sync_all()

    @jit
    def execute(self, decay, beta, fa, x, wa, wb, wbeta, rows, policy):
        for group in range(get_block_idx(), x.shape[0] // rows, get_block_num()):
            self.beta_panel(beta, x, wbeta, group, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        for group in range(get_block_idx(), x.shape[0] // rows, get_block_num()):
            self.decay_panel(decay, x, wa, wb, group, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)


class SequenceProjectionStage:
    """共用 Cube 投影，支持 BF16 输出和 FP32 原始 QKV。"""

    def __init__(self):
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((128 * 128,)),
            depth=2,
            data_format="nd",
        )
        self.b_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((128, 128)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((128 * 128,)),
            depth=2,
            data_format="nd",
        )
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((128, 128)), depth=2
        )
        self.l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((128 * 128,)),
            depth=1,
            data_format="nd",
        )

    @jit
    def project(self, out, x, weight):
        columns = weight.shape[0] // 128
        for task in range(
            get_block_idx(), x.shape[0] // 128 * columns, get_block_num()
        ):
            row = task // columns
            column = task % columns
            c = from_tensor(
                self.l0c.next_write(), layout=make_layout((128, 128)), data_format="nz"
            )
            for ki in range(x.shape[1] // 128):
                a1 = from_tensor(
                    self.a_l1.next_write(),
                    layout=make_layout((128, 128)),
                    data_format="nz",
                )
                b1 = self.b_l1.next_write()
                mem_copy(a1, tile_slice(x, (128, 128), (row, ki)))
                mem_copy(b1, tile_slice(weight, (128, 128), (column, ki)))
                a0 = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout((128, 128)),
                    data_format="nz",
                )
                b0 = self.l0b.next_write()
                mem_copy(a0, a1)
                mem_copy(b0, b1)
                matmul(c, a0, b0, init=ki == 0)
            mem_copy(tile_slice(out, (128, 128), (row, column)), c)


class TwofoldQKVDotStage:
    """使用 TwoSum 残差进行补偿投影累加。"""

    @kernel
    def kernel(self, out, x, weight):
        onchip_rewind(reset_sync_id=True)
        wx = Buffer(MemLoc.UB, dtypes.float32, make_layout((32, 128)))
        xx = Buffer(MemLoc.UB, dtypes.float32, make_layout((32,)))
        carry = Buffer(MemLoc.UB, dtypes.float32, make_layout((2, 128)))
        result = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 128)))
        columns = out.shape[1] // 128
        for task in range(get_block_idx(), x.shape[0] * columns, get_block_num()):
            row = task // columns
            column = task % columns
            with vf(mode="simd"):
                mask1, _ = update_mask(64, elem_bits=32)
                zero = vdups(0.0, dtypes.float32)
                for segment in range(2):
                    vstore(carry, segment * 64, zero, mask1)
                    vstore(carry, 128 + segment * 64, zero, mask1)
            for chunk in range(x.shape[1] // 32):
                mem_copy(xx, x[row, chunk * 32 : (chunk + 1) * 32].view(32))
                mem_copy(wx, tile_slice(weight, (32, 128), (chunk, column)))
                with vf(mode="simd"):
                    mask2, _ = update_mask(64, elem_bits=32)
                    for segment in range(2):
                        off = segment * 64
                        hi = vload(carry, off)
                        lo = vload(carry, 128 + off)
                        for k in range(32):
                            a = vload_broadcast(xx, k)
                            b = vload(wx, k * 128 + off)
                            product = vmul(a, b, mask=mask2)
                            summed = vadd(hi, product, mask=mask2)
                            bp = vsub(summed, hi, mask=mask2)
                            residual = vadd(
                                vsub(hi, vsub(summed, bp, mask=mask2), mask=mask2),
                                vsub(product, bp, mask=mask2),
                                mask=mask2,
                            )
                            lo = vadd(lo, residual, mask=mask2)
                            hi = summed
                        vstore(carry, off, hi, mask2)
                        vstore(carry, 128 + off, lo, mask2)
            with vf(mode="simd"):
                mask3, _ = update_mask(64, elem_bits=32)
                for segment in range(2):
                    off = segment * 64
                    total = vadd(vload(carry, off), vload(carry, 128 + off), mask=mask3)
                    vstore(result, off, total, mask3)
            mem_copy(tile_slice(out, (1, 128), (row, column)), result)


class TwofoldQKVKernel:
    @host
    def run(self, out, x, weight, cores):
        stage = TwofoldQKVDotStage()
        stage.kernel[2 * cores](out, x, weight)


_TWOFOLD_COMPILED = {}


def _get_twofold_compiled(width):
    if width not in _TWOFOLD_COMPILED:
        rows = cannbotdsl.Dim("twofold_rows", min=1)
        outrows = cannbotdsl.Dim("twofold_padded_rows", min=128, multiple_of=128)
        _TWOFOLD_COMPILED[width] = cannbotdsl.compile(
            TwofoldQKVKernel().run,
            TensorSpec((outrows, width), dtypes.float32),
            TensorSpec((rows, HIDDEN_SIZE), dtypes.float32),
            TensorSpec((HIDDEN_SIZE, width), dtypes.float32),
            dtypes.int64,
        )
    return _TWOFOLD_COMPILED[width]


def _twofold_qkv_projection(hidden, weight, raw, cores):
    import torch_npu

    inputs = hidden.reshape(-1, HIDDEN_SIZE).float()
    weights = torch_npu.npu_format_cast(weight, 2).transpose(0, 1).contiguous().float()
    width = weight.shape[0]
    _get_twofold_compiled(width)
    _TWOFOLD_COMPILED[width](raw, inputs, weights, cores)
    return (inputs, weights, raw)


_AUXILIARY_COMPILED = {}


def _get_auxiliary_compiled(reduction, width):
    key = (reduction, width)
    if key not in _AUXILIARY_COMPILED:
        rows = cannbotdsl.Dim("auxiliary_rows", min=1)
        _AUXILIARY_COMPILED[key] = cannbotdsl.compile(
            TwofoldQKVKernel().run,
            TensorSpec((rows, width), dtypes.float32),
            TensorSpec((rows, reduction), dtypes.float32),
            TensorSpec((reduction, width), dtypes.float32),
            dtypes.int64,
        )
    return _AUXILIARY_COMPILED[key]


def _auxiliary_twofold_projection(x, weight, cores):
    import torch_npu

    reduction = int(weight.shape[1])
    width = int(weight.shape[0])
    operands = x.reshape(-1, reduction).float()
    weights = torch_npu.npu_format_cast(weight, 2).transpose(0, 1).contiguous().float()
    projected = torch.empty(
        (operands.shape[0], width), dtype=torch.float32, device=x.device
    )
    key = (reduction, width)
    _get_auxiliary_compiled(reduction, width)
    _AUXILIARY_COMPILED[key](projected, operands, weights, cores)
    # Fa 和 Fb 分别保留原始 BF16 舍入边界。
    rounded = projected.to(torch.bfloat16)
    return rounded, (operands, weights, projected, rounded)


def _accurate_auxiliary_projection(hidden, weight_a, weight_b, cores):
    fa, keep_a = _auxiliary_twofold_projection(hidden, weight_a, cores)
    fb, keep_b = _auxiliary_twofold_projection(fa, weight_b, cores)
    return fa, fb, (keep_a, keep_b)


def prepare_optimized_precision(hidden, arguments, heads, cores):
    """为 S4/S8 计算补偿投影，保持 Fa、Fb 各自的 BF16 边界。"""
    rows = hidden.shape[0]
    padded_rows = (rows + TILE - 1) // TILE * TILE
    raw = torch.empty(
        (padded_rows, 3 * heads * TILE), dtype=torch.float32, device=hidden.device
    )
    qkv_keep = _twofold_qkv_projection(
        hidden, arguments["qkv_projection_weight"], raw, cores
    )
    _, decay, decay_keep = _accurate_auxiliary_projection(
        hidden,
        arguments["decay_projection_a_weight"],
        arguments["decay_projection_b_weight"],
        cores,
    )
    return raw[:rows], decay, (qkv_keep, decay_keep)


class SequenceConvStage:
    """使用调用方提供的 PA 状态步长执行变长 Conv/SiLU。"""

    def __init__(self):
        self.history = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((3, 128)))
        self.weight = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((4, 128)))
        self.raw = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 128)))
        self.rounded = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 128)))
        self.out = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, 128)))

    @jit
    def convolve(self, out, raw, conv, indices, accepted, cu, weight_gm, conv_stride0):
        width = raw.shape[1]
        channels = width // 128
        for item in range(
            get_block_idx(), indices.shape[0] * channels, get_block_num()
        ):
            batch = item // channels
            channel = item % channels
            slot = dtypes.int64(indices[batch])
            count = dtypes.int64(accepted[batch])
            start = dtypes.int64(cu[batch])
            end = dtypes.int64(cu[batch + 1])
            history_offset = count - 1 if count > 0 else 0
            base = slot * conv_stride0 + channel * 128
            history_begin = slot * conv_stride0 + history_offset * width
            history_gm = conv[history_begin : history_begin + 3 * width,].view(
                3, width
            )[None, channel * 128 : (channel + 1) * 128]
            mem_copy(self.history, history_gm)
            mem_copy(self.weight, tile_slice(weight_gm, (4, 128), (0, channel)))
            mem_copy(
                conv[slot * conv_stride0 : slot * conv_stride0 + 2 * width,].view(
                    2, width
                )[None, channel * 128 : (channel + 1) * 128],
                self.history[1:3, None],
            )
            for token in range(start, end):
                mem_copy(self.raw, tile_slice(raw, (1, 128), (token, channel)))
                with vf(mode="simd"):
                    mask, _ = update_mask(64, elem_bits=32)
                    for segment in range(2):
                        off = segment * 64
                        h0 = vcast(
                            vload_unpack(
                                self.history, off, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        h1 = vcast(
                            vload_unpack(
                                self.history,
                                128 + off,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        h2 = vcast(
                            vload_unpack(
                                self.history,
                                256 + off,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        x = vload(self.raw, off)
                        w0 = vcast(
                            vload_unpack(
                                self.weight, off, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        w1 = vcast(
                            vload_unpack(
                                self.weight,
                                128 + off,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        w2 = vcast(
                            vload_unpack(
                                self.weight,
                                256 + off,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        w3 = vcast(
                            vload_unpack(
                                self.weight,
                                384 + off,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        acc = vadd(
                            vmadd(h0, w0, vmul(h1, w1, mask=mask), mask=mask),
                            vmadd(h2, w2, vmul(x, w3, mask=mask), mask=mask),
                            mask=mask,
                        )
                        silu = vdiv(
                            acc,
                            vadds(
                                vexp(vneg(acc, mask=mask), mask=mask), 1.0, mask=mask
                            ),
                            mask=mask,
                        )
                        # 小参数采用补偿计算，其余范围保留原 SiLU 结果。
                        regular_activated = silu
                        hi = vdups(0.0, dtypes.float32)
                        lo = vdups(0.0, dtypes.float32)
                        product = vmul(h0, w0, mask=mask)
                        negative_product = vneg(product, mask=mask)
                        product_error = vmadd(h0, w0, negative_product, mask=mask)
                        total = vadd(hi, product, mask=mask)
                        recovered = vsub(total, hi, mask=mask)
                        prior = vsub(total, recovered, mask=mask)
                        prior_error = vsub(hi, prior, mask=mask)
                        new_error = vsub(product, recovered, mask=mask)
                        sum_error = vadd(prior_error, new_error, mask=mask)
                        term_error = vadd(sum_error, product_error, mask=mask)
                        lo = vadd(lo, term_error, mask=mask)
                        hi = total
                        product = vmul(h1, w1, mask=mask)
                        negative_product = vneg(product, mask=mask)
                        product_error = vmadd(h1, w1, negative_product, mask=mask)
                        total = vadd(hi, product, mask=mask)
                        recovered = vsub(total, hi, mask=mask)
                        prior = vsub(total, recovered, mask=mask)
                        prior_error = vsub(hi, prior, mask=mask)
                        new_error = vsub(product, recovered, mask=mask)
                        sum_error = vadd(prior_error, new_error, mask=mask)
                        term_error = vadd(sum_error, product_error, mask=mask)
                        lo = vadd(lo, term_error, mask=mask)
                        hi = total
                        product = vmul(h2, w2, mask=mask)
                        negative_product = vneg(product, mask=mask)
                        product_error = vmadd(h2, w2, negative_product, mask=mask)
                        total = vadd(hi, product, mask=mask)
                        recovered = vsub(total, hi, mask=mask)
                        prior = vsub(total, recovered, mask=mask)
                        prior_error = vsub(hi, prior, mask=mask)
                        new_error = vsub(product, recovered, mask=mask)
                        sum_error = vadd(prior_error, new_error, mask=mask)
                        term_error = vadd(sum_error, product_error, mask=mask)
                        lo = vadd(lo, term_error, mask=mask)
                        hi = total
                        product = vmul(x, w3, mask=mask)
                        negative_product = vneg(product, mask=mask)
                        product_error = vmadd(x, w3, negative_product, mask=mask)
                        total = vadd(hi, product, mask=mask)
                        recovered = vsub(total, hi, mask=mask)
                        prior = vsub(total, recovered, mask=mask)
                        prior_error = vsub(hi, prior, mask=mask)
                        new_error = vsub(product, recovered, mask=mask)
                        sum_error = vadd(prior_error, new_error, mask=mask)
                        term_error = vadd(sum_error, product_error, mask=mask)
                        lo = vadd(lo, term_error, mask=mask)
                        hi = total
                        total = vadd(hi, lo, mask=mask)
                        recovered = vsub(total, hi, mask=mask)
                        prior = vsub(total, recovered, mask=mask)
                        prior_error = vsub(hi, prior, mask=mask)
                        new_error = vsub(lo, recovered, mask=mask)
                        lo = vadd(prior_error, new_error, mask=mask)
                        hi = total
                        absolute_hi = vabs(hi, mask=mask)
                        small_limit = vdups(0.0078125, dtypes.float32)
                        small = vle(absolute_hi, small_limit, mask=mask)
                        zero = vdups(0.0, dtypes.float32)
                        hi = vselect(hi, zero, cond_mask=small)
                        lo = vselect(lo, zero, cond_mask=small)
                        # SiLU = x/2 + x^2/4 - x^4/48 + x^6/480 + O(x^8).
                        # 保留保护位到 FP32 舍入，再按原边界转换到 BF16。
                        square = vmul(hi, hi, mask=mask)
                        negative_square = vneg(square, mask=mask)
                        square_error = vmadd(hi, hi, negative_square, mask=mask)
                        half = vmuls(hi, 0.5, mask=mask)
                        quarter_square = vmuls(square, 0.25, mask=mask)
                        major = vadd(half, quarter_square, mask=mask)
                        recovered = vsub(major, half, mask=mask)
                        prior = vsub(major, recovered, mask=mask)
                        prior_error = vsub(half, prior, mask=mask)
                        new_error = vsub(quarter_square, recovered, mask=mask)
                        major_error = vadd(prior_error, new_error, mask=mask)
                        quartic = vmul(square, square, mask=mask)
                        sixth = vmul(quartic, square, mask=mask)
                        fourth_term = vmuls(quartic, -0.020833333333333332, mask=mask)
                        sixth_term = vmuls(sixth, 0.0020833333333333333, mask=mask)
                        tail = vadd(fourth_term, sixth_term, mask=mask)
                        half_hi = vmuls(hi, 0.5, mask=mask)
                        derivative_main = vadds(half_hi, 0.5, mask=mask)
                        cubic = vmul(hi, square, mask=mask)
                        derivative_tail = vmuls(cubic, 0.08333333333333333, mask=mask)
                        derivative = vsub(derivative_main, derivative_tail, mask=mask)
                        square_correction = vmuls(square_error, 0.25, mask=mask)
                        low_correction = vmul(derivative, lo, mask=mask)
                        correction = vadd(square_correction, low_correction, mask=mask)
                        correction = vadd(correction, tail, mask=mask)
                        correction = vadd(major_error, correction, mask=mask)
                        accurate_activated = vadd(major, correction, mask=mask)
                        silu = vselect(
                            accurate_activated, regular_activated, cond_mask=small
                        )
                        xb = vcast(x, dtypes.bfloat16, mask=mask)
                        vstore_pack(
                            self.out,
                            off,
                            vcast(silu, dtypes.bfloat16, mask=mask),
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                        vstore_pack(
                            self.rounded, off, xb, mask, pack_mode=PackMode.B32_TO_B16
                        )
                        vstore_pack(
                            self.history,
                            off,
                            vcast(h1, dtypes.bfloat16, mask=mask),
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                        vstore_pack(
                            self.history,
                            128 + off,
                            vcast(h2, dtypes.bfloat16, mask=mask),
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                        vstore_pack(
                            self.history,
                            256 + off,
                            xb,
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    vmem_bar("vst_vld")
                mem_copy(tile_slice(out, (1, 128), (token, channel)), self.out)
                target_begin = base + (token - start + 2) * width
                mem_copy(
                    conv[target_begin : target_begin + 128,].view(1, 128), self.rounded
                )


class TP4GridScheduleFrontendOperations:
    @jit
    def _run_qkvg_task(
        self,
        projection_stage: ProjectionStage,
        conv_stage: ConvSiluStage,
        hidden_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        qkv_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        projection_kind,
        projection_head,
        batch_start,
        batch_count,
        subblock_idx,
        k_tiles,
    ):
        """执行一个完整的 QKV+Conv 任务或门控投影任务。"""
        if projection_kind == 0:
            projection_stage.matmul_hidden(
                hidden_gm, qkv_weight_gm, projection_head, k_tiles
            )
            batches_per_subblock = batch_count // 2
            selected_history = Buffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((batches_per_subblock * (CONV_KERNEL - 1), TILE)),
            )
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    projection = candidate_head // self.num_heads
                    value_head = candidate_head - projection * self.num_heads
                    conv_stage.load_weight(
                        tile_slice(
                            conv_weight_gm[projection, value_head, None, None],
                            (CONV_KERNEL, TILE),
                            (0, 0),
                        )
                    )
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                        history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                        if cache_index >= 0:
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for history_row in range(CONV_KERNEL - 1):
                                mem_copy(
                                    tile_slice(
                                        selected_history,
                                        (1, TILE),
                                        (
                                            local_batch * (CONV_KERNEL - 1)
                                            + history_row,
                                            0,
                                        ),
                                    ),
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (history_offset + history_row, candidate_head),
                                    ),
                                )
            projection_stage.acquire_result()
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index < 0:
                            conv_stage.zero_sequence(local_batch)
                        else:
                            history_view = tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            )
                            conv_stage.install_history(history_view)
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for cache_row in range(2):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_row, candidate_head),
                                    ),
                                    tile_slice(
                                        history_view, (1, TILE), (cache_row + 1, 0)
                                    ),
                                )
                            conv_stage.load_accurate_sequence(
                                conv_state_gm, batch_index, candidate_head
                            )
                            conv_stage.convolve_sequence(
                                local_batch,
                                projection_stage.projection_fp32,
                                tile_slice(
                                    conv_stage.weight, (CONV_KERNEL, TILE), (0, 0)
                                ),
                            )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index >= 0:
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for cache_token in range(self.seq_len):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_token + 2, candidate_head),
                                    ),
                                    tile_slice(
                                        conv_stage.raw_bf16,
                                        (1, TILE),
                                        (local_batch * self.seq_len + cache_token, 0),
                                    ),
                                )
            vec_sync_all()
            if projection_head < self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        q_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            elif projection_head < 2 * self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        k_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            else:
                projection_stage.publish_source(
                    tile_slice(
                        v_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - 2 * self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
        else:
            projection_stage.project(
                hidden_gm, output_gate_weight_gm, projection_head, k_tiles
            )
            projection_stage.publish(
                tile_slice(
                    output_gate_workspace_gm,
                    (projection_stage.row_capacity, TILE),
                    (0, projection_head),
                ),
                subblock_idx,
            )
        vec_sync_all()

    @jit
    def _prefetch_n384_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """投影进行时，预取一个数据面板对应的 Conv 元数据。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                if cache_index >= 0:
                    state_view = _conv_slot(conv_state_gm, cache_index)
                    mem_copy(
                        tile_slice(
                            selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                        ),
                        tile_slice(
                            state_view[
                                history_offset : history_offset + CONV_KERNEL - 1, None
                            ],
                            (CONV_KERNEL - 1, TILE),
                            (0, logical_n),
                        ),
                    )

    @jit
    def _consume_n384_panel(
        self,
        projection_stage: _ResidentN384ProjectionStage,
        conv_stage: ConvSiluStage,
        selected_history,
        weight_ub,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
        panel,
    ):
        """处理一个驻留的 N128 累加器，同时让 Cube 保留另外两个。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                if cache_index < 0:
                    conv_stage.zero_sequence(local_batch)
                else:
                    history_view = tile_slice(
                        selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                    )
                    conv_stage.install_history(history_view)
                    state_view = _conv_slot(conv_state_gm, cache_index)
                    mem_copy(
                        tile_slice(state_view[0:2, None], (2, TILE), (0, logical_n)),
                        tile_slice(history_view[1:3, None], (2, TILE), (0, 0)),
                    )
                    conv_stage.load_accurate_sequence(
                        conv_state_gm, batch_index, logical_n
                    )
                    conv_stage.convolve_sequence(
                        local_batch, projection_stage.projection_fp32, weight_ub
                    )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    mem_copy(
                        tile_slice(
                            state_view[2 : 2 + self.seq_len, None],
                            (self.seq_len, TILE),
                            (0, logical_n),
                        ),
                        tile_slice(
                            conv_stage.raw_bf16, (self.seq_len, TILE), (local_batch, 0)
                        ),
                    )
            vec_sync_all()
            if projection == 0:
                projection_stage.publish(
                    q_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            elif projection == 1:
                projection_stage.publish(
                    k_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            else:
                projection_stage.publish(
                    v_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
        else:
            gate_head = logical_n - 3 * self.num_heads
            projection_stage.cast_result()
            vec_sync_all()
            projection_stage.publish(
                output_gate_workspace_gm,
                m_group,
                gate_head,
                subblock_idx,
                projection_stage.projection_bf16,
            )
        vec_sync_all()

    @jit
    def _consume_n384_triplet(
        self,
        projection_stage: _ResidentN384ProjectionStage,
        conv_stage: ConvSiluStage,
        selected_history,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
    ):
        history0 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (0, 0)
        )
        history1 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (1, 0)
        )
        history2 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (2, 0)
        )
        weight0 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (0, 0))
        weight1 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (1, 0))
        weight2 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (2, 0))
        self._prefetch_n384_panel(
            conv_stage,
            history0,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            0,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history1,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            1,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history2,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            2,
        )
        vec_sync_all()
        projection_stage.acquire_result0()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history0,
            weight0,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            0,
        )
        projection_stage.acquire_result1()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history1,
            weight1,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            1,
        )
        projection_stage.acquire_result2()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history2,
            weight2,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            2,
        )


class TPScheduleFrontendOperations:
    @jit
    def _consume_n384_panel(
        self,
        projection_stage: _BaseProjection384,
        conv_stage: ConvSiluStage,
        selected_history,
        weight_ub,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
        panel,
    ):
        """处理一个驻留的 N128 累加器，同时让 Cube 保留另外两个。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                if batch_index < conv_state_indices_gm.shape[0]:
                    cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                    if cache_index < 0:
                        conv_stage.zero_sequence(local_batch)
                    else:
                        history_view = tile_slice(
                            selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                        )
                        conv_stage.install_history(history_view)
                        state_view = _conv_slot(conv_state_gm, cache_index)
                        mem_copy(
                            tile_slice(
                                state_view[0:2, None], (2, TILE), (0, logical_n)
                            ),
                            tile_slice(history_view[1:3, None], (2, TILE), (0, 0)),
                        )
                        conv_stage.load_accurate_sequence(
                            conv_state_gm, batch_index, logical_n
                        )
                        conv_stage.convolve_sequence(
                            local_batch, projection_stage.projection_fp32, weight_ub
                        )
                        vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        mem_copy(
                            tile_slice(
                                state_view[2 : 2 + self.seq_len, None],
                                (self.seq_len, TILE),
                                (0, logical_n),
                            ),
                            tile_slice(
                                conv_stage.raw_bf16,
                                (self.seq_len, TILE),
                                (local_batch, 0),
                            ),
                        )
            vec_sync_all()
            if projection == 0:
                projection_stage.publish(
                    q_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            elif projection == 1:
                projection_stage.publish(
                    k_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            else:
                projection_stage.publish(
                    v_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
        else:
            gate_head = logical_n - 3 * self.num_heads
            projection_stage.cast_result()
            vec_sync_all()
            projection_stage.publish(
                output_gate_workspace_gm,
                m_group,
                gate_head,
                subblock_idx,
                projection_stage.projection_bf16,
            )
        vec_sync_all()

    @jit
    def _prefetch_n256_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """投影进行时，预取一个数据面板对应的 Conv 元数据。"""
        logical_n = n_macro * 2 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                if batch_index < conv_state_indices_gm.shape[0]:
                    cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                    conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                    history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                    if cache_index >= 0:
                        state_view = _conv_slot(conv_state_gm, cache_index)
                        mem_copy(
                            tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            ),
                            tile_slice(
                                state_view[
                                    history_offset : history_offset + CONV_KERNEL - 1,
                                    None,
                                ],
                                (CONV_KERNEL - 1, TILE),
                                (0, logical_n),
                            ),
                        )

    @jit
    def frontend_main(
        self, q, k, v, gate, x, wqkv, wz, conv, state_indices, conv_accepted, wc
    ):
        groups = (
            ceil_div(x.shape[0], self.main_rows)
            if const_expr(self.config.ragged_main)
            else x.shape[0] // self.main_rows
        )
        subblock = get_subblock_id()
        if const_expr(self.config.main_width == 256):
            task_count = groups * (2 * self.num_heads)
            batches = self.main_rows // 16
            for task in range(get_block_idx(), task_count, 32):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = _BaseProjection256(self.main_rows, self.config.ragged_main)
                projection.weight_l2_cache_ctl = self.config.main_weight_l2_cache_ctl
                convolution = ConvSiluStage(8, self.main_rows // 2, self.main_rows // 2)
                history = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((2 * batches * 3, 128))
                )
                batch_base = group * (self.main_rows // 8) + subblock * batches
                if macro < 3 * self.num_heads // 2:
                    projection.matmul_hidden2(x, wqkv, group, macro)
                else:
                    projection.matmul_hidden2(
                        x, wz, group, macro - 3 * self.num_heads // 2
                    )
                h0 = tile_slice(history, (batches * 3, 128), (0, 0))
                h1 = tile_slice(history, (batches * 3, 128), (1, 0))
                self._prefetch_n256_panel(
                    convolution,
                    h0,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    macro,
                    batch_base,
                    batches,
                    0,
                )
                self._prefetch_n256_panel(
                    convolution,
                    h1,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    macro,
                    batch_base,
                    batches,
                    1,
                )
                vec_sync_all()
                projection.acquire_result0()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    h0,
                    tile_slice(convolution.weight, (4, 128), (0, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    group,
                    0,
                    batch_base,
                    batches,
                    subblock,
                    macro * 2,
                )
                projection.acquire_result1()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    h1,
                    tile_slice(convolution.weight, (4, 128), (1, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    group,
                    0,
                    batch_base,
                    batches,
                    subblock,
                    macro * 2 + 1,
                )
        elif const_expr(self.config.main_width == 384):
            task_count = groups * (4 * self.num_heads // 3)
            for task in range(get_block_idx(), task_count, CORE_NUM):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = _BaseProjection384(
                    self.main_rows, 0, self.config.main_weight_l2_cache_ctl
                )
                convolution = ConvSiluStage(8, self.main_rows // 2, self.main_rows // 2)
                batches = self.main_rows // 16
                history = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((3 * batches * 3, TILE))
                )
                if macro < self.num_heads:
                    projection.matmul_hidden3(x, wqkv, group, macro, 56)
                else:
                    projection.matmul_hidden3(x, wz, group, macro - self.num_heads, 56)
                self._consume_n384_triplet(
                    projection,
                    convolution,
                    history,
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    group,
                    macro,
                    group * (self.main_rows // 8) + subblock * batches,
                    batches,
                    subblock,
                )
        else:
            for task in range(get_block_idx(), groups * 4 * self.num_heads, CORE_NUM):
                head = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = ProjectionStage(
                    self.main_rows,
                    0,
                    self.config.main_weight_l2_cache_ctl,
                    row_capacity=self.main_rows,
                )
                convolution = ConvSiluStage(8, self.main_rows // 2, self.main_rows // 2)
                self._run_qkvg_task(
                    projection,
                    convolution,
                    tile_slice(x, (self.main_rows, HIDDEN_SIZE), (group, 0)),
                    tile_slice(q, (self.main_rows, self.num_heads * TILE), (group, 0)),
                    tile_slice(k, (self.main_rows, self.num_heads * TILE), (group, 0)),
                    tile_slice(v, (self.main_rows, self.num_heads * TILE), (group, 0)),
                    tile_slice(
                        gate, (self.main_rows, self.num_heads * TILE), (group, 0)
                    ),
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    0 if head < 3 * self.num_heads else 1,
                    head if head < 3 * self.num_heads else head - 3 * self.num_heads,
                    group * (self.main_rows // 8),
                    self.main_rows // 8,
                    subblock,
                    56,
                )

    @jit
    def frontend_auxiliary_partitioned(self, decay, beta, fa, x, wa, wb, wbeta):
        groups = x.shape[0] // self.auxiliary_rows
        for task in range(
            get_block_idx(), groups * (1 + 128 // self.config.fa_width), 32
        ):
            onchip_rewind(reset_sync_id=True)
            group = task % groups
            part = task // groups
            hidden = tile_slice(x, (self.auxiliary_rows, HIDDEN_SIZE), (group, 0))
            if part == 0:
                stage = BetaProjectionStage(
                    self.auxiliary_rows,
                    0,
                    0,
                    tile_width=(self.num_heads + 15) // 16 * 16,
                    row_capacity=self.auxiliary_rows,
                )
                stage.project_to_gm(
                    hidden,
                    wbeta,
                    tile_slice(
                        beta,
                        (self.auxiliary_rows, (self.num_heads + 15) // 16 * 16),
                        (group, 0),
                    ),
                    0,
                    56,
                )
            else:
                if const_expr(self.config.fa_width <= 32):
                    stage = FaResident(
                        self.auxiliary_rows,
                        0,
                        0,
                        tile_width=self.config.fa_width,
                        row_capacity=self.auxiliary_rows,
                    )
                else:
                    stage = ProjectionStage(
                        self.auxiliary_rows,
                        0,
                        0,
                        tile_width=self.config.fa_width,
                        row_capacity=self.auxiliary_rows,
                    )
                stage.project_to_gm(
                    hidden,
                    tile_slice(wa, (self.config.fa_width, HIDDEN_SIZE), (part - 1, 0)),
                    tile_slice(
                        fa,
                        (self.auxiliary_rows, self.config.fa_width),
                        (group, part - 1),
                    ),
                    0,
                    56,
                )
            vec_sync_all()

    @jit
    def frontend_auxiliary(self, decay, beta, fa, x, wa, wb, wbeta):
        # 每核只有一个任务时，保持原有 Beta/Fa 的并行执行。
        # 第二个任务可能改变 L0A 的分块几何，因此需要在此处排空整个网格。
        groups = x.shape[0] // self.auxiliary_rows
        if groups * (1 + 128 // self.config.fa_width) <= CORE_NUM:
            self._frontend_auxiliary_concurrent(decay, beta, fa, x, wa, wb, wbeta)
        else:
            self._frontend_auxiliary_separate(decay, beta, fa, x, wa, wb, wbeta)

    @jit
    def _frontend_auxiliary_concurrent(self, decay, beta, fa, x, wa, wb, wbeta):
        if const_expr(self.config.fa_width != 128):
            self.frontend_auxiliary_partitioned(decay, beta, fa, x, wa, wb, wbeta)
        else:
            groups = x.shape[0] // self.auxiliary_rows
            subblock = get_subblock_id()
            for task in range(get_block_idx(), 2 * groups, CORE_NUM):
                onchip_rewind(reset_sync_id=True)
                group = task % groups
                hidden = tile_slice(x, (self.auxiliary_rows, HIDDEN_SIZE), (group, 0))
                if task < groups:
                    stage = BetaProjectionStage(
                        self.auxiliary_rows,
                        0,
                        self.config.auxiliary_weight_l2_cache_ctl,
                        tile_width=(self.num_heads + 15) // 16 * 16,
                        row_capacity=self.auxiliary_rows,
                    )
                    stage.project_to_gm(
                        hidden,
                        wbeta,
                        tile_slice(
                            beta,
                            (self.auxiliary_rows, (self.num_heads + 15) // 16 * 16),
                            (group, 0),
                        ),
                        0,
                        56,
                    )
                    vec_sync_all()
                else:
                    stage = ProjectionStage(
                        self.auxiliary_rows,
                        0,
                        self.config.auxiliary_weight_l2_cache_ctl,
                        row_capacity=self.auxiliary_rows,
                    )
                    if const_expr(self.config.split_fb):
                        stage.project_to_gm(
                            hidden,
                            wa,
                            tile_slice(fa, (self.auxiliary_rows, TILE), (group, 0)),
                            0,
                            56,
                        )
                        vec_sync_all()
                    else:
                        low_rank = LowRankDecayStage(stage, self.num_heads)
                        low_rank.project(
                            tile_slice(
                                decay,
                                (self.auxiliary_rows, self.num_heads * TILE),
                                (group, 0),
                            ),
                            hidden,
                            wa,
                            wb,
                            56,
                            subblock,
                        )

    @jit
    def _frontend_auxiliary_separate(self, decay, beta, fa, x, wa, wb, wbeta):
        if const_expr(self.config.fa_width != 128):
            self.frontend_auxiliary_partitioned(decay, beta, fa, x, wa, wb, wbeta)
        else:
            groups = x.shape[0] // self.auxiliary_rows
            subblock = get_subblock_id()
            for group in range(get_block_idx(), groups, CORE_NUM):
                onchip_rewind(reset_sync_id=True)
                stage = BetaProjectionStage(
                    self.auxiliary_rows,
                    0,
                    self.config.auxiliary_weight_l2_cache_ctl,
                    tile_width=(self.num_heads + 15) // 16 * 16,
                    row_capacity=self.auxiliary_rows,
                )
                hidden = tile_slice(x, (self.auxiliary_rows, HIDDEN_SIZE), (group, 0))
                stage.project_to_gm(
                    hidden,
                    wbeta,
                    tile_slice(
                        beta,
                        (self.auxiliary_rows, (self.num_heads + 15) // 16 * 16),
                        (group, 0),
                    ),
                    0,
                    56,
                )
                vec_sync_all()
            global_sync_all()
            onchip_rewind(reset_sync_id=True)
            for group in range(get_block_idx(), groups, CORE_NUM):
                onchip_rewind(reset_sync_id=True)
                stage = ProjectionStage(
                    self.auxiliary_rows,
                    0,
                    self.config.auxiliary_weight_l2_cache_ctl,
                    row_capacity=self.auxiliary_rows,
                )
                hidden = tile_slice(x, (self.auxiliary_rows, HIDDEN_SIZE), (group, 0))
                if const_expr(self.config.split_fb):
                    stage.project_to_gm(
                        hidden,
                        wa,
                        tile_slice(fa, (self.auxiliary_rows, TILE), (group, 0)),
                        0,
                        56,
                    )
                    vec_sync_all()
                else:
                    low_rank = LowRankDecayStage(stage, self.num_heads)
                    low_rank.project(
                        tile_slice(
                            decay,
                            (self.auxiliary_rows, self.num_heads * TILE),
                            (group, 0),
                        ),
                        hidden,
                        wa,
                        wb,
                        56,
                        subblock,
                    )

    @jit
    def frontend_fb(self, decay, fa, wb):
        groups = fa.shape[0] // self.auxiliary_rows
        for task in range(get_block_idx(), groups * self.num_heads, CORE_NUM):
            group = task // self.num_heads
            head = task % self.num_heads
            onchip_rewind(reset_sync_id=True)
            stage = ProjectionStage(
                self.auxiliary_rows, 0, 0, row_capacity=self.auxiliary_rows
            )
            a = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.auxiliary_rows, TILE)),
                depth=1,
            ).next_write()
            b = RingBuffer(
                MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=1
            ).next_write()
            mem_copy(a, tile_slice(fa, (self.auxiliary_rows, TILE), (group, 0)))
            mem_copy(b, tile_slice(wb, (TILE, TILE), (head, 0)))
            stage.matmul_resident(a, b)
            stage.publish_result(
                tile_slice(decay, (self.auxiliary_rows, TILE), (group, head))
            )
            vec_sync_all()


class ProjectionScheduleOperationsFrontendOperations:
    @jit
    def _prefetch_n384_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """投影进行时，预取一个数据面板对应的 Conv 元数据。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                if cache_index >= 0:
                    state_view = _conv_slot(conv_state_gm, cache_index)
                    mem_copy(
                        tile_slice(
                            selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                        ),
                        tile_slice(
                            state_view[
                                history_offset : history_offset + CONV_KERNEL - 1,
                                None,
                            ],
                            (CONV_KERNEL - 1, TILE),
                            (0, logical_n),
                        ),
                    )

    @jit
    def _prefetch_n256_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """投影进行时，预取一个数据面板对应的 Conv 元数据。"""
        logical_n = n_macro * 2 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                if batch_index < conv_state_indices_gm.shape[0]:
                    cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                    conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                    history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                    if cache_index >= 0:
                        state_view = _conv_slot(conv_state_gm, cache_index)
                        mem_copy(
                            tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            ),
                            tile_slice(
                                state_view[
                                    history_offset : history_offset + CONV_KERNEL - 1,
                                    None,
                                ],
                                (CONV_KERNEL - 1, TILE),
                                (0, logical_n),
                            ),
                        )

    @jit
    def _consume_n384_triplet(
        self,
        projection_stage: _Projection384Operations,
        conv_stage: ConvSiluStage,
        selected_history,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
    ):
        history0 = selected_history[
            0 * (batches_per_subblock * (CONV_KERNEL - 1)) : (0 + 1)
            * (batches_per_subblock * (CONV_KERNEL - 1)),
            0 * TILE : (0 + 1) * TILE,
        ]
        history1 = selected_history[
            1 * (batches_per_subblock * (CONV_KERNEL - 1)) : (1 + 1)
            * (batches_per_subblock * (CONV_KERNEL - 1)),
            0 * TILE : (0 + 1) * TILE,
        ]
        history2 = selected_history[
            2 * (batches_per_subblock * (CONV_KERNEL - 1)) : (2 + 1)
            * (batches_per_subblock * (CONV_KERNEL - 1)),
            0 * TILE : (0 + 1) * TILE,
        ]
        weight0 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (0, 0))
        weight1 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (1, 0))
        weight2 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (2, 0))
        self._prefetch_n384_panel(
            conv_stage,
            history0,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            0,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history1,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            1,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history2,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            2,
        )
        vec_sync_all()
        projection_stage.acquire_result0()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history0,
            weight0,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            0,
        )
        projection_stage.acquire_result1()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history1,
            weight1,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            1,
        )
        projection_stage.acquire_result2()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history2,
            weight2,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            2,
        )

    @jit
    def _consume_n384_panel(
        self,
        projection_stage: _Projection384Operations,
        conv_stage: ConvSiluStage,
        selected_history,
        weight_ub,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
        panel,
    ):
        """处理一个驻留的 N128 累加器，同时让 Cube 保留另外两个。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                if batch_index < conv_state_indices_gm.shape[0]:
                    cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                    if cache_index < 0:
                        conv_stage.zero_sequence(local_batch)
                    else:
                        history_view = tile_slice(
                            selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                        )
                        conv_stage.install_history(history_view)
                        state_view = _conv_slot(conv_state_gm, cache_index)
                        mem_copy(
                            tile_slice(
                                state_view[0:2, None], (2, TILE), (0, logical_n)
                            ),
                            tile_slice(history_view[1:3, None], (2, TILE), (0, 0)),
                        )
                        conv_stage.load_accurate_sequence(
                            conv_state_gm, batch_index, logical_n
                        )
                        conv_stage.convolve_sequence(
                            local_batch, projection_stage.projection_fp32, weight_ub
                        )
                        vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        mem_copy(
                            tile_slice(
                                state_view[2 : 2 + self.seq_len, None],
                                (self.seq_len, TILE),
                                (0, logical_n),
                            ),
                            tile_slice(
                                conv_stage.raw_bf16,
                                (self.seq_len, TILE),
                                (local_batch, 0),
                            ),
                        )
            vec_sync_all()
            if projection == 0:
                projection_stage.publish(
                    q_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            elif projection == 1:
                projection_stage.publish(
                    k_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            else:
                projection_stage.publish(
                    v_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
        else:
            gate_head = logical_n - 3 * self.num_heads
            projection_stage.cast_result()
            vec_sync_all()
            projection_stage.publish(
                output_gate_workspace_gm,
                m_group,
                gate_head,
                subblock_idx,
                projection_stage.projection_bf16,
            )
        vec_sync_all()


class MultiCoreProjectionSchedule(
    ProjectionScheduleOperationsFrontendOperations,
    TP4GridScheduleFrontendOperations,
):
    """TP4/TP8 S8 默认路径共用的多核投影和卷积调度。"""

    def __init__(self, heads, rows, width):
        self.config = Schedule(
            main_rows=rows,
            main_width=width,
            ragged_main=width == 256,
            split_fb=True,
            ragged_output=True,
        )
        self.heads = heads
        self.num_heads = heads
        self.main_rows = rows
        self.row_capacity = rows
        self.seq_len = S8_SEQUENCE_LENGTH

    @jit
    def main(
        self,
        q,
        k,
        v,
        gate,
        x,
        wqkv,
        wz,
        conv,
        state_indices,
        conv_accepted,
        wc,
        n,
        logical_rows,
        policy=4,
    ):
        self.main_policy = policy
        self.main_rows = logical_rows
        self.num_heads = self.heads
        self.frontend_main(
            q, k, v, gate, x, wqkv, wz, conv, state_indices, conv_accepted, wc
        )

    @jit
    def frontend_main(
        self, q, k, v, gate, x, wqkv, wz, conv, state_indices, conv_accepted, wc
    ):
        groups = (
            ceil_div(x.shape[0], self.main_rows)
            if const_expr(self.config.ragged_main)
            else x.shape[0] // self.main_rows
        )
        subblock = get_subblock_id()
        if const_expr(self.config.main_width == 256):
            task_count = groups * (2 * self.num_heads)
            batches = self.main_rows // 16
            for task in range(get_block_idx(), task_count, get_block_num()):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = N256ProjectionStage(
                    self.main_rows, self.config.ragged_main
                )
                projection.weight_l2_cache_ctl = self.main_policy
                convolution = ConvSiluStage(
                    8, self.main_rows // 2, self.row_capacity // 2
                )
                history = Buffer(
                    MemLoc.UB,
                    dtypes.bfloat16,
                    make_layout((3 * (self.row_capacity // 16) * 3 * TILE,)),
                ).view(2 * batches * 3, 128)
                batch_base = group * (self.main_rows // 8) + subblock * batches
                if macro < 3 * self.num_heads // 2:
                    projection.matmul_hidden2(x, wqkv, group, macro)
                else:
                    projection.matmul_hidden2(
                        x, wz, group, macro - 3 * self.num_heads // 2
                    )
                h0 = history[
                    0 * (batches * 3) : (0 + 1) * (batches * 3), 0 * 128 : (0 + 1) * 128
                ]
                h1 = history[
                    1 * (batches * 3) : (1 + 1) * (batches * 3), 0 * 128 : (0 + 1) * 128
                ]
                self._prefetch_n256_panel(
                    convolution,
                    h0,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    macro,
                    batch_base,
                    batches,
                    0,
                )
                self._prefetch_n256_panel(
                    convolution,
                    h1,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    macro,
                    batch_base,
                    batches,
                    1,
                )
                vec_sync_all()
                projection.acquire_result0()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    h0,
                    tile_slice(convolution.weight, (4, 128), (0, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    group,
                    0,
                    batch_base,
                    batches,
                    subblock,
                    macro * 2,
                )
                projection.acquire_result1()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    h1,
                    tile_slice(convolution.weight, (4, 128), (1, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    group,
                    0,
                    batch_base,
                    batches,
                    subblock,
                    macro * 2 + 1,
                )
        elif const_expr(self.config.main_width == 384):
            task_count = groups * (4 * self.num_heads // 3)
            for task in range(get_block_idx(), task_count, get_block_num()):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = N384ProjectionStage(
                    self.main_rows, 0, self.config.main_weight_l2_cache_ctl
                )
                projection.weight_l2_cache_ctl = self.main_policy
                convolution = ConvSiluStage(
                    8, self.main_rows // 2, self.row_capacity // 2
                )
                batches = self.main_rows // 16
                history = Buffer(
                    MemLoc.UB,
                    dtypes.bfloat16,
                    make_layout((3 * (self.row_capacity // 16) * 3 * TILE,)),
                ).view(3 * batches * 3, TILE)
                if macro < self.num_heads:
                    projection.matmul_hidden3(x, wqkv, group, macro, 56)
                else:
                    projection.matmul_hidden3(x, wz, group, macro - self.num_heads, 56)
                self._consume_n384_triplet(
                    projection,
                    convolution,
                    history,
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    group,
                    macro,
                    group * (self.main_rows // 8) + subblock * batches,
                    batches,
                    subblock,
                )
        else:
            for task in range(
                get_block_idx(), groups * 4 * self.num_heads, get_block_num()
            ):
                head = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = ProjectionStage(
                    self.main_rows,
                    0,
                    self.config.main_weight_l2_cache_ctl,
                    row_capacity=self.main_rows,
                )
                convolution = ConvSiluStage(
                    8, self.main_rows // 2, self.row_capacity // 2
                )
                self._run_qkvg_task(
                    projection,
                    convolution,
                    tile_slice(
                        x,
                        make_tiler((self.main_rows, HIDDEN_SIZE), alignment=(16, 16)),
                        (group, 0),
                    ),
                    tile_slice(
                        q,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    tile_slice(
                        k,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    tile_slice(
                        v,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    tile_slice(
                        gate,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    0 if head < 3 * self.num_heads else 1,
                    head if head < 3 * self.num_heads else head - 3 * self.num_heads,
                    group * (self.main_rows // 8),
                    self.main_rows // 8,
                    subblock,
                    56,
                )


class TP4AuxiliaryScheduleFrontendOperations:
    @jit
    def beta_tasks(self, beta, x, wbeta, width):
        groups = x.shape[0] // self.auxiliary_rows
        for group in range(get_block_idx(), groups, get_block_num()):
            onchip_rewind(reset_sync_id=True)
            stage = DynamicBetaStage(
                self.auxiliary_rows,
                0,
                4,
                tile_width=width,
                row_capacity=self.auxiliary_rows,
            )
            stage.project_to_gm(
                tile_slice(
                    x,
                    make_tiler((self.auxiliary_rows, HIDDEN_SIZE), alignment=(16, 16)),
                    (group, 0),
                ),
                wbeta,
                tile_slice(
                    beta,
                    make_tiler((self.auxiliary_rows, width), alignment=(16, 16)),
                    (group, 0),
                ),
                0,
                56,
            )
            vec_sync_all()

    @jit
    def execute(self, decay, beta, fa, x, wa, wb, wbeta):
        self.num_heads = self.heads
        self.beta_tasks(beta, x, wbeta, 32)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        groups = x.shape[0] // self.auxiliary_rows
        for group in range(get_block_idx(), groups, get_block_num()):
            onchip_rewind(reset_sync_id=True)
            stage = ProjectionStage(
                self.auxiliary_rows, 0, 4, row_capacity=self.auxiliary_rows
            )
            stage.project_to_gm(
                tile_slice(
                    x,
                    make_tiler((self.auxiliary_rows, HIDDEN_SIZE), alignment=(16, 16)),
                    (group, 0),
                ),
                wa,
                tile_slice(
                    fa,
                    make_tiler((self.auxiliary_rows, TILE), alignment=(16, 16)),
                    (group, 0),
                ),
                0,
                56,
            )
            vec_sync_all()
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.frontend_fb(decay, fa, wb)

    @jit
    def frontend_fb(self, decay, fa, wb):
        groups = fa.shape[0] // self.auxiliary_rows
        for task in range(get_block_idx(), groups * self.num_heads, get_block_num()):
            group = task // self.num_heads
            head = task % self.num_heads
            onchip_rewind(reset_sync_id=True)
            stage = ProjectionStage(
                self.auxiliary_rows, 0, 0, row_capacity=self.auxiliary_rows
            )
            a = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.auxiliary_rows, TILE)),
                depth=1,
            ).next_write()
            b = RingBuffer(
                MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=1
            ).next_write()
            mem_copy(
                a,
                tile_slice(
                    fa,
                    make_tiler((self.auxiliary_rows, TILE), alignment=(16, 16)),
                    (group, 0),
                ),
            )
            mem_copy(b, tile_slice(wb, (TILE, TILE), (head, 0)))
            stage.matmul_resident(a, b)
            stage.publish_result(
                tile_slice(
                    decay,
                    make_tiler((self.auxiliary_rows, TILE), alignment=(16, 16)),
                    (group, head),
                )
            )
            vec_sync_all()


class S4ProjectionScheduleFrontendOperations:
    @jit
    def main(
        self,
        q,
        k,
        v,
        gate,
        x,
        wqkv,
        wz,
        conv,
        indices,
        accepted,
        wc,
        rows,
        width,
        policy,
        qkv_only=False,
    ):
        self.main_rows = rows
        self.num_heads = self.heads
        groups = x.shape[0] // rows
        subblock = get_subblock_id()
        if width == 128:
            count = groups * (
                3 * self.heads if const_expr(qkv_only) else 4 * self.heads
            )
            for task in range(get_block_idx(), count, get_block_num()):
                head = task // groups
                group = task % groups
                self.panel(
                    q,
                    k,
                    v,
                    gate,
                    x,
                    wqkv,
                    wz,
                    conv,
                    indices,
                    accepted,
                    wc,
                    group,
                    head,
                    rows,
                    policy,
                )
        elif width == 256:
            count = groups * (
                3 * self.heads // 2 if const_expr(qkv_only) else 2 * self.heads
            )
            for task in range(get_block_idx(), count, get_block_num()):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = N256ProjectionStage(rows)
                projection.weight_l2_cache_ctl = policy
                convolution = ConvSiluStage(4, rows // 2, 128)
                batches = rows // 8
                history = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((32 * 3 * TILE,))
                ).view(batches * 3, TILE)
                if macro < 3 * self.heads // 2:
                    projection.matmul_hidden2(x, wqkv, group, macro)
                else:
                    projection.matmul_hidden2(x, wz, group, macro - 3 * self.heads // 2)
                first = group * (rows // 4) + subblock * batches
                self._prefetch_n256_panel(
                    convolution,
                    history,
                    conv,
                    indices,
                    accepted,
                    wc,
                    macro,
                    first,
                    batches,
                    0,
                )
                vec_sync_all()
                projection.acquire_result0()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    history,
                    tile_slice(convolution.weight, (4, 128), (0, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    indices,
                    group,
                    0,
                    first,
                    batches,
                    subblock,
                    macro * 2,
                )
                # 将共用历史槽位复用于下一个 PA head 前，
                # 数据面板消费者先完成其 Vector 搬运。
                self._prefetch_n256_panel(
                    convolution,
                    history,
                    conv,
                    indices,
                    accepted,
                    wc,
                    macro,
                    first,
                    batches,
                    1,
                )
                vec_sync_all()
                projection.acquire_result1()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    history,
                    tile_slice(convolution.weight, (4, 128), (1, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    indices,
                    group,
                    0,
                    first,
                    batches,
                    subblock,
                    macro * 2 + 1,
                )
        else:
            count = groups * (
                self.heads if const_expr(qkv_only) else 4 * self.heads // 3
            )
            for task in range(get_block_idx(), count, get_block_num()):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = N384ProjectionStage(rows, 0, 0)
                projection.weight_l2_cache_ctl = policy
                convolution = ConvSiluStage(4, rows // 2, 64)
                batches = rows // 8
                history = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((16 * 3 * TILE,))
                ).view(batches * 3, TILE)
                if macro < self.heads:
                    projection.matmul_hidden3(x, wqkv, group, macro, 56)
                else:
                    projection.matmul_hidden3(x, wz, group, macro - self.heads, 56)
                self.consume_triplet(
                    projection,
                    convolution,
                    history,
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    indices,
                    accepted,
                    wc,
                    group,
                    macro,
                    group * (rows // 4) + subblock * batches,
                    batches,
                    subblock,
                )

    @jit
    def consume_triplet(
        self,
        projection,
        convolution,
        history,
        q,
        k,
        v,
        gate,
        conv,
        indices,
        accepted,
        wc,
        group,
        macro,
        first,
        batches,
        subblock,
    ):
        # 复用一个元数据槽位，避免 S4 的 M 维容纳两倍 batch 时，
        # 将 S8 三个数据面板对应的历史缓存分配扩大一倍。
        self._prefetch_n384_panel(
            convolution, history, conv, indices, accepted, wc, macro, first, batches, 0
        )
        vec_sync_all()
        projection.acquire_result0()
        self._consume_n384_panel(
            projection,
            convolution,
            history,
            tile_slice(convolution.weight, (4, 128), (0, 0)),
            q,
            k,
            v,
            gate,
            conv,
            indices,
            group,
            macro,
            first,
            batches,
            subblock,
            0,
        )
        self._prefetch_n384_panel(
            convolution, history, conv, indices, accepted, wc, macro, first, batches, 1
        )
        vec_sync_all()
        projection.acquire_result1()
        self._consume_n384_panel(
            projection,
            convolution,
            history,
            tile_slice(convolution.weight, (4, 128), (1, 0)),
            q,
            k,
            v,
            gate,
            conv,
            indices,
            group,
            macro,
            first,
            batches,
            subblock,
            1,
        )
        self._prefetch_n384_panel(
            convolution, history, conv, indices, accepted, wc, macro, first, batches, 2
        )
        vec_sync_all()
        projection.acquire_result2()
        self._consume_n384_panel(
            projection,
            convolution,
            history,
            tile_slice(convolution.weight, (4, 128), (2, 0)),
            q,
            k,
            v,
            gate,
            conv,
            indices,
            group,
            macro,
            first,
            batches,
            subblock,
            2,
        )

    @jit
    def panel(
        self,
        q,
        k,
        v,
        gate,
        x,
        wqkv,
        wz,
        conv,
        indices,
        accepted,
        wc,
        group,
        head,
        rows,
        policy,
    ):
        onchip_rewind(reset_sync_id=True)
        projection = S4ProjectionStage(rows)
        projection.weight_l2_cache_ctl = policy
        convolution = ConvSiluStage(4, rows // 2, 64)
        hidden = tile_slice(
            x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
        )
        tile = make_tiler((rows, self.heads * TILE), alignment=(16, 16))
        self.run_qkvg_task(
            projection,
            convolution,
            hidden,
            tile_slice(q, tile, (group, 0)),
            tile_slice(k, tile, (group, 0)),
            tile_slice(v, tile, (group, 0)),
            tile_slice(gate, tile, (group, 0)),
            wqkv,
            wz,
            conv,
            indices,
            accepted,
            wc,
            0 if head < 3 * self.heads else 1,
            head if head < 3 * self.heads else head - 3 * self.heads,
            group * (rows // 4),
            rows // 4,
            get_subblock_id(),
            56,
        )

    @jit
    def run_qkvg_task(
        self,
        projection_stage: ProjectionStage,
        conv_stage: ConvSiluStage,
        hidden_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        qkv_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        projection_kind,
        projection_head,
        batch_start,
        batch_count,
        subblock_idx,
        k_tiles,
    ):
        """执行一个完整的 QKV+Conv 任务或门控投影任务。"""
        if projection_kind == 0:
            projection_stage.matmul_hidden(
                hidden_gm, qkv_weight_gm, projection_head, k_tiles
            )
            batches_per_subblock = batch_count // 2
            selected_history = Buffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((16 * (CONV_KERNEL - 1) * TILE,)),
            ).view(batches_per_subblock * (CONV_KERNEL - 1), TILE)
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    projection = candidate_head // self.num_heads
                    value_head = candidate_head - projection * self.num_heads
                    conv_stage.load_weight(
                        tile_slice(
                            conv_weight_gm[projection, value_head, None, None],
                            (CONV_KERNEL, TILE),
                            (0, 0),
                        )
                    )
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                        history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                        if cache_index >= 0:
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for history_row in range(CONV_KERNEL - 1):
                                mem_copy(
                                    tile_slice(
                                        selected_history,
                                        (1, TILE),
                                        (
                                            local_batch * (CONV_KERNEL - 1)
                                            + history_row,
                                            0,
                                        ),
                                    ),
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (history_offset + history_row, candidate_head),
                                    ),
                                )
            projection_stage.acquire_result()
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index < 0:
                            conv_stage.zero_sequence(local_batch)
                        else:
                            history_view = tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            )
                            conv_stage.install_history(history_view)
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for cache_row in range(2):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_row, candidate_head),
                                    ),
                                    tile_slice(
                                        history_view, (1, TILE), (cache_row + 1, 0)
                                    ),
                                )
                            conv_stage.load_accurate_sequence(
                                conv_state_gm, batch_index, candidate_head
                            )
                            conv_stage.convolve_sequence(
                                local_batch,
                                projection_stage.projection_fp32,
                                tile_slice(
                                    conv_stage.weight, (CONV_KERNEL, TILE), (0, 0)
                                ),
                            )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index >= 0:
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for cache_token in range(self.seq_len):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_token + 2, candidate_head),
                                    ),
                                    tile_slice(
                                        conv_stage.raw_bf16,
                                        (1, TILE),
                                        (local_batch * self.seq_len + cache_token, 0),
                                    ),
                                )
            vec_sync_all()
            if projection_head < self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        q_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            elif projection_head < 2 * self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        k_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            else:
                projection_stage.publish_source(
                    tile_slice(
                        v_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - 2 * self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
        else:
            projection_stage.project(
                hidden_gm, output_gate_weight_gm, projection_head, k_tiles
            )
            projection_stage.publish(
                tile_slice(
                    output_gate_workspace_gm,
                    (projection_stage.row_capacity, TILE),
                    (0, projection_head),
                ),
                subblock_idx,
            )
        vec_sync_all()


class Core32ProjectionSchedule(MultiCoreProjectionSchedule):
    """TP4/TP8 共用的 32 核主投影调度。"""

    def __init__(self, heads, rows, width):
        super().__init__(heads, rows, width)
        self.projection_groups = 4 * heads // 3

    @jit
    def main(
        self,
        q,
        k,
        v,
        gate,
        x,
        wqkv,
        wz,
        conv,
        state_indices,
        conv_accepted,
        wc,
        n,
        logical_rows,
        enabled=True,
    ):
        # 在共用辅助阶段期间，保持选中的主投影阶段及其资源有效。
        # 外层的宽度分支会强制提前排空跨核通道。
        self.enabled = enabled
        self.main_rows = logical_rows
        self.num_heads = self.heads
        self.frontend_main(
            q, k, v, gate, x, wqkv, wz, conv, state_indices, conv_accepted, wc
        )

    @jit
    def frontend_main(
        self, q, k, v, gate, x, wqkv, wz, conv, state_indices, conv_accepted, wc
    ):
        groups = (
            ceil_div(x.shape[0], self.main_rows)
            if const_expr(self.config.ragged_main)
            else x.shape[0] // self.main_rows
        )
        subblock = get_subblock_id()
        if const_expr(self.config.main_width == 256):
            task_count = groups * (2 * self.num_heads)
            if not self.enabled:
                task_count = 0
            batches = self.main_rows // 16
            for task in range(get_block_idx(), task_count, get_block_num()):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = _Core32Projection256(
                    self.main_rows, self.config.ragged_main
                )
                projection.weight_l2_cache_ctl = self.main_policy
                convolution = ConvSiluStage(
                    8, self.main_rows // 2, self.row_capacity // 2
                )
                history = Buffer(
                    MemLoc.UB,
                    dtypes.bfloat16,
                    make_layout((3 * (self.row_capacity // 16) * 3 * TILE,)),
                ).view(2 * batches * 3, 128)
                batch_base = group * (self.main_rows // 8) + subblock * batches
                if macro < 3 * self.num_heads // 2:
                    projection.matmul_hidden2(x, wqkv, group, macro)
                else:
                    projection.matmul_hidden2(
                        x, wz, group, macro - 3 * self.num_heads // 2
                    )
                h0 = history[
                    0 * (batches * 3) : (0 + 1) * (batches * 3), 0 * 128 : (0 + 1) * 128
                ]
                h1 = history[
                    1 * (batches * 3) : (1 + 1) * (batches * 3), 0 * 128 : (0 + 1) * 128
                ]
                self._prefetch_n256_panel(
                    convolution,
                    h0,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    macro,
                    batch_base,
                    batches,
                    0,
                )
                self._prefetch_n256_panel(
                    convolution,
                    h1,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    macro,
                    batch_base,
                    batches,
                    1,
                )
                vec_sync_all()
                projection.acquire_result0()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    h0,
                    tile_slice(convolution.weight, (4, 128), (0, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    group,
                    0,
                    batch_base,
                    batches,
                    subblock,
                    macro * 2,
                )
                projection.acquire_result1()
                self._consume_n384_panel(
                    projection,
                    convolution,
                    h1,
                    tile_slice(convolution.weight, (4, 128), (1, 0)),
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    group,
                    0,
                    batch_base,
                    batches,
                    subblock,
                    macro * 2 + 1,
                )
        elif const_expr(self.config.main_width == 384):
            task_count = groups * self.projection_groups
            if not self.enabled:
                task_count = 0
            for task in range(get_block_idx(), task_count, get_block_num()):
                macro = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = _Core32Projection384(
                    self.main_rows, 0, self.config.main_weight_l2_cache_ctl
                )
                projection.weight_l2_cache_ctl = self.main_policy
                convolution = ConvSiluStage(
                    8, self.main_rows // 2, self.row_capacity // 2
                )
                batches = self.main_rows // 16
                history = Buffer(
                    MemLoc.UB,
                    dtypes.bfloat16,
                    make_layout((3 * (self.row_capacity // 16) * 3 * TILE,)),
                ).view(3 * batches * 3, TILE)
                if macro < self.num_heads:
                    projection.matmul_hidden3(x, wqkv, group, macro, 56)
                else:
                    projection.matmul_hidden3(x, wz, group, macro - self.num_heads, 56)
                self._consume_n384_triplet(
                    projection,
                    convolution,
                    history,
                    q,
                    k,
                    v,
                    gate,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    group,
                    macro,
                    group * (self.main_rows // 8) + subblock * batches,
                    batches,
                    subblock,
                )
        else:
            for task in range(
                get_block_idx(), groups * 4 * self.num_heads, get_block_num()
            ):
                head = task // groups
                group = task % groups
                onchip_rewind(reset_sync_id=True)
                projection = ProjectionStage(
                    self.main_rows,
                    0,
                    self.config.main_weight_l2_cache_ctl,
                    row_capacity=self.main_rows,
                )
                projection.weight_l2_cache_ctl = self.main_policy
                convolution = ConvSiluStage(
                    8, self.main_rows // 2, self.row_capacity // 2
                )
                self._run_qkvg_task(
                    projection,
                    convolution,
                    tile_slice(
                        x,
                        make_tiler((self.main_rows, HIDDEN_SIZE), alignment=(16, 16)),
                        (group, 0),
                    ),
                    tile_slice(
                        q,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    tile_slice(
                        k,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    tile_slice(
                        v,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    tile_slice(
                        gate,
                        make_tiler(
                            (self.main_rows, self.num_heads * TILE), alignment=(16, 16)
                        ),
                        (group, 0),
                    ),
                    wqkv,
                    wz,
                    conv,
                    state_indices,
                    conv_accepted,
                    wc,
                    0 if head < 3 * self.num_heads else 1,
                    head if head < 3 * self.num_heads else head - 3 * self.num_heads,
                    group * (self.main_rows // 8),
                    self.main_rows // 8,
                    subblock,
                    56,
                )


class TP8S4AuxiliarySchedule:
    """计算 Beta 并融合 Fa/Fb，避免 beta 重排和 Fa 经 GM 中转。"""

    def __init__(self, heads, pipelined=False):
        self.heads = heads
        self.beta_width = 16
        self.pipelined = pipelined

    @jit
    def beta_panel(self, beta, x, weight, group, rows, policy):
        onchip_rewind(reset_sync_id=True)
        stage = S4BetaProjectionStage(rows, self.beta_width)
        stage.weight_l2_cache_ctl = policy
        hidden = tile_slice(
            x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
        )
        target = tile_slice(
            beta, make_tiler((rows, self.beta_width), alignment=(16, 16)), (group, 0)
        )
        stage.project_to_gm(hidden, weight, target, 0, 56)
        vec_sync_all()

    @jit
    def decay_panel(self, decay, x, wa, wb, group, rows, policy):
        onchip_rewind(reset_sync_id=True)
        stage = S4ProjectionStage(rows)
        stage.weight_l2_cache_ctl = policy
        low_rank = S4LowRankStage(stage, self.heads)
        low_rank.project(
            tile_slice(
                decay,
                make_tiler((rows, self.heads * TILE), alignment=(16, 16)),
                (group, 0),
            ),
            tile_slice(
                x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
            ),
            wa,
            wb,
            56,
            get_subblock_id(),
        )
        vec_sync_all()

    @jit
    def split_decay(self, decay, beta, fa, x, wa, wb, wbeta, rows, policy):
        """让 Beta 与 Fa 共用任务池，再按行组与 head 分散 Fb。"""
        groups = x.shape[0] // rows
        rank = get_block_idx()
        if const_expr(self.pipelined):
            qkv_groups = x.shape[0] // TILE
            offset = (qkv_groups * self.heads) % get_block_num()
            rank = (rank - offset + get_block_num()) % get_block_num()
        for task in range(rank, 2 * groups, get_block_num()):
            if task < groups:
                self.beta_panel(beta, x, wbeta, task, rows, policy)
            else:
                group = task - groups
                onchip_rewind(reset_sync_id=True)
                stage = S4ProjectionStage(rows)
                stage.weight_l2_cache_ctl = policy
                hidden = tile_slice(
                    x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
                )
                stage.project_to_gm(
                    hidden,
                    wa,
                    tile_slice(
                        fa, make_tiler((rows, TILE), alignment=(16, 16)), (group, 0)
                    ),
                    0,
                    56,
                )
                cube_sync_all()
                vec_sync_all()
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        for task in range(get_block_idx(), groups * self.heads, get_block_num()):
            group = task // self.heads
            head = task % self.heads
            onchip_rewind(reset_sync_id=True)
            stage = S4ProjectionStage(rows)
            panels = LowRankDecayStage(stage, self.heads)
            a, b = panels.fa_l1, panels.fb_l1
            mem_copy(
                a,
                tile_slice(
                    fa, make_tiler((rows, TILE), alignment=(16, 16)), (group, 0)
                ),
            )
            _copy_weight_with_policy(b, tile_slice(wb, (TILE, TILE), (head, 0)), policy)
            stage.matmul_resident(a, b)
            stage.publish_result(
                tile_slice(
                    decay, make_tiler((rows, TILE), alignment=(16, 16)), (group, head)
                )
            )
            cube_sync_all()
            vec_sync_all()
        global_sync_all()
        onchip_rewind(reset_sync_id=True)

    @jit
    def execute(self, decay, beta, fa, x, wa, wb, wbeta, rows, policy):
        # 完整行块较多时，让原先串行的 Fb head 在 AIC 间并行。
        if const_expr(self.pipelined):
            self.split_decay(decay, beta, fa, x, wa, wb, wbeta, rows // 2, policy)
        elif get_block_num() == 32 and rows == TILE and x.shape[0] >= 2 * TILE:
            self.split_decay(decay, beta, fa, x, wa, wb, wbeta, rows, policy)
        else:
            self.execute_fused(decay, beta, x, wa, wb, wbeta, rows, policy)

    @jit
    def execute_fused(self, decay, beta, x, wa, wb, wbeta, rows, policy):
        for group in range(get_block_idx(), x.shape[0] // rows, get_block_num()):
            self.beta_panel(beta, x, wbeta, group, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        for group in range(get_block_idx(), x.shape[0] // rows, get_block_num()):
            self.decay_panel(decay, x, wa, wb, group, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)


class _TailMaskedProjection384(_Core32Projection384):
    def __init__(self, rows, input_l2_cache_ctl=0, weight_l2_cache_ctl=0):
        super().__init__(rows, input_l2_cache_ctl, weight_l2_cache_ctl)
        self.allow_tail = True

    @jit
    def matmul_hidden3(self, hidden_gm, weight_gm, m_group, n_macro, k_tiles):
        for kc_idx in range(k_tiles * TILE // PROJECTION_KC):
            a_l1_slot = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            valid_rows = hidden_gm.shape[0] - m_group * self.rows
            if valid_rows > self.rows:
                valid_rows = self.rows
            mem_copy(
                a_l1_slot[0:valid_rows, None],
                hidden_gm[
                    m_group * self.rows : m_group * self.rows + valid_rows,
                    kc_idx * PROJECTION_KC : (kc_idx + 1) * PROJECTION_KC,
                ],
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            b_l1_slot = self.b_l1.next_write()
            self._copy_projection_weight(
                b_l1_slot,
                tile_slice(weight_gm, (3 * TILE, PROJECTION_KC), (n_macro, kc_idx)),
            )
            for ks_idx in range(PROJECTION_KC // PROJECTION_KS):
                l0a_slot = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                l0a_slot_source_4 = tile_slice(
                    a_l1_slot, (128, 128), (0, ks_idx * PROJECTION_KS // 128)
                )[0 : self.rows, None]
                l0a_slot = l0a_slot[tuple(map(slice, l0a_slot_source_4.shape))]
                mem_copy(l0a_slot, l0a_slot_source_4)
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_11 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (0, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_11.shape))]
                mem_copy(l0b_slot, l0b_slot_source_11)
                matmul(self.l0c0, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half0, self.l0c0.partition(dim=0))
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_12 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (1, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_12.shape))]
                mem_copy(l0b_slot, l0b_slot_source_12)
                matmul(self.l0c1, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half1, self.l0c1.partition(dim=0))
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_13 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (2, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_13.shape))]
                mem_copy(l0b_slot, l0b_slot_source_13)
                matmul(self.l0c2, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half2, self.l0c2.partition(dim=0))


class TP8AuxiliaryScheduleFrontendOperations:
    @jit
    def beta_tasks(self, beta, x, wbeta, width):
        groups = x.shape[0] // self.auxiliary_rows
        for group in range(get_block_idx(), groups, get_block_num()):
            onchip_rewind(reset_sync_id=True)
            stage = DynamicBetaStage(
                self.auxiliary_rows,
                0,
                4,
                tile_width=width,
                row_capacity=self.auxiliary_rows,
            )
            stage.project_to_gm(
                tile_slice(
                    x,
                    make_tiler((self.auxiliary_rows, HIDDEN_SIZE), alignment=(16, 16)),
                    (group, 0),
                ),
                wbeta,
                tile_slice(
                    beta,
                    make_tiler((self.auxiliary_rows, width), alignment=(16, 16)),
                    (group, 0),
                ),
                0,
                56,
            )
            vec_sync_all()

    @jit
    def execute(self, decay, beta, fa, x, wa, wb, wbeta):
        self.num_heads = self.heads
        self.beta_tasks(beta, x, wbeta, 16)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        groups = x.shape[0] // self.auxiliary_rows
        for group in range(get_block_idx(), groups, get_block_num()):
            onchip_rewind(reset_sync_id=True)
            stage = ProjectionStage(
                self.auxiliary_rows, 0, 4, row_capacity=self.auxiliary_rows
            )
            stage.project_to_gm(
                tile_slice(
                    x,
                    make_tiler((self.auxiliary_rows, HIDDEN_SIZE), alignment=(16, 16)),
                    (group, 0),
                ),
                wa,
                tile_slice(
                    fa,
                    make_tiler((self.auxiliary_rows, TILE), alignment=(16, 16)),
                    (group, 0),
                ),
                0,
                56,
            )
            vec_sync_all()
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        self.frontend_fb(decay, fa, wb)

    @jit
    def frontend_fb(self, decay, fa, wb):
        groups = fa.shape[0] // self.auxiliary_rows
        for task in range(get_block_idx(), groups * self.num_heads, get_block_num()):
            group = task // self.num_heads
            head = task % self.num_heads
            onchip_rewind(reset_sync_id=True)
            stage = ProjectionStage(
                self.auxiliary_rows, 0, 0, row_capacity=self.auxiliary_rows
            )
            a = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((self.auxiliary_rows, TILE)),
                depth=1,
            ).next_write()
            b = RingBuffer(
                MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=1
            ).next_write()
            mem_copy(
                a,
                tile_slice(
                    fa,
                    make_tiler((self.auxiliary_rows, TILE), alignment=(16, 16)),
                    (group, 0),
                ),
            )
            mem_copy(b, tile_slice(wb, (TILE, TILE), (head, 0)))
            stage.matmul_resident(a, b)
            stage.publish_result(
                tile_slice(
                    decay,
                    make_tiler((self.auxiliary_rows, TILE), alignment=(16, 16)),
                    (group, head),
                )
            )
            vec_sync_all()


class TailMaskedCore32ProjectionSchedule(Core32ProjectionSchedule):
    """TP8 尾块的 N384 投影与 Conv，预取时检查有效行边界。"""

    @jit
    def _prefetch_n384_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """投影进行时，预取一个数据面板对应的 Conv 元数据。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                if batch_index < conv_state_indices_gm.shape[0]:
                    cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                    conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                    history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                    if cache_index >= 0:
                        state_view = _conv_slot(conv_state_gm, cache_index)
                        mem_copy(
                            tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            ),
                            tile_slice(
                                state_view[
                                    history_offset : history_offset + CONV_KERNEL - 1,
                                    None,
                                ],
                                (CONV_KERNEL - 1, TILE),
                                (0, logical_n),
                            ),
                        )

    @jit
    def frontend_main(
        self, q, k, v, gate, x, wqkv, wz, conv, state_indices, conv_accepted, wc
    ):
        groups = ceil_div(x.shape[0], self.main_rows)
        subblock = get_subblock_id()
        task_count = groups * self.projection_groups
        if not self.enabled:
            task_count = 0
        for task in range(get_block_idx(), task_count, get_block_num()):
            macro = task // groups
            group = task % groups
            onchip_rewind(reset_sync_id=True)
            projection = _TailMaskedProjection384(
                self.main_rows, 0, self.config.main_weight_l2_cache_ctl
            )
            projection.weight_l2_cache_ctl = self.main_policy
            convolution = ConvSiluStage(8, self.main_rows // 2, self.row_capacity // 2)
            batches = self.main_rows // 16
            history = Buffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((3 * (self.row_capacity // 16) * 3 * TILE,)),
            ).view(3 * batches * 3, TILE)
            if macro < self.num_heads:
                projection.matmul_hidden3(x, wqkv, group, macro, 56)
            else:
                projection.matmul_hidden3(x, wz, group, macro - self.num_heads, 56)
            self._consume_n384_triplet(
                projection,
                convolution,
                history,
                q,
                k,
                v,
                gate,
                conv,
                state_indices,
                conv_accepted,
                wc,
                group,
                macro,
                group * (self.main_rows // 8) + subblock * batches,
                batches,
                subblock,
            )


class GateDecayProjectionStage:
    """为门控 head 3～5 和 decay Fa 共享一次 hidden 的 K128 遍历。"""

    def __init__(self, input_l2_cache_ctl: int = 0, weight_l2_cache_ctl: int = 0):
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.a_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=2
        )
        self.a_l1_shape = (TILE, TILE)
        self.g_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((3 * TILE, TILE)), depth=2
        )
        self.fa_weight_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=2
        )
        self.l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((TILE, TILE)), depth=2
        )
        self.l0a_shape = (TILE, TILE)
        self.l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((TILE, TILE)), depth=2
        )
        self.l0c0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.l0c1 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.l0c2 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.l0c_fa = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.fa_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=1
        ).next_write()
        self.fb_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((TILE, TILE)), depth=1
        ).next_write()

    @jit
    def matmul_hidden3_fa(
        self, hidden_gm, gate_weight_gm, fa_weight_gm, m_group, k_tiles
    ):
        for k_idx in range(k_tiles):
            a_l1_slot = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                a_l1_slot,
                tile_slice(hidden_gm, (TILE, TILE), (m_group, k_idx)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            g_l1_slot = self.g_l1.next_write()
            mem_copy(
                g_l1_slot,
                tile_slice(gate_weight_gm, (3 * TILE, TILE), (1, k_idx)),
                options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
            )
            fa_weight_l1_slot = self.fa_weight_l1.next_write()
            mem_copy(
                fa_weight_l1_slot,
                tile_slice(fa_weight_gm, (TILE, TILE), (0, k_idx)),
                options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
            )
            l0a_slot = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            l0a_slot = l0a_slot[tuple(map(slice, a_l1_slot.shape))]
            mem_copy(l0a_slot, a_l1_slot)
            l0b_slot = self.l0b.next_write()
            l0b_slot_source_2 = tile_slice(g_l1_slot, (TILE, TILE), (0, 0))
            l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_2.shape))]
            mem_copy(l0b_slot, l0b_slot_source_2)
            matmul(self.l0c0, l0a_slot, l0b_slot, init=k_idx == 0)
            l0b_slot = self.l0b.next_write()
            l0b_slot_source_3 = tile_slice(g_l1_slot, (TILE, TILE), (1, 0))
            l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_3.shape))]
            mem_copy(l0b_slot, l0b_slot_source_3)
            matmul(self.l0c1, l0a_slot, l0b_slot, init=k_idx == 0)
            l0b_slot = self.l0b.next_write()
            l0b_slot_source_4 = tile_slice(g_l1_slot, (TILE, TILE), (2, 0))
            l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_4.shape))]
            mem_copy(l0b_slot, l0b_slot_source_4)
            matmul(self.l0c2, l0a_slot, l0b_slot, init=k_idx == 0)
            l0b_slot = self.l0b.next_write()
            l0b_slot = l0b_slot[tuple(map(slice, fa_weight_l1_slot.shape))]
            mem_copy(l0b_slot, fa_weight_l1_slot)
            matmul(self.l0c_fa, l0a_slot, l0b_slot, init=k_idx == 0)

    @jit
    def _publish_l0c(
        self, workspace_gm, source_l0c, m_group, output_head, subblock_idx
    ):
        whole_tile = tile_slice(workspace_gm, (TILE, TILE), (m_group, output_head))
        mem_copy(whole_tile, source_l0c)

    @jit
    def publish_gate(self, output_gate_workspace_gm, m_group, subblock_idx):
        self._publish_l0c(output_gate_workspace_gm, self.l0c0, m_group, 3, subblock_idx)
        self._publish_l0c(output_gate_workspace_gm, self.l0c1, m_group, 4, subblock_idx)
        self._publish_l0c(output_gate_workspace_gm, self.l0c2, m_group, 5, subblock_idx)

    @jit
    def project_fb(self, decay_workspace_gm, fb_weight_gm, m_group, subblock_idx):
        mem_copy(self.fa_l1, self.l0c_fa)
        for output_head in range(TP16_HEADS):
            mem_copy(
                self.fb_l1,
                tile_slice(fb_weight_gm, (TILE, TILE), (output_head, 0)),
                options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
            )
            l0a_slot = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            l0a_slot = l0a_slot[tuple(map(slice, self.fa_l1.shape))]
            mem_copy(l0a_slot, self.fa_l1)
            l0b_slot = self.l0b.next_write()
            l0b_slot = l0b_slot[tuple(map(slice, self.fb_l1.shape))]
            mem_copy(l0b_slot, self.fb_l1)
            matmul(self.l0c_fa, l0a_slot, l0b_slot, init=True)
            self._publish_l0c(
                decay_workspace_gm, self.l0c_fa, m_group, output_head, subblock_idx
            )


class BetaZSharedStage:
    """为最后一个门控 head 和 Beta 共享一次 hidden KC 读取。"""

    def __init__(self, rows, input_l2_cache_ctl=0, weight_l2_cache_ctl=0):
        self.input_l2_cache_ctl = int(input_l2_cache_ctl)
        self.weight_l2_cache_ctl = int(weight_l2_cache_ctl)
        self.rows = rows
        self.half = rows // 2
        self.a = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((128 * 256,)),
            depth=2,
            data_format="nd",
        )
        self.a_shape = (rows, 256)
        self.z = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((128, 256)), depth=2
        )
        self.la = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((128 * 128,)),
            depth=2,
            data_format="nd",
        )
        self.la_shape = (rows, 128)
        self.lz = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((128, 128)), depth=1
        ).next_write()
        self.lb = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((16, 128)), depth=2
        )
        self.cz = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * 128,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((rows, 128)),
            data_format="nz",
        )
        self.cb = from_tensor(
            RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((128 * 16,)),
                depth=1,
                data_format="nd",
            ).next_write(),
            layout=make_layout((rows, 16)),
            data_format="nz",
        )

    @jit
    def project(self, hidden_gm, z_weight_gm, beta_weight_gm, z_gm, beta_gm, subblock):
        beta_l1 = Buffer(
            MemLoc.L1,
            beta_weight_gm.dtype,
            beta_weight_gm.layout,
            physical_layout=beta_weight_gm.physical_layout,
        )
        mem_copy(
            beta_l1,
            beta_weight_gm,
            options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
        )
        for kc in range(HIDDEN_SIZE // PROJECTION_KC):
            a_slot = from_tensor(
                self.a.next_write(), layout=make_layout(self.a_shape), data_format="nz"
            )
            mem_copy(
                a_slot,
                tile_slice(hidden_gm, (128, 256), (0, kc)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            z_slot = self.z.next_write()
            mem_copy(
                z_slot,
                tile_slice(z_weight_gm, (128, 256), (5, kc)),
                options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
            )
            for ks in range(2):
                la_slot = from_tensor(
                    self.la.next_write(),
                    layout=make_layout(self.la_shape),
                    data_format="nz",
                )
                la_slot_source = tile_slice(a_slot, (128, 128), (0, ks))
                la_slot = la_slot[tuple(map(slice, la_slot_source.shape))]
                mem_copy(la_slot, la_slot_source)
                mem_copy(self.lz, tile_slice(z_slot, (128, 128), (0, ks)))
                matmul(self.cz, la_slot, self.lz, init=kc == 0 and ks == 0)
                lb_slot = self.lb.next_write()
                mem_copy(lb_slot, tile_slice(beta_l1, (16, 128), (0, kc * 2 + ks)))
                matmul(self.cb, la_slot, lb_slot, init=kc == 0 and ks == 0)
        mem_copy(z_gm[0 : self.rows, None], self.cz)
        mem_copy(beta_gm[0 : self.rows, None], self.cb)
        cube_sync_all()


class TP16S4AuxiliarySchedule:
    """计算 Beta 并融合 Fa/Fb，避免 beta 重排和 Fa 经 GM 中转。"""

    def __init__(self, heads):
        self.heads = heads
        self.beta_width = 16

    @jit
    def beta_panel(self, beta, x, weight, group, rows, policy):
        onchip_rewind(reset_sync_id=True)
        stage = S4BetaProjectionStage(rows, self.beta_width)
        stage.weight_l2_cache_ctl = policy
        hidden = tile_slice(
            x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
        )
        target = tile_slice(
            beta, make_tiler((rows, self.beta_width), alignment=(16, 16)), (group, 0)
        )
        stage.project_to_gm(hidden, weight, target, 0, 56)
        vec_sync_all()

    @jit
    def decay_panel(self, decay, x, wa, wb, group, rows, policy):
        onchip_rewind(reset_sync_id=True)
        stage = S4ProjectionStage(rows)
        stage.weight_l2_cache_ctl = policy
        low_rank = S4LowRankStage(stage, self.heads)
        low_rank.project(
            tile_slice(
                decay,
                make_tiler((rows, self.heads * TILE), alignment=(16, 16)),
                (group, 0),
            ),
            tile_slice(
                x, make_tiler((rows, HIDDEN_SIZE), alignment=(16, 16)), (group, 0)
            ),
            wa,
            wb,
            56,
            get_subblock_id(),
        )
        vec_sync_all()

    @jit
    def execute(self, decay, beta, fa, x, wa, wb, wbeta, rows, policy):
        for group in range(get_block_idx(), x.shape[0] // rows, get_block_num()):
            self.beta_panel(beta, x, wbeta, group, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)
        for group in range(get_block_idx(), x.shape[0] // rows, get_block_num()):
            self.decay_panel(decay, x, wa, wb, group, rows, policy)
        global_sync_all()
        onchip_rewind(reset_sync_id=True)


# TP16 前端核分配与投影任务数。
PROJECTION_TASK_COUNT = 4 * TP16_HEADS
THREE_GROUP_HALF_PROJECTION_CORE_COUNT = 12
FOUR_GROUP_BETA_CORE_START = 24
FULL_FA_K_TILE_COUNT = 6
PROJECTION_CORES_PER_CORE_GROUP = 12


class TP16GridScheduleFrontendOperations:
    @jit
    def _run_qkvg_task(
        self,
        projection_stage: ProjectionStage,
        conv_stage: ConvSiluStage,
        hidden_gm: Tensor,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        qkv_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        projection_kind,
        projection_head,
        batch_start,
        batch_count,
        subblock_idx,
        k_tiles,
    ):
        """执行一个完整的 QKV+Conv 任务或门控投影任务。"""
        if projection_kind == 0:
            projection_stage.matmul_hidden(
                hidden_gm, qkv_weight_gm, projection_head, k_tiles
            )
            batches_per_subblock = batch_count // 2
            selected_history = Buffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((batches_per_subblock * (CONV_KERNEL - 1), TILE)),
            )
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    projection = candidate_head // self.num_heads
                    value_head = candidate_head - projection * self.num_heads
                    conv_stage.load_weight(
                        tile_slice(
                            conv_weight_gm[projection, value_head, None, None],
                            (CONV_KERNEL, TILE),
                            (0, 0),
                        )
                    )
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                        history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                        if cache_index >= 0:
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for history_row in range(CONV_KERNEL - 1):
                                mem_copy(
                                    tile_slice(
                                        selected_history,
                                        (1, TILE),
                                        (
                                            local_batch * (CONV_KERNEL - 1)
                                            + history_row,
                                            0,
                                        ),
                                    ),
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (history_offset + history_row, candidate_head),
                                    ),
                                )
            projection_stage.acquire_result()
            for candidate_head in range(3 * self.num_heads):
                if projection_head == candidate_head:
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index < 0:
                            conv_stage.zero_sequence(local_batch)
                        else:
                            history_view = tile_slice(
                                selected_history,
                                (CONV_KERNEL - 1, TILE),
                                (local_batch, 0),
                            )
                            conv_stage.install_history(history_view)
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for cache_row in range(2):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_row, candidate_head),
                                    ),
                                    tile_slice(
                                        history_view, (1, TILE), (cache_row + 1, 0)
                                    ),
                                )
                            conv_stage.load_accurate_sequence(
                                conv_state_gm, batch_index, candidate_head
                            )
                            conv_stage.convolve_sequence(
                                local_batch,
                                projection_stage.projection_fp32,
                                tile_slice(
                                    conv_stage.weight, (CONV_KERNEL, TILE), (0, 0)
                                ),
                            )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    for local_batch in range(batches_per_subblock):
                        batch_index = (
                            batch_start
                            + subblock_idx * batches_per_subblock
                            + local_batch
                        )
                        cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                        if cache_index >= 0:
                            state_view = _conv_slot(conv_state_gm, cache_index)
                            for cache_token in range(self.seq_len):
                                mem_copy(
                                    tile_slice(
                                        state_view,
                                        (1, TILE),
                                        (cache_token + 2, candidate_head),
                                    ),
                                    tile_slice(
                                        conv_stage.raw_bf16,
                                        (1, TILE),
                                        (local_batch * self.seq_len + cache_token, 0),
                                    ),
                                )
            vec_sync_all()
            if projection_head < self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        q_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            elif projection_head < 2 * self.num_heads:
                projection_stage.publish_source(
                    tile_slice(
                        k_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
            else:
                projection_stage.publish_source(
                    tile_slice(
                        v_workspace_gm,
                        (projection_stage.row_capacity, TILE),
                        (0, projection_head - 2 * self.num_heads),
                    ),
                    conv_stage.output_bf16,
                    subblock_idx,
                )
        else:
            projection_stage.project(
                hidden_gm, output_gate_weight_gm, projection_head, k_tiles
            )
            projection_stage.publish(
                tile_slice(
                    output_gate_workspace_gm,
                    (projection_stage.row_capacity, TILE),
                    (0, projection_head),
                ),
                subblock_idx,
            )
        vec_sync_all()

    @jit
    def _run_frontend_main_task(
        self,
        projection_stage: ProjectionStage,
        conv_stage: ConvSiluStage,
        low_rank_stage: LowRankDecayStage,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        projection_kind,
        projection_head,
        batch_start,
        batch_count,
        subblock_idx,
        k_tiles,
    ):
        rows = projection_stage.row_capacity
        row_tile = batch_start // batch_count
        hidden_tile = make_tiler((rows, self.hidden_size), alignment=(16, 16))
        qkv_tile = make_tiler((rows, self.num_heads * TILE), alignment=(16, 16))
        if projection_kind < 2:
            self._run_qkvg_task(
                projection_stage,
                conv_stage,
                tile_slice(hidden_gm, hidden_tile, (row_tile, 0)),
                tile_slice(q_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(k_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(v_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(output_gate_workspace_gm, qkv_tile, (row_tile, 0)),
                qkv_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                projection_kind,
                projection_head,
                batch_start,
                batch_count,
                subblock_idx,
                k_tiles,
            )
        else:
            low_rank_stage.project(
                tile_slice(decay_workspace_gm, qkv_tile, (row_tile, 0)),
                tile_slice(hidden_gm, hidden_tile, (row_tile, 0)),
                fa_weight_gm,
                fb_weight_gm,
                k_tiles,
                subblock_idx,
            )

    @jit
    def _prefetch_n384_panel(
        self,
        conv_stage: ConvSiluStage,
        selected_history,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        n_macro,
        batch_base,
        batches_per_subblock,
        panel,
    ):
        """投影进行时，预取一个数据面板对应的 Conv 元数据。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            mem_copy(
                tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (panel, 0)),
                tile_slice(
                    conv_weight_gm[projection, value_head, None, None],
                    (CONV_KERNEL, TILE),
                    (0, 0),
                ),
            )
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                history_offset = conv_accepted - 1 if conv_accepted > 0 else 0
                if cache_index >= 0:
                    state_view = _conv_slot(conv_state_gm, cache_index)
                    mem_copy(
                        tile_slice(
                            selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                        ),
                        tile_slice(
                            state_view[
                                history_offset : history_offset + CONV_KERNEL - 1, None
                            ],
                            (CONV_KERNEL - 1, TILE),
                            (0, logical_n),
                        ),
                    )

    @jit
    def _consume_n384_panel(
        self,
        projection_stage: _ResidentN384ProjectionStage,
        conv_stage: ConvSiluStage,
        selected_history,
        weight_ub,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
        panel,
    ):
        """处理一个驻留的 N128 累加器，同时让 Cube 保留另外两个。"""
        logical_n = n_macro * 3 + panel
        if logical_n < 3 * self.num_heads:
            projection = logical_n // self.num_heads
            value_head = logical_n - projection * self.num_heads
            for local_batch in range(batches_per_subblock):
                batch_index = batch_base + local_batch
                cache_index = dtypes.int64(conv_state_indices_gm[batch_index])
                if cache_index < 0:
                    conv_stage.zero_sequence(local_batch)
                else:
                    history_view = tile_slice(
                        selected_history, (CONV_KERNEL - 1, TILE), (local_batch, 0)
                    )
                    conv_stage.install_history(history_view)
                    state_view = _conv_slot(conv_state_gm, cache_index)
                    mem_copy(
                        tile_slice(state_view[0:2, None], (2, TILE), (0, logical_n)),
                        tile_slice(history_view[1:3, None], (2, TILE), (0, 0)),
                    )
                    conv_stage.load_accurate_sequence(
                        conv_state_gm, batch_index, logical_n
                    )
                    conv_stage.convolve_sequence(
                        local_batch, projection_stage.projection_fp32, weight_ub
                    )
                    vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                    mem_copy(
                        tile_slice(
                            state_view[2 : 2 + self.seq_len, None],
                            (self.seq_len, TILE),
                            (0, logical_n),
                        ),
                        tile_slice(
                            conv_stage.raw_bf16, (self.seq_len, TILE), (local_batch, 0)
                        ),
                    )
            vec_sync_all()
            if projection == 0:
                projection_stage.publish(
                    q_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            elif projection == 1:
                projection_stage.publish(
                    k_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
            else:
                projection_stage.publish(
                    v_workspace_gm,
                    m_group,
                    value_head,
                    subblock_idx,
                    conv_stage.output_bf16,
                )
        else:
            gate_head = logical_n - 3 * self.num_heads
            projection_stage.cast_result()
            vec_sync_all()
            projection_stage.publish(
                output_gate_workspace_gm,
                m_group,
                gate_head,
                subblock_idx,
                projection_stage.projection_bf16,
            )
        vec_sync_all()

    @jit
    def _consume_n384_triplet(
        self,
        projection_stage: _ResidentN384ProjectionStage,
        conv_stage: ConvSiluStage,
        selected_history,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
        m_group,
        n_macro,
        batch_base,
        batches_per_subblock,
        subblock_idx,
    ):
        history0 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (0, 0)
        )
        history1 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (1, 0)
        )
        history2 = tile_slice(
            selected_history, (batches_per_subblock * (CONV_KERNEL - 1), TILE), (2, 0)
        )
        weight0 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (0, 0))
        weight1 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (1, 0))
        weight2 = tile_slice(conv_stage.weight, (CONV_KERNEL, TILE), (2, 0))
        self._prefetch_n384_panel(
            conv_stage,
            history0,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            0,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history1,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            1,
        )
        self._prefetch_n384_panel(
            conv_stage,
            history2,
            conv_state_gm,
            conv_state_indices_gm,
            conv_accepted_gm,
            conv_weight_gm,
            n_macro,
            batch_base,
            batches_per_subblock,
            2,
        )
        vec_sync_all()
        projection_stage.acquire_result0()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history0,
            weight0,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            0,
        )
        projection_stage.acquire_result1()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history1,
            weight1,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            1,
        )
        projection_stage.acquire_result2()
        self._consume_n384_panel(
            projection_stage,
            conv_stage,
            history2,
            weight2,
            q_workspace_gm,
            k_workspace_gm,
            v_workspace_gm,
            output_gate_workspace_gm,
            conv_state_gm,
            conv_state_indices_gm,
            m_group,
            n_macro,
            batch_base,
            batches_per_subblock,
            subblock_idx,
            2,
        )

    @jit
    def _frontend_adaptive_n384(
        self,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
    ):
        """将每组的八个 M128xN384 任务均衡分配到启动网格。"""
        block_idx = get_block_idx()
        block_num = self.block_num
        subblock_idx = get_subblock_id()
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        task_count = self.batch_groups * self.projection_groups
        batches_per_subblock = BATCH_TILE // 2

        # 按 N 优先排序，保留已经验证的 B64 映射。运行时网格步长
        # 使 28 AIC 和 32 AIC 启动时的任务数都保持均衡。
        for task_idx in range(block_idx, task_count, block_num):
            n_macro = task_idx // self.batch_groups
            m_group = task_idx - n_macro * self.batch_groups
            batch_base = m_group * BATCH_TILE + subblock_idx * batches_per_subblock
            onchip_rewind(reset_sync_id=True)
            if n_macro == 7 and task_count <= block_num:
                projection_stage = GateDecayProjectionStage(
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                projection_stage.matmul_hidden3_fa(
                    hidden_gm, output_gate_weight_gm, fa_weight_gm, m_group, k_tiles
                )
                projection_stage.publish_gate(
                    output_gate_workspace_gm, m_group, subblock_idx
                )
                projection_stage.project_fb(
                    decay_workspace_gm, fb_weight_gm, m_group, subblock_idx
                )
            else:
                projection_stage = _ResidentN384ProjectionStage(
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                conv_stage = ConvSiluStage(self.seq_len, TILE // 2)
                selected_history = Buffer(
                    MemLoc.UB,
                    dtypes.bfloat16,
                    make_layout((3 * batches_per_subblock * (CONV_KERNEL - 1), TILE)),
                )
                if n_macro < QKV_PANEL_COUNT:
                    projection_stage.matmul_hidden3(
                        hidden_gm, qkv_weight_gm, m_group, n_macro, k_tiles
                    )
                else:
                    projection_stage.matmul_hidden3(
                        hidden_gm,
                        output_gate_weight_gm,
                        m_group,
                        n_macro - QKV_PANEL_COUNT,
                        k_tiles,
                    )
                self._consume_n384_triplet(
                    projection_stage,
                    conv_stage,
                    selected_history,
                    q_workspace_gm,
                    k_workspace_gm,
                    v_workspace_gm,
                    output_gate_workspace_gm,
                    conv_state_gm,
                    conv_state_indices_gm,
                    conv_accepted_gm,
                    conv_weight_gm,
                    m_group,
                    n_macro,
                    batch_base,
                    batches_per_subblock,
                    subblock_idx,
                )

    @jit
    def _frontend_adaptive_auxiliary(
        self,
        decay_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        hidden_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
    ):
        """将 Beta 和多波次 decay 任务分配给负载最轻的 AIC。"""
        block_idx = get_block_idx()
        block_num = self.block_num
        subblock_idx = get_subblock_id()
        task_count = self.batch_groups * 8
        extra_task_cores = task_count % block_num
        aux_start = block_num - 8 if extra_task_cores == 0 else extra_task_cores
        aux_slot = (block_idx - aux_start + block_num) % block_num
        auxiliary_count = (
            self.batch_groups if task_count <= block_num else 2 * self.batch_groups
        )
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        for auxiliary_idx in range(aux_slot, auxiliary_count, block_num):
            onchip_rewind(reset_sync_id=True)
            if auxiliary_idx < self.batch_groups:
                m_group = auxiliary_idx
                beta_stage = ProjectionStage(
                    row_count=TILE,
                    row_capacity=TILE,
                    tile_width=BETA_TILE,
                    input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                    weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
                )
                beta_stage.project_to_gm(
                    tile_slice(hidden_gm, (TILE, self.hidden_size), (m_group, 0)),
                    beta_weight_gm,
                    tile_slice(beta_workspace_gm, (TILE, BETA_TILE), (m_group, 0)),
                    0,
                    k_tiles,
                )
                vec_sync_all()
            else:
                m_group = auxiliary_idx - self.batch_groups
                projection_stage = ProjectionStage(
                    row_count=TILE,
                    row_capacity=TILE,
                    input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                    weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
                )
                low_rank_stage = LowRankDecayStage(projection_stage, self.num_heads)
                low_rank_stage.project(
                    tile_slice(
                        decay_workspace_gm, (TILE, self.num_heads * TILE), (m_group, 0)
                    ),
                    tile_slice(hidden_gm, (TILE, self.hidden_size), (m_group, 0)),
                    fa_weight_gm,
                    fb_weight_gm,
                    k_tiles,
                    subblock_idx,
                )

    @jit
    def _frontend_tail(
        self,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        replay_decay_gm: ReplayPool,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
    ):
        """严格沿用主机模型的任务顺序，各核独立推进，不使用网格屏障。"""
        self.seq_len = S8_SEQUENCE_LENGTH
        self.row_count = 128
        self.projection_row_block = 64
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        groups = replay_decay_gm.indices.shape[0] // 16
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        frontend_weight_l2_cache_ctl = 1
        full_qkv_count = 32 if groups == 2 else groups * 18
        full_gate_count = 0 if groups == 2 else 7 if groups == 3 else 20
        full_stage = ProjectionStage(
            row_count=128,
            row_capacity=128,
            input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
            weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
        )
        full_conv = ConvSiluStage(8, 64, row_capacity=64)
        full_decay = LowRankDecayStage(full_stage, self.num_heads)
        for local_round in range(groups - 1):
            ordinal = local_round * 32 + block_idx
            projection_kind = 0
            logical = ordinal
            if ordinal >= full_qkv_count:
                projection_kind = 1
                logical = ordinal - full_qkv_count
                if logical >= full_gate_count:
                    projection_kind = 2
                    logical = logical - full_gate_count
            projection_head = logical // groups
            group = logical % groups
            self._run_frontend_main_task(
                full_stage,
                full_conv,
                full_decay,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                projection_kind,
                projection_head,
                group * 16,
                16,
                subblock_idx,
                k_tiles,
            )

        # M128 占用 L0C 空间。将地址和标识复用于 M64 前，先排空本核流水线；
        # LowerRingBuffer 负责结束配对 AIV 的 CrossCore 资源生命周期。
        # 这些操作都在核内完成，不使用网格屏障。
        cube_sync_all()
        vec_sync_all()
        onchip_rewind(reset_sync_id=True)
        if groups == 2 or (
            groups == 3 and block_idx < THREE_GROUP_HALF_PROJECTION_CORE_COUNT
        ):
            half_stage = ProjectionStage(
                row_count=64,
                row_capacity=64,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
            )
            half_conv = ConvSiluStage(8, 32, row_capacity=32)
            half_decay = LowRankDecayStage(half_stage, self.num_heads)
            projection_kind = 1
            logical = block_idx // 2 + 7
            if groups == 2:
                if block_idx < 8:
                    projection_kind = 0
                    logical = block_idx // 2 + 32
                elif block_idx < 28:
                    logical = (block_idx - 8) // 2
                else:
                    projection_kind = 2
                    logical = (block_idx - 28) // 2
            projection_head = logical // groups
            group = logical % groups
            batch_start = group * 16 + (block_idx % 2) * 8
            self._run_frontend_main_task(
                half_stage,
                half_conv,
                half_decay,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                decay_workspace_gm,
                output_gate_workspace_gm,
                hidden_gm,
                qkv_weight_gm,
                fa_weight_gm,
                fb_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                projection_kind,
                projection_head,
                batch_start,
                8,
                subblock_idx,
                k_tiles,
            )
            cube_sync_all()
            vec_sync_all()

        onchip_rewind(reset_sync_id=True)
        quarter_core = (
            0
            if groups == 2
            else THREE_GROUP_HALF_PROJECTION_CORE_COUNT
            if groups == 3
            else 8
        )
        quarter_count = 8 if groups == 2 else 20 if groups == 3 else 16
        if block_idx >= quarter_core and block_idx < quarter_core + quarter_count:
            quarter_stage = ProjectionStage(
                row_count=32,
                row_capacity=32,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
            )
            piece = block_idx - quarter_core
            first_gate = 10 if groups == 2 else 13 if groups == 3 else 20
            logical = first_gate + piece // 4
            projection_head = logical // groups
            group = logical % groups
            row_tile = group * 4 + piece % 4
            quarter_stage.project(
                tile_slice(hidden_gm, (32, self.hidden_size), (row_tile, 0)),
                output_gate_weight_gm,
                projection_head,
                k_tiles,
            )
            quarter_stage.publish(
                tile_slice(
                    output_gate_workspace_gm, (32, TILE), (row_tile, projection_head)
                ),
                subblock_idx,
            )
            cube_sync_all()
            vec_sync_all()

        onchip_rewind(reset_sync_id=True)
        beta_group = -1
        if groups == 2:
            if block_idx >= 8 and block_idx < 10:
                beta_group = block_idx - 8
        elif groups == 3:
            if (
                block_idx >= THREE_GROUP_HALF_PROJECTION_CORE_COUNT
                and block_idx <= 20
                and block_idx % 4 == 0
            ):
                beta_group = (block_idx - THREE_GROUP_HALF_PROJECTION_CORE_COUNT) // 4
        else:
            if block_idx >= FOUR_GROUP_BETA_CORE_START and block_idx < 28:
                beta_group = block_idx - FOUR_GROUP_BETA_CORE_START
        if beta_group >= 0:
            beta_stage = ProjectionStage(
                row_count=128,
                row_capacity=128,
                tile_width=BETA_TILE,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=frontend_weight_l2_cache_ctl,
            )
            beta_stage.project_to_gm(
                tile_slice(hidden_gm, (128, self.hidden_size), (beta_group, 0)),
                beta_weight_gm,
                tile_slice(beta_workspace_gm, (128, BETA_TILE), (beta_group, 0)),
                0,
                k_tiles,
            )

    @jit
    def _frontend_flat_m128(
        self,
        q_workspace_gm: Tensor,
        k_workspace_gm: Tensor,
        v_workspace_gm: Tensor,
        decay_workspace_gm: Tensor,
        output_gate_workspace_gm: Tensor,
        beta_workspace_gm: Tensor,
        hidden_gm: Tensor,
        qkv_weight_gm: Tensor,
        fa_weight_gm: Tensor,
        fb_weight_gm: Tensor,
        beta_weight_gm: Tensor,
        output_gate_weight_gm: Tensor,
        replay_decay_gm: ReplayPool,
        conv_state_gm,
        conv_state_indices_gm: Tensor,
        conv_accepted_gm: Tensor,
        conv_weight_gm: Tensor,
    ):
        self.seq_len = S8_SEQUENCE_LENGTH
        self.row_count = 128
        self.projection_row_block = 64
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        block_num = self.block_num
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        groups = replay_decay_gm.indices.shape[0] // BATCH_TILE
        task_count = groups * PROJECTION_TASK_COUNT
        round_count = ceil_div(task_count, block_num)
        hidden_tile = make_tiler((128, self.hidden_size), alignment=(16, 16))
        qkv_tile = make_tiler((128, self.num_heads * TILE), alignment=(16, 16))
        beta_tile = make_tiler((128, BETA_TILE), alignment=(16, 16))

        projection_stage = ProjectionStage(
            row_count=128,
            input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
            weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
            row_capacity=128,
        )
        conv_stage = ConvSiluStage(self.seq_len, 64, row_capacity=64)
        for local_round in range(round_count):
            task = local_round * block_num + block_idx
            if task < task_count:
                group = task // PROJECTION_TASK_COUNT
                projection = task - group * PROJECTION_TASK_COUNT
                batch_start = group * 16
                self._run_qkvg_task(
                    projection_stage,
                    conv_stage,
                    tile_slice(hidden_gm, hidden_tile, (group, 0)),
                    tile_slice(q_workspace_gm, qkv_tile, (group, 0)),
                    tile_slice(k_workspace_gm, qkv_tile, (group, 0)),
                    tile_slice(v_workspace_gm, qkv_tile, (group, 0)),
                    tile_slice(output_gate_workspace_gm, qkv_tile, (group, 0)),
                    qkv_weight_gm,
                    output_gate_weight_gm,
                    conv_state_gm,
                    conv_state_indices_gm,
                    conv_accepted_gm,
                    conv_weight_gm,
                    projection_kind=(0 if projection < 18 else 1),
                    projection_head=(
                        projection if projection < 18 else projection - 18
                    ),
                    batch_start=batch_start,
                    batch_count=16,
                    subblock_idx=subblock_idx,
                    k_tiles=k_tiles,
                )

        cube_sync_all()
        onchip_rewind(reset_sync_id=True)
        aux_kind = -1
        aux_group = 0
        tail_tasks = task_count % get_block_num()
        aux_start = 2 if tail_tasks == 0 else tail_tasks + 2
        beta_gap = groups if tail_tasks == 0 else 0
        beta_start = aux_start + groups + beta_gap
        if block_idx >= aux_start and block_idx < aux_start + groups:
            aux_kind = 0
            aux_group = block_idx - aux_start
        elif block_idx >= beta_start and block_idx < beta_start + groups:
            aux_kind = 1
            aux_group = block_idx - beta_start

        if aux_kind == 0:
            auxiliary_stage = ProjectionStage(
                row_count=128,
                row_capacity=128,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
            )
            low_rank_stage = LowRankDecayStage(auxiliary_stage, self.num_heads)
            low_rank_stage.project(
                tile_slice(decay_workspace_gm, qkv_tile, (aux_group, 0)),
                tile_slice(hidden_gm, hidden_tile, (aux_group, 0)),
                fa_weight_gm,
                fb_weight_gm,
                k_tiles,
                subblock_idx,
            )
        elif aux_kind == 1:
            beta_stage = ProjectionStage(
                row_count=128,
                row_capacity=128,
                tile_width=BETA_TILE,
                input_l2_cache_ctl=self.projection_input_l2_cache_ctl,
                weight_l2_cache_ctl=self.projection_weight_l2_cache_ctl,
            )
            beta_stage.project_to_gm(
                tile_slice(hidden_gm, hidden_tile, (aux_group, 0)),
                beta_weight_gm,
                tile_slice(beta_workspace_gm, beta_tile, (aux_group, 0)),
                0,
                k_tiles,
            )

    @jit
    def _frontend_24_single_group(
        self,
        q_workspace_gm,
        k_workspace_gm,
        v_workspace_gm,
        decay_workspace_gm,
        output_gate_workspace_gm,
        beta_workspace_gm,
        fa_workspace_gm,
        norm_exchange_gm,
        hidden_gm,
        qkv_weight_gm,
        fa_weight_gm,
        fb_weight_gm,
        beta_weight_gm,
        output_gate_weight_gm,
        conv_state_gm,
        conv_state_indices_gm,
        conv_accepted_gm,
        conv_weight_gm,
    ):
        block_idx = get_block_idx()
        subblock_idx = get_subblock_id()
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        if block_idx < 18:
            vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 3, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 4, mode=0)
            projection_stage = ProjectionStage(
                self.row_count,
                self.projection_input_l2_cache_ctl,
                self.projection_weight_l2_cache_ctl,
            )
            conv_stage = ConvSiluStage(self.seq_len, self.projection_row_block)
            self._run_qkvg_task(
                projection_stage,
                conv_stage,
                hidden_gm,
                q_workspace_gm,
                k_workspace_gm,
                v_workspace_gm,
                output_gate_workspace_gm,
                qkv_weight_gm,
                output_gate_weight_gm,
                conv_state_gm,
                conv_state_indices_gm,
                conv_accepted_gm,
                conv_weight_gm,
                0,
                block_idx,
                0,
                BATCH_TILE,
                subblock_idx,
                k_tiles,
            )
            vec_sync_block_wait(PIPE.S, 0, mode=0)
            vec_sync_block_wait(PIPE.S, 3, mode=0)
            vec_sync_block_wait(PIPE.S, 4, mode=0)
        else:
            onchip_rewind(reset_sync_id=True)
            if block_idx < 23:
                vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 4, mode=0)
                vec_sync_block_wait(PIPE.S, 0, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 1, mode=2)
                cube_sync_block_wait(PIPE.S, 1, mode=2)
                projection_stage = ProjectionStage(
                    self.row_count,
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                first_kc = (
                    (block_idx - 18) * FULL_FA_K_TILE_COUNT
                    if block_idx < 21
                    else 18 + (block_idx - 21) * 5
                )
                kc_count = FULL_FA_K_TILE_COUNT if block_idx < 21 else 5
                accumulator = from_tensor(
                    projection_stage.l0c.next_write(),
                    layout=make_layout(projection_stage.l0c_shape),
                    data_format="nz",
                )
                for local_kc in range(kc_count):
                    kc = first_kc + local_kc
                    a_slot = from_tensor(
                        projection_stage.a_l1.next_write(),
                        layout=make_layout(projection_stage.a_l1_shape),
                        data_format="nz",
                    )
                    b_slot = projection_stage.b_l1.next_write()
                    mem_copy(
                        a_slot,
                        tile_slice(hidden_gm, (128, 256), (0, kc)),
                        options=copy_options(
                            l2_cache_ctl=self.projection_input_l2_cache_ctl
                        ),
                    )
                    mem_copy(
                        b_slot,
                        tile_slice(fa_weight_gm, (128, 256), (0, kc)),
                        options=copy_options(
                            l2_cache_ctl=self.projection_weight_l2_cache_ctl
                        ),
                    )
                    for ks in range(2):
                        a_input = from_tensor(
                            projection_stage.l0a.next_write(),
                            layout=make_layout(projection_stage.l0a_shape),
                            data_format="nz",
                        )
                        b_input = projection_stage.l0b.next_write()
                        mem_copy(a_input, tile_slice(a_slot, (128, 128), (0, ks)))
                        mem_copy(b_input, tile_slice(b_slot, (128, 128), (0, ks)))
                        matmul(
                            accumulator,
                            a_input,
                            b_input,
                            init=local_kc == 0 and ks == 0,
                        )
                mem_copy(
                    norm_exchange_gm,
                    from_tensor(
                        projection_stage.l0c.next_read(),
                        layout=make_layout(projection_stage.l0c_shape),
                        data_format="nz",
                    ),
                    options=copy_options(atomic_add=True),
                )
                cube_sync_block_arrive(PIPE.FIXPIPE, 2, mode=2)
                vec_sync_block_wait(PIPE.S, 2, mode=2)
                vec_sync_block_arrive(PIPE.MTE3, 3, mode=0)
                vec_sync_block_wait(PIPE.S, 4, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 5, mode=2)
                projection_stage.project(
                    hidden_gm, output_gate_weight_gm, block_idx - 18, k_tiles
                )
                projection_stage.publish(
                    tile_slice(
                        output_gate_workspace_gm, (128, 128), (0, block_idx - 18)
                    ),
                    subblock_idx,
                )
                vec_sync_all()
                cube_sync_block_wait(PIPE.S, 5, mode=2)
                f_l1 = from_tensor(
                    RingBuffer(
                        MemLoc.L1,
                        dtypes.bfloat16,
                        make_layout((128 * 128,)),
                        depth=1,
                        data_format="nd",
                    ).next_write(),
                    layout=make_layout((self.row_count, 128)),
                    data_format="nz",
                )
                fb_l1 = RingBuffer(
                    MemLoc.L1, dtypes.bfloat16, make_layout((128, 128)), depth=1
                ).next_write()
                mem_copy(f_l1, fa_workspace_gm)
                mem_copy(
                    fb_l1, tile_slice(fb_weight_gm, (128, 128), (block_idx - 18, 0))
                )
                projection_stage.matmul_resident(f_l1, fb_l1)
                mem_copy(
                    tile_slice(decay_workspace_gm, (128, 128), (0, block_idx - 18)),
                    from_tensor(
                        projection_stage.l0c.next_read(),
                        layout=make_layout(projection_stage.l0c_shape),
                        data_format="nz",
                    ),
                )
                vec_sync_all()
                vec_sync_block_wait(PIPE.S, 3, mode=0)
            else:
                acc_half = Buffer(
                    MemLoc.UB, dtypes.float32, make_layout((64 * 128,))
                ).view(self.projection_row_block, 128)
                f_half = Buffer(
                    MemLoc.UB, dtypes.bfloat16, make_layout((64 * 128,))
                ).view(self.projection_row_block, 128)
                with vf(mode="simd"):
                    full, _ = update_mask(VL, elem_bits=32)
                    zero = vdups(0.0, dtypes.float32, mask=full)
                    for segment in range(self.projection_row_block * 128 // VL):
                        vstore(acc_half, segment * VL, zero, full)
                vec_sync_all()
                mem_copy(
                    tile_slice(
                        norm_exchange_gm,
                        make_tiler((self.projection_row_block, 128), alignment=(1, 1)),
                        (subblock_idx, 0),
                    ),
                    acc_half,
                )
                vec_sync_block_arrive(PIPE.MTE3, 0, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 3, mode=0)
                vec_sync_block_wait(PIPE.S, 3, mode=0)
                mem_copy(
                    acc_half,
                    tile_slice(
                        norm_exchange_gm,
                        make_tiler((self.projection_row_block, 128), alignment=(1, 1)),
                        (subblock_idx, 0),
                    ),
                )
                vec_sync_all()
                with vf(mode="simd"):
                    full, _ = update_mask(VL, elem_bits=32)
                    for segment in range(self.projection_row_block * 128 // VL):
                        offset = segment * VL
                        value = vcast(
                            vload(acc_half, offset), dtypes.bfloat16, mask=full
                        )
                        vstore_pack(
                            f_half, offset, value, full, pack_mode=PackMode.B32_TO_B16
                        )
                vec_sync_all()
                mem_copy(
                    tile_slice(
                        fa_workspace_gm,
                        make_tiler((self.projection_row_block, 128), alignment=(1, 1)),
                        (subblock_idx, 0),
                    ),
                    f_half,
                )
                vec_sync_block_arrive(PIPE.MTE3, 4, mode=0)
                vec_sync_block_wait(PIPE.S, 4, mode=0)
                vec_sync_block_arrive(PIPE.MTE3, 5, mode=2)
                fused = BetaZSharedStage(
                    self.row_count,
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                fused.project(
                    hidden_gm,
                    output_gate_weight_gm,
                    beta_weight_gm,
                    tile_slice(output_gate_workspace_gm, (128, 128), (0, 5)),
                    beta_workspace_gm,
                    subblock_idx,
                )
                cube_sync_block_wait(PIPE.S, 5, mode=2)
                onchip_rewind(reset_sync_id=True)
                projection_stage = ProjectionStage(
                    self.row_count,
                    self.projection_input_l2_cache_ctl,
                    self.projection_weight_l2_cache_ctl,
                )
                f_l1 = from_tensor(
                    RingBuffer(
                        MemLoc.L1,
                        dtypes.bfloat16,
                        make_layout((128 * 128,)),
                        depth=1,
                        data_format="nd",
                    ).next_write(),
                    layout=make_layout((self.row_count, 128)),
                    data_format="nz",
                )
                fb_l1 = RingBuffer(
                    MemLoc.L1, dtypes.bfloat16, make_layout((128, 128)), depth=1
                ).next_write()
                mem_copy(f_l1, fa_workspace_gm)
                mem_copy(fb_l1, tile_slice(fb_weight_gm, (128, 128), (5, 0)))
                projection_stage.matmul_resident(f_l1, fb_l1)
                mem_copy(
                    tile_slice(decay_workspace_gm, (128, 128), (0, 5)),
                    from_tensor(
                        projection_stage.l0c.next_read(),
                        layout=make_layout(projection_stage.l0c_shape),
                        data_format="nz",
                    ),
                )
                vec_sync_all()
                vec_sync_block_wait(PIPE.S, 0, mode=0)

    @jit
    def _frontend_large(
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
        self.row_count = hidden_gm.shape[0]
        self.projection_row_block = self.row_count // 2
        self.recurrent_wave_rows = self.row_count // self.recurrent_wave_count
        batches_per_subblock = self.batch_tile // 2
        block_idx = get_block_idx()
        block_num = get_block_num()
        subblock_idx = get_subblock_id()
        n_tiles = self.qkv_block_count
        k_tiles = ceil_div(hidden_gm.shape[1], TILE)
        if const_expr(self.block_num == 32):
            group_idx = block_idx // 16
            local_idx = block_idx % 16
            block_idx = (
                group_idx * 9 + local_idx
                if local_idx < 9
                else (
                    18 + group_idx * 3 + local_idx - 9
                    if local_idx < PROJECTION_CORES_PER_CORE_GROUP
                    else PROJECTION_TASK_COUNT + group_idx
                    if local_idx == PROJECTION_CORES_PER_CORE_GROUP
                    else 26 + group_idx * 3 + local_idx - 13
                )
            )
        if block_idx == self.beta_block:
            beta_stage = ProjectionStage(
                self.row_count,
                self.projection_input_l2_cache_ctl,
                self.projection_weight_l2_cache_ctl,
                tile_width=BETA_TILE,
            )
            beta_stage.project_to_gm(
                hidden_gm, beta_weight_gm, beta_workspace_gm, 0, k_tiles
            )
        else:
            onchip_rewind(reset_sync_id=True)
            projection_stage = ProjectionStage(
                self.row_count,
                self.projection_input_l2_cache_ctl,
                self.projection_weight_l2_cache_ctl,
            )
            low_rank_stage = LowRankDecayStage(projection_stage, self.num_heads)
            conv_stage = ConvSiluStage(self.seq_len, self.projection_row_block)
            selected_history = Buffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((batches_per_subblock * (CONV_KERNEL - 1), TILE)),
            )
            for n_idx in range(block_idx, n_tiles, block_num):
                projection_stage.matmul_hidden(hidden_gm, qkv_weight_gm, n_idx, k_tiles)
                for projection_head in range(3 * self.num_heads):
                    if n_idx == projection_head:
                        projection = projection_head // self.num_heads
                        value_head = projection_head - projection * self.num_heads
                        channel_tile = projection * self.num_heads + value_head
                        conv_stage.load_weight(
                            tile_slice(
                                conv_weight_gm[projection, value_head, None, None],
                                (CONV_KERNEL, TILE),
                                (0, 0),
                            )
                        )
                        for local_batch in range(batches_per_subblock):
                            batch_index = (
                                subblock_idx * batches_per_subblock + local_batch
                            )
                            cache_index = dtypes.int64(
                                conv_state_indices_gm[batch_index]
                            )
                            conv_accepted = dtypes.int64(conv_accepted_gm[batch_index])
                            history_offset = (
                                conv_accepted - 1 if conv_accepted > 0 else 0
                            )
                            if cache_index >= 0:
                                state_view = _conv_slot(conv_state_gm, cache_index)
                                for history_row in range(CONV_KERNEL - 1):
                                    mem_copy(
                                        tile_slice(
                                            selected_history,
                                            (1, TILE),
                                            (
                                                local_batch * (CONV_KERNEL - 1)
                                                + history_row,
                                                0,
                                            ),
                                        ),
                                        tile_slice(
                                            state_view,
                                            (1, TILE),
                                            (
                                                history_offset + history_row,
                                                channel_tile,
                                            ),
                                        ),
                                    )
                projection_stage.acquire_result()
                for projection_head in range(3 * self.num_heads):
                    if n_idx == projection_head:
                        projection = projection_head // self.num_heads
                        for local_batch in range(batches_per_subblock):
                            batch_index = (
                                subblock_idx * batches_per_subblock + local_batch
                            )
                            cache_index = dtypes.int64(
                                conv_state_indices_gm[batch_index]
                            )
                            if cache_index < 0:
                                conv_stage.zero_sequence(local_batch)
                            else:
                                history_view = tile_slice(
                                    selected_history,
                                    (CONV_KERNEL - 1, TILE),
                                    (local_batch, 0),
                                )
                                conv_stage.install_history(history_view)
                                state_view = _conv_slot(conv_state_gm, cache_index)
                                for cache_row in range(2):
                                    mem_copy(
                                        tile_slice(
                                            state_view,
                                            (1, TILE),
                                            (cache_row, projection_head),
                                        ),
                                        tile_slice(
                                            history_view, (1, TILE), (cache_row + 1, 0)
                                        ),
                                    )
                                conv_stage.load_accurate_sequence(
                                    conv_state_gm, batch_index, projection_head
                                )
                                conv_stage.convolve_sequence(
                                    local_batch,
                                    projection_stage.projection_fp32,
                                    tile_slice(
                                        conv_stage.weight, (CONV_KERNEL, TILE), (0, 0)
                                    ),
                                )
                        vec_sync_notify(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        vec_sync_wait(PIPE.V, PIPE.MTE3, CONV_RAW_V_TO_MTE3_EVENT)
                        for local_batch in range(batches_per_subblock):
                            batch_index = (
                                subblock_idx * batches_per_subblock + local_batch
                            )
                            cache_index = dtypes.int64(
                                conv_state_indices_gm[batch_index]
                            )
                            if cache_index >= 0:
                                state_view = _conv_slot(conv_state_gm, cache_index)
                                for cache_token in range(self.seq_len):
                                    mem_copy(
                                        tile_slice(
                                            state_view,
                                            (1, TILE),
                                            (cache_token + 2, projection_head),
                                        ),
                                        tile_slice(
                                            conv_stage.raw_bf16,
                                            (1, TILE),
                                            (
                                                local_batch * self.seq_len
                                                + cache_token,
                                                0,
                                            ),
                                        ),
                                    )
                vec_sync_all()
                if n_idx < self.num_heads:
                    projection_stage.publish_source(
                        tile_slice(q_workspace_gm, (128, TILE), (0, n_idx)),
                        conv_stage.output_bf16,
                        subblock_idx,
                    )
                elif n_idx < 2 * self.num_heads:
                    projection_stage.publish_source(
                        tile_slice(
                            k_workspace_gm, (128, TILE), (0, n_idx - self.num_heads)
                        ),
                        conv_stage.output_bf16,
                        subblock_idx,
                    )
                else:
                    projection_stage.publish_source(
                        tile_slice(
                            v_workspace_gm, (128, TILE), (0, n_idx - 2 * self.num_heads)
                        ),
                        conv_stage.output_bf16,
                        subblock_idx,
                    )
            if block_idx >= self.qkv_block_count and block_idx < self.beta_block:
                output_head = block_idx - self.qkv_block_count
                projection_stage.project(
                    hidden_gm, output_gate_weight_gm, output_head, k_tiles
                )
                projection_stage.publish(
                    tile_slice(output_gate_workspace_gm, (128, TILE), (0, output_head)),
                    subblock_idx,
                )
            if block_idx == self.low_rank_block:
                low_rank_stage.project(
                    decay_workspace_gm,
                    hidden_gm,
                    fa_weight_gm,
                    fb_weight_gm,
                    k_tiles,
                    subblock_idx,
                )
