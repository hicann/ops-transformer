# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mega KDA 的 PA 视图、通用矩阵乘及存储搬运。"""

from __future__ import annotations

import dataclasses
import torch
from cannbotdsl import MemLoc, Tensor, dtypes, from_tensor, make_layout
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import PackMode, update_mask, vcast, vload, vstore_pack
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.tensor import make_tiler, tile_slice
from typing import NamedTuple


# 模型形状
MODEL_HEAD_COUNT = 96
HIDDEN_SIZE = 7168
D = 128

# TP 度数和每分片 head 数
TP4_TP_DEGREE = 4
TP8_TP_DEGREE = 8
TP16_TP_DEGREE = 16
TP4_HEADS = MODEL_HEAD_COUNT // TP4_TP_DEGREE
TP8_HEADS = MODEL_HEAD_COUNT // TP8_TP_DEGREE
TP16_HEADS = MODEL_HEAD_COUNT // TP16_TP_DEGREE

# 序列及卷积形状
S4_SEQUENCE_LENGTH = 4
S8_SEQUENCE_LENGTH = 8
CONV_KERNEL = 4

# 共享 tile 与向量宽度
TILE = 128
VL = 64
BETA_TILE = 16
BATCH_TILE = 16

# 矩阵乘 K 分块
PROJECTION_KC = 256
PROJECTION_KS = 128
OPROJ_KC = 256
OPROJ_KS = 64

# TP16 N384 投影面板
QKV_PROJECTION_COUNT = 3
PROJECTION_PANELS_PER_N384_TASK = 3
QKV_PANEL_COUNT = QKV_PROJECTION_COUNT * TP16_HEADS // PROJECTION_PANELS_PER_N384_TASK


class ReplayPool(NamedTuple):
    """将 PA 存储展平，并保留独立的元素步长和逻辑 batch 映射。"""

    data: Tensor
    indices: Tensor
    stride0: int
    heads: int
    capacity: int


class _CheckpointPool(NamedTuple):
    data: Tensor
    indices: Tensor
    stride0: int
    accurate_decay: Tensor


class _ConvPool(NamedTuple):
    data: Tensor
    stride0: int
    length: int
    width: int
    accurate_raw: Tensor


@jit
def replay_head(pool, batch, head):
    slot = dtypes.int64(pool.indices[batch])
    begin = slot * pool.stride0 + head * pool.capacity * TILE
    return pool.data[begin : begin + pool.capacity * TILE,].view(pool.capacity, TILE)


@jit
def replay_batches(pool, begin, end):
    return ReplayPool(
        pool.data, pool.indices[begin:end,], pool.stride0, pool.heads, pool.capacity
    )


@jit
def _checkpoint_head(pool, batch, head):
    slot = dtypes.int64(pool.indices[batch])
    begin = slot * pool.stride0 + head * TILE * TILE
    return pool.data[begin : begin + TILE * TILE,].view(TILE, TILE)


@jit
def _checkpoint_batches(pool, begin, end):
    """切分逻辑 batch，同时保留原有 PA 状态池的完整物理存储。"""
    return _CheckpointPool(
        pool.data,
        pool.indices[begin:end,],
        pool.stride0,
        pool.accurate_decay[
            begin * S8_SEQUENCE_LENGTH : end * S8_SEQUENCE_LENGTH, None
        ],
    )


@jit
def _conv_slot(pool, slot):
    begin = slot * pool.stride0
    return pool.data[begin : begin + pool.length * pool.width,].view(
        pool.length, pool.width
    )


def _physical_span(tensor):
    span = (tensor.shape[0] - 1) * tensor.stride(0) + tensor[0].numel()
    return tensor.as_strided((span,), (1,))


class _ProjectionOperations:
    """为缓冲区布局各异的计算阶段提供共用投影操作。"""

    projection_fp32: object

    @jit
    def matmul_hidden(self, hidden_gm: Tensor, weight_gm: Tensor, output_tile, k_tiles):
        """使用模型布局 [out,in] 的权重计算 hidden @ weight.T。"""
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        if const_expr(self.tile_width == BETA_TILE):
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
        for kc_idx in range(k_tiles * TILE // PROJECTION_KC):
            a_l1_slot = from_tensor(
                self.a_l1.next_write(),
                layout=make_layout(self.a_l1_shape),
                data_format="nz",
            )
            mem_copy(
                a_l1_slot,
                tile_slice(hidden_gm, (self.row_capacity, PROJECTION_KC), (0, kc_idx)),
                options=copy_options(l2_cache_ctl=self.input_l2_cache_ctl),
            )
            if const_expr(self.tile_width == BETA_TILE):
                self._accumulate_panel(
                    accumulator,
                    a_l1_slot,
                    beta_l1,
                    kc_idx * (PROJECTION_KC // PROJECTION_KS),
                    kc_idx == 0,
                )
            else:
                b_l1_slot = self.b_l1.next_write()
                mem_copy(
                    b_l1_slot,
                    tile_slice(
                        weight_gm,
                        (self.tile_width, PROJECTION_KC),
                        (output_tile, kc_idx),
                    ),
                    options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl),
                )
                self._accumulate_panel(
                    accumulator, a_l1_slot, b_l1_slot, 0, kc_idx == 0
                )

    @jit
    def _accumulate_panel(self, accumulator, a_l1_slot, b_l1_slot, first_k, init):
        for ks_idx in range(PROJECTION_KC // PROJECTION_KS):
            l0a_slot = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            l0b_slot = self.l0b.next_write()
            mem_copy(
                l0a_slot,
                tile_slice(a_l1_slot, (self.row_capacity, PROJECTION_KS), (0, ks_idx)),
            )
            mem_copy(
                l0b_slot,
                tile_slice(
                    b_l1_slot, (self.tile_width, PROJECTION_KS), (0, first_k + ks_idx)
                ),
            )
            matmul(accumulator, l0a_slot, l0b_slot, init=init and ks_idx == 0)

    @jit
    def matmul_resident(self, resident_l1, weight_l1):
        """使用驻留 L1 的 BF16 分块计算一次 K=128 投影。"""
        accumulator = from_tensor(
            self.l0c.next_write(), layout=make_layout(self.l0c_shape), data_format="nz"
        )
        for ks_idx in range(TILE // PROJECTION_KS):
            l0a_slot = from_tensor(
                self.l0a.next_write(),
                layout=make_layout(self.l0a_shape),
                data_format="nz",
            )
            l0a_slot_source = tile_slice(
                resident_l1, (self.row_capacity, PROJECTION_KS), (0, ks_idx)
            )
            l0a_slot = l0a_slot[tuple(map(slice, l0a_slot_source.shape))]
            mem_copy(l0a_slot, l0a_slot_source)
            l0b_slot = self.l0b.next_write()
            l0b_slot_source = tile_slice(
                weight_l1, (self.tile_width, PROJECTION_KS), (0, ks_idx)
            )
            l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source.shape))]
            mem_copy(l0b_slot, l0b_slot_source)
            matmul(accumulator, l0a_slot, l0b_slot, init=ks_idx == 0)

    @jit
    def acquire_result(self):
        """将 L0C 发送给两个配对 AIV，并保留 FP32 半块。"""
        mem_copy(
            self.projection_half,
            from_tensor(
                self.l0c.next_read(),
                layout=make_layout(self.l0c_shape),
                data_format="nz",
            ).partition(dim=0),
        )
        self.projection_fp32 = self.projection_half

    @jit
    def cast_result(self):
        """将 L0C 发送给两个配对 AIV，再分别转换各 AIV 负责的半块。"""
        self.acquire_result()
        with vf(mode="simd"):
            full, _ = update_mask(VL, elem_bits=32)
            for segment in range(self.row_block * self.tile_width // VL):
                offset = segment * VL
                projected = vload(self.projection_fp32, offset)
                projected_bf16 = vcast(projected, dtypes.bfloat16, mask=full)
                vstore_pack(
                    self.projection_bf16,
                    offset,
                    projected_bf16,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
        vec_sync_all()

    @jit
    def project(self, hidden_gm: Tensor, weight_gm: Tensor, n_idx, k_tiles):
        """计算一个输出分块，并转换当前 AIV 负责的半块。"""
        self.matmul_hidden(hidden_gm, weight_gm, n_idx, k_tiles)
        self.cast_result()

    @jit
    def project_to_gm(self, hidden_gm, weight_gm, workspace_tile, n_idx, k_tiles):
        """通过 Fixpipe 将完整投影舍入为 BF16，无需 AIV 再做类型转换。"""
        self.matmul_hidden(hidden_gm, weight_gm, n_idx, k_tiles)
        self.publish_result(workspace_tile)

    @jit
    def publish_result(self, workspace_tile):
        mem_copy(
            workspace_tile[0 : self.row_count, 0 : self.tile_width],
            from_tensor(
                self.l0c.next_read(),
                layout=make_layout(self.l0c_shape),
                data_format="nz",
            ),
        )

    def publish(self, workspace_tile, subblock_idx):
        out_half = tile_slice(
            workspace_tile,
            make_tiler((self.row_block, self.tile_width), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, self.projection_bf16)

    def publish_source(self, workspace_tile, source, subblock_idx):
        out_half = tile_slice(
            workspace_tile,
            make_tiler((self.row_block, self.tile_width), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, source)


@dataclasses.dataclass(frozen=True)
class Schedule:
    """保存当前 TP 的编译期分块与缓存策略。"""

    ragged_output: bool = False
    ragged_main: bool = False
    main_rows: int = 128
    auxiliary_rows: int = 0
    main_width: int = 384
    split_fb: bool = True
    publish_rows: int = 64
    resident_output_weight: bool = True
    state_update_unroll: int = 2
    main_weight_l2_cache_ctl: int = 4
    state_store_l2_cache_ctl: int = 4
    state_load_l2_cache_ctl: int = 2
    fa_width: int = 128
    auxiliary_weight_l2_cache_ctl: int = 4
    output_kc: int = 0

    def __post_init__(self):
        if self.fa_width not in (32, 64, 128):
            raise ValueError("unsupported Fa partition width")
        if self.fa_width != 128 and (not self.split_fb):
            raise ValueError("partitioned Fa requires split_fb")
        if self.main_weight_l2_cache_ctl not in (0, 4):
            raise ValueError("main_weight_l2_cache_ctl must be 0 or 4")
        if self.auxiliary_weight_l2_cache_ctl not in (0, 4):
            raise ValueError("auxiliary_weight_l2_cache_ctl must be 0 or 4")
        if self.output_kc not in (0, 128, 256, 384):
            raise ValueError("output_kc must be 0,128,256 or384")
        if self.state_store_l2_cache_ctl not in (0, 4):
            raise ValueError("state_store_l2_cache_ctl must be 0 or 4")
        if self.state_load_l2_cache_ctl not in (0, 2, 4):
            raise ValueError("state_load_l2_cache_ctl must be 0, 2 or 4")
        if self.ragged_main and self.main_width != 256:
            raise ValueError("ragged_main requires N256")
        if self.main_width not in (128, 256, 384):
            raise ValueError("main_width must be 128,256,384")
        if self.main_rows not in (32, 64, 128, 256):
            raise ValueError("main_rows must be 32,64,128,256")
        if self.main_rows == 256 and self.main_width != 256:
            raise ValueError("M256 requires the N256 projection stage")
        if self.auxiliary_rows not in (0, 16, 32, 64, 128):
            raise ValueError("auxiliary_rows must be 0,16,32,64,128")
        if self.publish_rows not in (16, 32, 64, 128, 256):
            raise ValueError("publish_rows must be 16,32,64,128,256")
        if self.state_update_unroll not in (1, 2, 4):
            raise ValueError("state_update_unroll must be 1,2,4")


def _pack_conv_weights(conv1d_weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    if tuple(conv1d_weight.shape) == (3, num_heads, CONV_KERNEL, TILE):
        return conv1d_weight
    if tuple(conv1d_weight.shape) == (3, CONV_KERNEL, TILE):
        return (
            conv1d_weight[:, None, :, :]
            .expand(3, num_heads, CONV_KERNEL, TILE)
            .contiguous()
        )
    if tuple(conv1d_weight.shape) == (CONV_KERNEL, 3 * num_heads * TILE):
        return (
            conv1d_weight.view(CONV_KERNEL, 3, num_heads, TILE)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
    raise ValueError("conv1d_weight must be [3,N,4,D], [3,4,D], or [4,3*N*D]")


def _mega_workspace_views(workspace, row_count, widths):
    regions = torch.split(workspace, [row_count * width for width in widths])
    views = []
    for region, width in zip(regions, widths):
        views.append(region.view(row_count, width))
    return tuple(views)


@jit
def _copy_weight_with_policy(dst, src, policy):
    if policy == 0:
        mem_copy(dst, src, options=copy_options(l2_cache_ctl=0))
    else:
        mem_copy(dst, src, options=copy_options(l2_cache_ctl=4))


class _Projection384Operations:
    """为缓冲区布局各异的计算阶段提供共用投影操作。"""

    projection_fp32: object

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
                tile_slice(hidden_gm, (self.rows, PROJECTION_KC), (m_group, kc_idx)),
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
                l0a_slot_source_3 = tile_slice(
                    a_l1_slot, (self.rows, PROJECTION_KS), (0, ks_idx)
                )
                l0a_slot = l0a_slot[tuple(map(slice, l0a_slot_source_3.shape))]
                mem_copy(l0a_slot, l0a_slot_source_3)
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_8 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (0, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_8.shape))]
                mem_copy(l0b_slot, l0b_slot_source_8)
                matmul(self.l0c0, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half0, self.l0c0.partition(dim=0))
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_9 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (1, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_9.shape))]
                mem_copy(l0b_slot, l0b_slot_source_9)
                matmul(self.l0c1, l0a_slot, l0b_slot, init=kc_idx == 0 and ks_idx == 0)
                if (
                    kc_idx == k_tiles * TILE // PROJECTION_KC - 1
                    and ks_idx == PROJECTION_KC // PROJECTION_KS - 1
                ):
                    mem_copy(self.projection_half1, self.l0c1.partition(dim=0))
                l0b_slot = self.l0b.next_write()
                l0b_slot_source_10 = tile_slice(
                    b_l1_slot, (TILE, PROJECTION_KS), (2, ks_idx)
                )
                l0b_slot = l0b_slot[tuple(map(slice, l0b_slot_source_10.shape))]
                mem_copy(l0b_slot, l0b_slot_source_10)
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
            for segment in range(self.rows // 2 * TILE // VL):
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

    @jit
    def publish(self, workspace_gm, m_group, output_head, subblock_idx, source):
        if const_expr(self.allow_tail):
            start = m_group * self.rows + subblock_idx * (self.rows // 2)
            valid = workspace_gm.shape[0] - start
            if valid > self.rows // 2:
                valid = self.rows // 2
            if valid > 0:
                mem_copy(
                    workspace_gm[
                        start : start + valid,
                        output_head * TILE : (output_head + 1) * TILE,
                    ],
                    source[0:valid, None],
                )
        else:
            self.publish_full(workspace_gm, m_group, output_head, subblock_idx, source)

    def publish_full(self, workspace_gm, m_group, output_head, subblock_idx, source):
        whole_tile = tile_slice(workspace_gm, (self.rows, TILE), (m_group, output_head))
        out_half = tile_slice(
            whole_tile,
            make_tiler((self.rows // 2, TILE), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, source)


class _Core32ProjectionOperations(_Projection384Operations):
    """为缓冲区布局各异的计算阶段提供共用投影操作。"""

    @jit
    def _copy_projection_weight(self, dst, src):
        mem_copy(dst, src, options=copy_options(l2_cache_ctl=self.weight_l2_cache_ctl))

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
                tile_slice(
                    hidden_gm,
                    make_tiler((self.rows, PROJECTION_KC), alignment=(16, 16)),
                    (m_group, kc_idx),
                ),
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

    def publish_full(self, workspace_gm, m_group, output_head, subblock_idx, source):
        whole_tile = tile_slice(
            workspace_gm,
            make_tiler((self.rows, TILE), alignment=(16, 16)),
            (m_group, output_head),
        )
        out_half = tile_slice(
            whole_tile,
            make_tiler((self.rows // 2, TILE), alignment=(1, 1)),
            (subblock_idx, 0),
        )
        mem_copy(out_half, source)

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
                    tile_slice(
                        x, make_tiler((self.rows, 256), alignment=(16, 16)), (group, kc)
                    ),
                    options=copy_options(l2_cache_ctl=0),
                )
            self._copy_projection_weight(b, tile_slice(w, (256, 256), (macro, kc)))
            for ks in range(4):
                l0a = from_tensor(
                    self.l0a.next_write(),
                    layout=make_layout(self.l0a_shape),
                    data_format="nz",
                )
                l0a_source_2 = tile_slice(a, (256, 64), (0, ks * 64 // 64))[
                    0 : self.rows, None
                ]
                l0a = l0a[tuple(map(slice, l0a_source_2.shape))]
                mem_copy(l0a, l0a_source_2)
                l0b = self.l0b.next_write()
                l0b_source_3 = tile_slice(b, (128, 64), (0, ks))
                l0b = l0b[tuple(map(slice, l0b_source_3.shape))]
                mem_copy(l0b, l0b_source_3)
                matmul(self.l0c0, l0a, l0b, init=kc == 0 and ks == 0)
                l0b = self.l0b.next_write()
                l0b_source_4 = tile_slice(b, (128, 64), (1, ks))
                l0b = l0b[tuple(map(slice, l0b_source_4.shape))]
                mem_copy(l0b, l0b_source_4)
                matmul(self.l0c1, l0a, l0b, init=kc == 0 and ks == 0)


class _DynamicProjectionOperations(_Core32ProjectionOperations):
    """复用投影组件，并在运行时选择权重缓存策略。"""

    @jit
    def _copy_projection_weight(self, dst, src):
        _copy_weight_with_policy(dst, src, self.weight_l2_cache_ctl)
