# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""mla_prolog_v4: device computation, Host planning and public integration."""

import atexit
import hashlib
import inspect
import json
import math
import os
import struct
import threading
from dataclasses import dataclass, fields, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cannbotdsl as dsl
import torch
from cannbotdsl import Dim, MemLoc, ProfileSpec, Tensor, TensorSpec, dtypes, from_tensor
from cannbotdsl.aot import compile_cache, export
from cannbotdsl.aot.cache import observe_compilations
from cannbotdsl.aot.collection import current_collection
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    RegLayout,
    RoundingMode,
    UnpackMode,
    full_mask,
    update_mask,
    vabs,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdups,
    veq,
    vload,
    vload_broadcast,
    vload_deinterleave,
    vload_unpack,
    vmax,
    vmaxs,
    vmem_bar,
    vmins,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_sum,
    vreinterpret,
    vselect,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.sync import cube_sync_all, global_sync_all, onchip_rewind
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import make_layout, make_tiler, tile_slice

# Device computation, layouts and synchronization.

PA_NZ_PLANE_ELEMS = 16
KV_BYTES = 512
DESCALE_BYTES = 16
DTILE_MAX = KV_BYTES + 2 * 64 + DESCALE_BYTES
CUBE_BN = 96
CUBE_BK = 128
QB_GROUP_HEADS = 4
BKW = 64
NZ_PANEL_K = 256
WKB_STREAM_MIN_HEADS = 64
L2_LOAD_NOTALLOC_KEEP = 4
M_TAIL_ROWS = 16
QB_COLUMNS = 128
BF16_NZ_COLUMNS = 16
CUBE_M_ALIGNMENT = 16
VECTOR_SEG = 64
# The partial-QA reduction deliberately keeps four pairwise-summed inputs.
QA_SPLIT_PARTS = 4
GLOBAL_FRONT_L1_K = 256
GLOBAL_FRONT_L0_K = 128
GLOBAL_FRONT_L0_PANELS = GLOBAL_FRONT_L1_K // GLOBAL_FRONT_L0_K
GLOBAL_FRONT_BUFFER_DEPTH = 2
HALF = VECTOR_SEG // 2
PAD_SEG = 2 * VECTOR_SEG
ROPE_HEAD_BATCH = 4
ROPE_BATCH_MIN_TILE_ROWS = 128
VECTOR_Q_LORA_MAX = 2048
VECTOR_KV_LORA = 512
VECTOR_KVR_MAX = 576
GROUPS = 4
GROUP_SIZE = 128
GROUP_AMAX_FLOOR = 0.0001
FP8_MAX = 448.0
HEAD_DIM_MAX = 256
PA_NZ_FP8_PLANE_ELEMS = 32
HIF8_MAX = 32768.0
AMAX_VECTOR_LANES = 64
QB_WIDE_PANEL_MAX_ROWS = 8
QB_MEDIUM_PANEL_MAX_ROWS = 64
FRONT_ACTUAL_ROWS_MAX = 64
# All modes use BF16 CQ and WQB output; every Matmul accumulates in FP32 L0C.
# RoPE buffers are planned for the enabled path; runtime flags control execution.
_SCRATCH_TOKEN_CAPACITY = 1024
_SCRATCH_MIN_CHUNK_ROWS = 128
_SCRATCH_TARGET_BYTES = 32 * 1024 * 1024
_QB_RUNTIME_WIDTHS = (96, 128, 192, 256)
BN = 96
BK = 128
BKA = 64
SEG = 64
KV_LORA = 512
DESCALE_TAIL_BYTES = 16
TOKEN_PROFILE_BOUNDS = {
    1: (1, 1),
    8: (2, 8),
    16: (9, 16),
    32: (17, 32),
    64: (33, 64),
    128: (65, 128),
}
GENERIC_PROFILE = 0
TOKEN_PROFILES = (8, 32, 64, GENERIC_PROFILE)
L1_BYTES = 512 * 1024
L0A_BYTES = 64 * 1024
L0B_BYTES = 64 * 1024
L0C_BYTES = 256 * 1024
UB_BYTES = 248 * 1024
L2_BYTES = 128 * 1024 * 1024
Q_LORA_VALUES = (640, 1024, 1536, 2048)
Q_LORA_MAX = 2048
KVR_MAX = 576
HEADS_DOMAIN_MAX = 128
ITEM_A16 = 2
ITEM_F32 = 4
_SCRATCH_ALIGNMENT_BYTES = 32


def dtile_for(rope_dim):
    return KV_BYTES + 2 * rope_dim + DESCALE_BYTES


@jit
def cache_row_view(cache, page):
    return cache[page, None, 0, None].view((cache.shape[1], cache.shape[3]))


@jit
def write_row_panz(cache, cache_index, payload, token):
    slot = cache_index[token]
    if slot >= 0:
        planes = cache.shape[3] // PA_NZ_PLANE_ELEMS
        block = slot // cache.shape[1]
        physical = cache[block, None, 0, None].view(
            (planes * cache.shape[1], PA_NZ_PLANE_ELEMS)
        )
        row = slot % cache.shape[1]
        for plane in dsl.range(0, planes):
            mem_copy(
                tile_slice(
                    physical,
                    (1, PA_NZ_PLANE_ELEMS),
                    (plane * cache.shape[1] + row, 0),
                ),
                tile_slice(payload, (1, PA_NZ_PLANE_ELEMS), (0, plane)),
            )


@jit
def write_cache_row(cache, cache_index, payload, token, row_tiler, cache_mode):
    if cache_mode == 0:
        slot = cache_index[token]
        if slot >= 0:
            cache_2d = cache_row_view(cache, slot // cache.shape[1])
            valid_payload = from_tensor(
                payload, layout=make_layout((1, cache.shape[3]))
            )
            mem_copy(
                tile_slice(cache_2d, row_tiler, (slot % cache.shape[1], 0)),
                valid_payload,
            )
    else:
        write_row_panz(cache, cache_index, payload, token)


@jit
def write_cache_row_group_fp8(cache, cache_index, payload_read, token, rope_dim):
    slot = cache_index[token]
    if slot >= 0:
        dtile = dtile_for(rope_dim)
        row_tiler = make_tiler((1, dtile), alignment=(1, DESCALE_BYTES))
        cache_2d = cache_row_view(cache, slot // cache.shape[1])
        payload = from_tensor(payload_read, layout=make_layout((1, dtile)))
        mem_copy(
            tile_slice(cache_2d, row_tiler, (slot % cache.shape[1], 0)),
            payload,
        )


def nz_tile_width(output_extent):
    for width in (96, 128, 64):
        if output_extent % width == 0:
            return width
    raise ValueError(f"no divisible NZ tile width for O={output_extent}")


def nz_qb_tile_width(output_extent, heads, qkv_quant_mode):
    if qkv_quant_mode == 0 and heads >= WKB_STREAM_MIN_HEADS:
        for width in (256, 192):
            if output_extent % width == 0:
                return width
    return nz_tile_width(output_extent)


def nz_front_tile_width(q_lora, kvr):
    common = math.gcd(int(q_lora), int(kvr))
    for width in (96, 128, 64):
        if common % width == 0:
            return width
    raise ValueError(
        f"no divisible shared front NZ tile width for (Ql={q_lora}, Kvr={kvr})"
    )


class NdProjection:
    def __init__(self, bm, l2_hint):
        self.bm = bm
        self.l2_hint = l2_hint
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((bm, CUBE_BK)),
            depth=2,
            data_format="nz",
        )
        self.b_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CUBE_BN, CUBE_BK)),
            depth=2,
            data_format="nz",
        )
        self.a_l0 = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((bm, CUBE_BK)), depth=2
        )
        # Cube consumes B as N x K; 0.8 no longer rebinds its layout on copy.
        self.b_l0 = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((CUBE_BN, CUBE_BK)), depth=2
        )
        self.c_l0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((bm, CUBE_BN)), depth=2
        )

    @jit
    def project_tile(self, a_gm, weight, out_gm, k_total, n_off, width, m_tiles):
        bm = self.bm
        for m_tile in dsl.range(0, m_tiles):
            c_view = self.c_l0.next_write()
            for k_idx in dsl.range(0, k_total // CUBE_BK):
                mem_copy(
                    self.a_l1.next_write(),
                    tile_slice(a_gm, (bm, CUBE_BK), (m_tile, k_idx)),
                    options=copy_options(l2_cache_ctl=self.l2_hint),
                )
                mem_copy(self.a_l0.next_write(), self.a_l1.next_read())
                mem_copy(
                    self.b_l1.next_write(),
                    tile_slice(weight, (width, CUBE_BK), (n_off // width, k_idx)),
                )
                mem_copy(self.b_l0.next_write(), self.b_l1.next_read())
                matmul(
                    c_view,
                    self.a_l0.next_read(),
                    self.b_l0.next_read(),
                    init=k_idx == 0,
                )
            mem_copy(
                tile_slice(out_gm, (bm, CUBE_BN), (m_tile, n_off // CUBE_BN)),
                self.c_l0.next_read(),
            )

    @jit
    def project_owner(
        self, a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
    ):
        full = n_total // CUBE_BN
        for c in dsl.range(block, full, blocks):
            self.project_tile(a_gm, weight, out_gm, k_total, c * CUBE_BN, 96, m_tiles)
        tail = n_total - full * CUBE_BN
        if tail == 64:
            if full % blocks == block:
                self.project_tile(
                    a_gm, weight, out_gm, k_total, full * CUBE_BN, 64, m_tiles
                )
        if tail == 32:
            if full % blocks == block:
                self.project_tile(
                    a_gm, weight, out_gm, k_total, full * CUBE_BN, 32, m_tiles
                )


class QbFourHeads:
    def __init__(self, bm, head_dim, q_lora):
        self.bm = bm
        self.head_dim = head_dim
        self.group_heads = 2 if bm == 16 else QB_GROUP_HEADS
        self.panel_k = 256 if bm == 16 else CUBE_BK
        self.width = self.group_heads * head_dim
        self.q_lora = q_lora
        self.b_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((self.panel_k, self.width)),
            depth=2,
            data_format="nz",
        )
        self.a_full = Buffer(MemLoc.L0A, dtypes.bfloat16, make_layout((bm, q_lora)))
        self.b_l0 = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((head_dim, CUBE_BK)), depth=1
        )
        self.c_all = Buffer(MemLoc.L0C, dtypes.float32, make_layout((bm, self.width)))

    @jit
    def project_tile(self, weight, out_gm, group, m_tile, valid_heads):
        columns = valid_heads * self.head_dim
        for k_idx in range_constexpr(self.q_lora // self.panel_k):
            mem_copy(
                tile_slice(self.b_l1.next_write(), (self.panel_k, columns), (0, 0)),
                tile_slice(weight, (self.panel_k, columns), (k_idx, group)),
                options=copy_options(l2_cache_ctl=4),
            )
            b_panel = self.b_l1.next_read().permute(1, 0)
            for inner in range_constexpr(self.panel_k // CUBE_BK):
                a_view = from_tensor(
                    self.a_full,
                    layout=make_layout((self.bm, CUBE_BK)),
                    offset=(k_idx * self.panel_k + inner * CUBE_BK) * 16 * ITEM_A16,
                    data_format="nz",
                )
                for head_part in range_constexpr(valid_heads):
                    mem_copy(
                        self.b_l0.next_write(),
                        tile_slice(
                            b_panel, (self.head_dim, CUBE_BK), (head_part, inner)
                        ),
                    )
                    c_view = from_tensor(
                        self.c_all,
                        layout=make_layout((self.bm, self.head_dim)),
                        offset=head_part * self.head_dim * 16 * ITEM_F32,
                        data_format="nz",
                    )
                    matmul(
                        c_view,
                        a_view,
                        self.b_l0.next_read(),
                        init=k_idx == 0 and inner == 0,
                    )
        mem_copy(
            tile_slice(out_gm, (self.bm, columns), (m_tile, group)),
            tile_slice(self.c_all, (self.bm, columns), (0, 0)),
        )

    @jit
    def project_owner(self, a_gm, weight, out_gm, n_total, m_tiles, block, blocks):
        full_groups = n_total // self.width
        tail_columns = n_total % self.width
        groups = (n_total + self.width - 1) // self.width
        full_weight = weight
        full_output = out_gm
        if block < groups:
            for m_tile in dsl.range(0, m_tiles):
                staging_write = from_tensor(
                    self.b_l1.next_write(),
                    layout=make_layout((self.bm, self.q_lora)),
                    data_format="nz",
                )
                mem_copy(
                    staging_write,
                    tile_slice(a_gm, (self.bm, self.q_lora), (m_tile, 0)),
                    options=copy_options(l2_cache_ctl=1),
                )
                staging_read = from_tensor(
                    self.b_l1.next_read(),
                    layout=make_layout((self.bm, self.q_lora)),
                    data_format="nz",
                )
                mem_copy(self.a_full, staging_read)
                for group in dsl.range(block, groups, blocks):
                    if group < full_groups:
                        self.project_tile(
                            full_weight,
                            full_output,
                            group,
                            m_tile,
                            self.group_heads,
                        )
                    else:
                        for tail_heads in range_constexpr(1, self.group_heads):
                            if tail_columns == tail_heads * self.head_dim:
                                self.project_tile(
                                    tile_slice(
                                        weight,
                                        (self.q_lora, self.width),
                                        (0, full_groups),
                                    ),
                                    tile_slice(
                                        out_gm,
                                        make_tiler(
                                            (out_gm.shape[0], self.width),
                                            alignment=(1, 1),
                                        ),
                                        (0, full_groups),
                                    ),
                                    0,
                                    m_tile,
                                    tail_heads,
                                )


class _TransposedWeightRing:
    def __init__(self, ring):
        self._ring = ring

    def next_write(self):
        return self._ring.next_write().permute(1, 0)

    def next_read(self):
        return self._ring.next_read().permute(1, 0)


class NzProjection:
    def __init__(
        self,
        bm,
        width,
        l2_hint,
        weight_nz=True,
        resident_capable=False,
        panel_k=CUBE_BK,
        weight_l2_hint=0,
        use_actual_rows=False,
    ):
        self.weight_nz = weight_nz
        self.bm = bm
        self.width = width
        self.l2_hint = l2_hint
        self.weight_l2_hint = weight_l2_hint
        self.use_actual_rows = use_actual_rows
        self.physical_rows = (
            (bm + CUBE_M_ALIGNMENT - 1) // CUBE_M_ALIGNMENT * CUBE_M_ALIGNMENT
        )
        self.panel_k = CUBE_BK if resident_capable else panel_k
        self.a_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((bm, self.panel_k)),
            depth=2,
            data_format="nz",
        )
        self.b_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((self.panel_k, width) if weight_nz else (width, self.panel_k)),
            depth=2,
            data_format="nz",
        )
        if weight_nz:
            self.b_l1 = _TransposedWeightRing(self.b_l1)
        self.a_l0 = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((bm, CUBE_BK)), depth=2
        )
        self.b_l0 = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((width, CUBE_BK)),
            depth=1 if width > 128 else 2,
        )
        self.c_l0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((bm, width)), depth=2
        )

        self.resident_capable = resident_capable
        if resident_capable:
            self.a_full = Buffer(MemLoc.L1, dtypes.bfloat16, make_layout((bm, 1024)))

    @jit
    def copy_weight(self, destination, weight, shape, coord, options=None):
        if self.weight_nz:
            mem_copy(
                destination.permute(1, 0),
                tile_slice(weight, (shape[1], shape[0]), (coord[1], coord[0])),
                options=options,
            )
        else:
            mem_copy(destination, tile_slice(weight, shape, coord), options=options)

    @jit
    def project_tile(self, a_gm, weight, out_gm, k_total, n_idx, m_tiles, m_begin=0):
        bm = self.bm
        width = self.width
        for m_tile in dsl.range(m_begin, m_tiles):
            c_view = self.c_l0.next_write()
            for k_idx in dsl.range(0, k_total // self.panel_k):
                mem_copy(
                    self.a_l1.next_write(),
                    tile_slice(a_gm, (bm, self.panel_k), (m_tile, k_idx)),
                    options=copy_options(l2_cache_ctl=self.l2_hint),
                )
                self.copy_weight(
                    self.b_l1.next_write(),
                    weight,
                    (width, self.panel_k),
                    (n_idx, k_idx),
                    options=copy_options(l2_cache_ctl=self.weight_l2_hint),
                )
                a_panel = self.a_l1.next_read()
                b_panel = self.b_l1.next_read()
                for inner in range_constexpr(self.panel_k // CUBE_BK):
                    mem_copy(
                        self.a_l0.next_write(),
                        tile_slice(a_panel, (bm, CUBE_BK), (0, inner)),
                    )
                    mem_copy(
                        self.b_l0.next_write(),
                        tile_slice(b_panel, (width, CUBE_BK), (0, inner)),
                    )
                    matmul(
                        c_view,
                        self.a_l0.next_read(),
                        self.b_l0.next_read(),
                        init=k_idx == 0 and inner == 0,
                    )
            mem_copy(
                tile_slice(out_gm, (bm, width), (m_tile, n_idx)), self.c_l0.next_read()
            )

    @jit
    def project_owner(
        self, a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
    ):
        full = n_total // self.width
        for n_idx in dsl.range(block, full, blocks):
            self.project_tile(a_gm, weight, out_gm, k_total, n_idx, m_tiles)

    @jit
    def actual_row_view(self, slot, valid_rows, width, physical_rows):
        storage = from_tensor(
            slot, layout=make_layout((physical_rows, width)), data_format="nz"
        )
        return from_tensor(storage, layout=make_layout((valid_rows, width)))

    @jit
    def project_actual_rows(self, a_gm, weight, out_gm, k_total, n_idx, m_tile):
        valid_rows = tile_slice(a_gm, (self.bm, self.panel_k), (m_tile, 0)).shape[0]
        if self.physical_rows <= CUBE_M_ALIGNMENT:
            self.project_row_tile(
                a_gm,
                weight,
                out_gm,
                k_total,
                n_idx,
                m_tile,
                valid_rows,
                CUBE_M_ALIGNMENT,
            )
        else:
            if valid_rows <= CUBE_M_ALIGNMENT:
                self.project_row_tile(
                    a_gm,
                    weight,
                    out_gm,
                    k_total,
                    n_idx,
                    m_tile,
                    valid_rows,
                    CUBE_M_ALIGNMENT,
                )
            else:
                if self.physical_rows <= 2 * CUBE_M_ALIGNMENT:
                    self.project_row_tile(
                        a_gm,
                        weight,
                        out_gm,
                        k_total,
                        n_idx,
                        m_tile,
                        valid_rows,
                        2 * CUBE_M_ALIGNMENT,
                    )
                else:
                    if valid_rows <= 2 * CUBE_M_ALIGNMENT:
                        self.project_row_tile(
                            a_gm,
                            weight,
                            out_gm,
                            k_total,
                            n_idx,
                            m_tile,
                            valid_rows,
                            2 * CUBE_M_ALIGNMENT,
                        )
                    else:
                        if valid_rows <= 3 * CUBE_M_ALIGNMENT:
                            self.project_row_tile(
                                a_gm,
                                weight,
                                out_gm,
                                k_total,
                                n_idx,
                                m_tile,
                                valid_rows,
                                3 * CUBE_M_ALIGNMENT,
                            )
                        else:
                            self.project_row_tile(
                                a_gm,
                                weight,
                                out_gm,
                                k_total,
                                n_idx,
                                m_tile,
                                valid_rows,
                                4 * CUBE_M_ALIGNMENT,
                            )

    @jit
    def project_row_tile(
        self,
        a_gm,
        weight,
        out_gm,
        k_total,
        n_idx,
        m_tile,
        valid_rows,
        physical_rows,
    ):
        bm = self.bm
        width = self.width
        c_view = self.actual_row_view(
            self.c_l0.next_write(), valid_rows, width, physical_rows
        )
        for k_idx in dsl.range(0, k_total // self.panel_k):
            mem_copy(
                self.actual_row_view(
                    self.a_l1.next_write(), valid_rows, self.panel_k, physical_rows
                ),
                tile_slice(a_gm, (bm, self.panel_k), (m_tile, k_idx)),
                options=copy_options(l2_cache_ctl=self.l2_hint),
            )
            self.copy_weight(
                self.b_l1.next_write(),
                weight,
                (width, self.panel_k),
                (n_idx, k_idx),
                options=copy_options(l2_cache_ctl=self.weight_l2_hint),
            )
            a_panel = self.actual_row_view(
                self.a_l1.next_read(), valid_rows, self.panel_k, physical_rows
            )
            b_panel = self.b_l1.next_read()
            for inner in range_constexpr(self.panel_k // CUBE_BK):
                mem_copy(
                    self.actual_row_view(
                        self.a_l0.next_write(), valid_rows, CUBE_BK, physical_rows
                    ),
                    tile_slice(a_panel, (physical_rows, CUBE_BK), (0, inner)),
                )
                mem_copy(
                    self.b_l0.next_write(),
                    tile_slice(b_panel, (width, CUBE_BK), (0, inner)),
                )
                matmul(
                    c_view,
                    self.actual_row_view(
                        self.a_l0.next_read(), valid_rows, CUBE_BK, physical_rows
                    ),
                    self.b_l0.next_read(),
                    init=k_idx == 0 and inner == 0,
                )
        mem_copy(
            out_gm[
                m_tile * bm : m_tile * bm + valid_rows,
                n_idx * width : (n_idx + 1) * width,
            ],
            self.actual_row_view(
                self.c_l0.next_read(), valid_rows, width, physical_rows
            ),
        )

    @jit
    def project_grid(
        self, a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
    ):
        n_tiles = n_total // self.width
        for task in dsl.range(block, m_tiles * n_tiles, blocks):
            m_idx = task // n_tiles
            n_idx = task % n_tiles
            if self.use_actual_rows:
                self.project_actual_rows(a_gm, weight, out_gm, k_total, n_idx, m_idx)
            else:
                self.project_tile(
                    a_gm, weight, out_gm, k_total, n_idx, m_idx + 1, m_idx
                )

    @jit
    def project_partial_front(
        self, a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
    ):
        n_tiles = n_total // self.width
        panels = k_total // self.panel_k
        for task in dsl.range(block, m_tiles * n_tiles, blocks):
            m_idx = task // n_tiles
            n_idx = task % n_tiles
            for part in range_constexpr(QA_SPLIT_PARTS):
                first = panels * part // QA_SPLIT_PARTS
                last = panels * (part + 1) // QA_SPLIT_PARTS
                c_view = self.c_l0.next_write()
                for k_idx in dsl.range(first, last):
                    mem_copy(
                        self.a_l1.next_write(),
                        tile_slice(a_gm, (self.bm, self.panel_k), (m_idx, k_idx)),
                        options=copy_options(l2_cache_ctl=self.l2_hint),
                    )
                    self.copy_weight(
                        self.b_l1.next_write(),
                        weight,
                        (self.width, self.panel_k),
                        (n_idx, k_idx),
                    )
                    a_panel = self.a_l1.next_read()
                    b_panel = self.b_l1.next_read()
                    for inner in range_constexpr(self.panel_k // CUBE_BK):
                        mem_copy(
                            self.a_l0.next_write(),
                            tile_slice(a_panel, (self.bm, CUBE_BK), (0, inner)),
                        )
                        mem_copy(
                            self.b_l0.next_write(),
                            tile_slice(b_panel, (self.width, CUBE_BK), (0, inner)),
                        )
                        matmul(
                            c_view,
                            self.a_l0.next_read(),
                            self.b_l0.next_read(),
                            init=k_idx == first and inner == 0,
                        )
                mem_copy(
                    tile_slice(
                        out_gm, (self.bm, self.width), (m_idx + part * m_tiles, n_idx)
                    ),
                    self.c_l0.next_read(),
                )

    @jit
    def project_resident_row(
        self, a_gm, weight, out_gm, m_idx, n_start, n_step, n_tiles
    ):
        mem_copy(
            self.a_full,
            tile_slice(a_gm, (self.bm, 1024), (m_idx, 0)),
            options=copy_options(l2_cache_ctl=self.l2_hint),
        )
        for n_idx in dsl.range(n_start, n_tiles, n_step):
            c_view = self.c_l0.next_write()
            for k_idx in dsl.range(0, 1024 // CUBE_BK):
                mem_copy(
                    self.a_l0.next_write(),
                    tile_slice(self.a_full, (self.bm, CUBE_BK), (0, k_idx)),
                )
                self.copy_weight(
                    self.b_l1.next_write(),
                    weight,
                    (self.width, CUBE_BK),
                    (n_idx, k_idx),
                )
                mem_copy(self.b_l0.next_write(), self.b_l1.next_read())
                matmul(
                    c_view,
                    self.a_l0.next_read(),
                    self.b_l0.next_read(),
                    init=k_idx == 0,
                )
            mem_copy(
                tile_slice(out_gm, (self.bm, self.width), (m_idx, n_idx)),
                self.c_l0.next_read(),
            )

    @jit
    def project_front(
        self, a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
    ):
        if self.resident_capable:
            if k_total == 1024:
                if m_tiles >= blocks:
                    for m_idx in dsl.range(block, m_tiles, blocks):
                        self.project_resident_row(
                            a_gm, weight, out_gm, m_idx, 0, 1, n_total // self.width
                        )
                else:
                    group = block % m_tiles
                    lane = block // m_tiles
                    lanes = (blocks + m_tiles - 1 - group) // m_tiles
                    if lane < n_total // self.width:
                        self.project_resident_row(
                            a_gm,
                            weight,
                            out_gm,
                            group,
                            lane,
                            lanes,
                            n_total // self.width,
                        )
            else:
                self.project_grid(
                    a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
                )
        else:
            self.project_grid(
                a_gm, weight, out_gm, k_total, n_total, m_tiles, block, blocks
            )


class WkbProjection:
    def __init__(self, bm, group_columns=False, heads=0, weight_l2_hint=0):
        self.bm = bm
        self.group_columns = group_columns
        self.weight_l2_hint = weight_l2_hint
        self.columns = 4 * CUBE_BK if group_columns else CUBE_BK
        operand_dtype = dtypes.bfloat16
        self.a_l1 = RingBuffer(
            MemLoc.L1, operand_dtype, make_layout((bm, BKW)), depth=2, data_format="nz"
        )
        self.b_l1 = RingBuffer(
            MemLoc.L1,
            operand_dtype,
            make_layout((BKW, self.columns)),
            depth=2,
            data_format="nz",
        )
        self.a_l0 = RingBuffer(
            MemLoc.L0A, operand_dtype, make_layout((bm, BKW)), depth=1
        )
        self.b_l0 = RingBuffer(
            MemLoc.L0B, operand_dtype, make_layout((CUBE_BK, BKW)), depth=1
        )
        self.c_l0 = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((bm, self.columns)), depth=2
        )

    @jit
    def project_full_head(self, q_proj, wkb2d, q2d, head, dn, head_dim, kvr, m_tile):
        row = m_tile * self.bm
        c_view = self.c_l0.next_write()
        for k_idx in dsl.range(0, dn // BKW):
            mem_copy(
                self.a_l1.next_write(),
                q_proj[
                    row : row + self.bm,
                    head * head_dim + k_idx * BKW : head * head_dim + (k_idx + 1) * BKW,
                ],
                options=copy_options(l2_cache_ctl=1),
            )
            mem_copy(
                self.b_l1.next_write(),
                tile_slice(
                    wkb2d, (BKW, self.columns), ((head * dn + k_idx * BKW) // BKW, 0)
                ),
                options=copy_options(l2_cache_ctl=self.weight_l2_hint),
            )
            mem_copy(self.a_l0.next_write(), self.a_l1.next_read())
            a_view = self.a_l0.next_read()
            b_panel = self.b_l1.next_read()
            for n_chunk in range_constexpr(4):
                mem_copy(
                    self.b_l0.next_write(),
                    tile_slice(b_panel, (BKW, CUBE_BK), (0, n_chunk)).permute(1, 0),
                )
                matmul(
                    tile_slice(c_view, (self.bm, CUBE_BK), (0, n_chunk)),
                    a_view,
                    self.b_l0.next_read(),
                    init=k_idx == 0,
                )
        mem_copy(
            q2d[row : row + self.bm, head * kvr : head * kvr + self.columns],
            self.c_l0.next_read(),
        )

    @jit
    def load_weight(self, wkb2d, head, dn, k_idx, n_chunk):
        mem_copy(
            self.b_l1.next_write(),
            tile_slice(
                wkb2d, (BKW, CUBE_BK), ((head * dn + k_idx * BKW) // BKW, n_chunk)
            ),
            options=copy_options(l2_cache_ctl=self.weight_l2_hint),
        )

    @jit
    def project_m_tile(
        self, q_proj, wkb2d, q2d, head, dn, head_dim, kvr, m_tile, n_chunk, tile_rows
    ):
        row = m_tile * self.bm
        col_base = head * head_dim
        out_base = head * kvr
        c_view = from_tensor(
            self.c_l0.next_write(),
            layout=make_layout((tile_rows, CUBE_BK)),
            data_format="nz",
        )
        for k_idx in dsl.range(0, dn // BKW):
            mem_copy(
                from_tensor(
                    self.a_l1.next_write(),
                    layout=make_layout((tile_rows, BKW)),
                    data_format="nz",
                ),
                q_proj[
                    row : row + tile_rows,
                    col_base + k_idx * BKW : col_base + (k_idx + 1) * BKW,
                ],
                options=copy_options(l2_cache_ctl=1),
            )
            mem_copy(
                from_tensor(
                    self.a_l0.next_write(),
                    layout=make_layout((tile_rows, BKW)),
                    data_format="nz",
                ),
                from_tensor(
                    self.a_l1.next_read(),
                    layout=make_layout((tile_rows, BKW)),
                    data_format="nz",
                ),
            )
            self.load_weight(wkb2d, head, dn, k_idx, n_chunk)
            mem_copy(self.b_l0.next_write(), self.b_l1.next_read().permute(1, 0))
            matmul(
                c_view,
                from_tensor(
                    self.a_l0.next_read(),
                    layout=make_layout((tile_rows, BKW)),
                    data_format="nz",
                ),
                self.b_l0.next_read(),
                init=k_idx == 0,
            )
        mem_copy(
            q2d[
                row : row + tile_rows,
                out_base + n_chunk * CUBE_BK : out_base + (n_chunk + 1) * CUBE_BK,
            ],
            from_tensor(
                self.c_l0.next_read(),
                layout=make_layout((tile_rows, CUBE_BK)),
                data_format="nz",
            ),
        )

    @jit
    def project_head(
        self,
        q_proj,
        wkb2d,
        q2d,
        head,
        dn,
        head_dim,
        kvr,
        m_tiles,
        tokens,
        m_begin=0,
        n_begin=0,
        n_end=4,
    ):
        bm = self.bm
        for m_tile in dsl.range(m_begin, m_tiles):
            for n_chunk in dsl.range(n_begin, n_end):
                if bm > M_TAIL_ROWS:
                    if m_tile * bm + M_TAIL_ROWS >= tokens:
                        self.project_m_tile(
                            q_proj,
                            wkb2d,
                            q2d,
                            head,
                            dn,
                            head_dim,
                            kvr,
                            m_tile,
                            n_chunk,
                            M_TAIL_ROWS,
                        )
                    else:
                        self.project_m_tile(
                            q_proj,
                            wkb2d,
                            q2d,
                            head,
                            dn,
                            head_dim,
                            kvr,
                            m_tile,
                            n_chunk,
                            bm,
                        )
                else:
                    self.project_m_tile(
                        q_proj, wkb2d, q2d, head, dn, head_dim, kvr, m_tile, n_chunk, bm
                    )

    @jit
    def project_grid(
        self,
        q_proj,
        wkb2d,
        q2d,
        heads,
        dn,
        head_dim,
        kvr,
        m_tiles,
        block,
        blocks,
        tokens,
    ):
        if self.group_columns:
            for task in dsl.range(block, m_tiles * heads, blocks):
                self.project_full_head(
                    q_proj,
                    wkb2d,
                    q2d,
                    task % heads,
                    dn,
                    head_dim,
                    kvr,
                    task // heads,
                )
        else:
            for task in dsl.range(block, m_tiles * heads * 4, blocks):
                n_chunk = task % 4
                head = task // 4 % heads
                m_tile = task // (heads * 4)
                self.project_head(
                    q_proj,
                    wkb2d,
                    q2d,
                    head,
                    dn,
                    head_dim,
                    kvr,
                    m_tile + 1,
                    tokens,
                    m_tile,
                    n_chunk,
                    n_chunk + 1,
                )


@jit
def wkb_projection_path(
    bm,
    group_columns,
    load_hint,
    q_proj,
    wkb2d,
    q2d,
    heads,
    dn,
    head_dim,
    kvr,
    m_tiles,
    block,
    blocks,
    tokens,
):
    # The preceding AIC phase is complete. Runtime branches are exclusive;
    # the planner preserves any live AIV resources across this boundary.
    onchip_rewind(reset_sync_id=True)
    projection = WkbProjection(
        bm, group_columns=group_columns, weight_l2_hint=load_hint
    )
    projection.project_grid(
        q_proj, wkb2d, q2d, heads, dn, head_dim, kvr, m_tiles, block, blocks, tokens
    )
    cube_sync_all()
    onchip_rewind(reset_sync_id=True)


@jit
def project_wkb(
    bm,
    allow_grouped,
    load_hint,
    q_proj,
    wkb2d,
    q2d,
    heads,
    dn,
    head_dim,
    kvr,
    m_tiles,
    block,
    blocks,
    tokens,
):
    if const_expr(8 < bm <= 64 and allow_grouped):
        if heads >= 64:
            wkb_projection_path(
                bm,
                True,
                load_hint,
                q_proj,
                wkb2d,
                q2d,
                heads,
                dn,
                head_dim,
                kvr,
                m_tiles,
                block,
                blocks,
                tokens,
            )
        else:
            wkb_projection_path(
                bm,
                False,
                0,
                q_proj,
                wkb2d,
                q2d,
                heads,
                dn,
                head_dim,
                kvr,
                m_tiles,
                block,
                blocks,
                tokens,
            )
    elif const_expr(bm <= 8 and allow_grouped):
        if m_tiles * heads >= 2 * blocks:
            wkb_projection_path(
                bm,
                True,
                load_hint,
                q_proj,
                wkb2d,
                q2d,
                heads,
                dn,
                head_dim,
                kvr,
                m_tiles,
                block,
                blocks,
                tokens,
            )
        else:
            wkb_projection_path(
                bm,
                False,
                load_hint,
                q_proj,
                wkb2d,
                q2d,
                heads,
                dn,
                head_dim,
                kvr,
                m_tiles,
                block,
                blocks,
                tokens,
            )
    else:
        wkb_projection_path(
            bm,
            False,
            load_hint,
            q_proj,
            wkb2d,
            q2d,
            heads,
            dn,
            head_dim,
            kvr,
            m_tiles,
            block,
            blocks,
            tokens,
        )


@jit
def cast_hif8(value):
    mask = full_mask()
    zero = vdups(0.0, dtypes.float32, mask=mask)
    value = vselect(value, zero, cond_mask=veq(value, value, mask=mask))
    value = vmins(vmaxs(value, -HIF8_MAX, mask=mask), HIF8_MAX, mask=mask)
    return vcast(
        value,
        dtypes.hifloat8,
        mask=mask,
        rounding=RoundingMode.RNA,
        reg_layout=RegLayout.ZERO,
    )


@jit
def write_row_panz_fp8(cache, cache_index, payload_read, token):
    slot = cache_index[token]
    if slot >= 0:
        planes = cache.shape[3] // PA_NZ_FP8_PLANE_ELEMS
        block = slot // cache.shape[1]
        physical = cache[block, None, 0, None].view(
            (planes * cache.shape[1], PA_NZ_FP8_PLANE_ELEMS)
        )
        row = slot % cache.shape[1]
        for plane in dsl.range(0, planes):
            mem_copy(
                tile_slice(
                    physical,
                    (1, PA_NZ_FP8_PLANE_ELEMS),
                    (plane * cache.shape[1] + row, 0),
                ),
                tile_slice(payload_read, (1, PA_NZ_FP8_PLANE_ELEMS), (0, plane)),
            )


@jit
def write_cache_row_fp8(cache, cache_index, payload_read, token, kvr, cache_mode):
    row_tiler = make_tiler((1, kvr), alignment=(1, 16))
    if const_expr(isinstance(kvr, int)):
        byte_view = tile_slice(payload_read, (1, kvr), (0, 0))
    else:
        byte_view = from_tensor(
            payload_read,
            dtype=dtypes.float8_e4m3fn,
            layout=make_layout((1, kvr)),
            offset=0,
        )
    if cache_mode == 0:
        slot = cache_index[token]
        if slot >= 0:
            cache_2d = cache_row_view(cache, slot // cache.shape[1])
            mem_copy(
                tile_slice(cache_2d, row_tiler, (slot % cache.shape[1], 0)), byte_view
            )
    else:
        write_row_panz_fp8(cache, cache_index, payload_read, token)


@jit
def write_row_panz_hif8(cache, cache_index, payload_read, token):
    slot = cache_index[token]
    if slot >= 0:
        planes = cache.shape[3] // PA_NZ_FP8_PLANE_ELEMS
        block = slot // cache.shape[1]
        physical = cache[block, None, 0, None].view(
            (planes * cache.shape[1], PA_NZ_FP8_PLANE_ELEMS)
        )
        row = slot % cache.shape[1]
        for plane in dsl.range(0, planes):
            mem_copy(
                tile_slice(
                    physical,
                    (1, PA_NZ_FP8_PLANE_ELEMS),
                    (plane * cache.shape[1] + row, 0),
                ),
                tile_slice(payload_read, (1, PA_NZ_FP8_PLANE_ELEMS), (0, plane)),
            )


@jit
def write_cache_row_hif8(cache, cache_index, payload_read, token, kvr, cache_mode):
    row_tiler = make_tiler((1, kvr), alignment=(1, 16))
    if const_expr(isinstance(kvr, int)):
        byte_view = tile_slice(payload_read, (1, kvr), (0, 0))
    else:
        byte_view = from_tensor(
            payload_read, dtype=dtypes.uint8, layout=make_layout((1, kvr)), offset=0
        )
    if cache_mode == 0:
        slot = cache_index[token]
        if slot >= 0:
            cache_2d = cache_row_view(cache, slot // cache.shape[1])
            mem_copy(
                tile_slice(cache_2d, row_tiler, (slot % cache.shape[1], 0)), byte_view
            )
    else:
        write_row_panz_hif8(cache, cache_index, payload_read, token)


class Bf16RowReader:
    def __init__(self, capacity):
        self.capacity = capacity
        self.input = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, capacity)), depth=2
        )
        self.output = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, capacity)), depth=2
        )

    @jit
    def read(self, source, columns, token):
        tiler = make_tiler((1, columns), alignment=(1, VECTOR_SEG))
        mem_copy(self.input.next_write(), tile_slice(source, tiler, (token, 0)))
        src = self.input.next_read()
        dst = self.output.next_write()
        with vf(mode="simd"):
            for offset in dsl.range(0, columns, VECTOR_SEG):
                value = vload_unpack(src, offset, unpack_mode=UnpackMode.B16_TO_B32)
                value = vcast(value, dtypes.float32, mask=full_mask())
                vstore(dst, offset, value, full_mask())
            vmem_bar(mode="vst_vld")
        return self.output.next_read()


class PartialQaRowReader:
    def __init__(self):
        self.input = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((QA_SPLIT_PARTS, VECTOR_Q_LORA_MAX)),
            depth=2,
        )
        self.output = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_Q_LORA_MAX)), depth=2
        )

    @jit
    def read(self, source, columns, token):
        tiler = make_tiler((1, columns), alignment=(1, VECTOR_SEG))
        partials = self.input.next_write()
        for part in range_constexpr(QA_SPLIT_PARTS):
            mem_copy(
                tile_slice(partials, (1, VECTOR_Q_LORA_MAX), (part, 0)),
                tile_slice(
                    source,
                    tiler,
                    (token + part * (source.shape[0] // QA_SPLIT_PARTS), 0),
                ),
            )
        src = self.input.next_read()
        dst = self.output.next_write()
        with vf(mode="simd"):
            for offset in dsl.range(0, columns, VECTOR_SEG):
                first = vadd(
                    vload(src, offset),
                    vload(src, VECTOR_Q_LORA_MAX + offset),
                    mask=full_mask(),
                )
                second = vadd(
                    vload(src, 2 * VECTOR_Q_LORA_MAX + offset),
                    vload(src, 3 * VECTOR_Q_LORA_MAX + offset),
                    mask=full_mask(),
                )
                value = vadd(first, second, mask=full_mask())
                value = vcast(
                    value, dtypes.bfloat16, mask=full_mask(), rounding=RoundingMode.RN
                )
                value = vcast(value, dtypes.float32, mask=full_mask())
                vstore(dst, offset, value, full_mask())
            vmem_bar(mode="vst_vld")
        return self.output.next_read()


class QaRmsNorm:
    def __init__(self, partial_front=False):
        self.row = (
            PartialQaRowReader() if partial_front else Bf16RowReader(VECTOR_Q_LORA_MAX)
        )
        self.gamma = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_Q_LORA_MAX)), depth=1
        )
        self.out = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_Q_LORA_MAX)), depth=2
        )
        self.out_bf16 = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, VECTOR_Q_LORA_MAX)), depth=2
        )
        self.scalar = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 8)))

    @jit
    def zero_tail(self, out_gm, q_lora, start, stride):
        row_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        for row in dsl.range(start, out_gm.shape[0], stride):
            out_write = self.out_bf16.next_write()
            with vf(mode="simd"):
                zero_mask = update_mask(2 * VECTOR_SEG, elem_bits=16)[0]
                zero = vdups(0.0, dtypes.bfloat16, mask=zero_mask)
                for column in range_constexpr(VECTOR_Q_LORA_MAX // (2 * VECTOR_SEG)):
                    vstore(out_write, column * (2 * VECTOR_SEG), zero, zero_mask)
                vmem_bar(mode="vst_vld")
            mem_copy(
                tile_slice(out_gm, row_tiler, (row, 0)),
                from_tensor(self.out_bf16.next_read(), layout=make_layout((1, q_lora))),
            )

    @jit
    def process_rows(
        self, ws_raw, gamma_gm, q_lora, norm_eps, qr_scale, start, stride, rows
    ):
        row_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        gamma_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        inv_q_lora = 1.0 / dtypes.float32(q_lora)
        mem_copy(self.gamma.next_write(), tile_slice(gamma_gm, gamma_tiler, (0, 0)))
        gamma_read = self.gamma.next_read()
        for token in dsl.range(start, rows, stride):
            row_read = self.row.read(ws_raw, q_lora, token)
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                total = vdups(0.0, dtypes.float32, mask=one)
                for segment in range_constexpr(VECTOR_Q_LORA_MAX // VECTOR_SEG):
                    lanes = q_lora - segment * VECTOR_SEG
                    lanes = min(max(lanes, 0), VECTOR_SEG)
                    seg_mask = update_mask(lanes, elem_bits=32)[0]
                    value = vload(row_read, segment * VECTOR_SEG)
                    total = vadd(
                        total,
                        vreduce_sum(vmul(value, value, mask=seg_mask), mask=seg_mask),
                        mask=one,
                    )
                mean = vmuls(total, inv_q_lora, mask=one)
                root = vsqrt(vadds(mean, norm_eps, mask=one), mask=one)
                vstore_first(self.scalar, 0, root)
                vmem_bar(mode="vst_vld")
            out_write = self.out.next_write()
            with vf(mode="simd"):
                divisor = vload_broadcast(self.scalar, 0)
                for segment in dsl.range(0, q_lora, VECTOR_SEG):
                    value = vload(row_read, segment)
                    normalized = vdiv(
                        value, divisor, mask=full_mask(), precision_mode=True
                    )
                    normalized = vmul(
                        normalized, vload(gamma_read, segment), mask=full_mask()
                    )
                    normalized = vmuls(normalized, qr_scale, mask=full_mask())
                    vstore(out_write, segment, normalized, full_mask())
                vmem_bar(mode="vst_vld")
            mem_copy(
                tile_slice(ws_raw, row_tiler, (token, 0)),
                from_tensor(self.out.next_read(), layout=make_layout((1, q_lora))),
            )

    @jit
    def process_rows_bf16(
        self, ws_raw, gamma_gm, out_gm, q_lora, norm_eps, qr_scale, start, stride, rows
    ):
        row_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        gamma_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        inv_q_lora = 1.0 / dtypes.float32(q_lora)
        mem_copy(self.gamma.next_write(), tile_slice(gamma_gm, gamma_tiler, (0, 0)))
        gamma_read = self.gamma.next_read()
        for token in dsl.range(start, rows, stride):
            row_read = self.row.read(ws_raw, q_lora, token)
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                total = vdups(0.0, dtypes.float32, mask=one)
                for segment in range_constexpr(VECTOR_Q_LORA_MAX // VECTOR_SEG):
                    lanes = q_lora - segment * VECTOR_SEG
                    lanes = min(max(lanes, 0), VECTOR_SEG)
                    seg_mask = update_mask(lanes, elem_bits=32)[0]
                    value = vload(row_read, segment * VECTOR_SEG)
                    total = vadd(
                        total,
                        vreduce_sum(vmul(value, value, mask=seg_mask), mask=seg_mask),
                        mask=one,
                    )
                mean = vmuls(total, inv_q_lora, mask=one)
                root = vsqrt(vadds(mean, norm_eps, mask=one), mask=one)
                vstore_first(self.scalar, 0, root)
                vmem_bar(mode="vst_vld")
            out_write = self.out_bf16.next_write()
            with vf(mode="simd"):
                divisor = vload_broadcast(self.scalar, 0)
                for segment in dsl.range(0, q_lora, VECTOR_SEG):
                    value = vload(row_read, segment)
                    normalized = vdiv(
                        value, divisor, mask=full_mask(), precision_mode=True
                    )
                    normalized = vmul(
                        normalized, vload(gamma_read, segment), mask=full_mask()
                    )
                    normalized = vmuls(normalized, qr_scale, mask=full_mask())
                    vstore_pack(
                        out_write,
                        segment,
                        vcast(
                            normalized,
                            dtypes.bfloat16,
                            mask=full_mask(),
                            rounding=RoundingMode.RN,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B16,
                    )
                vmem_bar(mode="vst_vld")
            mem_copy(
                tile_slice(out_gm, row_tiler, (token, 0)),
                from_tensor(self.out_bf16.next_read(), layout=make_layout((1, q_lora))),
            )

    @jit
    def process_rows_qr(
        self, ws_raw, gamma_gm, qr_gm, q_lora, norm_eps, qr_scale, start, stride, rows
    ):
        row_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        gamma_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        inv_q_lora = 1.0 / dtypes.float32(q_lora)
        mem_copy(self.gamma.next_write(), tile_slice(gamma_gm, gamma_tiler, (0, 0)))
        gamma_read = self.gamma.next_read()
        for token in dsl.range(start, rows, stride):
            row_read = self.row.read(ws_raw, q_lora, token)
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                total = vdups(0.0, dtypes.float32, mask=one)
                for segment in range_constexpr(VECTOR_Q_LORA_MAX // VECTOR_SEG):
                    lanes = q_lora - segment * VECTOR_SEG
                    lanes = min(max(lanes, 0), VECTOR_SEG)
                    seg_mask = update_mask(lanes, elem_bits=32)[0]
                    value = vload(row_read, segment * VECTOR_SEG)
                    total = vadd(
                        total,
                        vreduce_sum(vmul(value, value, mask=seg_mask), mask=seg_mask),
                        mask=one,
                    )
                mean = vmuls(total, inv_q_lora, mask=one)
                root = vsqrt(vadds(mean, norm_eps, mask=one), mask=one)
                vstore_first(self.scalar, 0, root)
                vmem_bar(mode="vst_vld")
            out_write = self.out.next_write()
            bf16_write = self.out_bf16.next_write()
            with vf(mode="simd"):
                divisor = vload_broadcast(self.scalar, 0)
                for segment in dsl.range(0, q_lora, VECTOR_SEG):
                    value = vload(row_read, segment)
                    normalized = vdiv(
                        value, divisor, mask=full_mask(), precision_mode=True
                    )
                    normalized = vmul(
                        normalized, vload(gamma_read, segment), mask=full_mask()
                    )
                    normalized = vmuls(normalized, qr_scale, mask=full_mask())
                    vstore(out_write, segment, normalized, full_mask())
                    vstore_pack(
                        bf16_write,
                        segment,
                        vcast(
                            normalized,
                            dtypes.bfloat16,
                            mask=full_mask(),
                            rounding=RoundingMode.RN,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B16,
                    )
                vmem_bar(mode="vst_vld")
            mem_copy(
                tile_slice(ws_raw, row_tiler, (token, 0)),
                from_tensor(self.out.next_read(), layout=make_layout((1, q_lora))),
            )
            mem_copy(
                tile_slice(qr_gm, row_tiler, (token, 0)),
                from_tensor(self.out_bf16.next_read(), layout=make_layout((1, q_lora))),
            )

    @jit
    def process_rows_bf16_qr(
        self,
        ws_raw,
        gamma_gm,
        out_gm,
        qr_gm,
        q_lora,
        norm_eps,
        qr_scale,
        start,
        stride,
        rows,
    ):
        row_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        gamma_tiler = make_tiler((1, q_lora), alignment=(1, VECTOR_SEG))
        inv_q_lora = 1.0 / dtypes.float32(q_lora)
        mem_copy(self.gamma.next_write(), tile_slice(gamma_gm, gamma_tiler, (0, 0)))
        gamma_read = self.gamma.next_read()
        for token in dsl.range(start, rows, stride):
            row_read = self.row.read(ws_raw, q_lora, token)
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                total = vdups(0.0, dtypes.float32, mask=one)
                for segment in range_constexpr(VECTOR_Q_LORA_MAX // VECTOR_SEG):
                    lanes = q_lora - segment * VECTOR_SEG
                    lanes = min(max(lanes, 0), VECTOR_SEG)
                    seg_mask = update_mask(lanes, elem_bits=32)[0]
                    value = vload(row_read, segment * VECTOR_SEG)
                    total = vadd(
                        total,
                        vreduce_sum(vmul(value, value, mask=seg_mask), mask=seg_mask),
                        mask=one,
                    )
                mean = vmuls(total, inv_q_lora, mask=one)
                root = vsqrt(vadds(mean, norm_eps, mask=one), mask=one)
                vstore_first(self.scalar, 0, root)
                vmem_bar(mode="vst_vld")
            out_write = self.out_bf16.next_write()
            with vf(mode="simd"):
                divisor = vload_broadcast(self.scalar, 0)
                for segment in dsl.range(0, q_lora, VECTOR_SEG):
                    value = vload(row_read, segment)
                    normalized = vdiv(
                        value, divisor, mask=full_mask(), precision_mode=True
                    )
                    normalized = vmul(
                        normalized, vload(gamma_read, segment), mask=full_mask()
                    )
                    normalized = vmuls(normalized, qr_scale, mask=full_mask())
                    vstore_pack(
                        out_write,
                        segment,
                        vcast(
                            normalized,
                            dtypes.bfloat16,
                            mask=full_mask(),
                            rounding=RoundingMode.RN,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B16,
                    )
                vmem_bar(mode="vst_vld")
            bf16_read = self.out_bf16.next_read()
            mem_copy(
                tile_slice(out_gm, row_tiler, (token, 0)),
                from_tensor(bf16_read, layout=make_layout((1, q_lora))),
            )
            mem_copy(
                tile_slice(qr_gm, row_tiler, (token, 0)),
                from_tensor(bf16_read, layout=make_layout((1, q_lora))),
            )


class Rope:
    def __init__(self):
        self.tbl = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, VECTOR_SEG)), depth=2
        )
        self.wide = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, PAD_SEG)), depth=2
        )
        self.zpad = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, PAD_SEG)))
        self.out = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_SEG)))

    @jit
    def load_table(self, table_gm, token):
        mem_copy(
            self.tbl.next_write(), tile_slice(table_gm, (1, VECTOR_SEG), (token, 0))
        )
        tbl_read = self.tbl.next_read()
        wide_write = self.wide.next_write()
        with vf(mode="simd"):
            value = vload_unpack(tbl_read, 0, unpack_mode=UnpackMode.B16_TO_B32)
            vstore(
                wide_write,
                0,
                vcast(value, dtypes.float32, mask=full_mask()),
                full_mask(),
            )
            vmem_bar(mode="vst_vld")
        return self.wide.next_read()

    @jit
    def rotate(self, z_read, z_offset, sin_read, cos_read, rope_mode):
        with vf(mode="simd"):
            vstore(self.zpad, 0, vload(z_read, z_offset), full_mask())
            vmem_bar(mode="vst_vld")
        if rope_mode == 1:
            with vf(mode="simd"):
                half = update_mask(HALF, elem_bits=32)[0]
                cos_low = vload(cos_read, 0)
                cos_high = vload(cos_read, HALF)
                sin_low = vload(sin_read, 0)
                sin_high = vload(sin_read, HALF)
                (first, second) = vload_deinterleave(self.zpad, 0)
                lower = vsub(
                    vmul(first, cos_low, mask=half),
                    vmul(second, sin_low, mask=half),
                    mask=half,
                )
                upper = vadd(
                    vmul(second, cos_high, mask=half),
                    vmul(first, sin_high, mask=half),
                    mask=half,
                )
                vstore(self.out, 0, lower, half)
                vstore(self.out, HALF, upper, half)
                vmem_bar(mode="vst_vld")
        else:
            with vf(mode="simd"):
                half = update_mask(HALF, elem_bits=32)[0]
                cos_low = vload(cos_read, 0)
                cos_high = vload(cos_read, HALF)
                sin_low = vload(sin_read, 0)
                sin_high = vload(sin_read, HALF)
                first = vload(self.zpad, 0)
                second = vload(self.zpad, HALF)
                lower = vsub(
                    vmul(first, cos_low, mask=half),
                    vmul(second, sin_low, mask=half),
                    mask=half,
                )
                upper = vadd(
                    vmul(second, cos_high, mask=half),
                    vmul(first, sin_high, mask=half),
                    mask=half,
                )
                vstore(self.out, 0, lower, half)
                vstore(self.out, HALF, upper, half)
                vmem_bar(mode="vst_vld")


class KvChain:
    def __init__(self, page_size, rope_active):
        self.page_size = page_size
        self.row = Bf16RowReader(VECTOR_KVR_MAX)
        self.gamma = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_KV_LORA)), depth=1
        )
        self.payload = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.payload_q = RingBuffer(
            MemLoc.UB, dtypes.float8_e4m3fn, make_layout((1, DTILE_MAX)), depth=2
        )
        self.payload_m2 = RingBuffer(
            MemLoc.UB, dtypes.float8_e4m3fn, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.payload_m3 = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.scalar = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 64)))
        self.scale_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 8)))
        if rope_active:
            self.rope = Rope()

    @jit
    def latent_root(self, row_read, norm_eps):
        with vf(mode="simd"):
            one = update_mask(1, elem_bits=32)[0]
            total = vdups(0.0, dtypes.float32, mask=one)
            for segment in range_constexpr(VECTOR_KV_LORA // VECTOR_SEG):
                value = vload(row_read, segment * VECTOR_SEG)
                total = vadd(
                    total,
                    vreduce_sum(vmul(value, value, mask=full_mask()), mask=full_mask()),
                    mask=one,
                )
            mean = vmuls(total, 1.0 / VECTOR_KV_LORA, mask=one)
            root = vsqrt(vadds(mean, norm_eps, mask=one), mask=one)
            vstore_first(self.scalar, 0, root)
            vmem_bar(mode="vst_vld")

    @jit
    def pack_payload(
        self,
        payload_write,
        row_read,
        gamma_read,
        kr_read,
        kr_offset,
        kvr_scale,
        rope_dim,
    ):
        with vf(mode="simd"):
            kr_mask = update_mask(rope_dim, elem_bits=32)[0]
            divisor = vload_broadcast(self.scalar, 0)
            for segment in range_constexpr(VECTOR_KV_LORA // VECTOR_SEG):
                value = vload(row_read, segment * VECTOR_SEG)
                normalized = vdiv(value, divisor, mask=full_mask(), precision_mode=True)
                normalized = vmul(
                    normalized,
                    vload(gamma_read, segment * VECTOR_SEG),
                    mask=full_mask(),
                )
                normalized = vmuls(normalized, kvr_scale, mask=full_mask())
                vstore_pack(
                    payload_write,
                    segment * VECTOR_SEG,
                    vcast(
                        normalized,
                        dtypes.bfloat16,
                        mask=full_mask(),
                        rounding=RoundingMode.RN,
                    ),
                    full_mask(),
                    pack_mode=PackMode.B32_TO_B16,
                )
            kr_value = vload(kr_read, kr_offset)
            vstore_pack(
                payload_write,
                VECTOR_KV_LORA,
                vcast(
                    kr_value,
                    dtypes.bfloat16,
                    mask=full_mask(),
                    rounding=RoundingMode.RN,
                ),
                kr_mask,
                pack_mode=PackMode.B32_TO_B16,
            )
            vmem_bar(mode="vst_vld")

    @jit
    def quant_pack_payload(
        self,
        payload_write,
        row_read,
        gamma_read,
        kr_read,
        kr_offset,
        kvr_scale,
        rope_dim,
    ):
        kr_view = from_tensor(
            payload_write,
            dtype=dtypes.bfloat16,
            layout=make_layout((1, (DTILE_MAX - KV_BYTES) // 2)),
            offset=KV_BYTES,
        )
        tail_low = from_tensor(
            payload_write,
            dtype=dtypes.float32,
            layout=make_layout((1, GROUPS)),
            offset=KV_BYTES,
        )
        tail_high = from_tensor(
            payload_write,
            dtype=dtypes.float32,
            layout=make_layout((1, GROUPS)),
            offset=KV_BYTES + 2 * 64,
        )
        with vf(mode="simd"):
            one = update_mask(1, elem_bits=32)[0]
            limit = vdups(FP8_MAX, dtypes.float32, mask=one)
            divisor = vload_broadcast(self.scalar, 0)
            for g in range_constexpr(GROUPS):
                g_amax = vdups(GROUP_AMAX_FLOOR, dtypes.float32, mask=one)
                for s in range_constexpr(GROUP_SIZE // VECTOR_SEG):
                    value = vload(row_read, g * GROUP_SIZE + s * VECTOR_SEG)
                    normalized = vdiv(
                        value, divisor, mask=full_mask(), precision_mode=True
                    )
                    normalized = vmul(
                        normalized,
                        vload(gamma_read, g * GROUP_SIZE + s * VECTOR_SEG),
                        mask=full_mask(),
                    )
                    normalized = vmuls(normalized, kvr_scale, mask=full_mask())
                    g_amax = vmax(
                        g_amax,
                        vreduce_max(
                            vabs(normalized, mask=full_mask()), mask=full_mask()
                        ),
                        mask=one,
                    )
                g_descale = vdiv(g_amax, limit, mask=one, precision_mode=True)
                vstore_first(self.scalar, g, g_descale)
            vmem_bar(mode="vst_vld")
            descale_vec = vload(self.scalar, 0)
            mask4 = update_mask(GROUPS, elem_bits=32)[0]
            if const_expr(isinstance(rope_dim, int)):
                if rope_dim == 0:
                    vstore(tail_low, 0, descale_vec, mask4)
                else:
                    vstore(tail_high, 0, descale_vec, mask4)
            else:
                mask_dr0 = update_mask((64 - rope_dim) // 64 * GROUPS, elem_bits=32)[0]
                vstore(tail_high, 0, descale_vec, mask4)
                vstore(tail_low, 0, descale_vec, mask_dr0)
            for g in range_constexpr(GROUPS):
                g_descale = vload_broadcast(self.scalar, g)
                for s in range_constexpr(GROUP_SIZE // VECTOR_SEG):
                    value = vload(row_read, g * GROUP_SIZE + s * VECTOR_SEG)
                    normalized = vdiv(
                        value, divisor, mask=full_mask(), precision_mode=True
                    )
                    normalized = vmul(
                        normalized,
                        vload(gamma_read, g * GROUP_SIZE + s * VECTOR_SEG),
                        mask=full_mask(),
                    )
                    normalized = vmuls(normalized, kvr_scale, mask=full_mask())
                    quantized = vdiv(
                        normalized, g_descale, mask=full_mask(), precision_mode=True
                    )
                    vstore_pack(
                        payload_write,
                        g * GROUP_SIZE + s * VECTOR_SEG,
                        vcast(
                            quantized,
                            dtypes.float8_e4m3fn,
                            mask=full_mask(),
                            reg_layout=RegLayout.ZERO,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B8,
                    )
            if rope_dim != 0:
                kr_mask = update_mask(rope_dim, elem_bits=32)[0]
                kr_value = vload(kr_read, kr_offset)
                vstore_pack(
                    kr_view,
                    0,
                    vcast(
                        kr_value,
                        dtypes.bfloat16,
                        mask=full_mask(),
                        rounding=RoundingMode.RN,
                    ),
                    kr_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
            vmem_bar(mode="vst_vld")

    @jit
    def scale_pack_payload(
        self,
        payload_write,
        row_read,
        gamma_read,
        kr_read,
        kr_offset,
        kvr_scale,
        rope_dim,
        scale_read,
    ):
        with vf(mode="simd"):
            divisor = vload_broadcast(self.scalar, 0)
            scale_vec = vload_broadcast(scale_read, 0)
            for segment in range_constexpr(VECTOR_KV_LORA // VECTOR_SEG):
                value = vload(row_read, segment * VECTOR_SEG)
                normalized = vdiv(value, divisor, mask=full_mask(), precision_mode=True)
                normalized = vmul(
                    normalized,
                    vload(gamma_read, segment * VECTOR_SEG),
                    mask=full_mask(),
                )
                normalized = vmuls(normalized, kvr_scale, mask=full_mask())
                scaled = vmul(normalized, scale_vec, mask=full_mask())
                scaled = vmaxs(scaled, -FP8_MAX, mask=full_mask())
                scaled = vmins(scaled, FP8_MAX, mask=full_mask())
                vstore_pack(
                    payload_write,
                    segment * VECTOR_SEG,
                    vcast(
                        scaled,
                        dtypes.float8_e4m3fn,
                        mask=full_mask(),
                        reg_layout=RegLayout.ZERO,
                    ),
                    full_mask(),
                    pack_mode=PackMode.B32_TO_B8,
                )
            kr_mask = update_mask(rope_dim, elem_bits=32)[0]
            kr_value = vload(kr_read, kr_offset)
            kr_scaled = vmul(kr_value, scale_vec, mask=full_mask())
            kr_scaled = vmaxs(kr_scaled, -FP8_MAX, mask=full_mask())
            kr_scaled = vmins(kr_scaled, FP8_MAX, mask=full_mask())
            vstore_pack(
                payload_write,
                VECTOR_KV_LORA,
                vcast(
                    kr_scaled,
                    dtypes.float8_e4m3fn,
                    mask=full_mask(),
                    reg_layout=RegLayout.ZERO,
                ),
                kr_mask,
                pack_mode=PackMode.B32_TO_B8,
            )
            vmem_bar(mode="vst_vld")

    @jit
    def scale_pack_payload_hif8(
        self,
        payload_write,
        row_read,
        gamma_read,
        kr_read,
        kr_offset,
        kvr_scale,
        rope_dim,
        scale_read,
    ):
        with vf(mode="simd"):
            divisor = vload_broadcast(self.scalar, 0)
            scale_vec = vload_broadcast(scale_read, 0)
            for segment in range_constexpr(VECTOR_KV_LORA // VECTOR_SEG):
                value = vload(row_read, segment * VECTOR_SEG)
                normalized = vdiv(value, divisor, mask=full_mask(), precision_mode=True)
                normalized = vmul(
                    normalized,
                    vload(gamma_read, segment * VECTOR_SEG),
                    mask=full_mask(),
                )
                normalized = vmuls(normalized, kvr_scale, mask=full_mask())
                scaled = vmul(normalized, scale_vec, mask=full_mask())
                h8 = cast_hif8(scaled)
                vstore_pack(
                    payload_write,
                    segment * VECTOR_SEG,
                    vreinterpret(h8, dtypes.float8_e4m3fn),
                    full_mask(),
                    pack_mode=PackMode.B32_TO_B8,
                )
            kr_mask = update_mask(rope_dim, elem_bits=32)[0]
            kr_value = vload(kr_read, kr_offset)
            kr_scaled = vmul(kr_value, scale_vec, mask=full_mask())
            kr_h8 = cast_hif8(kr_scaled)
            vstore_pack(
                payload_write,
                VECTOR_KV_LORA,
                vreinterpret(kr_h8, dtypes.float8_e4m3fn),
                kr_mask,
                pack_mode=PackMode.B32_TO_B8,
            )
            vmem_bar(mode="vst_vld")

    @jit
    def process_rows(
        self,
        ws_raw,
        gamma_gm,
        cache,
        cache_index,
        scale_gm,
        kvr,
        norm_eps,
        kvr_scale,
        start,
        stride,
        rows,
        cache_mode,
    ):
        row_tiler = make_tiler((1, kvr), alignment=(1, VECTOR_SEG))
        rope_dim = kvr - VECTOR_KV_LORA
        is_hif8_quant = scale_gm is not None and cache.dtype == dtypes.uint8
        is_fp8_tensor_quant = scale_gm is not None and (not is_hif8_quant)
        is_fp8_group_quant = scale_gm is None and cache.dtype == dtypes.float8_e4m3fn
        mem_copy(
            self.gamma.next_write(), tile_slice(gamma_gm, (1, VECTOR_KV_LORA), (0, 0))
        )
        gamma_read = self.gamma.next_read()
        scale_read = self.scale_ub
        if is_fp8_tensor_quant or is_hif8_quant:
            scale_2d = scale_gm.view((1, 1))
            mem_copy(
                tile_slice(self.scale_ub, (1, 1), (0, 0)),
                tile_slice(scale_2d, (1, 1), (0, 0)),
            )
        for token in dsl.range(start, rows, stride):
            row_read = self.row.read(ws_raw, kvr, token)
            self.latent_root(row_read, norm_eps)
            if is_hif8_quant:
                payload_write = self.payload_m3.next_write()
                self.scale_pack_payload_hif8(
                    payload_write,
                    row_read,
                    gamma_read,
                    row_read,
                    VECTOR_KV_LORA,
                    kvr_scale,
                    rope_dim,
                    scale_read,
                )
                payload_read = self.payload_m3.next_read()
                write_cache_row_hif8(
                    cache, cache_index, payload_read, token, kvr, cache_mode
                )
            elif is_fp8_tensor_quant:
                payload_write = self.payload_m2.next_write()
                self.scale_pack_payload(
                    payload_write,
                    row_read,
                    gamma_read,
                    row_read,
                    VECTOR_KV_LORA,
                    kvr_scale,
                    rope_dim,
                    scale_read,
                )
                payload_read = self.payload_m2.next_read()
                write_cache_row_fp8(
                    cache, cache_index, payload_read, token, kvr, cache_mode
                )
            elif is_fp8_group_quant:
                payload_write = self.payload_q.next_write()
                self.quant_pack_payload(
                    payload_write,
                    row_read,
                    gamma_read,
                    row_read,
                    VECTOR_KV_LORA,
                    kvr_scale,
                    rope_dim,
                )
                payload_read = self.payload_q.next_read()
                write_cache_row_group_fp8(
                    cache, cache_index, payload_read, token, rope_dim
                )
            else:
                payload_write = self.payload.next_write()
                self.pack_payload(
                    payload_write,
                    row_read,
                    gamma_read,
                    row_read,
                    VECTOR_KV_LORA,
                    kvr_scale,
                    rope_dim,
                )
                payload_read = self.payload.next_read()
                write_cache_row(
                    cache, cache_index, payload_read, token, row_tiler, cache_mode
                )

    @jit
    def process_rows_rope(
        self,
        ws_raw,
        gamma_gm,
        cache,
        cache_index,
        rope_sin,
        rope_cos,
        rope_mode,
        scale_gm,
        kvr,
        norm_eps,
        kvr_scale,
        start,
        stride,
        rows,
        cache_mode,
    ):
        row_tiler = make_tiler((1, kvr), alignment=(1, VECTOR_SEG))
        rope_dim = kvr - VECTOR_KV_LORA
        is_hif8_quant = scale_gm is not None and cache.dtype == dtypes.uint8
        is_fp8_tensor_quant = scale_gm is not None and (not is_hif8_quant)
        is_fp8_group_quant = scale_gm is None and cache.dtype == dtypes.float8_e4m3fn
        mem_copy(
            self.gamma.next_write(), tile_slice(gamma_gm, (1, VECTOR_KV_LORA), (0, 0))
        )
        gamma_read = self.gamma.next_read()
        scale_read = self.scale_ub
        if is_fp8_tensor_quant or is_hif8_quant:
            scale_2d = scale_gm.view((1, 1))
            mem_copy(
                tile_slice(self.scale_ub, (1, 1), (0, 0)),
                tile_slice(scale_2d, (1, 1), (0, 0)),
            )
        for token in dsl.range(start, rows, stride):
            row_read = self.row.read(ws_raw, kvr, token)
            self.latent_root(row_read, norm_eps)
            sin_read = self.rope.load_table(rope_sin, token)
            cos_read = self.rope.load_table(rope_cos, token)
            self.rope.rotate(row_read, VECTOR_KV_LORA, sin_read, cos_read, rope_mode)
            if is_hif8_quant:
                payload_write = self.payload_m3.next_write()
                self.scale_pack_payload_hif8(
                    payload_write,
                    row_read,
                    gamma_read,
                    self.rope.out,
                    0,
                    kvr_scale,
                    rope_dim,
                    scale_read,
                )
                payload_read = self.payload_m3.next_read()
                write_cache_row_hif8(
                    cache, cache_index, payload_read, token, kvr, cache_mode
                )
            elif is_fp8_tensor_quant:
                payload_write = self.payload_m2.next_write()
                self.scale_pack_payload(
                    payload_write,
                    row_read,
                    gamma_read,
                    self.rope.out,
                    0,
                    kvr_scale,
                    rope_dim,
                    scale_read,
                )
                payload_read = self.payload_m2.next_read()
                write_cache_row_fp8(
                    cache, cache_index, payload_read, token, kvr, cache_mode
                )
            elif is_fp8_group_quant:
                payload_write = self.payload_q.next_write()
                self.quant_pack_payload(
                    payload_write,
                    row_read,
                    gamma_read,
                    self.rope.out,
                    0,
                    kvr_scale,
                    rope_dim,
                )
                payload_read = self.payload_q.next_read()
                write_cache_row_group_fp8(
                    cache, cache_index, payload_read, token, rope_dim
                )
            else:
                payload_write = self.payload.next_write()
                self.pack_payload(
                    payload_write,
                    row_read,
                    gamma_read,
                    self.rope.out,
                    0,
                    kvr_scale,
                    rope_dim,
                )
                payload_read = self.payload.next_read()
                write_cache_row(
                    cache, cache_index, payload_read, token, row_tiler, cache_mode
                )


class InputPad:
    def __init__(self):
        self.row = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, 8192)), depth=1
        )

    @jit
    def run(self, x, padded, tokens, start, stride):
        capacity = padded.shape[0]
        dim = x.shape[1]
        copy_tiler = make_tiler((1, dim), alignment=(1, 256))
        for row in dsl.range(start, capacity, stride):
            if row < tokens:
                mem_copy(self.row.next_write(), tile_slice(x, copy_tiler, (row, 0)))
                mem_copy(
                    tile_slice(padded, copy_tiler, (row, 0)),
                    from_tensor(self.row.next_read(), layout=make_layout((1, dim))),
                )
            else:
                zero_row = self.row.next_write()
                with vf(mode="simd"):
                    zero_mask = update_mask(128, elem_bits=16)[0]
                    zero = vdups(0.0, dtypes.bfloat16, mask=zero_mask)
                    for column in range_constexpr(8192 // 128):
                        vstore(zero_row, column * 128, zero, zero_mask)
                    vmem_bar(mode="vst_vld")
                mem_copy(
                    tile_slice(padded, copy_tiler, (row, 0)),
                    from_tensor(self.row.next_read(), layout=make_layout((1, dim))),
                )


class QHeadBatch:
    def __init__(self, head_batch=ROPE_HEAD_BATCH):
        self.batch_heads = head_batch
        self.src = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((self.batch_heads, VECTOR_SEG)),
            depth=2,
        )
        self.dst = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((self.batch_heads, VECTOR_SEG)),
            depth=2,
        )
        self.zpad = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.batch_heads, PAD_SEG))
        )
        self.rope = Rope()

    @jit
    def process_rows_rope(
        self,
        q_proj,
        q2d,
        rope_sin,
        rope_cos,
        rope_mode,
        heads,
        nope_dim,
        head_dim,
        kvr,
        start,
        stride,
        rows,
        batch_heads,
    ):
        groups = heads // batch_heads
        for item in dsl.range(start, rows * groups, stride):
            token = item // groups
            group = item % groups
            src_heads = q_proj[token : token + 1, 0 : heads * head_dim].view(
                (heads, head_dim)
            )
            dst_heads = q2d[token : token + 1, 0 : heads * kvr].view((heads, kvr))
            sin_read = self.rope.load_table(rope_sin, token)
            cos_read = self.rope.load_table(rope_cos, token)
            source = tile_slice(
                src_heads,
                make_tiler((batch_heads, head_dim), alignment=(1, 64)),
                (group, 0),
            )
            mem_copy(
                tile_slice(self.src.next_write(), (batch_heads, VECTOR_SEG), (0, 0)),
                source[0:batch_heads, nope_dim:head_dim],
            )
            src_read = self.src.next_read()
            with vf(mode="simd"):
                zero = vdups(0.0, dtypes.float32, mask=full_mask())
                for head in range_constexpr(batch_heads):
                    value = vload_unpack(
                        src_read, head * VECTOR_SEG, unpack_mode=UnpackMode.B16_TO_B32
                    )
                    vstore(
                        self.zpad,
                        head * PAD_SEG,
                        vcast(value, dtypes.float32, mask=full_mask()),
                        full_mask(),
                    )
                    vstore(self.zpad, head * PAD_SEG + VECTOR_SEG, zero, full_mask())
                vmem_bar(mode="vst_vld")
            dst_write = self.dst.next_write()
            if rope_mode == 1:
                with vf(mode="simd"):
                    half = update_mask(HALF, elem_bits=32)[0]
                    cos_low = vload(cos_read, 0)
                    cos_high = vload(cos_read, HALF)
                    sin_low = vload(sin_read, 0)
                    sin_high = vload(sin_read, HALF)
                    for head in range_constexpr(batch_heads):
                        (first, second) = vload_deinterleave(self.zpad, head * PAD_SEG)
                        lower = vsub(
                            vmul(first, cos_low, mask=half),
                            vmul(second, sin_low, mask=half),
                            mask=half,
                        )
                        upper = vadd(
                            vmul(second, cos_high, mask=half),
                            vmul(first, sin_high, mask=half),
                            mask=half,
                        )
                        vstore_pack(
                            dst_write,
                            head * VECTOR_SEG,
                            vcast(
                                lower,
                                dtypes.bfloat16,
                                mask=half,
                                rounding=RoundingMode.RN,
                            ),
                            half,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                        vstore_pack(
                            dst_write,
                            head * VECTOR_SEG + HALF,
                            vcast(
                                upper,
                                dtypes.bfloat16,
                                mask=half,
                                rounding=RoundingMode.RN,
                            ),
                            half,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    vmem_bar(mode="vst_vld")
            else:
                with vf(mode="simd"):
                    half = update_mask(HALF, elem_bits=32)[0]
                    cos_low = vload(cos_read, 0)
                    cos_high = vload(cos_read, HALF)
                    sin_low = vload(sin_read, 0)
                    sin_high = vload(sin_read, HALF)
                    for head in range_constexpr(batch_heads):
                        first = vload(self.zpad, head * PAD_SEG)
                        second = vload(self.zpad, head * PAD_SEG + HALF)
                        lower = vsub(
                            vmul(first, cos_low, mask=half),
                            vmul(second, sin_low, mask=half),
                            mask=half,
                        )
                        upper = vadd(
                            vmul(second, cos_high, mask=half),
                            vmul(first, sin_high, mask=half),
                            mask=half,
                        )
                        vstore_pack(
                            dst_write,
                            head * VECTOR_SEG,
                            vcast(
                                lower,
                                dtypes.bfloat16,
                                mask=half,
                                rounding=RoundingMode.RN,
                            ),
                            half,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                        vstore_pack(
                            dst_write,
                            head * VECTOR_SEG + HALF,
                            vcast(
                                upper,
                                dtypes.bfloat16,
                                mask=half,
                                rounding=RoundingMode.RN,
                            ),
                            half,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    vmem_bar(mode="vst_vld")
            destination = tile_slice(
                dst_heads, make_tiler((batch_heads, kvr), alignment=(1, 64)), (group, 0)
            )
            mem_copy(
                destination[0:batch_heads, VECTOR_KV_LORA:kvr],
                tile_slice(self.dst.next_read(), (batch_heads, VECTOR_SEG), (0, 0)),
            )


class QTailCopy:
    def __init__(
        self,
        rope_active,
        input_dtype=dtypes.float32,
        batch_capable=False,
        head_batch=ROPE_HEAD_BATCH,
        batch_min_heads=0,
    ):
        self.bf16_input = input_dtype == dtypes.bfloat16
        self.batch_capable = batch_capable
        self.head_batch_size = head_batch
        self.batch_min_heads = batch_min_heads
        if batch_capable:
            self.head_batch = QHeadBatch(head_batch)
        self.src = RingBuffer(
            MemLoc.UB, input_dtype, make_layout((1, VECTOR_SEG)), depth=2
        )
        if self.bf16_input:
            self.fp32_source = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_SEG))
            )
        self.dst = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, VECTOR_SEG)), depth=2
        )
        if rope_active:
            self.rope = Rope()

    @jit
    def load_source(self, q_proj, token, src_col):
        mem_copy(
            self.src.next_write(),
            tile_slice(q_proj, (1, VECTOR_SEG), (token, src_col // VECTOR_SEG)),
        )
        return self.src.next_read()

    @jit
    def process_rows(
        self, q_proj, q2d, heads, nope_dim, head_dim, kvr, start, stride, rows
    ):
        for item in dsl.range(start, rows * heads, stride):
            token = item // heads
            head = item % heads
            src_col = head * head_dim + nope_dim
            dst_col = head * kvr + VECTOR_KV_LORA
            src_read = self.load_source(q_proj, token, src_col)
            dst_write = self.dst.next_write()
            if self.bf16_input:
                with vf(mode="simd"):
                    value = vcast(
                        vload_unpack(src_read, 0, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full_mask(),
                    )
                    vstore_pack(
                        dst_write,
                        0,
                        vcast(
                            value,
                            dtypes.bfloat16,
                            mask=full_mask(),
                            rounding=RoundingMode.RN,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vmem_bar(mode="vst_vld")
            else:
                with vf(mode="simd"):
                    value = vload(src_read, 0)
                    vstore_pack(
                        dst_write,
                        0,
                        vcast(
                            value,
                            dtypes.bfloat16,
                            mask=full_mask(),
                            rounding=RoundingMode.RN,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vmem_bar(mode="vst_vld")
            mem_copy(
                tile_slice(q2d, (1, VECTOR_SEG), (token, dst_col // VECTOR_SEG)),
                self.dst.next_read(),
            )

    @jit
    def process_rows_rope(
        self,
        q_proj,
        q2d,
        rope_sin,
        rope_cos,
        rope_mode,
        heads,
        nope_dim,
        head_dim,
        kvr,
        start,
        stride,
        rows,
    ):
        if const_expr(self.batch_capable):
            if heads >= self.batch_min_heads and heads % ROPE_HEAD_BATCH == 0:
                if const_expr(self.head_batch_size == 8):
                    if heads % 8 == 0:
                        self.process_rows_rope_batch(
                            q_proj,
                            q2d,
                            rope_sin,
                            rope_cos,
                            rope_mode,
                            heads,
                            nope_dim,
                            head_dim,
                            kvr,
                            start,
                            stride,
                            rows,
                            8,
                        )
                    else:
                        self.process_rows_rope_batch(
                            q_proj,
                            q2d,
                            rope_sin,
                            rope_cos,
                            rope_mode,
                            heads,
                            nope_dim,
                            head_dim,
                            kvr,
                            start,
                            stride,
                            rows,
                            ROPE_HEAD_BATCH,
                        )
                else:
                    self.process_rows_rope_batch(
                        q_proj,
                        q2d,
                        rope_sin,
                        rope_cos,
                        rope_mode,
                        heads,
                        nope_dim,
                        head_dim,
                        kvr,
                        start,
                        stride,
                        rows,
                        ROPE_HEAD_BATCH,
                    )
            else:
                self.process_rows_rope_scalar(
                    q_proj,
                    q2d,
                    rope_sin,
                    rope_cos,
                    rope_mode,
                    heads,
                    nope_dim,
                    head_dim,
                    kvr,
                    start,
                    stride,
                    rows,
                )
        else:
            self.process_rows_rope_scalar(
                q_proj,
                q2d,
                rope_sin,
                rope_cos,
                rope_mode,
                heads,
                nope_dim,
                head_dim,
                kvr,
                start,
                stride,
                rows,
            )

    @jit
    def process_rows_rope_batch(
        self,
        q_proj,
        q2d,
        rope_sin,
        rope_cos,
        rope_mode,
        heads,
        nope_dim,
        head_dim,
        kvr,
        start,
        stride,
        rows,
        batch_heads,
    ):
        if rows * (heads // batch_heads) >= stride:
            self.head_batch.process_rows_rope(
                q_proj,
                q2d,
                rope_sin,
                rope_cos,
                rope_mode,
                heads,
                nope_dim,
                head_dim,
                kvr,
                start,
                stride,
                rows,
                batch_heads,
            )
        else:
            self.process_rows_rope_scalar(
                q_proj,
                q2d,
                rope_sin,
                rope_cos,
                rope_mode,
                heads,
                nope_dim,
                head_dim,
                kvr,
                start,
                stride,
                rows,
            )

    @jit
    def process_rows_rope_scalar(
        self,
        q_proj,
        q2d,
        rope_sin,
        rope_cos,
        rope_mode,
        heads,
        nope_dim,
        head_dim,
        kvr,
        start,
        stride,
        rows,
    ):
        for item in dsl.range(start, rows * heads, stride):
            token = item // heads
            head = item % heads
            sin_read = self.rope.load_table(rope_sin, token)
            cos_read = self.rope.load_table(rope_cos, token)
            src_col = head * head_dim + nope_dim
            dst_col = head * kvr + VECTOR_KV_LORA
            src_read = self.load_source(q_proj, token, src_col)
            if self.bf16_input:
                with vf(mode="simd"):
                    value = vload_unpack(src_read, 0, unpack_mode=UnpackMode.B16_TO_B32)
                    vstore(
                        self.fp32_source,
                        0,
                        vcast(value, dtypes.float32, mask=full_mask()),
                        full_mask(),
                    )
                    vmem_bar(mode="vst_vld")
                self.rope.rotate(self.fp32_source, 0, sin_read, cos_read, rope_mode)
            else:
                self.rope.rotate(src_read, 0, sin_read, cos_read, rope_mode)
            dst_write = self.dst.next_write()
            with vf(mode="simd"):
                value = vload(self.rope.out, 0)
                vstore_pack(
                    dst_write,
                    0,
                    vcast(
                        value,
                        dtypes.bfloat16,
                        mask=full_mask(),
                        rounding=RoundingMode.RN,
                    ),
                    full_mask(),
                    pack_mode=PackMode.B32_TO_B16,
                )
                vmem_bar(mode="vst_vld")
            mem_copy(
                tile_slice(q2d, (1, VECTOR_SEG), (token, dst_col // VECTOR_SEG)),
                self.dst.next_read(),
            )


class QFpTailCopy:
    def __init__(self, rope_active):
        self.src = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, VECTOR_SEG)), depth=2
        )
        self.fp32_source = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_SEG))
        )
        self.tail = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_SEG)), depth=2
        )
        if rope_active:
            self.rope = Rope()

    @jit
    def load_source(self, q_proj, token, head, head_dim, nope_dim):
        col = head * head_dim + nope_dim
        mem_copy(
            self.src.next_write(),
            tile_slice(q_proj, (1, VECTOR_SEG), (token, col // VECTOR_SEG)),
        )
        src_read = self.src.next_read()
        with vf(mode="simd"):
            value = vload_unpack(src_read, 0, unpack_mode=UnpackMode.B16_TO_B32)
            vstore(
                self.fp32_source,
                0,
                vcast(value, dtypes.float32, mask=full_mask()),
                full_mask(),
            )
            vmem_bar(mode="vst_vld")

    @jit
    def store_tail(self, tail_write, q_fp2d, token, head, kvr):
        with vf(mode="simd"):
            vstore(tail_write, 0, vload(self.rope.out, 0), full_mask())
            vmem_bar(mode="vst_vld")
        mem_copy(
            tile_slice(
                q_fp2d,
                (1, VECTOR_SEG),
                (token, (head * kvr + VECTOR_KV_LORA) // VECTOR_SEG),
            ),
            self.tail.next_read(),
        )

    @jit
    def process_rows(
        self, q_proj, q_fp2d, heads, nope_dim, head_dim, kvr, start, stride, rows
    ):
        for token in dsl.range(start, rows, stride):
            for head in dsl.range(0, heads):
                self.load_source(q_proj, token, head, head_dim, nope_dim)
                if kvr != 512:
                    tail_write = self.tail.next_write()
                    with vf(mode="simd"):
                        vstore(tail_write, 0, vload(self.fp32_source, 0), full_mask())
                        vmem_bar(mode="vst_vld")
                    mem_copy(
                        tile_slice(
                            q_fp2d,
                            (1, VECTOR_SEG),
                            (token, (head * kvr + VECTOR_KV_LORA) // VECTOR_SEG),
                        ),
                        self.tail.next_read(),
                    )

    @jit
    def process_rows_rope(
        self,
        q_proj,
        q_fp2d,
        rope_sin,
        rope_cos,
        rope_mode,
        heads,
        nope_dim,
        head_dim,
        kvr,
        start,
        stride,
        rows,
    ):
        for token in dsl.range(start, rows, stride):
            sin_read = self.rope.load_table(rope_sin, token)
            cos_read = self.rope.load_table(rope_cos, token)
            for head in dsl.range(0, heads):
                self.load_source(q_proj, token, head, head_dim, nope_dim)
                if kvr != 512:
                    self.rope.rotate(self.fp32_source, 0, sin_read, cos_read, rope_mode)
                    self.store_tail(self.tail.next_write(), q_fp2d, token, head, kvr)


class QRowQuant:
    def __init__(self):
        self.row = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.payload = RingBuffer(
            MemLoc.UB, dtypes.float8_e4m3fn, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.scalar = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 8)))

    @jit
    def process_rows(self, q_fp2d, q2d, descale_q2d, heads, kvr, start, stride, rows):
        row_tiler = make_tiler((1, kvr), alignment=(1, VECTOR_SEG))
        for r in dsl.range(start, rows, stride):
            token = r // heads
            head = r % heads
            mem_copy(
                self.row.next_write(), tile_slice(q_fp2d, row_tiler, (token, head))
            )
            row_read = self.row.next_read()
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                amax = vdups(0.0, dtypes.float32, mask=one)
                for segment in range_constexpr(VECTOR_KVR_MAX // VECTOR_SEG):
                    lanes = kvr - segment * VECTOR_SEG
                    lanes = min(max(lanes, 0), VECTOR_SEG)
                    seg_mask = update_mask(lanes, elem_bits=32)[0]
                    value = vload(row_read, segment * VECTOR_SEG)
                    amax = vmax(
                        amax,
                        vreduce_max(vabs(value, mask=seg_mask), mask=seg_mask),
                        mask=one,
                    )
                limit = vdups(FP8_MAX, dtypes.float32, mask=one)
                descale = vdiv(amax, limit, mask=one, precision_mode=True)
                zero = vdups(0.0, dtypes.float32, mask=one)
                zero_mask = veq(amax, zero, mask=one)
                ones = vdups(1.0, dtypes.float32, mask=one)
                denom = vselect(ones, descale, cond_mask=zero_mask)
                vstore_first(self.scalar, 0, descale)
                vstore_first(self.scalar, 1, denom)
                vstore_first(self.scalar, 2, amax)
                vmem_bar(mode="vst_vld")
            payload_write = self.payload.next_write()
            with vf(mode="simd"):
                denom_vec = vload_broadcast(self.scalar, 1)
                amax_vec = vload_broadcast(self.scalar, 2)
                zero_full = veq(
                    amax_vec,
                    vdups(0.0, dtypes.float32, mask=full_mask()),
                    mask=full_mask(),
                )
                zeros_vec = vdups(0.0, dtypes.float32, mask=full_mask())
                for segment in dsl.range(0, kvr, VECTOR_SEG):
                    value = vload(row_read, segment)
                    normalized = vdiv(
                        value, denom_vec, mask=full_mask(), precision_mode=True
                    )
                    normalized = vselect(zeros_vec, normalized, cond_mask=zero_full)
                    normalized = vmaxs(normalized, -FP8_MAX, mask=full_mask())
                    normalized = vmins(normalized, FP8_MAX, mask=full_mask())
                    vstore_pack(
                        payload_write,
                        segment,
                        vcast(
                            normalized,
                            dtypes.float8_e4m3fn,
                            mask=full_mask(),
                            reg_layout=RegLayout.ZERO,
                        ),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B8,
                    )
                vmem_bar(mode="vst_vld")
            payload_read = self.payload.next_read()
            if const_expr(isinstance(kvr, int)):
                byte_view = tile_slice(payload_read, (1, kvr), (0, 0))
            else:
                byte_view = from_tensor(
                    payload_read,
                    dtype=dtypes.float8_e4m3fn,
                    layout=make_layout((1, kvr)),
                    offset=0,
                )
            mem_copy(tile_slice(q2d, row_tiler, (token, head)), byte_view)
            mem_copy(
                tile_slice(descale_q2d, (1, 1), (r, 0)),
                tile_slice(self.scalar, (1, 1), (0, 0)),
            )


class QAmaxStage:
    def __init__(self):
        self.row = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.carry = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, 64)), depth=2
        )
        self.scalar = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 8)))

    @jit
    def partial_amax(
        self, q_fp2d, ws_amax, heads, kvr, block, blocks, rows, accumulate=False
    ):
        row_tiler = make_tiler((1, kvr), alignment=(1, VECTOR_SEG))
        ws_col = ws_amax.view((ws_amax.shape[0], 1))
        w_init = self.carry.next_write()
        if accumulate:
            mem_copy(
                tile_slice(w_init, (1, 1), (0, 0)),
                tile_slice(ws_col, (1, 1), (block, 0)),
            )
            mem_copy(
                tile_slice(self.scalar, (1, 1), (0, 0)),
                tile_slice(ws_col, (1, 1), (block, 0)),
            )
        else:
            with vf(mode="simd"):
                vstore(
                    w_init, 0, vdups(0.0, dtypes.float32, mask=full_mask()), full_mask()
                )
                vstore_first(
                    self.scalar, 0, vdups(0.0, dtypes.float32, mask=full_mask())
                )
                vmem_bar(mode="vst_vld")
        for r in dsl.range(block, rows, blocks):
            token = r // heads
            head = r % heads
            carry_read = self.carry.next_read()
            carry_write = self.carry.next_write()
            mem_copy(
                self.row.next_write(), tile_slice(q_fp2d, row_tiler, (token, head))
            )
            row_read = self.row.next_read()
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                row_amax = vdups(0.0, dtypes.float32, mask=one)
                for segment in range_constexpr(VECTOR_KVR_MAX // VECTOR_SEG):
                    lanes = kvr - segment * VECTOR_SEG
                    lanes = min(max(lanes, 0), VECTOR_SEG)
                    seg_mask = update_mask(lanes, elem_bits=32)[0]
                    value = vload(row_read, segment * VECTOR_SEG)
                    row_amax = vmax(
                        row_amax,
                        vreduce_max(vabs(value, mask=seg_mask), mask=seg_mask),
                        mask=one,
                    )
                vstore_first(self.scalar, 2, row_amax)
                vmem_bar(mode="vst_vld")
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                prev = vload(carry_read, 0)
                row_amax = vload_broadcast(self.scalar, 2)
                total = vmax(prev, row_amax, mask=one)
                vstore(carry_write, 0, total, one)
                vstore_first(self.scalar, 0, total)
                vmem_bar(mode="vst_vld")
        # The next stage reads this slot only after the launch-wide barrier.
        mem_copy(
            tile_slice(ws_col, (1, 1), (block, 0)),
            tile_slice(self.scalar, (1, 1), (0, 0)),
        )


class QGlobalQuant:
    def __init__(self):
        self.row = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.slots = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, 64)), depth=1
        )
        self.payload = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((1, VECTOR_KVR_MAX)), depth=2
        )
        self.scalar = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 8)))

    @jit
    def quantize_rows(
        self,
        q_fp2d,
        q2d,
        descale_q_gm,
        ws_amax,
        heads,
        kvr,
        blocks,
        start,
        stride,
        rows,
        write_descale,
    ):
        row_tiler = make_tiler((1, kvr), alignment=(1, VECTOR_SEG))
        descale_2d = descale_q_gm.view((1, 1))
        ws_row = ws_amax.view((1, ws_amax.shape[0]))
        with vf(mode="simd"):
            init_mask = update_mask(1, elem_bits=32)[0]
            vstore_first(self.scalar, 2, vdups(0.0, dtypes.float32, mask=init_mask))
            vmem_bar(mode="vst_vld")
        for first_slot in dsl.range(0, blocks, AMAX_VECTOR_LANES):
            mem_copy(
                self.slots.next_write(),
                tile_slice(
                    ws_row, (1, AMAX_VECTOR_LANES), (0, first_slot // AMAX_VECTOR_LANES)
                ),
            )
            slots_read = self.slots.next_read()
            with vf(mode="simd"):
                partial_mask = update_mask(1, elem_bits=32)[0]
                slot_mask = update_mask(
                    min(blocks - first_slot, AMAX_VECTOR_LANES), elem_bits=32
                )[0]
                partial_max = vmax(
                    vload_broadcast(self.scalar, 2),
                    vreduce_max(vload(slots_read, 0), mask=slot_mask),
                    mask=partial_mask,
                )
                vstore_first(self.scalar, 2, partial_max)
                vmem_bar(mode="vst_vld")
        with vf(mode="simd"):
            one = update_mask(1, elem_bits=32)[0]
            amax = vload_broadcast(self.scalar, 2)
            limit = vdups(HIF8_MAX, dtypes.float32, mask=one)
            descale = vdiv(amax, limit, mask=one, precision_mode=True)
            zero = vdups(0.0, dtypes.float32, mask=one)
            zero_mask = veq(amax, zero, mask=one)
            ones = vdups(1.0, dtypes.float32, mask=one)
            denom = vselect(ones, descale, cond_mask=zero_mask)
            vstore_first(self.scalar, 0, descale)
            vstore_first(self.scalar, 1, denom)
            vstore_first(self.scalar, 2, amax)
            vmem_bar(mode="vst_vld")
        if write_descale:
            mem_copy(
                tile_slice(descale_2d, (1, 1), (0, 0)),
                tile_slice(self.scalar, (1, 1), (0, 0)),
            )
        for r in dsl.range(start, rows, stride):
            token = r // heads
            head = r % heads
            mem_copy(
                self.row.next_write(), tile_slice(q_fp2d, row_tiler, (token, head))
            )
            row_read = self.row.next_read()
            payload_write = self.payload.next_write()
            with vf(mode="simd"):
                denom_vec = vload_broadcast(self.scalar, 1)
                amax_vec = vload_broadcast(self.scalar, 2)
                zero_full = veq(
                    amax_vec,
                    vdups(0.0, dtypes.float32, mask=full_mask()),
                    mask=full_mask(),
                )
                zeros_vec = vdups(0.0, dtypes.float32, mask=full_mask())
                for segment in dsl.range(0, kvr, VECTOR_SEG):
                    value = vload(row_read, segment)
                    normalized = vdiv(
                        value, denom_vec, mask=full_mask(), precision_mode=True
                    )
                    normalized = vselect(zeros_vec, normalized, cond_mask=zero_full)
                    h8 = cast_hif8(normalized)
                    vstore_pack(
                        payload_write,
                        segment,
                        vreinterpret(h8, dtypes.float8_e4m3fn),
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B8,
                    )
                vmem_bar(mode="vst_vld")
            payload_read = self.payload.next_read()
            if const_expr(isinstance(kvr, int)):
                byte_view = tile_slice(payload_read, (1, kvr), (0, 0))
            else:
                byte_view = from_tensor(
                    payload_read,
                    dtype=dtypes.uint8,
                    layout=make_layout((1, kvr)),
                    offset=0,
                )
            mem_copy(tile_slice(q2d, row_tiler, (token, head)), byte_view)


@dataclass(frozen=True, slots=True)
class _ArgumentField:
    name: str
    annotation: object
    role: str


# Order and type are defined once, shared by @host, @kernel and launch binding.
# ws_qa_fp32 retains inference from its TensorSpec, as in the published ABI.
_DEVICE_ARGUMENT_FIELDS = (
    _ArgumentField("x", Tensor, "input"),
    _ArgumentField("wqa", Tensor, "input"),
    _ArgumentField("wqb", Tensor, "input"),
    _ArgumentField("wkva", Tensor, "input"),
    _ArgumentField("wkb", Tensor, "input"),
    _ArgumentField("norm_weight_qa", Tensor, "input"),
    _ArgumentField("norm_weight_kva", Tensor, "input"),
    _ArgumentField("kv_cache", Tensor, "input"),
    _ArgumentField("cache_index", Tensor, "input"),
    _ArgumentField("quant_scale_kv", Tensor, "quant_input"),
    _ArgumentField("rope_sin", Tensor, "rope"),
    _ArgumentField("rope_cos", Tensor, "rope"),
    _ArgumentField("q", Tensor, "output"),
    _ArgumentField("descale_q", Tensor, "quant_output"),
    _ArgumentField("qr", Tensor, "qr"),
    _ArgumentField("ws_x_pad", Tensor, "scratch"),
    _ArgumentField("ws_qa_raw", Tensor, "scratch"),
    _ArgumentField("ws_qa_fp32", inspect.Parameter.empty, "qa_fp32"),
    _ArgumentField("ws_kva_raw", Tensor, "scratch"),
    _ArgumentField("ws_q_proj", Tensor, "scratch"),
    _ArgumentField("ws_q_fp", Tensor, "quant_scratch"),
    _ArgumentField("ws_amax", Tensor, "amax"),
    _ArgumentField("ws_qb_aux", Tensor, "scratch"),
    _ArgumentField("norm_eps", dtypes.float32, "scalar"),
    _ArgumentField("qr_scale", dtypes.float32, "scalar"),
    _ArgumentField("kvr_scale", dtypes.float32, "scalar"),
    _ArgumentField("cache_mode", dtypes.int64, "scalar"),
    _ArgumentField("rope_mode", dtypes.int64, "scalar"),
    _ArgumentField("front_load_hint", dtypes.int32, "hints"),
    _ArgumentField("qb_columns", dtypes.int32, "hints"),
    _ArgumentField("qb_load_hint", dtypes.int32, "hints"),
    _ArgumentField("wkb_load_hint", dtypes.int32, "hints"),
)
_HOST_CORE_FIELD = _ArgumentField("core_num", dtypes.int64, "host")


def argument_surface(quant_mode, rope, return_qr, *, host_surface=False):
    roles = {"input", "output", "scratch", "scalar"}
    if quant_mode in (2, 3):
        roles.update(("quant_input", "quant_output", "quant_scratch"))
    if quant_mode == 3:
        roles.add("amax")
    if rope:
        roles.add("rope")
    if return_qr:
        roles.add("qr")
    if rope and return_qr:
        roles.update(("qa_fp32", "hints"))
    fields = tuple(field for field in _DEVICE_ARGUMENT_FIELDS if field.role in roles)
    if host_surface:
        fields += (_HOST_CORE_FIELD,)
    return (
        tuple(field.name for field in fields),
        {
            field.name: field.annotation
            for field in fields
            if field.annotation is not inspect.Parameter.empty
        },
    )


ROPE_QR_KERNEL_ARGUMENTS, ROPE_QR_KERNEL_ANNOTATIONS = argument_surface(
    0, True, True, host_surface=False
)
ROPE_QR_HOST_ARGUMENTS, ROPE_QR_HOST_ANNOTATIONS = argument_surface(
    0, True, True, host_surface=True
)
M2_ROPE_QR_KERNEL_ARGUMENTS, M2_ROPE_QR_KERNEL_ANNOTATIONS = argument_surface(
    2, True, True, host_surface=False
)
M2_ROPE_QR_HOST_ARGUMENTS, M2_ROPE_QR_HOST_ANNOTATIONS = argument_surface(
    2, True, True, host_surface=True
)
M3_ROPE_QR_KERNEL_ARGUMENTS, M3_ROPE_QR_KERNEL_ANNOTATIONS = argument_surface(
    3, True, True, host_surface=False
)
M3_ROPE_QR_HOST_ARGUMENTS, M3_ROPE_QR_HOST_ANNOTATIONS = argument_surface(
    3, True, True, host_surface=True
)
_LAUNCH_ARGUMENTS_BY_MODE = (
    ROPE_QR_HOST_ARGUMENTS,
    ROPE_QR_HOST_ARGUMENTS,
    M2_ROPE_QR_HOST_ARGUMENTS,
    M3_ROPE_QR_HOST_ARGUMENTS,
)


def surface_signature(names, annotations):
    def decorate(function):
        parameters = [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        for name in names:
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotations.get(name, inspect.Parameter.empty),
                )
            )
        function.__signature__ = inspect.Signature(parameters)
        function.__annotations__ = {
            name: annotations[name] for name in names if name in annotations
        }
        return function

    return decorate


@jit
def chunk_rows_2d(tensor, rows):
    return tile_slice(
        tensor, make_tiler((rows, tensor.shape[1]), alignment=(1, 1)), (0, 0)
    )


@jit
def token_chunk(tensor, chunk, rows, capacity):
    if const_expr(len(tensor.shape) == 1):
        window = tile_slice(tensor, make_tiler((capacity,), alignment=(1,)), (chunk,))
        return tile_slice(window, make_tiler((rows,), alignment=(1,)), (0,))
    elif const_expr(len(tensor.shape) == 2):
        window = tile_slice(
            tensor,
            make_tiler((capacity, tensor.shape[1]), alignment=(1, 1)),
            (chunk, 0),
        )
        return chunk_rows_2d(window, rows)
    else:
        window = tile_slice(
            tensor,
            make_tiler(
                (capacity, tensor.shape[1], tensor.shape[2]), alignment=(1, 1, 1)
            ),
            (chunk, 0, 0),
        )
        return tile_slice(
            window,
            make_tiler((rows, tensor.shape[1], tensor.shape[2]), alignment=(1, 1, 1)),
            (0, 0, 0),
        )


@jit
def qb_projection_path(
    bm,
    width,
    load_hint,
    allow_wide,
    a_gm,
    weight,
    out_gm,
    q_lora,
    rows_qb,
    m_tiles,
    block,
    blocks,
):
    onchip_rewind(reset_sync_id=True)
    panel_k = (
        (512 if width <= 192 else NZ_PANEL_K)
        if bm <= QB_WIDE_PANEL_MAX_ROWS and q_lora != 640
        else (
            NZ_PANEL_K
            if bm <= QB_MEDIUM_PANEL_MAX_ROWS and q_lora != 640 and allow_wide
            else CUBE_BK
        )
    )
    projection = NzProjection(bm, width, 1, weight_l2_hint=load_hint, panel_k=panel_k)
    projection.project_owner(
        a_gm, weight, out_gm, q_lora, rows_qb, m_tiles, block, blocks
    )
    cube_sync_all()
    onchip_rewind(reset_sync_id=True)


@jit
def qb_runtime_path(
    bm,
    columns,
    load_hint,
    allow_wide,
    a_gm,
    weight,
    out_gm,
    q_lora,
    rows_qb,
    m_tiles,
    block,
    blocks,
):
    # Structural if/elif exclusivity is required for Buffer-ID/address reuse.
    if columns == 96:
        qb_projection_path(
            bm,
            96,
            load_hint,
            allow_wide,
            a_gm,
            weight,
            out_gm,
            q_lora,
            rows_qb,
            m_tiles,
            block,
            blocks,
        )
    elif columns == 128:
        qb_projection_path(
            bm,
            128,
            load_hint,
            allow_wide,
            a_gm,
            weight,
            out_gm,
            q_lora,
            rows_qb,
            m_tiles,
            block,
            blocks,
        )
    elif const_expr(allow_wide):
        if columns == 192:
            qb_projection_path(
                bm,
                192,
                load_hint,
                allow_wide,
                a_gm,
                weight,
                out_gm,
                q_lora,
                rows_qb,
                m_tiles,
                block,
                blocks,
            )
        else:
            qb_projection_path(
                bm,
                256,
                load_hint,
                allow_wide,
                a_gm,
                weight,
                out_gm,
                q_lora,
                rows_qb,
                m_tiles,
                block,
                blocks,
            )


@jit
def qb_grouped_path(
    bm, head_width, a_gm, weight, out_gm, q_lora, rows_qb, m_tiles, block, blocks
):
    onchip_rewind(reset_sync_id=True)
    projection = QbFourHeads(bm, head_width, q_lora)
    projection.project_owner(a_gm, weight, out_gm, rows_qb, m_tiles, block, blocks)
    cube_sync_all()
    onchip_rewind(reset_sync_id=True)


# Dispatch outside the token loop so mutually exclusive schedules have
# independent Buffer-ID lifetimes while sharing one Native entry.
@jit
def run_all_profiles(
    bm,
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    rope_sin,
    rope_cos,
    q,
    qr,
    ws_x_pad,
    ws_qa_raw,
    ws_qa_fp32,
    ws_kva_raw,
    ws_q_proj,
    ws_qb_aux,
    norm_eps,
    qr_scale,
    kvr_scale,
    cache_mode,
    rope_mode,
    rope_active,
    return_qr,
    nz_active,
    nz_wfront,
    nz_wqb,
    head_hint,
    head_width,
    scale_gm,
    descale_q,
    ws_q_fp,
    is_fp8_tensor_quant,
    ws_amax,
    is_hif8_quant,
    front_load_hint,
    qb_columns,
    qb_load_hint,
    wkb_load_hint,
):
    if x.shape[0] <= 8:
        if const_expr(nz_active):
            if const_expr(wqa.shape[1] == 1536):
                onchip_rewind(reset_sync_id=True)
                run_schedule(
                    8,
                    x,
                    wqa,
                    wqb,
                    wkva,
                    wkb,
                    norm_weight_qa,
                    norm_weight_kva,
                    kv_cache,
                    cache_index,
                    rope_sin,
                    rope_cos,
                    q,
                    qr,
                    ws_x_pad,
                    ws_qa_raw,
                    ws_kva_raw,
                    ws_q_proj,
                    ws_qb_aux,
                    norm_eps,
                    qr_scale,
                    kvr_scale,
                    cache_mode,
                    rope_mode,
                    rope_active,
                    return_qr,
                    nz_active,
                    48,
                    nz_wqb,
                    head_hint,
                    head_width,
                    scale_gm,
                    descale_q,
                    ws_q_fp,
                    is_fp8_tensor_quant,
                    ws_amax,
                    is_hif8_quant,
                    front_load_hint,
                    qb_columns,
                    qb_load_hint,
                    wkb_load_hint,
                )
            elif const_expr(wqa.shape[1] == 640 or wqa.shape[1] == 1024):
                onchip_rewind(reset_sync_id=True)
                run_schedule(
                    8,
                    x,
                    wqa,
                    wqb,
                    wkva,
                    wkb,
                    norm_weight_qa,
                    norm_weight_kva,
                    kv_cache,
                    cache_index,
                    rope_sin,
                    rope_cos,
                    q,
                    qr,
                    ws_x_pad,
                    ws_qa_raw,
                    ws_kva_raw,
                    ws_q_proj,
                    ws_qb_aux,
                    norm_eps,
                    qr_scale,
                    kvr_scale,
                    cache_mode,
                    rope_mode,
                    rope_active,
                    return_qr,
                    nz_active,
                    32,
                    nz_wqb,
                    head_hint,
                    head_width,
                    scale_gm,
                    descale_q,
                    ws_q_fp,
                    is_fp8_tensor_quant,
                    ws_amax,
                    is_hif8_quant,
                    front_load_hint,
                    qb_columns,
                    qb_load_hint,
                    wkb_load_hint,
                )
            else:
                onchip_rewind(reset_sync_id=True)
                run_schedule(
                    8,
                    x,
                    wqa,
                    wqb,
                    wkva,
                    wkb,
                    norm_weight_qa,
                    norm_weight_kva,
                    kv_cache,
                    cache_index,
                    rope_sin,
                    rope_cos,
                    q,
                    qr,
                    ws_x_pad,
                    ws_qa_raw,
                    ws_kva_raw,
                    ws_q_proj,
                    ws_qb_aux,
                    norm_eps,
                    qr_scale,
                    kvr_scale,
                    cache_mode,
                    rope_mode,
                    rope_active,
                    return_qr,
                    nz_active,
                    64,
                    nz_wqb,
                    head_hint,
                    head_width,
                    scale_gm,
                    descale_q,
                    ws_q_fp,
                    is_fp8_tensor_quant,
                    ws_amax,
                    is_hif8_quant,
                    front_load_hint,
                    qb_columns,
                    qb_load_hint,
                    wkb_load_hint,
                )
        else:
            onchip_rewind(reset_sync_id=True)
            run_schedule(
                8,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_fp32,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                64,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
    elif x.shape[0] <= 32:
        if const_expr(
            nz_active and kv_cache.dtype == dtypes.bfloat16 and (wqa.shape[1] == 1536)
        ):
            onchip_rewind(reset_sync_id=True)
            run_schedule(
                32,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_fp32,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                64,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
        else:
            onchip_rewind(reset_sync_id=True)
            run_schedule(
                32,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                64,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
    elif x.shape[0] <= 64:
        if const_expr(
            nz_active and kv_cache.dtype == dtypes.bfloat16 and (wqa.shape[1] == 1536)
        ):
            onchip_rewind(reset_sync_id=True)
            run_schedule(
                64,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_fp32,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                64,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
        else:
            onchip_rewind(reset_sync_id=True)
            run_schedule(
                64,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                64,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
    else:
        onchip_rewind(reset_sync_id=True)
        run_schedule(
            128,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_active,
            return_qr,
            nz_active,
            128,
            nz_wqb,
            head_hint,
            head_width,
            scale_gm,
            descale_q,
            ws_q_fp,
            is_fp8_tensor_quant,
            ws_amax,
            is_hif8_quant,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )


@jit
def run_schedule(
    bm,
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    rope_sin,
    rope_cos,
    q,
    qr,
    ws_x_pad,
    ws_qa_raw,
    ws_kva_raw,
    ws_q_proj,
    ws_qb_aux,
    norm_eps,
    qr_scale,
    kvr_scale,
    cache_mode,
    rope_mode,
    rope_active,
    return_qr,
    nz_active,
    nz_wfront,
    nz_wqb,
    head_hint,
    head_width,
    scale_gm,
    descale_q,
    ws_q_fp,
    is_fp8_tensor_quant,
    ws_amax,
    is_hif8_quant,
    front_load_hint,
    qb_columns,
    qb_load_hint,
    wkb_load_hint,
):
    if const_expr(bm == 8):
        if x.shape[0] == 1:
            run_schedule_impl(
                1,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
        else:
            run_schedule_impl(
                8,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
    elif const_expr(bm == 32):
        if x.shape[0] <= 16:
            run_schedule_impl(
                16,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
        else:
            run_schedule_impl(
                32,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
    elif const_expr(bm == 128):
        if x.shape[0] <= 128:
            run_schedule_impl(
                128,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                64,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
        else:
            run_schedule_impl(
                128,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                128,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
    else:
        run_schedule_impl(
            bm,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_active,
            return_qr,
            nz_active,
            nz_wfront,
            nz_wqb,
            head_hint,
            head_width,
            scale_gm,
            descale_q,
            ws_q_fp,
            is_fp8_tensor_quant,
            ws_amax,
            is_hif8_quant,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )


@jit
def run_schedule_impl(
    bm,
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    rope_sin,
    rope_cos,
    q,
    qr,
    ws_x_pad,
    ws_qa_raw,
    ws_kva_raw,
    ws_q_proj,
    ws_qb_aux,
    norm_eps,
    qr_scale,
    kvr_scale,
    cache_mode,
    rope_mode,
    rope_active,
    return_qr,
    nz_active,
    nz_wfront,
    nz_wqb,
    head_hint,
    head_width,
    scale_gm,
    descale_q,
    ws_q_fp,
    is_fp8_tensor_quant,
    ws_amax,
    is_hif8_quant,
    front_load_hint,
    qb_columns,
    qb_load_hint,
    wkb_load_hint,
):
    if const_expr(bm < 128 or not nz_active):
        run_phases(
            bm,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_active,
            return_qr,
            nz_active,
            nz_wfront,
            nz_wqb,
            head_hint,
            head_width,
            scale_gm,
            descale_q,
            ws_q_fp,
            is_fp8_tensor_quant,
            ws_amax,
            is_hif8_quant,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )
    else:
        if qb_columns == 96:
            onchip_rewind(reset_sync_id=True)
            run_phases(
                bm,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                96,
                qb_load_hint,
                wkb_load_hint,
            )
        elif qb_columns == 128:
            onchip_rewind(reset_sync_id=True)
            run_phases(
                bm,
                x,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                q,
                qr,
                ws_x_pad,
                ws_qa_raw,
                ws_kva_raw,
                ws_q_proj,
                ws_qb_aux,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_q,
                ws_q_fp,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                front_load_hint,
                128,
                qb_load_hint,
                wkb_load_hint,
            )
        elif const_expr(kv_cache.dtype == dtypes.bfloat16):
            if qb_columns == 192:
                onchip_rewind(reset_sync_id=True)
                run_phases(
                    bm,
                    x,
                    wqa,
                    wqb,
                    wkva,
                    wkb,
                    norm_weight_qa,
                    norm_weight_kva,
                    kv_cache,
                    cache_index,
                    rope_sin,
                    rope_cos,
                    q,
                    qr,
                    ws_x_pad,
                    ws_qa_raw,
                    ws_kva_raw,
                    ws_q_proj,
                    ws_qb_aux,
                    norm_eps,
                    qr_scale,
                    kvr_scale,
                    cache_mode,
                    rope_mode,
                    rope_active,
                    return_qr,
                    nz_active,
                    nz_wfront,
                    nz_wqb,
                    head_hint,
                    head_width,
                    scale_gm,
                    descale_q,
                    ws_q_fp,
                    is_fp8_tensor_quant,
                    ws_amax,
                    is_hif8_quant,
                    front_load_hint,
                    192,
                    qb_load_hint,
                    wkb_load_hint,
                )
            else:
                onchip_rewind(reset_sync_id=True)
                run_phases(
                    bm,
                    x,
                    wqa,
                    wqb,
                    wkva,
                    wkb,
                    norm_weight_qa,
                    norm_weight_kva,
                    kv_cache,
                    cache_index,
                    rope_sin,
                    rope_cos,
                    q,
                    qr,
                    ws_x_pad,
                    ws_qa_raw,
                    ws_kva_raw,
                    ws_q_proj,
                    ws_qb_aux,
                    norm_eps,
                    qr_scale,
                    kvr_scale,
                    cache_mode,
                    rope_mode,
                    rope_active,
                    return_qr,
                    nz_active,
                    nz_wfront,
                    nz_wqb,
                    head_hint,
                    head_width,
                    scale_gm,
                    descale_q,
                    ws_q_fp,
                    is_fp8_tensor_quant,
                    ws_amax,
                    is_hif8_quant,
                    front_load_hint,
                    256,
                    qb_load_hint,
                    wkb_load_hint,
                )


@jit
def run_phases(
    bm,
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    rope_sin,
    rope_cos,
    q,
    qr,
    ws_x_pad,
    ws_qa_raw,
    ws_kva_raw,
    ws_q_proj,
    ws_qb_aux,
    norm_eps,
    qr_scale,
    kvr_scale,
    cache_mode,
    rope_mode,
    rope_active,
    return_qr,
    nz_active,
    nz_wfront,
    nz_wqb,
    head_hint,
    head_width,
    scale_gm,
    descale_q,
    ws_q_fp,
    is_fp8_tensor_quant,
    ws_amax,
    is_hif8_quant,
    front_load_hint=0,
    qb_columns=128,
    qb_load_hint=0,
    wkb_load_hint=0,
):
    if const_expr(bm < 128):
        run_chunk(
            bm,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_active,
            return_qr,
            nz_active,
            nz_wfront,
            nz_wqb,
            head_hint,
            head_width,
            scale_gm,
            descale_q,
            ws_q_fp,
            is_fp8_tensor_quant,
            ws_amax,
            is_hif8_quant,
            0,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )
    else:
        tokens = x.shape[0]
        padding_slot = 0
        if tokens % bm != 0:
            padding_slot = 1
        chunk_capacity = ws_q_proj.shape[0]
        for start in dsl.range(0, tokens, chunk_capacity):
            chunk = start // chunk_capacity
            x_chunk = tile_slice(
                x,
                make_tiler((chunk_capacity, x.shape[1]), alignment=(1, 1)),
                (chunk, 0),
            )
            valid = x_chunk.shape[0]
            rows = (valid + bm - 1) // bm * bm
            q_chunk = token_chunk(q, chunk, rows, chunk_capacity)
            cache_index_chunk = token_chunk(cache_index, chunk, valid, chunk_capacity)
            x_pad_chunk = chunk_rows_2d(ws_x_pad, rows)
            front_input = [x_chunk, x_pad_chunk][padding_slot]
            qa_chunk = chunk_rows_2d(ws_qa_raw, rows)
            kva_chunk = chunk_rows_2d(ws_kva_raw, rows)
            q_proj_chunk = chunk_rows_2d(ws_q_proj, rows)
            qb_aux_index = 0
            if return_qr:
                qb_aux_index = chunk
            qb_aux_chunk = token_chunk(ws_qb_aux, qb_aux_index, rows, chunk_capacity)
            # Empty views carry disabled optional inputs; only the runtime
            # enabled branches access their storage.
            sin_chunk = token_chunk(rope_sin, chunk, valid, chunk_capacity)
            cos_chunk = token_chunk(rope_cos, chunk, valid, chunk_capacity)
            qr_chunk = token_chunk(qr, chunk, rows, chunk_capacity)
            if const_expr(is_hif8_quant):
                q_fp_index = 0
                if ws_q_fp.shape[0] >= q.shape[0]:
                    q_fp_index = chunk
                q_fp_chunk = token_chunk(ws_q_fp, q_fp_index, rows, chunk_capacity)
                descale_chunk = descale_q
            elif const_expr(is_fp8_tensor_quant):
                q_fp_chunk = tile_slice(
                    ws_q_fp,
                    make_tiler(
                        (rows, ws_q_fp.shape[1], ws_q_fp.shape[2]), alignment=(1, 1, 1)
                    ),
                    (0, 0, 0),
                )
                descale_chunk = token_chunk(descale_q, chunk, valid, chunk_capacity)
            else:
                q_fp_chunk = None
                descale_chunk = None
            run_chunk(
                bm,
                x_chunk,
                wqa,
                wqb,
                wkva,
                wkb,
                norm_weight_qa,
                norm_weight_kva,
                kv_cache,
                cache_index_chunk,
                sin_chunk,
                cos_chunk,
                q_chunk,
                qr_chunk,
                front_input,
                qa_chunk,
                kva_chunk,
                q_proj_chunk,
                qb_aux_chunk,
                norm_eps,
                qr_scale,
                kvr_scale,
                cache_mode,
                rope_mode,
                rope_active,
                return_qr,
                nz_active,
                nz_wfront,
                nz_wqb,
                head_hint,
                head_width,
                scale_gm,
                descale_chunk,
                q_fp_chunk,
                is_fp8_tensor_quant,
                ws_amax,
                is_hif8_quant,
                padding_slot,
                front_load_hint,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
            if const_expr(is_hif8_quant):
                if ws_q_fp.shape[0] < q.shape[0]:
                    if get_subblock_id() == 0:
                        QAmaxStage().partial_amax(
                            q_fp_chunk.view((rows, q.shape[1] * q.shape[2])),
                            ws_amax,
                            q.shape[1],
                            q.shape[2],
                            get_block_idx(),
                            get_block_num(),
                            valid * q.shape[1],
                            start != 0,
                        )
            # All readers finish before the next chunk overwrites the arena.
            global_sync_all(flag_ids=(13, 14, 15))
            onchip_rewind(reset_sync_id=True)
    if const_expr(is_hif8_quant):
        if ws_q_fp.shape[0] < q.shape[0]:
            recompute_global_q_chunks(
                bm,
                x,
                wqb,
                wkb,
                ws_qb_aux,
                ws_q_proj,
                q,
                ws_q_fp,
                descale_q,
                ws_amax,
                rope_sin,
                rope_cos,
                rope_mode,
                rope_active,
                nz_active,
                qb_columns,
                qb_load_hint,
                wkb_load_hint,
            )
        else:
            block = get_block_idx()
            blocks = get_block_num()
            subblock = get_subblock_id()
            owner = block * 2 + subblock
            tokens = x.shape[0]
            heads = wkb.shape[0]
            kvr = q.shape[2]
            ws_q_fp2d = ws_q_fp.view((ws_q_fp.shape[0], heads * kvr))
            q2d = q.view((q.shape[0], heads * kvr))
            amax_stage = QAmaxStage()
            if subblock == 0:
                amax_stage.partial_amax(
                    ws_q_fp2d, ws_amax, heads, kvr, block, blocks, tokens * heads
                )
            global_sync_all(flag_ids=(13, 14, 15))
            onchip_rewind(reset_sync_id=True)
            global_quant = QGlobalQuant()
            if owner < tokens * heads:
                global_quant.quantize_rows(
                    ws_q_fp2d,
                    q2d,
                    descale_q,
                    ws_amax,
                    heads,
                    kvr,
                    blocks,
                    owner,
                    blocks * 2,
                    tokens * heads,
                    block + subblock == 0,
                )


@jit
def recompute_global_q_chunks(
    bm,
    x,
    wqb,
    wkb,
    qr_input,
    ws_q_proj,
    q,
    ws_q_fp,
    descale_q,
    ws_amax,
    rope_sin,
    rope_cos,
    rope_mode,
    rope_active,
    nz_active,
    qb_columns,
    qb_load_hint,
    wkb_load_hint,
):
    tokens = x.shape[0]
    capacity = ws_q_proj.shape[0]
    block = get_block_idx()
    blocks = get_block_num()
    owner = block * 2 + get_subblock_id()
    heads = wkb.shape[0]
    kvr = q.shape[2]
    for start in dsl.range(0, tokens, capacity):
        chunk = start // capacity
        valid = min(capacity, tokens - start)
        rows = (valid + bm - 1) // bm * bm
        qr_chunk = token_chunk(qr_input, chunk, rows, capacity)
        q_proj_chunk = chunk_rows_2d(ws_q_proj, rows)
        q_chunk = token_chunk(q, chunk, rows, capacity)
        q_fp_chunk = token_chunk(ws_q_fp, 0, rows, capacity)
        sin_chunk = token_chunk(rope_sin, chunk, valid, capacity)
        cos_chunk = token_chunk(rope_cos, chunk, valid, capacity)
        project_q_chunk(
            bm,
            wqb,
            wkb,
            qr_chunk,
            q_proj_chunk,
            q_chunk,
            q_fp_chunk,
            descale_q,
            sin_chunk,
            cos_chunk,
            rope_mode,
            rope_active,
            nz_active,
            False,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
            False,
            True,
            valid,
        )
        if owner < valid * heads:
            QGlobalQuant().quantize_rows(
                q_fp_chunk.view((rows, heads * kvr)),
                q_chunk.view((rows, heads * kvr)),
                descale_q,
                ws_amax,
                heads,
                kvr,
                blocks,
                owner,
                blocks * 2,
                valid * heads,
                owner == 0,
            )
        global_sync_all(flag_ids=(13, 14, 15))
        onchip_rewind(reset_sync_id=True)


@jit
def run_chunk(
    bm,
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    rope_sin,
    rope_cos,
    q,
    qr,
    ws_x_pad,
    ws_qa_raw,
    ws_kva_raw,
    ws_q_proj,
    ws_qb_aux,
    norm_eps,
    qr_scale,
    kvr_scale,
    cache_mode,
    rope_mode,
    rope_active,
    return_qr,
    nz_active,
    nz_wfront,
    nz_wqb,
    head_hint,
    head_width,
    scale_gm,
    descale_q,
    ws_q_fp,
    is_fp8_tensor_quant,
    ws_amax,
    is_hif8_quant,
    input_padding=0,
    front_load_hint=0,
    qb_columns=128,
    qb_load_hint=0,
    wkb_load_hint=0,
):
    block = get_block_idx()
    blocks = get_block_num()
    subblock = get_subblock_id()
    tokens = x.shape[0]
    dim = x.shape[1]
    q_lora = wqa.shape[1] if nz_active else wqa.shape[0]
    kvr = wkva.shape[1] if nz_active else wkva.shape[0]
    _heads = wkb.shape[0]
    nope_dim = wkb.shape[1]
    _head_dim = nope_dim + kvr - 512
    _rows_qb = wqb.shape[1] if nz_active else wqb.shape[0]
    m_tiles = ws_x_pad.shape[0] // bm
    owner = block * 2 + subblock
    if (
        nz_active
        and kv_cache.dtype == dtypes.bfloat16
        and ws_qa_raw.dtype == dtypes.float32
    ):
        front_bm = min(bm, 64)
        front_width = 192 if q_lora == 1536 else 64
        if q_lora == 2048:
            front_width = 128
        if tokens != ws_x_pad.shape[0]:
            InputPad().run(x, ws_x_pad, tokens, owner, blocks * 2)
            global_sync_all(flag_ids=(13, 14, 15))
            onchip_rewind(reset_sync_id=True)
        front = GlobalSplitFront(front_bm, front_width)
        front.project(ws_x_pad, wqa, ws_qa_raw, block, blocks)
        # All four K partials must be visible before any RMSNorm consumer.
        global_sync_all(flag_ids=(13, 14, 15))
        onchip_rewind(reset_sync_id=True)
        direct_front = bm <= FRONT_ACTUAL_ROWS_MAX
        kva_width = (
            32
            if nz_active and 8 < bm <= 64 and kv_cache.dtype == dtypes.bfloat16
            else (32 if nz_active and bm <= 8 else 64)
        )
        kva_projection = NzProjection(
            bm,
            kva_width,
            1,
            nz_active,
            panel_k=NZ_PANEL_K,
            use_actual_rows=direct_front,
        )
        if direct_front:
            kva_projection.project_grid(
                x, wkva, ws_kva_raw, dim, kvr, m_tiles, block, blocks
            )
        else:
            kva_projection.project_grid(
                ws_x_pad, wkva, ws_kva_raw, dim, kvr, m_tiles, block, blocks
            )
        front.normalize(
            ws_qa_raw,
            norm_weight_qa,
            ws_qb_aux,
            qr,
            tokens,
            ws_x_pad.shape[0],
            norm_eps,
            qr_scale,
            return_qr,
            block,
            blocks,
            subblock,
        )
        QaRmsNorm().zero_tail(ws_qb_aux, q_lora, tokens + owner, blocks * 2)
    else:
        partial_front = not nz_active and bm <= 8
        direct_front = (
            nz_active
            and bm <= FRONT_ACTUAL_ROWS_MAX
            and kv_cache.dtype == dtypes.bfloat16
        )
        if not direct_front:
            if x.shape[0] != ws_x_pad.shape[0] or input_padding != 0:
                input_pad = InputPad()
                input_pad.run(x, ws_x_pad, tokens, owner, blocks * 2)
                global_sync_all(flag_ids=(13, 14, 15))
                onchip_rewind(reset_sync_id=True)
        mm1_projection = NzProjection(
            bm,
            nz_wfront,
            1,
            nz_active,
            nz_wfront == 128,
            panel_k=NZ_PANEL_K,
            use_actual_rows=direct_front,
            weight_l2_hint=front_load_hint,
        )
        if partial_front:
            mm1_projection.project_partial_front(
                ws_x_pad, wqa, ws_qa_raw, dim, q_lora, m_tiles, block, blocks
            )
        elif direct_front:
            mm1_projection.project_front(
                x, wqa, ws_qa_raw, dim, q_lora, m_tiles, block, blocks
            )
        else:
            mm1_projection.project_front(
                ws_x_pad, wqa, ws_qa_raw, dim, q_lora, m_tiles, block, blocks
            )
        global_sync_all(flag_ids=(13, 14, 15))
        onchip_rewind(reset_sync_id=True)
        kva_width = (
            32
            if nz_active and 8 < bm <= 64 and kv_cache.dtype == dtypes.bfloat16
            else (32 if nz_active and bm <= 8 else 64)
        )
        kva_projection = NzProjection(
            bm,
            kva_width,
            1,
            nz_active,
            panel_k=NZ_PANEL_K,
            use_actual_rows=direct_front,
        )
        if direct_front:
            kva_projection.project_grid(
                x, wkva, ws_kva_raw, dim, kvr, m_tiles, block, blocks
            )
        else:
            kva_projection.project_grid(
                ws_x_pad, wkva, ws_kva_raw, dim, kvr, m_tiles, block, blocks
            )
        qa_norm = QaRmsNorm(partial_front)
        if owner < tokens:
            qa_norm.process_rows_bf16(
                ws_qa_raw,
                norm_weight_qa,
                ws_qb_aux,
                q_lora,
                norm_eps,
                qr_scale,
                owner,
                blocks * 2,
                tokens,
            )
        qa_norm.zero_tail(ws_qb_aux, q_lora, tokens + owner, blocks * 2)
    _wkb2d_src = wkb.view((wkb.shape[0] * wkb.shape[1], wkb.shape[2]))
    global_sync_all(flag_ids=(13, 14, 15))
    onchip_rewind(reset_sync_id=True)
    kv_chain = KvChain(128, True)
    if subblock == 0:
        if rope_active:
            kv_chain.process_rows_rope(
                ws_kva_raw,
                norm_weight_kva,
                kv_cache,
                cache_index,
                rope_sin,
                rope_cos,
                rope_mode,
                scale_gm,
                kvr,
                norm_eps,
                kvr_scale,
                block,
                blocks,
                tokens,
                cache_mode,
            )
        else:
            kv_chain.process_rows(
                ws_kva_raw,
                norm_weight_kva,
                kv_cache,
                cache_index,
                scale_gm,
                kvr,
                norm_eps,
                kvr_scale,
                block,
                blocks,
                tokens,
                cache_mode,
            )
    project_q_chunk(
        bm,
        wqb,
        wkb,
        ws_qb_aux,
        ws_q_proj,
        q,
        ws_q_fp,
        descale_q,
        rope_sin,
        rope_cos,
        rope_mode,
        rope_active,
        nz_active,
        kv_cache.dtype == dtypes.bfloat16,
        qb_columns,
        qb_load_hint,
        wkb_load_hint,
        is_fp8_tensor_quant,
        is_hif8_quant,
        tokens,
    )


@jit
def project_q_chunk(
    bm,
    wqb,
    wkb,
    qr_input,
    ws_q_proj,
    q,
    ws_q_fp,
    descale_q,
    rope_sin,
    rope_cos,
    rope_mode,
    rope_active,
    nz_active,
    bf16_cache,
    qb_columns,
    qb_load_hint,
    wkb_load_hint,
    is_fp8_tensor_quant,
    is_hif8_quant,
    tokens,
):
    block = get_block_idx()
    blocks = get_block_num()
    owner = block * 2 + get_subblock_id()
    heads = wkb.shape[0]
    nope_dim = wkb.shape[1]
    kvr = q.shape[2]
    head_dim = nope_dim + kvr - 512
    q_lora = qr_input.shape[1]
    rows_qb = heads * head_dim
    m_tiles = ws_q_proj.shape[0] // bm
    wkb2d_src = wkb.view((heads * nope_dim, 512))
    if nz_active:
        if const_expr(bm <= 16 and bf16_cache and (q_lora == 1536 or q_lora == 2048)):
            if heads >= 96:
                if head_dim == 192:
                    qb_grouped_path(
                        bm,
                        192,
                        qr_input,
                        wqb,
                        ws_q_proj,
                        q_lora,
                        rows_qb,
                        m_tiles,
                        block,
                        blocks,
                    )
                else:
                    qb_grouped_path(
                        bm,
                        256,
                        qr_input,
                        wqb,
                        ws_q_proj,
                        q_lora,
                        rows_qb,
                        m_tiles,
                        block,
                        blocks,
                    )
            else:
                qb_runtime_path(
                    bm,
                    qb_columns,
                    qb_load_hint,
                    True,
                    qr_input,
                    wqb,
                    ws_q_proj,
                    q_lora,
                    rows_qb,
                    m_tiles,
                    block,
                    blocks,
                )
        else:
            qb_runtime_path(
                bm,
                qb_columns,
                qb_load_hint,
                bf16_cache,
                qr_input,
                wqb,
                ws_q_proj,
                q_lora,
                rows_qb,
                m_tiles,
                block,
                blocks,
            )
    else:
        qb_nd = NdProjection(bm, 1)
        qb_nd.project_owner(
            qr_input, wqb, ws_q_proj, q_lora, rows_qb, m_tiles, block, blocks
        )
    global_sync_all(flag_ids=(13, 14, 15))
    onchip_rewind(reset_sync_id=True)
    q2d = q.view((q.shape[0], q.shape[1] * q.shape[2]))
    if is_fp8_tensor_quant or is_hif8_quant:
        ws_q_fp2d = ws_q_fp.view(
            (ws_q_fp.shape[0], ws_q_fp.shape[1] * ws_q_fp.shape[2])
        )
        head_stage = QFpTailCopy(True)
        if kvr != 512 and owner < tokens:
            if rope_active:
                head_stage.process_rows_rope(
                    ws_q_proj,
                    ws_q_fp2d,
                    rope_sin,
                    rope_cos,
                    rope_mode,
                    heads,
                    nope_dim,
                    head_dim,
                    kvr,
                    owner,
                    blocks * 2,
                    tokens,
                )
            else:
                head_stage.process_rows(
                    ws_q_proj,
                    ws_q_fp2d,
                    heads,
                    nope_dim,
                    head_dim,
                    kvr,
                    owner,
                    blocks * 2,
                    tokens,
                )
        project_wkb(
            bm,
            nz_active and not is_fp8_tensor_quant,
            wkb_load_hint,
            ws_q_proj,
            wkb2d_src,
            ws_q_fp2d,
            heads,
            nope_dim,
            head_dim,
            kvr,
            m_tiles,
            block,
            blocks,
            tokens,
        )
        global_sync_all(flag_ids=(13, 14, 15))
        onchip_rewind(reset_sync_id=True)
        if is_fp8_tensor_quant:
            descale_q2d = descale_q.view(
                (descale_q.shape[0] * descale_q.shape[1], descale_q.shape[2])
            )
            q_quant = QRowQuant()
            if owner < tokens * heads:
                q_quant.process_rows(
                    ws_q_fp2d,
                    q2d,
                    descale_q2d,
                    heads,
                    kvr,
                    owner,
                    blocks * 2,
                    tokens * heads,
                )
    else:
        tail_copy = QTailCopy(
            True,
            ws_q_proj.dtype,
            head_batch=8 if 8 <= bm <= 64 else ROPE_HEAD_BATCH,
            batch_capable=nz_active
            and (bm >= ROPE_BATCH_MIN_TILE_ROWS or (bm >= 8 and bf16_cache)),
            batch_min_heads=64 if bm < ROPE_BATCH_MIN_TILE_ROWS else 0,
        )
        if kvr != 512:
            if owner < tokens * heads:
                if rope_active:
                    tail_copy.process_rows_rope(
                        ws_q_proj,
                        q2d,
                        rope_sin,
                        rope_cos,
                        rope_mode,
                        heads,
                        nope_dim,
                        head_dim,
                        kvr,
                        owner,
                        blocks * 2,
                        tokens,
                    )
                else:
                    tail_copy.process_rows(
                        ws_q_proj,
                        q2d,
                        heads,
                        nope_dim,
                        head_dim,
                        kvr,
                        owner,
                        blocks * 2,
                        tokens,
                    )
        project_wkb(
            bm,
            nz_active,
            wkb_load_hint,
            ws_q_proj,
            wkb2d_src,
            q2d,
            heads,
            nope_dim,
            head_dim,
            kvr,
            m_tiles,
            block,
            blocks,
            tokens,
        )


@kernel(
    profile=ProfileSpec(
        name="mla_prolog_v4",
        op_type="mla_prolog_v4",
        inputs=(
            "x",
            "wqa",
            "wqb",
            "wkva",
            "wkb",
            "norm_weight_qa",
            "norm_weight_kva",
            "cache_index",
        ),
        outputs=("q",),
        inouts=("kv_cache",),
    )
)
class MlaPrologKernel:
    def __init__(
        self, bm, nz_active=False, nz_wfront=0, nz_wqb=0, head_hint=0, head_width=0
    ):
        self.bm = bm
        self.nz_active = nz_active
        self.nz_wfront = nz_wfront
        self.nz_wqb = nz_wqb
        self.head_hint = head_hint
        self.head_width = head_width

    @surface_signature(ROPE_QR_KERNEL_ARGUMENTS, ROPE_QR_KERNEL_ANNOTATIONS)
    def run_qr_rope(self, *args):
        (
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_qa_fp32,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        ) = args
        run_all_profiles(
            self.bm,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_qa_fp32,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_sin.shape[0] != 0,
            qr.shape[0] != 0,
            self.nz_active,
            self.nz_wfront,
            self.nz_wqb,
            self.head_hint,
            self.head_width,
            None,
            None,
            None,
            False,
            None,
            False,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )

    @surface_signature(M2_ROPE_QR_KERNEL_ARGUMENTS, M2_ROPE_QR_KERNEL_ANNOTATIONS)
    def run_m2_qr_rope(self, *args):
        (
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            quant_scale_kv,
            rope_sin,
            rope_cos,
            q,
            descale_q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_qa_fp32,
            ws_kva_raw,
            ws_q_proj,
            ws_q_fp,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        ) = args
        run_all_profiles(
            self.bm,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_qa_fp32,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_sin.shape[0] != 0,
            qr.shape[0] != 0,
            self.nz_active,
            self.nz_wfront,
            self.nz_wqb,
            self.head_hint,
            self.head_width,
            quant_scale_kv,
            descale_q,
            ws_q_fp,
            True,
            None,
            False,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )

    @surface_signature(M3_ROPE_QR_KERNEL_ARGUMENTS, M3_ROPE_QR_KERNEL_ANNOTATIONS)
    def run_m3_qr_rope(self, *args):
        (
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            quant_scale_kv,
            rope_sin,
            rope_cos,
            q,
            descale_q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_qa_fp32,
            ws_kva_raw,
            ws_q_proj,
            ws_q_fp,
            ws_amax,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        ) = args
        run_all_profiles(
            self.bm,
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            rope_sin,
            rope_cos,
            q,
            qr,
            ws_x_pad,
            ws_qa_raw,
            ws_qa_fp32,
            ws_kva_raw,
            ws_q_proj,
            ws_qb_aux,
            norm_eps,
            qr_scale,
            kvr_scale,
            cache_mode,
            rope_mode,
            rope_sin.shape[0] != 0,
            qr.shape[0] != 0,
            self.nz_active,
            self.nz_wfront,
            self.nz_wqb,
            self.head_hint,
            self.head_width,
            quant_scale_kv,
            descale_q,
            ws_q_fp,
            False,
            ws_amax,
            True,
            front_load_hint,
            qb_columns,
            qb_load_hint,
            wkb_load_hint,
        )


class GlobalSplitFront:
    """Four K partials per output tile, consumed after an all-core barrier."""

    def __init__(self, bm, width):
        self.bm = bm
        self.width = width

    @jit
    def project(self, x, weight, raw, block, blocks):
        activation_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((self.bm, GLOBAL_FRONT_L1_K)),
            depth=GLOBAL_FRONT_BUFFER_DEPTH,
            data_format="nz",
        )
        weight_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((GLOBAL_FRONT_L1_K, self.width)),
            depth=GLOBAL_FRONT_BUFFER_DEPTH,
            data_format="nz",
        )
        activation_l0 = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((self.bm, GLOBAL_FRONT_L0_K)),
            depth=GLOBAL_FRONT_BUFFER_DEPTH,
        )
        weight_l0 = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((self.width, GLOBAL_FRONT_L0_K)),
            depth=1,
        )
        accumulator_l0 = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((self.bm, self.width)),
            depth=GLOBAL_FRONT_BUFFER_DEPTH,
        )

        rows = x.shape[0]
        tiles = weight.shape[1] // self.width
        m_tiles = rows // self.bm
        panels = x.shape[1] // GLOBAL_FRONT_L1_K
        for task in dsl.range(block, m_tiles * tiles * QA_SPLIT_PARTS, blocks):
            part = task % QA_SPLIT_PARTS
            output_tile = task // QA_SPLIT_PARTS
            col = output_tile % tiles
            row = output_tile // tiles
            first = panels * part // QA_SPLIT_PARTS
            last = panels * (part + 1) // QA_SPLIT_PARTS
            acc = accumulator_l0.next_write()
            for panel in dsl.range(first, last):
                mem_copy(
                    activation_l1.next_write(),
                    tile_slice(x, (self.bm, GLOBAL_FRONT_L1_K), (row, panel)),
                    options=copy_options(l2_cache_ctl=1),
                )
                mem_copy(
                    weight_l1.next_write(),
                    tile_slice(weight, (GLOBAL_FRONT_L1_K, self.width), (panel, col)),
                )
                a = activation_l1.next_read()
                b = weight_l1.next_read().permute(1, 0)
                for inner in range_constexpr(GLOBAL_FRONT_L0_PANELS):
                    mem_copy(
                        activation_l0.next_write(),
                        tile_slice(a, (self.bm, GLOBAL_FRONT_L0_K), (0, inner)),
                    )
                    mem_copy(
                        weight_l0.next_write(),
                        tile_slice(b, (self.width, GLOBAL_FRONT_L0_K), (0, inner)),
                    )
                    matmul(
                        acc,
                        activation_l0.next_read(),
                        weight_l0.next_read(),
                        init=panel == first and inner == 0,
                    )
            mem_copy(
                tile_slice(raw, (self.bm, self.width), (part * m_tiles + row, col)),
                accumulator_l0.next_read(),
            )

    @jit
    def normalize(
        self,
        raw,
        gamma,
        cq,
        qr,
        tokens,
        rows,
        epsilon,
        scale,
        return_qr,
        block,
        blocks,
        subblock,
    ):
        columns = cq.shape[1]
        row_tiler = make_tiler((1, columns), alignment=(1, VECTOR_SEG))
        left_channel = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, columns)), depth=1
        )
        right_channel = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, columns)), depth=1
        )
        third_channel = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, columns)), depth=1
        )
        fourth_channel = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, columns)), depth=1
        )
        gamma_channel = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, columns)), depth=1
        )
        output_channel = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, columns)), depth=1
        )
        scalar = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 16)))
        mem_copy(gamma_channel.next_write(), tile_slice(gamma, row_tiler, (0, 0)))
        gamma_read = gamma_channel.next_read()
        for token in dsl.range(block * 2 + subblock, tokens, blocks * 2):
            mem_copy(left_channel.next_write(), tile_slice(raw, row_tiler, (token, 0)))
            mem_copy(
                right_channel.next_write(),
                tile_slice(raw, row_tiler, (token + rows, 0)),
            )
            mem_copy(
                third_channel.next_write(),
                tile_slice(raw, row_tiler, (token + 2 * rows, 0)),
            )
            mem_copy(
                fourth_channel.next_write(),
                tile_slice(raw, row_tiler, (token + 3 * rows, 0)),
            )
            third = third_channel.next_read()
            fourth = fourth_channel.next_read()
            left = left_channel.next_read()
            right = right_channel.next_read()
            with vf(mode="simd"):
                one = update_mask(1, elem_bits=32)[0]
                sum_square = vdups(0.0, dtypes.float32, mask=one)
                for offset in range_constexpr(0, columns, VECTOR_SEG):
                    first_pair = vadd(
                        vload(left, offset), vload(right, offset), mask=full_mask()
                    )
                    second_pair = vadd(
                        vload(third, offset), vload(fourth, offset), mask=full_mask()
                    )
                    value = vadd(first_pair, second_pair, mask=full_mask())
                    value = vcast(
                        value,
                        dtypes.bfloat16,
                        mask=full_mask(),
                        rounding=RoundingMode.RN,
                    )
                    value = vcast(value, dtypes.float32, mask=full_mask())
                    vstore(left, offset, value, full_mask())
                    sum_square = vadd(
                        sum_square,
                        vreduce_sum(
                            vmul(value, value, mask=full_mask()), mask=full_mask()
                        ),
                        mask=one,
                    )
                root = vsqrt(
                    vadds(
                        vmuls(sum_square, 1.0 / dtypes.float32(columns), mask=one),
                        epsilon,
                        mask=one,
                    ),
                    mask=one,
                )
                vstore_first(scalar, 0, root)
                vmem_bar(mode="vst_vld")
            write = output_channel.next_write()
            with vf(mode="simd"):
                divisor = vload_broadcast(scalar, 0)
                for offset in range_constexpr(0, columns, VECTOR_SEG):
                    normalized = vdiv(
                        vload(left, offset),
                        divisor,
                        mask=full_mask(),
                        precision_mode=True,
                    )
                    normalized = vmul(
                        normalized, vload(gamma_read, offset), mask=full_mask()
                    )
                    normalized = vmuls(normalized, scale, mask=full_mask())
                    packed = vcast(
                        normalized,
                        dtypes.bfloat16,
                        mask=full_mask(),
                        rounding=RoundingMode.RN,
                    )
                    vstore_pack(
                        write,
                        offset,
                        packed,
                        full_mask(),
                        pack_mode=PackMode.B32_TO_B16,
                    )
                vmem_bar(mode="vst_vld")
            output = output_channel.next_read()
            mem_copy(tile_slice(cq, row_tiler, (token, 0)), output)


# Host validation, scheduling and scratch planning.


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _InputTensors:
    x: torch.Tensor
    wqa: torch.Tensor
    wqb: torch.Tensor
    wkva: torch.Tensor
    wkb: torch.Tensor
    norm_weight_qa: torch.Tensor
    norm_weight_kva: torch.Tensor
    kv_cache: torch.Tensor
    cache_index: torch.Tensor
    quant_scale_kv: Optional[torch.Tensor]
    rope_sin: Optional[torch.Tensor]
    rope_cos: Optional[torch.Tensor]


@dataclass(frozen=True, slots=True)
class _Attributes:
    norm_eps: float
    qkv_quant_mode: int
    cache_mode: "CacheMode"
    qr_scale: float
    kvr_scale: float
    rope_mode: int
    return_qr: bool


@dataclass(frozen=True, slots=True)
class _Metadata:
    tokens: int
    dim: int
    q_lora: int
    rows_qb: int
    kvr: int
    heads: int
    nope_dim: int
    rope_dim: int
    weight_format: str
    qkv_quant_mode: int
    rope_active: bool
    return_qr: bool


@dataclass(frozen=True, slots=True)
class _RuntimeControls:
    front_load_hint: int = 0
    qb_columns: int = 0
    qb_load_hint: int = 0
    wkb_load_hint: int = 0


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _Outputs:
    q: torch.Tensor
    descale_q: torch.Tensor
    qr: torch.Tensor

    def as_tuple(self):
        return self.q, self.descale_q, self.qr


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _ScratchBuffers:
    ws_qa_raw: torch.Tensor
    ws_kva_raw: torch.Tensor
    ws_q_proj: torch.Tensor
    ws_qb_aux: torch.Tensor
    ws_x_pad: Optional[torch.Tensor] = None
    ws_qa_fp32: Optional[torch.Tensor] = None
    ws_q_fp: Optional[torch.Tensor] = None
    ws_amax: Optional[torch.Tensor] = None


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _QaWorkspace:
    bf16: torch.Tensor
    fp32: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CompileDimensions:
    tokens: object
    rows: object
    scratch_rows: object
    bm: int
    dim: object
    q_lora: object
    kvr: object
    heads: object
    nope: object
    rows_qb: object
    blocks: object
    block_size: object
    q_lora_padded: object
    kvr_padded: object
    rows_qb_padded: object


def record_values(*records):
    return {
        field.name: getattr(record, field.name)
        for record in records
        for field in fields(record)
    }


if hasattr(torch, "library"):
    import torch_npu  # noqa: F401


def scratch_token_rows(rows, meta):
    if rows <= _SCRATCH_TOKEN_CAPACITY:
        return rows
    row_bytes = (
        2 * max(ceil_align(meta.q_lora, BN), ceil_align(meta.rows_qb, BN))
        + 2 * ceil_align(meta.kvr, BN)
        + 2 * meta.q_lora
        + 2 * meta.dim
    )
    if meta.qkv_quant_mode in (2, 3):
        row_bytes += 4 * meta.heads * meta.kvr
    capacity = _SCRATCH_TOKEN_CAPACITY
    while (
        capacity > _SCRATCH_MIN_CHUNK_ROWS
        and capacity * row_bytes > _SCRATCH_TARGET_BYTES
    ):
        capacity //= 2
    return min(rows, capacity)


def recompute_global_q(meta, rows):
    return (
        meta.qkv_quant_mode == 3
        and rows > scratch_token_rows(rows, meta)
        and rows * meta.heads * meta.kvr * ITEM_F32 > _SCRATCH_TARGET_BYTES
        and (
            meta.return_qr or meta.q_lora * ITEM_A16 < meta.heads * meta.kvr * ITEM_F32
        )
    )


@dataclass(frozen=True, slots=True)
class CompileKey:
    fmt: str
    qkv_quant_mode: int
    profile: int
    shapes: tuple = ()


def select_qb_columns(rows_qb, heads, cores):
    if type(cores) is not int or cores <= 0:
        raise ValueError("cores must be a positive integer")
    current = nz_qb_tile_width(rows_qb, heads, 0)
    if current <= CUBE_BK or rows_qb % CUBE_BK:
        return 0
    current_tasks = rows_qb // current
    narrow_tasks = rows_qb // CUBE_BK
    current_peak = (current_tasks + cores - 1) // cores * current
    narrow_peak = (narrow_tasks + cores - 1) // cores * CUBE_BK
    return CUBE_BK if narrow_peak < current_peak else 0


def token_profile(tokens):
    for capacity, (lower, upper) in TOKEN_PROFILE_BOUNDS.items():
        if lower <= tokens <= upper:
            return capacity
    return GENERIC_PROFILE


def validate_resource_budget(bm):
    front_l1 = bm * BK * ITEM_A16 * 2 + BN * BK * ITEM_A16 * 2
    qb_columns = QB_COLUMNS
    qb_l1 = bm * BK * ITEM_A16 * 2 + qb_columns * BK * ITEM_A16 * 2
    wkb_l1 = bm * BKA * ITEM_A16 * 2 + BKA * 128 * ITEM_A16 * 2
    if not (front_l1 <= L1_BYTES):
        raise ValueError(
            f"front projection L1 budget exceeded: front_l1={front_l1} bytes exceeds L1_BYTES={L1_BYTES} bytes (tile_rows={bm})"
        )
    if not (qb_l1 <= L1_BYTES):
        raise ValueError(
            f"QB projection L1 budget exceeded: qb_l1={qb_l1} bytes exceeds L1_BYTES={L1_BYTES} bytes (tile_rows={bm})"
        )
    if not (wkb_l1 <= L1_BYTES):
        raise ValueError(
            f"wkb projection L1 budget exceeded: wkb_l1={wkb_l1} bytes exceeds L1_BYTES={L1_BYTES} bytes (tile_rows={bm})"
        )
    front_l0a = bm * BK * ITEM_A16 * 2
    front_l0b = BK * BN * ITEM_A16 * 2
    front_l0c = bm * BN * ITEM_F32 * 2
    qb_l0a = bm * BK * ITEM_A16
    qb_l0b = BK * qb_columns * ITEM_A16
    qb_l0c = bm * qb_columns * ITEM_F32 * 2
    wkb_l0a = bm * BKA * ITEM_A16
    wkb_l0b = BKA * 128 * ITEM_A16
    wkb_l0c = bm * 128 * ITEM_F32 * 2
    if not (front_l0a <= L0A_BYTES):
        raise ValueError(
            f"L0A budget exceeded: front_l0a={front_l0a} bytes exceeds L0A_BYTES={L0A_BYTES} bytes (tile_rows={bm})"
        )
    if not (front_l0b <= L0B_BYTES):
        raise ValueError(
            f"L0B budget exceeded: front_l0b={front_l0b} bytes exceeds L0B_BYTES={L0B_BYTES} bytes (tile_rows={bm})"
        )
    if not (front_l0c <= L0C_BYTES):
        raise ValueError(
            f"L0C budget exceeded: front_l0c={front_l0c} bytes exceeds L0C_BYTES={L0C_BYTES} bytes (tile_rows={bm})"
        )
    if not (qb_l0a <= L0A_BYTES):
        raise ValueError(
            f"QB L0A budget exceeded: qb_l0a={qb_l0a} bytes exceeds L0A_BYTES={L0A_BYTES} bytes (tile_rows={bm})"
        )
    if not (qb_l0b <= L0B_BYTES):
        raise ValueError(
            f"QB L0B budget exceeded: qb_l0b={qb_l0b} bytes exceeds L0B_BYTES={L0B_BYTES} bytes (tile_rows={bm})"
        )
    if not (qb_l0c <= L0C_BYTES):
        raise ValueError(
            f"QB L0C budget exceeded: qb_l0c={qb_l0c} bytes exceeds L0C_BYTES={L0C_BYTES} bytes (tile_rows={bm})"
        )
    if not (wkb_l0a <= L0A_BYTES):
        raise ValueError(
            f"wkb L0A budget exceeded: wkb_l0a={wkb_l0a} bytes exceeds L0A_BYTES={L0A_BYTES} bytes (tile_rows={bm})"
        )
    if not (wkb_l0b <= L0B_BYTES):
        raise ValueError(
            f"wkb L0B budget exceeded: wkb_l0b={wkb_l0b} bytes exceeds L0B_BYTES={L0B_BYTES} bytes (tile_rows={bm})"
        )
    if not (wkb_l0c <= L0C_BYTES):
        raise ValueError(
            f"wkb L0C budget exceeded: wkb_l0c={wkb_l0c} bytes exceeds L0C_BYTES={L0C_BYTES} bytes (tile_rows={bm})"
        )
    if bm <= 8:
        # Runtime branches may retain both channel sets; NZ rows align to 16.
        wkb_rows = max(bm, 16)
        adaptive_l1 = (
            2 * wkb_rows * BKW * ITEM_A16 * 2
            + BKW * (4 * CUBE_BK + CUBE_BK) * ITEM_A16 * 2
        )
        adaptive_l0a = 2 * wkb_rows * BKW * ITEM_A16
        adaptive_l0b = 2 * BKW * CUBE_BK * ITEM_A16
        adaptive_l0c = wkb_rows * (4 * CUBE_BK + CUBE_BK) * ITEM_F32 * 2
        if not (adaptive_l1 <= L1_BYTES):
            raise ValueError(
                f"adaptive WKB L1 budget exceeded: adaptive_l1={adaptive_l1} bytes exceeds L1_BYTES={L1_BYTES} bytes (tile_rows={bm})"
            )
        if not (adaptive_l0a <= L0A_BYTES):
            raise ValueError(
                f"adaptive WKB L0A budget exceeded: adaptive_l0a={adaptive_l0a} bytes exceeds L0A_BYTES={L0A_BYTES} bytes (tile_rows={bm})"
            )
        if not (adaptive_l0b <= L0B_BYTES):
            raise ValueError(
                f"adaptive WKB L0B budget exceeded: adaptive_l0b={adaptive_l0b} bytes exceeds L0B_BYTES={L0B_BYTES} bytes (tile_rows={bm})"
            )
        if not (adaptive_l0c <= L0C_BYTES):
            raise ValueError(
                f"adaptive WKB L0C budget exceeded: adaptive_l0c={adaptive_l0c} bytes exceeds L0C_BYTES={L0C_BYTES} bytes (tile_rows={bm})"
            )
    qa_ub = (
        Q_LORA_MAX * ITEM_F32 * 2
        + Q_LORA_MAX * ITEM_F32
        + Q_LORA_MAX * ITEM_F32 * 2
        + Q_LORA_MAX * ITEM_A16 * 2
    )
    rope_ub = SEG * ITEM_A16 * 2 + 2 * SEG * ITEM_F32 * 2 + 2 * (2 * SEG) * ITEM_F32
    kv_ub = (
        KVR_MAX * ITEM_F32 * 2
        + 512 * ITEM_F32
        + KVR_MAX * ITEM_A16 * 2
        + (512 + 2 * 64 + 16) * 1 * 2
        + KVR_MAX * 1 * 2
        + KVR_MAX * 1 * 2
        + 64 * ITEM_F32
        + rope_ub
    )
    tail_ub = 2 * (SEG * ITEM_F32 * 2) + 2 * (SEG * ITEM_A16 * 2) + rope_ub
    fp8_quant_ub_bytes = (
        256 * ITEM_F32 * 2
        + 256 * ITEM_A16 * 2
        + SEG * ITEM_F32 * 2
        + rope_ub
        + KVR_MAX * ITEM_F32 * 2
        + KVR_MAX * 1 * 2
        + 64 * ITEM_F32
    )
    hif8_quant_ub_bytes = (
        KVR_MAX * ITEM_F32 * 2
        + 64 * ITEM_F32
        + KVR_MAX * ITEM_F32 * 2
        + 64 * ITEM_F32
        + KVR_MAX * 1 * 2
        + 64 * ITEM_F32
    )
    if not (qa_ub <= UB_BYTES):
        raise ValueError(
            f"QA norm UB budget exceeded: qa_ub={qa_ub} bytes exceeds UB_BYTES={UB_BYTES} bytes (tile_rows={bm})"
        )
    if not (kv_ub <= UB_BYTES):
        raise ValueError(
            f"KV chain UB budget exceeded: kv_ub={kv_ub} bytes exceeds UB_BYTES={UB_BYTES} bytes (tile_rows={bm})"
        )
    if not (tail_ub <= UB_BYTES):
        raise ValueError(
            f"q tail UB budget exceeded: tail_ub={tail_ub} bytes exceeds UB_BYTES={UB_BYTES} bytes (tile_rows={bm})"
        )
    if not (fp8_quant_ub_bytes <= UB_BYTES):
        raise ValueError(
            f"mode-2 quantization UB budget exceeded: fp8_quant_ub_bytes={fp8_quant_ub_bytes} bytes exceeds UB_BYTES={UB_BYTES} bytes (tile_rows={bm})"
        )
    if not (hif8_quant_ub_bytes <= UB_BYTES):
        raise ValueError(
            f"mode-3 global amax UB budget exceeded: hif8_quant_ub_bytes={hif8_quant_ub_bytes} bytes exceeds UB_BYTES={UB_BYTES} bytes (tile_rows={bm})"
        )
    return True


def validate_resource_matrix():
    for profile in TOKEN_PROFILES:
        bm = 128 if profile == GENERIC_PROFILE else profile
        validate_resource_budget(bm)
        a_slab = bm * Q_LORA_MAX * ITEM_A16
        if not (a_slab <= L2_BYTES):
            raise ValueError(
                f"QB A slab exceeds L2 residency: a_slab={a_slab} bytes exceeds L2_BYTES={L2_BYTES} bytes (tile_rows={bm})"
            )
        for q_lora in Q_LORA_VALUES:
            for rope_dim, kvr in ((64, 576), (0, 512)):
                front_width = nz_front_tile_width(q_lora, kvr)
                for dn in (128, 192) if rope_dim else (256,):
                    for heads in (1, HEADS_DOMAIN_MAX):
                        rows_qb = heads * (dn + rope_dim)
                        wqb_width = nz_tile_width(rows_qb)
                        validate_nz_resource_budget(bm, (front_width, wqb_width))
    return True


def validate_nz_resource_budget(bm, widths, panel_k=BK):
    for width in widths:
        l1 = bm * panel_k * ITEM_A16 * 2 + width * panel_k * ITEM_A16 * 2
        if not (l1 <= L1_BYTES):
            raise ValueError(
                f"NZ projection L1 budget exceeded: l1={l1} bytes exceeds L1_BYTES={L1_BYTES} bytes (tile_rows={bm})"
            )
        l0a = bm * BK * ITEM_A16 * 2
        l0b = BK * width * ITEM_A16 * (1 if width > 128 else 2)
        l0c = bm * width * ITEM_F32 * 2
        if not (l0a <= L0A_BYTES):
            raise ValueError(
                f"NZ projection L0A budget exceeded: l0a={l0a} bytes exceeds L0A_BYTES={L0A_BYTES} bytes (tile_rows={bm})"
            )
        if not (l0b <= L0B_BYTES):
            raise ValueError(
                f"NZ projection L0B budget exceeded: l0b={l0b} bytes exceeds L0B_BYTES={L0B_BYTES} bytes (tile_rows={bm})"
            )
        if not (l0c <= L0C_BYTES):
            raise ValueError(
                f"NZ projection L0C budget exceeded: l0c={l0c} bytes exceeds L0C_BYTES={L0C_BYTES} bytes (tile_rows={bm})"
            )
    return True


class CacheMode(Enum):
    PA_BSND = 0
    PA_NZ = 1


class QkvQuantMode(Enum):
    A16C16 = 0
    A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE = 1
    A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR = 2
    A8C8_QKV_HIFLOAT8_PER_TENSOR = 3


class RopeMode(Enum):
    NO_ROPE = 0
    ROPE_INTERLEAVE_HALF = 1
    ROPE_HALF = 2


_QKV_QUANT_MODE_VALUES = tuple(_m.value for _m in QkvQuantMode)
_CACHE_MODE_VALUES = tuple(_m.value for _m in CacheMode)
_ROPE_MODE_VALUES = tuple(_m.value for _m in RopeMode)


def validate_mode_enums(qkv_quant_mode, cache_mode, rope_mode):
    for name, value, enum_type in (
        ("qkv_quant_mode", qkv_quant_mode, QkvQuantMode),
        ("cache_mode", cache_mode, CacheMode),
        ("rope_mode", rope_mode, RopeMode),
    ):
        if type(value) is not enum_type:
            raise TypeError(f"{name} must be a {enum_type.__name__} enum")


def validate_scalar_types(norm_eps, qr_scale, kvr_scale):
    for name, value in (
        ("norm_eps", norm_eps),
        ("qr_scale", qr_scale),
        ("kvr_scale", kvr_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite float")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_rank(name, value, rank, shape_description):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a rank-{rank} Tensor {shape_description}")
    actual_rank = value.ndim
    if actual_rank != rank:
        raise ValueError(
            f"{name} must have rank {rank} {shape_description}, "
            f"got rank {actual_rank} with shape {tuple(value.shape)}"
        )


def describe_npu_format(value):
    code = int(value)
    name = {0: "NCHW", 2: "ND", 29: "FRACTAL_NZ"}.get(code)
    return f"{name} ({code})" if name is not None else f"NPU format {code}"


def validate_tensor(name, value, shape, dtype, device, *, nz=False, cache_axis0=False):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    require(
        tuple(value.shape) == tuple(shape),
        f"{name} shape must be {tuple(shape)}, got {tuple(value.shape)}",
    )
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype must be {dtype}, got {value.dtype}")
    require(value.device == device, f"{name} must be on {device}")
    if cache_axis0:
        width = shape[-1]
        strides = value.stride()
        expected_inner_strides = (width, width, 1)
        if tuple(strides[1:]) != expected_inner_strides:
            raise ValueError(
                f"{name} inner strides must be {expected_inner_strides} elements, "
                f"got {tuple(strides[1:])}"
            )
        minimum_stride = shape[1] * width
        if strides[0] < minimum_stride:
            raise ValueError(
                f"{name}.stride(0) is {strides[0]} elements, "
                f"expected at least {minimum_stride} for shape {tuple(shape)}"
            )
    else:
        require(value.is_contiguous(), f"{name} must be contiguous")
    fmt = torch_npu.get_npu_format(value)
    require(
        fmt == 29 if nz else fmt in (0, 2),
        f"{name} Format must be {('true NZ' if nz else 'ND')}",
    )


def validate(inputs, attributes):
    x = inputs.x
    wqa = inputs.wqa
    wqb = inputs.wqb
    wkva = inputs.wkva
    wkb = inputs.wkb
    norm_weight_qa = inputs.norm_weight_qa
    norm_weight_kva = inputs.norm_weight_kva
    kv_cache = inputs.kv_cache
    cache_index = inputs.cache_index
    quant_scale_kv = inputs.quant_scale_kv
    rope_sin = inputs.rope_sin
    rope_cos = inputs.rope_cos
    norm_eps = attributes.norm_eps
    qkv_quant_mode = attributes.qkv_quant_mode
    cache_mode = attributes.cache_mode
    qr_scale = attributes.qr_scale
    kvr_scale = attributes.kvr_scale
    rope_mode = attributes.rope_mode
    return_qr = attributes.return_qr
    for name, value, domain in (
        ("qkv_quant_mode", qkv_quant_mode, (0, 1, 2, 3)),
        ("rope_mode", rope_mode, (0, 1, 2)),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be int, not bool or float")
        require(value in domain, f"{name} must be in {domain}")
    if type(cache_mode) is not CacheMode:
        raise TypeError("cache_mode must be a CacheMode enum, not integer or string")
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")
    validate_scalar_types(norm_eps, qr_scale, kvr_scale)
    for name, value in (
        ("norm_eps", norm_eps),
        ("qr_scale", qr_scale),
        ("kvr_scale", kvr_scale),
    ):
        require(math.isfinite(float(value)), f"{name} must be finite")
        try:
            fp32_value = struct.unpack("f", struct.pack("f", float(value)))[0]
        except OverflowError as error:
            raise ValueError(f"{name} must be finite in FP32") from error
        require(math.isfinite(fp32_value), f"{name} must be finite in FP32")
        if name == "norm_eps":
            require(fp32_value > 0, "norm_eps must be positive in FP32")
    require(
        not (qkv_quant_mode == 1 and cache_mode == CacheMode.PA_NZ),
        "mode 1 only supports PA_BSND; PA_NZ is forbidden",
    )
    require_rank("x", x, 2, "(T, dim)")
    require(x.device.type == "npu", "x must be on NPU")
    (tokens, dim) = x.shape
    require(
        1024 <= dim <= 8192 and dim % 256 == 0,
        "dim must be [1024,8192] and 256-aligned",
    )
    require_rank("norm_weight_qa", norm_weight_qa, 1, "(q_lora,)")
    q_lora = norm_weight_qa.shape[0]
    require(q_lora in (640, 1024, 1536, 2048), "q_lora must be 640/1024/1536/2048")
    require_rank("wkb", wkb, 3, "(N, Dn, 512)")
    (heads, nope_dim, kv_lora) = wkb.shape
    require(1 <= heads <= 128, "N must be [1,128]")
    require(kv_lora == 512, "kv_lora must be 512")
    for name, weight in (("wqa", wqa), ("wqb", wqb), ("wkva", wkva)):
        require_rank(name, weight, 2, "ND (O, K) or NZ (K, O)")
        if weight.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {weight.device}")
    formats = tuple((torch_npu.get_npu_format(weight) for weight in (wqa, wqb, wkva)))
    if formats not in ((2, 2, 2), (29, 29, 29)):
        actual_formats = ", ".join(
            f"{name}={describe_npu_format(fmt)}"
            for name, fmt in zip(("wqa", "wqb", "wkva"), formats)
        )
        raise ValueError(
            "wqa/wqb/wkva must all use ND (2) or all use FRACTAL_NZ (29), "
            f"got {actual_formats}"
        )
    nz = formats[0] == 29
    rope_dim = wkva.shape[1 if nz else 0] - 512
    if (nope_dim, rope_dim) not in ((128, 64), (192, 64), (256, 0)):
        raise ValueError(
            f"unsupported (Dn, Dr)={(nope_dim, rope_dim)}, "
            "expected (128, 64), (192, 64), or (256, 0)"
        )
    require(rope_dim != 0 or rope_mode == 0, "Dr=0 requires rope_mode=0")
    device = x.device
    validate_tensor("x", x, (tokens, dim), torch.bfloat16, device)
    for name, weight, shape in (
        ("wqa", wqa, (q_lora, dim)),
        ("wqb", wqb, (heads * (nope_dim + rope_dim), q_lora)),
        ("wkva", wkva, (512 + rope_dim, dim)),
    ):
        validate_tensor(
            name, weight, shape[::-1] if nz else shape, torch.bfloat16, device, nz=nz
        )
    validate_tensor("wkb", wkb, (heads, nope_dim, 512), torch.bfloat16, device)
    validate_tensor("norm_weight_qa", norm_weight_qa, (q_lora,), torch.float32, device)
    validate_tensor("norm_weight_kva", norm_weight_kva, (512,), torch.float32, device)
    require(
        isinstance(kv_cache, torch.Tensor) and kv_cache.ndim == 4,
        "kv_cache shape must be rank 4",
    )
    (blocks, block_size, cache_heads, _) = kv_cache.shape
    require(blocks > 0, "block_number must be positive")
    require(
        16 <= block_size <= 1024 and block_size % 16 == 0,
        "kv_cache block_size must be [16,1024] and divisible by 16",
    )
    require(cache_heads == 1, "kv_cache Nkv must be 1")
    expected_width = 512 + 2 * rope_dim + 16 if qkv_quant_mode == 1 else 512 + rope_dim
    cache_dtype = (
        torch.bfloat16
        if qkv_quant_mode == 0
        else torch.uint8
        if qkv_quant_mode == 3
        else torch.float8_e4m3fn
    )
    validate_tensor(
        "kv_cache",
        kv_cache,
        (blocks, block_size, 1, expected_width),
        cache_dtype,
        device,
        cache_axis0=True,
    )
    validate_tensor("cache_index", cache_index, (tokens,), torch.int64, device)
    if qkv_quant_mode in (2, 3):
        if quant_scale_kv is None:
            raise TypeError(
                f"qkv_quant_mode={qkv_quant_mode} requires quant_scale_kv as an FP32 "
                "Tensor of shape (1,) on the same device as x, got None"
            )
        validate_tensor("quant_scale_kv", quant_scale_kv, (1,), torch.float32, device)
    else:
        require(quant_scale_kv is None, "quant_scale_kv must be None in mode 0/1")
    if rope_dim and rope_mode:
        validate_tensor(
            "rope_sin", rope_sin, (tokens, rope_dim), torch.bfloat16, device
        )
        validate_tensor(
            "rope_cos", rope_cos, (tokens, rope_dim), torch.bfloat16, device
        )
    else:
        require(
            (rope_sin is None) == (rope_cos is None), "rope_sin/rope_cos must be paired"
        )
        if rope_sin is not None:
            validate_tensor(
                "rope_sin", rope_sin, (tokens, rope_dim), torch.bfloat16, device
            )
            validate_tensor(
                "rope_cos", rope_cos, (tokens, rope_dim), torch.bfloat16, device
            )
    return _Metadata(
        tokens=int(tokens),
        dim=int(dim),
        q_lora=int(q_lora),
        rows_qb=int(heads * (nope_dim + rope_dim)),
        kvr=int(512 + rope_dim),
        heads=int(heads),
        nope_dim=int(nope_dim),
        rope_dim=int(rope_dim),
        weight_format="nz" if nz else "nd",
        qkv_quant_mode=int(qkv_quant_mode),
        rope_active=bool(rope_dim and rope_mode),
        return_qr=return_qr,
    )


def ceil_align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def get_core_num(x, meta):
    info = dsl.get_platform_info(stream=torch.npu.current_stream(x.device))
    require(
        info.available and info.cube_core_num > 0 and (info.vector_core_num > 1),
        "current stream has no valid MIX-core capacity",
    )
    return min(int(info.cube_core_num), int(info.vector_core_num) // 2)


def output_specs(meta, q_rows=None):
    q_dtype = (
        torch.bfloat16
        if meta.qkv_quant_mode < 2
        else torch.float8_e4m3fn
        if meta.qkv_quant_mode == 2
        else torch.uint8
    )
    descale_shape = (
        (0,)
        if meta.qkv_quant_mode < 2 or (meta.tokens == 0 and meta.qkv_quant_mode == 3)
        else (meta.tokens, meta.heads, 1)
        if meta.qkv_quant_mode == 2
        else (1,)
    )
    return {
        "q": (
            (meta.tokens if q_rows is None else q_rows, meta.heads, meta.kvr),
            q_dtype,
        ),
        "descale_q": (descale_shape, torch.float32),
        "qr": (
            ((meta.tokens if q_rows is None else q_rows), meta.q_lora)
            if meta.return_qr
            else (0,),
            torch.bfloat16,
        ),
    }


def empty_outputs(meta, device, q_rows=None):
    return _Outputs(
        **{
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in output_specs(meta, q_rows).items()
        }
    )


@dataclass(frozen=True, slots=True)
class _ScratchSpec:
    name: str
    shape: tuple
    dtype: object


@dataclass(frozen=True, slots=True)
class _ScratchView:
    name: str
    shape: tuple
    dtype: object
    stride: tuple
    offset_elements: int


@dataclass(frozen=True, slots=True)
class _ScratchLayout:
    allocation_bytes: int
    views: tuple


@dataclass(frozen=True, slots=True)
class _ScratchPlan:
    rows: int
    specs: tuple


@lru_cache(maxsize=256)
def scratch_layout(specs, reuse_groups=()):
    groups = {buffer_name: group for group in reuse_groups for buffer_name in group}
    buffer_bytes = {
        spec.name: math.prod(spec.shape) * spec.dtype.itemsize for spec in specs
    }
    cursor_bytes = 0
    group_offsets_bytes = {}
    layout = []
    for spec in specs:
        group = groups.get(spec.name, (spec.name,))
        if group not in group_offsets_bytes:
            cursor_bytes = ceil_align(cursor_bytes, _SCRATCH_ALIGNMENT_BYTES)
            group_offsets_bytes[group] = cursor_bytes
            cursor_bytes += max(buffer_bytes[buffer_name] for buffer_name in group)
        stride = []
        stride_elements = 1
        for extent in reversed(spec.shape):
            stride.append(stride_elements)
            stride_elements *= extent
        layout.append(
            _ScratchView(
                name=spec.name,
                shape=spec.shape,
                dtype=spec.dtype,
                stride=tuple(reversed(stride)),
                offset_elements=group_offsets_bytes[group] // spec.dtype.itemsize,
            )
        )
    return _ScratchLayout(
        allocation_bytes=ceil_align(cursor_bytes, _SCRATCH_ALIGNMENT_BYTES),
        views=tuple(layout),
    )


def allocate_scratch(specs, device, reuse_groups=()):
    layout = scratch_layout(specs, reuse_groups)
    storage = torch.empty((layout.allocation_bytes,), dtype=torch.uint8, device=device)
    typed = {
        dtype: storage.view(dtype)
        for dtype in dict.fromkeys(spec.dtype for spec in specs)
    }
    return _ScratchBuffers(
        **{
            view.name: typed[view.dtype].as_strided(
                view.shape,
                view.stride,
                typed[view.dtype].storage_offset() + view.offset_elements,
            )
            for view in layout.views
        }
    )


def scratch_reuse_groups(meta):
    front_group = ("ws_qa_raw", "ws_q_proj")
    groups = (front_group,)
    # KVA cache consumption ends at the WQB join; mode2 Q is produced afterwards.
    profile = token_profile(meta.tokens)
    rows = ceil_align(meta.tokens, 128) if profile == GENERIC_PROFILE else profile
    if meta.qkv_quant_mode == 2 or recompute_global_q(meta, rows):
        groups += (("ws_kva_raw", "ws_q_fp"),)
    return groups


def execution_scratch_specs(meta, core_num):
    profile = token_profile(meta.tokens)
    rows = ceil_align(meta.tokens, 128) if profile == GENERIC_PROFILE else profile
    scratch_rows = scratch_token_rows(rows, meta)
    q_lora_padded = ceil_align(meta.q_lora, BN)
    kvr_padded = ceil_align(meta.kvr, BN)
    rows_qb_padded = ceil_align(meta.rows_qb, BN)
    is_tensor_quantized = meta.qkv_quant_mode in (2, 3)
    partial_front = meta.weight_format == "nd" and 0 < profile <= 8
    global_front = (
        meta.weight_format == "nz"
        and meta.qkv_quant_mode == 0
        and 8 < meta.tokens <= 64
        and meta.q_lora == 1536
    )
    qa_shape = (
        scratch_rows * (QA_SPLIT_PARTS if global_front or partial_front else 1),
        q_lora_padded,
    )
    qa_dtype = torch.float32 if global_front or partial_front else torch.bfloat16
    specs = (
        _ScratchSpec("ws_qa_raw", qa_shape, qa_dtype),
        _ScratchSpec("ws_kva_raw", (scratch_rows, kvr_padded), torch.bfloat16),
        _ScratchSpec("ws_q_proj", (scratch_rows, rows_qb_padded), torch.bfloat16),
        _ScratchSpec(
            "ws_qb_aux",
            (
                (
                    0
                    if meta.return_qr
                    else rows
                    if recompute_global_q(meta, rows)
                    else scratch_rows
                ),
                meta.q_lora,
            ),
            torch.bfloat16,
        ),
    )
    if is_tensor_quantized:
        q_fp_rows = (
            rows
            if meta.qkv_quant_mode == 3 and not recompute_global_q(meta, rows)
            else scratch_rows
        )
        specs += (
            _ScratchSpec("ws_q_fp", (q_fp_rows, meta.heads, meta.kvr), torch.float32),
        )
    if meta.qkv_quant_mode == 3:
        specs += (
            _ScratchSpec(
                "ws_amax", (ceil_align(core_num, AMAX_VECTOR_LANES),), torch.float32
            ),
        )
    if rows != meta.tokens:
        specs += (_ScratchSpec("ws_x_pad", (scratch_rows, meta.dim), torch.bfloat16),)
    return _ScratchPlan(rows=rows, specs=specs)


def make_compile_key(meta, core_num):
    shapes = (meta.q_lora,) if meta.weight_format == "nz" else ()
    return CompileKey(meta.weight_format, meta.qkv_quant_mode, GENERIC_PROFILE, shapes)


def runtime_controls(meta, core_num):
    if meta.weight_format != "nz":
        return _RuntimeControls()
    profile = token_profile(meta.tokens)
    bm = 128 if profile == GENERIC_PROFILE else profile
    columns = nz_qb_tile_width(meta.rows_qb, meta.heads, meta.qkv_quant_mode)
    if meta.qkv_quant_mode == 0 and 0 < profile <= 8:
        columns = select_qb_columns(meta.rows_qb, meta.heads, core_num) or columns
    front_hint = 4 if bm <= 8 and meta.qkv_quant_mode == 0 and meta.heads >= 64 else 0
    qb_hint = (
        4
        if 8 < bm <= 64 and meta.qkv_quant_mode == 0 and meta.heads >= 96
        else 2
        if bm <= 8 and meta.heads == 64
        else 0
    )
    wkb_hint = (
        4
        if meta.heads >= 96 and (bm <= 8 or bm <= 64 and meta.qkv_quant_mode != 2)
        else 0
    )
    return _RuntimeControls(
        front_load_hint=front_hint,
        qb_columns=columns,
        qb_load_hint=qb_hint,
        wkb_load_hint=wkb_hint,
    )


def qa_workspace_args(raw, norm_qa_matrix, qr_matrix):
    # Only the selected QA dtype is accessed by the device schedule. Reuse
    # already-needed views for the other slot; account for 96-column padding.
    width = raw.shape[1]
    if raw.dtype == torch.float32:
        inactive = qr_matrix
        if inactive.shape[1] != width:
            inactive = inactive.as_strided((0, width), (width, 1))
        return _QaWorkspace(bf16=inactive, fp32=raw)
    inactive = norm_qa_matrix
    if inactive.shape[1] != width:
        inactive = inactive.as_strided((0, width), (width, 1))
    return _QaWorkspace(bf16=raw, fp32=inactive)


def validate_quant_scale_value(quant_scale_kv):
    """Check the optional quantization scale in Eager before any Cache write.

    This check reads the scale back to CPU. Graph callers guarantee its value
    outside capture/replay. Cache index values are always caller-validated.
    """
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError(
            "checked Torch mla_prolog_v2 does not support graph capture; "
            "validate inputs before capture and use the public mla_prolog_v2 entry"
        )
    if quant_scale_kv is not None:
        scale = quant_scale_kv.detach().cpu().item()
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("quant_scale_kv must be finite and non-negative")


# NPU dispatcher delegates to the same single-launch implementation; cache_mode is
# serialized as an integer.
def validate_schema_modes(qkv_quant_mode, cache_mode, rope_mode):
    for name, value, allowed in (
        ("qkv_quant_mode", qkv_quant_mode, _QKV_QUANT_MODE_VALUES),
        ("cache_mode", cache_mode, _CACHE_MODE_VALUES),
        ("rope_mode", rope_mode, _ROPE_MODE_VALUES),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} schema value must be int")
        if value not in allowed:
            raise ValueError(f"{name} has unsupported schema value {value}")


# Public API, Torch registration and Native export.

if hasattr(torch, "library"):
    import torch_npu  # noqa: F401
    from torch.library import impl


__all__ = [
    "CacheMode",
    "QkvQuantMode",
    "RopeMode",
    "mla_prolog_v2",
    "torch_mla_prolog_v2",
    "export_mla_prolog_v4",
]


def native_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_resource(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid MLA Native resource: {relative!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def native_read_checked(root, relative, files):
    path = native_resource(root, relative)
    if native_sha(path) != files.get(relative):
        raise ValueError(f"MLA Native metadata SHA mismatch: {relative}")
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def native_catalog(source_file, operator):
    source = Path(source_file).resolve()
    root = source.parent / "_native"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["target"] != "dav-3510":
        raise ValueError("MLA Native target mismatch")
    metadata = native_read_checked(
        root, f"operators/{operator}/operator.json", manifest["files"]
    )
    source_sha = native_sha(source)
    programs = {}
    for entry in metadata.get("mla_dispatch", ()):
        if entry["source_sha256"] != source_sha:
            continue
        if entry["schema_version"] != 1:
            raise ValueError("Unsupported MLA Native dispatch schema")
        token, binary_key = entry["compile_key"], entry["binary_key"]
        if token in programs and programs[token] != binary_key:
            raise ValueError("Conflicting MLA Native compile key")
        programs[token] = binary_key
    if not programs:
        return None
    native_index = native_read_checked(root, manifest["index"], manifest["files"])
    return root, manifest, programs, native_index


def load_native_program(source_file, operator, compile_key):
    """Prefer an indexed Native; an absent resource or key uses the original builder."""
    try:
        return indexed_native_program(source_file, operator, compile_key)
    except FileNotFoundError:
        return None


def indexed_native_program(source_file, operator, compile_key):
    catalog = native_catalog(str(Path(source_file).resolve()), operator)
    if catalog is None:
        return None
    root, manifest, programs, native_index = catalog
    binary_key = programs.get(compile_key)
    if binary_key is None:
        return None
    indexed = native_index.get(binary_key)
    if indexed is None:
        return None
    descriptor = indexed["descriptor"]
    if not descriptor.startswith(f"operators/{operator}/variants/"):
        raise ValueError("MLA Native operator/descriptor mismatch")
    variant = native_read_checked(root, descriptor, manifest["files"])
    if variant["binary_key"] != binary_key or variant["target"] != manifest["target"]:
        raise ValueError("MLA Native variant identity mismatch")
    binary = native_resource(root, variant["binary"])
    sidecar = native_resource(root, variant["sidecar"])
    if sidecar != Path(str(binary) + ".meta.json"):
        raise ValueError("MLA Native sidecar path mismatch")
    metadata = native_read_checked(root, variant["sidecar"], manifest["files"])
    contract_digest = variant["key_inputs"]["call_contract_sha256"]
    if metadata["binary_sha256"] != manifest["files"].get(variant["binary"]):
        raise ValueError("MLA Native binary SHA metadata mismatch")
    if metadata["host_abi_version"] != manifest["host_abi_version"]:
        raise ValueError("MLA Native ABI mismatch")
    if metadata["contract_digest"] != contract_digest:
        raise ValueError("MLA Native call contract mismatch")
    program = dsl.load(str(binary))
    if (
        program.host_abi_version != manifest["host_abi_version"]
        or program.contract_digest != contract_digest
    ):
        program.close()
        raise ValueError("Loaded MLA Native ABI/contract mismatch")
    return program


class MlaPrologLauncher:
    def __init__(self, key, dims):
        self.key = key
        self.nz_active = key.fmt == "nz"
        self.head_hint = 0
        self.head_width = 0
        self.nz_wqb = 0
        self.nz_wfront = 128 if key.profile == 0 else 64
        if self.nz_active and 0 < key.profile <= 8:
            if dims.q_lora == 1536:
                self.nz_wfront = 48
            elif dims.q_lora in (640, 1024):
                self.nz_wfront = 32

        # This handle records static arguments; it does not allocate device buffers.
        self.kernel = MlaPrologKernel(
            dims.bm,
            self.nz_active,
            self.nz_wfront,
            self.nz_wqb,
            self.head_hint,
            self.head_width,
        )

    @host
    @surface_signature(ROPE_QR_HOST_ARGUMENTS, ROPE_QR_HOST_ANNOTATIONS)
    def run_qr_rope(self, *args):
        self.kernel[args[-1]].run_qr_rope(*args[:-1])

    @host
    @surface_signature(M2_ROPE_QR_HOST_ARGUMENTS, M2_ROPE_QR_HOST_ANNOTATIONS)
    def run_m2_qr_rope(self, *args):
        self.kernel[args[-1]].run_m2_qr_rope(*args[:-1])

    @host
    @surface_signature(M3_ROPE_QR_HOST_ARGUMENTS, M3_ROPE_QR_HOST_ANNOTATIONS)
    def run_m3_qr_rope(self, *args):
        self.kernel[args[-1]].run_m3_qr_rope(*args[:-1])


_PROGRAM_CACHE = {}
_PROGRAM_LOCK = threading.Lock()


def dims_for(key):
    tokens = Dim("T", min=1)
    rows = Dim("OutputRows", min=1)
    scratch_rows = Dim("ScratchRows", min=1, max=_SCRATCH_TOKEN_CAPACITY)
    bm = 128
    dim = Dim("Dm", min=1024, max=8192, multiple_of=NZ_PANEL_K)
    if key.fmt == "nz":
        (q_lora,) = key.shapes
        heads = Dim("N", min=1, max=128)
        kvr = Dim("Kvr", min=512, max=576, multiple_of=64)
        nope = Dim("Dn", min=128, max=256, multiple_of=64)
        rope_dim = kvr - KV_LORA
    else:
        q_lora = Dim("Ql", min=640, max=2048, multiple_of=128)
        kvr = Dim("Kvr", min=512, max=576, multiple_of=64)
        heads = Dim("N", min=1, max=128)
        nope = Dim("Dn", min=128, max=256, multiple_of=64)
        rope_dim = kvr - 512
    rows_qb = heads * (nope + rope_dim)
    blocks = Dim("Bn", min=1)
    block_size = Dim("Bs", min=16, max=1024, multiple_of=16)
    q_lora_padded = (q_lora + BN - 1) // BN * BN
    kvr_padded = (kvr + BN - 1) // BN * BN
    rows_qb_padded = (rows_qb + BN - 1) // BN * BN
    return _CompileDimensions(
        tokens=tokens,
        rows=rows,
        scratch_rows=scratch_rows,
        bm=bm,
        dim=dim,
        q_lora=q_lora,
        kvr=kvr,
        heads=heads,
        nope=nope,
        rows_qb=rows_qb,
        blocks=blocks,
        block_size=block_size,
        q_lora_padded=q_lora_padded,
        kvr_padded=kvr_padded,
        rows_qb_padded=rows_qb_padded,
    )


def build_program(key):
    if key.fmt not in ("nd", "nz"):
        raise ValueError(f"unsupported compile key: {key}")
    if key.fmt == "nz" and (not key.shapes):
        raise ValueError(
            f"unsupported compile key: {key} "
            "(NZ keys require the static resource tuple "
            "(q_lora,))"
        )
    if key.qkv_quant_mode not in (0, 1, 2, 3):
        raise ValueError(f"unsupported compile key: {key}")
    validate_resource_matrix()
    d = dims_for(key)
    validate_resource_budget(d.bm)
    front_width = 128 if key.profile == GENERIC_PROFILE else 64
    validate_nz_resource_budget(d.bm, (front_width, 64), NZ_PANEL_K)
    if key.fmt == "nz":
        for width in _QB_RUNTIME_WIDTHS:
            if key.qkv_quant_mode == 0 or width <= CUBE_BK:
                panel = (
                    (512 if width <= 192 else NZ_PANEL_K)
                    if 0 < key.profile <= 8 and d.q_lora != 640
                    else (
                        NZ_PANEL_K
                        if 0 < key.profile <= 64
                        and d.q_lora != 640
                        and key.qkv_quant_mode == 0
                        else CUBE_BK
                    )
                )
                validate_nz_resource_budget(d.bm, (width,), panel)
    rope_dim = d.kvr - KV_LORA
    cache_last_dim = (
        KV_LORA + 2 * rope_dim + DESCALE_TAIL_BYTES
        if key.qkv_quant_mode == 1
        else d.kvr
    )
    cache_dtype = (
        dtypes.uint8
        if key.qkv_quant_mode == 3
        else dtypes.bfloat16
        if key.qkv_quant_mode == 0
        else dtypes.float8_e4m3fn
    )
    spec = TensorSpec
    if key.fmt == "nz":
        weight_specs = (
            spec((d.dim, d.q_lora), dtypes.bfloat16, storage_format="nz"),
            spec((d.q_lora, d.rows_qb), dtypes.bfloat16, storage_format="nz"),
            spec((d.dim, d.kvr), dtypes.bfloat16, storage_format="nz"),
        )
    else:
        weight_specs = (
            spec((d.q_lora, d.dim), dtypes.bfloat16),
            spec((d.rows_qb, d.q_lora), dtypes.bfloat16),
            spec((d.kvr, d.dim), dtypes.bfloat16),
        )
    qb_aux_spec = spec((Dim("QrInputRows", min=1), d.q_lora), dtypes.bfloat16)
    launcher = MlaPrologLauncher(key, d)
    argument_specs = {
        "x": spec((d.tokens, d.dim), dtypes.bfloat16),
        "wqa": weight_specs[0],
        "wqb": weight_specs[1],
        "wkva": weight_specs[2],
        "wkb": spec((d.heads, d.nope, 512), dtypes.bfloat16),
        "norm_weight_qa": spec((1, d.q_lora), dtypes.float32),
        "norm_weight_kva": spec((1, 512), dtypes.float32),
        "kv_cache": spec(
            (d.blocks, d.block_size, 1, cache_last_dim),
            cache_dtype,
            stride=(Dim("cache_stride0", min=1), cache_last_dim, cache_last_dim, 1),
        ),
        "cache_index": spec((d.tokens,), dtypes.int64),
    }
    if key.qkv_quant_mode in (2, 3):
        argument_specs["quant_scale_kv"] = spec((1,), dtypes.float32)
    argument_specs.update(
        {
            "rope_sin": spec((Dim("RopeRows", min=0), SEG), dtypes.bfloat16),
            "rope_cos": spec((Dim("RopeRows", min=0), SEG), dtypes.bfloat16),
        }
    )
    q_dtype = (
        dtypes.uint8
        if key.qkv_quant_mode == 3
        else dtypes.float8_e4m3fn
        if key.qkv_quant_mode == 2
        else dtypes.bfloat16
    )
    q_spec = spec((d.rows, d.heads, d.kvr), q_dtype)
    descale_spec = (
        spec((1,), dtypes.float32)
        if key.qkv_quant_mode == 3
        else spec((d.tokens, d.heads, 1), dtypes.float32)
    )
    qr_spec = spec((Dim("QrRows", min=0), d.q_lora), dtypes.bfloat16)
    argument_specs["q"] = q_spec
    if key.qkv_quant_mode in (2, 3):
        argument_specs["descale_q"] = descale_spec
    argument_specs["qr"] = qr_spec
    argument_specs["ws_x_pad"] = spec((d.scratch_rows, d.dim), dtypes.bfloat16)
    argument_specs["ws_qa_raw"] = spec(
        (Dim("QaBf16Rows", min=0, max=_SCRATCH_TOKEN_CAPACITY), d.q_lora_padded),
        dtypes.bfloat16,
    )
    argument_specs["ws_qa_fp32"] = spec(
        (
            Dim("QaFp32Rows", min=0, max=QA_SPLIT_PARTS * _SCRATCH_TOKEN_CAPACITY),
            d.q_lora_padded,
        ),
        dtypes.float32,
    )
    argument_specs["ws_kva_raw"] = spec((d.scratch_rows, d.kvr_padded), dtypes.bfloat16)
    argument_specs["ws_q_proj"] = spec(
        (d.scratch_rows, d.rows_qb_padded),
        dtypes.bfloat16,
    )
    if key.qkv_quant_mode in (2, 3):
        argument_specs["ws_q_fp"] = spec(
            (
                Dim("QFpRows", min=1) if key.qkv_quant_mode == 3 else d.scratch_rows,
                d.heads,
                d.kvr,
            ),
            dtypes.float32,
        )
    if key.qkv_quant_mode == 3:
        amax_slots = Dim(
            "AmaxSlots", min=AMAX_VECTOR_LANES, multiple_of=AMAX_VECTOR_LANES
        )
        argument_specs["ws_amax"] = spec((amax_slots,), dtypes.float32)
    argument_specs["ws_qb_aux"] = qb_aux_spec
    argument_specs.update(
        {
            field.name: field.annotation
            for field in _DEVICE_ARGUMENT_FIELDS
            if field.role in ("scalar", "hints")
        }
    )
    argument_specs[_HOST_CORE_FIELD.name] = _HOST_CORE_FIELD.annotation
    entry = (
        launcher.run_m3_qr_rope
        if key.qkv_quant_mode == 3
        else launcher.run_m2_qr_rope
        if key.qkv_quant_mode == 2
        else launcher.run_qr_rope
    )
    return dsl.compile(
        entry,
        *(
            argument_specs[name]
            for name in _LAUNCH_ARGUMENTS_BY_MODE[key.qkv_quant_mode]
        ),
    )


def native_dispatch_key(key):
    return json.dumps(
        (key.fmt, key.qkv_quant_mode, key.profile, key.shapes), separators=(",", ":")
    )


def get_program(key):
    program = _PROGRAM_CACHE.get(key)
    if program is not None:
        return program
    with _PROGRAM_LOCK:
        program = _PROGRAM_CACHE.get(key)
        if program is None:
            program = load_native_program(
                __file__, "mla_prolog_v4", native_dispatch_key(key)
            )
            if program is None:
                program = build_program(key)
            _PROGRAM_CACHE[key] = program
        return program


def launch(meta, inputs, attributes):
    core_num = get_core_num(inputs.x, meta)
    key = make_compile_key(meta, core_num)
    program = get_program(key)
    plan = execution_scratch_specs(meta, core_num)
    outputs = empty_outputs(meta, inputs.x.device, plan.rows)
    scratch = allocate_scratch(plan.specs, inputs.x.device, scratch_reuse_groups(meta))
    norm_qa_matrix = inputs.norm_weight_qa.view(1, -1)
    qr_matrix = outputs.qr.view(-1, meta.q_lora)
    if not meta.return_qr and recompute_global_q(meta, plan.rows):
        qr_matrix = scratch.ws_qb_aux
    if meta.return_qr:
        scratch = replace(scratch, ws_qb_aux=qr_matrix)
    qa_workspace = qa_workspace_args(scratch.ws_qa_raw, norm_qa_matrix, qr_matrix)
    scratch = replace(
        scratch,
        ws_qa_raw=qa_workspace.bf16,
        ws_qa_fp32=qa_workspace.fp32,
        ws_x_pad=(
            scratch.ws_x_pad
            if plan.rows != meta.tokens
            else inputs.x[: scratch.ws_q_proj.shape[0]]
        ),
    )
    if not meta.rope_active:
        rope_sin = outputs.qr.as_strided((0, SEG), (SEG, 1))
        inputs = replace(inputs, rope_sin=rope_sin, rope_cos=rope_sin)
    inputs = replace(
        inputs,
        norm_weight_qa=norm_qa_matrix,
        norm_weight_kva=inputs.norm_weight_kva.view(1, -1),
    )
    launch_values = record_values(
        inputs, outputs, scratch, runtime_controls(meta, core_num)
    )
    launch_values.update(
        qr=qr_matrix,
        norm_eps=float(attributes.norm_eps),
        qr_scale=float(attributes.qr_scale),
        kvr_scale=float(attributes.kvr_scale),
        cache_mode=int(attributes.cache_mode == CacheMode.PA_NZ),
        rope_mode=int(attributes.rope_mode),
        core_num=core_num,
    )
    program(
        *(
            launch_values[name]
            for name in _LAUNCH_ARGUMENTS_BY_MODE[meta.qkv_quant_mode]
        )
    )
    return (outputs.q[: meta.tokens], outputs.descale_q, outputs.qr[: meta.tokens])


# Direct Python entry: fixed (q, descale_q, qr) slots; kv_cache is mutated in place.
def mla_prolog_v2_native(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qkv_quant_mode: QkvQuantMode,
    *,
    quant_scale_kv: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    cache_mode: CacheMode = CacheMode.PA_BSND,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode = RopeMode.ROPE_INTERLEAVE_HALF,
    return_qr: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute MLA Prolog and update kv_cache in place.

    Args:
        x: BF16 ND input, (T, dim).
        wqa: BF16 weight, ND (q_lora, dim), NZ (dim, q_lora).
        wqb: BF16 weight, ND (N*(Dn+Dr), q_lora), NZ (q_lora, N*(Dn+Dr)).
        wkva: BF16 weight, ND (512+Dr, dim), NZ (dim, 512+Dr).
            wqa/wqb/wkva must be all ND or all true NZ.
        wkb: BF16 ND weight, (N, Dn, 512).
        norm_weight_qa: FP32 gamma, (q_lora,).
        norm_weight_kva: FP32 gamma, (512,).
        kv_cache: Mutable Cache; layout selected by cache_mode.
        cache_index: INT64 indices, (T,); negative indices skip Cache writes.
        quant_scale_kv: FP32 (1,), required only for A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR/A8C8_QKV_HIFLOAT8_PER_TENSOR.
        rope_sin: BF16 (T, Dr), paired with rope_cos when RoPE is enabled.
        rope_cos: BF16 (T, Dr), paired with rope_sin.
        norm_eps: Shared positive FP32-representable RMSNorm epsilon.
        qkv_quant_mode: A16C16; A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE (PA_BSND only);
            A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR (Q scale per Token/Head); A8C8_QKV_HIFLOAT8_PER_TENSOR (global Q scale).
        cache_mode: PA_BSND or PA_NZ physical Cache addressing.
        qr_scale: Scales QA RMSNorm output before the BF16 CQ cast.
        kvr_scale: Scales normalized Ckv only, never KR.
        rope_mode: NO_ROPE, ROPE_INTERLEAVE_HALF or ROPE_HALF; Dr=0 requires NO_ROPE.
        return_qr: Whether the fixed qr slot contains the BF16 CQ.

    Returns:
        (q, descale_q, qr). Disabled descale_q/qr slots have shape (0,).
        HiF8 q/Cache use uint8 raw-byte carriers, not UINT8 quantization.
        T=0 returns empty outputs without a Kernel launch; Cache is unchanged.
        Empty descale_q is (0, N, 1) in A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR and (0,) in other modes.
        No Cache clone, copy-back or extra Cache output is introduced.
    """
    validate_mode_enums(qkv_quant_mode, cache_mode, rope_mode)
    inputs = _InputTensors(
        x=x,
        wqa=wqa,
        wqb=wqb,
        wkva=wkva,
        wkb=wkb,
        norm_weight_qa=norm_weight_qa,
        norm_weight_kva=norm_weight_kva,
        kv_cache=kv_cache,
        cache_index=cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
    )
    attributes = _Attributes(
        norm_eps=norm_eps,
        qkv_quant_mode=qkv_quant_mode.value,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode.value,
        return_qr=return_qr,
    )
    meta = validate(inputs, attributes)
    if meta.tokens == 0:
        return empty_outputs(meta, x.device).as_tuple()
    return launch(meta, inputs, attributes)


def native_nz_q_lora_values():
    encoded = os.environ.get("MLA_PROLOG_NATIVE_NZ_PROFILES")
    if not encoded:
        return Q_LORA_VALUES
    raw_profiles = json.loads(encoded)
    require(
        isinstance(raw_profiles, (list, tuple)) and raw_profiles,
        "Native NZ profile manifest must be non-empty",
    )
    profiles = []
    for number, raw in enumerate(raw_profiles):
        require(
            isinstance(raw, (list, tuple)) and len(raw) == 5,
            f"Native NZ profile {number} must be [dim,q_lora,N,Dn,Dr]",
        )
        require(
            all((type(value) is int for value in raw)),
            f"Native NZ profile {number} values must be integers",
        )
        (dim, q_lora, heads, nope_dim, rope_dim) = raw
        require(
            1024 <= dim <= 8192 and dim % 256 == 0,
            f"Native NZ profile {number} has illegal dim",
        )
        require(
            q_lora in (640, 1024, 1536, 2048),
            f"Native NZ profile {number} has illegal q_lora",
        )
        require(1 <= heads <= 128, f"Native NZ profile {number} has illegal N")
        require(
            (nope_dim, rope_dim) in ((128, 64), (192, 64), (256, 0)),
            f"Native NZ profile {number} has illegal (Dn,Dr)",
        )
        profile = tuple(raw)
        if profile not in profiles:
            profiles.append(profile)
    return tuple(dict.fromkeys(profile[1] for profile in profiles))


def export_keys():
    keys = {}
    for mode in (0, 1, 2, 3):
        for profile in (GENERIC_PROFILE,):
            keys[CompileKey("nd", mode, profile)] = None
    for q_lora in native_nz_q_lora_values():
        for mode in (0, 1, 2, 3):
            keys[CompileKey("nz", mode, GENERIC_PROFILE, (q_lora,))] = None
    return tuple(keys)


@compile_cache
def compile_export_variant(fmt, qkv_quant_mode, profile, q_lora=0):
    shapes = (q_lora,) if fmt == "nz" else ()
    key = CompileKey(fmt, qkv_quant_mode, profile, shapes)
    session = current_collection()
    if session is None:
        return build_program(key)
    compilations = []
    with observe_compilations(lambda artifact, source: compilations.append(artifact)):
        program = build_program(key)
    try:
        if len(compilations) != 1:
            raise RuntimeError("MLA export must compile exactly one Program")
        session.add_task_metadata(
            "mla_dispatch",
            dict(
                schema_version=1,
                compile_key=native_dispatch_key(key),
                binary_key=compilations[0].binary_key,
                source_sha256=native_sha(Path(__file__)),
            ),
        )
    except BaseException:
        program.close()
        raise
    return program


@export("mla_prolog_v4")
def export_mla_prolog_v4():
    for key in export_keys():
        compile_export_variant(key.fmt, key.qkv_quant_mode, key.profile, *key.shapes)


def close_programs():
    for program in tuple(_PROGRAM_CACHE.values()):
        program.close()
    _PROGRAM_CACHE.clear()


atexit.register(close_programs)


# Torch schema: Tensor(a!) declares Cache mutation. Outputs do not alias Cache; no
# clone/copy-back is inserted here.
_LIBRARY = None
_TORCH_SCHEMA = (
    "mla_prolog_v2("
    "Tensor x, "
    "Tensor wqa, "
    "Tensor wqb, "
    "Tensor wkva, "
    "Tensor wkb, "
    "Tensor norm_weight_qa, "
    "Tensor norm_weight_kva, "
    "Tensor(a!) kv_cache, "
    "Tensor cache_index, int qkv_quant_mode, *, "
    "Tensor? quant_scale_kv=None, "
    "Tensor? rope_sin=None, "
    "Tensor? rope_cos=None, "
    "float norm_eps=1e-5, "
    "int cache_mode=0, "
    "float qr_scale=1.0, "
    "float kvr_scale=1.0, "
    "int rope_mode=1, "
    "bool return_qr=False) -> (Tensor, Tensor, Tensor)"
)


def mla_prolog_v2_npu(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qkv_quant_mode: int,
    *,
    quant_scale_kv: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    cache_mode: int = 0,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager dispatch checks quantization scale; index values are caller-owned."""
    validate_schema_modes(qkv_quant_mode, cache_mode, rope_mode)
    validate(
        _InputTensors(
            x=x,
            wqa=wqa,
            wqb=wqb,
            wkva=wkva,
            wkb=wkb,
            norm_weight_qa=norm_weight_qa,
            norm_weight_kva=norm_weight_kva,
            kv_cache=kv_cache,
            cache_index=cache_index,
            quant_scale_kv=quant_scale_kv,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
        ),
        _Attributes(
            norm_eps=norm_eps,
            qkv_quant_mode=qkv_quant_mode,
            cache_mode=CacheMode(cache_mode),
            qr_scale=qr_scale,
            kvr_scale=kvr_scale,
            rope_mode=rope_mode,
            return_qr=return_qr,
        ),
    )
    validate_quant_scale_value(quant_scale_kv)
    return mla_prolog_v2_native(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        qkv_quant_mode=QkvQuantMode(qkv_quant_mode),
        cache_mode=CacheMode(cache_mode),
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=RopeMode(rope_mode),
        return_qr=return_qr,
    )


# Fake tensors describe outputs only, without a device launch. Disabled outputs and
# empty descale use shape (0,).
def mla_prolog_v2_fake(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qkv_quant_mode: int,
    *,
    quant_scale_kv: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    cache_mode: int = 0,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe fixed output slots without reading device data or launching DSL."""
    validate_schema_modes(qkv_quant_mode, cache_mode, rope_mode)
    require_rank("x", x, 2, "(T, dim)")
    require_rank("wkb", wkb, 3, "(N, Dn, 512)")
    if return_qr:
        require_rank("norm_weight_qa", norm_weight_qa, 1, "(q_lora,)")
    (t, heads, dr) = (x.shape[0], wkb.shape[0], 0 if wkb.shape[1] == 256 else 64)
    q_dtype = (
        torch.bfloat16
        if qkv_quant_mode < 2
        else torch.float8_e4m3fn
        if qkv_quant_mode == 2
        else torch.uint8
    )
    scale_shape = (
        (0,)
        if qkv_quant_mode < 2 or (t == 0 and qkv_quant_mode == 3)
        else (t, heads, 1)
        if qkv_quant_mode == 2
        else (1,)
    )
    return (
        x.new_empty((t, heads, 512 + dr), dtype=q_dtype),
        x.new_empty(scale_shape, dtype=torch.float32),
        x.new_empty(
            (t, norm_weight_qa.shape[0]) if return_qr else (0,), dtype=torch.bfloat16
        ),
    )


def mla_prolog_v2_graph_npu(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qkv_quant_mode,
    *,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    norm_eps=1e-5,
    cache_mode=0,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
):
    """Graph dispatch requires device values validated outside capture/replay."""
    validate_schema_modes(qkv_quant_mode, cache_mode, rope_mode)
    return mla_prolog_v2_native(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        qkv_quant_mode=QkvQuantMode(qkv_quant_mode),
        cache_mode=CacheMode(cache_mode),
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=RopeMode(rope_mode),
        return_qr=return_qr,
    )


def register_torch():
    global _LIBRARY
    schemas = (
        ("mla_prolog_v2", _TORCH_SCHEMA, mla_prolog_v2_npu),
        (
            "_mla_prolog_v2_graph",
            _TORCH_SCHEMA.replace("mla_prolog_v2(", "_mla_prolog_v2_graph(", 1),
            mla_prolog_v2_graph_npu,
        ),
    )
    existing = getattr(torch.ops.cannbot_dsl_mla_prolog_v4, "mla_prolog_v2", None)
    if existing is not None:
        for name, schema, _ in schemas:
            registered = getattr(torch.ops.cannbot_dsl_mla_prolog_v4, name, None)
            expected = torch._C.parse_schema("cannbot_dsl_mla_prolog_v4::" + schema)
            if registered is None or str(registered.default._schema) != str(expected):
                raise RuntimeError(
                    "mla_prolog_v2 is already registered with another schema"
                )
        return
    # Native discovery reimports this file under a temporary module name. Keep the
    # first registration alive and do not register the same Torch schema twice.
    _LIBRARY = torch.library.Library("cannbot_dsl_mla_prolog_v4", "FRAGMENT")
    for name, schema, implementation in schemas:
        _LIBRARY.define(schema)
        impl(_LIBRARY, name, "PrivateUse1")(implementation)
        torch.library.register_fake("cannbot_dsl_mla_prolog_v4::" + name)(
            mla_prolog_v2_fake
        )


if hasattr(torch, "library"):
    register_torch()


# Checked Eager entry. Preserve enum validation before entering the dispatcher.
def mla_prolog_v2(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qkv_quant_mode: QkvQuantMode,
    *,
    quant_scale_kv: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-5,
    cache_mode: CacheMode = CacheMode.PA_BSND,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode = RopeMode.ROPE_INTERLEAVE_HALF,
    return_qr: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public Eager, ACLGraph and torch.compile entry.

    The schema serializes the three enums as integers. Cache mutation is
    declared by Tensor(a!); this entry does not clone or return Cache.
    Cache index values are not read back or checked; callers guarantee valid
    indices in every execution mode. Eager checks quant_scale_kv on CPU.
    Manual capture and torch.compile use asynchronous graph dispatch; callers
    guarantee scale validity outside capture/replay. No result is cached.
    """
    validate_mode_enums(qkv_quant_mode, cache_mode, rope_mode)
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")
    # Torch's float schema coerces bool, so check Python types before dispatch.
    validate_scalar_types(norm_eps, qr_scale, kvr_scale)
    dispatch = (
        torch.ops.cannbot_dsl_mla_prolog_v4._mla_prolog_v2_graph.default
        if torch.compiler.is_compiling()
        or (
            isinstance(x, torch.Tensor)
            and x.device.type in ("npu", "privateuseone")
            and torch.npu.is_current_stream_capturing()
        )
        else torch.ops.cannbot_dsl_mla_prolog_v4.mla_prolog_v2.default
    )
    return dispatch(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        qkv_quant_mode=qkv_quant_mode.value,
        cache_mode=cache_mode.value,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode.value,
        return_qr=return_qr,
    )


# Compatibility name; registration and graph routing stay behind the public API.
torch_mla_prolog_v2 = mla_prolog_v2


for _entry in (mla_prolog_v2, mla_prolog_v2_native):
    _entry.QkvQuantMode = QkvQuantMode
    _entry.CacheMode = CacheMode
    _entry.RopeMode = RopeMode
