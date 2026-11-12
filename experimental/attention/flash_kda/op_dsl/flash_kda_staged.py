# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# fmt: off
# 保持参数分组紧凑，与原有 FlashKDA 样例风格一致。

"""训练前向的分阶段 chunk 分组调度、矩阵和向量资源及序列视图。"""
from cannbotdsl import from_tensor
from cannbotdsl.ops.memcpy import cube_fill_l1_zero

import dataclasses
import os
from typing import NamedTuple, Optional, Tuple

from cannbotdsl import make_layout, PIPE, MemLoc, Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang import vf
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import vpack, vreinterpret, vreinterpret_lanes, vstore_strided
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    mask_xor,
    update_mask,
    vabs,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdups,
    vexp,
    vexp_sub,
    vload,
    vload_broadcast,
    vload_unpack,
    vlog,
    vmaxs,
    vmins,
    vmul,
    vmuls,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.reg import vselect as vselect_raw
from cannbotdsl.ops.sync import cube_sync_all, cube_sync_block_arrive, cube_sync_block_wait, cube_sync_pipe, onchip_rewind, vec_sync_all, vec_sync_block_arrive, vec_sync_block_wait, vec_sync_notify, vec_sync_wait
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import idx2crd, tile_slice
from cannbotdsl import Array
import torch

from .flash_kda_validation import CHUNK_SIZE, SUPPORTED_HEAD_DIM
from . import flash_kda as fwd
from .flash_kda_metadata import (
    TEMPLATE_STAGED,
    HEADER_CORE_NUM,
    HEADER_STAGE12_ROUND_NUM,
    HEADER_STATUS,
    HEADER_WORDS,
    MAX_AIC_CORES,
    STATUS_OK,
)

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


L2_NORM_EPS = 1e-6
MAX_INVERSE_NORM = L2_NORM_EPS ** -0.5
NZ_TILE = 16
NZ_TILE_ELEMENTS = NZ_TILE * NZ_TILE
TRANSFER_BLOCK_BITS = 256


def nz_layout(rows, cols, stride=None):
    """按 NZ 分形排列行列；新缓冲区每列分形补一行，别名视图沿用源步长。"""
    row_tiles = (rows + NZ_TILE - 1) // NZ_TILE
    col_tiles = (cols + NZ_TILE - 1) // NZ_TILE
    if stride is None:
        stride = (row_tiles * NZ_TILE_ELEMENTS + NZ_TILE, NZ_TILE_ELEMENTS, NZ_TILE, 1)
    return make_layout((col_tiles, row_tiles, NZ_TILE, NZ_TILE), stride=stride)


HALF_CHUNK_SIZE = CHUNK_SIZE // 2
STABLE_SCORE_ROWS = 7 * HALF_CHUNK_SIZE
D_HALF = SUPPORTED_HEAD_DIM // 2
GATE_MODE_PREPROCESSED = 0
GATE_MODE_FUSED_WITH_DT_BIAS = 1
GATE_MODE_FUSED_WITHOUT_DT_BIAS = 2
_L2_CACHE_LAST_USE = 1
_L2_CACHE_DISABLE = 4
_SAVED_OUTPUT_L2_CACHE_CTL = _L2_CACHE_DISABLE
_STAGE2_INPUT_L2_CACHE_CTL = 0


@jit
def cast_tile(dst, src):
    rows, cols = dst.shape
    src_stride = src.physical_layout.stride[0]
    dst_stride = dst.physical_layout.stride[0]
    with vf(mode="simd"):
        for row in range(rows):
            if const_expr(isinstance(cols, int) and cols == 1):
                value = vload_broadcast(src, row * src_stride)
                converted = vcast(value, dst.dtype, mask=full_mask())
                vstore_first(dst, row * dst_stride, converted)
            else:
                for col in range(0, cols, 64):
                    mask, _ = update_mask(cols - col, elem_bits=32)
                    if const_expr(src.dtype == dtypes.float32):
                        value = vload(src, row * src_stride + col)
                    else:
                        value = vload_unpack(src, row * src_stride + col, unpack_mode=UnpackMode.B16_TO_B32)
                    converted = vcast(value, dst.dtype, mask=mask)
                    if const_expr(dst.dtype == dtypes.float32):
                        vstore(dst, row * dst_stride + col, converted, mask)
                    else:
                        vstore_pack(dst, row * dst_stride + col, converted, mask, pack_mode=PackMode.B32_TO_B16)


@jit
def prefix_sum_rows(dst, src):
    rows, cols = dst.shape
    src_stride = src.physical_layout.stride[0]
    dst_stride = dst.physical_layout.stride[0]
    with vf(mode="simd"):
        for col in range(0, cols, 64):
            mask, _ = update_mask(cols - col, elem_bits=32)
            prefix = vload(src, col)
            vstore(dst, col, prefix, mask)
            for row in range(1, rows):
                prefix = vadd(prefix, vload(src, row * src_stride + col), mask=mask)
                vstore(dst, row * dst_stride + col, prefix, mask)


def _valid_rows_view(full_tile, valid_rows, cols: int):
    """将重定位后的紧凑 TND 分块裁剪到当前逻辑序列。"""
    return full_tile[:valid_rows, :cols]


def _o_head_chunk_tile(gm_O, seq_idx, chunk_idx, dv_base, dv_idx, valid_rows):
    """返回一个逻辑 BNSD 输出 chunk，并保留物理步长。"""
    nv = gm_O.shape[1]
    batch_idx = seq_idx // nv
    head_idx = seq_idx - batch_idx * nv
    full_chunk = tile_slice(gm_O[batch_idx, head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
    column_tile = tile_slice(full_chunk, (CHUNK_SIZE, dv_base), (0, dv_idx))
    return _valid_rows_view(column_tile, valid_rows, dv_base)


@jit
def nd2nz_tile(dst, src, elem_bits: int = 16):
    rows, cols = src.shape
    src_stride = src.physical_layout.stride[0]
    dst_stride = dst.physical_layout.stride[0]
    n0 = TRANSFER_BLOCK_BITS // elem_bits
    block_stride = dst_stride * elem_bits // TRANSFER_BLOCK_BITS
    with vf(mode="simd"):
        for row in range(rows):
            remaining = rr.mask_counter(cols)
            for col in range(0, cols, 2048 // elem_bits):
                mask, remaining = rr.update_mask(remaining, elem_bits=elem_bits)
                values = rr.vload(src, row * src_stride + col)
                offset = row * n0 + col // n0 * dst_stride
                rr.vstore_strided(dst, offset, values, mask, block_stride=block_stride, repeat_stride=0)


class _Stage1Constants(NamedTuple):
    identity16: torch.Tensor


_STAGE1_CONSTANTS_CACHE: dict[
    tuple[str, int | None, torch.dtype], _Stage1Constants
] = {}


@dataclasses.dataclass
class StageOneWorkspace:
    Q_decayed: torch.Tensor
    gamma_C: torch.Tensor
    identity16: torch.Tensor


def _empty_like_device(
    shape: tuple[int, ...], *, ref: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=ref.device)


def allocate_qk_exchange_workspace(
    *, active_block_num: int, ref: torch.Tensor
) -> torch.Tensor:
    """在每个 AIV 交换方向分配 Q/K 两组部分和。"""
    if not isinstance(active_block_num, int) or isinstance(active_block_num, bool) or active_block_num <= 0:
        raise ValueError("active_block_num must be a positive integer")
    return _empty_like_device((active_block_num, 2, 2, CHUNK_SIZE), ref=ref, dtype=torch.float32)


def qk_exchange_slices(
    workspace: torch.Tensor, *, block_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回当前块的两个 AIV 间交换方向。"""
    if not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.float32:
        raise ValueError("qk exchange workspace must be a float32 Tensor")
    if not (workspace.dim() == 4 and tuple(workspace.shape[1:]) == (2, 2, CHUNK_SIZE)):
        raise ValueError("qk exchange workspace must have shape [cores, 2, 2, 64]")
    if not workspace.is_contiguous():
        raise ValueError("qk exchange workspace must be contiguous")
    if not isinstance(block_idx, int) or isinstance(block_idx, bool) or not 0 <= block_idx < workspace.shape[0]:
        raise ValueError("block_idx is outside the exchange workspace")
    return (workspace[block_idx, 0], workspace[block_idx, 1])


def _resolved_device_key(ref: torch.Tensor) -> tuple[str, int | None]:
    device = ref.device
    device_type = getattr(device, "type", "cpu")
    device_index = getattr(device, "index", None)
    if device_type in {"npu", "privateuseone"} and device_index is None:
        npu = getattr(torch, "npu", None)
        current_device = (
            getattr(npu, "current_device", None) if npu is not None else None
        )
        if current_device is not None:
            device_index = int(current_device())
    return (device_type, device_index)


def _stage1_constants_for(
    ref: torch.Tensor, *, identity_dtype: torch.dtype = torch.float16
) -> _Stage1Constants:
    device_type, device_index = _resolved_device_key(ref)
    key = (device_type, device_index, identity_dtype)
    cached = _STAGE1_CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached
    constants = _Stage1Constants(identity16=torch.eye(NEUMANN_BLOCK_SIZE, dtype=identity_dtype, device=ref.device).contiguous())
    _STAGE1_CONSTANTS_CACHE[key] = constants
    return constants


def allocate_stage1_workspace(
    batch: int, heads: int, eff_group: int, dim: int, *, ref: torch.Tensor
) -> StageOneWorkspace:
    bn_chunks = batch * heads * eff_group
    constants = _stage1_constants_for(ref)
    return StageOneWorkspace(
        Q_decayed=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=torch.bfloat16),
        gamma_C=_empty_like_device((bn_chunks * dim, 1), ref=ref, dtype=torch.float32), identity16=constants.identity16,
    )


ELEM_BYTES = 4
VL = 64
NEUMANN_BLOCK_SIZE = 16
NEUMANN_DIAG_BLOCK_NUM = 4
NEUMANN_FRACTAL_ELEMS = NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
NEUMANN_DIAG_SRC_STRIDE = NEUMANN_BLOCK_SIZE * CHUNK_SIZE + NEUMANN_BLOCK_SIZE
NEUMANN_DIAG_DST_STRIDE = (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS


@jit
def zero_recurrent_state(ub_state, elements):
    """在片上初始化当前分块的 FP32 递推状态。"""
    with vf(mode="simd"):
        mask, _ = update_mask(VL, elem_bits=32)
        zero = vdups(0.0, dtypes.float32, mask=mask)
        for offset in dsl_range(0, elements, VL, unroll=4):
            vstore(ub_state, offset, zero, mask)


class StageOneMatmul:
    """StageOne Cube 侧矩阵乘计算。"""

    def __init__(self):
        self.tile_power_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.packed_inv64_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.tile_inv_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.scratch_l1 = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1).next_write()
        self.neumann_power_handoff_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.packed_identity_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.zero_l1 = Buffer(MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)))
        self.resident_inv_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.l0a_db = RingBuffer(MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=2)
        self.l0b_db = RingBuffer(MemLoc.L0B, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=1)
        self.beta_l0b = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((SUPPORTED_HEAD_DIM, CHUNK_SIZE)), depth=1
        ).next_write()
        self.l0c_db = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=2)
        self.per_d_left_k_l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), depth=1
        ).next_write()
        self.per_d_left_q_l0a = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), depth=1
        ).next_write()
        self.per_d_l0b_db = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), depth=2
        )
        self.raw_l0a_db = RingBuffer(MemLoc.L0A, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.raw_l0b_db = RingBuffer(MemLoc.L0B, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.raw_l0c_main = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.raw_l0c_aux = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.per_d_shared_extra = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), depth=2
        )
        self.per_d_shared_right_cache = (
            self.per_d_l0b_db.next_write(),
            self.per_d_l0b_db.next_write(),
            self.per_d_shared_extra.next_write(),
            self.per_d_shared_extra.next_write(),
        )
        self.l1_shared_score_cache = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((14 * NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), depth=1
        ).next_write()
        score_alias0 = self.raw_l0b_db.next_write()
        self.raw_l0b_db.next_write()  # 推进双缓冲写指针，当前阶段不使用第二个别名。
        self.shared_rhs_cache = self.per_d_shared_right_cache + (
            from_tensor(score_alias0, layout=make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), offset=0, data_format="nz", dtype=dtypes.bfloat16),
            from_tensor(score_alias0, layout=make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), offset=NEUMANN_BLOCK_SIZE * SUPPORTED_HEAD_DIM * 2, data_format="nz", dtype=dtypes.bfloat16),
        )

    @staticmethod
    def packed64_block(l1_matrix, row_block: int, col_block: int):
        return (None, l1_matrix, row_block, col_block)

    @staticmethod
    def l0_matrix_block_offset_elems(row_block: int, col_block: int):
        return (row_block * NEUMANN_DIAG_BLOCK_NUM + col_block) * NEUMANN_FRACTAL_ELEMS

    @staticmethod
    def c64_diag_nz_offset_elems(diag_block: int):
        return diag_block * (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS

    @staticmethod
    def zero_half_l1(l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    @staticmethod
    def fixpipe_packed64_full_to_half_l1(dst_ring_buffer, c_tile):
        mem_copy(dst_ring_buffer, c_tile, options=copy_options(unit_flag=3))

    @staticmethod
    def fixpipe_packed64_full_to_bf16_l1(dst_ring_buffer, c_tile):
        mem_copy(dst_ring_buffer, c_tile, options=copy_options(unit_flag=3))

    @staticmethod
    def _packed64_block_offset_elems(row_block: int, col_block: int):
        return (
            col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
            + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
        )

    def load_k_decayed(self, k_decayed_l1):
        mem_copy(self.l0a_db.next_write(), k_decayed_l1)

    def load_q_decayed(self, q_decayed_l1):
        mem_copy(self.l0a_db.next_write(), q_decayed_l1)

    def load_k_inv(self, k_inv_l1):
        mem_copy(self.l0b_db.next_write(), k_inv_l1)

    def mm_kkt(self):
        k_inv = self.l0b_db.next_read()
        result = self.l0c_db.next_write()[0:CHUNK_SIZE, 0:CHUNK_SIZE]
        matmul(result, self.l0a_db.next_read(), k_inv, init=True)
        return k_inv

    def mm_mqk(self, k_inv_l0b):
        result = self.l0c_db.next_write()[0:CHUNK_SIZE, 0:CHUNK_SIZE]
        matmul(result, self.l0a_db.next_read(), k_inv_l0b, init=True)

    def load_per_d_left_k(self, l1_ring_buffer):
        mem_copy(self.per_d_left_k_l0a, l1_ring_buffer.next_read())

    def load_per_d_left_q(self, l1_ring_buffer):
        mem_copy(self.per_d_left_q_l0a, l1_ring_buffer.next_read())

    def load_per_d_right_k(self, l1_ring_buffer):
        destination = self.per_d_l0b_db.next_write()
        destination_source = l1_ring_buffer.next_read()
        destination = destination[tuple(map(slice, destination_source.shape))]
        mem_copy(destination, destination_source)
        return destination

    def mm_per_d_kkt(self, row_block: int, col_block: int, right, init=True):
        kkt_tile = tile_slice(self.raw_l0c_main, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        matmul(from_tensor(kkt_tile, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), data_format="nz"), self.per_d_left_k_l0a, right, init=init)

    def mm_per_d_mqk(self, row_block: int, col_block: int, right, init=True):
        mqk_tile = tile_slice(self.raw_l0c_aux, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        matmul(from_tensor(mqk_tile, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), data_format="nz"), self.per_d_left_q_l0a, right, init=init)

    @jit
    def mm_safe_diagonal_tile(
        self,
        vector,
        normalized_k_dhalf,
        normalized_q_dhalf,
        gate_cumsum_dhalf,
        row_block: int,
    ):
        """使用非正指数，逐行累加因果结果。"""
        vector.prepare_safe_diagonal_left()
        for active_row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1):
            vector.update_safe_diagonal_left(
                normalized_k_dhalf, normalized_q_dhalf, row_block, active_row, vector.l1_per_d_operand,
            )
            self.load_per_d_left_k(vector.l1_per_d_operand)
            self.load_per_d_left_q(vector.l1_per_d_operand)
            vector.build_safe_diagonal_right(
                normalized_k_dhalf, gate_cumsum_dhalf, row_block, active_row, vector.l1_per_d_operand,
            )
            right = self.load_per_d_right_k(vector.l1_per_d_operand)
            self.mm_per_d_kkt(row_block, row_block, right, init=active_row == 0)
            self.mm_per_d_mqk(row_block, row_block, right, init=active_row == 0)

    def finish_per_d_kkt(self, ub_kkt):
        mem_copy(ub_kkt, (self.raw_l0c_main).partition(dim=0))

    def finish_per_d_mqk(self, ub_mqk):
        mem_copy(ub_mqk.next_write(), (self.raw_l0c_aux).partition(dim=0))

    def store_l0c_to_gm(self, gm_tile):
        value = self.l0c_db.next_read()[0:CHUNK_SIZE, 0:gm_tile.shape[1]]
        mem_copy(gm_tile, value[:gm_tile.shape[0], :gm_tile.shape[1]])

    def c64_odd_even_pair_dst_gap_fractals(self):
        return self.c64_diag_nz_offset_elems(2) // NEUMANN_FRACTAL_ELEMS - 1

    def zero_packed64_l0_operands(self, l0a_slot, l0b_slot):
        mem_copy(l0a_slot, self.zero_l1)
        mem_copy(l0b_slot, self.zero_l1)

    def load_packed64_diag_only_operands_to_l0(
        self, left_l1, right_l1, l0a_slot, l0b_slot, *, clear_l0: bool = True
    ):
        if clear_l0:
            self.zero_packed64_l0_operands(l0a_slot, l0b_slot)
        for diagonal_tile in range_constexpr(NEUMANN_DIAG_BLOCK_NUM):
            mem_copy(
                tile_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (diagonal_tile, diagonal_tile)),
                tile_slice(left_l1, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (diagonal_tile, diagonal_tile)),
            )
        for diagonal_tile in range_constexpr(NEUMANN_DIAG_BLOCK_NUM):
            mem_copy(
                tile_slice(l0b_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (diagonal_tile, diagonal_tile)),
                (tile_slice(right_l1, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (diagonal_tile, diagonal_tile))).permute(1, 0),
            )

    def issue_packed64_diag_only_operands_to_l0_db(
        self, left_l1, right_l1, *, clear_l0: bool = True
    ):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.load_packed64_diag_only_operands_to_l0(left_l1, right_l1, l0a_write, l0b_write, clear_l0=clear_l0)

    def begin_mmad_raw_l0ab_db_packed64(self, c_tile, *, init_c: bool, unit_flag: int):
        """发起 full64 raw MMAD，显式指定 init/accumulate。"""
        l0a_read = self.raw_l0a_db.next_read()
        l0b_read = self.raw_l0b_db.next_read()
        matmul(c_tile, l0a_read, l0b_read, init=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only(
        self,
        left_l1,
        right_l1,
        c_tile,
        *,
        init_c: bool,
        unit_flag: int,
        clear_l0: bool = True,
    ):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l1, right_l1, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c, unit_flag=unit_flag)

    def raw_mmad_packed64_diag_only_db_pair(
        self,
        left_l10,
        right_l10,
        left_l11,
        right_l11,
        c_tile,
        *,
        init_c0: bool,
        unit_flag0: int,
        init_c1: bool,
        unit_flag1: int,
        clear_l0: bool = True,
    ):
        self.issue_packed64_diag_only_operands_to_l0_db(left_l10, right_l10, clear_l0=clear_l0)
        self.issue_packed64_diag_only_operands_to_l0_db(left_l11, right_l11, clear_l0=clear_l0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c0, unit_flag=unit_flag0)
        self.begin_mmad_raw_l0ab_db_packed64(c_tile, init_c=init_c1, unit_flag=unit_flag1)

    def half_l1_power_packed64_diag_only_on_c(
        self, left_l1, right_l1, dst_ring_buffer, *, clear_l0: bool
    ):
        c_power = self.raw_l0c_aux
        self.raw_mmad_packed64_diag_only(left_l1, right_l1, c_power, init_c=True, unit_flag=3, clear_l0=clear_l0)
        c_power_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(dst_ring_buffer, c_power_read)

    def prepare_packed64_identity_l1(self, gm_identity16):
        """构造 packed 64x64 identity。"""
        identity_write = self.packed_identity_l1
        self.zero_half_l1(identity_write)
        cube_sync_pipe(PIPE.MTE2)
        for diagonal_tile in range_constexpr(NEUMANN_DIAG_BLOCK_NUM):
            mem_copy(
                tile_slice(identity_write, (16, 16), (diagonal_tile, diagonal_tile)),
                tile_slice(gm_identity16, (16, 16), (0, 0)),
            )
        return self.packed_identity_l1

    def issue_negative_t_identity_operands(self, negative_t, identity):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        mem_copy(l0a_write, negative_t)
        mem_copy(l0b_write, identity)

    def issue_identity_identity_operands(self, identity):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        mem_copy(l0a_write, identity)
        mem_copy(l0b_write, identity)

    def prepare_inv_init_from_negative_t_cube(self, negative_t, identity):
        """Cube 侧在 L0C 构造 scratch_l1 = (-T) @ I + I @ I。"""
        self.issue_negative_t_identity_operands(negative_t, identity)
        self.issue_identity_identity_operands(identity)
        c_inv_init = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=True, unit_flag=2)
        self.begin_mmad_raw_l0ab_db_packed64(c_inv_init, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.scratch_l1, c_inv_read)
        return self.scratch_l1

    def load_packed64_diagonal_inputs_from_l1_handoff(self, power_handoff_l1, identity):
        """消费 AIV `-T` handoff，并在 Cube 侧构造 `I + (-T)`。"""
        negative_t = power_handoff_l1
        scratch = self.prepare_inv_init_from_negative_t_cube(negative_t, identity)
        return (negative_t, identity, scratch)

    def neumann_diag_power2_update(self, negative_t, identity, scratch):
        self.half_l1_power_packed64_diag_only_on_c(negative_t, negative_t, self.tile_inv_l1, clear_l0=True)
        power2 = self.tile_inv_l1
        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only_db_pair(
            scratch, identity, scratch, power2, c_inv, init_c0=True, unit_flag0=2, init_c1=False, unit_flag1=3,
            clear_l0=False,
        )
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return (power2, self.packed_inv64_l1)

    def neumann_diag_power4_update(self, power2, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power2, power2, self.tile_power_l1, clear_l0=False)
        power4 = self.tile_power_l1
        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power4, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return (power4, self.packed_inv64_l1)

    def neumann_diag_power8_update(self, power4, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(power4, power4, self.tile_inv_l1, clear_l0=False)
        power8 = self.tile_inv_l1
        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(packed_inv, power8, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap(
            scratch, l0a_write, src0_row_block=1, src0_col_block=0, src1_row_block=3, src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            packed_inv, l0b_write, src0_row_block=0, src0_col_block=0, src1_row_block=2, src1_col_block=2,
            dst_offset_elems=self.c64_diag_nz_offset_elems(0),
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def issue_odd_inv_odd_tmp_full64(self, packed_inv, tmp64):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 1, 1), l0a_write, 1, 1)
        self._load_l1_block_to_l0a_block(self.packed64_block(packed_inv, 3, 3), l0a_write, 3, 3)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            tmp64, l0b_write, src0_row_block=1, src0_col_block=0, src1_row_block=3, src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def compose_odd_even_lower16_to32_accum_full64(self, scratch, packed_inv):
        """组合两个 32x32 对角子块内部的 lower 逆矩。"""
        self.issue_odd_lower_even_inv_full64(scratch, packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1
        self.issue_odd_inv_odd_tmp_full64(packed_inv, tmp64)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd32_lower_even32_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, scratch)
        self._load_packed32_block_to_full64_l0b(
            packed_inv, l0b_write, src_row=0, src_col=0, dst_row=0, dst_col=0, transpose=True,
        )

    def prepare_odd32_inv_odd32_tmp_full64(self, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, packed_inv)
        return (l0a_write, l0b_write)

    def complete_odd32_inv_odd32_tmp_full64(self, tmp64, l0a_write, l0b_write):
        self._load_packed32_block_to_full64_l0b(
            tmp64, l0b_write, src_row=HALF_CHUNK_SIZE, src_col=0, dst_row=HALF_CHUNK_SIZE, dst_col=0, transpose=True,
        )

    def compose_odd_even_lower32_to64_accum_full64(self, scratch, packed_inv):
        """组合 64x64 左下 32x32 lower 逆矩，保留 FP32 累加器供两次写出。"""
        self.issue_odd32_lower_even32_inv_full64(scratch, packed_inv)
        second_l0a, second_l0b = self.prepare_odd32_inv_odd32_tmp_full64(packed_inv)
        c_tmp = self.raw_l0c_aux
        self.begin_mmad_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1
        self.complete_odd32_inv_odd32_tmp_full64(tmp64, second_l0a, second_l0b)
        c_inv = self.raw_l0c_main
        self.begin_mmad_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)

    @jit
    def store_inverse(self, gm_saved, head_idx, chunk_idx, valid_rows, save_matrices):
        if save_matrices != 0:
            saved = tile_slice(gm_saved[0, head_idx, chunk_idx, None, None], (CHUNK_SIZE, CHUNK_SIZE), (0, 0))
            mem_copy(
                _valid_rows_view(saved, valid_rows, CHUNK_SIZE), self.raw_l0c_main[:valid_rows, :CHUNK_SIZE],
                options=copy_options(unit_flag=2, l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )
        self.fixpipe_packed64_full_to_bf16_l1(self.resident_inv_l1, self.raw_l0c_main)

    def compute_u_pre_and_w_from_resident_inv_beta(
        self, gm_u, gm_w, head_idx, chunk_idx, valid_rows, v_beta_l1, k_decayed_beta_l1
    ):
        l0a_resident = self._load_resident_inv_to_l0a()
        l0b_beta = self._load_beta_to_l0b(k_decayed_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        gm_W_chunk = tile_slice(gm_w[0, head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        mem_copy(
            _valid_rows_view(gm_W_chunk, valid_rows, SUPPORTED_HEAD_DIM), (self.l0c_db.next_read())[:valid_rows, :SUPPORTED_HEAD_DIM],
            options=copy_options(l2_cache_ctl=_STAGE2_INPUT_L2_CACHE_CTL),
        )
        l0b_beta = self._load_beta_to_l0b(v_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        u_pre_part = tile_slice(gm_u[0, head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        self.store_l0c_to_gm(_valid_rows_view(u_pre_part, valid_rows, SUPPORTED_HEAD_DIM))

    def _raw_l0a_block(self, l0a_slot, row_block: int, col_block: int):
        return from_tensor(l0a_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE, l0a_slot.physical_layout.stride))

    def _raw_l0b_block(self, l0b_slot, row_block: int, col_block: int):
        return from_tensor(l0b_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE, l0b_slot.physical_layout.stride))

    def _load_l1_block_to_l0a_block(
        self, l1_block, l0a_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 装入 full64 L0A 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0a_block(l0a_slot, row_block, col_block)
        l1_tile = tile_slice(l1_parent, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        mem_copy(l0_tile, l1_tile)

    def _load_l1_block_to_l0b_block(
        self, l1_block, l0b_slot, row_block: int, col_block: int,
    ):
        """把一个 packed L1 16x16 fractal 转置装入 full64 L0B 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0b_block(l0b_slot, row_block, col_block)
        l1_tile = tile_slice(l1_parent, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (row_block, col_block))
        mem_copy(l0_tile, (l1_tile).permute(1, 0))

    def _load_packed32_block_to_full64_l0b(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = True,
    ):
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems(
                (dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE,
            )
            l1_tile = from_tensor(l1_matrix, layout=make_layout((NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)), offset=l1_offset * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE, l1_matrix.physical_layout.stride))
            if const_expr(transpose):
                mem_copy(
                    from_tensor(l0b_slot, layout=make_layout((HALF_CHUNK_SIZE, NEUMANN_BLOCK_SIZE)), offset=l0_offset * 2, data_format="nz"),
                    (l1_tile).permute(1, 0,),
                )
            else:
                l0_tile = from_tensor(l0b_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)), offset=l0_offset * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE, l0b_slot.physical_layout.stride))
                mem_copy(l0_tile, l1_tile)

    def _load_packed32_block_to_full64_l0a(
        self,
        l1_matrix,
        l0a_slot,
        *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = False,
    ):
        """把 packed32 L1 block 装入 full64 L0A。"""
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_matrix_block_offset_elems(
                (dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE,
            )
            l0_offset -= 6 * NEUMANN_FRACTAL_ELEMS
            l0_tile = from_tensor(l0a_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)), offset=l0_offset * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE, l0a_slot.physical_layout.stride))
            l1_tile = from_tensor(l1_matrix, layout=make_layout((NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)), offset=l1_offset * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE, l1_matrix.physical_layout.stride))
            if const_expr(transpose):
                mem_copy(l0_tile, (l1_tile).permute(1, 0))
            else:
                mem_copy(l0_tile, l1_tile)

    def _load_packed16_pair_to_full64_l0a_with_gap(
        self,
        l1_matrix,
        l0a_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 装入 full64 L0A，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        for src_row_block, src_col_block, dst_block in (
            (src0_row_block, src0_col_block, dst0_block),
            (src1_row_block, src1_col_block, dst1_block),
        ):
            # 3510 L0A 的 16x16 单块地址仍需补偿一个 full64 M span。
            dst_offset = (dst_block - (NEUMANN_DIAG_BLOCK_NUM - 1)) * NEUMANN_FRACTAL_ELEMS
            l0_tile = from_tensor(l0a_slot, layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)), offset=dst_offset * 2, physical_layout=nz_layout(NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE, l0a_slot.physical_layout.stride))
            l1_tile = tile_slice(l1_matrix, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), (src_row_block, src_col_block))
            mem_copy(l0_tile, l1_tile)

    def _load_packed16_pair_to_full64_l0b_trans_with_gap(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src0_row_block: int,
        src0_col_block: int,
        src1_row_block: int,
        src1_col_block: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        """把一对 packed16 L1 block 转置装入 full64 L0B，目标块之间保留 gap。"""
        dst0_block = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_block = dst0_block + dst_gap_fractals + 1
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src0_row_block, src0_col_block), l0b_slot,
            dst0_block // NEUMANN_DIAG_BLOCK_NUM, dst0_block % NEUMANN_DIAG_BLOCK_NUM,
        )
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src1_row_block, src1_col_block), l0b_slot,
            dst1_block // NEUMANN_DIAG_BLOCK_NUM, dst1_block % NEUMANN_DIAG_BLOCK_NUM,
        )

    def _load_resident_inv_to_l0a(self):
        destination_2 = self.l0a_db.next_write()
        destination_2 = destination_2[tuple(map(slice, self.resident_inv_l1.shape))]
        mem_copy(destination_2, self.resident_inv_l1)
        return destination_2

    def _load_beta_to_l0b(self, beta_ring_buffer):
        destination_3 = self.beta_l0b
        destination_3_source = beta_ring_buffer.permute(1, 0)
        destination_3 = destination_3[tuple(map(slice, destination_3_source.shape))]
        mem_copy(destination_3, destination_3_source)
        return destination_3

    def _mm_resident_inv_beta(self, l0a_resident, l0b_beta):
        l0c_write = self.l0c_db.next_write()
        matmul(l0c_write, l0a_resident[0:CHUNK_SIZE, 0:CHUNK_SIZE], l0b_beta, init=True)


    @jit
    def compute_shared_per_d_scores(self, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf):
        self.pack_shared_score_operands(vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf)
        self.issue_cached_score_mmads()


    @jit
    def pack_shared_score_operands(self, vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf):
        vector.pack_shared_score_panel(normalized_q_dhalf, normalized_k_dhalf, gate_cumsum_dhalf, 0)
        vector.pack_shared_score_panel(normalized_q_dhalf, normalized_k_dhalf, gate_cumsum_dhalf, 1)
        destination = self.l1_shared_score_cache.partition(dim=1, part_id=vector.subblock_idx)
        mem_copy(destination, vector.ub_shared_score_nz[: destination.shape[0], : destination.shape[1]])

    @jit
    def issue_cached_score_mmads(self):
        for index in range_constexpr(6):
            right_source = tile_slice(self.l1_shared_score_cache, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (8 + index, 0))
            mem_copy(self.shared_rhs_cache[index], right_source)
        for kind in range_constexpr(2):
            for row_block in range_constexpr(4):
                left_source = tile_slice(
                    self.l1_shared_score_cache, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (2 * row_block + kind, 0)
                )
                if const_expr(kind == 0):
                    mem_copy(self.per_d_left_k_l0a, left_source)
                else:
                    mem_copy(self.per_d_left_q_l0a, left_source)
                for col_block in range_constexpr(row_block + 1):
                    right = self.shared_rhs_cache[col_block if row_block < 2 else 2 + col_block]
                    if const_expr(kind == 0):
                        self.mm_per_d_kkt(row_block, col_block, right)
                    else:
                        self.mm_per_d_mqk(row_block, col_block, right)


@jit
def _copy_padded_normalized_k(dst, src):
    """保存全部已初始化行的快照，包括尾部补零行。

    原始写入初始化填充区，但不扩大先前 DMA 的有效范围。
    显式使用完整分块视图，包含延后计算所需的填充区。"""
    full_tile = src[0:CHUNK_SIZE, 0:D_HALF]
    mem_copy(dst, full_tile)


class StageOneVector:
    """Vector 侧标量、mask、搬运和状态更新封装。"""

    def __init__(self, *, scale_value, subblock_idx):
        self.scale_value = scale_value
        self.subblock_idx = subblock_idx
        self.ub_q = [Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF))) for _ in range(2)]
        self.ub_k = [Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF))) for _ in range(2)]
        self.ub_qk_exchange = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF)))
        self.ub_g_raw = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_gate_activated = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF)))
        self.ub_dt_bias = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((D_HALF,)), depth=1).next_write()
        self.ub_alpha = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1,)), depth=1).next_write()
        self.ub_beta_raw = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, 1)), depth=1).next_write()
        self.ub_beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, 1)))
        self.ub_v_raw = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_gamma_c = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((1, D_HALF)), depth=1).next_write()
        self.ub_cs_snapshot = [Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, D_HALF))) for _ in range(2)]
        self.ub_mqk = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.ub_bf16_nz = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, D_HALF)),
            depth=2,
            physical_layout=make_layout(
                ((D_HALF + 16 - 1) // 16, (CHUNK_SIZE + 15) // 16, 16, 16),
                stride=((CHUNK_SIZE + 15) // 16 * 256 + 16, 256, 16, 1),
            ),
            data_format="nz",
        )
        self.ub_cast_bf16 = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF)), depth=1).next_write()
        self.ub_kkt_fp16 = Buffer(MemLoc.UB, dtypes.float16, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)))
        self.ub_mqk_bf16 = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.ub_raw_k = [Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF))) for _ in range(2)]
        self.ub_kkt = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.ub_kkt_nz = RingBuffer(
            MemLoc.UB,
            dtypes.float16,
            make_layout((HALF_CHUNK_SIZE, CHUNK_SIZE)),
            depth=1,
            physical_layout=make_layout(
                ((CHUNK_SIZE + 16 - 1) // 16, (HALF_CHUNK_SIZE + 15) // 16, 16, 16),
                stride=((HALF_CHUNK_SIZE + 15) // 16 * 256 + 16, 256, 16, 1),
            ),
            data_format="nz",
        ).next_write()
        self.ub_per_d_nd = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, D_HALF)), depth=2)
        self.ub_per_d_nz = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((NEUMANN_BLOCK_SIZE, D_HALF)),
            depth=2,
            physical_layout=make_layout(
                ((D_HALF + 16 - 1) // 16, (NEUMANN_BLOCK_SIZE + 15) // 16, 16, 16),
                stride=((NEUMANN_BLOCK_SIZE + 15) // 16 * 256 + 16, 256, 16, 1),
            ),
            data_format="nz",
        )
        self.ub_safe_left_k = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, D_HALF)))
        self.ub_safe_left_q = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, D_HALF)))
        self.ub_safe_right_k = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, D_HALF)))
        self.l1_per_d_operand = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)), depth=2
        )
        self.l1_V_beta = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=1
        ).next_write()
        self.l1_K_decayed_beta_for_W = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=1
        ).next_write()
        self.ub_shared_score_nz = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((14 * NEUMANN_BLOCK_SIZE, D_HALF)),
            depth=1,
            physical_layout=nz_layout(STABLE_SCORE_ROWS, D_HALF),
            data_format="nz",
        ).next_write()

    @staticmethod
    def _copy_bf16_gm_half_tile_to_ub(
        gm_tile, ub_ring_buffer, *, cols: int = SUPPORTED_HEAD_DIM
    ):
        mem_copy(ub_ring_buffer, gm_tile)

    def scratch_half_tile(self, gm_scratch, block_idx, cols: int):
        return tile_slice(gm_scratch, (HALF_CHUNK_SIZE, cols), (block_idx * 2 + self.subblock_idx, 0))

    def scratch_dhalf_tile(self, gm_scratch, block_idx):
        return tile_slice(gm_scratch, (CHUNK_SIZE, D_HALF), (block_idx, self.subblock_idx))

    @jit
    def publish_qk_partials(self, exchange, outgoing_partials):
        """在独立的 gate 计算前发布当前 AIV 的 Q/K 部分和。"""
        vec_sync_notify(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        mem_copy(exchange.flatten(), outgoing_partials.flatten())
        vec_sync_block_arrive(PIPE.MTE3, QK_PUBLISH_FLAG_ID, mode=1)

    @jit
    def issue_qk_partials(self, exchange, incoming_partials):
        """在独立的 gate 计算前发起对端 Q/K 部分和的接收。"""
        vec_sync_block_wait(PIPE.MTE2, QK_PUBLISH_FLAG_ID, mode=1)
        mem_copy(incoming_partials.flatten(), exchange.flatten())
        vec_sync_notify(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)

    @jit
    def finish_issued_qk_partials(self):
        """等待对端部分和并完成跨 AIV 握手。"""
        vec_sync_wait(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)
        vec_sync_block_arrive(PIPE.MTE2, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_block_wait(PIPE.MTE3, QK_CONSUME_FLAG_ID, mode=1)

    @jit
    def compute_qk_partial_sums(self, q_half, k_half, partials):
        """将当前 AIV 的 Q/K 半分块按 token 行归约为 FP32 和。"""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                q_unpacked = vload_unpack(q_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(q_unpacked, dtypes.float32, mask=lane_mask)
                k_unpacked = vload_unpack(k_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_unpacked, dtypes.float32, mask=lane_mask)
                q_partial = vreduce_sum(vmul(q_value, q_value, mask=lane_mask), mask=lane_mask)
                k_partial = vreduce_sum(vmul(k_value, k_value, mask=lane_mask), mask=lane_mask)
                vstore_first(partials, row, q_partial)
                vstore_first(partials, CHUNK_SIZE + row, k_partial)

    @jit
    def compute_qk_inverse_vectors(self, local_partials, peer_partials, inverse_norms):
        """通过两次向量操作计算全部 64 行 Q/K 范数的倒数。"""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            # 平方和非负，低精度硬件除法的舍入不能使倒数超过 1/sqrt(eps)。
            q_square_sum = vadd(vload(local_partials, 0), vload(peer_partials, 0), mask=lane_mask)
            q_inverse = vdiv(one, vsqrt(vadds(q_square_sum, L2_NORM_EPS, mask=lane_mask), mask=lane_mask), mask=lane_mask)
            vstore(inverse_norms, 0, vmins(q_inverse, MAX_INVERSE_NORM, mask=lane_mask), lane_mask)
            k_square_sum = vadd(vload(local_partials, CHUNK_SIZE), vload(peer_partials, CHUNK_SIZE), mask=lane_mask)
            k_inverse = vdiv(one, vsqrt(vadds(k_square_sum, L2_NORM_EPS, mask=lane_mask), mask=lane_mask), mask=lane_mask)
            vstore(inverse_norms, CHUNK_SIZE, vmins(k_inverse, MAX_INVERSE_NORM, mask=lane_mask), lane_mask)

    @jit
    def normalize_local_qk_from_inverse_vectors(self, q_half, k_half, inverse_norms):
        """使用打包的 Q/K 范数倒数向量归一化本地两个 BF16 半分块。"""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                q_unpacked = vload_unpack(q_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(q_unpacked, dtypes.float32, mask=lane_mask)
                q_inverse = vload_broadcast(inverse_norms, row)
                q_normalized = vmul(q_value, q_inverse, mask=lane_mask)
                q_bf16 = vcast(q_normalized, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(q_half, offset, q_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
                k_unpacked = vload_unpack(k_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(k_unpacked, dtypes.float32, mask=lane_mask)
                k_inverse = vload_broadcast(inverse_norms, CHUNK_SIZE + row)
                k_normalized = vmul(k_value, k_inverse, mask=lane_mask)
                k_bf16 = vcast(k_normalized, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(k_half, offset, k_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)

    @jit
    def prefetch_qk_local_halves(
        self,
        gm_Q,
        gm_K,
        batch_idx,
        qk_head_idx,
        chunk_idx,
        valid_rows,
        buffer_parity: int,
    ):
        """预取当前 AIV 的 Q/K 半分块，暂不发布事件。"""
        q_local_half = tile_slice(
            gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx),
        )
        k_local_half = tile_slice(
            gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx),
        )
        q_local_half = _valid_rows_view(q_local_half, valid_rows, D_HALF)
        k_local_half = _valid_rows_view(k_local_half, valid_rows, D_HALF)
        mem_copy(self.ub_q[buffer_parity], q_local_half)
        mem_copy(self.ub_k[buffer_parity], k_local_half)

    @jit
    def notify_qk_local_halves_ready(self):
        if self.subblock_idx == 0:
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        else:
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)

    @jit
    def issue_qk_local_halves(
        self,
        gm_Q,
        gm_K,
        batch_idx,
        qk_head_idx,
        chunk_idx,
        valid_rows,
        buffer_parity: int,
    ):
        self.prefetch_qk_local_halves(gm_Q, gm_K, batch_idx, qk_head_idx, chunk_idx, valid_rows, buffer_parity)
        self.notify_qk_local_halves_ready()

    @jit
    def finish_and_publish_qk_partials(
        self, gm_qk_exchange, block_idx, valid_rows, buffer_parity: int
    ):
        """归约本地半分块，并发布两组部分和向量。"""
        if self.subblock_idx == 0:
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        else:
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
        q_half = self.ub_q[buffer_parity]
        k_half = self.ub_k[buffer_parity]
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16_invalid_rows(q_half, valid_rows=valid_rows, cols=D_HALF, value=0.0)
            self._fill_bf16_invalid_rows(k_half, valid_rows=valid_rows, cols=D_HALF, value=0.0)
        local_partials = self.ub_qk_exchange.flatten()[0:2 * CHUNK_SIZE,].reshape((2, CHUNK_SIZE))
        self.compute_qk_partial_sums(q_half, k_half, local_partials)
        local_partials = self.ub_qk_exchange.flatten()[0:2 * CHUNK_SIZE,].reshape((2, CHUNK_SIZE))
        if self.subblock_idx == 0:
            self.publish_qk_partials(gm_qk_exchange[block_idx, 0, None, None], local_partials)
        else:
            self.publish_qk_partials(gm_qk_exchange[block_idx, 1, None, None], local_partials)

    @jit
    def issue_peer_qk_partials(self, gm_qk_exchange, block_idx):
        """开始接收对端 AIV 的两组部分和向量。"""
        incoming_partials = self.ub_qk_exchange.flatten()[2 * CHUNK_SIZE:2 * CHUNK_SIZE + 2 * CHUNK_SIZE,].reshape((2, CHUNK_SIZE))
        if self.subblock_idx == 0:
            self.issue_qk_partials(gm_qk_exchange[block_idx, 1, None, None], incoming_partials)
        else:
            self.issue_qk_partials(gm_qk_exchange[block_idx, 0, None, None], incoming_partials)

    @jit
    def finish_qk_partial_exchange(self, buffer_parity: int, use_qk_l2norm_in_kernel):
        """完成一轮交换，并归一化本地两个半分块。"""
        self.finish_issued_qk_partials()
        local_partials = self.ub_qk_exchange
        peer_partials = self.ub_qk_exchange.flatten()[2 * CHUNK_SIZE:2 * CHUNK_SIZE + 2 * CHUNK_SIZE,].reshape((2, CHUNK_SIZE))
        inverse_norms = self.ub_qk_exchange.flatten()[4 * CHUNK_SIZE:4 * CHUNK_SIZE + 2 * CHUNK_SIZE,].reshape((2, CHUNK_SIZE))
        q_half = self.ub_q[buffer_parity]
        k_half = self.ub_k[buffer_parity]
        if use_qk_l2norm_in_kernel != 0:
            self.compute_qk_inverse_vectors(local_partials, peer_partials, inverse_norms)
            self.normalize_local_qk_from_inverse_vectors(q_half, k_half, inverse_norms)
        return (k_half, q_half)

    @jit
    def store_normalized_qk(
        self,
        gm_q,
        gm_k,
        gm_q_rstd,
        gm_k_rstd,
        q_half,
        k_half,
        head_idx,
        chunk_idx,
        valid_rows,
    ):
        q_tile = tile_slice(gm_q[0, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        k_tile = tile_slice(gm_k[0, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        mem_copy(
            _valid_rows_view(q_tile, valid_rows, D_HALF), q_half[:valid_rows, :D_HALF],
            options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
        )
        mem_copy(
            _valid_rows_view(k_tile, valid_rows, D_HALF), k_half[:valid_rows, :D_HALF],
            options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
        )
        if self.subblock_idx == 0:
            q_rstd = self.ub_qk_exchange.flatten()[4 * CHUNK_SIZE:4 * CHUNK_SIZE + CHUNK_SIZE * 1,].reshape((CHUNK_SIZE, 1))
            k_rstd = self.ub_qk_exchange.flatten()[5 * CHUNK_SIZE:5 * CHUNK_SIZE + CHUNK_SIZE * 1,].reshape((CHUNK_SIZE, 1))
            q_dst = tile_slice(gm_q_rstd[0, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, 0))
            k_dst = tile_slice(gm_k_rstd[0, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, 0))
            mem_copy(
                _valid_rows_view(q_dst, valid_rows, 1), q_rstd[:valid_rows, :1],
                options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )
            mem_copy(
                _valid_rows_view(k_dst, valid_rows, 1), k_rstd[:valid_rows, :1],
                options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )

    @jit
    def store_beta(self, gm_beta, head_idx, chunk_idx, valid_rows):
        if self.subblock_idx == 0:
            dst = tile_slice(gm_beta[0, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, 0))
            mem_copy(
                _valid_rows_view(dst, valid_rows, 1), self.ub_beta[:valid_rows, :1],
                options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )

    @jit
    def store_saved_mqk(self, gm_saved, head_idx, chunk_idx, valid_rows):
        row_start = self.subblock_idx * HALF_CHUNK_SIZE
        if row_start < valid_rows:
            rows = valid_rows - row_start
            if rows > HALF_CHUNK_SIZE:
                rows = HALF_CHUNK_SIZE
            dst = tile_slice(
                gm_saved[0, head_idx, None, None], (HALF_CHUNK_SIZE, CHUNK_SIZE),
                (chunk_idx * 2 + self.subblock_idx, 0),
            )
            mem_copy(
                _valid_rows_view(dst, rows, CHUNK_SIZE), self.ub_mqk_bf16[:rows, :CHUNK_SIZE],
                options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )

    @jit
    def issue_raw_gate(
        self,
        gm_g,
        gm_A_log,
        gm_dt_bias,
        batch_idx,
        head_idx,
        chunk_idx,
        valid_rows,
        use_gate_in_kernel,
    ):
        """在 Q/K 归一化和交换前发起原始 gate 输入搬运。"""
        g_chunk = tile_slice(
            gm_g[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx),
        )
        g_chunk = _valid_rows_view(g_chunk, valid_rows, D_HALF)
        mem_copy(self.ub_g_raw, g_chunk)
        if use_gate_in_kernel != 0:
            # 预处理路径不读取这两个 GM；无 bias 的融合路径只读取 A_log。
            alpha = tile_slice(gm_A_log, (1,), (head_idx,))
            if use_gate_in_kernel == GATE_MODE_FUSED_WITH_DT_BIAS:
                bias_half = tile_slice(gm_dt_bias[head_idx, None], (D_HALF,), (self.subblock_idx,))
                mem_copy(self.ub_dt_bias, bias_half)
            mem_copy(self.ub_alpha, alpha)
        vec_sync_notify(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)

    @jit
    def activate_issued_gate(self, lower_bound, valid_rows, buffer_parity: int, use_gate_in_kernel, has_lower_bound):
        """等待已发起的 gate 输入，完成激活后计算前缀和。"""
        vec_sync_wait(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)
        g_slot = self.ub_g_raw
        rows_to_activate = valid_rows
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        if use_gate_in_kernel != 0:
            if has_lower_bound != 0:
                if use_gate_in_kernel == GATE_MODE_FUSED_WITH_DT_BIAS:
                    self._vf_activate_bounded_gate(g_slot, rows_to_activate, lower_bound, True, gate_cumsum_snapshot)
                else:
                    self._vf_activate_bounded_gate(g_slot, rows_to_activate, lower_bound, False, gate_cumsum_snapshot)
            elif use_gate_in_kernel == GATE_MODE_FUSED_WITH_DT_BIAS:
                self._vf_activate_softplus_gate(g_slot, rows_to_activate, True, gate_cumsum_snapshot)
            else:
                self._vf_activate_softplus_gate(g_slot, rows_to_activate, False, gate_cumsum_snapshot)
        else:
            with vf(mode="simd"):
                lane_mask, _ = update_mask(D_HALF, elem_bits=32)
                prefix = vdups(0.0, dtypes.float32, mask=lane_mask)
                for row in dsl_range(0, rows_to_activate, 1, unroll=4):
                    offset = row * D_HALF
                    unpacked_g = vload_unpack(g_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                    gate = vcast(unpacked_g, dtypes.float32, mask=lane_mask)
                    prefix = vadd(prefix, gate, mask=lane_mask)
                    vstore(gate_cumsum_snapshot, offset, prefix, lane_mask)
                for row in dsl_range(rows_to_activate, CHUNK_SIZE, 1, unroll=4):
                    vstore(gate_cumsum_snapshot, row * D_HALF, prefix, lane_mask)
        return gate_cumsum_snapshot

    @jit
    def issue_raw_beta(self, gm_beta, batch_idx, head_idx, chunk_idx, valid_rows):
        """在按维度计算前，发起一个原始 FP32 beta-logit chunk 的搬运。"""
        beta_chunk = tile_slice(gm_beta[batch_idx, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, 0))
        beta_chunk = _valid_rows_view(beta_chunk, valid_rows, 1)
        mem_copy(self.ub_beta_raw, beta_chunk)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)

    @jit
    def activate_issued_beta(
        self, valid_rows, use_beta_sigmoid_in_kernel, allow_neg_eigval
    ):
        """进入 VF 前静态选择 beta 计算公式。"""
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        if use_beta_sigmoid_in_kernel != 0:
            if allow_neg_eigval != 0:
                self._activate_beta_values(valid_rows, True, True)
            else:
                self._activate_beta_values(valid_rows, True, False)
        else:
            self._activate_beta_values(valid_rows, False, False)
        return self.ub_beta

    @jit
    def apply_beta_and_finish_t_for_neumann(
        self, beta_slot, power_handoff_l1, subblock_base: int, buffer_parity: int
    ):
        row_base = subblock_base * HALF_CHUNK_SIZE
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                row_off = row * CHUNK_SIZE
                t_true = vload(self.ub_kkt, row_off)
                beta_brc = vload_broadcast(beta_slot, row_base + row)
                t_beta = vmul(t_true, beta_brc, mask=row_mask)
                tril_mask, _ = update_mask(row_base + row, elem_bits=32)
                t_tril = vselect_raw(t_beta, zero_reg, cond_mask=tril_mask)
                neg_t = vmuls(t_tril, -1.0, mask=row_mask)
                neg_t_fp16 = vcast(neg_t, dtypes.float16, mask=row_mask)
                vstore_pack(self.ub_kkt_fp16, row_off, neg_t_fp16, row_mask, pack_mode="b32_to_b16")
        nd2nz_tile(self.ub_kkt_nz, self.ub_kkt_fp16)
        self._store_neumann_tmp_to_handoff_l1(power_handoff_l1, subblock_base)

    @jit
    def issue_beta_weighted_v(self, gm_V, batch_idx, head_idx, chunk_idx, valid_rows):
        """提前一个任务发起 V 搬运，供 C3 向量变换使用。"""
        v_half = tile_slice(gm_V[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        v_half = _valid_rows_view(v_half, valid_rows, D_HALF)
        self._copy_bf16_gm_half_tile_to_ub(v_half, self.ub_v_raw, cols=D_HALF)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)

    @jit
    def prepare_beta_weighted_c3_operands(
        self, beta_slot, buffer_parity: int, valid_rows
    ):
        """等待上一任务预取的 V 就绪后，构建 C3 操作数。"""
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        self._finish_beta_weighted_c3_operands(
            self.ub_v_raw, beta_slot, raw_k_snapshot, gate_cumsum_snapshot, valid_rows,
        )

    @jit
    def prepare_normalized_chunk_inputs(
        self,
        normalized_k_dhalf,
        normalized_q_dhalf,
        gate_cumsum_dhalf,
        buffer_parity: int,
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_tile = gate_cumsum_dhalf[0:CHUNK_SIZE, 0:D_HALF]
        _copy_padded_normalized_k(raw_k_snapshot, normalized_k_dhalf)
        cast_write = self.ub_cast_bf16
        qg_write = self.ub_gate_activated.view(dtype=dtypes.bfloat16).flatten()[0:CHUNK_SIZE * D_HALF,].reshape((CHUNK_SIZE, D_HALF))
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                gamma = vload(gate_cumsum_tile, offset)
                exp_value = vexp(gamma, mask=row_mask)
                raw_q = vload_unpack(normalized_q_dhalf, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(raw_q, dtypes.float32, mask=row_mask)
                qg_value = vmul(q_value, exp_value, mask=row_mask)
                qg_bf16 = vcast(qg_value, dtypes.bfloat16, mask=row_mask)
                vstore_pack(qg_write, offset, qg_bf16, row_mask, pack_mode="b32_to_b16")
                q_decayed = vmuls(qg_value, self.scale_value, mask=row_mask)
                q_decayed_bf16 = vcast(q_decayed, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, q_decayed_bf16, row_mask, pack_mode="b32_to_b16")
        return (
            self.ub_cast_bf16,
            gate_cumsum_dhalf,
            normalized_k_dhalf,
            normalized_q_dhalf,
        )

    @jit
    def store_g_cumsum(
        self, gm_g_cumsum, gate_cumsum_dhalf, head_idx, chunk_idx, valid_rows
    ):
        g_dst = tile_slice(gm_g_cumsum[0, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        mem_copy(
            _valid_rows_view(g_dst, valid_rows, D_HALF), gate_cumsum_dhalf[:valid_rows, :D_HALF],
            options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
        )

    @jit
    def store_qg(self, gm_qg, head_idx, chunk_idx, valid_rows):
        qg_dst = tile_slice(gm_qg[0, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        qg_read = self.ub_gate_activated.view(dtype=dtypes.bfloat16).flatten()[0:CHUNK_SIZE * D_HALF,].reshape((CHUNK_SIZE, D_HALF))
        mem_copy(
            _valid_rows_view(qg_dst, valid_rows, D_HALF), qg_read[:valid_rows, :D_HALF],
            options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
        )

    @jit
    def finish_mqk(self, gm_Aqk, head_idx, chunk_idx, valid_rows, buffer_parity: int):
        row_start = self.subblock_idx * HALF_CHUNK_SIZE
        rows = valid_rows - row_start
        if rows > HALF_CHUNK_SIZE:
            rows = HALF_CHUNK_SIZE
        self._store_masked_mqk_to_bf16_gm(gm_Aqk, head_idx, chunk_idx, rows, buffer_parity)
        return self.ub_cs_snapshot[buffer_parity][CHUNK_SIZE - 1:CHUNK_SIZE, None]

    def store_q_decayed_scratch(self, gm_Q_decayed_scratch, block_idx, q_decayed_slot):
        Q_decayed_slot_part = self.scratch_dhalf_tile(gm_Q_decayed_scratch, block_idx)
        mem_copy(Q_decayed_slot_part,
                 q_decayed_slot[:Q_decayed_slot_part.shape[0], :Q_decayed_slot_part.shape[1]])

    @jit
    def store_restored_state_inputs(
        self,
        gm_kg,
        gm_gamma_C_scratch,
        block_idx,
        gamma_c_slot,
        buffer_parity: int,
        head_idx,
        chunk_idx,
        valid_rows,
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        K_restored_dhalf = tile_slice(
            gm_kg[0, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx),
        )
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            cs_last = vload(gamma_c_slot, 0)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                raw_k = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(raw_k, dtypes.float32, mask=row_mask)
                gate_cumsum = vload(gate_cumsum_snapshot, offset)
                restore_scale = vexp_sub(cs_last, gate_cumsum, mask=row_mask)
                restored = vmul(k_value, restore_scale, mask=row_mask)
                restored_bf16 = vcast(restored, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, restored_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        mem_copy(
            _valid_rows_view(K_restored_dhalf, valid_rows, D_HALF), cast_read[:valid_rows, None],
            options=copy_options(l2_cache_ctl=_STAGE2_INPUT_L2_CACHE_CTL),
        )
        gamma_c_write = self.ub_gamma_c
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            gamma_c = vload(gamma_c_slot, 0)
            gamma_c_exp = vexp(gamma_c, mask=row_mask)
            vstore(gamma_c_write, 0, gamma_c_exp, row_mask)
        gamma_C_gm = tile_slice(gm_gamma_C_scratch, (D_HALF, 1), (block_idx * 2 + self.subblock_idx, 0))
        gamma_c_read = self.ub_gamma_c
        mem_copy(gamma_C_gm.view(1, D_HALF), gamma_c_read)

    @jit
    def build_per_d_left_k_and_decay(
        self, normalized_k_dhalf, gate_cumsum_dhalf, row_block: int, left_k_l1_operand
    ):
        k_read = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        decay_scratch = tile_slice(self.ub_qk_exchange, (NEUMANN_BLOCK_SIZE, D_HALF), (0, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF
        nd_write = self.ub_per_d_nd.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (row_block * NEUMANN_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(gate_cumsum, ref, mask=mask)
                vstore(decay_scratch, offset, decay, mask)
                k_decayed = vmul(k, decay, mask=mask)
                k_decayed_bf16 = vcast(k_decayed, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, k_decayed_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.next_read()
        self._write_per_d_full_d_handoff(left_k_l1_operand, nd_read)

    @jit
    def build_per_d_left_q_from_decay(
        self, normalized_q_dhalf, row_block: int, left_q_l1_operand
    ):
        q_read = tile_slice(normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        decay_scratch = tile_slice(self.ub_qk_exchange, (NEUMANN_BLOCK_SIZE, D_HALF), (0, 0))
        nd_write = self.ub_per_d_nd.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                unpacked_q = vload_unpack(q_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q = vcast(unpacked_q, dtypes.float32, mask=mask)
                decay = vload(decay_scratch, offset)
                q_decayed = vmul(q, decay, mask=mask)
                scaled_q = vmuls(q_decayed, self.scale_value, mask=mask)
                scaled_q_bf16 = vcast(scaled_q, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, scaled_q_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.next_read()
        self._write_per_d_full_d_handoff(left_q_l1_operand, nd_read)

    @jit
    def build_per_d_right(
        self,
        normalized_k_dhalf,
        gate_cumsum_dhalf,
        row_block: int,
        col_block: int,
        right_k_l1_operand,
    ):
        k_read = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (col_block, 0))
        ref_offset = row_block * NEUMANN_BLOCK_SIZE * D_HALF
        nd_write = self.ub_per_d_nd.next_write()
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (col_block * NEUMANN_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(ref, gate_cumsum, mask=mask)
                restored_k = vmul(k, decay, mask=mask)
                restored_k_bf16 = vcast(restored_k, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, restored_k_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_per_d_nd.next_read()
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_read)

    @jit
    def prepare_safe_diagonal_left(self):
        """每个对角块仅初始化一次两个持久行掩码。"""
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            zero = vcast(vdups(0.0, dtypes.float32, mask=mask), dtypes.bfloat16, mask=mask)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1):
                offset = row * D_HALF
                vstore_pack(self.ub_safe_left_k, offset, zero, mask, pack_mode="b32_to_b16")
                vstore_pack(self.ub_safe_left_q, offset, zero, mask, pack_mode="b32_to_b16")
                vstore_pack(self.ub_safe_right_k, offset, zero, mask, pack_mode="b32_to_b16")

    @jit
    def update_safe_diagonal_left(
        self,
        normalized_k_dhalf,
        normalized_q_dhalf,
        row_block: int,
        active_row,
        left_l1_operand,
    ):
        k_source = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        q_source = tile_slice(normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        if active_row > 0:
            self._update_safe_diagonal_left_values(k_source, q_source, active_row, left_l1_operand, True)
        else:
            self._update_safe_diagonal_left_values(k_source, q_source, active_row, left_l1_operand, False)

    @jit
    def build_safe_diagonal_right(
        self,
        normalized_k_dhalf,
        gate_cumsum_dhalf,
        row_block: int,
        active_row,
        right_k_l1_operand,
    ):
        source = tile_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), (row_block, 0))
        nd_write = self.ub_safe_right_k
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, (row_block * NEUMANN_BLOCK_SIZE + active_row) * D_HALF)
            for row in dsl_range(0, active_row + 1, 1):
                offset = row * D_HALF
                unpacked_k = vload_unpack(source, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate = vload(gate_cumsum_dhalf, (row_block * NEUMANN_BLOCK_SIZE + row) * D_HALF)
                decay = vexp_sub(ref, gate, mask=mask)
                decayed_k = vmul(k, decay, mask=mask)
                decayed_bf16 = vcast(decayed_k, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, decayed_bf16, mask, pack_mode="b32_to_b16")
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_write)

    @jit
    def _fill_bf16_invalid_rows(self, ub_tile, *, valid_rows, cols: int, value: float):
        """用真实 BF16 常量初始化 [valid_rows, 64) 的无效行。"""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(cols, elem_bits=16)
            fill_bf16 = vdups(value, dtypes.bfloat16, mask=lane_mask)
            for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                vstore(ub_tile, row * cols, fill_bf16, lane_mask)

    @jit
    def _vf_activate_bounded_gate(self, g_slot, rows_to_activate, lower_bound, has_dt_bias, gate_cumsum_snapshot):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            bound = vdups(lower_bound, dtypes.float32, mask=lane_mask)
            alpha = vload_broadcast(self.ub_alpha, 0)
            alpha_exp = vexp(alpha, mask=lane_mask)
            neg_exp_a = vmuls(alpha_exp, -1.0, mask=lane_mask)
            dt_bias = vdups(0.0, dtypes.float32, mask=lane_mask)
            prefix = vdups(0.0, dtypes.float32, mask=lane_mask)
            if const_expr(has_dt_bias):
                dt_bias = vload(self.ub_dt_bias, 0)
            for row in dsl_range(0, rows_to_activate, 1, unroll=4):
                offset = row * D_HALF
                unpacked_g = vload_unpack(g_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                raw_g = vcast(unpacked_g, dtypes.float32, mask=lane_mask)
                biased_g = vadd(raw_g, dt_bias, mask=lane_mask)
                gate_logit = vmul(neg_exp_a, biased_g, mask=lane_mask)
                exp_gate_logit = vexp(gate_logit, mask=lane_mask)
                gate_denominator = vadd(one, exp_gate_logit, mask=lane_mask)
                gate = vdiv(bound, gate_denominator, mask=lane_mask)
                prefix = vadd(prefix, gate, mask=lane_mask)
                vstore(gate_cumsum_snapshot, offset, prefix, lane_mask)
            for row in dsl_range(rows_to_activate, CHUNK_SIZE, 1, unroll=4):
                vstore(gate_cumsum_snapshot, row * D_HALF, prefix, lane_mask)

    @jit
    def _vf_activate_softplus_gate(self, g_slot, rows_to_activate, has_dt_bias, gate_cumsum_snapshot):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            alpha = vload_broadcast(self.ub_alpha, 0)
            alpha_exp = vexp(alpha, mask=lane_mask)
            dt_bias = vdups(0.0, dtypes.float32, mask=lane_mask)
            prefix = vdups(0.0, dtypes.float32, mask=lane_mask)
            if const_expr(has_dt_bias):
                dt_bias = vload(self.ub_dt_bias, 0)
            for row in dsl_range(0, rows_to_activate, 1, unroll=4):
                offset = row * D_HALF
                unpacked_g = vload_unpack(g_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                raw_g = vcast(unpacked_g, dtypes.float32, mask=lane_mask)
                biased_g = vadd(raw_g, dt_bias, mask=lane_mask)
                abs_x = vabs(biased_g, mask=lane_mask)
                neg_abs_x = vmuls(abs_x, -1.0, mask=lane_mask)
                softplus_tail = vlog(vadd(one, vexp(neg_abs_x, mask=lane_mask), mask=lane_mask), mask=lane_mask)
                positive_x = vmaxs(biased_g, 0.0, mask=lane_mask)
                softplus = vadd(positive_x, softplus_tail, mask=lane_mask)
                gate = vmul(vmuls(alpha_exp, -1.0, mask=lane_mask), softplus, mask=lane_mask)
                prefix = vadd(prefix, gate, mask=lane_mask)
                vstore(gate_cumsum_snapshot, offset, prefix, lane_mask)
            for row in dsl_range(rows_to_activate, CHUNK_SIZE, 1, unroll=4):
                vstore(gate_cumsum_snapshot, row * D_HALF, prefix, lane_mask)

    @jit
    def _activate_beta_values(
        self, valid_rows, use_sigmoid: bool, allow_negative: bool
    ):
        beta_slot = self.ub_beta_raw
        with vf(mode="simd"):
            lane_mask, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            raw_beta = vload(beta_slot, 0)
            beta = raw_beta
            if const_expr(use_sigmoid):
                neg_beta = vmuls(raw_beta, -1.0, mask=lane_mask)
                exp_neg_beta = vexp(neg_beta, mask=lane_mask)
                sigmoid_denominator = vadd(one, exp_neg_beta, mask=lane_mask)
                beta = vdiv(one, sigmoid_denominator, mask=lane_mask)
                if const_expr(allow_negative):
                    beta = vmuls(beta, 2.0, mask=lane_mask)
            zero = vdups(0.0, dtypes.float32, mask=lane_mask)
            valid_mask, _ = update_mask(valid_rows, elem_bits=32)
            invalid_mask = mask_xor(lane_mask, valid_mask, exec_mask=lane_mask)
            vstore(self.ub_beta, 0, beta, valid_mask)
            vstore(self.ub_beta, 0, zero, invalid_mask)

    @jit
    def _finish_beta_weighted_c3_operands(
        self, v_slot, beta_slot, raw_k_snapshot, gate_cumsum_snapshot, valid_rows
    ):
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                cs_value = vload(gate_cumsum_snapshot, offset)
                raw_k = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(raw_k, dtypes.float32, mask=row_mask)
                decay = vexp(cs_value, mask=row_mask)
                k_decayed = vmul(k_value, decay, mask=row_mask)
                k_decayed_beta = vmul(k_decayed, beta, mask=row_mask)
                k_decayed_beta_bf16 = vcast(k_decayed_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, k_decayed_beta_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        nd2nz_tile(self.ub_bf16_nz.next_write(), cast_read)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_K_decayed_beta_for_W, self.ub_bf16_nz.next_read())
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16_invalid_rows(v_slot, valid_rows=valid_rows, cols=D_HALF, value=0.0)
        cast_write = self.ub_cast_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                raw_v = vload_unpack(v_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                v_value = vcast(raw_v, dtypes.float32, mask=row_mask)
                v_beta = vmul(v_value, beta, mask=row_mask)
                v_beta_bf16 = vcast(v_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, v_beta_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        nd2nz_tile(self.ub_bf16_nz.next_write(), cast_read)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_V_beta, self.ub_bf16_nz.next_read())

    def _store_dhalf_nz_ub_to_full_l1(self, l1_dst, ub_nz_src):
        copy_destination = l1_dst.partition(dim=1, part_id=self.subblock_idx)
        copy_source = ub_nz_src
        mem_copy(copy_destination, copy_source[:copy_destination.shape[0], :copy_destination.shape[1]])

    def _store_neumann_tmp_to_handoff_l1(self, l1_dst, subblock_base: int):
        copy_destination = l1_dst.partition(dim=0, part_id=subblock_base)
        copy_source = self.ub_kkt_nz
        mem_copy(copy_destination, copy_source[:copy_destination.shape[0], :copy_destination.shape[1]])

    @jit
    def _store_masked_mqk_to_bf16_gm(
        self, gm_Aqk, head_idx, chunk_idx, valid_rows, buffer_parity: int
    ):
        mqk_slot = self.ub_mqk.next_read()
        bf16_write = self.ub_mqk_bf16
        with vf(mode="simd"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                off = row * CHUNK_SIZE
                keep, _ = update_mask(self.subblock_idx * HALF_CHUNK_SIZE + row + 1, elem_bits=32)
                mqk = vload(mqk_slot, off)
                mqk_tril = vselect_raw(mqk, zero_reg, cond_mask=keep)
                mqk_bf16 = vcast(mqk_tril, dtypes.bfloat16, mask=row_mask)
                vstore_pack(bf16_write, off, mqk_bf16, row_mask, pack_mode="b32_to_b16")
        if valid_rows > 0:
            Aqk_part = tile_slice(
                gm_Aqk[0, head_idx, chunk_idx, None, None], (HALF_CHUNK_SIZE, CHUNK_SIZE), (self.subblock_idx, 0),
            )
            mem_copy(
                _valid_rows_view(Aqk_part, valid_rows, CHUNK_SIZE), self.ub_mqk_bf16[:valid_rows, :CHUNK_SIZE],
                options=copy_options(l2_cache_ctl=_STAGE2_INPUT_L2_CACHE_CTL),
            )

    @jit
    def _write_per_d_full_d_handoff(self, full_ring_buffer, nd_slot):
        nz_slot = self.ub_per_d_nz.next_write()
        nd2nz_tile(nz_slot, nd_slot)
        copy_destination = full_ring_buffer.next_write().partition(dim=1, part_id=self.subblock_idx)
        copy_source = nz_slot
        mem_copy(copy_destination, copy_source[:copy_destination.shape[0], :copy_destination.shape[1]])

    @jit
    def _update_safe_diagonal_left_values(
        self, k_source, q_source, active_row, left_l1_operand, clear_previous: bool
    ):
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            if const_expr(clear_previous):
                zero = vcast(vdups(0.0, dtypes.float32, mask=mask), dtypes.bfloat16, mask=mask)
                old_offset = (active_row - 1) * D_HALF
                vstore_pack(self.ub_safe_left_k, old_offset, zero, mask, pack_mode="b32_to_b16")
                vstore_pack(self.ub_safe_left_q, old_offset, zero, mask, pack_mode="b32_to_b16")
            offset = active_row * D_HALF
            unpacked_k = vload_unpack(k_source, offset, unpack_mode=UnpackMode.B16_TO_B32)
            unpacked_q = vload_unpack(q_source, offset, unpack_mode=UnpackMode.B16_TO_B32)
            k = vcast(unpacked_k, dtypes.float32, mask=mask)
            q = vcast(unpacked_q, dtypes.float32, mask=mask)
            scaled_q = vmuls(q, self.scale_value, mask=mask)
            vstore_pack(self.ub_safe_left_k, offset, vcast(k, dtypes.bfloat16, mask=mask), mask, pack_mode="b32_to_b16")
            vstore_pack(
                self.ub_safe_left_q, offset, vcast(scaled_q, dtypes.bfloat16, mask=mask), mask, pack_mode="b32_to_b16",
            )
        self._write_per_d_full_d_handoff(left_l1_operand, self.ub_safe_left_k)
        self._write_per_d_full_d_handoff(left_l1_operand, self.ub_safe_left_q)





    @jit
    def pack_shared_score_panel(self, normalized_q_dhalf, normalized_k_dhalf, gate_cumsum_dhalf, panel_idx):
        """使用 32 行共享的中心参考值打包各因果分数块。"""
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            reference_front = vload(gate_cumsum_dhalf, 16 * D_HALF)
            reference_back = vload(gate_cumsum_dhalf, 48 * D_HALF)
            if const_expr(panel_idx == 0):
                reference = reference_front
            else:
                reference = reference_back
            for qk_half_idx in range_constexpr(2):
                col_block = panel_idx * 2 + qk_half_idx
                for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                    panel_row = qk_half_idx * NEUMANN_BLOCK_SIZE + row
                    logical_row = panel_idx * HALF_CHUNK_SIZE + panel_row
                    element_offset = logical_row * D_HALF
                    q_value = vcast(
                        vload_unpack(normalized_q_dhalf, element_offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=lane_mask,
                    )
                    k_value = vcast(
                        vload_unpack(normalized_k_dhalf, element_offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=lane_mask,
                    )
                    gate = vload(gate_cumsum_dhalf, element_offset)
                    decay = vexp_sub(gate, reference, mask=lane_mask)
                    self.store_shared_score_operand16(
                        cast_and_pack_bf16(vmul(k_value, decay, mask=lane_mask), lane_mask), 2 * col_block, row, low_mask
                    )
                    self.store_shared_score_operand16(
                        cast_and_pack_bf16(
                            vmuls(vmul(q_value, decay, mask=lane_mask), self.scale_value, mask=lane_mask), lane_mask
                        ),
                        2 * col_block + 1,
                        row,
                        low_mask,
                    )
                    for row_group in range_constexpr(2):
                        if const_expr(col_block < 2 * (row_group + 1)):
                            if const_expr(row_group == 0):
                                right_reference = reference_front
                            else:
                                right_reference = reference_back
                            right_decay = vexp_sub(right_reference, gate, mask=lane_mask)
                            right_tile = 8 + row_group * 2 + col_block
                            self.store_shared_score_operand16(
                                cast_and_pack_bf16(vmul(k_value, right_decay, mask=lane_mask), lane_mask),
                                right_tile,
                                row,
                                low_mask,
                            )

    def store_shared_score_operand16(self, value, tile, row, lane_mask):
        vstore_strided(
            self.ub_shared_score_nz,
            (tile * NEUMANN_BLOCK_SIZE + row) * NZ_TILE,
            value,
            lane_mask,
            block_stride=self.ub_shared_score_nz.physical_layout.stride[0] // NZ_TILE,
        )


@jit
def _run_stage1_body(
    tail_group,
    tail_rows,
    gm_K,
    gm_V,
    gm_Q,
    gm_beta_brc,
    gm_g,
    gm_qk_exchange,
    gm_A_log,
    gm_dt_bias,
    gm_q_decayed,
    gm_kg,
    gm_gamma,
    gm_u,
    gm_w,
    gm_qg,
    gm_g_cumsum,
    gm_identity16,
    gm_q_norm,
    gm_k_norm,
    gm_q_rstd,
    gm_k_rstd,
    gm_beta_activated,
    gm_Aqk,
    gm_Akk,
    workspace_task_base: int,
    head_count: int,
    chunks_total: int,
    seq_len: int,
    head_base: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    active_core_num: int,
    task_start: int,
    task_end: int,
    scale_value: float,
    lower_bound: float,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    has_lower_bound,
    save_qg,
    save_g_cumsum,
    save_matrices,
):
    """Chunk Kimi Delta Attention Stage1。

    Stage1 对每个 chunk 计算 K_inv、K_decayed、Q_decayed 和 beta 加权的 KKT 下三角矩阵，
    使用 Neumann 展开求解块内三角系统，生成 Mqk、U_pre、W、K_restored 和 gamma_C。

    计算 [head_base, head_base+head_count) 这些(平铺后)head 一组 chunk 的 StageOne workspace。
    输入保持原始 (B, N, S, D)；用 idx2crd 把平铺 head 反解回 (batch, head) 再 2D 索引（免 reshape）。"""
    subblock_idx = get_subblock_id()
    block_idx = get_block_idx()
    batch_count = gm_K.shape[0]
    value_head_count = gm_V.shape[1]
    key_head_count = gm_K.shape[1]
    heads_per_key_head = value_head_count // key_head_count
    if block_idx < active_core_num:
        stage1_matmul = StageOneMatmul()
        vector = StageOneVector(scale_value=scale_value, subblock_idx=subblock_idx)
        task_indices = Array(dtypes.int64, 2)
        task_valid_row_counts = Array(dtypes.int64, 2)
        identity = stage1_matmul.prepare_packed64_identity_l1(gm_identity16)
        stage1_matmul.zero_half_l1(stage1_matmul.zero_l1)
        pipeline_tick = 0
        issued_task_count = 0
        qk_halves_prefetched = False
        for local_task_index in range(task_start, task_end):
            current_local_head_index = local_task_index // group_count
            current_group_chunk_index = local_task_index - current_local_head_index * group_count
            current_chunk_index = group_start + current_group_chunk_index
            current_global_head_index = head_base + current_local_head_index
            current_batch_index, current_value_head_index = idx2crd(
                current_global_head_index, [batch_count, value_head_count]
            )
            current_query_key_head_index = current_value_head_index // heads_per_key_head
            current_valid_rows = CHUNK_SIZE
            if tail_group:
                if current_group_chunk_index + 1 == group_count:
                    current_valid_rows = tail_rows
            buffer_parity = local_task_index % 2
            vector.issue_raw_gate(
                gm_g,
                gm_A_log,
                gm_dt_bias,
                current_batch_index,
                current_value_head_index,
                current_chunk_index,
                current_valid_rows,
                use_gate_in_kernel,
            )
            if not qk_halves_prefetched:
                vector.issue_qk_local_halves(
                    gm_Q,
                    gm_K,
                    current_batch_index,
                    current_query_key_head_index,
                    current_chunk_index,
                    current_valid_rows,
                    buffer_parity,
                )
            else:
                vector.notify_qk_local_halves_ready()
            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_slot = (pipeline_tick - 1) % 2
                delayed_task_index = task_indices[delayed_task_slot]
                local_head_index = delayed_task_index // group_count
                group_chunk_index = delayed_task_index - local_head_index * group_count
                chunk_index = group_start + group_chunk_index
                global_head_index = head_base + local_head_index
                batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
                buffer_parity = delayed_task_index % 2
                delayed_valid_rows = task_valid_row_counts[delayed_task_slot]
                vector.apply_beta_and_finish_t_for_neumann(
                    vector.ub_beta, stage1_matmul.neumann_power_handoff_l1, subblock_idx, buffer_parity
                )
                vector.prepare_beta_weighted_c3_operands(vector.ub_beta, buffer_parity, delayed_valid_rows)
                negative_lower_matrix, identity_matrix, neumann_scratch = (
                    stage1_matmul.load_packed64_diagonal_inputs_from_l1_handoff(
                        stage1_matmul.neumann_power_handoff_l1, identity
                    )
                )
                second_power, packed_inverse = stage1_matmul.neumann_diag_power2_update(
                    negative_lower_matrix, identity_matrix, neumann_scratch
                )
                fourth_power, packed_inverse = stage1_matmul.neumann_diag_power4_update(second_power, packed_inverse)
                packed_inverse = stage1_matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
                packed_inverse = stage1_matmul.compose_odd_even_lower16_to32_accum_full64(
                    neumann_scratch, packed_inverse
                )
                stage1_matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)
                stage1_matmul.store_inverse(gm_Akk, value_head_index, chunk_index, delayed_valid_rows, save_matrices)
            vector.issue_beta_weighted_v(
                gm_V, current_batch_index, current_value_head_index, current_chunk_index, current_valid_rows
            )
            task_write_slot = pipeline_tick % 2
            task_indices[task_write_slot] = dtypes.int64(local_task_index)
            task_valid_row_counts[task_write_slot] = dtypes.int64(current_valid_rows)
            local_head_index = current_local_head_index
            group_chunk_index = current_group_chunk_index
            chunk_index = current_chunk_index
            global_head_index = current_global_head_index
            batch_index = current_batch_index
            value_head_index = current_value_head_index
            workspace_task_index = workspace_task_base + global_head_index * eff_group + group_chunk_index
            buffer_parity = local_task_index % 2
            vector.finish_and_publish_qk_partials(gm_qk_exchange, block_idx, current_valid_rows, buffer_parity)
            vector.issue_peer_qk_partials(gm_qk_exchange, block_idx)
            gate_cumsum_dhalf = vector.activate_issued_gate(
                lower_bound, current_valid_rows, buffer_parity, use_gate_in_kernel, has_lower_bound
            )
            if save_g_cumsum != 0:
                vector.store_g_cumsum(
                    gm_g_cumsum, gate_cumsum_dhalf, current_value_head_index, current_chunk_index, current_valid_rows
                )
            normalized_k_dhalf, normalized_q_dhalf = vector.finish_qk_partial_exchange(
                buffer_parity, use_qk_l2norm_in_kernel
            )
            # 关闭保存时 norm/rstd 使用只读占位，归一化仍在 UB 内计算。
            if save_matrices != 0 and use_qk_l2norm_in_kernel != 0:
                if current_value_head_index % heads_per_key_head == 0:
                    vector.store_normalized_qk(
                        gm_q_norm,
                        gm_k_norm,
                        gm_q_rstd,
                        gm_k_rstd,
                        normalized_q_dhalf,
                        normalized_k_dhalf,
                        current_query_key_head_index,
                        current_chunk_index,
                        current_valid_rows,
                    )
            vector.issue_raw_beta(gm_beta_brc, batch_index, value_head_index, chunk_index, current_valid_rows)
            decayed_q_dhalf, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf = (
                vector.prepare_normalized_chunk_inputs(
                    normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity
                )
            )
            vector.store_q_decayed_scratch(gm_q_decayed, workspace_task_index, decayed_q_dhalf)
            if local_task_index + 1 < task_end:
                next_local_task_index = local_task_index + 1
                next_local_head_index = next_local_task_index // group_count
                next_group_chunk_index = next_local_task_index - next_local_head_index * group_count
                next_chunk_index = group_start + next_group_chunk_index
                next_global_head_index = head_base + next_local_head_index
                next_batch_index, next_value_head_index = idx2crd(
                    next_global_head_index, [batch_count, value_head_count]
                )
                next_query_key_head_index = next_value_head_index // heads_per_key_head
                next_valid_rows = CHUNK_SIZE
                if tail_group:
                    if next_group_chunk_index + 1 == group_count:
                        next_valid_rows = tail_rows
                next_buffer_parity = next_local_task_index % 2
                vector.prefetch_qk_local_halves(
                    gm_Q,
                    gm_K,
                    next_batch_index,
                    next_query_key_head_index,
                    next_chunk_index,
                    next_valid_rows,
                    next_buffer_parity,
                )
                qk_halves_prefetched = True
            else:
                qk_halves_prefetched = False
            if (
                use_gate_in_kernel != 0
                and has_lower_bound != 0
                and (scale_value >= -0.125)
                and (scale_value <= 0.125)
                and (lower_bound >= -5.0)
                and allow_neg_eigval == 0
            ):
                stage1_matmul.compute_shared_per_d_scores(
                    vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf
                )
            else:
                if use_gate_in_kernel != 0 and has_lower_bound != 0:
                    vector.build_per_d_left_k_and_decay(
                        normalized_k_dhalf, gate_cumsum_dhalf, 0, vector.l1_per_d_operand
                    )
                    stage1_matmul.load_per_d_left_k(vector.l1_per_d_operand)
                    vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 0, vector.l1_per_d_operand)
                    stage1_matmul.load_per_d_left_q(vector.l1_per_d_operand)
                    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 0, 0, vector.l1_per_d_operand)
                    right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                    stage1_matmul.mm_per_d_kkt(0, 0, right)
                    stage1_matmul.mm_per_d_mqk(0, 0, right)
                else:
                    stage1_matmul.mm_safe_diagonal_tile(
                        vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, 0
                    )
                vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 1, vector.l1_per_d_operand)
                stage1_matmul.load_per_d_left_k(vector.l1_per_d_operand)
                vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 1, vector.l1_per_d_operand)
                stage1_matmul.load_per_d_left_q(vector.l1_per_d_operand)
                vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 0, vector.l1_per_d_operand)
                right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                stage1_matmul.mm_per_d_kkt(1, 0, right)
                stage1_matmul.mm_per_d_mqk(1, 0, right)
                if use_gate_in_kernel != 0 and has_lower_bound != 0:
                    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 1, 1, vector.l1_per_d_operand)
                    right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                    stage1_matmul.mm_per_d_kkt(1, 1, right)
                    stage1_matmul.mm_per_d_mqk(1, 1, right)
                else:
                    stage1_matmul.mm_safe_diagonal_tile(
                        vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, 1
                    )
                vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 2, vector.l1_per_d_operand)
                stage1_matmul.load_per_d_left_k(vector.l1_per_d_operand)
                vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 2, vector.l1_per_d_operand)
                stage1_matmul.load_per_d_left_q(vector.l1_per_d_operand)
                vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 0, vector.l1_per_d_operand)
                right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                stage1_matmul.mm_per_d_kkt(2, 0, right)
                stage1_matmul.mm_per_d_mqk(2, 0, right)
                vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 1, vector.l1_per_d_operand)
                right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                stage1_matmul.mm_per_d_kkt(2, 1, right)
                stage1_matmul.mm_per_d_mqk(2, 1, right)
                if use_gate_in_kernel != 0 and has_lower_bound != 0:
                    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 2, 2, vector.l1_per_d_operand)
                    right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                    stage1_matmul.mm_per_d_kkt(2, 2, right)
                    stage1_matmul.mm_per_d_mqk(2, 2, right)
                else:
                    stage1_matmul.mm_safe_diagonal_tile(
                        vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, 2
                    )
                vector.build_per_d_left_k_and_decay(normalized_k_dhalf, gate_cumsum_dhalf, 3, vector.l1_per_d_operand)
                stage1_matmul.load_per_d_left_k(vector.l1_per_d_operand)
                vector.build_per_d_left_q_from_decay(normalized_q_dhalf, 3, vector.l1_per_d_operand)
                stage1_matmul.load_per_d_left_q(vector.l1_per_d_operand)
                vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 0, vector.l1_per_d_operand)
                right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                stage1_matmul.mm_per_d_kkt(3, 0, right)
                stage1_matmul.mm_per_d_mqk(3, 0, right)
                vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 1, vector.l1_per_d_operand)
                right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                stage1_matmul.mm_per_d_kkt(3, 1, right)
                stage1_matmul.mm_per_d_mqk(3, 1, right)
                vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 2, vector.l1_per_d_operand)
                right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                stage1_matmul.mm_per_d_kkt(3, 2, right)
                stage1_matmul.mm_per_d_mqk(3, 2, right)
                if use_gate_in_kernel != 0 and has_lower_bound != 0:
                    vector.build_per_d_right(normalized_k_dhalf, gate_cumsum_dhalf, 3, 3, vector.l1_per_d_operand)
                    right = stage1_matmul.load_per_d_right_k(vector.l1_per_d_operand)
                    stage1_matmul.mm_per_d_kkt(3, 3, right)
                    stage1_matmul.mm_per_d_mqk(3, 3, right)
                else:
                    stage1_matmul.mm_safe_diagonal_tile(
                        vector, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, 3
                    )
            if save_qg != 0:
                vector.store_qg(gm_qg, current_value_head_index, current_chunk_index, current_valid_rows)
            stage1_matmul.finish_per_d_kkt(vector.ub_kkt)
            vector.activate_issued_beta(current_valid_rows, use_beta_sigmoid_in_kernel, allow_neg_eigval)
            if save_matrices != 0 and use_beta_sigmoid_in_kernel != 0:
                vector.store_beta(gm_beta_activated, current_value_head_index, current_chunk_index, current_valid_rows)
            stage1_matmul.finish_per_d_mqk(vector.ub_mqk)
            issued_task_count = issued_task_count + 1
            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_slot = (pipeline_tick - 1) % 2
                delayed_task_index = task_indices[delayed_task_slot]
                local_head_index = delayed_task_index // group_count
                group_chunk_index = delayed_task_index - local_head_index * group_count
                chunk_index = group_start + group_chunk_index
                global_head_index = head_base + local_head_index
                _, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
                workspace_task_index = global_head_index * eff_group + group_chunk_index + workspace_task_base
                buffer_parity = delayed_task_index % 2
                delayed_valid_rows = task_valid_row_counts[delayed_task_slot]
                stage1_matmul.compute_u_pre_and_w_from_resident_inv_beta(
                    gm_u,
                    gm_w,
                    value_head_index,
                    chunk_index,
                    delayed_valid_rows,
                    vector.l1_V_beta,
                    vector.l1_K_decayed_beta_for_W,
                )
                log_gamma_c_dhalf = vector.finish_mqk(
                    gm_Aqk, head_base + local_head_index, chunk_index, delayed_valid_rows, buffer_parity
                )
                vector.store_restored_state_inputs(
                    gm_kg,
                    gm_gamma,
                    workspace_task_index,
                    log_gamma_c_dhalf,
                    buffer_parity,
                    value_head_index,
                    chunk_index,
                    delayed_valid_rows,
                )
            pipeline_tick = pipeline_tick + 1
        if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
            delayed_task_slot = (pipeline_tick - 1) % 2
            delayed_task_index = task_indices[delayed_task_slot]
            local_head_index = delayed_task_index // group_count
            group_chunk_index = delayed_task_index - local_head_index * group_count
            chunk_index = group_start + group_chunk_index
            global_head_index = head_base + local_head_index
            batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
            workspace_task_index = workspace_task_base + global_head_index * eff_group + group_chunk_index
            buffer_parity = delayed_task_index % 2
            delayed_valid_rows = task_valid_row_counts[delayed_task_slot]
            vector.apply_beta_and_finish_t_for_neumann(
                vector.ub_beta, stage1_matmul.neumann_power_handoff_l1, subblock_idx, buffer_parity
            )
            vector.prepare_beta_weighted_c3_operands(vector.ub_beta, buffer_parity, delayed_valid_rows)
            negative_lower_matrix, identity_matrix, neumann_scratch = (
                stage1_matmul.load_packed64_diagonal_inputs_from_l1_handoff(
                    stage1_matmul.neumann_power_handoff_l1, identity
                )
            )
            second_power, packed_inverse = stage1_matmul.neumann_diag_power2_update(
                negative_lower_matrix, identity_matrix, neumann_scratch
            )
            fourth_power, packed_inverse = stage1_matmul.neumann_diag_power4_update(second_power, packed_inverse)
            packed_inverse = stage1_matmul.neumann_diag_power8_update(fourth_power, packed_inverse)
            packed_inverse = stage1_matmul.compose_odd_even_lower16_to32_accum_full64(neumann_scratch, packed_inverse)
            stage1_matmul.compose_odd_even_lower32_to64_accum_full64(neumann_scratch, packed_inverse)
            stage1_matmul.store_inverse(gm_Akk, value_head_index, chunk_index, delayed_valid_rows, save_matrices)
            stage1_matmul.compute_u_pre_and_w_from_resident_inv_beta(
                gm_u,
                gm_w,
                value_head_index,
                chunk_index,
                delayed_valid_rows,
                vector.l1_V_beta,
                vector.l1_K_decayed_beta_for_W,
            )
            log_gamma_c_dhalf = vector.finish_mqk(
                gm_Aqk, head_base + local_head_index, chunk_index, delayed_valid_rows, buffer_parity
            )
            vector.store_restored_state_inputs(
                gm_kg,
                gm_gamma,
                workspace_task_index,
                log_gamma_c_dhalf,
                buffer_parity,
                value_head_index,
                chunk_index,
                delayed_valid_rows,
            )


class StageTwoMatmul:
    def __init__(self, *, dv_base: int, shared=None):
        self.dv_base = dv_base
        if shared is None:
            self.l0a_chunk = RingBuffer(
                MemLoc.L0A,
                dtypes.bfloat16,
                make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                depth=2,
            )
            self.l0a_kr_t = RingBuffer(
                MemLoc.L0A,
                dtypes.bfloat16,
                make_layout((dv_base, CHUNK_SIZE)),
                depth=1,
            ).next_write()
            self.l0a_mqk = RingBuffer(
                MemLoc.L0A,
                dtypes.bfloat16,
                make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                depth=1,
            ).next_write()
            self.l0b_u = RingBuffer(
                MemLoc.L0B,
                dtypes.bfloat16,
                make_layout((dv_base, CHUNK_SIZE)),
                depth=1,
            ).next_write()
            self.l0b_state = RingBuffer(
                MemLoc.L0B,
                dtypes.bfloat16,
                make_layout((dv_base, SUPPORTED_HEAD_DIM)),
                depth=1,
            ).next_write()
            self.l0b_kr = RingBuffer(
                MemLoc.L0B,
                dtypes.bfloat16,
                make_layout((SUPPORTED_HEAD_DIM, CHUNK_SIZE)),
                depth=1,
            ).next_write()
            self.l0c_u = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, dv_base)), depth=1).next_write()
            self.l0c_o = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, dv_base)), depth=1).next_write()
            self.l0c_state = RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((dv_base, SUPPORTED_HEAD_DIM)),
                depth=1,
            ).next_write()
        else:
            self.l0a_chunk = shared.l0a_chunk
            self.l0a_kr_t = from_tensor(shared.l0a_kr_t, layout=make_layout((dv_base, CHUNK_SIZE)), data_format="nz")
            self.l0a_mqk = shared.l0a_mqk
            self.l0b_u = from_tensor(shared.l0b_u, layout=make_layout((dv_base, CHUNK_SIZE)), data_format="nz")
            self.l0b_state = from_tensor(shared.l0b_state, layout=make_layout((dv_base, SUPPORTED_HEAD_DIM)), data_format="nz")
            self.l0b_kr = shared.l0b_kr
            self.l0c_u = shared.l0c_u
            self.l0c_o = shared.l0c_o
            self.l0c_state = shared.l0c_state

    @staticmethod
    def load_q_decayed(q_decayed_l1, gm_tile):
        mem_copy(q_decayed_l1.next_write(), gm_tile)

    def load_k_cumdecay(self, k_cumdecay_l1, gm_tile, valid_rows):
        self._load_chunk_rows(k_cumdecay_l1, gm_tile, valid_rows)

    def load_k_restored(self, k_restored_l1, gm_tile, valid_rows):
        self._load_chunk_rows(k_restored_l1, gm_tile, valid_rows, l2_cache_ctl=_L2_CACHE_LAST_USE)

    def load_mqk(self, mqk_l1, gm_tile, valid_rows):
        self._load_chunk_rows(mqk_l1, gm_tile, valid_rows, l2_cache_ctl=_L2_CACHE_LAST_USE)

    def load_state(self, state_l1):
        mem_copy(
            self.l0b_state,
            from_tensor(state_l1.next_read(), layout=make_layout((self.dv_base, SUPPORTED_HEAD_DIM)), data_format="nz"),
        )

    def compute_w_state(self, w_l1):
        mem_copy(self.l0a_chunk.next_write(), w_l1.next_read())
        matmul(
            self.l0c_u[0:CHUNK_SIZE, 0:self.dv_base], self.l0a_chunk.next_read(),
            self.l0b_state, init=True,
        )

    def compute_q_state(self, q_decayed_l1):
        mem_copy(self.l0a_chunk.next_write(), q_decayed_l1.next_read())
        matmul(
            self.l0c_o[0:CHUNK_SIZE, 0:self.dv_base], self.l0a_chunk.next_read(),
            self.l0b_state, init=True,
        )

    @jit
    def load_k_restored_operand(self, k_restored_l1, valid_rows):
        source = k_restored_l1.next_read()
        mem_copy(self.l0b_kr, (source).permute(1, 0))

    @jit
    def load_mqk_operand(self, mqk_l1, valid_rows):
        source = mqk_l1.next_read()
        mem_copy(self._mqk_operand(CHUNK_SIZE), source)

    @jit
    def update_state(self, u_slot, valid_rows):
        self._update_state_tail(u_slot, valid_rows)

    @jit
    def accumulate_mqk_u(self, u_slot, valid_rows):
        self._accumulate_mqk_u_tail(u_slot, valid_rows)

    def store_u_delta(self, ub):
        mem_copy(
            ub,
            (tile_slice(
                self.l0c_u[0:CHUNK_SIZE, 0:self.dv_base], (CHUNK_SIZE, self.dv_base),
                (0, 0),
            )).partition(
                dim=0,
            ),
        )

    def store_output(self, gm_tile):
        """用源视图的有效行列限定 FIXPIPE 尾块写回。"""
        mem_copy(
            gm_tile, self.l0c_o[0:CHUNK_SIZE, 0:self.dv_base][:gm_tile.shape[0], :gm_tile.shape[1]],
            options=copy_options(l2_cache_ctl=_L2_CACHE_DISABLE),
        )

    def store_state_delta(self, ub):
        mem_copy(
            ub,
            (from_tensor(self.l0c_state, layout=make_layout((self.dv_base, SUPPORTED_HEAD_DIM)), data_format="nz")).partition(dim=1),
        )

    @jit
    def _load_chunk_rows(self, ring_buffer, gm_tile, valid_rows, *, l2_cache_ctl=0):
        slot = ring_buffer.next_write()
        if valid_rows < CHUNK_SIZE:
            cube_fill_l1_zero(slot, repeat=slot.shape[0] * slot.shape[1] // 16)
            cube_sync_pipe(PIPE.MTE2)
        if valid_rows < CHUNK_SIZE:
            for col_block in range(gm_tile.shape[1] // 16):
                gm_col = tile_slice(gm_tile, (CHUNK_SIZE, 16), (0, col_block))
                l1_col = tile_slice(slot, (CHUNK_SIZE, 16), (0, col_block))
                mem_copy(
                    l1_col, _valid_rows_view(gm_col, valid_rows, 16), options=copy_options(l2_cache_ctl=l2_cache_ctl),
                )
        else:
            mem_copy(slot, gm_tile, options=copy_options(l2_cache_ctl=l2_cache_ctl))

    def _k_restored_operand(self, valid_rows):
        return self.l0b_kr[0:SUPPORTED_HEAD_DIM, 0:valid_rows]

    def _mqk_operand(self, valid_rows):
        return self.l0a_mqk[0:CHUNK_SIZE, 0:valid_rows]

    def _update_state_tail(self, u_slot, valid_rows):
        source = u_slot[0:CHUNK_SIZE, 0:self.dv_base][
            :CHUNK_SIZE, None
        ]
        operand = self.l0a_kr_t
        write_operand = self.l0a_kr_t
        mem_copy(write_operand, source.permute(1, 0))
        matmul(
            from_tensor(self.l0c_state, layout=make_layout((self.dv_base, SUPPORTED_HEAD_DIM)), data_format="nz"), operand,
            self._k_restored_operand(CHUNK_SIZE), init=True,
        )

    def _accumulate_mqk_u_tail(self, u_slot, valid_rows):
        source = u_slot[0:CHUNK_SIZE, 0:self.dv_base][
            :CHUNK_SIZE, None
        ]
        operand = self.l0b_u
        write_operand = self.l0b_u
        mem_copy(write_operand, source.permute(1, 0))
        matmul(
            self.l0c_o[0:CHUNK_SIZE, 0:self.dv_base], self._mqk_operand(CHUNK_SIZE),
            operand, init=False,
        )


class StageTwoVector:
    def __init__(self, *, u_row_block: int, dv_base: int, shared=None):
        self.dv_base = dv_base
        self.u_row_block = u_row_block
        self.half_dk = D_HALF
        if shared is None:
            self.ub_state_fp32 = Buffer(MemLoc.UB, dtypes.float32, make_layout((self.dv_base, self.half_dk)))
            self.tmp_u_bf16 = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((u_row_block, dv_base)))
            self.state_nd = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((self.dv_base, self.half_dk)), depth=1).next_write()
            self.state_nz = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((self.dv_base, self.half_dk)),
                depth=1,
                physical_layout=nz_layout(self.dv_base, self.half_dk),
                data_format="nz",
            ).next_write()
            self.u_nz = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((u_row_block, dv_base)),
                depth=1,
                physical_layout=nz_layout(u_row_block, dv_base),
                data_format="nz",
            ).next_write()
            self.state_in = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((self.dv_base, self.half_dk)), depth=1).next_write()
            self.state_out = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((self.dv_base, self.half_dk)), depth=1).next_write()
            self.u_in = RingBuffer(MemLoc.UB, dtypes.bfloat16, make_layout((u_row_block, dv_base)), depth=2)
        else:
            self.ub_state_fp32 = shared.ub_state_fp32
            self.tmp_u_bf16 = shared.tmp_u_bf16
            self.state_nd = shared.state_nd.flatten()[0:dv_base * self.half_dk,].reshape((dv_base, self.half_dk))
            self.state_nz = shared.state_nz
            self.u_nz = shared.u_nz
            self.state_in = shared.state_in
            self.state_out = shared.state_out
            self.u_in = shared.u_in

    @staticmethod
    def load_gamma_col(gamma_ub, gamma_col):
        slot = gamma_ub.next_write()
        mem_copy(slot, gamma_col)

    def store_h(self, gm_h_rows):
        mem_copy(
            gm_h_rows, self.ub_state_fp32.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk)),
            options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
        )

    def store_group_state_to_h(self, gm_h_rows):
        mem_copy(
            gm_h_rows, self.ub_state_fp32.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk)),
            options=copy_options(l2_cache_ctl=_STAGE2_INPUT_L2_CACHE_CTL),
        )

    def load_initial_state_and_write_l1(self, current_state, state_l1, subblock_idx):
        in_slot = self.state_in.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        in_read = in_slot
        in_read = in_read[tuple(map(slice, current_state.shape))]
        mem_copy(in_read, current_state)
        ub_state = self.ub_state_fp32.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        mem_copy(ub_state, in_read)
        self._produce_state_l1(state_l1, subblock_idx)

    def write_state_l1_snapshot(self, state_l1, subblock_idx):
        """发布累加后的状态，供下一步 Cube 计算使用。"""
        self._produce_state_l1(state_l1, subblock_idx)

    @jit
    def scale_and_add_delta(self, tmp_delta, gamma_ub):
        """在一次全状态 VF 遍历中融合状态衰减与增量累加。"""
        gamma_read = gamma_ub.next_read()
        ub_gamma = gamma_read
        ub_state = self.ub_state_fp32.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        delta = tmp_delta
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            gamma_vec = vload(ub_gamma, 0)
            for base in dsl_range(0, self.dv_base * self.half_dk, VL, unroll=4):
                state_v = vload(ub_state, base)
                delta_v = vload(delta, base)
                state_scaled = vmul(state_v, gamma_vec, mask=mask)
                state_next = vadd(state_scaled, delta_v, mask=mask)
                vstore(ub_state, base, state_next, mask)

    @jit
    def issue_u_rows(self, gm_u_tile, subblock_idx, valid_rows):
        row_start = subblock_idx * self.u_row_block
        u_in_write = self.u_in.next_write().flatten()[0:self.u_row_block * self.dv_base,].reshape((self.u_row_block, self.dv_base))
        if row_start < valid_rows:
            rows = valid_rows - row_start
            if rows > self.u_row_block:
                rows = self.u_row_block
            gm_u_rows = tile_slice(gm_u_tile, (self.u_row_block, self.dv_base), (subblock_idx, 0))
            mem_copy(_valid_rows_view(u_in_write, rows, self.dv_base), _valid_rows_view(gm_u_rows, rows, self.dv_base))

    @jit
    def finish_u_delta_to_l1(
        self, u_l1, tmp_u_delta, subblock_idx, gm_v_new_chunk, valid_rows
    ):
        u_l1_rows = tile_slice(u_l1.next_write(), (self.u_row_block, self.dv_base), (subblock_idx, 0))
        delta = tmp_u_delta
        row_start = subblock_idx * self.u_row_block
        rows = valid_rows - row_start
        if rows > self.u_row_block:
            rows = self.u_row_block
        if rows < 0:
            rows = 0
        u_in_read = self.u_in.next_read().flatten()[0:self.u_row_block * self.dv_base,].reshape((self.u_row_block, self.dv_base))
        u_bf16_write = self.tmp_u_bf16.flatten()[0:self.u_row_block * self.dv_base,].reshape((self.u_row_block, self.dv_base))
        with vf(mode="simd"):
            bf16_mask, _ = update_mask(self.dv_base, elem_bits=16)
            zero = vdups(0.0, dtypes.bfloat16, mask=bf16_mask)
            for row in range(0, self.u_row_block):
                vstore(u_bf16_write, row * self.dv_base, zero, bf16_mask)
        with vf(mode="simd"):
            for row in dsl_range(0, rows, 1, unroll=2):
                for col in range(0, self.dv_base, VL):
                    mask, _ = update_mask(self.dv_base - col, elem_bits=32)
                    base = row * self.dv_base + col
                    u_raw = vload_unpack(u_in_read, base, unpack_mode=UnpackMode.B16_TO_B32)
                    u_fp32 = vcast(u_raw, dtypes.float32, mask=mask)
                    delta_value = vload(delta, base)
                    u_next = vsub(u_fp32, delta_value, mask=mask)
                    u_bf16 = vcast(u_next, dtypes.bfloat16, mask=mask)
                    vstore_pack(u_bf16_write, base, u_bf16, mask, pack_mode=PackMode.B32_TO_B16)
        u_nz_slot = self.u_nz[0:self.u_row_block, 0:self.dv_base]
        nd2nz_tile(u_nz_slot, self.tmp_u_bf16.flatten()[0:self.u_row_block * self.dv_base,].reshape((self.u_row_block, self.dv_base)))
        u_nz_read = self.u_nz[0:self.u_row_block, 0:self.dv_base]
        mem_copy(u_l1_rows, u_nz_read)
        if rows > 0:
            v_new_rows = tile_slice(gm_v_new_chunk, (self.u_row_block, self.dv_base), (subblock_idx, 0))
            mem_copy(
                _valid_rows_view(v_new_rows, rows, self.dv_base),
                (self.tmp_u_bf16.flatten()[0:self.u_row_block * self.dv_base,].reshape((self.u_row_block, self.dv_base)))[:rows, :self.dv_base],
                options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )

    def store_state_to_gm(self, gm_state_rows):
        """末态 ub_state_fp32 → GM（FP32）。累加在 V pipe、写 GM 在 MTE3，跨 pipe，经 RingBuffer 交接。"""
        ub_state = self.ub_state_fp32.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        out_slot = self.state_out.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        mem_copy(out_slot, ub_state)
        out_read = self.state_out.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        mem_copy(gm_state_rows, out_read)

    def _produce_state_l1(self, state_l1, subblock_idx):
        ub_state = self.ub_state_fp32.flatten()[0:self.dv_base * self.half_dk,].reshape((self.dv_base, self.half_dk))
        nd_slot = self.state_nd
        cast_tile(nd_slot, ub_state)
        nd_read = self.state_nd
        # 按当前 DV 高度重新打包共享槽位，避免切片保留最大高度的 N1 步长。
        nz_slot = from_tensor(self.state_nz, layout=make_layout((self.dv_base, self.half_dk)), data_format="nz", physical_layout=nz_layout(self.dv_base, self.half_dk))
        nd2nz_tile(nz_slot, nd_read)
        nz_read = from_tensor(self.state_nz, layout=make_layout((self.dv_base, self.half_dk)), data_format="nz", physical_layout=nz_layout(self.dv_base, self.half_dk))
        slot = from_tensor(state_l1.next_write(), layout=make_layout((self.dv_base, SUPPORTED_HEAD_DIM)), data_format="nz")
        state_l1_rows = tile_slice(slot, (self.dv_base, self.half_dk), (0, subblock_idx))
        mem_copy(state_l1_rows, nz_read)


class _StageTwoResources:
    """每轮仅分配一次，由互斥的 DV 变体共享。"""

    def __init__(self):
        self.q_l1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=2)
        self.w_l1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=2)
        self.k_restored_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.mqk_l1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2)
        self.state_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.u_l1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)), depth=2)
        self.matmul = StageTwoMatmul(dv_base=SUPPORTED_HEAD_DIM)
        self.gamma_ub = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((D_HALF, 1)), depth=2)
        self.tmp_delta = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((SUPPORTED_HEAD_DIM, D_HALF)), depth=1).next_write()
        self.tmp_u_delta = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)),
            depth=1,
        ).next_write()
        self.vector = StageTwoVector(u_row_block=CHUNK_SIZE // 2, dv_base=SUPPORTED_HEAD_DIM)


@jit
def _run_stage2_body(
    gm_kg,
    gm_gamma,
    gm_q_decayed,
    gm_w,
    gm_u,
    gm_v_new,
    gm_h,
    gm_O,
    gm_state_in,
    gm_state_out,
    gm_Mqk,
    *,
    workspace_task_base: int,
    dv_base: int,
    head_base: int,
    head_count: int,
    chunks_total: int,
    seq_len: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    logical_core_num: int,
    item_start: int,
    item_end: int,
    resources,
    state_batch,
    state_batch_idx: int,
    state_chunk_base: int,
    has_initial_state,
    save_final_state,
):
    block_idx = get_block_idx()
    subblock_idx = get_subblock_id()
    num_dv_base = SUPPORTED_HEAD_DIM // dv_base
    half_dk = D_HALF
    u_row_block = CHUNK_SIZE // 2
    q_l1 = resources.q_l1
    w_l1 = resources.w_l1
    k_restored_l1 = resources.k_restored_l1
    mqk_l1 = resources.mqk_l1
    state_l1 = resources.state_l1
    u_l1 = resources.u_l1
    s2_matmul = StageTwoMatmul(dv_base=dv_base, shared=resources.matmul)
    gamma_ub = resources.gamma_ub
    tmp_delta = resources.tmp_delta.flatten()[0:dv_base * half_dk,].reshape((dv_base, half_dk))
    tmp_u_delta = resources.tmp_u_delta.flatten()[0:u_row_block * dv_base,].reshape((u_row_block, dv_base))
    s2_vector = StageTwoVector(u_row_block=u_row_block, dv_base=dv_base, shared=resources.vector)
    if block_idx < logical_core_num:
        for item in range(item_start, item_end):
            head_id, dv_idx = idx2crd(item, [head_count, num_dv_base])
            seq_idx = head_base + head_id
            write_state_ref = gm_state_out[state_batch, seq_idx, None, None]
            first_linear = workspace_task_base + seq_idx * eff_group
            first_w_chunk = tile_slice(gm_w[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (group_start, 0))
            first_q_chunk = tile_slice(gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_k_restored_chunk = tile_slice(
                gm_kg[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (group_start, 0),
            )
            first_mqk_tile = tile_slice(
                gm_Mqk[0, seq_idx, group_start, None, None], (CHUNK_SIZE, CHUNK_SIZE), (0, 0),
            )
            first_gamma_col_full = tile_slice(gm_gamma, (SUPPORTED_HEAD_DIM, 1), (first_linear, 0))
            first_gamma_col = tile_slice(first_gamma_col_full, (half_dk, 1), (subblock_idx, 0))
            first_u_tile_full = tile_slice(
                gm_u[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (group_start, 0),
            )
            first_u_tile = tile_slice(first_u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx))
            first_valid_rows = CHUNK_SIZE
            if group_start + 1 == chunks_total:
                first_valid_rows = seq_len - group_start * CHUNK_SIZE
            s2_matmul.load_k_cumdecay(w_l1, first_w_chunk, first_valid_rows)
            s2_matmul.load_q_decayed(q_l1, first_q_chunk)
            s2_matmul.load_k_restored(k_restored_l1, first_k_restored_chunk, first_valid_rows)
            s2_matmul.load_mqk(mqk_l1, first_mqk_tile, first_valid_rows)
            if group_start == 0 and has_initial_state == 0:
                zero_recurrent_state(s2_vector.ub_state_fp32, dv_base * half_dk)
                s2_vector.write_state_l1_snapshot(state_l1, subblock_idx)
            elif group_start == 0:
                in_state_ref = gm_state_in[state_batch, seq_idx, None, None]
                first_state_rows = tile_slice(in_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            else:
                first_h_state = gm_h[
                    state_batch_idx, seq_idx, state_chunk_base + group_start, None, None
                ]
                first_state_rows = tile_slice(first_h_state, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            s2_vector.load_gamma_col(gamma_ub, first_gamma_col)
            s2_vector.issue_u_rows(first_u_tile, subblock_idx, first_valid_rows)
            for chunk_id in range(0, group_count):
                k = group_start + chunk_id
                linear = workspace_task_base + seq_idx * eff_group + chunk_id
                valid_rows = CHUNK_SIZE
                if k + 1 == chunks_total:
                    valid_rows = seq_len - k * CHUNK_SIZE
                next_valid_rows = CHUNK_SIZE
                if k + 2 == chunks_total:
                    next_valid_rows = seq_len - (k + 1) * CHUNK_SIZE
                o_chunk = _o_head_chunk_tile(gm_O, seq_idx, k, dv_base, dv_idx, valid_rows)
                v_new_chunk_full = tile_slice(
                    gm_v_new[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (k, 0),
                )
                v_new_chunk = tile_slice(v_new_chunk_full, (CHUNK_SIZE, dv_base), (0, dv_idx))
                h_state = gm_h[
                    state_batch_idx, seq_idx, state_chunk_base + k, None, None
                ]
                h_rows = tile_slice(h_state, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_matmul.load_state(state_l1)
                s2_matmul.compute_w_state(w_l1)
                s2_matmul.compute_q_state(q_l1)
                s2_matmul.load_k_restored_operand(k_restored_l1, valid_rows)
                s2_matmul.load_mqk_operand(mqk_l1, valid_rows)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_q_chunk = tile_slice(gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_w_chunk = tile_slice(
                        gm_w[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (k + 1, 0),
                    )
                    next_gamma_col_full = tile_slice(gm_gamma, (SUPPORTED_HEAD_DIM, 1), (next_linear, 0))
                    next_gamma_col = tile_slice(next_gamma_col_full, (half_dk, 1), (subblock_idx, 0))
                    s2_matmul.load_q_decayed(q_l1, next_q_chunk)
                    s2_matmul.load_k_cumdecay(w_l1, next_w_chunk, next_valid_rows)
                    s2_vector.load_gamma_col(gamma_ub, next_gamma_col)
                    next_u_tile_full = tile_slice(
                        gm_u[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (k + 1, 0),
                    )
                    next_u_tile = tile_slice(next_u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx))
                    s2_vector.issue_u_rows(next_u_tile, subblock_idx, next_valid_rows)
                s2_matmul.store_u_delta(tmp_u_delta)
                s2_vector.finish_u_delta_to_l1(u_l1, tmp_u_delta, subblock_idx, v_new_chunk, valid_rows)
                u_slot = u_l1.next_read()
                s2_matmul.update_state(u_slot, valid_rows)
                s2_matmul.store_state_delta(tmp_delta)
                s2_matmul.accumulate_mqk_u(u_slot, valid_rows)
                if group_start == 0 or chunk_id > 0:
                    s2_vector.store_h(h_rows)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_k_restored_chunk = tile_slice(
                        gm_kg[0, seq_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (k + 1, 0),
                    )
                    next_mqk_tile = tile_slice(
                        gm_Mqk[0, seq_idx, k + 1, None, None], (CHUNK_SIZE, CHUNK_SIZE), (0, 0),
                    )
                    s2_matmul.load_k_restored(k_restored_l1, next_k_restored_chunk, next_valid_rows)
                    s2_matmul.load_mqk(mqk_l1, next_mqk_tile, next_valid_rows)
                s2_matmul.store_output(o_chunk)
                s2_vector.scale_and_add_delta(tmp_delta, gamma_ub)
                if chunk_id + 1 < group_count:
                    s2_vector.write_state_l1_snapshot(state_l1, subblock_idx)
                elif k + 1 < chunks_total:
                    next_h_state = gm_h[
                        state_batch_idx, seq_idx, state_chunk_base + k + 1, None, None
                    ]
                    next_h_rows = tile_slice(next_h_state, (dv_base, half_dk), (dv_idx, subblock_idx))
                    s2_vector.store_group_state_to_h(next_h_rows)
                elif save_final_state != 0:
                    state_rows = tile_slice(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                    s2_vector.store_state_to_gm(state_rows)


STAGE1_BOUNDARY_FLAG_ID = 0
STAGE2_BOUNDARY_FLAG_ID = 1
_STAGE_BOUNDARY_AIC_FLAG_OFFSET = 4
_STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET = 8
QK_PUBLISH_FLAG_ID = 10
QK_CONSUME_FLAG_ID = 11
MTE2_TO_V_EVENT_ID = 0
MTE2_TO_V_V_STAGING_EVENT_ID = 1
MTE2_TO_V_SUBBLOCK1_EVENT_ID = 4
QK_V_TO_MTE3_EVENT_ID = 2
QK_MTE2_TO_V_EVENT_ID = 3
GATE_MTE2_TO_V_EVENT_ID = 5


def _rewind_buffers():
    onchip_rewind(reset_sync_id=False)


def _stage_boundary_reset(flag_id: int):
    cube_sync_all()
    vec_sync_all()
    aiv_ready_flag_id = flag_id
    aic_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_FLAG_OFFSET
    aic_aiv_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET
    vec_sync_block_arrive(PIPE.MTE3, aiv_ready_flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, aiv_ready_flag_id, mode=2)
    cube_sync_block_arrive(PIPE.FIXPIPE, aic_ready_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, aic_ready_flag_id, mode=0)
    cube_sync_block_arrive(PIPE.MTE3, aic_aiv_ready_flag_id, mode=2)
    vec_sync_block_wait(PIPE.S, aic_aiv_ready_flag_id, mode=2)
    _rewind_buffers()


def _tnd_state_chunk_capacity(storage_length: int, batch: int) -> int:
    return (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch


def _tnd_state_chunk_base(token_start: int, logical_batch: int) -> int:
    return int(token_start) // CHUNK_SIZE + int(logical_batch)


def _public_bnsd_tensor(tensor: torch.Tensor, layout_qkv: str) -> torch.Tensor:
    if layout_qkv == "TND":
        return tensor[0].transpose(0, 1).contiguous()
    return tensor


def _public_bnsd_scalar(tensor: torch.Tensor, layout_qkv: str) -> torch.Tensor:
    if layout_qkv == "TND":
        return tensor[0].transpose(0, 1).contiguous()
    return tensor


def _public_chunk_tensor(
    tensor: torch.Tensor, layout_qkv: str, width: int
) -> torch.Tensor:
    if tensor.dim() == 5:
        return (
            tensor[0].transpose(0, 1).contiguous()
            if layout_qkv == "TND"
            else tensor.contiguous()
        )
    batch, heads, storage_length, _ = tensor.shape
    chunk_count = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded_length = chunk_count * CHUNK_SIZE
    if padded_length != storage_length:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padded_length - storage_length))
    chunks = tensor.reshape(batch, heads, chunk_count, CHUNK_SIZE, width)
    return chunks[0].contiguous() if layout_qkv == "TND" else chunks.contiguous()


def _public_chunk_state(tensor: torch.Tensor, layout_qkv: str) -> torch.Tensor:
    return (
        tensor[0].transpose(0, 1).contiguous()
        if layout_qkv == "TND"
        else tensor.contiguous()
    )


@jit
def _run_stage2_round(
    gm_kg,
    gm_gamma,
    gm_q_decayed,
    gm_saved_w,
    gm_u,
    gm_v_new,
    gm_h,
    gm_o,
    gm_state_in,
    gm_state_out,
    gm_mqk,
    gm_metadata,
    *,
    dv_base: int,
    value_heads,
    active_batch_num,
    batch_idx_base,
    storage_batch_idx_base,
    token_start_base,
    valid_seq_len_base,
    chunk_start_base,
    chunk_num_base,
    stage1_task_prefix_base,
    stage2_task_prefix_base,
    stage2_core_start,
    stage2_core_end,
    logical_core_num,
    resources,
    state_is_compact,
    has_initial_state,
    save_final_state,
):
    for active_batch_idx in range(0, active_batch_num):
        prefix_start = gm_metadata[stage2_task_prefix_base + active_batch_idx]
        prefix_end = gm_metadata[stage2_task_prefix_base + active_batch_idx + 1]
        intersection_start = stage2_core_start
        if intersection_start < prefix_start:
            intersection_start = prefix_start
        intersection_end = stage2_core_end
        if intersection_end > prefix_end:
            intersection_end = prefix_end
        item_start = 0
        item_end = 0
        if intersection_start < intersection_end:
            item_start = intersection_start - prefix_start
            item_end = intersection_end - prefix_start
        logical_batch = gm_metadata[batch_idx_base + active_batch_idx]
        storage_batch = gm_metadata[storage_batch_idx_base + active_batch_idx]
        storage_batch_end = storage_batch + 1
        token_start = gm_metadata[token_start_base + active_batch_idx]
        seq_len = dtypes.int64(gm_metadata[valid_seq_len_base + active_batch_idx])
        group_start = gm_metadata[chunk_start_base + active_batch_idx]
        chunk_num_per_group = gm_metadata[chunk_num_base + active_batch_idx]
        chunks_total = (seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE
        chunk_storage_base = 0
        if state_is_compact != 0:
            chunk_storage_base = token_start // CHUNK_SIZE + logical_batch
        token_stop = token_start + seq_len
        o = gm_o[storage_batch, None, token_start:token_stop, None]
        kg = gm_kg[storage_batch, None, token_start:token_stop, None]
        saved_w = gm_saved_w[storage_batch, None, token_start:token_stop, None]
        u = gm_u[storage_batch, None, token_start:token_stop, None]
        mqk = gm_mqk[
            storage_batch:storage_batch_end, None, chunk_storage_base:, None, None
        ]
        v_new = gm_v_new[storage_batch, None, token_start:token_stop, None]
        workspace_slot_start = gm_metadata[stage1_task_prefix_base + active_batch_idx]
        _run_stage2_body(
            kg, gm_gamma, gm_q_decayed, saved_w, u, v_new, gm_h, o, gm_state_in, gm_state_out, mqk,
            workspace_task_base=workspace_slot_start, dv_base=dv_base, head_base=0, head_count=value_heads,
            chunks_total=chunks_total, seq_len=seq_len, group_start=group_start, group_count=chunk_num_per_group,
            eff_group=chunk_num_per_group, logical_core_num=logical_core_num, item_start=item_start, item_end=item_end,
            resources=resources, state_batch=logical_batch,
            has_initial_state=has_initial_state, save_final_state=save_final_state,
            state_batch_idx=0 if state_is_compact != 0 else logical_batch,
            state_chunk_base=chunk_storage_base if state_is_compact != 0 else 0,
        )


@jit
def _run_stage2_dynamic(
    gm_kg,
    gm_gamma,
    gm_q_decayed,
    gm_saved_w,
    gm_u,
    gm_v_new,
    gm_h,
    gm_o,
    gm_state_in,
    gm_state_out,
    gm_mqk,
    gm_metadata,
    *,
    dv_splits_num,
    value_heads,
    active_batch_num,
    batch_idx_base,
    storage_batch_idx_base,
    token_start_base,
    valid_seq_len_base,
    chunk_start_base,
    chunk_num_base,
    stage1_task_prefix_base,
    stage2_task_prefix_base,
    stage2_core_start,
    stage2_core_end,
    logical_core_num,
    state_is_compact,
    has_initial_state,
    save_final_state,
):
    resources = _StageTwoResources()
    if dv_splits_num == 1:
        _run_stage2_round(
            gm_kg, gm_gamma, gm_q_decayed, gm_saved_w, gm_u, gm_v_new, gm_h, gm_o, gm_state_in, gm_state_out, gm_mqk,
            gm_metadata, dv_base=128, value_heads=value_heads, active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base, storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base, valid_seq_len_base=valid_seq_len_base, chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base, stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base, stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end, logical_core_num=logical_core_num, resources=resources,
            state_is_compact=state_is_compact, has_initial_state=has_initial_state, save_final_state=save_final_state,
        )
    elif dv_splits_num == 2:
        _run_stage2_round(
            gm_kg, gm_gamma, gm_q_decayed, gm_saved_w, gm_u, gm_v_new, gm_h, gm_o, gm_state_in, gm_state_out, gm_mqk,
            gm_metadata, dv_base=64, value_heads=value_heads, active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base, storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base, valid_seq_len_base=valid_seq_len_base, chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base, stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base, stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end, logical_core_num=logical_core_num, resources=resources,
            state_is_compact=state_is_compact, has_initial_state=has_initial_state, save_final_state=save_final_state,
        )
    elif dv_splits_num == 4:
        _run_stage2_round(
            gm_kg, gm_gamma, gm_q_decayed, gm_saved_w, gm_u, gm_v_new, gm_h, gm_o, gm_state_in, gm_state_out, gm_mqk,
            gm_metadata, dv_base=32, value_heads=value_heads, active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base, storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base, valid_seq_len_base=valid_seq_len_base, chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base, stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base, stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end, logical_core_num=logical_core_num, resources=resources,
            state_is_compact=state_is_compact, has_initial_state=has_initial_state, save_final_state=save_final_state,
        )
    else:
        _run_stage2_round(
            gm_kg, gm_gamma, gm_q_decayed, gm_saved_w, gm_u, gm_v_new, gm_h, gm_o, gm_state_in, gm_state_out, gm_mqk,
            gm_metadata, dv_base=16, value_heads=value_heads, active_batch_num=active_batch_num,
            batch_idx_base=batch_idx_base, storage_batch_idx_base=storage_batch_idx_base,
            token_start_base=token_start_base, valid_seq_len_base=valid_seq_len_base, chunk_start_base=chunk_start_base,
            chunk_num_base=chunk_num_base, stage1_task_prefix_base=stage1_task_prefix_base,
            stage2_task_prefix_base=stage2_task_prefix_base, stage2_core_start=stage2_core_start,
            stage2_core_end=stage2_core_end, logical_core_num=logical_core_num, resources=resources,
            state_is_compact=state_is_compact, has_initial_state=has_initial_state, save_final_state=save_final_state,
        )


@jit
def run_staged_flash_kda_fwd(
    gm_K: Tensor,
    gm_V: Tensor,
    gm_Q: Tensor,
    gm_beta: Tensor,
    gm_g: Tensor,
    gm_qk_exchange: Tensor,
    gm_A_log: Tensor,
    gm_dt_bias: Tensor,
    gm_kg: Tensor,
    gm_gamma: Tensor,
    gm_q_decayed: Tensor,
    gm_u: Tensor,
    gm_saved_w: Tensor,
    gm_qg: Tensor,
    gm_g_cumsum: Tensor,
    gm_v_new: Tensor,
    gm_h: Tensor,
    gm_O: Tensor,
    gm_state_in: Tensor,
    gm_state_out: Tensor,
    gm_identity16: Tensor,
    gm_q_norm: Tensor,
    gm_k_norm: Tensor,
    gm_q_rstd: Tensor,
    gm_k_rstd: Tensor,
    gm_beta_activated: Tensor,
    gm_Aqk: Tensor,
    gm_Akk: Tensor,
    gm_metadata: Tensor,
    scale_value: float,
    lower_bound: float,
    use_qk_l2norm_in_kernel: int,
    use_gate_in_kernel: int,
    use_beta_sigmoid_in_kernel: int,
    allow_neg_eigval: int,
    has_lower_bound: int,
    state_is_compact: int,
    save_qg: int,
    save_g_cumsum: int,
    save_intermediate_state: int,
    save_matrices: int,
    has_initial_state: int,
    save_final_state: int,
):
    block_idx = get_block_idx()
    block_num = get_block_num()
    metadata_status = gm_metadata[HEADER_STATUS]
    metadata_core_num = gm_metadata[HEADER_CORE_NUM]
    if metadata_status == STATUS_OK and metadata_core_num == block_num:
        stage12_round_num = gm_metadata[HEADER_STAGE12_ROUND_NUM]
        for global_round_idx in range(0, stage12_round_num):
            record = gm_metadata[HEADER_WORDS + global_round_idx]
            active_batch_num = gm_metadata[record + 1]
            batch_idx_base = record + 2
            storage_batch_idx_base = batch_idx_base + active_batch_num
            token_start_base = storage_batch_idx_base + active_batch_num
            valid_seq_len_base = token_start_base + active_batch_num
            group_num_index = valid_seq_len_base + active_batch_num
            chunk_start_base = group_num_index + 1
            chunk_num_base = chunk_start_base + active_batch_num
            stage1_task_prefix_base = chunk_num_base + active_batch_num
            stage1_core_ranges_base = stage1_task_prefix_base + active_batch_num + 1
            dv_splits_num_base = stage1_core_ranges_base + MAX_AIC_CORES * 2
            stage2_task_prefix_base = dv_splits_num_base + active_batch_num
            stage2_core_ranges_base = stage2_task_prefix_base + active_batch_num + 1
            stage1_core_range_word = stage1_core_ranges_base + block_idx * 2
            stage1_core_start = gm_metadata[stage1_core_range_word]
            stage1_core_end = gm_metadata[stage1_core_range_word + 1]
            for active_batch_idx in range(0, active_batch_num):
                prefix_start = gm_metadata[stage1_task_prefix_base + active_batch_idx]
                prefix_end = gm_metadata[stage1_task_prefix_base + active_batch_idx + 1]
                intersection_start = stage1_core_start
                if intersection_start < prefix_start:
                    intersection_start = prefix_start
                intersection_end = stage1_core_end
                if intersection_end > prefix_end:
                    intersection_end = prefix_end
                task_start = 0
                task_end = 0
                if intersection_start < intersection_end:
                    task_start = intersection_start - prefix_start
                    task_end = intersection_end - prefix_start
                logical_batch = gm_metadata[batch_idx_base + active_batch_idx]
                storage_batch = gm_metadata[storage_batch_idx_base + active_batch_idx]
                storage_batch_end = storage_batch + 1
                token_start = gm_metadata[token_start_base + active_batch_idx]
                seq_len = dtypes.int64(gm_metadata[valid_seq_len_base + active_batch_idx])
                group_start = gm_metadata[chunk_start_base + active_batch_idx]
                chunk_num_per_group = gm_metadata[chunk_num_base + active_batch_idx]
                chunks_total = (seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE
                tail_rows = seq_len - (chunks_total - 1) * CHUNK_SIZE
                tail_group = group_start + chunk_num_per_group == chunks_total
                chunk_storage_base = 0
                if state_is_compact != 0:
                    chunk_storage_base = token_start // CHUNK_SIZE + logical_batch
                token_stop = token_start + seq_len
                k = gm_K[storage_batch, None, token_start:token_stop, None]
                v = gm_V[storage_batch, None, token_start:token_stop, None]
                q = gm_Q[storage_batch, None, token_start:token_stop, None]
                beta = gm_beta[storage_batch, None, token_start:token_stop, None]
                gate = gm_g[storage_batch, None, token_start:token_stop, None]
                value_heads = v.shape[1]
                gate_batch, gate_start, gate_stop = storage_batch, token_start, token_stop
                if save_g_cumsum == 0:
                    gate_batch, gate_start, gate_stop = 0, 0, 1
                gate_cache = gm_g_cumsum[gate_batch:gate_batch + 1, None, gate_start:gate_stop, None]
                _run_stage1_body(
                    tail_group, tail_rows, k, v, q, beta, gate, gm_qk_exchange, gm_A_log, gm_dt_bias, gm_q_decayed,
                    gm_kg[storage_batch, None, token_start:token_stop, None], gm_gamma,
                    gm_u[storage_batch, None, token_start:token_stop, None],
                    gm_saved_w[storage_batch, None, token_start:token_stop, None],
                    gm_qg[storage_batch, None, token_start:token_stop, None],
                    gate_cache, gm_identity16,
                    gm_q_norm[storage_batch, None, token_start:token_stop, None],
                    gm_k_norm[storage_batch, None, token_start:token_stop, None],
                    gm_q_rstd[storage_batch, None, token_start:token_stop, None],
                    gm_k_rstd[storage_batch, None, token_start:token_stop, None],
                    gm_beta_activated[storage_batch, None, token_start:token_stop, None],
                    gm_Aqk[storage_batch:storage_batch_end, None, chunk_storage_base:, None, None],
                    gm_Akk[storage_batch:storage_batch_end, None, chunk_storage_base:, None, None], prefix_start,
                    value_heads, chunks_total, seq_len, 0, group_start, chunk_num_per_group, chunk_num_per_group,
                    block_num, task_start, task_end, scale_value, lower_bound, use_qk_l2norm_in_kernel,
                    use_gate_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval, has_lower_bound, save_qg,
                    save_g_cumsum, save_matrices,
                )
            _stage_boundary_reset(STAGE1_BOUNDARY_FLAG_ID)
            stage2_core_range_word = stage2_core_ranges_base + block_idx * 2
            stage2_core_start = gm_metadata[stage2_core_range_word]
            stage2_core_end = gm_metadata[stage2_core_range_word + 1]
            round_dv_splits_num = gm_metadata[dv_splits_num_base]
            _run_stage2_dynamic(
                gm_kg, gm_gamma, gm_q_decayed, gm_saved_w, gm_u, gm_v_new, gm_h, gm_O, gm_state_in, gm_state_out,
                gm_Aqk, gm_metadata, dv_splits_num=round_dv_splits_num, value_heads=gm_V.shape[1],
                active_batch_num=active_batch_num, batch_idx_base=batch_idx_base,
                storage_batch_idx_base=storage_batch_idx_base, token_start_base=token_start_base,
                valid_seq_len_base=valid_seq_len_base, chunk_start_base=chunk_start_base, chunk_num_base=chunk_num_base,
                stage1_task_prefix_base=stage1_task_prefix_base, stage2_task_prefix_base=stage2_task_prefix_base,
                stage2_core_start=stage2_core_start, stage2_core_end=stage2_core_end, logical_core_num=block_num,
                state_is_compact=state_is_compact, has_initial_state=has_initial_state, save_final_state=save_final_state,
            )
            if global_round_idx + 1 < stage12_round_num:
                _stage_boundary_reset(STAGE2_BOUNDARY_FLAG_ID)
            cube_sync_all()
            vec_sync_all()


def staged_flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Optional[torch.Tensor], ...]:

    return fwd._flash_kda_fwd_impl(
        q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
        cu_seqlens=cu_seqlens, seqused=seqused, scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, disable_recompute=disable_recompute,
        lower_bound=lower_bound, layout=layout,
        internal_template=TEMPLATE_STAGED,
    )


@export("staged_flash_kda")
def export_staged_flash_kda():
    """收集直接模板编译请求，不启动 Provider。"""
    try:
        fwd._get_compiled_kernel(TEMPLATE_STAGED)
    finally:
        fwd.clear_caches()


def cast_and_pack_bf16(value, lane_mask):
    value_bf16 = vcast(value, dtypes.bfloat16, mask=lane_mask)
    packed_value = vpack(vreinterpret_lanes(value_bf16, dtypes.uint32), dtypes.uint16)
    return vreinterpret(packed_value, dtypes.bfloat16)
