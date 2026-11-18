# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""训练前向的分阶段 chunk 分组调度、矩阵和向量资源及序列视图。"""

import dataclasses
import os
from typing import NamedTuple, Optional, Tuple

from cannbotdsl import PIPE, MemLoc, Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang import vf
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, cube_fill_l1_zero, mem_copy
from cannbotdsl.ops.reg import vmem_bar, vpack, vreinterpret, vreinterpret_lanes
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
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    cube_sync_intra_arrive,
    cube_sync_intra_wait,
    cube_sync_pipe,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_intra_arrive,
    vec_sync_intra_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import (
    from_tensor,
    idx2crd,
    make_layout,
    tile_slice,
    tile_slice_multi,
)
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


# Set only while compiling under the entry module compilation lock.
_COMPILE_DV_SPLITS = 4

L2_NORM_EPS = 1e-6
MAX_INVERSE_NORM = L2_NORM_EPS**-0.5
NZ_TILE = 16
NZ_TILE_ELEMENTS = NZ_TILE * NZ_TILE
TRANSFER_BLOCK_BITS = 256


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


def _padded_nz_layout(shape):
    """Physical BF16/FP16 NZ layout with a 16-element N1 stride gap.

    Keep the existing 32-byte padding between N1 planes. The gap is in
    elements, not rows or fractals; logical extents remain unchanged.
    """
    rows, cols = shape
    m1 = (rows + NZ_TILE - 1) // NZ_TILE
    n1 = (cols + NZ_TILE - 1) // NZ_TILE
    return make_layout(
        (n1, m1, NZ_TILE, NZ_TILE),
        stride=(m1 * NZ_TILE_ELEMENTS + NZ_TILE, NZ_TILE_ELEMENTS, NZ_TILE, 1),
    )


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
                        value = vload_unpack(
                            src,
                            row * src_stride + col,
                            unpack_mode=UnpackMode.B16_TO_B32,
                        )
                    converted = vcast(value, dst.dtype, mask=mask)
                    if const_expr(dst.dtype == dtypes.float32):
                        vstore(dst, row * dst_stride + col, converted, mask)
                    else:
                        vstore_pack(
                            dst,
                            row * dst_stride + col,
                            converted,
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )


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
    full_chunk = tile_slice(
        gm_O[batch_idx, head_idx, None, None],
        (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
        (chunk_idx, 0),
    )
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
        for row in dsl_range(0, rows):
            remaining = rr.mask_counter(cols)
            for col in range_constexpr(0, cols, 2048 // elem_bits):
                mask, remaining = rr.update_mask(remaining, elem_bits=elem_bits)
                values = rr.vload(src, row * src_stride + col)
                offset = row * n0 + col // n0 * dst_stride
                rr.vstore_strided(
                    dst,
                    offset,
                    values,
                    mask,
                    block_stride=block_stride,
                    repeat_stride=0,
                )


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
    if (
        not isinstance(active_block_num, int)
        or isinstance(active_block_num, bool)
        or active_block_num <= 0
    ):
        raise ValueError("active_block_num must be a positive integer")
    return _empty_like_device(
        (active_block_num, 2, 2, CHUNK_SIZE), ref=ref, dtype=torch.float32
    )


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
    if (
        not isinstance(block_idx, int)
        or isinstance(block_idx, bool)
        or not 0 <= block_idx < workspace.shape[0]
    ):
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
    ref: torch.Tensor,
    *,
    identity_dtype: torch.dtype = torch.float16,
    capture: bool = False,
) -> _Stage1Constants:
    device_type, device_index = _resolved_device_key(ref)
    key = (device_type, device_index, identity_dtype)
    cached = _STAGE1_CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached
    if capture:
        raise ValueError(
            "warm up FlashKDA arch92 constants on the target device before capture"
        )
    constants = _Stage1Constants(
        identity16=torch.eye(
            NEUMANN_BLOCK_SIZE, dtype=identity_dtype, device=ref.device
        ).contiguous()
    )
    _STAGE1_CONSTANTS_CACHE[key] = constants
    return constants


def allocate_stage1_workspace(
    batch: int,
    heads: int,
    eff_group: int,
    dim: int,
    *,
    ref: torch.Tensor,
    capture: bool = False,
) -> StageOneWorkspace:
    bn_chunks = batch * heads * eff_group
    constants = _stage1_constants_for(ref, capture=capture)
    return StageOneWorkspace(
        Q_decayed=_empty_like_device(
            (bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=torch.bfloat16
        ),
        gamma_C=_empty_like_device((bn_chunks * dim, 1), ref=ref, dtype=torch.float32),
        identity16=constants.identity16,
    )


ELEM_BYTES = 4
VL = 64
NEUMANN_BLOCK_SIZE = 16
NEUMANN_DIAG_BLOCK_NUM = 4
NEUMANN_FRACTAL_ELEMS = NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
NEUMANN_DIAG_SRC_STRIDE = NEUMANN_BLOCK_SIZE * CHUNK_SIZE + NEUMANN_BLOCK_SIZE
NEUMANN_DIAG_DST_STRIDE = (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS


class StageOneMatmul:
    """StageOne Cube 侧矩阵乘计算。"""

    def __init__(self, *, split_fixpipe: bool = False):
        self.split_fixpipe = split_fixpipe
        self.tile_power_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.packed_inv64_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.tile_inv_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.scratch_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.neumann_power_handoff_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.packed_identity_l1 = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.zero_l1 = Buffer(
            MemLoc.L1, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE))
        )
        self.resident_inv_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.l0a_db = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.l0b_db = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.l0c_db = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.per_d_left_k_l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        ).next_write()
        self.per_d_left_q_l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        ).next_write()
        self.per_d_l0b_db = RingBuffer(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.raw_l0a_db = RingBuffer(
            MemLoc.L0A, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2
        )
        self.raw_l0b_db = RingBuffer(
            MemLoc.L0B, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2
        )
        self.raw_l0c_main = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.raw_l0c_aux = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()

    def load_per_d_left_k(self, l1_channel):
        mem_copy(self.per_d_left_k_l0a, l1_channel.next_read())

    def load_per_d_left_q(self, l1_channel):
        mem_copy(self.per_d_left_q_l0a, l1_channel.next_read())

    def load_per_d_right_k(self, l1_channel):
        right = self.per_d_l0b_db.next_write()
        mem_copy(right, l1_channel.next_read())
        return right

    def _per_d_l0c_tile(self, l0c, row_block: int, col_block: int):
        fractal = col_block * NEUMANN_DIAG_BLOCK_NUM + row_block
        return from_tensor(
            l0c,
            offset=fractal * NEUMANN_FRACTAL_ELEMS * 4,
            data_format="nz",
            layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
        )

    def mm_per_d_kkt(self, row_block: int, col_block: int, right, init=True):
        matmul(
            self._per_d_l0c_tile(self.raw_l0c_main, row_block, col_block),
            self.per_d_left_k_l0a,
            right,
            init=init,
        )

    def mm_per_d_mqk(self, row_block: int, col_block: int, right, init=True):
        matmul(
            self._per_d_l0c_tile(self.raw_l0c_aux, row_block, col_block),
            self.per_d_left_q_l0a,
            right,
            init=init,
        )

    def _l0c_fixpipe_source(self, src):
        if self.split_fixpipe:
            return src.partition(dim=0)
        return src

    def finish_per_d_kkt(self, ub_kkt):
        mem_copy(ub_kkt, self._l0c_fixpipe_source(self.raw_l0c_main))

    def finish_per_d_mqk(self, ub_mqk):
        mem_copy(ub_mqk, self._l0c_fixpipe_source(self.raw_l0c_aux))

    def packed64_block(self, l1_matrix, row_block: int, col_block: int):
        return (None, l1_matrix, row_block, col_block)

    def _packed64_block_offset_elems(self, row_block: int, col_block: int):
        return (
            col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
            + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
        )

    def l0_matrix_block_offset_elems(self, row_block: int, col_block: int):
        return (row_block * NEUMANN_DIAG_BLOCK_NUM + col_block) * NEUMANN_FRACTAL_ELEMS

    def c64_diag_nz_offset_elems(self, diag_block: int):
        return diag_block * (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS

    def c64_odd_even_pair_dst_gap_fractals(self):
        return self.c64_diag_nz_offset_elems(2) // NEUMANN_FRACTAL_ELEMS - 1

    def _raw_l0a_block(self, l0a_slot, row_block: int, col_block: int):
        # NZ stores the column block on the outer physical axis.
        return from_tensor(
            l0a_slot,
            offset=self._packed64_block_offset_elems(row_block, col_block) * 2,
            layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
        )

    def _load_l1_block_to_l0a_block(
        self, l1_block, l0a_slot, row_block: int, col_block: int
    ):
        """把一个 packed L1 16x16 fractal 装入 full64 L0A 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = self._raw_l0a_block(l0a_slot, row_block, col_block)
        l1_tile = from_tensor(
            l1_parent,
            offset=(
                col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
                + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
            )
            * 2,
            layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
        )
        mem_copy(l0_tile, l1_tile)

    def _load_l1_block_to_l0b_block(
        self, l1_block, l0b_slot, row_block: int, col_block: int
    ):
        """把一个 packed L1 16x16 fractal 转置装入 full64 L0B 的指定块位置。"""
        _, l1_parent, _, _ = l1_block
        l0_tile = from_tensor(
            l0b_slot,
            offset=self.l0_matrix_block_offset_elems(row_block, col_block) * 2,
            data_format="zn",
            layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
        )
        l1_tile = from_tensor(
            l1_parent,
            offset=(
                col_block * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
                + row_block * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
            )
            * 2,
            layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
        )
        mem_copy(l0_tile, l1_tile)

    def _load_packed32_block_to_full64_l0b_raw(
        self,
        l1_matrix,
        l0b_slot,
        *,
        src_row: int,
        dst_row: int,
        dst_col: int,
        transpose: bool = True,
    ):
        row_shape = (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE)
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            # The two horizontal fractals are separated by the parent NZ pitch.
            l1_tile = tile_slice(
                l1_matrix, row_shape, ((src_row + row_offset) // NEUMANN_BLOCK_SIZE, 0)
            )
            l0_offset = self.l0_matrix_block_offset_elems(
                (dst_row + row_offset) // NEUMANN_BLOCK_SIZE,
                dst_col // NEUMANN_BLOCK_SIZE,
            )
            if transpose:
                l0_tile = from_tensor(
                    l0b_slot,
                    offset=l0_offset * 2,
                    data_format="zn",
                    layout=make_layout(row_shape),
                )
            else:
                l0_tile = from_tensor(
                    l0b_slot, offset=l0_offset * 2, layout=make_layout(row_shape)
                )
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
            src_offset = self._packed64_block_offset_elems(src_row_block, src_col_block)
            dst_offset = (
                dst_block - (NEUMANN_DIAG_BLOCK_NUM - 1)
            ) * NEUMANN_FRACTAL_ELEMS
            l0_tile = from_tensor(
                l0a_slot,
                offset=dst_offset * 2,
                layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
            )
            l1_tile = from_tensor(
                l1_matrix,
                offset=src_offset * 2,
                layout=make_layout((NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)),
            )
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
            self.packed64_block(l1_matrix, src0_row_block, src0_col_block),
            l0b_slot,
            dst0_block // NEUMANN_DIAG_BLOCK_NUM,
            dst0_block % NEUMANN_DIAG_BLOCK_NUM,
        )
        self._load_l1_block_to_l0b_block(
            self.packed64_block(l1_matrix, src1_row_block, src1_col_block),
            l0b_slot,
            dst1_block // NEUMANN_DIAG_BLOCK_NUM,
            dst1_block % NEUMANN_DIAG_BLOCK_NUM,
        )

    def zero_half_l1(self, l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    def zero_packed64_l0_operands(self, l0a_slot, l0b_slot):
        mem_copy(l0a_slot, self.zero_l1)
        mem_copy(l0b_slot, self.zero_l1)

    def load_packed64_diag_only_operands_to_l0(
        self, left_l1, right_l1, l0a_slot, l0b_slot, *, clear_l0: bool = True
    ):
        if clear_l0:
            self.zero_packed64_l0_operands(l0a_slot, l0b_slot)
        block = (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)
        mem_copy(
            tile_slice_multi(
                l0a_slot, block, repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16)
            ),
            tile_slice_multi(
                left_l1, block, repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16)
            ),
        )
        mem_copy(
            tile_slice_multi(
                l0b_slot, block, repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16)
            ),
            tile_slice_multi(
                right_l1.permute(1, 0),
                block,
                repeats=NEUMANN_DIAG_BLOCK_NUM,
                step=(16, 16),
            ),
        )

    def issue_packed64_diag_only_operands_to_l0_db(
        self, left_l1, right_l1, *, clear_l0: bool = True
    ):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.load_packed64_diag_only_operands_to_l0(
            left_l1, right_l1, l0a_write, l0b_write, clear_l0=clear_l0
        )

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
        self.issue_packed64_diag_only_operands_to_l0_db(
            left_l1, right_l1, clear_l0=clear_l0
        )
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
        self.issue_packed64_diag_only_operands_to_l0_db(
            left_l10, right_l10, clear_l0=clear_l0
        )
        self.issue_packed64_diag_only_operands_to_l0_db(
            left_l11, right_l11, clear_l0=clear_l0
        )
        self.begin_mmad_raw_l0ab_db_packed64(
            c_tile, init_c=init_c0, unit_flag=unit_flag0
        )
        self.begin_mmad_raw_l0ab_db_packed64(
            c_tile, init_c=init_c1, unit_flag=unit_flag1
        )

    def fixpipe_packed64_full_to_half_l1(self, dst_channel, c_tile):
        mem_copy(dst_channel, c_tile, copy_options(unit_flag=3))

    def fixpipe_packed64_full_to_bf16_l1(self, dst_channel, c_tile):
        mem_copy(dst_channel, c_tile, copy_options(unit_flag=3))

    def half_l1_power_packed64_diag_only_on_c(
        self, left_l1, right_l1, dst_channel, *, clear_l0: bool
    ):
        c_power = self.raw_l0c_aux
        self.raw_mmad_packed64_diag_only(
            left_l1, right_l1, c_power, init_c=True, unit_flag=3, clear_l0=clear_l0
        )
        c_power_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(dst_channel, c_power_read)

    def prepare_packed64_identity_l1(self, gm_identity16):
        """构造 packed 64x64 identity。"""
        identity_write = self.packed_identity_l1
        self.zero_half_l1(identity_write)
        cube_sync_pipe(PIPE.MTE2)
        block = (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE)
        mem_copy(
            tile_slice_multi(
                identity_write, block, repeats=NEUMANN_DIAG_BLOCK_NUM, step=(16, 16)
            ),
            tile_slice_multi(
                gm_identity16, block, repeats=NEUMANN_DIAG_BLOCK_NUM, step=(0, 0)
            ),
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
        self.half_l1_power_packed64_diag_only_on_c(
            negative_t, negative_t, self.tile_inv_l1, clear_l0=True
        )
        power2 = self.tile_inv_l1
        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only_db_pair(
            scratch,
            identity,
            scratch,
            power2,
            c_inv,
            init_c0=True,
            unit_flag0=2,
            init_c1=False,
            unit_flag1=3,
            clear_l0=False,
        )
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return (power2, self.packed_inv64_l1)

    def neumann_diag_power4_update(self, power2, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(
            power2, power2, self.tile_power_l1, clear_l0=False
        )
        power4 = self.tile_power_l1
        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(
            packed_inv, power4, c_inv, init_c=False, unit_flag=3, clear_l0=False
        )
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return (power4, self.packed_inv64_l1)

    def neumann_diag_power8_update(self, power4, packed_inv):
        self.half_l1_power_packed64_diag_only_on_c(
            power4, power4, self.tile_inv_l1, clear_l0=False
        )
        power8 = self.tile_inv_l1
        c_inv = self.raw_l0c_main
        self.raw_mmad_packed64_diag_only(
            packed_inv, power8, c_inv, init_c=False, unit_flag=3, clear_l0=False
        )
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def issue_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap(
            scratch,
            l0a_write,
            src0_row_block=1,
            src0_col_block=0,
            src1_row_block=3,
            src1_col_block=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            packed_inv,
            l0b_write,
            src0_row_block=0,
            src0_col_block=0,
            src1_row_block=2,
            src1_col_block=2,
            dst_offset_elems=self.c64_diag_nz_offset_elems(0),
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals(),
        )

    def issue_odd_inv_odd_tmp_full64(self, packed_inv, tmp64):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        self._load_l1_block_to_l0a_block(
            self.packed64_block(packed_inv, 1, 1), l0a_write, 1, 1
        )
        self._load_l1_block_to_l0a_block(
            self.packed64_block(packed_inv, 3, 3), l0a_write, 3, 3
        )
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(
            tmp64,
            l0b_write,
            src0_row_block=1,
            src0_col_block=0,
            src1_row_block=3,
            src1_col_block=2,
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
        self._load_packed32_block_to_full64_l0b_raw(
            packed_inv,
            l0b_write,
            src_row=0,
            dst_row=0,
            dst_col=0,
            transpose=True,
        )

    def prepare_odd32_inv_odd32_tmp_full64(self, packed_inv):
        l0a_write = self.raw_l0a_db.next_write()
        l0b_write = self.raw_l0b_db.next_write()
        self.zero_packed64_l0_operands(l0a_write, l0b_write)
        mem_copy(l0a_write, packed_inv)
        return (l0a_write, l0b_write)

    def complete_odd32_inv_odd32_tmp_full64(self, tmp64, l0a_write, l0b_write):
        self._load_packed32_block_to_full64_l0b_raw(
            tmp64,
            l0b_write,
            src_row=HALF_CHUNK_SIZE,
            dst_row=HALF_CHUNK_SIZE,
            dst_col=0,
            transpose=True,
        )

    def compose_odd_even_lower32_to64_accum_full64(self, scratch, packed_inv):
        """组合 64x64 左下 32x32 lower 逆矩并生成 BF16 resident inverse。"""
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
        # store_inverse writes the optional public matrix first, then publishes
        # this same inverse once to the BF16 L1 consumer.

    def _load_resident_inv_to_l0a(self):
        resident_inv = self.resident_inv_l1
        l0a_write = self.l0a_db.next_write()
        mem_copy(
            from_tensor(
                l0a_write, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            resident_inv,
        )
        return l0a_write

    def _patch_resident_inv_tail(self, l0a_slot):
        """Reload the last inverse row block into the wide L0A window.

        Preserve the A6 reference's explicit tail transfer before both W and U.
        """
        inv = self.resident_inv_l1
        for col_block in range(NEUMANN_DIAG_BLOCK_NUM):
            self._load_l1_block_to_l0a_block(
                self.packed64_block(inv, 3, col_block), l0a_slot, 3, col_block
            )

    def _load_beta_to_l0b(self, beta_channel):
        beta_read = beta_channel
        l0b_write = self.l0b_db.next_write()
        l0b_zn = from_tensor(
            l0b_write,
            data_format="zn",
            layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
        )
        mem_copy(l0b_zn, beta_read)
        return l0b_zn

    def _mm_resident_inv_beta(self, l0a_resident, l0b_beta):
        l0c_write = self.l0c_db.next_write()
        matmul(
            l0c_write,
            from_tensor(
                l0a_resident, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            l0b_beta,
            init=True,
        )

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
                normalized_k_dhalf,
                normalized_q_dhalf,
                row_block,
                active_row,
                vector.l1_per_d,
            )
            self.load_per_d_left_k(vector.l1_per_d)
            self.load_per_d_left_q(vector.l1_per_d)
            vector.build_safe_diagonal_right(
                normalized_k_dhalf,
                gate_cumsum_dhalf,
                row_block,
                active_row,
                vector.l1_per_d,
            )
            right = self.load_per_d_right_k(vector.l1_per_d)
            self.mm_per_d_kkt(row_block, row_block, right, init=active_row == 0)
            self.mm_per_d_mqk(row_block, row_block, right, init=active_row == 0)

    @jit
    def store_inverse(self, gm_saved, head_idx, chunk_idx, valid_rows, save_matrices):
        if save_matrices != 0:
            saved = tile_slice(
                gm_saved[0, head_idx, None, None],
                (CHUNK_SIZE, CHUNK_SIZE),
                (chunk_idx, 0),
            )
            mem_copy(
                _valid_rows_view(saved, valid_rows, CHUNK_SIZE),
                self.raw_l0c_main[:valid_rows, :CHUNK_SIZE],
                options=copy_options(
                    unit_flag=2, l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL
                ),
            )
        self.fixpipe_packed64_full_to_bf16_l1(self.resident_inv_l1, self.raw_l0c_main)

    def compute_u_pre_and_w_from_resident_inv_beta(
        self, gm_u, gm_w, head_idx, chunk_idx, valid_rows, v_beta_l1, k_decayed_beta_l1
    ):
        l0a_resident = self._load_resident_inv_to_l0a()
        self._patch_resident_inv_tail(l0a_resident)
        l0b_beta = self._load_beta_to_l0b(k_decayed_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        gm_W_chunk = tile_slice(
            gm_w[0, head_idx, None, None],
            (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            (chunk_idx, 0),
        )
        mem_copy(
            _valid_rows_view(gm_W_chunk, valid_rows, SUPPORTED_HEAD_DIM),
            self.l0c_db.next_read()[:valid_rows, :SUPPORTED_HEAD_DIM],
            options=copy_options(l2_cache_ctl=_STAGE2_INPUT_L2_CACHE_CTL),
        )
        l0b_beta = self._load_beta_to_l0b(v_beta_l1)
        self._mm_resident_inv_beta(l0a_resident, l0b_beta)
        u_pre_part = tile_slice(
            gm_u[0, head_idx, None, None],
            (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            (chunk_idx, 0),
        )
        mem_copy(
            _valid_rows_view(u_pre_part, valid_rows, SUPPORTED_HEAD_DIM),
            self.l0c_db.next_read()[:valid_rows, :SUPPORTED_HEAD_DIM],
            options=copy_options(l2_cache_ctl=_L2_CACHE_LAST_USE),
        )


class StageOneVector:
    def __init__(self, *, scale_value):
        self.scale_value = scale_value
        self.ub_safe_left_k = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((16, 128)))
        self.ub_safe_left_q = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((16, 128)))
        self.ub_safe_right_k = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((16, 128))
        )
        d = SUPPORTED_HEAD_DIM
        self.ub_q = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, d)))
        self.ub_k = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, d)))
        self.ub_q_partials = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE))
        )
        self.ub_k_partials = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE))
        )
        self.ub_g_raw = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, d)), depth=1
        ).next_write()
        self.ub_cs = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, d)))
        self.ub_dt_bias = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, d)))
        self.ub_alpha = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1,)), depth=1
        ).next_write()
        self.ub_beta_raw = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, 1)), depth=1
        ).next_write()
        self.ub_beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, 1)))
        self.ub_v = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, d)), depth=1
        ).next_write()
        self.ub_gamma = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, d)), depth=1
        ).next_write()
        self.ub_cast = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, d)),
            depth=1,
            data_format="nd",
        ).next_write()
        self.ub_kkt = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.ub_kkt_fp16 = Buffer(
            MemLoc.UB, dtypes.float16, make_layout((CHUNK_SIZE, CHUNK_SIZE))
        )
        self.ub_mqk = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.ub_mqk_bf16 = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, CHUNK_SIZE)),
            depth=1,
            data_format="nd",
        ).next_write()
        self.ub_per_d_nd = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, d)), depth=1
        ).next_write()
        self.ub_per_d_nz = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((NEUMANN_BLOCK_SIZE, d)),
            depth=1,
            data_format="nz",
            physical_layout=_padded_nz_layout((NEUMANN_BLOCK_SIZE, d)),
        ).next_write()
        self.ub_nz = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, d)),
            depth=1,
            data_format="nz",
            physical_layout=_padded_nz_layout((CHUNK_SIZE, d)),
        ).next_write()
        self.ub_kkt_nz = RingBuffer(
            MemLoc.UB,
            dtypes.float16,
            make_layout((CHUNK_SIZE, CHUNK_SIZE)),
            depth=1,
            data_format="nz",
            physical_layout=_padded_nz_layout((CHUNK_SIZE, CHUNK_SIZE)),
        ).next_write()
        self.ub_decay = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((NEUMANN_BLOCK_SIZE, d))
        )
        self.l1_per_d = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((NEUMANN_BLOCK_SIZE, d)), depth=3
        )
        self.l1_v_beta = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, d)), depth=1
        ).next_write()
        self.l1_k_beta = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, d)), depth=1
        ).next_write()

    def _full_d(self, gm, task):
        return tile_slice(gm, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (task, 0))

    @jit
    def load_and_normalize_qk(
        self, gm_Q, gm_K, batch_idx, qk_head, chunk_idx, valid_rows, normalize
    ):
        self.issue_qk(gm_Q, gm_K, batch_idx, qk_head, chunk_idx, valid_rows)
        self.finish_qk(valid_rows, normalize)

    @jit
    def issue_qk(self, gm_Q, gm_K, batch_idx, qk_head, chunk_idx, valid_rows):
        """Load into the current Q/K buffers after their previous consumers drain."""
        d = SUPPORTED_HEAD_DIM
        q_tile = _valid_rows_view(
            tile_slice(
                gm_Q[batch_idx, qk_head, None, None], (CHUNK_SIZE, d), (chunk_idx, 0)
            ),
            valid_rows,
            d,
        )
        k_tile = _valid_rows_view(
            tile_slice(
                gm_K[batch_idx, qk_head, None, None], (CHUNK_SIZE, d), (chunk_idx, 0)
            ),
            valid_rows,
            d,
        )
        mem_copy(self.ub_q, q_tile, copy_options(l2_cache_ctl=_L2_CACHE_DISABLE))
        mem_copy(self.ub_k, k_tile, copy_options(l2_cache_ctl=_L2_CACHE_DISABLE))
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)

    @jit
    def finish_qk(self, valid_rows, normalize):
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16(self.ub_q, valid_rows)
            self._fill_bf16(self.ub_k, valid_rows)
        if normalize != 0:
            self._reduce_qk()
            self._inverse_from_sums()
            self._apply_inverse()

    @jit
    def _fill_bf16(self, ub, valid_rows):
        cols = SUPPORTED_HEAD_DIM
        with vf(mode="simd"):
            for col in range(0, cols, VL):
                lane_mask, _ = update_mask(cols - col, elem_bits=16)
                fill = vdups(0.0, dtypes.bfloat16, mask=lane_mask)
                for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                    vstore(ub, row * cols + col, fill, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def _reduce_qk(self):
        d = SUPPORTED_HEAD_DIM
        for row0 in range_constexpr(0, CHUNK_SIZE, 4):
            with vf(mode="simd"):
                acc_mask, _ = update_mask(VL, elem_bits=32)
                for row in range_constexpr(row0, row0 + 4):
                    q_acc = vdups(0.0, dtypes.float32, mask=acc_mask)
                    k_acc = vdups(0.0, dtypes.float32, mask=acc_mask)
                    for col in range_constexpr(0, d, VL):
                        col_mask, _ = update_mask(d - col, elem_bits=32)
                        offset = row * d + col
                        q_raw = vload_unpack(
                            self.ub_q, offset, unpack_mode=UnpackMode.B16_TO_B32
                        )
                        q_val = vcast(q_raw, dtypes.float32, mask=col_mask)
                        q_acc = vadd(
                            q_acc,
                            vreduce_sum(
                                vmul(q_val, q_val, mask=col_mask), mask=col_mask
                            ),
                            mask=acc_mask,
                        )
                        k_raw = vload_unpack(
                            self.ub_k, offset, unpack_mode=UnpackMode.B16_TO_B32
                        )
                        k_val = vcast(k_raw, dtypes.float32, mask=col_mask)
                        k_acc = vadd(
                            k_acc,
                            vreduce_sum(
                                vmul(k_val, k_val, mask=col_mask), mask=col_mask
                            ),
                            mask=acc_mask,
                        )
                    vstore_first(self.ub_q_partials, row, q_acc)
                    vstore_first(self.ub_k_partials, row, k_acc)
                vmem_bar("vst_vld")

    @jit
    def _inverse_from_sums(self):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            q_sum = vload(self.ub_q_partials, 0)
            q_inv = vdiv(
                one,
                vsqrt(vadds(q_sum, 1e-06, mask=lane_mask), mask=lane_mask),
                mask=lane_mask,
            )
            vstore(
                self.ub_q_partials,
                0,
                vmins(q_inv, MAX_INVERSE_NORM, mask=lane_mask),
                lane_mask,
            )
            k_sum = vload(self.ub_k_partials, 0)
            k_inv = vdiv(
                one,
                vsqrt(vadds(k_sum, 1e-06, mask=lane_mask), mask=lane_mask),
                mask=lane_mask,
            )
            vstore(
                self.ub_k_partials,
                0,
                vmins(k_inv, MAX_INVERSE_NORM, mask=lane_mask),
                lane_mask,
            )
            vmem_bar("vst_vld")

    @jit
    def _apply_inverse(self):
        d = SUPPORTED_HEAD_DIM
        for row0 in range_constexpr(0, CHUNK_SIZE, 4):
            with vf(mode="simd"):
                for col in range_constexpr(0, d, VL):
                    col_mask, _ = update_mask(d - col, elem_bits=32)
                    for row in range_constexpr(row0, row0 + 4):
                        offset = row * d + col
                        q_inv = vload_broadcast(self.ub_q_partials, row)
                        q_raw = vload_unpack(
                            self.ub_q, offset, unpack_mode=UnpackMode.B16_TO_B32
                        )
                        q_val = vmul(
                            vcast(q_raw, dtypes.float32, mask=col_mask),
                            q_inv,
                            mask=col_mask,
                        )
                        vstore_pack(
                            self.ub_q,
                            offset,
                            vcast(q_val, dtypes.bfloat16, mask=col_mask),
                            col_mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                        k_inv = vload_broadcast(self.ub_k_partials, row)
                        k_raw = vload_unpack(
                            self.ub_k, offset, unpack_mode=UnpackMode.B16_TO_B32
                        )
                        k_val = vmul(
                            vcast(k_raw, dtypes.float32, mask=col_mask),
                            k_inv,
                            mask=col_mask,
                        )
                        vstore_pack(
                            self.ub_k,
                            offset,
                            vcast(k_val, dtypes.bfloat16, mask=col_mask),
                            col_mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")

    @jit
    def store_q_decayed(self, gm_q_decayed, task):
        d = SUPPORTED_HEAD_DIM
        # Rows are independent; retain one completion barrier per produced tile.
        with vf(mode="simd"):
            for col in range_constexpr(0, d, VL):
                col_mask, _ = update_mask(d - col, elem_bits=32)
                for row in dsl_range(0, CHUNK_SIZE):
                    offset = row * d + col
                    gamma = vload(self.ub_cs, offset)
                    q_raw = vload_unpack(
                        self.ub_q, offset, unpack_mode=UnpackMode.B16_TO_B32
                    )
                    q_val = vcast(q_raw, dtypes.float32, mask=col_mask)
                    decayed = vmuls(
                        vmul(q_val, vexp(gamma, mask=col_mask), mask=col_mask),
                        self.scale_value,
                        mask=col_mask,
                    )
                    vstore_pack(
                        self.ub_cast,
                        offset,
                        vcast(decayed, dtypes.bfloat16, mask=col_mask),
                        col_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            vmem_bar("vst_vld")
        mem_copy(
            self._full_d(gm_q_decayed, task),
            self.ub_cast,
            copy_options(l2_cache_ctl=_L2_CACHE_LAST_USE),
        )

    @jit
    def issue_beta(self, gm_beta, batch_idx, head_idx, chunk_idx, valid_rows):
        beta = _valid_rows_view(
            tile_slice(
                gm_beta[batch_idx, head_idx, None, None],
                (CHUNK_SIZE, 1),
                (chunk_idx, 0),
            ),
            valid_rows,
            1,
        )
        mem_copy(self.ub_beta_raw, beta, copy_options(l2_cache_ctl=_L2_CACHE_DISABLE))
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)

    def _publish_per_d(self):
        nd2nz_tile(self.ub_per_d_nz, self.ub_per_d_nd)
        mem_copy(self.l1_per_d.next_write(), self.ub_per_d_nz)

    @jit
    def build_left_k(self, row_block: int):
        d = SUPPORTED_HEAD_DIM
        k_read = tile_slice(self.ub_k, (NEUMANN_BLOCK_SIZE, d), (row_block, 0))
        ref_row = row_block * NEUMANN_BLOCK_SIZE
        for row0 in range_constexpr(0, NEUMANN_BLOCK_SIZE, 4):
            with vf(mode="simd"):
                for col in range_constexpr(0, d, VL):
                    mask, _ = update_mask(d - col, elem_bits=32)
                    ref = vload(self.ub_cs, ref_row * d + col)
                    for row in range_constexpr(row0, row0 + 4):
                        offset = row * d + col
                        g_offset = (ref_row + row) * d + col
                        k = vcast(
                            vload_unpack(
                                k_read, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        decay = vexp_sub(vload(self.ub_cs, g_offset), ref, mask=mask)
                        vstore(self.ub_decay, offset, decay, mask)
                        packed = vcast(
                            vmul(k, decay, mask=mask), dtypes.bfloat16, mask=mask
                        )
                        vstore_pack(
                            self.ub_per_d_nd,
                            offset,
                            packed,
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")
        self._publish_per_d()

    @jit
    def build_left_q(self, row_block: int):
        d = SUPPORTED_HEAD_DIM
        q_read = tile_slice(self.ub_q, (NEUMANN_BLOCK_SIZE, d), (row_block, 0))
        for row0 in range_constexpr(0, NEUMANN_BLOCK_SIZE, 4):
            with vf(mode="simd"):
                for col in range_constexpr(0, d, VL):
                    mask, _ = update_mask(d - col, elem_bits=32)
                    for row in range_constexpr(row0, row0 + 4):
                        offset = row * d + col
                        q = vcast(
                            vload_unpack(
                                q_read, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        decay = vload(self.ub_decay, offset)
                        scaled = vmuls(
                            vmul(q, decay, mask=mask), self.scale_value, mask=mask
                        )
                        vstore_pack(
                            self.ub_per_d_nd,
                            offset,
                            vcast(scaled, dtypes.bfloat16, mask=mask),
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")
        self._publish_per_d()

    @jit
    def build_right_k(self, row_block: int, col_block: int):
        d = SUPPORTED_HEAD_DIM
        k_read = tile_slice(self.ub_k, (NEUMANN_BLOCK_SIZE, d), (col_block, 0))
        ref_row = row_block * NEUMANN_BLOCK_SIZE
        src_row = col_block * NEUMANN_BLOCK_SIZE
        for row0 in range_constexpr(0, NEUMANN_BLOCK_SIZE, 4):
            with vf(mode="simd"):
                for col in range_constexpr(0, d, VL):
                    mask, _ = update_mask(d - col, elem_bits=32)
                    ref = vload(self.ub_cs, ref_row * d + col)
                    for row in range_constexpr(row0, row0 + 4):
                        offset = row * d + col
                        g_offset = (src_row + row) * d + col
                        k = vcast(
                            vload_unpack(
                                k_read, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        decay = vexp_sub(ref, vload(self.ub_cs, g_offset), mask=mask)
                        packed = vcast(
                            vmul(k, decay, mask=mask), dtypes.bfloat16, mask=mask
                        )
                        vstore_pack(
                            self.ub_per_d_nd,
                            offset,
                            packed,
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")
        self._publish_per_d()

    @jit
    def apply_beta_to_fp16(self):
        for row0 in range_constexpr(0, CHUNK_SIZE, 4):
            with vf(mode="simd"):
                row_mask, _ = update_mask(VL, elem_bits=32)
                zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
                for row in range_constexpr(row0, row0 + 4):
                    row_off = row * CHUNK_SIZE
                    t_true = vload(self.ub_kkt, row_off)
                    t_beta = vmul(
                        t_true, vload_broadcast(self.ub_beta, row), mask=row_mask
                    )
                    tril_mask, _ = update_mask(row, elem_bits=32)
                    neg_t = vmuls(
                        vselect_raw(t_beta, zero_reg, cond_mask=tril_mask),
                        -1.0,
                        mask=row_mask,
                    )
                    vstore_pack(
                        self.ub_kkt_fp16,
                        row_off,
                        vcast(neg_t, dtypes.float16, mask=row_mask),
                        row_mask,
                        pack_mode="b32_to_b16",
                    )

    @jit
    def apply_beta_and_publish_neumann(self, handoff_l1):
        self.apply_beta_to_fp16()
        nd2nz_tile(self.ub_kkt_nz, self.ub_kkt_fp16)
        mem_copy(handoff_l1, self.ub_kkt_nz)

    @jit
    def store_masked_mqk(self, destination, valid_rows):
        for row0 in range_constexpr(0, CHUNK_SIZE, 4):
            with vf(mode="simd"):
                row_mask, _ = update_mask(VL, elem_bits=32)
                zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
                for row in range_constexpr(row0, row0 + 4):
                    off = row * CHUNK_SIZE
                    keep, _ = update_mask(row + 1, elem_bits=32)
                    mqk = vselect_raw(vload(self.ub_mqk, off), zero_reg, cond_mask=keep)
                    vstore_pack(
                        self.ub_mqk_bf16,
                        off,
                        vcast(mqk, dtypes.bfloat16, mask=row_mask),
                        row_mask,
                        pack_mode="b32_to_b16",
                    )
        mem_copy(
            destination[:valid_rows, None],
            self.ub_mqk_bf16[:valid_rows, None],
            copy_options(l2_cache_ctl=_L2_CACHE_LAST_USE),
        )

    @jit
    def store_restored_k(self, destination, valid_rows):
        d = SUPPORTED_HEAD_DIM
        last = (CHUNK_SIZE - 1) * d
        for row0 in range_constexpr(0, CHUNK_SIZE, 4):
            with vf(mode="simd"):
                for col in range_constexpr(0, d, VL):
                    col_mask, _ = update_mask(d - col, elem_bits=32)
                    cs_last = vload(self.ub_cs, last + col)
                    for row in range_constexpr(row0, row0 + 4):
                        offset = row * d + col
                        k = vcast(
                            vload_unpack(
                                self.ub_k, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=col_mask,
                        )
                        restored = vmul(
                            k,
                            vexp_sub(cs_last, vload(self.ub_cs, offset), mask=col_mask),
                            mask=col_mask,
                        )
                        vstore_pack(
                            self.ub_cast,
                            offset,
                            vcast(restored, dtypes.bfloat16, mask=col_mask),
                            col_mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")
        mem_copy(
            destination[:valid_rows, None],
            self.ub_cast[:valid_rows, None],
            copy_options(l2_cache_ctl=_L2_CACHE_LAST_USE),
        )

    @jit
    def store_gamma_row(self, gm_gamma, task):
        d = SUPPORTED_HEAD_DIM
        last = (CHUNK_SIZE - 1) * d
        with vf(mode="simd"):
            for col in range_constexpr(0, d, VL):
                col_mask, _ = update_mask(d - col, elem_bits=32)
                vstore(
                    self.ub_gamma,
                    col,
                    vexp(vload(self.ub_cs, last + col), mask=col_mask),
                    col_mask,
                )
            vmem_bar("vst_vld")
        mem_copy(
            tile_slice(gm_gamma, (d, 1), (task, 0)),
            self.ub_gamma.permute(1, 0),
            copy_options(l2_cache_ctl=_L2_CACHE_LAST_USE),
        )

    @jit
    def publish_c3(self, gm_V, batch_idx, head_idx, chunk_idx, valid_rows):
        d = SUPPORTED_HEAD_DIM
        v_tile = _valid_rows_view(
            tile_slice(
                gm_V[batch_idx, head_idx, None, None], (CHUNK_SIZE, d), (chunk_idx, 0)
            ),
            valid_rows,
            d,
        )
        mem_copy(self.ub_v, v_tile, copy_options(l2_cache_ctl=_L2_CACHE_DISABLE))
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16(self.ub_v, valid_rows)
        # Rows are independent; retain one completion barrier per produced tile.
        with vf(mode="simd"):
            for col in range_constexpr(0, d, VL):
                col_mask, _ = update_mask(d - col, elem_bits=32)
                for row in dsl_range(0, CHUNK_SIZE):
                    offset = row * d + col
                    beta = vload_broadcast(self.ub_beta, row)
                    cs = vload(self.ub_cs, offset)
                    k = vcast(
                        vload_unpack(
                            self.ub_k, offset, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=col_mask,
                    )
                    k_out = vmul(
                        vmul(k, vexp(cs, mask=col_mask), mask=col_mask),
                        beta,
                        mask=col_mask,
                    )
                    vstore_pack(
                        self.ub_cast,
                        offset,
                        vcast(k_out, dtypes.bfloat16, mask=col_mask),
                        col_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            vmem_bar("vst_vld")
        nd2nz_tile(self.ub_nz, self.ub_cast)
        mem_copy(self.l1_k_beta, self.ub_nz)
        # Rows are independent; retain one completion barrier per produced tile.
        with vf(mode="simd"):
            for col in range_constexpr(0, d, VL):
                col_mask, _ = update_mask(d - col, elem_bits=32)
                for row in dsl_range(0, CHUNK_SIZE):
                    offset = row * d + col
                    beta = vload_broadcast(self.ub_beta, row)
                    v_val = vcast(
                        vload_unpack(
                            self.ub_v, offset, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=col_mask,
                    )
                    vstore_pack(
                        self.ub_cast,
                        offset,
                        vcast(
                            vmul(v_val, beta, mask=col_mask),
                            dtypes.bfloat16,
                            mask=col_mask,
                        ),
                        col_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            vmem_bar("vst_vld")
        nd2nz_tile(self.ub_nz, self.ub_cast)
        mem_copy(self.l1_v_beta, self.ub_nz)

    @jit
    def prepare_safe_diagonal_left(self):
        """每个对角块仅初始化一次两个持久行掩码。"""
        with vf(mode="simd"):
            for col in range_constexpr(0, SUPPORTED_HEAD_DIM, VL):
                mask, _ = update_mask(VL, elem_bits=32)
                zero = vcast(
                    vdups(0.0, dtypes.float32, mask=mask), dtypes.bfloat16, mask=mask
                )
                for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1):
                    offset = row * SUPPORTED_HEAD_DIM
                    vstore_pack(
                        self.ub_safe_left_k,
                        offset + col,
                        zero,
                        mask,
                        pack_mode="b32_to_b16",
                    )
                    vstore_pack(
                        self.ub_safe_left_q,
                        offset + col,
                        zero,
                        mask,
                        pack_mode="b32_to_b16",
                    )
                    vstore_pack(
                        self.ub_safe_right_k,
                        offset + col,
                        zero,
                        mask,
                        pack_mode="b32_to_b16",
                    )

    @jit
    def update_safe_diagonal_left(
        self,
        normalized_k_dhalf,
        normalized_q_dhalf,
        row_block: int,
        active_row,
        left_l1_operand,
    ):
        k_source = tile_slice(
            normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (row_block, 0)
        )
        q_source = tile_slice(
            normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (row_block, 0)
        )
        if active_row > 0:
            self._update_safe_diagonal_left_values(
                k_source, q_source, active_row, left_l1_operand, True
            )
        else:
            self._update_safe_diagonal_left_values(
                k_source, q_source, active_row, left_l1_operand, False
            )

    @jit
    def build_safe_diagonal_right(
        self,
        normalized_k_dhalf,
        gate_cumsum_dhalf,
        row_block: int,
        active_row,
        right_k_l1_operand,
    ):
        source = tile_slice(
            normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), (row_block, 0)
        )
        nd_write = self.ub_safe_right_k
        with vf(mode="simd"):
            for col in range_constexpr(0, SUPPORTED_HEAD_DIM, VL):
                mask, _ = update_mask(VL, elem_bits=32)
                ref = vload(
                    gate_cumsum_dhalf,
                    (row_block * NEUMANN_BLOCK_SIZE + active_row) * SUPPORTED_HEAD_DIM
                    + col,
                )
                for row in dsl_range(0, active_row + 1, 1):
                    offset = row * SUPPORTED_HEAD_DIM
                    unpacked_k = vload_unpack(
                        source, offset + col, unpack_mode=UnpackMode.B16_TO_B32
                    )
                    k = vcast(unpacked_k, dtypes.float32, mask=mask)
                    gate = vload(
                        gate_cumsum_dhalf,
                        (row_block * NEUMANN_BLOCK_SIZE + row) * SUPPORTED_HEAD_DIM
                        + col,
                    )
                    decay = vexp_sub(ref, gate, mask=mask)
                    decayed_k = vmul(k, decay, mask=mask)
                    decayed_bf16 = vcast(decayed_k, dtypes.bfloat16, mask=mask)
                    vstore_pack(
                        nd_write,
                        offset + col,
                        decayed_bf16,
                        mask,
                        pack_mode="b32_to_b16",
                    )
        self._write_per_d_full_d_handoff(right_k_l1_operand, nd_write)

    @jit
    def _update_safe_diagonal_left_values(
        self, k_source, q_source, active_row, left_l1_operand, clear_previous: bool
    ):
        with vf(mode="simd"):
            for col in range_constexpr(0, SUPPORTED_HEAD_DIM, VL):
                mask, _ = update_mask(VL, elem_bits=32)
                if const_expr(clear_previous):
                    zero = vcast(
                        vdups(0.0, dtypes.float32, mask=mask),
                        dtypes.bfloat16,
                        mask=mask,
                    )
                    old_offset = (active_row - 1) * SUPPORTED_HEAD_DIM
                    vstore_pack(
                        self.ub_safe_left_k,
                        old_offset + col,
                        zero,
                        mask,
                        pack_mode="b32_to_b16",
                    )
                    vstore_pack(
                        self.ub_safe_left_q,
                        old_offset + col,
                        zero,
                        mask,
                        pack_mode="b32_to_b16",
                    )
                offset = active_row * SUPPORTED_HEAD_DIM
                unpacked_k = vload_unpack(
                    k_source, offset + col, unpack_mode=UnpackMode.B16_TO_B32
                )
                unpacked_q = vload_unpack(
                    q_source, offset + col, unpack_mode=UnpackMode.B16_TO_B32
                )
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                q = vcast(unpacked_q, dtypes.float32, mask=mask)
                scaled_q = vmuls(q, self.scale_value, mask=mask)
                vstore_pack(
                    self.ub_safe_left_k,
                    offset + col,
                    vcast(k, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode="b32_to_b16",
                )
                vstore_pack(
                    self.ub_safe_left_q,
                    offset + col,
                    vcast(scaled_q, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode="b32_to_b16",
                )
        self._write_per_d_full_d_handoff(left_l1_operand, self.ub_safe_left_k)
        self._write_per_d_full_d_handoff(left_l1_operand, self.ub_safe_left_q)

    def _write_per_d_full_d_handoff(self, full_ring_buffer, nd_slot):
        nd2nz_tile(self.ub_per_d_nz, nd_slot)
        mem_copy(full_ring_buffer.next_write(), self.ub_per_d_nz)

    @jit
    def load_gate(
        self,
        gm_g,
        gm_A_log,
        gm_dt_bias,
        batch,
        head,
        chunk,
        rows,
        mode,
        bounded,
        lower_bound,
    ):
        source = tile_slice(
            gm_g[batch, head, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)
        )
        mem_copy(self.ub_g_raw[:rows, None], source[:rows, None])
        if mode != GATE_MODE_PREPROCESSED:
            mem_copy(self.ub_alpha, tile_slice(gm_A_log, (1,), (head,)))
            if mode == GATE_MODE_FUSED_WITH_DT_BIAS:
                mem_copy(
                    self.ub_dt_bias,
                    tile_slice(gm_dt_bias, (1, SUPPORTED_HEAD_DIM), (head, 0)),
                )
        vec_sync_notify(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)
        if mode == GATE_MODE_PREPROCESSED:
            self._gate_values(rows, lower_bound, False, False, False)
        elif mode == GATE_MODE_FUSED_WITH_DT_BIAS:
            if bounded != 0:
                self._gate_values(rows, lower_bound, True, True, True)
            else:
                self._gate_values(rows, lower_bound, True, True, False)
        elif bounded != 0:
            self._gate_values(rows, lower_bound, True, False, True)
        else:
            self._gate_values(rows, lower_bound, True, False, False)

    @jit
    def _gate_values(self, rows, lower_bound, fused: bool, bias: bool, bounded: bool):
        for col in range_constexpr(0, SUPPORTED_HEAD_DIM, VL):
            with vf(mode="simd"):
                mask, _ = update_mask(VL, elem_bits=32)
                one = vdups(1.0, dtypes.float32, mask=mask)
                prefix = vdups(0.0, dtypes.float32, mask=mask)
                if const_expr(fused):
                    alpha = vexp(vload_broadcast(self.ub_alpha, 0), mask=mask)
                    dt = vdups(0.0, dtypes.float32, mask=mask)
                    if const_expr(bias):
                        dt = vload(self.ub_dt_bias, col)
                for row in dsl_range(0, rows, 1, unroll=4):
                    offset = row * SUPPORTED_HEAD_DIM + col
                    gate = vcast(
                        vload_unpack(
                            self.ub_g_raw, offset, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    if const_expr(fused):
                        x = vadd(gate, dt, mask=mask)
                        if const_expr(bounded):
                            exponent = vexp(
                                vmul(vmuls(alpha, -1.0, mask=mask), x, mask=mask),
                                mask=mask,
                            )
                            gate = vdiv(
                                vdups(lower_bound, dtypes.float32, mask=mask),
                                vadd(one, exponent, mask=mask),
                                mask=mask,
                            )
                        else:
                            tail = vlog(
                                vadd(
                                    one,
                                    vexp(
                                        vmuls(vabs(x, mask=mask), -1.0, mask=mask),
                                        mask=mask,
                                    ),
                                    mask=mask,
                                ),
                                mask=mask,
                            )
                            gate = vmul(
                                vmuls(alpha, -1.0, mask=mask),
                                vadd(vmaxs(x, 0.0, mask=mask), tail, mask=mask),
                                mask=mask,
                            )
                    prefix = vadd(prefix, gate, mask=mask)
                    vstore(self.ub_cs, offset, prefix, mask)
                for row in dsl_range(rows, CHUNK_SIZE, 1, unroll=4):
                    vstore(self.ub_cs, row * SUPPORTED_HEAD_DIM + col, prefix, mask)
                vmem_bar("vst_vld")

    @jit
    def activate_beta(self, rows, sigmoid, negative):
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        if sigmoid != 0:
            if negative != 0:
                self._beta_values(rows, True, True)
            else:
                self._beta_values(rows, True, False)
        else:
            self._beta_values(rows, False, False)

    @jit
    def _beta_values(self, rows, sigmoid: bool, negative: bool):
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            value = vload(self.ub_beta_raw, 0)
            if const_expr(sigmoid):
                one = vdups(1.0, dtypes.float32, mask=mask)
                value = vdiv(
                    one,
                    vadd(
                        one, vexp(vmuls(value, -1.0, mask=mask), mask=mask), mask=mask
                    ),
                    mask=mask,
                )
                if const_expr(negative):
                    value = vmuls(value, 2.0, mask=mask)
            valid, _ = update_mask(rows, elem_bits=32)
            invalid = mask_xor(mask, valid, exec_mask=mask)
            vstore(self.ub_beta, 0, value, valid)
            vstore(self.ub_beta, 0, vdups(0.0, dtypes.float32, mask=mask), invalid)
            vmem_bar("vst_vld")

    def store_norm(self, q, k, qr, kr, head, chunk, rows):
        mem_copy(
            tile_slice(q[0, head, None, None], (64, 128), (chunk, 0))[:rows, None],
            self.ub_q[:rows, None],
        )
        mem_copy(
            tile_slice(k[0, head, None, None], (64, 128), (chunk, 0))[:rows, None],
            self.ub_k[:rows, None],
        )
        mem_copy(
            tile_slice(qr[0, head, None, None], (64, 1), (chunk, 0))[:rows, None],
            from_tensor(self.ub_q_partials, layout=make_layout((64, 1), stride=(1, 1)))[
                :rows, None
            ],
        )
        mem_copy(
            tile_slice(kr[0, head, None, None], (64, 1), (chunk, 0))[:rows, None],
            from_tensor(self.ub_k_partials, layout=make_layout((64, 1), stride=(1, 1)))[
                :rows, None
            ],
        )

    @jit
    def store_qg(self, destination, rows):
        for row0 in range_constexpr(0, CHUNK_SIZE, 4):
            with vf(mode="simd"):
                mask, _ = update_mask(VL, elem_bits=32)
                for col in range_constexpr(0, SUPPORTED_HEAD_DIM, VL):
                    for row in range_constexpr(row0, row0 + 4):
                        offset = row * SUPPORTED_HEAD_DIM + col
                        q = vcast(
                            vload_unpack(
                                self.ub_q, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        value = vmul(
                            q, vexp(vload(self.ub_cs, offset), mask=mask), mask=mask
                        )
                        vstore_pack(
                            self.ub_cast,
                            offset,
                            vcast(value, dtypes.bfloat16, mask=mask),
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")
        mem_copy(destination[:rows, None], self.ub_cast[:rows, None])


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
    """A6 full-tile preparation; prefetch next Q/K during the current Cube inverse."""
    block_idx = get_block_idx()
    if block_idx < active_core_num:
        cube = StageOneMatmul()
        vector = StageOneVector(scale_value=scale_value)
        identity = cube.prepare_packed64_identity_l1(gm_identity16)
        cube.zero_half_l1(cube.zero_l1)
        value_heads = gm_V.shape[1]
        heads_per_key = value_heads // gm_K.shape[1]
        for task in range(task_start, task_end):
            local_head = task // group_count
            local_chunk = task - local_head * group_count
            chunk = group_start + local_chunk
            global_head = head_base + local_head
            batch, head = idx2crd(global_head, [gm_K.shape[0], value_heads])
            qk_head = head // heads_per_key
            rows = CHUNK_SIZE
            if tail_group and local_chunk + 1 == group_count:
                rows = tail_rows
            slot = workspace_task_base + global_head * eff_group + local_chunk
            if task == task_start:
                vector.issue_qk(gm_Q, gm_K, batch, qk_head, chunk, rows)
            vector.finish_qk(rows, use_qk_l2norm_in_kernel)
            if save_matrices != 0 and use_qk_l2norm_in_kernel != 0:
                if head % heads_per_key == 0:
                    vector.store_norm(
                        gm_q_norm, gm_k_norm, gm_q_rstd, gm_k_rstd, qk_head, chunk, rows
                    )
            vector.load_gate(
                gm_g,
                gm_A_log,
                gm_dt_bias,
                batch,
                head,
                chunk,
                rows,
                use_gate_in_kernel,
                has_lower_bound,
                lower_bound,
            )
            if save_g_cumsum != 0:
                gate_tile = tile_slice(
                    gm_g_cumsum[0, head, None, None], (64, 128), (chunk, 0)
                )
                mem_copy(gate_tile[:rows, None], vector.ub_cs[:rows, None])
            vector.store_q_decayed(gm_q_decayed, slot)
            if save_qg != 0:
                vector.store_qg(
                    tile_slice(gm_qg[0, head, None, None], (64, 128), (chunk, 0)), rows
                )
            vector.issue_beta(gm_beta_brc, batch, head, chunk, rows)
            vector.activate_beta(rows, use_beta_sigmoid_in_kernel, allow_neg_eigval)
            if save_matrices != 0 and use_beta_sigmoid_in_kernel != 0:
                beta_tile = tile_slice(
                    gm_beta_activated[0, head, None, None], (64, 1), (chunk, 0)
                )
                mem_copy(beta_tile[:rows, None], vector.ub_beta[:rows, None])
            for row_block in range_constexpr(4):
                vector.build_left_k(row_block)
                cube.load_per_d_left_k(vector.l1_per_d)
                vector.build_left_q(row_block)
                cube.load_per_d_left_q(vector.l1_per_d)
                for col_block in range_constexpr(row_block):
                    vector.build_right_k(row_block, col_block)
                    right = cube.load_per_d_right_k(vector.l1_per_d)
                    cube.mm_per_d_kkt(row_block, col_block, right)
                    cube.mm_per_d_mqk(row_block, col_block, right)
                if use_gate_in_kernel != 0 and has_lower_bound != 0:
                    vector.build_right_k(row_block, row_block)
                    right = cube.load_per_d_right_k(vector.l1_per_d)
                    cube.mm_per_d_kkt(row_block, row_block, right)
                    cube.mm_per_d_mqk(row_block, row_block, right)
                else:
                    cube.mm_safe_diagonal_tile(
                        vector, vector.ub_k, vector.ub_q, vector.ub_cs, row_block
                    )
            cube.finish_per_d_kkt(vector.ub_kkt)
            cube.finish_per_d_mqk(vector.ub_mqk)
            vector.apply_beta_and_publish_neumann(cube.neumann_power_handoff_l1)
            vector.publish_c3(gm_V, batch, head, chunk, rows)
            vector.store_masked_mqk(
                tile_slice(
                    gm_Aqk[0, head, None, None],
                    (CHUNK_SIZE, CHUNK_SIZE),
                    (chunk, 0),
                ),
                rows,
            )
            vector.store_restored_k(
                tile_slice(gm_kg[0, head, None, None], (64, 128), (chunk, 0)), rows
            )
            vector.store_gamma_row(gm_gamma, slot)
            # All Q/K readers, including normalization cache DMA, finish before
            # reusing the same UB buffers. Cube now consumes only the published
            # L1 operands. No extra buffers/counters or weaker rewind boundary.
            vec_sync_all()
            if task + 1 < task_end:
                next_local_head = (task + 1) // group_count
                next_local_chunk = task + 1 - next_local_head * group_count
                next_batch, next_head = idx2crd(
                    head_base + next_local_head, [gm_K.shape[0], value_heads]
                )
                next_rows = CHUNK_SIZE
                if tail_group and next_local_chunk + 1 == group_count:
                    next_rows = tail_rows
                vector.issue_qk(
                    gm_Q,
                    gm_K,
                    next_batch,
                    next_head // heads_per_key,
                    group_start + next_local_chunk,
                    next_rows,
                )
            lower, identity_matrix, scratch = (
                cube.load_packed64_diagonal_inputs_from_l1_handoff(
                    cube.neumann_power_handoff_l1, identity
                )
            )
            square, inverse = cube.neumann_diag_power2_update(
                lower, identity_matrix, scratch
            )
            fourth, inverse = cube.neumann_diag_power4_update(square, inverse)
            inverse = cube.neumann_diag_power8_update(fourth, inverse)
            inverse = cube.compose_odd_even_lower16_to32_accum_full64(scratch, inverse)
            cube.compose_odd_even_lower32_to64_accum_full64(scratch, inverse)
            cube.store_inverse(gm_Akk, head, chunk, rows, save_matrices)
            cube.compute_u_pre_and_w_from_resident_inv_beta(
                gm_u, gm_w, head, chunk, rows, vector.l1_v_beta, vector.l1_k_beta
            )
            cube_sync_all()
            vec_sync_all()


def _matmul_transposed_a(dst, lhs, rhs, *, init=True):
    """MMAD where L0A is ZN and its second logical axis is M.

    LHS and RHS are ZN views of (K, M) and (K, N). Both output axes
    are axis 1, giving an (M, N) state tile without moving the operands.
    """
    import cannbotdsl._mlir.ir as ir
    from cannbotdsl._mlir.dialects import cannbotir
    from cannbotdsl.ops.cube import _managed_matmul_attributes, _record_matmul
    from cannbotdsl.tensor import Tensor, _raw, _require_tensor

    axis_1 = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)
    kwargs = {
        "init": bool(init),
        "unit_flag": 0,
        "disable_gemv": True,
        "mx_mode": False,
        "lhs_m_axis": axis_1,
        "rhs_n_axis": axis_1,
    }
    kwargs.update(_managed_matmul_attributes())
    destination = _require_tensor(dst)
    operation = cannbotir.MatMulOp(
        _raw(destination),
        _raw(_require_tensor(lhs)),
        _raw(_require_tensor(rhs)),
        **kwargs,
    )
    result = Tensor._from_ir(operation.res)
    if hasattr(operation, "operation"):
        operation = operation.operation
    _record_matmul(operation)
    return result


class StageTwoMatmul:
    def __init__(self, *, dv_base: int, split_fixpipe: bool = False):
        self.dv_base = dv_base
        self.l0a_chunk = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.l0a_kr_t = (
            RingBuffer(
                MemLoc.L0A,
                dtypes.bfloat16,
                make_layout((dv_base, CHUNK_SIZE)),
                depth=1,
                data_format="nz",
            )
            .next_write()
            .permute(1, 0)
        )
        self.l0a_mqk = RingBuffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        ).next_write()
        self.l0b_u = RingBuffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((CHUNK_SIZE, dv_base)), depth=1
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
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        ).next_write()
        self.l0c_u = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, dv_base)), depth=1
        ).next_write()
        self.l0c_o = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, dv_base)), depth=1
        ).next_write()
        self.l0c_state = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((dv_base, SUPPORTED_HEAD_DIM)),
            depth=1,
        ).next_write()
        self.split_fixpipe = split_fixpipe
        self.mqk_held = None
        if not split_fixpipe:
            self.mqk_held = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                depth=1,
            ).next_write()

    def load_q_decayed(self, q_decayed_l1, gm_tile):
        mem_copy(
            q_decayed_l1.next_write(),
            gm_tile,
            copy_options(l2_cache_ctl=_L2_CACHE_LAST_USE),
        )

    def load_state(self, state_l1):
        slot = state_l1.next_read()
        self._state_l1_slot = slot
        mem_copy(self.l0b_state, slot)

    def reload_state_l0b(self):
        """Copy the same state tile into L0B again.

        W@state and Q@state share one L0B buffer. The first matmul is enough
        for the state recurrence because W is ~1e-4, but Q@state is the output.
        """
        mem_copy(self.l0b_state, self._state_l1_slot)

    def compute_w_state(self, w_l1):
        mem_copy(self.l0a_chunk.next_write(), w_l1.next_read())
        matmul(self.l0c_u, self.l0a_chunk.next_read(), self.l0b_state, init=True)

    def compute_q_state(self, q_decayed_l1):
        mem_copy(self.l0a_chunk.next_write(), q_decayed_l1.next_read())
        matmul(self.l0c_o, self.l0a_chunk.next_read(), self.l0b_state, init=True)

    def _kr_zn(self):
        return from_tensor(
            self.l0b_kr,
            data_format="zn",
            layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
        )

    def _u_zn(self):
        return from_tensor(
            self.l0b_u, data_format="zn", layout=make_layout((CHUNK_SIZE, self.dv_base))
        )

    def load_k_restored_operand(self, k_restored_l1, valid_rows):
        slot = k_restored_l1.next_read()
        self._k_l1_slot = slot
        mem_copy(self._kr_zn(), slot)

    def load_mqk_operand(self, mqk_l1, valid_rows):
        slot = mqk_l1.next_read()
        self._mqk_l1_slot = slot
        mem_copy(self.l0a_mqk, slot)

    def stage_u_for_both(self, u_slot):
        """Copy U into L0A and L0B before either matmul.

        On the first chunk the copy after the state matmul sees an empty L1.
        The state copy is the one that still has the data.
        """
        u_tile = from_tensor(
            u_slot, offset=0, layout=make_layout((CHUNK_SIZE, self.dv_base))
        )
        mem_copy(self.l0a_kr_t, u_tile)
        mem_copy(self._u_zn(), u_tile)

    def update_state_staged(self):
        _matmul_transposed_a(self.l0c_state, self.l0a_kr_t, self._kr_zn(), init=True)

    def accumulate_mqk_u_staged(self, *, init=False):
        matmul(self.l0c_o, self.l0a_mqk, self._u_zn(), init=init)

    def store_u_delta(self, ub):
        src = tile_slice(self.l0c_u, (CHUNK_SIZE, self.dv_base), (0, 0))
        if self.split_fixpipe:
            src = src.partition(dim=0)
        mem_copy(ub, src)

    def store_state_delta(self, ub):
        src = self.l0c_state
        if self.split_fixpipe:
            src = src.partition(dim=1)
        mem_copy(ub, src)

    def load_k_cumdecay(self, k_cumdecay_l1, gm_tile, valid_rows):
        self._load_chunk_rows(k_cumdecay_l1, gm_tile, valid_rows)

    def load_k_restored(self, k_restored_l1, gm_tile, valid_rows):
        self._load_chunk_rows(
            k_restored_l1, gm_tile, valid_rows, l2_cache_ctl=_L2_CACHE_LAST_USE
        )

    def load_mqk(self, mqk_l1, gm_tile, valid_rows):
        self._load_chunk_rows(
            mqk_l1, gm_tile, valid_rows, l2_cache_ctl=_L2_CACHE_LAST_USE
        )

    def store_output(self, gm_tile):
        """用源视图的有效行列限定 FIXPIPE 尾块写回。"""
        mem_copy(
            gm_tile,
            self.l0c_o[0:CHUNK_SIZE, 0 : self.dv_base][
                : gm_tile.shape[0], : gm_tile.shape[1]
            ],
            options=copy_options(l2_cache_ctl=_L2_CACHE_DISABLE),
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
                    l1_col,
                    _valid_rows_view(gm_col, valid_rows, 16),
                    options=copy_options(l2_cache_ctl=l2_cache_ctl),
                )
        else:
            mem_copy(slot, gm_tile, options=copy_options(l2_cache_ctl=l2_cache_ctl))


class StageTwoVector:
    def __init__(self, *, u_row_block: int, dv_base: int, shared=None):
        self.dv_base = dv_base
        self.u_row_block = u_row_block
        self.half_dk = SUPPORTED_HEAD_DIM
        if shared is None:
            self.ub_state_fp32 = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((dv_base, SUPPORTED_HEAD_DIM))
            )
            self.tmp_u_bf16 = Buffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((u_row_block, dv_base))
            )
            self.state_nz = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((dv_base, SUPPORTED_HEAD_DIM)),
                depth=1,
                data_format="nz",
                physical_layout=_padded_nz_layout((dv_base, SUPPORTED_HEAD_DIM)),
            ).next_write()
            self.u_nz = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((u_row_block, dv_base)),
                depth=1,
                data_format="nz",
                physical_layout=_padded_nz_layout((u_row_block, dv_base)),
            ).next_write()
            self.u_in = RingBuffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((u_row_block, dv_base)), depth=1
            )
        else:
            self.ub_state_fp32 = shared.ub_state_fp32
            self.tmp_u_bf16 = shared.tmp_u_bf16
            self.state_nz = shared.state_nz
            self.u_nz = shared.u_nz
            self.u_in = shared.u_in

    @staticmethod
    def load_gamma_col(gamma_ub, gamma_col):
        slot = gamma_ub.next_write()
        mem_copy(slot, gamma_col)

    def store_h(self, gm_h_rows):
        mem_copy(
            gm_h_rows,
            self.ub_state_fp32,
            options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
        )

    def store_group_state_to_h(self, gm_h_rows):
        mem_copy(
            gm_h_rows,
            self.ub_state_fp32,
            options=copy_options(l2_cache_ctl=_STAGE2_INPUT_L2_CACHE_CTL),
        )

    def write_state_l1_snapshot(self, state_l1, subblock_idx):
        """发布累加后的状态，供下一步 Cube 计算使用。"""
        self._produce_state_l1(state_l1, subblock_idx)

    @jit
    def issue_u_rows(self, gm_u_tile, subblock_idx, valid_rows):
        row_start = subblock_idx * self.u_row_block
        u_in_write = self.u_in.next_write()
        if row_start < valid_rows:
            rows = valid_rows - row_start
            if rows > self.u_row_block:
                rows = self.u_row_block
            gm_u_rows = tile_slice(
                gm_u_tile, (self.u_row_block, self.dv_base), (subblock_idx, 0)
            )
            mem_copy(
                _valid_rows_view(u_in_write, rows, self.dv_base),
                _valid_rows_view(gm_u_rows, rows, self.dv_base),
            )

    @jit
    def finish_u_delta_to_l1(
        self, u_l1, tmp_u_delta, subblock_idx, gm_v_new_chunk, valid_rows, save_v_new
    ):
        u_l1_rows = tile_slice(
            u_l1.next_write(), (self.u_row_block, self.dv_base), (subblock_idx, 0)
        )
        delta = tmp_u_delta
        row_start = subblock_idx * self.u_row_block
        rows = valid_rows - row_start
        if rows > self.u_row_block:
            rows = self.u_row_block
        if rows < 0:
            rows = 0
        u_in_read = self.u_in.next_read()
        u_bf16_write = self.tmp_u_bf16
        # Valid rows are overwritten below. Only the tail needs initialization
        # before the full tile is converted to NZ and consumed by both matmuls.
        if rows < self.u_row_block:
            with vf(mode="simd"):
                bf16_mask, _ = update_mask(self.dv_base, elem_bits=16)
                zero = vdups(0.0, dtypes.bfloat16, mask=bf16_mask)
                for row in dsl_range(rows, self.u_row_block):
                    vstore(u_bf16_write, row * self.dv_base, zero, bf16_mask)
        with vf(mode="simd"):
            for row in dsl_range(0, rows, 1, unroll=2):
                for col in range(0, self.dv_base, VL):
                    mask, _ = update_mask(self.dv_base - col, elem_bits=32)
                    base = row * self.dv_base + col
                    u_raw = vload_unpack(
                        u_in_read, base, unpack_mode=UnpackMode.B16_TO_B32
                    )
                    u_fp32 = vcast(u_raw, dtypes.float32, mask=mask)
                    delta_value = vload(delta, base)
                    u_next = vsub(u_fp32, delta_value, mask=mask)
                    u_bf16 = vcast(u_next, dtypes.bfloat16, mask=mask)
                    vstore_pack(
                        u_bf16_write, base, u_bf16, mask, pack_mode=PackMode.B32_TO_B16
                    )
        u_nz_slot = self.u_nz[0 : self.u_row_block, 0 : self.dv_base]
        nd2nz_tile(u_nz_slot, self.tmp_u_bf16)
        u_nz_read = self.u_nz[0 : self.u_row_block, 0 : self.dv_base]
        mem_copy(u_l1_rows, u_nz_read)
        if rows > 0 and save_v_new != 0:
            v_new_rows = tile_slice(
                gm_v_new_chunk, (self.u_row_block, self.dv_base), (subblock_idx, 0)
            )
            mem_copy(
                _valid_rows_view(v_new_rows, rows, self.dv_base),
                self.tmp_u_bf16[:rows, : self.dv_base],
                options=copy_options(l2_cache_ctl=_SAVED_OUTPUT_L2_CACHE_CTL),
            )

    def load_initial_state_and_write_l1(self, current_state, state_l1, subblock_idx):
        for half in range_constexpr(2):
            mem_copy(
                tile_slice(self.ub_state_fp32, (self.dv_base, D_HALF), (0, half)),
                tile_slice(current_state, (self.dv_base, D_HALF), (0, half)),
                options=copy_options(l2_cache_ctl=1),
            )
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        self._produce_state_l1(state_l1, subblock_idx)

    def _produce_state_l1(self, state_l1, subblock_idx):
        self._cast_full_state_to_nz()
        mem_copy(state_l1.next_write(), self.state_nz)

    def store_state_to_gm(self, gm_state_rows):
        vec_sync_notify(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        mem_copy(gm_state_rows, self.ub_state_fp32)

    @jit
    def _cast_full_state_to_nz(self):
        """Cast the full fp32 state into NZ. A5 keeps the ND then NZ copies."""
        d = self.half_dk
        s_n1, s_m1, s_m0, s_n0 = self.state_nz.physical_layout.stride
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        block_stride = s_n1 // n0
        # One VF owns all rows; the NZ map and per-element cast stay unchanged.
        with vf(mode="simd"):
            full, _ = rr.update_mask(VL, elem_bits=32)
            b16_full, _ = rr.update_mask(d, elem_bits=16)
            for row in dsl_range(0, self.dv_base):
                ve, vo = rr.vload_deinterleave(self.ub_state_fp32, row * d, width="b32")
                he = rr.vcast(
                    ve, dtypes.bfloat16, mask=full, reg_layout=rr.RegLayout.ZERO
                )
                ho = rr.vcast(
                    vo, dtypes.bfloat16, mask=full, reg_layout=rr.RegLayout.ONE
                )
                rr.vstore_strided(
                    self.state_nz,
                    row // m0 * s_m1 + row % m0 * s_m0,
                    rr.vbitwise_or(he, ho, mask=b16_full),
                    b16_full,
                    block_stride=block_stride,
                    repeat_stride=0,
                )

    @jit
    def scale_and_add_delta(self, tmp_delta, gamma_ub):
        """Fuse state decay and delta accumulation into one full-state VF pass."""
        gamma_read = gamma_ub.next_read()
        ub_state = self.ub_state_fp32
        delta = tmp_delta
        cols = self.half_dk
        gamma_row = from_tensor(
            gamma_read, offset=0, layout=make_layout((1, cols), stride=(cols, 1))
        )
        # Hoist gamma loads across rows without changing FP32 mul/add ordering.
        with vf(mode="simd"):
            mask, _ = update_mask(VL, elem_bits=32)
            gamma_lo = vload(gamma_row, 0)
            gamma_hi = vload(gamma_row, VL)
            for row in dsl_range(0, self.dv_base):
                base = row * cols
                state_lo = vload(ub_state, base)
                delta_lo = vload(delta, base)
                vstore(
                    ub_state,
                    base,
                    vadd(vmul(state_lo, gamma_lo, mask=mask), delta_lo, mask=mask),
                    mask,
                )
                state_hi = vload(ub_state, base + VL)
                delta_hi = vload(delta, base + VL)
                vstore(
                    ub_state,
                    base + VL,
                    vadd(vmul(state_hi, gamma_hi, mask=mask), delta_hi, mask=mask),
                    mask,
                )
            vmem_bar("vst_vld")


class _StageTwoResources:
    """每轮仅分配一次，由互斥的 DV 变体共享。"""

    def __init__(self, dv_base: int):
        self.q_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.w_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.k_restored_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.mqk_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=2
        )
        self.state_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((dv_base, SUPPORTED_HEAD_DIM)),
            depth=2,
        )
        self.u_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, dv_base)), depth=2
        )
        self.matmul = StageTwoMatmul(dv_base=dv_base)
        self.gamma_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((SUPPORTED_HEAD_DIM, 1)), depth=2
        )
        self.tmp_delta = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((dv_base, SUPPORTED_HEAD_DIM)),
            depth=1,
        ).next_write()
        self.tmp_u_delta = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, dv_base)), depth=1
        ).next_write()
        self.vector = StageTwoVector(u_row_block=CHUNK_SIZE, dv_base=dv_base)


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
    save_final_state,
    save_intermediate_state,
    save_v_new,
):
    block_idx = get_block_idx()
    subblock_idx = 0
    num_dv_base = SUPPORTED_HEAD_DIM // dv_base
    half_dk = SUPPORTED_HEAD_DIM
    u_row_block = CHUNK_SIZE
    q_l1 = resources.q_l1
    w_l1 = resources.w_l1
    k_restored_l1 = resources.k_restored_l1
    mqk_l1 = resources.mqk_l1
    state_l1 = resources.state_l1
    u_l1 = resources.u_l1
    s2_matmul = resources.matmul
    gamma_ub = resources.gamma_ub
    tmp_delta = resources.tmp_delta
    tmp_u_delta = resources.tmp_u_delta
    s2_vector = StageTwoVector(
        u_row_block=u_row_block, dv_base=dv_base, shared=resources.vector
    )
    if block_idx < logical_core_num:
        for item in range(item_start, item_end):
            head_id, dv_idx = idx2crd(item, [head_count, num_dv_base])
            seq_idx = head_base + head_id
            write_state_ref = gm_state_out[state_batch, seq_idx, None, None]
            first_linear = workspace_task_base + seq_idx * eff_group
            first_w_chunk = tile_slice(
                gm_w[0, seq_idx, None, None],
                (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                (group_start, 0),
            )
            first_q_chunk = tile_slice(
                gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0)
            )
            first_k_restored_chunk = tile_slice(
                gm_kg[0, seq_idx, None, None],
                (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                (group_start, 0),
            )
            first_mqk_tile = tile_slice(
                gm_Mqk[0, seq_idx, None, None],
                (CHUNK_SIZE, CHUNK_SIZE),
                (group_start, 0),
            )
            first_gamma_col_full = tile_slice(
                gm_gamma, (SUPPORTED_HEAD_DIM, 1), (first_linear, 0)
            )
            first_gamma_col = tile_slice(
                first_gamma_col_full, (half_dk, 1), (subblock_idx, 0)
            )
            first_u_tile_full = tile_slice(
                gm_u[0, seq_idx, None, None],
                (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                (group_start, 0),
            )
            first_u_tile = tile_slice(
                first_u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx)
            )
            first_valid_rows = CHUNK_SIZE
            if group_start + 1 == chunks_total:
                first_valid_rows = seq_len - group_start * CHUNK_SIZE
            s2_matmul.load_k_cumdecay(w_l1, first_w_chunk, first_valid_rows)
            s2_matmul.load_q_decayed(q_l1, first_q_chunk)
            s2_matmul.load_k_restored(
                k_restored_l1, first_k_restored_chunk, first_valid_rows
            )
            s2_matmul.load_mqk(mqk_l1, first_mqk_tile, first_valid_rows)
            if group_start == 0:
                in_state_ref = gm_state_in[state_batch, seq_idx, None, None]
                first_state_rows = tile_slice(
                    in_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx)
                )
                s2_vector.load_initial_state_and_write_l1(
                    first_state_rows, state_l1, subblock_idx
                )
            else:
                first_state_chunk = 0
                if save_intermediate_state != 0:
                    first_state_chunk = state_chunk_base + group_start
                first_h_state = gm_h[
                    state_batch_idx, seq_idx, first_state_chunk, None, None
                ]
                first_state_rows = tile_slice(
                    first_h_state, (dv_base, half_dk), (dv_idx, subblock_idx)
                )
                s2_vector.load_initial_state_and_write_l1(
                    first_state_rows, state_l1, subblock_idx
                )
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
                o_chunk = _o_head_chunk_tile(
                    gm_O, seq_idx, k, dv_base, dv_idx, valid_rows
                )
                v_new_chunk_full = tile_slice(
                    gm_v_new[0, seq_idx, None, None],
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (k, 0),
                )
                v_new_chunk = tile_slice(
                    v_new_chunk_full, (CHUNK_SIZE, dv_base), (0, dv_idx)
                )
                state_chunk = 0
                if save_intermediate_state != 0:
                    state_chunk = state_chunk_base + k
                h_state = gm_h[state_batch_idx, seq_idx, state_chunk, None, None]
                h_rows = tile_slice(h_state, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_matmul.load_state(state_l1)
                s2_matmul.compute_w_state(w_l1)
                s2_matmul.reload_state_l0b()
                s2_matmul.compute_q_state(q_l1)
                s2_matmul.load_k_restored_operand(k_restored_l1, valid_rows)
                s2_matmul.load_mqk_operand(mqk_l1, valid_rows)
                # Define the view outside both dynamic regions; only the guarded
                # issue_u_rows below reads it when a next chunk exists.
                next_u_tile_full = tile_slice(
                    gm_u[0, seq_idx, None, None],
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (k + 1, 0),
                )
                next_u_tile = tile_slice(
                    next_u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx)
                )
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_q_chunk = tile_slice(
                        gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0)
                    )
                    next_w_chunk = tile_slice(
                        gm_w[0, seq_idx, None, None],
                        (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                        (k + 1, 0),
                    )
                    next_gamma_col_full = tile_slice(
                        gm_gamma, (SUPPORTED_HEAD_DIM, 1), (next_linear, 0)
                    )
                    next_gamma_col = tile_slice(
                        next_gamma_col_full, (half_dk, 1), (subblock_idx, 0)
                    )
                    s2_matmul.load_q_decayed(q_l1, next_q_chunk)
                    s2_matmul.load_k_cumdecay(w_l1, next_w_chunk, next_valid_rows)
                    s2_vector.load_gamma_col(gamma_ub, next_gamma_col)
                s2_matmul.store_u_delta(tmp_u_delta)
                s2_vector.finish_u_delta_to_l1(
                    u_l1, tmp_u_delta, subblock_idx, v_new_chunk, valid_rows, save_v_new
                )
                if chunk_id + 1 < group_count:
                    s2_vector.issue_u_rows(next_u_tile, subblock_idx, next_valid_rows)
                u_slot = u_l1.next_read()
                s2_matmul.stage_u_for_both(u_slot)
                s2_matmul.update_state_staged()
                s2_matmul.store_state_delta(tmp_delta)
                mem_copy(s2_matmul.l0a_mqk, s2_matmul._mqk_l1_slot)
                s2_matmul.accumulate_mqk_u_staged()
                if save_intermediate_state != 0 and (group_start == 0 or chunk_id > 0):
                    s2_vector.store_h(h_rows)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_k_restored_chunk = tile_slice(
                        gm_kg[0, seq_idx, None, None],
                        (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                        (k + 1, 0),
                    )
                    next_mqk_tile = tile_slice(
                        gm_Mqk[0, seq_idx, None, None],
                        (CHUNK_SIZE, CHUNK_SIZE),
                        (k + 1, 0),
                    )
                    s2_matmul.load_k_restored(
                        k_restored_l1, next_k_restored_chunk, next_valid_rows
                    )
                    s2_matmul.load_mqk(mqk_l1, next_mqk_tile, next_valid_rows)
                s2_matmul.store_output(o_chunk)
                s2_vector.scale_and_add_delta(tmp_delta, gamma_ub)
                if chunk_id + 1 < group_count:
                    s2_vector.write_state_l1_snapshot(state_l1, subblock_idx)
                elif k + 1 < chunks_total:
                    next_state_chunk = 0
                    if save_intermediate_state != 0:
                        next_state_chunk = state_chunk_base + k + 1
                    next_h_state = gm_h[
                        state_batch_idx, seq_idx, next_state_chunk, None, None
                    ]
                    next_h_rows = tile_slice(
                        next_h_state, (dv_base, half_dk), (dv_idx, subblock_idx)
                    )
                    s2_vector.store_group_state_to_h(next_h_rows)
                elif save_final_state != 0:
                    state_rows = tile_slice(
                        write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx)
                    )
                    s2_vector.store_state_to_gm(state_rows)
            cube_sync_all()
            vec_sync_all()


# Pair and grid barriers share two reserved counters. Each is a round trip:
# the sender acquires a free credit, publishes, and the receiver returns it.
# Every boundary closes both cores' ownership; prime/drain only once around
# all rounds. This leaves 14 of the 16 counters for automatic allocation.
STAGE_READY_SYNC_ID = 0
STAGE_RELEASE_SYNC_ID = 4
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


def _prime_stage_barriers():
    """Seed one free credit for each direction of the staged handshake."""
    cube_sync_intra_arrive(PIPE.FIXPIPE, STAGE_READY_SYNC_ID)
    vec_sync_intra_arrive(PIPE.V, STAGE_RELEASE_SYNC_ID)


def _drain_stage_barriers():
    """Consume the two returned credits, including when there are no rounds."""
    vec_sync_intra_wait(PIPE.V, STAGE_READY_SYNC_ID)
    cube_sync_intra_wait(PIPE.FIXPIPE, STAGE_RELEASE_SYNC_ID)
    cube_sync_all()
    vec_sync_all()


def _pair_boundary_reset():
    """Drain one AIC/AIV pair at a staged batch boundary."""
    cube_sync_all()
    vec_sync_all()
    # A6 mode=2 lowers to these same intra instructions, but only explicit
    # intra IR reserves the IDs against automatic RingBuffer counters in 0.8.3.
    vec_sync_intra_wait(PIPE.V, STAGE_READY_SYNC_ID)
    vec_sync_intra_arrive(PIPE.V, STAGE_READY_SYNC_ID)
    cube_sync_intra_wait(PIPE.FIXPIPE, STAGE_READY_SYNC_ID)
    cube_sync_intra_arrive(PIPE.FIXPIPE, STAGE_READY_SYNC_ID)
    cube_sync_intra_wait(PIPE.FIXPIPE, STAGE_RELEASE_SYNC_ID)
    cube_sync_intra_arrive(PIPE.FIXPIPE, STAGE_RELEASE_SYNC_ID)
    vec_sync_intra_wait(PIPE.V, STAGE_RELEASE_SYNC_ID)
    vec_sync_intra_arrive(PIPE.V, STAGE_RELEASE_SYNC_ID)
    # DSL 0.8.2 recognizes local completion immediately before the rewind.
    cube_sync_all()
    vec_sync_all()
    _rewind_buffers()


def _global_stage_boundary(*, cross_block_flag_id: int):
    """Keep the A6 grid barrier's intra IDs visible to resource allocation.

    Preserve the MIX 1:1 global rendezvous with explicit intra handoffs.
    Each wait/arrive pair uses the same counter on the same core, as required
    by 0.8.3 ownership verification. Free credits are returned on both lanes.
    The grid rendezvous uses its separate mode=0 FFTS namespace on both cores.
    """
    cube_sync_all()
    vec_sync_all()
    vec_sync_intra_wait(PIPE.V, STAGE_READY_SYNC_ID)
    vec_sync_intra_arrive(PIPE.V, STAGE_READY_SYNC_ID)
    cube_sync_intra_wait(PIPE.FIXPIPE, STAGE_READY_SYNC_ID)
    cube_sync_block_arrive(PIPE.FIXPIPE, cross_block_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, cross_block_flag_id, mode=0)
    vec_sync_block_arrive(PIPE.MTE3, cross_block_flag_id, mode=0)
    vec_sync_block_wait(PIPE.S, cross_block_flag_id, mode=0)
    cube_sync_intra_arrive(PIPE.FIXPIPE, STAGE_READY_SYNC_ID)
    cube_sync_intra_wait(PIPE.FIXPIPE, STAGE_RELEASE_SYNC_ID)
    cube_sync_intra_arrive(PIPE.FIXPIPE, STAGE_RELEASE_SYNC_ID)
    vec_sync_intra_wait(PIPE.V, STAGE_RELEASE_SYNC_ID)
    vec_sync_intra_arrive(PIPE.V, STAGE_RELEASE_SYNC_ID)
    # SDK 0.8.3 only recognizes mode=2 block funnels as global resource
    # completion. Explicit intra handoffs need adjacent local drains for both
    # Buffer-ID namespaces before rewind; the leading drains are not enough.
    cube_sync_all()
    # The A6 wait runs on V; drain it before subsequent DMA can advance.
    vec_sync_all()


def _tnd_state_chunk_capacity(storage_length: int, batch: int) -> int:
    return (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch


def _tnd_state_chunk_base(token_start: int, logical_batch: int) -> int:
    return int(token_start) // CHUNK_SIZE + int(logical_batch)


def _public_bnsd_tensor(tensor: torch.Tensor, layout_qkv: str) -> torch.Tensor:
    if layout_qkv == "TND":
        return tensor[0].transpose(0, 1).contiguous()
    return tensor.transpose(1, 2).contiguous() if layout_qkv == "BSND" else tensor


def _public_bnsd_scalar(tensor: torch.Tensor, layout_qkv: str) -> torch.Tensor:
    return tensor[0] if layout_qkv == "TND" else tensor


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
        tensor = torch.nn.functional.pad(
            tensor, (0, 0, 0, padded_length - storage_length)
        )
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
    save_final_state,
    save_intermediate_state,
    save_v_new,
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
            storage_batch:storage_batch_end, None, token_start:token_stop, None
        ]
        v_new = gm_v_new[storage_batch, None, token_start:token_stop, None]
        workspace_slot_start = gm_metadata[stage1_task_prefix_base + active_batch_idx]
        _run_stage2_body(
            kg,
            gm_gamma,
            gm_q_decayed,
            saved_w,
            u,
            v_new,
            gm_h,
            o,
            gm_state_in,
            gm_state_out,
            mqk,
            workspace_task_base=workspace_slot_start,
            dv_base=dv_base,
            head_base=0,
            head_count=value_heads,
            chunks_total=chunks_total,
            seq_len=seq_len,
            group_start=group_start,
            group_count=chunk_num_per_group,
            eff_group=chunk_num_per_group,
            logical_core_num=logical_core_num,
            item_start=item_start,
            item_end=item_end,
            resources=resources,
            state_batch=logical_batch,
            save_final_state=save_final_state,
            save_intermediate_state=save_intermediate_state,
            save_v_new=save_v_new,
            state_batch_idx=0
            if state_is_compact != 0 and save_intermediate_state != 0
            else logical_batch,
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
    dv_splits_num: int,
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
    save_final_state,
    save_intermediate_state,
    save_v_new,
):
    dv_base = SUPPORTED_HEAD_DIM // dv_splits_num
    resources = _StageTwoResources(dv_base)
    _run_stage2_round(
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
        dv_base=dv_base,
        value_heads=value_heads,
        active_batch_num=active_batch_num,
        batch_idx_base=batch_idx_base,
        storage_batch_idx_base=storage_batch_idx_base,
        token_start_base=token_start_base,
        valid_seq_len_base=valid_seq_len_base,
        chunk_start_base=chunk_start_base,
        chunk_num_base=chunk_num_base,
        stage1_task_prefix_base=stage1_task_prefix_base,
        stage2_task_prefix_base=stage2_task_prefix_base,
        stage2_core_start=stage2_core_start,
        stage2_core_end=stage2_core_end,
        logical_core_num=logical_core_num,
        resources=resources,
        state_is_compact=state_is_compact,
        save_final_state=save_final_state,
        save_intermediate_state=save_intermediate_state,
        save_v_new=save_v_new,
    )


@jit
def run_staged_flash_kda_fwd_arch92(
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
    save_final_state: int,
):
    block_idx = get_block_idx()
    block_num = get_block_num()
    metadata_status = gm_metadata[HEADER_STATUS]
    metadata_core_num = gm_metadata[HEADER_CORE_NUM]
    if metadata_status == STATUS_OK and metadata_core_num == block_num:
        stage12_round_num = gm_metadata[HEADER_STAGE12_ROUND_NUM]
        _prime_stage_barriers()
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
                storage_batch = gm_metadata[storage_batch_idx_base + active_batch_idx]
                storage_batch_end = storage_batch + 1
                token_start = gm_metadata[token_start_base + active_batch_idx]
                seq_len = dtypes.int64(
                    gm_metadata[valid_seq_len_base + active_batch_idx]
                )
                group_start = gm_metadata[chunk_start_base + active_batch_idx]
                chunk_num_per_group = gm_metadata[chunk_num_base + active_batch_idx]
                chunks_total = (seq_len + (CHUNK_SIZE - 1)) // CHUNK_SIZE
                tail_rows = seq_len - (chunks_total - 1) * CHUNK_SIZE
                tail_group = group_start + chunk_num_per_group == chunks_total
                token_stop = token_start + seq_len
                k = gm_K[storage_batch, None, token_start:token_stop, None]
                v = gm_V[storage_batch, None, token_start:token_stop, None]
                q = gm_Q[storage_batch, None, token_start:token_stop, None]
                beta = gm_beta[storage_batch, None, token_start:token_stop, None]
                gate = gm_g[storage_batch, None, token_start:token_stop, None]
                value_heads = v.shape[1]
                gate_batch, gate_start, gate_stop = (
                    storage_batch,
                    token_start,
                    token_stop,
                )
                if save_g_cumsum == 0:
                    gate_batch, gate_start, gate_stop = 0, 0, 1
                gate_cache = gm_g_cumsum[
                    gate_batch : gate_batch + 1, None, gate_start:gate_stop, None
                ]
                _run_stage1_body(
                    tail_group,
                    tail_rows,
                    k,
                    v,
                    q,
                    beta,
                    gate,
                    gm_qk_exchange,
                    gm_A_log,
                    gm_dt_bias,
                    gm_q_decayed,
                    gm_kg[storage_batch, None, token_start:token_stop, None],
                    gm_gamma,
                    gm_u[storage_batch, None, token_start:token_stop, None],
                    gm_saved_w[storage_batch, None, token_start:token_stop, None],
                    gm_qg[storage_batch, None, token_start:token_stop, None],
                    gate_cache,
                    gm_identity16,
                    gm_q_norm[storage_batch, None, token_start:token_stop, None],
                    gm_k_norm[storage_batch, None, token_start:token_stop, None],
                    gm_q_rstd[storage_batch, None, token_start:token_stop, None],
                    gm_k_rstd[storage_batch, None, token_start:token_stop, None],
                    gm_beta_activated[
                        storage_batch, None, token_start:token_stop, None
                    ],
                    gm_Aqk[
                        storage_batch:storage_batch_end,
                        None,
                        token_start:token_stop,
                        None,
                    ],
                    gm_Akk[
                        storage_batch:storage_batch_end,
                        None,
                        token_start:token_stop,
                        None,
                    ],
                    prefix_start,
                    value_heads,
                    chunks_total,
                    seq_len,
                    0,
                    group_start,
                    chunk_num_per_group,
                    chunk_num_per_group,
                    block_num,
                    task_start,
                    task_end,
                    scale_value,
                    lower_bound,
                    use_qk_l2norm_in_kernel,
                    use_gate_in_kernel,
                    use_beta_sigmoid_in_kernel,
                    allow_neg_eigval,
                    has_lower_bound,
                    save_qg,
                    save_g_cumsum,
                    save_matrices,
                )
                _pair_boundary_reset()
            _global_stage_boundary(cross_block_flag_id=7)
            _rewind_buffers()
            stage2_core_range_word = stage2_core_ranges_base + block_idx * 2
            stage2_core_start = gm_metadata[stage2_core_range_word]
            stage2_core_end = gm_metadata[stage2_core_range_word + 1]
            round_dv_splits_num = _COMPILE_DV_SPLITS
            _run_stage2_dynamic(
                gm_kg,
                gm_gamma,
                gm_q_decayed,
                gm_saved_w,
                gm_u,
                gm_v_new,
                gm_h,
                gm_O,
                gm_state_in,
                gm_state_out,
                gm_Aqk,
                gm_metadata,
                dv_splits_num=round_dv_splits_num,
                value_heads=gm_V.shape[1],
                active_batch_num=active_batch_num,
                batch_idx_base=batch_idx_base,
                storage_batch_idx_base=storage_batch_idx_base,
                token_start_base=token_start_base,
                valid_seq_len_base=valid_seq_len_base,
                chunk_start_base=chunk_start_base,
                chunk_num_base=chunk_num_base,
                stage1_task_prefix_base=stage1_task_prefix_base,
                stage2_task_prefix_base=stage2_task_prefix_base,
                stage2_core_start=stage2_core_start,
                stage2_core_end=stage2_core_end,
                logical_core_num=block_num,
                state_is_compact=state_is_compact,
                save_final_state=save_final_state,
                save_intermediate_state=save_intermediate_state,
                save_v_new=save_qg,
            )
            _global_stage_boundary(cross_block_flag_id=5)
            _rewind_buffers()
        _drain_stage_barriers()


def staged_flash_kda_fwd_arch92(
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
    cu_seqlens_cpu: Optional[list[int]] = None,
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
    return fwd._flash_kda_fwd_arch92_impl(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
        internal_template=TEMPLATE_STAGED,
    )


@export("staged_flash_kda_fwd_arch92")
def export_staged_flash_kda_fwd_arch92():
    """收集直接模板编译请求，不启动 Provider。"""
    try:
        for dv_splits in (1, 2, 4, 8):
            fwd._get_compiled_kernel(TEMPLATE_STAGED, dv_splits)
    finally:
        fwd.clear_caches()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_flash_kda_fwd_arch92", "FRAGMENT")
_GRAPH_LIBRARY.define(
    "staged_flash_kda_fwd_arch92(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "Tensor? metadata, Tensor? initial_state=None, Tensor? A_log=None, Tensor? dt_bias=None, "
    "Tensor? cu_seqlens=None, int[]? cu_seqlens_cpu=None, Tensor? seqused=None, *, "
    f"Scalar? scale={SUPPORTED_HEAD_DIM ** (-0.5)}, "
    "Scalar output_final_state=False, Scalar return_intermediate_states=False, "
    "Scalar use_qk_l2norm_in_kernel=True, Scalar use_gate_in_kernel=True, "
    "Scalar use_beta_sigmoid_in_kernel=True, Scalar allow_neg_eigval=False, "
    'Scalar disable_recompute=False, Scalar? lower_bound=None, str layout="TND") '
    "-> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
)


@torch.library.impl(_GRAPH_LIBRARY, "staged_flash_kda_fwd_arch92", "Meta")
def _staged_flash_kda_fwd_arch92_meta(
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
    cu_seqlens_cpu: Optional[list[int]] = None,
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
    return fwd._flash_kda_fwd_arch92_meta(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
    )


@torch.library.impl(_GRAPH_LIBRARY, "staged_flash_kda_fwd_arch92", "PrivateUse1")
def _staged_flash_kda_fwd_arch92_privateuse1(
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
    cu_seqlens_cpu: Optional[list[int]] = None,
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
    return staged_flash_kda_fwd_arch92(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
    )


staged_flash_kda_fwd_arch92_op = (
    torch.ops.cannbotdsl_flash_kda_fwd_arch92.staged_flash_kda_fwd_arch92
)


@torch._dynamo.substitute_in_graph(
    staged_flash_kda_fwd_arch92, can_constant_fold_through=False
)
def _staged_flash_kda_fwd_arch92_graph(
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
    cu_seqlens_cpu: Optional[list[int]] = None,
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
    return staged_flash_kda_fwd_arch92_op(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
    )


def cast_and_pack_bf16(value, lane_mask):
    value_bf16 = vcast(value, dtypes.bfloat16, mask=lane_mask)
    packed_value = vpack(vreinterpret_lanes(value_bf16, dtypes.uint32), dtypes.uint16)
    return vreinterpret(packed_value, dtypes.bfloat16)
