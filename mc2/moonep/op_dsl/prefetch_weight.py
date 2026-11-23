# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""prefetch_weight：把本 rank 的专家权重（连同可选的 MX scale）推到各消费
rank 的预取槽里。

- 权重按命名参数传入（gate/up/down 或 gate_up/down），形状 `[epn, ...]`，
  行宽从尾维展平推断；
- fp8 权重必须同序带上 `local_scales`（`float8_e8m0fnu`，`Sg` 32B 对齐）；
  bf16/fp16 不用传；
- 返回 `(views, scales)`：每组前 `epn` 行是本地专家，后 `epn` 行是预取槽；
  `async_finish=True` 时再追加一个完成事件。
"""

from __future__ import annotations

import threading
import os
from typing import NamedTuple, Union

import cannbotdsl
import torch
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_vf_len
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    mask_and,
    mask_counter,
    update_mask,
    vadds,
    vdups,
    vges,
    vload,
    vlts,
    vselect,
    vstore,
)
from cannbotdsl.ops.sync import PIPE, vec_sync_notify, vec_sync_wait
from cannbotdsl.tensor import MemLoc, make_layout, tile_slice
from cannbotdsl.types.datastruct import StructRef
from moonep_context import (
    HEAP_BASE,
    ROWS_PER_EXPERT,
    SCALE_ALIGN_BYTES,
    check_heap_capacity,
    finish_op,
    group_slot_rows,
    next_barrier_epoch,
    prefetch_weight_off,
    reduce_stg_off,
    resident_blocks,
    slot_stride_bytes,
    sync_transport_stream,
    workspace_base,
)
from moonep_layout import NOTIFY_STRIDE, PREFETCH_SLICE_FLAGS_OFF
from moonep_comm import (
    MoonepCommCtx,
    _gm_view,
    make_xfer_rb,
    spin_u32_ge,
    tensor_address,
    ub_flag_write_snapshot,
    ub_write,
    xfer_open,
)
from npu_blob import npu_from_blob

_WeightViews = tuple[torch.Tensor, ...]
_ScaleViews = tuple[torch.Tensor, ...]
_PrefetchResult = Union[
    tuple[_WeightViews, _ScaleViews],
    tuple[tuple[_WeightViews, _ScaleViews], torch.Event],
]

_MAX_PUSH_BLOCKS = 64
# 行分片上限：与 reduce 的 MOONEP_ACC_SPLIT 同构——每 peer 最多 8 个 block
# 并发驱动其链路（跨 supernode 链路单核仅 ~26 GB/s，需多核聚合）。
_PUSH_SPLIT = 8


def _push_block_limit():
    """Resolve the opt-in push block ceiling for concurrent workloads."""
    configured = int(os.environ.get("MOONEP_PUSH_BLOCKS", _MAX_PUSH_BLOCKS))
    if configured < 1:
        raise ValueError("MOONEP_PUSH_BLOCKS must be >= 1")
    return min(resident_blocks(_MAX_PUSH_BLOCKS), configured)


# UbMem 深环承载在飞量，prefetch 侧 UB 占用小。
_PREFETCH_XFER_DEPTH = 6
_MAX_PUSH_PROGRAMS = 64
_PUSH_EVENT_ID = 4
_SCAN_EVENT_ID = 5
_SNAP_EVENT_ID = 6  # epoch 快照 MTE2→S（旗语会合单读）
_MAX_WEIGHT_GROUPS = 3  # 权重组槽上限（2 组时第 3 槽不参与推送）
_MAX_SCALE_GROUPS = 3  # scale 组槽上限（fp8 形态每组权重配一个 scale 组）
_MAX_GROUPS = _MAX_WEIGHT_GROUPS + _MAX_SCALE_GROUPS  # kernel 入口固定 6 张量槽
_MIN_GROUPS = 2  # 支持的权重组数下限
_BITS_PER_BYTE = 8

_WEIGHT_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)  # 权重 dtype 白名单（bf16 / fp16 / e4m3fn / e5m2）


# --------------------------------------------------------------------------
# 参数与布局
# --------------------------------------------------------------------------
def _is_fp8_weight(dtype):
    """是否 fp8 权重。"""
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _push_layout(rank_size, block_limit=None):
    """块布局：(nblk, per_block, n_slices)。

    优先「每 peer n_slices 个 block 行分片」（n_slices ≤ _PUSH_SPLIT）；
    常驻核数不足时退回 block-per-peer（n_slices=1, per_block>1）。分片模式
    下 nblk 裁齐到 n_rem 整数倍，保证每个 peer 的分片数一致。
    """
    n_rem = max(rank_size - 1, 0)
    if n_rem <= 0:
        return 1, 1, 1
    resident = _push_block_limit() if block_limit is None else int(block_limit)
    want = min(n_rem * _PUSH_SPLIT, resident)
    if want >= n_rem:
        nblk = (want // n_rem) * n_rem
        return nblk, 1, nblk // n_rem
    nblk = max(want, 1)
    return nblk, (n_rem + nblk - 1) // nblk, 1


# --------------------------------------------------------------------------
# 设备侧 kernel
# --------------------------------------------------------------------------
@kernel
class _MoonepPrefetchPushKernel:
    """按组推送：窗口是一维 uint8 workspace 视图，每组以“本组一行”为瓦片。

    控制面在算子内：rank 取自 ``MoonepCommCtx``，组
    起始行号与槽步长由编译期 ``sizes`` 计算，收尾 barrier 内联在推送 kernel
    里（flag 与数据同通道保序）。单一入口固定 6 个张量槽（3 权重 + 3
    scale），``ngroups``（权重组数 + scale 组数，2..6）在 trace 阶段裁剪实际
    推送组数；缺组槽由 host 填入重复值、不读取（现状 2 组 pad 逻辑的自然
    延拓）。

    组宽 ``sizes``（元素数）与逐组要素字节宽 ``elem_bytes``（权重 bf16 2B /
    fp8 1B；scale 恒 1B）均为编译期常量（tile_slice 的瓦片尺寸必须静态）；
    tile_slice 的坐标是瓦片序号，因此组基址用“本组一行字节宽”为单位的行号
    表示。各槽区按 (sizes × elem_bytes) 字节前缀和紧密排布，组宽可以不同
    （交织型 W13=2W、down=W；scale 组 Sg 与权重组 Wg 独立）。
    """

    def __init__(
        self,
        rank_size,
        epn,
        sizes,
        elem_bytes,
        ngroups,
        nblk,
        per_block,
        n_slices=1,
    ):
        self._r = int(rank_size)
        self._epn = int(epn)
        # 固定 6 槽，缺省槽用末组占位、不推送
        self._sizes = tuple(int(s) for s in sizes)
        self._elem_bytes = tuple(int(b) for b in elem_bytes)
        self._ngroups = int(ngroups)
        # 逐组一行字节数
        self._row_bytes = tuple(s * b for s, b in zip(self._sizes, self._elem_bytes))
        ngroups = self._ngroups
        actual = self._row_bytes[:ngroups]
        self._group_rows, self._slot_rows = group_slot_rows(self._epn, actual)
        self._nblk = int(nblk)
        self._per_block = int(per_block)
        self._n_slices = int(n_slices)

    def __call__(
        self,
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        gm_etc,
        slot,
        gm_epoch,
        ctx: StructRef[MoonepCommCtx],
    ):
        world_size, _ngroups = self._r, self._ngroups
        # 真实组在前、pad 槽在后
        reals = (k0, k1, k2, k3, k4, k5)
        # rank 从 ctx 读，不进编译键
        my_rank = dtypes.int64(ctx.rankId)
        my_rank_i32 = dtypes.int32(ctx.rankId)
        my_rank_u64 = dtypes.uint64(ctx.rankId)
        xfer_rb = make_xfer_rb(_PREFETCH_XFER_DEPTH)
        epoch_addr = tensor_address(gm_epoch)
        push_base = dtypes.uint64(HEAP_BASE + prefetch_weight_off())

        blk = dtypes.int64(get_block_idx())
        n_rem = dtypes.int64(world_size - 1)
        ub_etc = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn)))
        ub_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, self._epn)))
        if const_expr(self._n_slices > 1):
            # 行分片模式：每 peer n_slices 个 block 并发驱动同一条链路。
            self._push_one(
                ctx,
                gm_etc,
                ub_etc,
                ub_row,
                my_rank,
                my_rank_i32,
                my_rank_u64,
                epoch_addr,
                push_base,
                reals,
                slot,
                xfer_rb,
                blk % n_rem,
                blk // n_rem,
            )
        else:
            for peer_slot in range_constexpr(self._per_block):
                peer_id = blk + dtypes.int64(peer_slot * self._nblk)
                if peer_id < n_rem:
                    self._push_one(
                        ctx,
                        gm_etc,
                        ub_etc,
                        ub_row,
                        my_rank,
                        my_rank_i32,
                        my_rank_u64,
                        epoch_addr,
                        push_base,
                        reals,
                        slot,
                        xfer_rb,
                        peer_id,
                        dtypes.int64(0),
                    )

    @jit
    def _push_one(
        self,
        ctx,
        gm_etc,
        ub_etc,
        ub_row,
        my_rank,
        my_rank_i32,
        my_rank_u64,
        epoch_addr,
        push_base,
        reals,
        slot,
        xfer_rb,
        peer_id,
        slice_id,
    ):
        """向单个 peer 推送本 block 分管的行（peer_id 为紧凑序号）。

        分片模式下 ``slice_id`` 为行分片号；slice-0 block 等本 peer 全部分片
        的本地完成标志后，才向对端写原有的 per-sender flag。
        """
        epn, ngroups = self._epn, self._ngroups
        row_bytes = self._row_bytes
        epn64 = dtypes.int64(epn)
        peer = peer_id if peer_id < my_rank else peer_id + dtypes.int64(1)
        # peer 索引按排序非自身 rank 紧凑编号（peer_id 本身即紧凑序号）
        peerh = xfer_open(ctx, dtypes.uint32(dtypes.uint64(peer_id)))
        # epoch 只在 kernel 开头读一次快照：自旋阈值、本地分片标志和
        # 远端 flag 必须同代；GM 分次读会漂移
        ub_epoch = Buffer(MemLoc.UB, dtypes.uint32, make_layout((1,)))
        mem_copy(ub_epoch, _gm_view(epoch_addr, 1, dtypes.uint32))
        vec_sync_notify(PIPE.MTE2, PIPE.S, _SNAP_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.S, _SNAP_EVENT_ID)
        epoch = dtypes.uint32(ub_epoch[dtypes.int64(0)])
        self._scan_peer(gm_etc, peer, ub_etc, ub_row, my_rank_i32)

        # 逐行直推，每行 ngroups 笔（分片模式以 slice_id 起步、n_slices 步进）
        for i in cannbotdsl.range(slice_id, epn64, dtypes.int64(self._n_slices)):
            row = dtypes.int64(ub_row[dtypes.int64(0), i])
            if row >= dtypes.int64(0):
                # 目的槽行号 = epn + i
                slot_row = epn64 + i
                for g in range_constexpr(_MAX_GROUPS):
                    if g < ngroups:  # 编译期裁剪：缺组槽不发射
                        dst = (
                            dtypes.int64(self._group_rows[g])
                            + slot * dtypes.int64(self._slot_rows[g])
                            + slot_row
                        )
                        remote_off = push_base + dtypes.uint64(dst) * dtypes.uint64(
                            row_bytes[g]
                        )
                        src_addr = tensor_address(reals[g]) + dtypes.uint64(
                            row
                        ) * dtypes.uint64(row_bytes[g])
                        ub_write(
                            peerh,
                            remote_off,
                            src_addr,
                            self._row_bytes[g],
                            xfer_rb,
                        )
        if const_expr(self._n_slices > 1):
            # 每个分片完成后写本地 (peer, slice) 标志（MTE3 同 pipe，数据
            # 先于标志）；slice-0 汇总本 peer 全部分片后再发远端 flag。
            slice_idx = dtypes.uint64(peer_id) * dtypes.uint64(
                int(self._n_slices)
            ) + dtypes.uint64(slice_id)
            slice_flag_addr = dtypes.uint64(ctx.localWsAddr) + (
                dtypes.uint64(PREFETCH_SLICE_FLAGS_OFF) + slice_idx * dtypes.uint64(4)
            )
            mem_copy(_gm_view(slice_flag_addr, 1, dtypes.uint32), ub_epoch)
            if slice_id == dtypes.int64(0):
                for k in range_constexpr(self._n_slices):
                    k_idx = dtypes.uint64(peer_id) * dtypes.uint64(
                        int(self._n_slices)
                    ) + dtypes.uint64(k)
                    spin_u32_ge(
                        ctx,
                        PREFETCH_SLICE_FLAGS_OFF + k_idx * dtypes.uint64(4),
                        epoch,
                    )
                ub_flag_write_snapshot(
                    peerh,
                    my_rank_u64 * dtypes.uint64(NOTIFY_STRIDE),
                    ub_epoch,
                )
                spin_u32_ge(
                    ctx,
                    dtypes.uint64(peer) * dtypes.uint64(NOTIFY_STRIDE),
                    epoch,
                )
        else:
            # 尾部 flag 与数据同通道保序（快照写首尾 fence——
            # MTE3 同 pipe：flag 落地即此前全部行已到对端）
            ub_flag_write_snapshot(
                peerh,
                my_rank_u64 * dtypes.uint64(NOTIFY_STRIDE),
                ub_epoch,
            )
            # 等对端收到本轮 flag
            spin_u32_ge(
                ctx,
                dtypes.uint64(peer) * dtypes.uint64(NOTIFY_STRIDE),
                epoch,
            )

    @jit
    def _scan_peer(self, gm_etc, peer, ub_etc, ub_row, my_rank_i32):
        epn = self._epn
        # VL 按 int32 lane 数换算
        vl = (
            get_vf_len()
            * dtypes.int64(_BITS_PER_BYTE)
            // dtypes.int64(dtypes.int32.bits)
        )
        # 按 VL 分块，保持硬件循环
        n_chunks = (dtypes.int64(epn) + vl - 1) // vl
        lo_e = my_rank_i32 * dtypes.int32(epn)  # 本 rank 负责的专家区间 [lo, hi)
        hi_e = lo_e + dtypes.int32(epn)
        mem_copy(
            tile_slice(ub_etc, (1, epn), (0, 0)),
            tile_slice(gm_etc, (1, epn), (peer, 0)),
        )
        vec_sync_notify(PIPE.MTE2, PIPE.V, _SCAN_EVENT_ID)
        vec_sync_wait(PIPE.MTE2, PIPE.V, _SCAN_EVENT_ID)
        with vf(mode="simd"):
            minus_one = vdups(dtypes.int32(-1), dtypes.int32)
            remaining = mask_counter(epn)
            for chunk in cannbotdsl.range(n_chunks):
                off = chunk * vl
                lanes, remaining = update_mask(remaining, elem_bits=32)
                expert = vload(ub_etc, off)
                in_range = mask_and(
                    vges(expert, lo_e, mask=lanes),
                    vlts(expert, hi_e, mask=lanes),
                    exec_mask=lanes,
                )
                row = vadds(expert, -lo_e, mask=lanes)
                vstore(
                    ub_row,
                    off,
                    vselect(row, minus_one, cond_mask=in_range),
                    mask=lanes,
                )
        vec_sync_notify(PIPE.V, PIPE.S, _PUSH_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.S, _PUSH_EVENT_ID)


# --------------------------------------------------------------------------
# Host 配置与编译缓存
# --------------------------------------------------------------------------
class _PushConfig(NamedTuple):
    rank_size: int
    epn: int
    sizes: tuple  # 6 槽元素宽（权重 3 + scale 3，缺省槽复用末组）
    elem_bytes: tuple  # 6 槽逐组要素字节宽（bf16 2B / fp8·scale 1B）
    ngroups: int  # 实际推送组数（权重组数 + scale 组数）
    window_bytes: int
    nblk: int
    per_block: int
    n_slices: int = 1  # 每 peer 行分片数（1 = block-per-peer）


def _make_push_host(cfg):
    @host
    def _push_host(
        k0,
        k1,
        k2,
        k3,
        k4,
        k5,
        gm_etc,
        slot: int,
        gm_epoch,
        ctx: StructRef[MoonepCommCtx],
    ):
        _MoonepPrefetchPushKernel(
            cfg.rank_size,
            cfg.epn,
            cfg.sizes,
            cfg.elem_bytes,
            cfg.ngroups,
            cfg.nblk,
            cfg.per_block,
            cfg.n_slices,
        )[cfg.nblk](k0, k1, k2, k3, k4, k5, gm_etc, slot, gm_epoch, ctx)

    return _push_host


_PUSH_PROGRAMS = {}
_PUSH_LOCK = threading.Lock()


def _get_push_program(
    epn,
    weight_sizes,
    scale_sizes,
    weight_bytes,
    window_bytes,
    rank_size,
    block_limit=None,
):
    """按配置编译/复用推送程序；sizes/elem_bytes 进键，rank 不进键。"""
    n_w = len(weight_sizes)
    n_s = len(scale_sizes)
    ngroups = n_w + n_s
    real_sizes = tuple(weight_sizes) + tuple(scale_sizes)
    real_elem = (weight_bytes,) * n_w + (1,) * n_s
    if n_s:
        pad_sizes = (weight_sizes[-1],) * (_MAX_WEIGHT_GROUPS - n_w) + (
            scale_sizes[-1],
        ) * (_MAX_SCALE_GROUPS - n_s)
        pad_elem = (weight_bytes,) * (_MAX_WEIGHT_GROUPS - n_w) + (1,) * (
            _MAX_SCALE_GROUPS - n_s
        )
    else:
        pad_sizes = (weight_sizes[-1],) * (_MAX_GROUPS - n_w)
        pad_elem = (weight_bytes,) * (_MAX_GROUPS - n_w)
    sizes = real_sizes + pad_sizes
    elem6 = real_elem + pad_elem
    nblk, per_blk, n_slices = _push_layout(rank_size, block_limit)
    cfg = _PushConfig(
        rank_size,
        epn,
        tuple(sizes),
        tuple(elem6),
        ngroups,
        int(window_bytes),
        nblk,
        per_blk,
        n_slices,
    )
    with _PUSH_LOCK:
        prog = _PUSH_PROGRAMS.get(cfg)
        if prog is None:
            # 1B 家族统一 uint8 视图入 TensorSpec
            specs = [
                TensorSpec((epn, s), _dsl_dtype(b))
                for s, b in zip(cfg.sizes, cfg.elem_bytes)
            ]
            specs.append(TensorSpec((rank_size, epn), dtypes.int32))
            specs.append(dtypes.int64)  # 窗口槽位
            specs.append(TensorSpec((1,), dtypes.int32))  # 屏障 epoch
            specs.append(StructRef[MoonepCommCtx])
            prog = cannbotdsl.compile(_make_push_host(cfg), *specs)
            if len(_PUSH_PROGRAMS) >= _MAX_PUSH_PROGRAMS:
                _PUSH_PROGRAMS.pop(next(iter(_PUSH_PROGRAMS)))
            _PUSH_PROGRAMS[cfg] = prog
    return prog


# --------------------------------------------------------------------------
# Host 参数与视图辅助
# --------------------------------------------------------------------------
def _dsl_dtype(elem_bytes):
    """元素字节宽 → DSL dtype（2B=bf16，1B=uint8）。"""
    if elem_bytes == 2:
        return dtypes.bfloat16
    return dtypes.uint8


def _torch_dtype_str(dtype):
    """torch dtype → npu_from_blob 的 dtype 字符串。"""
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.uint8:
        return "uint8"
    if dtype == torch.float8_e4m3fn:
        return "e4m3fn"
    if dtype == torch.float8_e5m2:
        return "e5m2"
    if dtype == torch.float8_e8m0fnu:
        return "e8m0fnu"
    if dtype == torch.float32:
        return "fp32"
    raise TypeError(f"不支持的视图 dtype {dtype}")


def _as_sizes(sizes):
    """归一化每组件宽（元素数），要求为正。"""
    sizes = tuple(int(s) for s in sizes)
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"每组行宽必须为正，实得 {sizes}")
    return sizes


def _freeze_layout(context, widths, scale_widths, *, what):
    """首调定型组数、各组行宽和 scale 行宽；后续变化立即报错。"""
    if context.layout_groups is None:
        context.layout_groups = len(widths)
    elif context.layout_groups != len(widths):
        raise RuntimeError(
            f"窗口组数不一致（已有 {context.layout_groups}，本次 {len(widths)}），"
            "会导致发布区重叠"
        )
    if context.layout_widths is None:
        context.layout_widths = tuple(int(w) for w in widths)
    elif tuple(int(w) for w in widths) != context.layout_widths:
        raise RuntimeError(
            f"窗口组宽不一致（已有 {context.layout_widths}，本次 {tuple(widths)}）"
            f"（{what}）：宽度变化会使梯度槽区基址错位、发布/槽区重叠"
        )
    scale_widths = tuple(int(s) for s in scale_widths) if scale_widths else ()
    if context.layout_scale_widths is None:
        context.layout_scale_widths = scale_widths
    elif scale_widths != context.layout_scale_widths:
        raise RuntimeError(
            f"scale 组宽不一致（已有 {context.layout_scale_widths}，"
            f"本次 {scale_widths}）（{what}）：scale 随 slot 窗口轮转，"
            "宽度变化会使窗口步长错位"
        )


def _create_weight_views(context, epn, sizes, scale_sizes, *, slot=0, slots=1, dtype):
    """创建（或复用）权重视图 ``[2*epn, Wg]``。"""
    sizes = _as_sizes(sizes)
    scale_sizes = tuple(int(s) for s in scale_sizes) if scale_sizes else ()
    weight_bytes = int(dtype.itemsize)
    group_slot_rows(
        epn, tuple(int(w) * weight_bytes for w in sizes) + tuple(scale_sizes)
    )
    _freeze_layout(context, sizes, scale_sizes, what="prefetch 权重视图")
    if context.last_plan_shape is None:
        raise RuntimeError(
            "创建权重视图前请设置 context.last_plan_shape（输入形状与窗口布局）"
        )
    if slot < 0 or slot >= slots:
        raise ValueError(f"slot={slot} 超出槽位数 {slots}")
    if context.window_slots is None:
        context.window_slots = slots
    elif context.window_slots != slots:
        raise RuntimeError(
            f"窗口槽位数不一致（已有 {context.window_slots}，本次 {slots}）"
        )
    key = ("views", int(epn), sizes, int(slot), int(slots), str(dtype))
    cached = context.weight_views.get(key)
    if cached is not None:
        return cached
    check_heap_capacity(
        context,
        HEAP_BASE
        + reduce_stg_off(
            epn,
            slots=slots,
            widths=sizes,
            scale_widths=scale_sizes or None,
            weight_bytes=weight_bytes,
        ),
        "workspace 布局（多组权重发布区）",
    )
    slot_bytes = slot_stride_bytes(epn, sizes, scale_sizes or None, weight_bytes)
    addr = (
        workspace_base(context) + HEAP_BASE + prefetch_weight_off() + slot * slot_bytes
    )
    views = []
    for w in sizes:
        views.append(
            npu_from_blob(addr, (ROWS_PER_EXPERT * epn, w), _torch_dtype_str(dtype))
        )
        addr += ROWS_PER_EXPERT * epn * w * weight_bytes
    views = tuple(views)
    # bugfix: 副本段不要本地 fill：空槽本来就没人读，而填充会和远端
    # 推过来的行抢写
    context.weight_views[key] = views
    return views


def _create_scale_views(
    context, epn, weight_sizes, scale_sizes, *, slot=0, slots=1, weight_bytes=2
):
    """创建（或复用）scale 视图 ``[2*epn, Sg]``。"""
    scale_sizes = _as_sizes(scale_sizes)
    group_slot_rows(
        epn, tuple(int(w) * weight_bytes for w in weight_sizes) + tuple(scale_sizes)
    )
    _freeze_layout(context, weight_sizes, scale_sizes, what="prefetch scale 视图")
    weight_bytes_total = sum(int(w) for w in weight_sizes) * weight_bytes
    key = (
        "scale_views",
        int(epn),
        tuple(int(w) for w in weight_sizes),
        tuple(int(s) for s in scale_sizes),
        int(weight_bytes),
        int(slot),
        int(slots),
    )
    cached = context.weight_views.get(key)
    if cached is not None:
        return cached
    slot_bytes = slot_stride_bytes(epn, weight_sizes, scale_sizes, weight_bytes)
    addr = (
        workspace_base(context)
        + HEAP_BASE
        + prefetch_weight_off()
        + slot * slot_bytes
        + weight_bytes_total * ROWS_PER_EXPERT * epn
    )
    views = []
    for s in scale_sizes:
        views.append(npu_from_blob(addr, (ROWS_PER_EXPERT * epn, s), "e8m0fnu"))
        addr += ROWS_PER_EXPERT * epn * s
    views = tuple(views)
    context.weight_views[key] = views
    return views


def _sync_transport(local_weights, views, epn):
    """把每组前 epn 行写入视图 home 段；同址时跳过。"""
    for real_w, view_w in zip(local_weights, views):
        if real_w.data_ptr() == view_w[:epn].data_ptr():
            continue
        if real_w.dtype.itemsize == 1:
            view_w[:epn].view(torch.uint8).copy_(real_w.view(torch.uint8))
        else:
            view_w[:epn] = real_w


def _window_bytes(context, epn, sizes, scale_sizes, weight_bytes):
    """发布窗口总字节数（slots × LCM 对齐步长——与 kernel SymBuffer 同式）。"""
    slots = context.window_slots or 1
    return slots * slot_stride_bytes(epn, sizes, scale_sizes or None, weight_bytes)


def _normalize_groups(local_weights, epn, context=None):
    """归一化 2/3 组权重，返回 ``(flats, widths, dtype)``。"""
    epn = int(epn)
    if epn <= 0:
        raise ValueError(f"epn 必须为正，实得 {epn}")
    if isinstance(local_weights, torch.Tensor):
        raise TypeError(
            "local_weights 须为 2/3 组 list/tuple（每组 [epn, ...]）——不支持单张量"
        )
    local_weights = tuple(local_weights)
    if len(local_weights) not in (_MIN_GROUPS, _MAX_WEIGHT_GROUPS):
        raise ValueError(
            f"prefetch 支持 2 组或 3 组权重（kernel 入口按组数固定），"
            f"实得 {len(local_weights)} 组"
        )
    flats, sizes = [], []
    dtype = None
    for g, weight in enumerate(local_weights):
        if not isinstance(weight, torch.Tensor):
            raise TypeError(
                f"第 {g} 组权重必须是 torch.Tensor，实得 {type(weight).__name__}"
            )
        if weight.dim() < 2 or int(weight.shape[0]) != int(epn):
            raise ValueError(
                f"第 {g} 组权重形状应为 (epn, ...) >= 2 维，实得 {tuple(weight.shape)}"
            )
        if not weight.is_contiguous():
            raise ValueError(f"第 {g} 组须连续（展平前置）")
        if dtype is None:
            dtype = weight.dtype
        elif weight.dtype != dtype:
            raise TypeError(
                f"多权重组的 dtype 必须一致（第 {g} 组 {weight.dtype} != {dtype}）"
            )
        w = int(weight.numel() // int(epn))
        flats.append(weight.reshape(int(epn), w))
        sizes.append(w)
    if dtype not in _WEIGHT_DTYPES:
        raise TypeError(
            "权重 dtype 仅支持 bf16 / fp16 / float8_e4m3fn / float8_e5m2"
            f"（dtype 终态，uint8 不支持），got {dtype}"
        )
    if context is not None:
        frozen = getattr(context, "weight_dtype", None)
        if frozen is None:
            context.weight_dtype = dtype  # 裸 API：首调冻结
        elif frozen != dtype:
            raise TypeError(
                f"权重 dtype 与定型值不符（构造/首调 {frozen}，本次 {dtype}）"
                "——dtype 进编译键与视图，不可运行期漂移；请以一致的 dtype "
                "重建或转换权重"
            )
    return tuple(flats), _as_sizes(sizes), dtype


def _normalize_scales(local_scales, epn, num_weight_groups, weight_dtype):
    """归一化 scale 组；fp8 权重必须传，bf16/fp16 不可传。"""
    if local_scales is None:
        if _is_fp8_weight(weight_dtype):
            raise ValueError(
                "mxfp8 契约：fp8 权重（e4m3fn/e5m2）⇒ 必须传 local_scales"
                "（ue8m0 block scale 与权重同推；bf16/fp16 形态不传 scale）"
            )
        return (), ()
    if not _is_fp8_weight(weight_dtype):
        raise ValueError(
            "mxfp8 契约：local_scales 仅支持 fp8 权重（e4m3fn/e5m2），"
            f"bf16/fp16 权重不能传，实得权重 dtype {weight_dtype}"
        )
    if isinstance(local_scales, torch.Tensor):
        raise TypeError(
            "local_scales 须为与权重组同长度的 list/tuple"
            f"（每组 [epn, ...] float8_e8m0fnu，共 {num_weight_groups} 组）"
            "——不支持单张量"
        )
    local_scales = tuple(local_scales)
    if len(local_scales) != num_weight_groups:
        raise ValueError(
            f"local_scales 长度须等于权重组数 {num_weight_groups}（同组同序"
            f"——量化 GEMM 按 (weights, scales) 成对消费），"
            f"实得 {len(local_scales)} 组"
        )
    flats, sizes = [], []
    for g, sc in enumerate(local_scales):
        if sc.dim() < 2 or int(sc.shape[0]) != int(epn):
            raise ValueError(
                f"第 {g} 组 scale 形状应为 (epn, ...) >= 2 维，实得 {tuple(sc.shape)}"
            )
        if not sc.is_contiguous():
            raise ValueError(f"第 {g} 组 scale 须连续（展平前置）")
        if sc.dtype != torch.float8_e8m0fnu:
            raise TypeError(
                f"第 {g} 组 scale 须为 float8_e8m0fnu（MX 标准 ue8m0 编码）"
                f"，实得 {sc.dtype}"
            )
        s = int(sc.numel() // int(epn))
        if s <= 0 or s % SCALE_ALIGN_BYTES:
            padded = (
                (s + SCALE_ALIGN_BYTES - 1) // SCALE_ALIGN_BYTES * SCALE_ALIGN_BYTES
            )
            raise ValueError(
                f"第 {g} 组 scale 行宽 Sg={s} 须为正且是 {SCALE_ALIGN_BYTES}B"
                f" 倍数（对齐约束）；请 pad 到 {padded}B"
                f"（消费端按 padded 行宽解）"
            )
        flats.append(sc.reshape(int(epn), s))
        sizes.append(s)
    return tuple(flats), tuple(sizes)


def _require_npu(tensors, name):
    for g, t in enumerate(tensors):
        if t.device.type != "npu":
            raise ValueError(f"{name}[{g}] 必须在 NPU 上，实得 {t.device}")


def _validate_etc(context, experts_to_copy, epn):
    if not isinstance(experts_to_copy, torch.Tensor):
        raise TypeError("experts_to_copy 必须是 int32 Tensor")
    if experts_to_copy.dtype != torch.int32:
        raise TypeError(f"experts_to_copy 须为 int32，实得 {experts_to_copy.dtype}")
    expected = (int(context.rank_size), int(epn))
    if experts_to_copy.dim() != 2 or tuple(experts_to_copy.shape) != expected:
        raise ValueError(
            f"experts_to_copy 形状应为 {expected}，实得 {tuple(experts_to_copy.shape)}"
        )
    if experts_to_copy.device.type != "npu":
        raise ValueError("experts_to_copy 必须在 NPU 上")
    if not experts_to_copy.is_contiguous():
        raise ValueError("experts_to_copy 必须连续")


# --------------------------------------------------------------------------
# 公开入口
# --------------------------------------------------------------------------
def prefetch_weight(
    context,
    experts_to_copy,
    local_weights,
    epn,
    *,
    local_scales=None,
    slot=0,
    slots=1,
    async_finish=False,
) -> _PrefetchResult:
    """按 experts_to_copy 把本地权重（+ 可选 scale）推到各 rank 的预取槽。

    返回 ``(weights, scales)``；``async_finish=True`` 时追加完成事件。
    """
    flats, sizes, weight_dtype = _normalize_groups(local_weights, epn, context)
    scale_flats, scale_sizes = _normalize_scales(
        local_scales, epn, len(flats), weight_dtype
    )
    _validate_etc(context, experts_to_copy, epn)
    _require_npu(flats, "local_weights")
    if scale_flats:
        _require_npu(scale_flats, "local_scales")
    weight_bytes = int(weight_dtype.itemsize)
    views = _create_weight_views(
        context,
        epn,
        sizes,
        scale_sizes,
        slot=slot,
        slots=slots,
        dtype=weight_dtype,
    )
    scale_views = (
        _create_scale_views(
            context,
            epn,
            sizes,
            scale_sizes,
            slot=slot,
            slots=slots,
            weight_bytes=weight_bytes,
        )
        if scale_sizes
        else ()
    )
    sync_transport_stream(context)
    _sync_transport(flats, views, epn)
    if scale_flats:
        _sync_transport(scale_flats, scale_views, epn)
    # 真实组在前、pad 槽在后；kernel 只按字节搬运。
    n_w = len(flats)
    n_s = len(scale_flats)
    if weight_bytes == 1:
        flats_k = tuple(f.view(torch.uint8) for f in flats)
    elif weight_dtype == torch.float16:
        flats_k = tuple(f.view(torch.bfloat16) for f in flats)
    else:
        flats_k = flats
    if n_s:
        scale_flats_k = tuple(s.view(torch.uint8) for s in scale_flats)
        reals6 = (*flats_k, *scale_flats_k) + (
            (flats_k[-1],) * (_MAX_WEIGHT_GROUPS - n_w)
            + (scale_flats_k[-1],) * (_MAX_SCALE_GROUPS - n_s)
        )
    else:
        # 无 scale：pad 全部复用末组权重张量（v1 行为等价）
        reals6 = (*flats_k, *flats_k[-1:] * (_MAX_GROUPS - n_w))
    gm_epoch = next_barrier_epoch(context)
    _get_push_program(
        epn,
        sizes,
        scale_sizes,
        weight_bytes,
        _window_bytes(context, epn, sizes, scale_sizes, weight_bytes),
        context.rank_size,
        _push_block_limit(),
    )(
        *reals6,
        experts_to_copy,
        int(slot),
        gm_epoch,
        context.ctx_tensor,
    )
    event = finish_op(context, async_finish)
    if async_finish:
        return (views, scale_views), event
    return views, scale_views
