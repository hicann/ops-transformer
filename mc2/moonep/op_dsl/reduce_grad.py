# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""reduce_grad：把散落在各 rank 的副本专家梯度收拢回 home（训练反向）。

核心是 owner 直累，数据一路这样走：

  FFN 反向产出 full-dW `t[2*epn, Wg]`（前 epn 行 home 种子、后 epn 行
  副本 dW）→ 发送侧 host 把副本行拷进本 rank workspace 的副本区
  `[epn, Wg]`（一次 GM 拷贝，大小不随 R 增长）→ 单 kernel（2/3 组一次
  launch）走三步：

  ① 头部屏障：per-peer flag 握手。对端 kernel 一启动，就说明它流序在前
     的 host 副本区写入已完成，上一轮的远读不会撞上本轮覆写。
  ② owner 映射远读副本区（etc_inv 逆映射命中）→ 确定性归并：行单写者
     分区 + 固定 peer 升序远读（滞后窗流水）+ UB fp32 累加 + 单次写回。
  ③ 尾部 sym_barrier，给下一轮 host 覆写让路。

  确定性从哪来：每行的加法次序固定（seed → peer 升序 → 终轮一次舍入），
  跨 run 逐位可重复，bf16/fp16 的舍入也从 R-1 次降到 1 次。代价是块内
  peer 串行读（链路级并发不变，吞吐对块内调度形态不敏感），UB 多出
  acc/seed/out 缓冲（16 位 chunk 16K 时环深 ≤3）。

本模块要求：E%R==0；grad_view 为 [2*epn, W]（home 段 + 副本段，行序同
prefetch 视图）。
"""

import os
import threading
from typing import NamedTuple

import cannbotdsl
import cannbotdsl as cbd
import torch
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    RoundingMode,
    UnpackMode,
    full_mask,
    vadd,
    vcast,
    vload,
    vload_unpack,
    vstore,
    vstore_pack,
)
from cannbotdsl.tensor import MemLoc, make_layout, make_pointer, make_tensor, tile_slice
from cannbotdsl.types.datastruct import StructRef
from moonep_context import (
    BAR_FLAGS_OFF,
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    check_heap_capacity,
    finish_op,
    grad_slot_off,
    next_barrier_epoch,
    normalize_group_list,
    pick_weight_chunk,
    resident_blocks,
    sync_transport_stream,
)
from moonep_context import sym_barrier
from moonep_comm import (
    MoonepCommCtx,
    connection_from_address,
    spin_u32_ge,
    tensor_address,
    ub_flag_write,
    xfer_open,
)

# _VL：fp32 归并步进（vf 寄存器 256B//4 = 64）。
_VL = 64
_MAX_GRAD_GROUPS = 3  # 梯度组槽上限（kernel 入口固定 3 张量槽，ngroups 裁剪）

# merge UB 预算：ub_row（grad dtype chunk）+ ub_acc（fp32 chunk）≤ ~96KB
_CHUNK_UB_LIMIT_BF16 = 16384
_CHUNK_UB_LIMIT_FP32 = 8192
_HEADER_BARRIER_CAP = 64  # 头部屏障/读块上限（950PR vector_core_num 满额）
_UB_BYTES = 253952
_MAX_ACC_DEPTH = 16


def _acc_depth(rank_size):
    """Pipeline depth for a rank group, with an optional environment override."""
    default_depth = 4 if int(rank_size) <= 8 else 3
    depth = int(os.environ.get("MOONEP_ACC_DEPTH", str(default_depth)))
    if not 2 <= depth <= _MAX_ACC_DEPTH:
        raise ValueError(f"MOONEP_ACC_DEPTH={depth} must be in [2, {_MAX_ACC_DEPTH}]")
    return depth


def _acc_chunk_limit(depth, grad_16bit):
    """Largest power-of-two chunk fitting the depth-aware UB budget."""
    bytes_per_element = 2 * depth + 8 if grad_16bit else 4 * (depth + 1)
    power = 1 << ((_UB_BYTES // bytes_per_element).bit_length() - 1)
    base_limit = _CHUNK_UB_LIMIT_BF16 if grad_16bit else _CHUNK_UB_LIMIT_FP32
    return min(base_limit, power)


@jit
def _barrier_flags_put(ctx, blk, my_rank_u64, n_peer, n_writer, per_writer, gm_epoch):
    """头部屏障写者块：把 my flag 写到本块分管的全部 peer（轮转分配）。

    先写完分管 peer 再进自旋（调用方在循环外等全部 peer）——写者块全在
    首波，写先于等，避免分波后互相等待。ub_flag_write 首尾 fence（MTE3
    同通道保序），无需额外 drain。
    """
    for slot in range_constexpr(int(per_writer)):
        p_slot = blk + dtypes.int64(slot * int(n_writer))
        if p_slot < dtypes.int64(n_peer):
            peer_idx = dtypes.uint32(dtypes.uint64(p_slot))
            peerh = xfer_open(ctx, peer_idx)
            ub_flag_write(
                peerh,
                dtypes.uint64(HEAP_BASE + BAR_FLAGS_OFF)
                + my_rank_u64 * dtypes.uint64(4),
                tensor_address(gm_epoch),
            )


def _normalize_grad_views(context, local_grads, epn):
    """归一化 full-dW 组列表并推导 (groups, widths, dtype)。

    契约：每组 ``[2*epn, Wg]`` 或任意尾维 ``[2*epn, ...]``（连续——内部
    展平为 [2*epn, Wg]）；行序与 prefetch 视图一致（前 epn 行 home 种子、
    后 epn 行副本 dW）。组数 2 或 3、各组行宽可不同；返回
    (flat_views, widths, grad_bf16, grad_fp16)。
    """
    local_grads, widths = normalize_group_list(context, local_grads, "reduce_grad")
    grad_dtype = local_grads[0].dtype
    if grad_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(
            f"local_grads 须为 fp32/bf16/fp16（16 位时传输减半、归并期 "
            f"fp32 累加保精度、写回原 dtype），got {grad_dtype}"
        )
    frozen = getattr(context, "grad_dtype", None)
    if frozen is not None and frozen != grad_dtype:
        raise TypeError(
            f"local_grads dtype 与构造定型值不符（定型 {frozen}，本次 {grad_dtype}）"
            "——梯度接收区按定型 dtype 预留（fp32 为 bf16 的 2×），"
            "不可运行期漂移；请以 grad_dtype=... 重建或转换 local_grads"
        )
    flats = []
    for g, t in enumerate(local_grads):
        if t.dtype != grad_dtype:
            raise TypeError("多梯度组的 dtype 必须一致")
        if int(t.shape[0]) != 2 * int(epn):
            raise ValueError(
                f"第 {g} 组行数须为 2*epn={2 * int(epn)}（full-dW 形态：前 epn 行 "
                f"home 种子、后 epn 行副本 dW——与 prefetch 权重视图同序），"
                f"实得 {tuple(t.shape)}"
            )
        if not t.is_contiguous():
            raise ValueError(f"第 {g} 组须连续（展平前置）")
        flats.append(t.reshape(2 * int(epn), widths[g]))
    return flats, widths, grad_dtype == torch.bfloat16, grad_dtype == torch.float16


def _grad_region_params(context):
    """(scale_widths, weight_bytes)——梯度接收区基址的两个布局参数。"""
    scale_widths = getattr(context, "layout_scale_widths", None) or None
    weight_dtype = getattr(context, "weight_dtype", None)
    if weight_dtype is None:
        weight_bytes = 2
    else:
        weight_bytes = int(weight_dtype.itemsize)  # bf16/fp16 占 2 字节，fp8 占 1 字节
    return scale_widths, weight_bytes


class _ReduceSpec(NamedTuple):
    """reduce 直累 kernel 编译键（副本区偏移 copy_off）。"""

    epn: int
    sizes: tuple  # 3 槽行宽
    elem_bytes: tuple  # 3 槽要素字节宽
    ngroups: int
    copy_off: int  # 副本区基址（[epn, Wg]——无 R 维度）
    chunks: tuple  # 逐组归并 chunk
    rank_size: int
    grad_bf16: bool
    grad_fp16: bool  # fp16 梯度（16 位路径与 bf16 同构，dtype 三选一）
    blocks: int
    depth: int = 3  # 流水环槽深（≥2；进键——UB 占用 = depth×chunk）


# --------------------------------------------------------------------------
# reduce 直累 kernel（仅 UbMem）
# --------------------------------------------------------------------------
@kernel
class _UbmemReduceAccKernel:
    """确定性归并 kernel（行单写者 + 固定 peer 升序流水远读 + UB fp32
    累加 + 单次写回）。

    行单写者分区：块 b 归并行 {b, b+blocks, ...}（同行恰一块——单写者，
    无原子）；逐行 etc_inv 直读标量（j≥0 = 该 peer 承载行 i 副本）；固定
    peer 升序流水远读（滞后窗）→ UB fp32 累加 → 单次写回 home 行（一次
    dtype 舍入）。home 种子由调用侧 local_grads 前段即位（seed 入 UB 后
    逐 peer 累加）。逐元素加法序列恒定——跨 run 逐位可重复，bf16/fp16
    舍入从 R-1 次降为 1 次。
    """

    def __init__(
        self,
        rank_size,
        epn,
        sizes,
        elem_bytes,
        ngroups,
        copy_off,
        chunks,
        blocks,
        grad_bf16,
        grad_fp16=False,
        depth=2,
    ):
        self._rank_size = int(rank_size)
        self._epn = int(epn)
        self._sizes = tuple(int(s) for s in sizes)
        self._elem_bytes = tuple(int(b) for b in elem_bytes)
        self._ngroups = int(ngroups)
        self._row_bytes = tuple(s * b for s, b in zip(self._sizes, self._elem_bytes))
        self._chunks = tuple(int(c) for c in chunks)
        self._blocks = int(blocks)
        self._depth = int(depth)
        # 读块划分：块 b 负责行 {b, b+blocks, ...}，每行只有一个写者，
        # 按 peer 升序远读；写 flag 的块从前 n_writer 个里轮转
        n_rem = max(self._rank_size - 1, 0)
        self._n_writer = min(n_rem, self._blocks)
        self._per_writer = (
            (n_rem + self._n_writer - 1) // self._n_writer if self._n_writer else 0
        )
        self._n_read = self._blocks
        self._grad_bf16 = bool(grad_bf16)
        self._grad_fp16 = bool(grad_fp16)
        group_off, gbase = [], int(copy_off)
        for g in range(3):
            group_off.append(gbase)
            gbase += self._epn * self._row_bytes[g]
        self._group_off = tuple(group_off)

    def __call__(
        self,
        k0,
        k1,
        k2,
        gm_etc_inv,
        gm_epoch,
        ctx: StructRef[MoonepCommCtx],
    ):
        rank_size = self._rank_size
        reals = (k0, k1, k2)
        my_rank = dtypes.int64(ctx.rankId)
        my_rank_u64 = dtypes.uint64(ctx.rankId)
        epoch = dtypes.uint32(gm_epoch[dtypes.int64(0)])
        grad_dt = (
            dtypes.float16
            if self._grad_fp16
            else (dtypes.bfloat16 if self._grad_bf16 else dtypes.float32)
        )
        max_chunk = max(self._chunks)
        # grad dtype 走流水环：读第 k+1 块和消费第 k 块重叠，
        # 单槽缓冲会退化成串行
        ub_pipe = RingBuffer(
            MemLoc.UB, grad_dt, make_layout((max_chunk,)), depth=self._depth
        )
        # UB 预算（16 位 @chunk=16K/depth=3）：环 96K + acc fp32 64K +
        # seed 32K + out 32K ≈ 224K < 248K；fp32：环 96K + acc 32K。
        ub_acc = Buffer(MemLoc.UB, dtypes.float32, make_layout((max_chunk,)))
        if const_expr(self._grad_bf16 or self._grad_fp16):
            ub_seed = Buffer(MemLoc.UB, grad_dt, make_layout((max_chunk,)))
            ub_out = Buffer(MemLoc.UB, grad_dt, make_layout((max_chunk,)))
        else:
            # fp32：seed 入/结果出都直通 acc（MTE2/MTE3），无 16 位转换
            ub_seed = ub_acc
            ub_out = ub_acc
        ub_peer_compact = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((self._rank_size,))
        )
        ub_peer_j = Buffer(MemLoc.UB, dtypes.int32, make_layout((self._rank_size,)))
        blk = dtypes.int64(get_block_idx())
        if blk < dtypes.int64(self._n_read):
            # —— ① 头部屏障写者：my flag 写到本块分管的全部 peer
            #     （写者块 ≤ 常驻核数，先写完再等，不会分波互等）
            if blk < dtypes.int64(self._n_writer):
                _barrier_flags_put(
                    ctx,
                    blk,
                    my_rank_u64,
                    rank_size - 1,
                    self._n_writer,
                    self._per_writer,
                    gm_epoch,
                )
            connection = connection_from_address(dtypes.uint64(ctx.connection))
            inv_base = tensor_address(gm_etc_inv)
            # —— 确定性归并：行单写者 + 固定 peer 升序 + UB fp32 累加 +
            #     单次写（逐位可重复）——
            if const_expr(self._rank_size <= 8):
                self._det_merge_compact(
                    connection,
                    reals,
                    ctx,
                    my_rank,
                    epoch,
                    inv_base,
                    ub_pipe,
                    ub_acc,
                    ub_seed,
                    ub_out,
                    ub_peer_compact,
                    ub_peer_j,
                )
            else:
                self._det_merge_dense(
                    connection,
                    reals,
                    ctx,
                    my_rank,
                    epoch,
                    inv_base,
                    ub_pipe,
                    ub_acc,
                    ub_seed,
                    ub_out,
                )

    @jit
    def _det_read_j(self, inv_base, i, p_actual, rank_size):
        """etc_inv[i, p_actual] 的 GM 标量读（load_bypass）。

        返回 int64 的 j（<0 = 无映射）。逐 tile 消费/发射侧各查一次——
        GM 标量读 ~百 ns 级，DCache 行内命中后随查随取。
        """
        j_t = make_tensor(
            make_pointer(
                dtypes.int32,
                inv_base + dtypes.uint64(i * rank_size + p_actual) * dtypes.uint64(4),
                MemLoc.GM,
            ),
            make_layout((1,)),
        )
        return dtypes.int64(cbd.scalar.load_bypass(j_t.ptr(0)))

    @jit
    def _det_build_peer_row(
        self,
        inv_base,
        i,
        my_rank,
        rank_size,
        ub_peer_compact,
        ub_peer_j,
    ):
        """Compact this row's contributing peers while preserving rank order."""
        count = dtypes.int32(0)
        for p_actual in cannbotdsl.range(
            dtypes.int64(0), dtypes.int64(rank_size), dtypes.int64(1)
        ):
            if p_actual != my_rank:
                j = self._det_read_j(inv_base, i, p_actual, rank_size)
                if j >= dtypes.int64(0):
                    if p_actual < my_rank:
                        ub_peer_compact[dtypes.int64(count)] = dtypes.int32(p_actual)
                    else:
                        ub_peer_compact[dtypes.int64(count)] = dtypes.int32(
                            p_actual - dtypes.int64(1)
                        )
                    ub_peer_j[dtypes.int64(count)] = dtypes.int32(j)
                    count = count + dtypes.int32(1)
        return count

    @jit
    def _det_wait_peer_flags(self, ctx, my_rank, epoch):
        """等全部 peer 的入槽 flag（本块逐 peer 远读：对端 kernel 启动即
        其流序在前的 host 副本区写入已完成——保护上轮远读与本轮覆写
        不重叠）。"""
        rank_size = int(self._rank_size)
        for p in cannbotdsl.range(
            dtypes.int64(0), dtypes.int64(rank_size), dtypes.int64(1)
        ):
            if p != my_rank:
                spin_u32_ge(
                    ctx,
                    dtypes.uint64(HEAP_BASE + BAR_FLAGS_OFF)
                    + dtypes.uint64(p) * dtypes.uint64(4),
                    epoch,
                )

    @jit
    def _det_seed_acc(self, reals, g, tile_idx, ub_acc, ub_seed):
        """seed 入 fp32 累加器（fp32 直通 acc；16 位经 seed 缓冲 + vf
        展开成 fp32）。"""
        chunk = int(self._chunks[g])
        home_tile = tile_slice(reals[g], (chunk,), (tile_idx,))
        if const_expr(not (self._grad_bf16 or self._grad_fp16)):
            mem_copy(tile_slice(ub_acc, (chunk,), (0,)), home_tile)
        else:
            mem_copy(tile_slice(ub_seed, (chunk,), (0,)), home_tile)
            with vf(mode="simd"):
                full = full_mask()
                for off in range_constexpr(0, chunk, _VL):
                    vstore(
                        ub_acc,
                        off,
                        vcast(
                            vload_unpack(
                                ub_seed,
                                off,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=full,
                        ),
                        full,
                    )

    @jit
    def _det_consume(self, t, g, ub_peer_j, ub_pipe, ub_acc):
        """滞后窗消费侧：peer tc = t-depth 的贡献入 fp32 累加器
        （先消费后发射——环在飞数恒 ≤ depth）。"""
        depth = int(self._depth)
        chunk = int(self._chunks[g])
        tc = t - dtypes.int64(depth)
        if tc >= dtypes.int64(0):
            j_c = dtypes.int64(ub_peer_j[tc])
            if j_c >= dtypes.int64(0):
                slot = ub_pipe.next_read()
                # vf: acc += slot（fp32 直加；16 位 unpack+cast 后加）
                with vf(mode="simd"):
                    full = full_mask()
                    if const_expr(not (self._grad_bf16 or self._grad_fp16)):
                        for off in range_constexpr(0, chunk, _VL):
                            acc_v = vadd(
                                vload(ub_acc, off), vload(slot, off), mask=full
                            )
                            vstore(ub_acc, off, acc_v, full)
                    else:
                        for off in range_constexpr(0, chunk, _VL):
                            contrib = vcast(
                                vload_unpack(
                                    slot, off, unpack_mode=UnpackMode.B16_TO_B32
                                ),
                                dtypes.float32,
                                mask=full,
                            )
                            vstore(
                                ub_acc,
                                off,
                                vadd(vload(ub_acc, off), contrib, mask=full),
                                full,
                            )

    @jit
    def _det_issue(
        self,
        connection,
        i,
        t,
        c,
        g,
        n_valid,
        ub_peer_compact,
        ub_peer_j,
        ub_pipe,
    ):
        """滞后窗发射侧：peer t 的副本 chunk 映射远读入环下一槽。

        get_peer 取 compact 序号（连接表 0..R-2）；etc_inv 列取 actual
        rank（pa_i）——两者域不同，混用则 get_peer 越界 → mapped 野值 →
        远读地址无效（MTE 507035）。
        """
        if t < n_valid:
            j_i = dtypes.int64(ub_peer_j[t])
            if j_i >= dtypes.int64(0):
                chunk = int(self._chunks[g])
                elem = 2 if (self._grad_bf16 or self._grad_fp16) else 4
                grad_dt = (
                    dtypes.float16
                    if self._grad_fp16
                    else (dtypes.bfloat16 if self._grad_bf16 else dtypes.float32)
                )
                slot_w = ub_pipe.next_write()
                peer_compact = dtypes.int64(ub_peer_compact[t])
                peer = connection.get_peer(dtypes.uint32(dtypes.uint64(peer_compact)))
                mapped = dtypes.uint64(peer.get_mapped_addr(0))
                j64 = dtypes.int64(j_i)
                row_base = dtypes.uint64(
                    HEAP_BASE + int(self._group_off[g]) + j64 * int(self._row_bytes[g])
                )
                remote_src = make_tensor(
                    make_pointer(
                        grad_dt,
                        dtypes.int64(
                            mapped + row_base + dtypes.uint64(c * chunk * elem)
                        ),
                        MemLoc.GM,
                    ),
                    make_layout((chunk,)),
                )
                mem_copy(tile_slice(slot_w, (chunk,), (0,)), remote_src)

    @jit
    def _det_writeback(self, reals, g, tile_idx, ub_acc, ub_out):
        """单次写回 home chunk（终轮一次 dtype 舍入——逐位可重复的
        关键）。"""
        chunk = int(self._chunks[g])
        home_tile = tile_slice(reals[g], (chunk,), (tile_idx,))
        if const_expr(not (self._grad_bf16 or self._grad_fp16)):
            mem_copy(home_tile, tile_slice(ub_acc, (chunk,), (0,)))
        else:
            grad_dt = (
                dtypes.float16
                if self._grad_fp16
                else (dtypes.bfloat16 if self._grad_bf16 else dtypes.float32)
            )
            with vf(mode="simd"):
                full = full_mask()
                for off in range_constexpr(0, chunk, _VL):
                    vstore_pack(
                        ub_out,
                        off,
                        vcast(
                            vload(ub_acc, off),
                            grad_dt,
                            mask=full,
                            rounding=RoundingMode.RN,
                        ),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            mem_copy(home_tile, tile_slice(ub_out, (chunk,), (0,)))

    @jit
    def _det_merge_compact(
        self,
        connection,
        reals,
        ctx,
        my_rank,
        epoch,
        inv_base,
        ub_pipe,
        ub_acc,
        ub_seed,
        ub_out,
        ub_peer_compact,
        ub_peer_j,
    ):
        """确定性归并主循环：行单写者 + 固定 peer 升序 + 单次写。

        块 b 归并行 {b, b+blocks, ...}（同行恰一块——单写者，无原子）。
        每 (行, 组, chunk) tile：seed（home chunk，本地快读）入 fp32
        累加器 → peer 升序流水累加（读块内环深滞后窗，MTE2 读 ∥ vf 加）→
        单次写回 home chunk（一次 dtype 舍入）。

        逐元素加法序列恒为 seed、peer 升序贡献、终轮一次舍入——与 host
        参照（fp32 串行累加后一次 cast）逐位一致，跨 run 逐位可重复。

        每行先压缩有效 peer 升序列表；tile 内 t 遍历
        [0, n_valid+depth)。该列表保持实际 rank 升序，因此逐元素加法顺序
        与全 peer 扫描完全一致。
        """
        rank_size = int(self._rank_size)
        epn_rt = dtypes.int64(self._epn)
        blocks_rt = dtypes.int64(self._blocks)
        # ① 等全部 peer 的入槽 flag
        self._det_wait_peer_flags(ctx, my_rank, epoch)
        # ② 行单写者分区：i = blk, blk+blocks, ...
        blk = dtypes.int64(get_block_idx())
        for i in cannbotdsl.range(blk, epn_rt, blocks_rt):
            n_valid = self._det_build_peer_row(
                inv_base,
                i,
                my_rank,
                rank_size,
                ub_peer_compact,
                ub_peer_j,
            )
            if n_valid > dtypes.int32(0):
                for g in range_constexpr(_MAX_GRAD_GROUPS):
                    if g < int(self._ngroups):
                        chunk = int(self._chunks[g])
                        n_c = int(self._sizes[g]) // chunk
                        for c in cannbotdsl.range(
                            dtypes.int64(0), dtypes.int64(n_c), dtypes.int64(1)
                        ):
                            # —— tile (i, g, c)：seed → peer 升序流水 → 单次写 ——
                            tile_idx = i * dtypes.int64(n_c) + c
                            self._det_seed_acc(reals, g, tile_idx, ub_acc, ub_seed)
                            # —— peer 升序 issue/consume（滞后窗 depth；先消费
                            #    后发射——环在飞数恒 ≤ depth）——
                            n_iter = dtypes.int64(n_valid) + dtypes.int64(
                                int(self._depth)
                            )
                            for t in cannbotdsl.range(
                                dtypes.int64(0), n_iter, dtypes.int64(1)
                            ):
                                self._det_consume(
                                    t,
                                    g,
                                    ub_peer_j,
                                    ub_pipe,
                                    ub_acc,
                                )
                                self._det_issue(
                                    connection,
                                    i,
                                    t,
                                    c,
                                    g,
                                    n_valid,
                                    ub_peer_compact,
                                    ub_peer_j,
                                    ub_pipe,
                                )
                            # —— 单次写回（终轮一次舍入）——
                            self._det_writeback(reals, g, tile_idx, ub_acc, ub_out)

    @jit
    def _det_row_has_remote(self, inv_base, i, my_rank, rank_size):
        """Return 1 when any peer contributes to local home row ``i``."""
        found = dtypes.int32(0)
        for p_actual in cannbotdsl.range(
            dtypes.int64(0), dtypes.int64(rank_size), dtypes.int64(1)
        ):
            if p_actual != my_rank:
                j = self._det_read_j(inv_base, i, p_actual, rank_size)
                if j >= dtypes.int64(0):
                    found = dtypes.int32(1)
        return found

    @jit
    def _det_consume_dense(
        self, t, i, g, my_rank, rank_size, inv_base, ub_pipe, ub_acc
    ):
        """Consume one dense peer slot after the pipeline depth."""
        depth = int(self._depth)
        chunk = int(self._chunks[g])
        tc = t - dtypes.int64(depth)
        if tc >= dtypes.int64(0):
            pa_c = tc
            if pa_c >= my_rank:
                pa_c = pa_c + dtypes.int64(1)
            j_c = self._det_read_j(inv_base, i, pa_c, rank_size)
            if j_c >= dtypes.int64(0):
                slot = ub_pipe.next_read()
                with vf(mode="simd"):
                    full = full_mask()
                    if const_expr(not (self._grad_bf16 or self._grad_fp16)):
                        for off in range_constexpr(0, chunk, _VL):
                            acc_v = vadd(
                                vload(ub_acc, off), vload(slot, off), mask=full
                            )
                            vstore(ub_acc, off, acc_v, full)
                    else:
                        for off in range_constexpr(0, chunk, _VL):
                            contrib = vcast(
                                vload_unpack(
                                    slot, off, unpack_mode=UnpackMode.B16_TO_B32
                                ),
                                dtypes.float32,
                                mask=full,
                            )
                            vstore(
                                ub_acc,
                                off,
                                vadd(vload(ub_acc, off), contrib, mask=full),
                                full,
                            )

    @jit
    def _det_issue_dense(
        self, connection, inv_base, i, t, c, g, my_rank, rank_size, ub_pipe
    ):
        """Issue one dense peer slot into the read pipeline."""
        if t < dtypes.int64(rank_size - 1):
            pa_i = t
            if pa_i >= my_rank:
                pa_i = pa_i + dtypes.int64(1)
            j_i = self._det_read_j(inv_base, i, pa_i, rank_size)
            if j_i >= dtypes.int64(0):
                chunk = int(self._chunks[g])
                elem = 2 if (self._grad_bf16 or self._grad_fp16) else 4
                grad_dt = (
                    dtypes.float16
                    if self._grad_fp16
                    else (dtypes.bfloat16 if self._grad_bf16 else dtypes.float32)
                )
                slot_w = ub_pipe.next_write()
                peer = connection.get_peer(dtypes.uint32(dtypes.uint64(t)))
                mapped = dtypes.uint64(peer.get_mapped_addr(0))
                j64 = dtypes.int64(j_i)
                row_base = dtypes.uint64(
                    HEAP_BASE + int(self._group_off[g]) + j64 * int(self._row_bytes[g])
                )
                remote_src = make_tensor(
                    make_pointer(
                        grad_dt,
                        dtypes.int64(
                            mapped + row_base + dtypes.uint64(c * chunk * elem)
                        ),
                        MemLoc.GM,
                    ),
                    make_layout((chunk,)),
                )
                mem_copy(tile_slice(slot_w, (chunk,), (0,)), remote_src)

    @jit
    def _det_merge_dense(
        self,
        connection,
        reals,
        ctx,
        my_rank,
        epoch,
        inv_base,
        ub_pipe,
        ub_acc,
        ub_seed,
        ub_out,
    ):
        """Deterministic dense-peer merge used above eight ranks."""
        rank_size = int(self._rank_size)
        epn_rt = dtypes.int64(self._epn)
        blocks_rt = dtypes.int64(self._blocks)
        self._det_wait_peer_flags(ctx, my_rank, epoch)
        blk = dtypes.int64(get_block_idx())
        for i in cannbotdsl.range(blk, epn_rt, blocks_rt):
            has_remote = self._det_row_has_remote(inv_base, i, my_rank, rank_size)
            if has_remote > dtypes.int32(0):
                for g in range_constexpr(_MAX_GRAD_GROUPS):
                    if g < int(self._ngroups):
                        chunk = int(self._chunks[g])
                        n_c = int(self._sizes[g]) // chunk
                        for c in cannbotdsl.range(
                            dtypes.int64(0), dtypes.int64(n_c), dtypes.int64(1)
                        ):
                            tile_idx = i * dtypes.int64(n_c) + c
                            self._det_seed_acc(reals, g, tile_idx, ub_acc, ub_seed)
                            n_iter = dtypes.int64(rank_size - 1) + dtypes.int64(
                                int(self._depth)
                            )
                            for t in cannbotdsl.range(
                                dtypes.int64(0), n_iter, dtypes.int64(1)
                            ):
                                self._det_consume_dense(
                                    t,
                                    i,
                                    g,
                                    my_rank,
                                    rank_size,
                                    inv_base,
                                    ub_pipe,
                                    ub_acc,
                                )
                                self._det_issue_dense(
                                    connection,
                                    inv_base,
                                    i,
                                    t,
                                    c,
                                    g,
                                    my_rank,
                                    rank_size,
                                    ub_pipe,
                                )
                            self._det_writeback(reals, g, tile_idx, ub_acc, ub_out)


# --------------------------------------------------------------------------
# reduce host 入口与 program 缓存
# --------------------------------------------------------------------------
_UBMEM_PROGRAMS = {}
_ACC_LOCK = threading.Lock()  # program 缓存写入互斥锁
# 编译期常量放在模块级全局，锁内赋值；声明和赋值要一起改，
# 漏掉 global 就会按默认值 trace
_ACC_R = 4
_ACC_EPN = 1
_ACC_SIZES = (1,)
_ACC_ELEM = (2,)
_ACC_NGROUPS = 2
_ACC_COPY_OFF = 0
_ACC_CHUNKS = (1,)
_ACC_BF16 = True
_ACC_FP16 = False  # fp16 梯度（16 位路径同 bf16——dtype 三选一）
_ACC_BLOCKS = 8
_ACC_DEPTH = 3  # 流水环槽深（UB = depth×max_chunk；2=issue#16 基线）


@host
def _reduce_ubmem_host(
    k0,
    k1,
    k2,
    gm_etc_inv,
    gm_epoch,
    ctx: StructRef[MoonepCommCtx],
):
    """reduce host entry（仅 UbMem）：单 kernel——行单写者确定性归并
    （固定 peer 升序远读 + UB fp32 累加 + 单次写回）。"""
    _UbmemReduceAccKernel(
        _ACC_R,
        _ACC_EPN,
        _ACC_SIZES,
        _ACC_ELEM,
        _ACC_NGROUPS,
        _ACC_COPY_OFF,
        _ACC_CHUNKS,
        _ACC_BLOCKS,
        _ACC_BF16,
        _ACC_FP16,
        depth=_ACC_DEPTH,
    )[_ACC_BLOCKS](k0, k1, k2, gm_etc_inv, gm_epoch, ctx)


def _get_acc_program(spec):
    """直累 program 缓存（_ReduceSpec 键控——锁内原子烘焙）。"""
    with _ACC_LOCK:
        prog = _UBMEM_PROGRAMS.get(spec)
        if prog is not None:
            return prog
        global _ACC_R, _ACC_EPN, _ACC_SIZES, _ACC_ELEM
        global _ACC_NGROUPS, _ACC_COPY_OFF, _ACC_CHUNKS
        global _ACC_BF16, _ACC_BLOCKS, _ACC_FP16
        global _ACC_DEPTH
        _ACC_R = spec.rank_size
        _ACC_EPN = spec.epn
        _ACC_SIZES = spec.sizes
        _ACC_ELEM = spec.elem_bytes
        _ACC_NGROUPS = spec.ngroups
        _ACC_COPY_OFF = spec.copy_off
        _ACC_CHUNKS = spec.chunks
        _ACC_BF16 = spec.grad_bf16
        _ACC_FP16 = spec.grad_fp16
        _ACC_BLOCKS = spec.blocks
        _ACC_DEPTH = spec.depth
        i32 = dtypes.int32
        dt = (
            dtypes.float16
            if spec.grad_fp16
            else (dtypes.bfloat16 if spec.grad_bf16 else dtypes.float32)
        )
        specs = [TensorSpec((2 * spec.epn * s,), dt) for s in spec.sizes]
        specs.append(TensorSpec((spec.epn, spec.rank_size), i32))
        specs.append(TensorSpec((1,), i32))  # epoch 标志
        specs.append(StructRef[MoonepCommCtx])
        prog = cannbotdsl.compile(_reduce_ubmem_host, *specs)
        _UBMEM_PROGRAMS[spec] = prog
        return prog


def _copy_grads_to_ws(
    context, experts_to_copy, local_grads, epn, widths, grad_bf16, grad_fp16, copy_off
):
    """发送侧：副本行 local_grads[g][epn:] 拷入 workspace 副本区。

    副本区 [epn, Wg] × 组（grad_slot_off 后，紧随权重发布区——
    每 rank 只放自己的副本行，owner 远读。布局与 push 接收区共享
    基址公式但无 R 维度——内存恒定 [epn, Wg]×组。

    dst 视图 1 槽缓存（同 shape/dtype/copy_off 复用——每拍
    narrow/view/view × 组的 aten 派发是 host 段 ~70 ops/拍的
    削减项之一；源侧 local_grads 每拍新张量不可缓存）。
    """
    dt = (
        torch.float16 if grad_fp16 else (torch.bfloat16 if grad_bf16 else torch.float32)
    )
    elem = 2 if (grad_bf16 or grad_fp16) else 4
    ws = context.workspace
    key = (
        tuple(int(w) for w in widths),
        bool(grad_bf16),
        bool(grad_fp16),
        int(copy_off),
        int(epn),
    )
    cached = getattr(context, "_pull_ws_copy_views", None)
    if cached is None or cached[0] != key:
        views = []
        base = HEAP_BASE + int(copy_off)
        for w in widths:
            region = epn * w * elem
            views.append(ws.narrow(0, base, region).view(dt).view(epn, w))
            base += region
        cached = (key, views)
        context._pull_ws_copy_views = cached

    index_cache = getattr(context, "_pull_ws_copy_indices", None)
    index_key = (
        int(context.reduce_copy_plan_token),
        int(context.rank),
        int(epn),
    )
    if index_cache is None or index_cache[0] != index_key:
        active_rows = (
            experts_to_copy[context.rank].ge(0).nonzero(as_tuple=False).flatten()
        )
        index_cache = (index_key, active_rows, experts_to_copy)
        context._pull_ws_copy_indices = index_cache
    for g, w in enumerate(widths):
        source = local_grads[g][epn:]
        cached[1][g].index_copy_(
            0, index_cache[1], source.index_select(0, index_cache[1])
        )


def reduce_grad(
    context,
    experts_to_copy,
    local_grads,
    epn,
    *,
    slots=None,
    async_finish=False,
):
    """owner 直累 reduce_grad：副本梯度归并回 home（full-dW 直入形态）。

    ``local_grads`` 为 2 或 3 组 ``[2*epn, Wg]`` 张量——GMM 对全量
    （home+副本）权重视图反传的原生形态，行序与 prefetch 视图一致（前
    epn 行 home 种子、后 epn 行副本 dW）。组数/各组行宽从张量解析；
    dtype 须与构造定型值一致（交织型传 ``[2*epn, 2W] + [2*epn, W]``）。

    梯度搬运（owner 直累协议）：发送侧 host 拷贝副本行入 workspace 副本区
    ``[epn, Wg]``（恒定 ~19.3MB，不随 R 增长）；owner 侧单 kernel 映射
    远读各 peer 副本区 → UB 流水环 → 拷贝引擎原子直累 home 段（原位）。

    Args:
        local_grads: list/tuple，每组 ``[2*epn, Wg]`` 连续张量（fp32/bf16，
                     全组同 dtype，与构造 grad_dtype 定型值一致）。
        async_finish: False（缺省）阻塞到全组归并完成（返回 event=None）；
                     True 返回发起流 torch.npu.Event（wait_event 汇合后读
                     归并结果）。
        slots: 窗口池层数（None = context.window_slots 定型值；显式给出时
            须与定型值一致，否则 ValueError）。

    Returns:
        变长返回：``async_finish=False``（缺省）只返回
        ``home_views`` 单组视图元组；``async_finish=True`` 返回
        ``(home_views, event)``。``home_views`` 为各组归并结果的
        ``[epn, Wg]`` 视图（与输入张量同存储——原位归并的直接读柄，归并
        完成后即全量 dW）；``event`` 同步模式为 None、异步为发起流事件。
    """
    # 归一化 + 行数/宽度/dtype/调用序守卫（组数与行宽从张量解析）。
    local_grads, widths, grad_bf16, grad_fp16 = _normalize_grad_views(
        context, local_grads, epn
    )

    elem_bytes = 2 if (grad_bf16 or grad_fp16) else 4
    # 跨流护栏：本算子与前序通信算子共用 epoch 握手机制，须按发起序
    # 执行（同流为快路径零开销）
    sync_transport_stream(context)

    if context.rank_size == 1:
        # R=1 时没有副本可归并，home 段就是全量 dW，直接返回；
        # 没有集合操作，事件立刻可用
        home_views = tuple(t[:epn] for t in local_grads)
        if async_finish:
            return home_views, None
        return home_views

    # —— 窗口池约定校验（与 prefetch 同族；归并落点与槽位无关）——
    if slots is None:
        slots = context.window_slots or 1
    elif context.window_slots is not None and int(slots) != int(context.window_slots):
        raise ValueError(
            f"reduce_grad slots={slots} 与窗口池定型层数 {context.window_slots} 不一致"
        )

    # —— 副本区布局（紧随权重发布区；[epn, Wg] × 组——pull 恒定，不随 R 增长）——
    scale_widths, weight_bytes = _grad_region_params(context)
    copy_off = grad_slot_off(
        epn,
        widths=widths,
        scale_widths=scale_widths,
        weight_bytes=weight_bytes,
        slots=slots,
    )
    rank_size = context.rank_size
    copy_bytes = epn * sum(widths) * elem_bytes
    check_heap_capacity(
        context,
        HEAP_BASE + copy_off + copy_bytes + LAYOUT_TAIL_BYTES,
        "pull 梯度副本区",
    )
    # —— 槽视图一致性守卫（pull_slots_used 契约沿用）——
    used = context.pull_slots_used
    if used is None:
        context.pull_slots_used = slots
    elif used != slots:
        raise ValueError(
            f"梯度槽视图 slots={slots} 与既有 slots={used}"
            " 不一致——副本区基址会错位（请保持全链 slots 一致）"
        )

    # owner 直累：host 先把副本行拷进 workspace，kernel 只读 etc
    # 映射到的行，没有陈旧行掺和
    _copy_grads_to_ws(
        context,
        experts_to_copy,
        local_grads,
        epn,
        widths,
        grad_bf16,
        grad_fp16,
        copy_off,
    )
    # 归并 kernel：owner 远读 peer 副本 + vf 流式归并
    gm_epoch = next_barrier_epoch(context)
    # 块数封顶常驻核数（950PR 64 块）——R>64 时不随 R 增长：屏障写者按
    # peer 轮转，读块按行条带划分，避免写者被分波调度推到第二波互等。
    blocks = resident_blocks(_HEADER_BARRIER_CAP)
    acc_depth = _acc_depth(rank_size)
    chunk_limit = _acc_chunk_limit(acc_depth, grad_bf16 or grad_fp16)
    chunks = tuple(pick_weight_chunk(w, chunk_limit) for w in widths)
    # 行宽须为 _VL=64 倍数（归并 tile 步进=_VL——非倍数时尾 tile 越界 UB；
    # Buffer 构造期已查，裸 API 路径在此 fail-fast）
    for w in widths:
        if w % _VL:
            raise ValueError(f"Wg={w} 须为 {_VL} 的倍数（归并 _VL 尾块越界防护）")
    ngroups = len(local_grads)
    # 3 槽：真实组前缀 + pad 尾置（缺省槽复用末组——kernel 不读取）
    sizes = tuple(widths) + (widths[-1],) * (_MAX_GRAD_GROUPS - ngroups)
    # 环深与 chunk 上限同源计算；depth 进入编译键。确定性归并保持
    # peer 升序、UB fp32 累加与单次写回。
    spec = _ReduceSpec(
        epn=epn,
        sizes=sizes,
        elem_bytes=(elem_bytes,) * _MAX_GRAD_GROUPS,
        ngroups=ngroups,
        copy_off=copy_off,
        chunks=chunks + (chunks[-1],) * (_MAX_GRAD_GROUPS - ngroups),
        rank_size=rank_size,
        grad_bf16=grad_bf16,
        grad_fp16=grad_fp16,
        blocks=blocks,
        depth=acc_depth,
    )
    # 3 槽实参（缺省槽复用末组张量——kernel 不读取）
    reals3 = tuple(local_grads) + (local_grads[-1],) * (_MAX_GRAD_GROUPS - ngroups)
    # etc 逆映射 [epn, R]：etc_inv[i, p] = j 表示 peer p 的副本区承载
    # 本 rank 行 i 的槽位（-1 表示没有）；device 侧向量化算一次；按 etc
    # 张量版本缓存，缓存里攥着 etc 防止地址复用误命中
    inv_cache = getattr(context, "_pull_etc_inv_cache", None)
    inv_key = (id(experts_to_copy), experts_to_copy._version, epn)
    if inv_cache is not None and inv_cache[0] == inv_key:
        etc_inv = inv_cache[1]
    else:
        dev = experts_to_copy.device
        targets = context.rank * epn + torch.arange(epn, device=dev, dtype=torch.int32)
        eq = experts_to_copy.unsqueeze(-1) == targets  # [R, epn(j), epn(i)]
        has = eq.any(dim=1)  # [R, epn(i)]
        # argmax 恒返 int64——须显式转回 int32（TensorSpec 声明 int32，
        # int64 张量按 4B 步长解读 = 静默错值 + 野偏移远读）
        j_idx = eq.to(torch.int32).argmax(dim=1).to(torch.int32)  # [R, epn(i)]
        etc_inv = (
            torch.where(has, j_idx, torch.tensor(-1, dtype=torch.int32, device=dev))
            .t()
            .contiguous()
        )  # [epn, R]
        context._pull_etc_inv_cache = (
            inv_key,
            etc_inv,
            experts_to_copy,
        )
    # 确定性归并路径（UbMem 映射读）
    _get_acc_program(spec)(
        *(t.reshape(-1) for t in reals3),
        etc_inv,
        gm_epoch,
        context.ctx_tensor,
    )
    # 尾部屏障：全组 pull 完才允许下一轮 host 拷贝副本区，否则对端
    # 可能读到新旧混合行
    sym_barrier(context, sync=False)

    # 变长返回契约：sync 只返回 home_views 单值；async 二元组。
    event = finish_op(context, async_finish)
    home_views = tuple(t[:epn] for t in local_grads)
    if async_finish:
        return home_views, event
    return home_views
