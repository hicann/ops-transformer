# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_plan：MoonEP 的均衡规划算子。

吃进本 rank 的 tokens_per_expert[E] 和 topk[S,K]，吐出 MoonepPlan 五件套：

    dst_n[N] / src_n[N]         零占位（vtopk-only 快路径不计算）
    cu_seqlens[2*epn]           本 rank 接收区各槽位累计行数
    experts_to_copy[R, epn]     各 rank 的副本清单（-1 填充，单源）
    vtopk[S,K]                  虚拟专家 id（扩展空间 2E）

约束：E <= 1024、R <= MAX_R（128）、E % R == 0；N = S*K 任意规模。

小 EP 组把 preparation 和 token 落位压进两个 launch，省一点开销；大 EP 组
拆成四段各自 launch，让 R16 的运行期资源互不踩脚：

    阶段 0  交换计数
    阶段 1  token 落位（冷专家 vgather；热专家按段二分定位）
    阶段 2  副本清单（Tables）与段查找表（Pfx）
    阶段 3  cap 均衡贪心 + 冷/热专家 vid
"""

import threading
from typing import NamedTuple

import cannbotdsl
import torch
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    full_mask,
    vadd,
    varange,
    vbitwise_or,
    vdups,
    veqs,
    vgather,
    vhistogram_frequency,
    vload,
    vload_deinterleave,
    vlts,
    vreinterpret,
    vsqueeze_and_storeunalign,
    vsqueeze_and_storeunalign_finalize,
    vsqueeze_and_storeunalign_init,
    vstore,
    vstorealign_squeeze_status,
    vunpack,
)
from cannbotdsl.ops.sync import global_sync_all, vec_sync_all
from cannbotdsl.tensor import MemLoc, make_layout, make_pointer, make_tensor, tile_slice
from cannbotdsl.types.datastruct import StructRef
from moonep_context import (
    DEM_ALL_OFF,
    HEAP_BASE,
    MAX_E,
    MAX_R,
    check_pending_overflow,
    next_barrier_epoch,
    record_transport_event,
    resident_blocks,
    sync_transport_stream,
)
from moonep_layout import NOTIFY_STRIDE
from moonep_comm import (
    MoonepCommCtx,
    make_xfer_rb,
    spin_u32_ge,
    tensor_address,
    ub_flag_write,
    ub_write,
    xfer_open,
)

try:
    from cannbotdsl.ops.reg import vstore_unalign_begin
except ImportError:
    vstore_unalign_begin = None


_HAS_LEGACY_SQUEEZE_STORE = vstore_unalign_begin is not None

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------
_MAX_PLAN_BLOCKS = 64  # 单次启动 block 上限，实际取 min(AIV 核数, cap)
# 编译期烘焙的块数上限（_get_plan_program 锁内按实际机型核数覆写）
_PLAN_BLOCK_LIMIT = _MAX_PLAN_BLOCKS
_CHUNK_MIN = 128  # chunk 下限（token/块），再小则每块 O(E) 重建占比过高
_VF_SAFE_CHUNK = 1024  # chunk 上限（vf 栈安全——见 _plan_chunking docstring）
_PLAN_UB_BUDGET = 192 * 1024  # Greedy 的 dem+order 两个 R*E 表的 UB 预算

# 向量粒度（2048-bit 向量）
_I32_LANES = 64  # int32 lanes（Assign 冷专家路径步长）
_HISTOGRAM_U8_LANES = 256  # uint8 lanes（Count 向量路径步长）
_HISTOGRAM_BINS_PER_VL = 128  # uint16 直方图 bins/向量
_HISTOGRAM_MAX_CHUNK = 1 << 16
_HISTOGRAM_SENTINEL = -(1 << 24)  # b3=0xFF 的 int32 表示
_WARP_SIZE = 32
_PLAN_INIT_THREADS = _WARP_SIZE

# 贪心的极值哨兵
_GREEDY_NEG_INF = -(1 << 30)
_GREEDY_POS_INF = 1 << 30

# 段序 order：0=非服务 rank，1=home，>=2=副本（按创建顺序）
_ORDER_UNSERVED = 0
_ORDER_HOME = 1
_ORDER_FIRST_REPLICA = 2
_VID_HOT = -1  # 热专家标记（需走 pcnt/ord 段查找）

_INT32_BYTES = 4  # int32 字节宽（计数换算与字节偏移共用）
_SLOTS_PER_RANK = 2  # 每 rank 两组槽位：本地专家 + 预取副本
_WS_ALIGN = 16  # workspace 每段按 16 元素（64B）对齐


# --------------------------------------------------------------------------
# 纯整数原语
# --------------------------------------------------------------------------
def ceil_div(x, y):
    """向上取整除法。"""
    return (x + y - 1) // y


def align_up(x, alignment):
    """向上对齐到 alignment 的整数倍。"""
    return ceil_div(x, alignment) * alignment


def ceil_pow2(x):
    """不小于 x 的最小 2 的幂。"""
    return 1 << max(x - 1, 0).bit_length()


# --------------------------------------------------------------------------
# 运行期状态
# --------------------------------------------------------------------------
_RANK_SIZE = 1  # 组内 R（编译锁内写入，参与编译缓存键）


# --------------------------------------------------------------------------
# plan 单 kernel：跨卡交换、规划与 token 落位
# --------------------------------------------------------------------------
@kernel
class _PlanKernel:
    """plan kernel：跨卡交换 → 贪心均衡 → 表生成 → token 落位.

    每 launch 处理一个阶段；需要全阶段会合时使用 global_sync_all()。
    """

    def __init__(
        self,
        rank_size,
        num_experts,
        num_rows,
        num_chunks,
        token_blocks,
        pfx_blocks,
        phase,
    ):
        """绑定编译期形状参数。"""
        self._phase = int(phase)
        self._rank_size = int(rank_size)
        self._num_experts = int(num_experts)
        self._epn = self._num_experts // self._rank_size
        self._chunk = int(num_rows) // int(num_chunks)
        self._num_chunks = int(num_chunks)
        self._token_blocks = int(token_blocks)
        self._pfx_blocks = int(pfx_blocks)

    def __call__(
        self,
        gm_tpe,
        gm_flag,
        gm_ws,
        gm_topk,
        gm_io,
        ctx: StructRef[MoonepCommCtx],
    ):
        """按 launch phase 分派交换、规划、表生成与落位。"""
        rank_size, num_experts = self._rank_size, self._num_experts
        epn = self._epn
        views = _bind_plan_views(gm_ws, rank_size, num_experts, self._num_chunks)
        # 三份输出共用一块 io：cu [2epn] | copy_list [R,epn] | vtopk [N]
        cu_elems = _SLOTS_PER_RANK * epn
        etc_end = cu_elems + rank_size * epn
        gm_cu = gm_io[0:cu_elems,]
        gm_copy_list = gm_io[cu_elems:etc_end,].view(rank_size, epn)
        vtopk_end = etc_end + self._chunk * self._num_chunks
        gm_vtopk = gm_io[etc_end:vtopk_end,]
        block_id = get_block_idx()
        ub_demand = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((rank_size, num_experts))
        )

        if const_expr(self._phase == 0 or self._phase == 3) and block_id < dtypes.int64(
            rank_size
        ):
            self._exchange_put_peer(ctx, gm_tpe, gm_flag, block_id)
        if const_expr(self._phase == 0 or self._phase == 3) and block_id == (
            dtypes.int64(0)
        ):
            self._exchange(ctx, gm_tpe, gm_flag, views.dem, ub_demand)
        if const_expr(self._phase == 0 or self._phase == 3) and block_id < dtypes.int64(
            self._token_blocks
        ):
            self._count(block_id, gm_topk, views.counts)

        if const_expr(self._phase == 4) and block_id == dtypes.int64(0):
            self._exchange(ctx, gm_tpe, gm_flag, views.dem, ub_demand)
        if const_expr(self._phase == 0 or self._phase == 4) and block_id == (
            dtypes.int64(0)
        ):
            self._greedy_all(ctx, ub_demand, views, gm_cu)

        if const_expr(self._phase == 0):
            global_sync_all()

        # ---- 阶段 1：Tables / Pfx（只依赖阶段 0 的 order/counts） ----
        num_blocks = get_block_num()
        if const_expr(self._phase == 0 or self._phase == 2):
            self._tables_loop(block_id, num_blocks, views, gm_copy_list)
        if const_expr(self._phase == 0 or self._phase == 2) and block_id < dtypes.int64(
            self._pfx_blocks
        ):
            self._pfx(ctx, block_id, views)

        if const_expr(self._phase == 0 or self._phase == 2):
            global_sync_all()

        # ---- 阶段 2：token 落位 ----
        if const_expr(self._phase == 1) and block_id < dtypes.int64(self._token_blocks):
            self._assign(block_id, gm_topk, gm_vtopk, views)

    # ---- 阶段 0：Exchange（单 block 通信） ----
    @jit
    def _exchange(self, ctx, gm_tpe, gm_flag, gm_dem, ub_demand):
        """块 0：本地 tpe 行落位 + 自旋收齐全部 peer，汇总为 dem 矩阵。"""
        world_size, num_experts = self._rank_size, self._num_experts
        my_rank_u64 = dtypes.uint64(ctx.rankId)  # 运行时取，rank 不进编译键
        my_rank = dtypes.int32(ctx.rankId)
        epoch = dtypes.uint32(gm_flag[dtypes.int64(0)])

        ws_base = dtypes.uint64(ctx.localWsAddr)
        # 本端 dem_all 视图（[MAX_R, MAX_E] int32，本地读路径）
        dem_all = make_tensor(
            make_pointer(
                dtypes.int32,
                dtypes.int64(ws_base + dtypes.uint64(HEAP_BASE + DEM_ALL_OFF)),
                MemLoc.GM,
            ),
            make_layout((MAX_R, MAX_E)),
        )

        ub_tpe = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, num_experts)))
        mem_copy(ub_tpe, gm_tpe.view(1, num_experts))
        mem_copy(tile_slice(dem_all, (1, num_experts), (my_rank, 0)), ub_tpe)
        # 收齐：全部对端到达槽 ≥ epoch（tpe 行落地证明）
        for p in range_constexpr(world_size):
            if dtypes.uint64(p) != my_rank_u64:
                spin_u32_ge(
                    ctx,
                    dtypes.uint64(p) * dtypes.uint64(NOTIFY_STRIDE),
                    epoch,
                )
        mem_copy(ub_demand, tile_slice(dem_all, (world_size, num_experts), (0, 0)))
        mem_copy(gm_dem, ub_demand)

    @jit
    def _exchange_put_peer(self, ctx, gm_tpe, gm_flag, block_id):
        """块并行发射 tpe 行 + 哨兵：按实际块数跨步覆盖全部 peer（B<R 安全）。"""
        world_size, num_experts = self._rank_size, self._num_experts
        my_rank_u64 = dtypes.uint64(ctx.rankId)
        xfer_rb = make_xfer_rb()
        tpe_addr = tensor_address(gm_tpe)
        num_blocks = get_block_num()
        for peer in cannbotdsl.range(
            dtypes.int64(block_id), dtypes.int64(world_size), num_blocks
        ):
            if dtypes.uint64(peer) != my_rank_u64:
                peer_idx = dtypes.uint32(dtypes.uint64(peer))
                if dtypes.uint64(peer) > my_rank_u64:
                    peer_idx = peer_idx - dtypes.uint32(1)
                peerh = xfer_open(ctx, peer_idx)
                # ① tpe 行 → 对端 dem_all[my_rank]
                ub_write(
                    peerh,
                    dtypes.uint64(HEAP_BASE + DEM_ALL_OFF)
                    + my_rank_u64 * dtypes.uint64(MAX_E * _INT32_BYTES),
                    tpe_addr,
                    num_experts * _INT32_BYTES,
                    xfer_rb,
                )
                # ② epoch 哨兵 → 对端到达槽[my_rank]（ub_flag_write 首尾 fence
                #    ——MTE3 同通道保序：哨兵到达即此前数据全部落地）
                ub_flag_write(
                    peerh,
                    my_rank_u64 * dtypes.uint64(NOTIFY_STRIDE),
                    tensor_address(gm_flag),
                )

    # ---- 阶段 0：Count ----
    @jit
    def _count(self, block_id, gm_topk, gm_counts):
        """按 chunk 并行统计 per-expert token 数。"""
        num_experts, chunk, num_chunks, token_blocks = (
            self._num_experts,
            self._chunk,
            self._num_chunks,
            self._token_blocks,
        )
        num_hist_groups = ceil_div(num_experts, _HISTOGRAM_U8_LANES)
        use_vec = chunk % _HISTOGRAM_U8_LANES == 0 and chunk < _HISTOGRAM_MAX_CHUNK

        for k in cannbotdsl.range(num_chunks // token_blocks):
            chunk_id = block_id + k * token_blocks
            if const_expr(use_vec):
                self._count_vec(chunk_id, num_hist_groups, gm_topk, gm_counts)
            else:
                self._count_scalar(chunk_id, gm_topk, gm_counts)

    @jit
    def _count_vec(self, chunk_id, num_hist_groups, gm_topk, gm_counts):
        """向量直方图路径（uint8 打包 + 双 bin）。"""
        num_experts, chunk = self._num_experts, self._chunk
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_hist = Buffer(
            MemLoc.UB,
            dtypes.int32,
            make_layout((num_hist_groups * _HISTOGRAM_U8_LANES,)),
        )
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (chunk_id,)))
        with vf(mode="simd"):
            bmask = full_mask(32)
            sentinel = vdups(
                dtypes.int32(_HISTOGRAM_SENTINEL), dtypes.int32, mask=bmask
            )
            for boff in range_constexpr(0, chunk, _I32_LANES):
                topk_vec = vload(ub_topk, boff)
                vstore(
                    ub_topk,
                    boff,
                    vbitwise_or(topk_vec, sentinel, mask=bmask),
                    mask=bmask,
                )
        with vf(mode="simd"):
            # full_mask 参数为元素位宽（8=uint8, 16=uint16, 32=int32）
            mask8 = full_mask(8)
            mask16 = full_mask(16)
            mask32 = full_mask(32)
            for group in range_constexpr(num_hist_groups):
                acc0 = vdups(0, dtypes.uint16, mask=mask16)
                acc1 = vdups(0, dtypes.uint16, mask=mask16)
                for off in range_constexpr(0, chunk, _HISTOGRAM_U8_LANES):
                    low0, group0 = vload_deinterleave(ub_topk, off, width="b8")
                    low1, group1 = vload_deinterleave(
                        ub_topk, off + _HISTOGRAM_BINS_PER_VL, width="b8"
                    )
                    mask0 = veqs(group0, group, mask=mask8)
                    mask1 = veqs(group1, group, mask=mask8)
                    acc0 = vadd(
                        acc0,
                        vhistogram_frequency(low0, mask=mask0, bin=0),
                        mask=mask16,
                    )
                    acc0 = vadd(
                        acc0,
                        vhistogram_frequency(low1, mask=mask1, bin=0),
                        mask=mask16,
                    )
                    acc1 = vadd(
                        acc1,
                        vhistogram_frequency(low0, mask=mask0, bin=1),
                        mask=mask16,
                    )
                    acc1 = vadd(
                        acc1,
                        vhistogram_frequency(low1, mask=mask1, bin=1),
                        mask=mask16,
                    )
                base = group * _HISTOGRAM_U8_LANES
                low_part = vreinterpret(
                    vunpack(acc0, dtypes.uint32, part="lower"), dtypes.int32
                )
                vstore(ub_hist, base, low_part, mask32)
                high_part = vreinterpret(
                    vunpack(acc0, dtypes.uint32, part="upper"), dtypes.int32
                )
                vstore(ub_hist, base + _I32_LANES, high_part, mask32)
                low_part1 = vreinterpret(
                    vunpack(acc1, dtypes.uint32, part="lower"), dtypes.int32
                )
                vstore(ub_hist, base + _HISTOGRAM_BINS_PER_VL, low_part1, mask32)
                high_part1 = vreinterpret(
                    vunpack(acc1, dtypes.uint32, part="upper"), dtypes.int32
                )
                vstore(
                    ub_hist,
                    base + _HISTOGRAM_BINS_PER_VL + _I32_LANES,
                    high_part1,
                    mask32,
                )
        mem_copy(
            tile_slice(gm_counts, (1, num_experts), (chunk_id, 0)),
            ub_hist.view(1, num_experts),
        )

    @jit
    def _count_scalar(self, chunk_id, gm_topk, gm_counts):
        """标量回退路径（chunk 非 256 倍数或过大时）。"""
        num_experts, chunk = self._num_experts, self._chunk
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_cnt = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        with vf(mode="simt", thread=1):
            for e in cannbotdsl.range(num_experts):
                ub_cnt[dtypes.int64(e)] = dtypes.int32(0)
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (chunk_id,)))
        with vf(mode="simt", thread=1):
            for i in cannbotdsl.range(chunk):
                e = ub_topk[dtypes.int64(i)]
                ub_cnt[e] = ub_cnt[e] + dtypes.int32(1)
        mem_copy(
            tile_slice(gm_counts, (1, num_experts), (chunk_id, 0)),
            ub_cnt.view(1, num_experts),
        )

    @jit
    def _greedy_all(self, ctx, ub_demand, views, gm_cu):
        """demand 矩阵 → rem/order 分配 + cu + 冷专家 vid。"""
        world_size, num_experts = self._rank_size, self._num_experts
        epn = self._epn
        # ub_demand 读出后复用为 rem（控制 UB 峰值）
        ub_rem = ub_demand.view(num_experts, world_size)
        ub_order = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((num_experts, world_size))
        )
        ub_served = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_loads = Buffer(MemLoc.UB, dtypes.int32, make_layout((world_size,)))
        ub_rhome = Buffer(MemLoc.UB, dtypes.int32, make_layout((world_size,)))
        ub_aux = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        # cu 与 vid 生命周期重叠，不能别名复用（否则 vid 写进 cu）
        ub_cu = Buffer(MemLoc.UB, dtypes.int32, make_layout((_SLOTS_PER_RANK * epn,)))
        # ctx.rankId 须在 vf scope 外读取（datastruct_ref_load 不支持 vf 内）
        my_rank = dtypes.int32(ctx.rankId)

        with vf(mode="simt", thread=_PLAN_INIT_THREADS):
            thread_id = simt.thread_idx()[0]
            num_threads = simt.thread_dim()[0]
            for e in cannbotdsl.range(thread_id, num_experts, num_threads):
                expert_total = dtypes.int32(0)
                for r in cannbotdsl.range(world_size):
                    expert_total = (
                        expert_total + ub_demand[dtypes.int64(r), dtypes.int64(e)]
                    )
                ub_aux[dtypes.int64(e)] = expert_total
            simt.syncthreads()
            for e in cannbotdsl.range(thread_id, num_experts, num_threads):
                home_rank = e // epn
                for r in cannbotdsl.range(world_size):
                    ub_rem[dtypes.int64(e), dtypes.int64(r)] = dtypes.int32(0)
                    ub_order[dtypes.int64(e), dtypes.int64(r)] = _ORDER_UNSERVED
                ub_rem[dtypes.int64(e), home_rank] = ub_aux[dtypes.int64(e)]
                ub_order[dtypes.int64(e), home_rank] = _ORDER_HOME
                ub_served[dtypes.int64(e)] = dtypes.int32(1)
            simt.syncthreads()
            for r in cannbotdsl.range(thread_id, world_size, num_threads):
                if epn == 1:
                    # epn=1 时内层循环会被折叠成直写；这里直接发射
                    # 直写形态，避免编译校验误报控制路径变化
                    ub_loads[dtypes.int64(r)] = ub_aux[dtypes.int64(r)]
                else:
                    load = dtypes.int32(0)
                    for e in cannbotdsl.range(r * epn, (r + 1) * epn):
                        load = load + ub_aux[dtypes.int64(e)]
                    ub_loads[dtypes.int64(r)] = load
                ub_rhome[dtypes.int64(r)] = dtypes.int32(-1)
            simt.syncthreads()
            if thread_id < _WARP_SIZE:
                lane = simt.lane_idx()
                cap = dtypes.int32(0)
                for j in range_constexpr((world_size + _WARP_SIZE - 1) // _WARP_SIZE):
                    k = lane + j * _WARP_SIZE
                    v = dtypes.int32(0)
                    if k < world_size:
                        v = ub_loads[dtypes.int64(k)]
                    cap = cap + simt.reduce_add(v)
                cap = cap // world_size
                for _ in cannbotdsl.range(world_size):
                    src_r, max_load = self._warp_extreme_idx(
                        ub_loads, world_size, lane, True
                    )
                    if max_load > cap:
                        dst_r, min_load = self._warp_min_source_idx(
                            ub_loads, ub_rhome, src_r, world_size, lane
                        )
                        moved = self._greedy_round_warp(
                            ub_rem,
                            ub_order,
                            ub_served,
                            ub_rhome,
                            src_r,
                            dst_r,
                            min_load,
                            cap,
                            lane,
                        )
                        if lane == 0:
                            ub_loads[dtypes.int64(src_r)] = (
                                ub_loads[dtypes.int64(src_r)] - moved
                            )
                            ub_loads[dtypes.int64(dst_r)] = cap
                        simt.syncthreads()
                if lane == 0:
                    self._fill_cu_seqlens(my_rank, ub_rem, ub_order, ub_rhome, ub_cu)
            simt.syncthreads()
            # ub_aux 复用为 vid 表：热专家 -1，冷专家直接给出虚拟 id
            for e in cannbotdsl.range(thread_id, num_experts, num_threads):
                home_rank = e // epn
                if ub_served[dtypes.int64(e)] > dtypes.int32(1):
                    ub_aux[dtypes.int64(e)] = _VID_HOT
                else:
                    ub_aux[dtypes.int64(e)] = dtypes.int32(
                        home_rank * (_SLOTS_PER_RANK * epn) + (e % epn)
                    )
            simt.syncthreads()
        mem_copy(views.rem, ub_rem)
        mem_copy(views.order, ub_order)
        mem_copy(views.rhome, ub_rhome)
        mem_copy(views.vid, ub_aux)
        mem_copy(gm_cu, tile_slice(ub_cu, (_SLOTS_PER_RANK * epn,), (0,)))

    @jit
    def _warp_extreme_idx(self, ub_values, num_values, lane, maximize):
        """warp 归约极值，并列取最小下标（maximize 选 True/False 对应 max/min）。"""
        best = dtypes.int32(_GREEDY_NEG_INF if maximize else _GREEDY_POS_INF)
        idx = dtypes.int32(0)
        pad = dtypes.int32(_GREEDY_NEG_INF if maximize else _GREEDY_POS_INF)
        for j in range_constexpr((num_values + _WARP_SIZE - 1) // _WARP_SIZE):
            k = lane + j * _WARP_SIZE
            v = pad
            if k < num_values:
                v = ub_values[dtypes.int64(k)]
            if const_expr(maximize):
                round_best = simt.reduce_max(v)
            else:
                round_best = simt.reduce_min(v)
            candidate = dtypes.int32(num_values)
            if (k < num_values) and (v == round_best):
                candidate = k
            round_idx = simt.reduce_min(candidate)
            if const_expr(maximize):
                better = round_best > best
            else:
                better = round_best < best
            if better:
                best = round_best
                idx = round_idx
        return idx, best

    @jit
    def _warp_min_source_idx(self, ub_loads, ub_rhome, src_r, world_size, lane):
        """选择与源 rank 兼容的最轻目标 rank。"""
        best = dtypes.int32(_GREEDY_POS_INF)
        idx = dtypes.int32(-1)
        for j in range_constexpr((world_size + _WARP_SIZE - 1) // _WARP_SIZE):
            k = lane + j * _WARP_SIZE
            value = dtypes.int32(_GREEDY_POS_INF)
            if k < world_size:
                source = ub_rhome[dtypes.int64(k)]
                if source < dtypes.int32(0) or source == src_r:
                    value = ub_loads[dtypes.int64(k)]
            round_best = simt.reduce_min(value)
            candidate = dtypes.int32(world_size)
            if k < world_size and value == round_best:
                candidate = k
            round_idx = simt.reduce_min(candidate)
            if round_best < best:
                best = round_best
                idx = round_idx
        if best == dtypes.int32(_GREEDY_POS_INF):
            idx = src_r
            best = ub_loads[dtypes.int64(src_r)]
        return idx, best

    @jit
    def _greedy_round_warp(
        self, ub_rem, ub_order, ub_served, ub_rhome, src_r, dst_r, min_load, cap, lane
    ):
        """一轮 src→dst 搬移；专家 e 的 rem/order/served 由 e%32 号 lane 独占，
        跨 lane 只经 warp 归约，内层无需栅栏。"""
        epn = self._epn
        move = cap - min_load
        moved = dtypes.int32(0)
        # 属主对齐起点：本 lane 持有的本地专家 i 满足 i ≡ lane - src*epn (mod 32)
        i_start = (lane + dtypes.int32(32) - (src_r * epn) % dtypes.int32(32)) % (
            dtypes.int32(32)
        )
        for _ in cannbotdsl.range(epn):
            if move > dtypes.int32(0):
                best = dtypes.int32(0)
                idx = dtypes.int32(-1)
                for i in cannbotdsl.range(i_start, epn, dtypes.int32(32)):
                    v = ub_rem[dtypes.int64(src_r * epn + i), dtypes.int64(src_r)]
                    if v > best:
                        best = v
                        idx = i
                round_best = simt.reduce_max(best)
                candidate = dtypes.int32(epn)
                if (best == round_best) and (round_best > dtypes.int32(0)):
                    candidate = idx
                round_idx = simt.reduce_min(candidate)
                take = round_best
                if move < take:
                    take = move
                if round_best > dtypes.int32(0):
                    e_best = src_r * epn + round_idx
                    if lane == e_best % dtypes.int32(32):
                        if ub_order[dtypes.int64(e_best), dtypes.int64(dst_r)] == (
                            _ORDER_UNSERVED
                        ):
                            ub_served[dtypes.int64(e_best)] = ub_served[
                                dtypes.int64(e_best)
                            ] + dtypes.int32(1)
                            ub_order[dtypes.int64(e_best), dtypes.int64(dst_r)] = (
                                ub_served[dtypes.int64(e_best)]
                            )
                        ub_rem[dtypes.int64(e_best), dtypes.int64(src_r)] = (
                            ub_rem[dtypes.int64(e_best), dtypes.int64(src_r)] - take
                        )
                        ub_rem[dtypes.int64(e_best), dtypes.int64(dst_r)] = (
                            ub_rem[dtypes.int64(e_best), dtypes.int64(dst_r)] + take
                        )
                    move = move - take
                    moved = moved + take
        # 仅在确有搬移时登记源 rank：零搬移轮写 rhome 会让清单漏掉实际副本。
        if lane == dtypes.int32(0) and moved > dtypes.int32(0):
            ub_rhome[dtypes.int64(dst_r)] = src_r
        return moved

    @jit
    def _fill_cu_seqlens(self, my_rank, ub_rem, ub_order, ub_rhome, ub_cu):
        """[2*epn] 槽位行数前缀和 → cu_seqlens；副本槽按源专家升序压缩。"""
        epn = self._epn
        src_rank = ub_rhome[dtypes.int64(my_rank)]
        for i in range_constexpr(epn):
            ub_cu[dtypes.int64(i)] = ub_rem[
                dtypes.int64(my_rank * epn + i), dtypes.int64(my_rank)
            ]
        for i in range_constexpr(epn):
            ub_cu[dtypes.int64(epn + i)] = dtypes.int32(0)
        if src_rank >= dtypes.int32(0):
            copy_cnt = dtypes.int32(0)
            for i in cannbotdsl.range(epn):
                e = src_rank * epn + i
                if (
                    ub_order[dtypes.int64(e), dtypes.int64(my_rank)]
                    >= _ORDER_FIRST_REPLICA
                ):
                    ub_cu[dtypes.int64(epn) + copy_cnt] = ub_rem[
                        dtypes.int64(e), dtypes.int64(my_rank)
                    ]
                    copy_cnt = copy_cnt + dtypes.int32(1)
        acc = dtypes.int32(0)
        for i in range_constexpr(_SLOTS_PER_RANK * epn):
            acc = acc + ub_cu[dtypes.int64(i)]
            ub_cu[dtypes.int64(i)] = acc

    # ---- 阶段 1：Tables ----
    @jit
    def _tables_loop(self, block_id, num_blocks, views, gm_copy_list):
        """按块步进轮询处理 rank 行（R 可大于块数）。"""
        for r in cannbotdsl.range(block_id, self._rank_size, num_blocks):
            self._tables(r, views, gm_copy_list)

    def _tables(self, r, views, gm_copy_list):
        """生成 rank r 的副本槽映射与副本清单行。"""
        epn = self._epn
        ub_rep_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, epn)))
        ub_copy_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, epn)))

        with vf(mode="simt", thread=1):
            self._fill_rep_copy(
                views.order, views.rhome, ub_rep_row, ub_copy_row, r, epn
            )
        mem_copy(tile_slice(views.rep_slot, (1, epn), (r, 0)), ub_rep_row)
        mem_copy(tile_slice(gm_copy_list, (1, epn), (r, 0)), ub_copy_row)

    @jit
    def _fill_rep_copy(self, gm_order, gm_rhome, ub_rep_row, ub_copy_row, r, epn):
        """压缩副本映射：rep_slot 行与 copy_list 行。"""
        for j in range_constexpr(epn):
            ub_rep_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(0)
            ub_copy_row[dtypes.int64(0), dtypes.int64(j)] = dtypes.int32(-1)
        home = simt.l2cache_load(gm_rhome.ptr((dtypes.int64(r),)))
        if home >= dtypes.int32(0):
            copy_idx = dtypes.int32(0)
            for i in cannbotdsl.range(epn):
                e = home * epn + i
                order_seg = simt.l2cache_load(
                    gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                )
                if order_seg >= _ORDER_FIRST_REPLICA:
                    ub_rep_row[dtypes.int64(0), dtypes.int64(i)] = copy_idx
                    ub_copy_row[dtypes.int64(0), copy_idx] = dtypes.int32(e)
                    copy_idx = copy_idx + dtypes.int32(1)

    # ---- 阶段 1：Pfx ----
    @jit
    def _pfx(self, ctx, block_id, views):
        """热专家的 chunk 前缀位置表与段查找表。"""
        world_size, num_experts = self._rank_size, self._num_experts
        num_chunks = self._num_chunks
        experts_per_block = num_experts // self._pfx_blocks
        expert_start = block_id * experts_per_block

        ord_rank_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, world_size)))
        ord_end_row = Buffer(MemLoc.UB, dtypes.int32, make_layout((1, world_size)))

        for k in cannbotdsl.range(experts_per_block):
            e = expert_start + k
            self._build_pfx_row(ctx, e, views, ord_rank_row, ord_end_row)
            mem_copy(tile_slice(views.ord_rank, (1, world_size), (e, 0)), ord_rank_row)
            mem_copy(tile_slice(views.ord_end, (1, world_size), (e, 0)), ord_end_row)

    @jit
    def _build_pfx_row(self, ctx, e, views, ord_rank_row, ord_end_row):
        """单个专家的 pcnt/ord_rank/ord_end 行。"""
        world_size = self._rank_size
        my_rank = dtypes.int32(ctx.rankId)  # 运行时取（self-serve 偏好序）
        num_chunks = self._num_chunks
        gm_dem, gm_counts = views.dem, views.counts
        gm_rem, gm_order, gm_vid = views.rem, views.order, views.vid
        with vf(mode="simt", thread=1):
            # 冷专家早退：Assign 冷专家路径直读 vid，不消费本行 pcnt/ord
            vid = simt.l2cache_load(gm_vid.ptr((dtypes.int64(e),)))
            if vid < dtypes.int32(0):
                # 先建段序表（偏好序依赖 home/副本段排列）
                for segment_index in cannbotdsl.range(world_size):
                    ord_rank_row[dtypes.int64(0), dtypes.int64(segment_index)] = (
                        dtypes.int32(-1)
                    )
                for r in cannbotdsl.range(world_size):
                    order_seg = simt.l2cache_load(
                        gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                    )
                    if order_seg >= _ORDER_HOME:
                        ord_rank_row[dtypes.int64(0), dtypes.int64(order_seg - 1)] = (
                            dtypes.int32(r)
                        )
                # 偏好序 [home, 副本按段序, 非服务 rank]，pcnt 取其前序需求之和
                prefix_base = dtypes.int32(0)
                seen_self = dtypes.int32(0)
                for segment_index in cannbotdsl.range(world_size):
                    r = ord_rank_row[dtypes.int64(0), dtypes.int64(segment_index)]
                    if r >= dtypes.int32(0) and seen_self == dtypes.int32(0):
                        if r == my_rank:
                            seen_self = dtypes.int32(1)
                        else:
                            prefix_base = prefix_base + simt.l2cache_load(
                                gm_dem.ptr((dtypes.int64(r), dtypes.int64(e)))
                            )
                for r in cannbotdsl.range(world_size):
                    if seen_self == dtypes.int32(0):
                        if r == my_rank:
                            seen_self = dtypes.int32(1)
                        else:
                            order_seg = simt.l2cache_load(
                                gm_order.ptr((dtypes.int64(e), dtypes.int64(r)))
                            )
                            if order_seg == _ORDER_UNSERVED:
                                prefix_base = prefix_base + simt.l2cache_load(
                                    gm_dem.ptr((dtypes.int64(r), dtypes.int64(e)))
                                )
                chunk_prefix = dtypes.int32(0)
                for c in cannbotdsl.range(num_chunks):
                    simt.l2cache_store(
                        views.pcnt.ptr((dtypes.int64(e), dtypes.int64(c))),
                        prefix_base + chunk_prefix,
                    )
                    chunk_prefix = chunk_prefix + simt.l2cache_load(
                        gm_counts.ptr((dtypes.int64(c), dtypes.int64(e)))
                    )
                seg_acc = dtypes.int32(0)
                for segment_index in cannbotdsl.range(world_size):
                    seg_rank = ord_rank_row[
                        dtypes.int64(0), dtypes.int64(segment_index)
                    ]
                    if seg_rank >= dtypes.int32(0):
                        seg_acc = seg_acc + simt.l2cache_load(
                            gm_rem.ptr((dtypes.int64(e), dtypes.int64(seg_rank)))
                        )
                    ord_end_row[dtypes.int64(0), dtypes.int64(segment_index)] = seg_acc

    # ---- 阶段 2：Assign ----
    @jit
    def _assign(self, block_id, gm_topk, gm_vtopk, views):
        """token 落位：按 chunk 并行处理冷/热专家路径。"""
        world_size, num_experts = self._rank_size, self._num_experts
        num_chunks, token_blocks = self._num_chunks, self._token_blocks

        # ord_end 参与二分，驻 UB；ord_rank / rep_slot 每轮直读 L2，省 buf_id。
        self._ub_ord_end = Buffer(
            MemLoc.UB, dtypes.int32, make_layout((num_experts, world_size))
        )
        mem_copy(self._ub_ord_end, views.ord_end)
        self._ub_vid = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        mem_copy(self._ub_vid, views.vid)

        for k in cannbotdsl.range(num_chunks // token_blocks):
            self._assign_chunk(block_id + k * token_blocks, gm_topk, gm_vtopk, views)

    @jit
    def _assign_chunk(self, chunk_id, gm_topk, gm_vtopk, views):
        """单 chunk 落位：冷专家直读 vid，热专家每 (专家, chunk) 定位一次，
        跨段 token 走逐 token 标量兜底。"""
        num_experts = self._num_experts
        chunk = self._chunk
        gm_pcnt, gm_ord_rank, gm_rep_slot = views.pcnt, views.ord_rank, views.rep_slot
        gm_counts = views.counts
        ub_topk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_vtopk = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_hot_idx = Buffer(MemLoc.UB, dtypes.int32, make_layout((chunk,)))
        ub_num_hot = Buffer(MemLoc.UB, dtypes.int32, make_layout((1,)))
        ub_vid_chunk = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_pos = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_seg_end = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))
        ub_cur_vid = Buffer(MemLoc.UB, dtypes.int32, make_layout((num_experts,)))

        with vf(mode="simt", thread=1):
            for e in cannbotdsl.range(num_experts):
                vid_e = self._ub_vid[dtypes.int64(e)]
                if vid_e < dtypes.int32(0):
                    pos0 = simt.l2cache_load(
                        gm_pcnt.ptr((dtypes.int64(e), dtypes.int64(chunk_id)))
                    )
                    cnt = simt.l2cache_load(
                        gm_counts.ptr((dtypes.int64(chunk_id), dtypes.int64(e)))
                    )
                    if cnt > dtypes.int32(0):
                        pos1 = pos0 + cnt
                        vid0, segment_index = self._relocate(
                            e, pos0, gm_ord_rank, gm_rep_slot
                        )
                        seg_end = self._ub_ord_end[
                            dtypes.int64(e), dtypes.int64(segment_index)
                        ]
                        if pos1 <= seg_end:
                            ub_vid_chunk[dtypes.int64(e)] = vid0
                        else:
                            # 本 chunk 内跨段：交给标量路径逐 token 定位
                            ub_vid_chunk[dtypes.int64(e)] = dtypes.int32(-1)
                            ub_pos[dtypes.int64(e)] = pos0
                            ub_seg_end[dtypes.int64(e)] = dtypes.int32(0)
                    else:
                        # 本 chunk 无该专家 token，值不会被读到
                        ub_vid_chunk[dtypes.int64(e)] = dtypes.int32(-1)
                else:
                    ub_vid_chunk[dtypes.int64(e)] = vid_e
        mem_copy(ub_topk, tile_slice(gm_topk, (chunk,), (chunk_id,)))
        vec_end = chunk - chunk % _I32_LANES
        if const_expr(vec_end > 0):
            with vf(mode="simd"):
                mask = full_mask(32)  # int32 全 lane
                if const_expr(_HAS_LEGACY_SQUEEZE_STORE):
                    cursor = vstore_unalign_begin(ub_hot_idx)
                    for off in range_constexpr(0, vec_end, _I32_LANES):
                        indices = vreinterpret(vload(ub_topk, off), dtypes.uint32)
                        values = vgather(ub_vid_chunk, indices, mask=mask)
                        vstore(ub_vtopk, off, values, mask)
                        hot = vlts(values, 0, mask=mask)
                        positions = varange(off, dtypes.int32)
                        squeezed = vsqueeze_and_storeunalign_init(positions, mask=hot)
                        vsqueeze_and_storeunalign(ub_hot_idx, 0, squeezed, cursor)
                else:
                    cursor = vsqueeze_and_storeunalign_init(ub_hot_idx)
                    for off in range_constexpr(0, vec_end, _I32_LANES):
                        indices = vreinterpret(vload(ub_topk, off), dtypes.uint32)
                        values = vgather(ub_vid_chunk, indices, mask=mask)
                        vstore(ub_vtopk, off, values, mask)
                        hot = vlts(values, 0, mask=mask)
                        positions = varange(off, dtypes.int32)
                        vsqueeze_and_storeunalign(
                            ub_hot_idx, 0, positions, cursor, mask=hot
                        )
                vsqueeze_and_storeunalign_finalize(ub_hot_idx, 0, cursor)
                vstorealign_squeeze_status(ub_num_hot, 0)
            vec_sync_all()
        with vf(mode="simt", thread=1):
            # squeeze 状态字节数 → int32 计数
            if const_expr(vec_end > 0):
                num_hot = ub_num_hot[dtypes.int64(0)] // _INT32_BYTES
            else:
                num_hot = dtypes.int32(0)
            for i in range_constexpr(vec_end, chunk):
                e = ub_topk[dtypes.int64(i)]
                cur_vid = ub_vid_chunk[dtypes.int64(e)]
                ub_vtopk[dtypes.int64(i)] = cur_vid
                if cur_vid < dtypes.int32(0):
                    ub_hot_idx[dtypes.int64(num_hot)] = dtypes.int32(i)
                    num_hot = num_hot + dtypes.int32(1)
            for j in cannbotdsl.range(num_hot):
                i = ub_hot_idx[dtypes.int64(j)]
                e = ub_topk[dtypes.int64(i)]
                pos = ub_pos[dtypes.int64(e)]
                if pos >= ub_seg_end[dtypes.int64(e)]:
                    vid, segment_index = self._relocate(
                        e, pos, gm_ord_rank, gm_rep_slot
                    )
                    ub_seg_end[dtypes.int64(e)] = self._ub_ord_end[
                        dtypes.int64(e), segment_index
                    ]
                    ub_cur_vid[dtypes.int64(e)] = vid
                ub_vtopk[dtypes.int64(i)] = ub_cur_vid[dtypes.int64(e)]
                ub_pos[dtypes.int64(e)] = pos + dtypes.int32(1)
        mem_copy(tile_slice(gm_vtopk, (chunk,), (chunk_id,)), ub_vtopk)

    @jit
    def _relocate(self, e, pos, gm_ord_rank, gm_rep_slot):
        """二分定位 pos 所在段，返回 (vid, segment_index)。"""
        world_size, epn = self._rank_size, self._epn
        # ord_end[e,:] 非降累计——二分找首个 pos < end 的段
        low_index = dtypes.int32(0)
        span = dtypes.int32(world_size)
        for _ in range_constexpr(world_size.bit_length()):
            half = span // dtypes.int32(2)
            mid_index = low_index + half
            end_mid_value = self._ub_ord_end[dtypes.int64(e), dtypes.int64(mid_index)]
            if pos < end_mid_value:
                span = half
            else:
                low_index = mid_index + dtypes.int32(1)
                span = span - half - dtypes.int32(1)
        segment_index = low_index
        seg_rank = simt.l2cache_load(
            gm_ord_rank.ptr((dtypes.int64(e), dtypes.int64(segment_index)))
        )
        vid = seg_rank * dtypes.int32(_SLOTS_PER_RANK * epn) + (e % dtypes.int32(epn))
        if segment_index >= dtypes.int32(1):
            vid = (
                seg_rank * dtypes.int32(_SLOTS_PER_RANK * epn)
                + dtypes.int32(epn)
                + simt.l2cache_load(
                    gm_rep_slot.ptr((dtypes.int64(seg_rank), dtypes.int64(e % epn)))
                )
            )
        return vid, segment_index


# --------------------------------------------------------------------------
# host 入口
# --------------------------------------------------------------------------


def _plan_chunking(num_rows, block_limit=None):
    """chunk 取整除 N 的 2 的幂，范围 [_CHUNK_MIN, _VF_SAFE_CHUNK]。

    上限来自 vf 栈：constexpr 展开次数随 chunk 增长，chunk=2048 时栈
    6720B 超 6144B 上限、编译失败（A5 S=8192）。
    """
    cap = resident_blocks(_MAX_PLAN_BLOCKS) if block_limit is None else int(block_limit)
    target_chunk = max(_CHUNK_MIN, ceil_div(num_rows, cap))
    chunk = ceil_pow2(target_chunk)
    if chunk > _VF_SAFE_CHUNK:
        chunk = _VF_SAFE_CHUNK
    while chunk > 1 and num_rows % chunk != 0:
        chunk //= 2
    num_chunks = num_rows // chunk
    token_blocks = min(num_chunks, cap)
    while token_blocks > 1 and num_chunks % token_blocks != 0:
        token_blocks -= 1
    return chunk, num_chunks, token_blocks


def _pfx_block_count(num_experts, block_limit=None):
    """Pfx 的 block 数：E 的可整除因子，且不超过本机可常驻核数。"""
    cap = resident_blocks(_MAX_PLAN_BLOCKS) if block_limit is None else int(block_limit)
    block_count = min(num_experts, cap)
    while block_count > 1 and num_experts % block_count != 0:
        block_count -= 1
    return block_count


class _PlanBlocks(NamedTuple):
    """plan 分块：chunk 总数、token block 数与 Pfx block 数。"""

    num_chunks: int
    token_blocks: int
    pfx_blocks: int


def _plan_blocks(num_rows, num_experts, block_limit=None):
    """分块与专家块一次算全；不做 lru_cache（块数依赖运行期核数）。"""
    _, num_chunks, token_blocks = _plan_chunking(num_rows, block_limit)
    return _PlanBlocks(
        num_chunks, token_blocks, _pfx_block_count(num_experts, block_limit)
    )


@host
def _moonep_plan_host(
    gm_tpe,
    gm_flag,
    gm_ws,
    gm_topk,
    gm_io,
    ctx: StructRef[MoonepCommCtx],
):
    """按 EP 组规模选择 launch 形态：小 EP 融合，大 EP 分阶段。"""
    num_experts = gm_tpe.shape[0]
    world_size = _RANK_SIZE  # trace 期烘焙；rank 运行时由 ctx.rankId 取
    num_rows = gm_topk.shape[0]  # 本 rank topk 扁平视图，直接作 Cnt/Asg 输入
    blocks = _plan_blocks(num_rows, num_experts, _PLAN_BLOCK_LIMIT)
    split_preparation = world_size >= 16
    if not split_preparation:
        _PlanKernel(
            world_size,
            num_experts,
            num_rows,
            blocks.num_chunks,
            blocks.token_blocks,
            blocks.pfx_blocks,
            0,
        )[
            min(
                max(blocks.token_blocks, blocks.pfx_blocks),
                _PLAN_BLOCK_LIMIT,
            )
        ](
            gm_tpe,
            gm_flag,
            gm_ws,
            gm_topk,
            gm_io,
            ctx,
        )
    if split_preparation:
        _PlanKernel(
            world_size,
            num_experts,
            num_rows,
            blocks.num_chunks,
            blocks.token_blocks,
            blocks.pfx_blocks,
            3,
        )[
            min(
                max(blocks.token_blocks, blocks.pfx_blocks),
                _PLAN_BLOCK_LIMIT,
            )
        ](
            gm_tpe,
            gm_flag,
            gm_ws,
            gm_topk,
            gm_io,
            ctx,
        )
    if split_preparation:
        _PlanKernel(
            world_size,
            num_experts,
            num_rows,
            blocks.num_chunks,
            blocks.token_blocks,
            blocks.pfx_blocks,
            4,
        )[1](
            gm_tpe,
            gm_flag,
            gm_ws,
            gm_topk,
            gm_io,
            ctx,
        )
    if split_preparation:
        _PlanKernel(
            world_size,
            num_experts,
            num_rows,
            blocks.num_chunks,
            blocks.token_blocks,
            blocks.pfx_blocks,
            2,
        )[
            min(
                max(blocks.token_blocks, blocks.pfx_blocks),
                _PLAN_BLOCK_LIMIT,
            )
        ](
            gm_tpe,
            gm_flag,
            gm_ws,
            gm_topk,
            gm_io,
            ctx,
        )
    _PlanKernel(
        world_size,
        num_experts,
        num_rows,
        blocks.num_chunks,
        blocks.token_blocks,
        blocks.pfx_blocks,
        1,
    )[blocks.token_blocks](gm_tpe, gm_flag, gm_ws, gm_topk, gm_io, ctx)


_PLAN_PROGRAMS = {}
_PLAN_LOCK = threading.Lock()

# dst_n/src_n 零占位：只读、kernel 不写，进程级缓存省 alloc+H2D。
_PLACEHOLDER_CACHE = {}


class _PlanDims(NamedTuple):
    """plan 形状键：单 rank token 数 / top-k / 专家总数 / EP rank 数。"""

    num_tokens: int
    top_k: int
    num_experts: int
    world_size: int


# workspace 布局：(表名, 维度名)；元素数为各维之积。
_WS_SPECS = (
    ("dem", ("rank_size", "num_experts")),
    ("rem", ("num_experts", "rank_size")),
    ("order", ("num_experts", "rank_size")),
    ("counts", ("num_chunks", "num_experts")),
    ("pcnt", ("num_experts", "num_chunks")),
    ("vid", ("num_experts",)),
    ("ord_rank", ("num_experts", "rank_size")),
    ("ord_end", ("num_experts", "rank_size")),
    ("rep_slot", ("rank_size", "epn")),
    ("rhome", ("rank_size",)),
)


def _ws_tables(rank_size, num_experts, num_chunks):
    """(name, offset, count, shape) 序列 + 总元素数（布局的唯一来源）。"""
    dims = {
        "rank_size": rank_size,
        "num_experts": num_experts,
        "num_chunks": num_chunks,
        "epn": num_experts // rank_size,
    }
    out = []
    cur = 0
    for name, dim_names in _WS_SPECS:
        shape = tuple(dims[d] for d in dim_names)
        count = 1
        for dim in shape:
            count *= dim
        cur = align_up(cur, _WS_ALIGN)
        out.append((name, cur, count, shape))
        cur += count
    return tuple(out), align_up(cur, _WS_ALIGN)


def _ws_layout(rank_size, num_experts, num_chunks):
    """workspace 布局：{表名: 元素偏移} + total（host 分配与测试使用）。"""
    tables, total = _ws_tables(rank_size, num_experts, num_chunks)
    offs = {name: off for name, off, _, _ in tables}
    offs["total"] = total
    return offs


class _PlanViews(NamedTuple):
    """plan 中间表在 workspace 上的具名视图（kernel 一次绑定、按需传参）。"""

    dem: object
    rem: object
    order: object
    counts: object
    pcnt: object
    vid: object
    ord_rank: object
    ord_end: object
    rep_slot: object
    rhome: object


def _bind_plan_views(gm_ws, rank_size, num_experts, num_chunks):
    """按 _WS_SPECS 把扁平 workspace 绑成具名视图（布局与切片同一来源）。"""
    tables, _ = _ws_tables(rank_size, num_experts, num_chunks)
    bound = {}
    for name, off, count, shape in tables:
        seg_end = off + count
        seg = gm_ws[off:seg_end,]
        bound[name] = seg.view(*shape) if len(shape) > 1 else seg
    return _PlanViews(
        dem=bound["dem"],
        rem=bound["rem"],
        order=bound["order"],
        counts=bound["counts"],
        pcnt=bound["pcnt"],
        vid=bound["vid"],
        ord_rank=bound["ord_rank"],
        ord_end=bound["ord_end"],
        rep_slot=bound["rep_slot"],
        rhome=bound["rhome"],
    )


def _zero_placeholder(num_rows):
    """进程级缓存的零占位张量（只读）。"""
    t = _PLACEHOLDER_CACHE.get(num_rows)
    if t is None:
        t = torch.zeros(num_rows, dtype=torch.int32, device="npu")
        _PLACEHOLDER_CACHE[num_rows] = t
    return t


def _get_plan_program(
    num_tokens,
    top_k,
    num_experts,
    world_size,
    block_limit=None,
):
    """按 (S,K,E,R,块数) 键控的编译产物缓存。"""
    limit = (
        resident_blocks(_MAX_PLAN_BLOCKS) if block_limit is None else int(block_limit)
    )
    # 块数也是编译期常量（角色分派），核数不同的机器不能复用同一 program
    key = (num_tokens, top_k, num_experts, world_size, limit)
    with _PLAN_LOCK:
        program = _PLAN_PROGRAMS.get(key)
        if program is not None:
            return program
        global _PLAN_BLOCK_LIMIT
        _PLAN_BLOCK_LIMIT = limit
        num_rows = num_tokens * top_k
        blocks = _plan_blocks(num_rows, num_experts, limit)
        ws_total = _ws_layout(world_size, num_experts, blocks.num_chunks)["total"]
        global _RANK_SIZE
        _RANK_SIZE = world_size
        epn = num_experts // world_size
        i32 = dtypes.int32
        program = cannbotdsl.compile(
            _moonep_plan_host,
            TensorSpec((num_experts,), i32),  # 每专家 token 数
            TensorSpec((1,), i32),  # epoch 标志
            TensorSpec((ws_total,), i32),  # dem 与打包 workspace
            TensorSpec((num_rows,), i32),  # topk 扁平视图
            TensorSpec(
                (_SLOTS_PER_RANK * epn + world_size * epn + num_rows,), i32
            ),  # 合并输出
            StructRef[MoonepCommCtx],
        )
        _PLAN_PROGRAMS[key] = program
        return program


class MoonepPlan(NamedTuple):
    """plan 五件套句柄（moonep_plan 输出的纯数据包装，零同步契约保持）。

    NamedTuple 选型：元组解包兼容（``dst_n, cu_seqlens, experts_to_copy,
    src_n, vtopk = plan`` 仍成立，迁移面最小）+ 属性访问 + 零运行时开销；
    不添加任何同步类方法（plan 返回路径零同步，溢出检查延迟到
    prefetch/reduce 尾部——既有模块行为透传）。语义对标 GPU MoonEPCommPlan。

    Fields:
        dst_n: [N] int32 零占位（vtopk-only 快路径不计算）。
        cu_seqlens: [2*epn] int32 分组偏移（GMM 分段边界）。
        experts_to_copy: [R, epn] int32 副本清单（-1 填充；每行至多一个
            远程 home rank），prefetch/reduce 共用。
        src_n: [N] int32 零占位（vtopk-only 快路径不计算）。
        vtopk: [S, K] int32 虚拟专家 id（扩展空间 2E，供 moe_ep dispatch 路由）。
    """

    dst_n: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    src_n: torch.Tensor
    vtopk: torch.Tensor


def moonep_plan(context, tokens_per_expert, topk_experts_sk):
    """规划入口。

    Args:
        context: MoonEP 通信上下文（见 moonep_context.py）。
        tokens_per_expert: [E] int32，本 rank 各专家的 token 计数。
        topk_experts_sk: [S, K] int32，本 rank 的 top-k 专家索引。

    Returns:
        MoonepPlan（NamedTuple，可按 5-tuple 解包）：
        dst_n[N] / src_n[N] 为零占位；cu_seqlens[2*epn]、experts_to_copy[R,epn]
        与 vtopk[S,K] 由 kernel 全量产出。

    Notes:
        - 返回路径零同步：溢出与布局不变量延迟到下一个 moonep 算子尾部校验。
        - 形状约束：E <= 1024、R <= MAX_R（128）、E % R == 0；S*K 任意 N 原生整除。
        - UB 约束：2*R*E*4 <= 192KiB；超出时在入口报错，避免 kernel 资源失败。
        - 规划结果可跨调用、跨前反向复用。
    """

    if tokens_per_expert.dtype != torch.int32 or topk_experts_sk.dtype != torch.int32:
        raise TypeError(
            f"tpe/topk 须为 int32（bincount/topk 默认 int64，需 .to(torch.int32)）："
            f"got {tokens_per_expert.dtype} / {topk_experts_sk.dtype}"
        )
    cur_stream = sync_transport_stream(context)  # 跨流护栏（握手状态机要求调用按序）
    world_size = context.rank_size
    # 硬约束：R ≤ MAX_R（dem/rhome 表维度；MAX_R=128 时 dem_all 区 512KB）
    if world_size > MAX_R:
        raise RuntimeError(
            f"EP 组规模 {world_size} 超过 MAX_R={MAX_R}（dem/rhome 表维度上限）"
        )
    num_tokens, top_k = topk_experts_sk.shape
    num_experts = tokens_per_expert.shape[0]
    if num_experts % world_size != 0:
        raise ValueError(f"E={num_experts} 不能被 R={world_size} 整除")
    if num_experts > MAX_E:
        raise ValueError(f"E={num_experts} 超过 v1 上限 MAX_E={MAX_E}")
    ub_need = 2 * world_size * num_experts * _INT32_BYTES
    if ub_need > _PLAN_UB_BUDGET:
        raise ValueError(
            f"plan 的 dem/order UB 需求 {ub_need}B 超过预算 {_PLAN_UB_BUDGET}B："
            f"R={world_size}, E={num_experts}；"
            f"当前支持 2*R*E*4 <= {_PLAN_UB_BUDGET}"
        )
    epn = num_experts // world_size
    num_rows = num_tokens * top_k  # 任意 N 原生整除（分块降档，无尾补机制）
    topk_flat = topk_experts_sk.reshape(-1)
    bad_topk = ((topk_flat < 0) | (topk_flat >= num_experts)).any()
    topk_safe = topk_flat.clamp(0, num_experts - 1)
    dims = _PlanDims(num_tokens, top_k, num_experts, world_size)
    block_limit = resident_blocks(_MAX_PLAN_BLOCKS)

    dst_n = _zero_placeholder(num_rows)
    # 三份输出共享一次分配；kernel 全量覆写，empty 即可
    cu_elems = _SLOTS_PER_RANK * epn
    etc_elems = world_size * epn
    gm_io = torch.empty(
        cu_elems + etc_elems + num_rows, dtype=torch.int32, device="npu"
    )
    cu_seqlens = gm_io[:cu_elems]
    etc_end = cu_elems + etc_elems
    experts_to_copy = gm_io[cu_elems:etc_end].view(world_size, epn)
    src_n = _zero_placeholder(num_rows)
    vtopk = gm_io[etc_end:].view(num_tokens, top_k)
    # epoch 全算子共用一个单调计数器（跨算子单调，哨兵不串轮）
    gm_flag = next_barrier_epoch(context)
    # workspace 单槽缓存：同 shape 复用，shape 变化时按新布局重分配。
    ws_entry = context.plan_workspace
    if ws_entry is None or ws_entry[0] != dims:
        blocks = _plan_blocks(num_rows, dims.num_experts, block_limit)
        ws_total = _ws_layout(dims.world_size, dims.num_experts, blocks.num_chunks)[
            "total"
        ]
        gm_ws = torch.zeros(ws_total, dtype=torch.int32, device="npu")
        ws_entry = (dims, gm_ws)
        context.plan_workspace = ws_entry
    gm_ws = ws_entry[1]
    # Cnt/Asg 只消费本 rank 的行，直接传 [S,K] 展平视图
    _get_plan_program(
        dims.num_tokens,
        dims.top_k,
        dims.num_experts,
        dims.world_size,
        block_limit,
    )(
        tokens_per_expert,
        gm_flag,
        gm_ws,
        topk_safe,
        gm_io,
        context.ctx_tensor,
    )
    # 跨流护栏 fork 点：只等我方算子，不等主流尾部
    record_transport_event(context, stream=cur_stream)
    # 先冲洗上一轮延迟检查（异步链曾在此静默丢弃溢出信号）；
    # 本轮结果延迟到下一个算子校验，返回路径零同步。
    check_pending_overflow(context)
    context.pending_overflow = (cu_seqlens, num_rows, bad_topk)
    context.last_plan_shape = (num_tokens, top_k)  # 供 prefetch/reduce 的调用顺序守卫
    context.reduce_copy_plan_token += 1
    return MoonepPlan(dst_n, cu_seqlens, experts_to_copy, src_n, vtopk)


# --------------------------------------------------------------------------
# AOT 收集器（离线构建期预编译入口；源码直跑不受影响）
# --------------------------------------------------------------------------
try:
    from cannbotdsl import aot
except ImportError:
    aot = None

if aot is not None and hasattr(aot, "export"):

    @aot.export("moonep_plan")
    def export_moonep_plan():
        """预编译推荐配置（S=8192,K=16,E=224,R=8）。"""
        _get_plan_program(8192, 16, 224, 8)
