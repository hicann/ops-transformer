# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP 的通信上下文与对称 workspace。

MoonepContext 手里攥着本 rank 的 workspace（uint8 NPU 张量）、UbMem 连接和
设备侧上下文字节（StructRef[MoonepCommCtx]，含 connection 地址）。

全组用同一套公式各自分配，区域偏移一致，moonep_layout 是唯一的尺子。
workspace 是本 rank 私有的注册内存，不受 HCCL_BUFFSIZE 约束。
"""

from __future__ import annotations

import threading
from collections import deque
import warnings

import cannbotdsl
import torch

try:
    import torch.distributed as dist
except Exception:
    dist = None

try:
    import torch_npu
except Exception:
    torch_npu = None

from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.types.datastruct import StructRef

# --------------------------------------------------------------------------
# 布局常量统一从 moonep_layout 导出
# --------------------------------------------------------------------------
from moonep_layout import (  # noqa: F401
    BAR_FLAGS_OFF,
    DEM_ALL_OFF,
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    MAX_E,
    MAX_R,
    NOTIFY_STRIDE,
    PLAN_REGION_END,
    READY_OFF,
    ROWS_PER_EXPERT,
    SCALE_ALIGN_BYTES,
    WEIGHT_BYTES,
    WEIGHT_DTYPE,
    grad_slot_off,
    group_slot_rows,
    lcm_slot_stride,
    prefetch_weight_off,
    reduce_region_end,
    reduce_stg_off,
    slot_stride_bytes,
)
from moonep_comm import (
    MoonepCommCtx,
    _unregister_connection,
    comm_barrier_all,
    connect_moonep_comm,
    marshal_ctx,
)

_RESIDENT_BLOCKS = None  # 本机有效 AIV 核数（首次查询后缓存）
_FALLBACK_BLOCKS = 16  # 平台信息不可用时的保守值（小核数机型也不会卡死在同步屏障）


def _device_ready() -> bool:
    """torch 侧是否已选定 NPU 设备（不触发惰性初始化）。

    `_npu_getDeviceWithoutSet()` 只读当前设备号：未选设备返回 -1，且不建
    流、不初始化设备（torch_npu 的 device 上下文管理器就是用它保存现场）。
    """
    if torch_npu is None:
        return False
    return torch_npu._C._npu_getDeviceWithoutSet() >= 0


def _vector_core_num(ready: bool) -> int:
    """AIV 核数；设备未就绪时走不碰设备的静态规格查询。

    `get_platform_info()` 缺省会取 torch 当前流以叠管控核配额；torch_npu
    尚未选设备时，这一步会做设备初始化——卡被占用/不可用时是分钟级重试，
    被误当成"编译慢"。编译期只需要一个 block 上界，静态 SoC 规格足够：
    传 `stream=False` 跳过当前流查询（拿不到 handle 时不叠配额），
    完全不触发 NPU 惰性初始化。
    """
    if ready:
        return int(cannbotdsl.get_platform_info().vector_core_num)
    return int(cannbotdsl.get_platform_info(stream=False).vector_core_num)


def resident_blocks(cap: int = 64) -> int:
    """本机可同时常驻的 AIV block 数（含 grid 级同步屏障 kernel 的硬上限）。

    `global_sync_all()` 要求全部 block 同时常驻：block_dim 超过本机有效
    AIV 核数（例如 56 核机型上排 64 块）时，调度器只能分时复用，grid 级同步屏障永远
    收不齐 → 死循环挂死。核数取 `get_platform_info().vector_core_num`
    （设备已初始化时含流的控核配额）；查询报错直接抛出，只有核数为 0 时
    退保守值而不是猜硬件规格。设备未初始化时只读静态 SoC 规格，不触发
    设备初始化；该结果不缓存，运行期设备就绪后再按流配额复核，保证
    实启块数不超过配额。

    cap 是"核数再多也不加块"的经验上限（块数收益递减）；返回值 ≤ cap。
    进程内只在设备已初始化时缓存，保证同一进程内编译期与运行期的 block
    数一致——二者不一致会让 kernel 的角色分派与实启块数错位（不一致时
    按新配置重新编译一次，不会用旧网格启动）。
    """
    global _RESIDENT_BLOCKS
    if _RESIDENT_BLOCKS is not None:
        return min(cap, _RESIDENT_BLOCKS)
    ready = _device_ready()
    cores = _vector_core_num(ready)
    if cores <= 0:
        warnings.warn(
            f"无法获取 AIV 核数，block 数退保守值 {_FALLBACK_BLOCKS}"
            "（使用保守的并行度）",
            RuntimeWarning,
            stacklevel=2,
        )
        _RESIDENT_BLOCKS = _FALLBACK_BLOCKS
    elif ready:
        _RESIDENT_BLOCKS = cores
    else:
        # 设备未初始化：静态规格不缓存，运行期初始化后再按配额复核
        return min(cap, cores)
    return min(cap, _RESIDENT_BLOCKS)


def finish_op(context, async_finish=False):
    """完成算子的同步收尾。

    数据到达证明由 kernel 内到达槽自旋收齐承担（尾部等待内联在
    各算子 kernel 中），本函数不追加独立 sym_barrier——既有实现里
    "kernel 内 barrier + finish_op 再 barrier" 的双屏障开销已消除。
    同步模式记录发起流事件并检查布局容量状态；异步模式仅返回事件。"""
    if not async_finish:
        record_transport_event(context)  # 跨流护栏 fork 点（只等我方算子）
        check_pending_overflow(context)
        return None
    event = torch.npu.Event()
    event.record(torch.npu.current_stream())
    record_transport_event(context)
    return event


def normalize_group_list(context, tensors, op_name, *, min_groups=2, max_groups=3):
    """将 Tensor、list 或 tuple 归一化为列表，并检查组数、形状和布局元数据。

    返回 ``(tensors, widths)``（组数/组宽定型守卫，
    布局错位的防线上移到首调）。
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if not tensors:
        raise ValueError(f"{op_name} 组列表为空（须 {min_groups}~{max_groups} 组）")
    if not (min_groups <= len(tensors) <= max_groups):
        raise ValueError(
            f"{op_name} 支持 {min_groups} 组或 {max_groups} 组（kernel 入口按组数"
            f"固定），实得 {len(tensors)} 组"
        )
    if context.layout_groups is None:
        context.layout_groups = len(tensors)
    elif context.layout_groups != len(tensors):
        raise RuntimeError(
            f"权重组数不一致（已有 {context.layout_groups}，本次 {len(tensors)}），"
            "会导致发布区/槽区重叠"
        )
    widths = []
    for g, t in enumerate(tensors):
        if t.dim() < 2 or t.shape[0] == 0:
            raise ValueError(
                f"{op_name} 第 {g} 组须为 (行, ...) >= 2 维，实得 {tuple(t.shape)}"
            )
        w = int(t.numel() // t.shape[0])
        if w <= 0:
            raise ValueError(
                f"{op_name} 第 {g} 组行宽须为正（尾维展平宽），实得 {w}"
                f"（张量形状 {tuple(t.shape)}）"
            )
        widths.append(w)
    widths = tuple(widths)
    if context.layout_widths is None:
        context.layout_widths = widths
    elif widths != context.layout_widths:
        raise RuntimeError(
            f"{op_name} 组宽不一致（已有 {context.layout_widths}，本次 {widths}）"
            "——宽度变化会使梯度槽区基址错位、发布/槽区重叠"
        )
    if context.last_plan_shape is None:
        raise RuntimeError(
            f"{op_name} 需要先设置 context.last_plan_shape（输入形状与窗口布局）"
        )
    return tensors, widths


def pick_weight_chunk(w_elems: int, ub_limit_elems: int) -> int:
    """UB 分块粒度：整行单块（W<=ub_limit）或最大 2 的幂因子（保证整除、无尾块）。

    DSL 的 tile_slice 拒绝静态部分宽度尾块与超尺寸 tile，因此分块必须
    满足 W % chunk == 0。W=H*Hp（H=3584=2^9*7）必能命中；W 非 512 倍数且超
    上限时抛 ValueError。
    """
    if w_elems <= ub_limit_elems:
        return w_elems
    chunk = ub_limit_elems
    while chunk > 512 and w_elems % chunk != 0:
        chunk //= 2
    if w_elems % chunk != 0:
        raise ValueError(
            f"W={w_elems} 在 UB 上限 {ub_limit_elems} 下无 2 的幂因子整除方案"
        )
    return chunk


class MoonepContext:
    """EP 组级通信上下文句柄：workspace + UbMem 连接 + 设备侧上下文字节。

    Attributes:
        ctx_tensor: torch.uint8 NPU 张量，MoonepCommCtx 的设备侧字节（传给
            算子的 ``StructRef[MoonepCommCtx]`` 入参）。
        rank: 本 rank 在 EP 组内的 id。
        rank_size: EP 组 rank 数。
        workspace: 本 rank 私有 workspace（uint8 NPU 张量；对称布局约定）。
        connection: cannbotdsl.comm.UbMemConnection（映射内存连接）。
    """

    def __init__(
        self,
        group_name: str,
        *,
        op_name: str = "moonep",
        rank=None,
        rank_size=None,
        workspace_bytes=None,
        group=None,
    ):
        self.group_name = group_name
        self.group = group

        # rank/rank_size 缺省回退 WORLD；EP 为子通信域时由 create_moonep_context(group=...) 显式传入
        self.rank = dist.get_rank() if rank is None else rank
        self.rank_size = dist.get_world_size() if rank_size is None else rank_size
        if self.rank_size > MAX_R:
            # MAX_R 为布局定宽区域上限；引擎侧 COMM_MAX_RANK_NUM 的同级
            # 上限尚未实机核对（>64 卡环境未验证）
            raise ValueError(
                f"moonep 支持 EP 组 rank 数 <= MAX_R={MAX_R}，当前 {self.rank_size}"
            )
        # workspace 是本 rank 私有的注册内存，分配多少就用多少；默认只含
        # plan 交换区下界，权重/梯度区由调用方按形状传进来
        if workspace_bytes is None:
            workspace_bytes = HEAP_BASE + PLAN_REGION_END
        workspace_bytes = int(workspace_bytes)
        if workspace_bytes < HEAP_BASE + PLAN_REGION_END:
            raise ValueError(
                f"workspace {workspace_bytes} B 不足以容纳 plan 交换区 "
                f"（需 >= {HEAP_BASE + PLAN_REGION_END} B）"
            )
        # 清零是硬要求：到达槽会被按 epoch 轮询，回收页里的旧值会冒充“已到达”
        self.workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="npu")
        self.workspace_bytes = workspace_bytes
        # 单 rank 无副本可交换：不建连（算子侧 R==1 早退）
        if self.rank_size >= 2:
            # 建连前先来一次全组 barrier：对端协商要求所有 rank 就绪，
            # 否则会撞上 HcclChannelAcquire 竞态
            if dist.is_available() and dist.is_initialized():
                if self.group is None:
                    dist.barrier()
                else:
                    dist.barrier(group=self.group)
            self.connection, self.ctx_tensor = connect_moonep_comm(
                group_name,
                self.rank,
                self.rank_size,
                self.workspace,
                tag=op_name,
            )
        else:
            self.connection = None
            self.ctx_tensor = marshal_single_rank_ctx(self.rank, self.rank_size)
        # 槽视图已用的 slots 值（一致性守卫——create_grad_slot_view 写入）。
        self.pull_slots_used = None
        self.plan_workspace = None
        self.transport_stream = None
        self.transport_event = None
        self.last_plan_shape = None
        self.reduce_copy_plan_token = 0
        self.layout_groups = None
        self.layout_widths = None
        self.layout_scale_widths = None
        self.weight_dtype = None
        self.grad_dtype = None
        self.window_slots = None
        self.pending_overflow = None
        self.weight_views = {}
        self.bar_epoch_tensors = deque(maxlen=128)
        self.barrier_epoch = 0

    def close(self):
        """断开连接（幂等；workspace/视图随张量引用回收）。"""
        if getattr(self, "connection", None) is not None:
            self.connection.disconnect()
            _unregister_connection(self.connection)
            self.connection = None


def marshal_single_rank_ctx(rank, rank_size):
    """单 rank（R==1）的占位上下文（无连接；算子侧早退不触通信面）。"""

    # connection 地址填 0：kernel 侧 R==1 路径在建连分支之前早退
    return marshal_ctx(rank, rank_size, _NullConnection(), 0)


class _NullConnection:
    """R==1 占位（get_address 返回 0；不进入设备侧使用路径）。"""

    def get_address(self):
        return 0


def workspace_base(context):
    """本 rank workspace 基址（取代旧 hccl_buffer_base；视图/寻址唯一基点）。"""
    return int(context.workspace.data_ptr())


def check_heap_capacity(context, need_bytes, what):
    """workspace 容量前置校验：不足即报所需/现有（沿用旧函数名与语义）。"""
    heap_bytes = context.workspace_bytes
    if need_bytes > heap_bytes:
        raise RuntimeError(
            f"{what}超出 workspace：需要 {need_bytes} B "
            f"({need_bytes / 2**20:.1f} MiB)，可用 {heap_bytes} B "
            f"({heap_bytes / 2**20:.1f} MiB)。请按形状精算显式传入 "
            "workspace_bytes（Buffer._workspace_need() 同式）。"
        )


def check_pending_overflow(context):
    """检查上下文中待验证的布局容量状态；没有待验证状态时直接返回。"""
    pending = context.pending_overflow
    if pending is None:
        return
    context.pending_overflow = None
    cu_seqlens, num_rows, *extra = pending
    cu_tail = int(cu_seqlens[-1].item())
    if cu_tail != num_rows:
        raise RuntimeError(
            f"副本槽溢出：极端路由下单 rank 副本数超过 epn={cu_seqlens.shape[0] // 2}"
            f" 个预留槽（cu_seqlens 尾值={cu_tail} != N={num_rows}）。"
            "请检查专家清单与布局元数据。"
        )
    if extra and bool(extra[0].item()):
        raise RuntimeError(
            "topk 专家 id 越界：要求 0 <= topk_experts_sk < E；"
            "越界项已被 clamp 到合法范围，结果不可用。"
        )


# --------------------------------------------------------------------------
# rank 自旋同步屏障（UbMem 直推 flag + 自旋收齐）
# --------------------------------------------------------------------------
# kernel 返回即本轮工作完成；epoch 单调防止陈旧 flag 提前放行；
# flag 与数据同走 MTE3 并加 fence 保序。
_BAR_EPOCH_MAX = (1 << 31) - 1
_BAR_PROGRAMS = {}
_BAR_LOCK = threading.Lock()


@kernel
class _MoonepBarrierKernel:
    """各 rank 把本轮 epoch 直推全部 peer 的 flag 槽，再自旋收齐（单 block）。"""

    def __init__(self, rank_size):
        self._r = int(rank_size)

    def __call__(self, gm_epoch, ctx: StructRef[MoonepCommCtx]):
        epoch = dtypes.uint32(gm_epoch[dtypes.int64(0)])
        comm_barrier_all(
            ctx,
            self._r,
            BAR_FLAGS_OFF,
            gm_epoch,
            epoch,
        )


def _make_barrier_host(rank_size):
    @host
    def _barrier_host(gm_epoch, ctx: StructRef[MoonepCommCtx]):
        _MoonepBarrierKernel(rank_size)[1](gm_epoch, ctx)

    return _barrier_host


def record_transport_event(context, stream=None):
    """各算子 wrapper 发射后记录完成事件（跨流护栏的 fork 点）。"""
    event = context.transport_event
    if event is None:
        event = context.transport_event = torch.npu.Event()
    event.record(stream if stream is not None else torch.npu.current_stream())


def _get_barrier_program(world_size):
    """world_size 键控的 rank 同步屏障 program（rank 运行期取自 ctx）。"""
    key = world_size
    with _BAR_LOCK:
        prog = _BAR_PROGRAMS.get(key)
        if prog is None:
            prog = cannbotdsl.compile(
                _make_barrier_host(key),
                TensorSpec((1,), dtypes.int32),
                StructRef[MoonepCommCtx],
            )
            _BAR_PROGRAMS[key] = prog
        return prog


def next_barrier_epoch(context):
    """推进本轮 epoch 并创建独立设备张量（1 元素 int32）。"""
    epoch = context.barrier_epoch + 1
    if epoch > _BAR_EPOCH_MAX:
        raise RuntimeError("barrier epoch 已达 int32 上限，请重建 MoonEP 上下文")
    context.barrier_epoch = epoch
    gm_epoch = torch.full((1,), epoch, dtype=torch.int32, device="npu")
    context.bar_epoch_tensors.append(gm_epoch)
    return gm_epoch


def sym_barrier(context, *, sync=True):
    """workspace flag 自旋 barrier：dist.barrier 的流级强化替代。

    单 block kernel 经 UbMem 把 epoch flag 直推全部 peer 的
    flag 槽（comm_barrier_all），自旋收齐后返回——返回即全组本轮此前流
    上工作全部完成（flag 与数据同通道保序）。

    sync=False 为"只发射"形态（跨 rank 保序由 kernel 自身
    的自旋协议保证，host 立即返回）。
    """
    if context.rank_size < 2:
        if sync:
            torch.npu.current_stream().synchronize()
        return
    gm_epoch = next_barrier_epoch(context)
    _get_barrier_program(context.rank_size)(gm_epoch, context.ctx_tensor)
    record_transport_event(context)
    if not sync:
        return
    # 等待范围为发起流而非整卡（避免封死跨层/跨流流水）
    torch.npu.current_stream().synchronize()


def sync_transport_stream(context):
    """跨流安全护栏（算子握手状态机要求调用严格按发起序执行）。"""
    cur = torch.npu.current_stream()
    transport_stream = context.transport_stream
    if transport_stream is None or transport_stream == cur:
        context.transport_stream = cur
        return cur
    if context.transport_event is not None:
        cur.wait_event(context.transport_event)
    context.transport_stream = cur
    return cur


def _hccl_comm_name(group, rank):
    """由 ProcessGroup 解析 HCCL comm name（新 API get_comm_by_name 的名字源）。"""
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def create_moonep_context(
    group_name: str | None = None,
    *,
    group=None,
    workspace_bytes: int | None = None,
) -> MoonepContext:
    """创建 moonep 通信上下文（workspace + UbMem 连接）。

    group_name 缺省时用 torch.distributed 默认组；EP 为子通信域时传 ``group``
    （ProcessGroup），rank/rank_size 与 HCCL comm name 均按组内取值。
    workspace_bytes 缺省 = plan 交换区下界（HEAP_BASE + PLAN_REGION_END，
    分配即所需——不做粗预留）；跑 reduce/prefetch 的裸 API 调用方须按形状
    精算显式传入（``Buffer._workspace_need()``；Buffer 路径
    构造期已自动注入）。
    裸 API 的形状在首调算子时才确定，按上界预留；Buffer 构造路径
    按形状精算注入。
    """
    if group is not None:
        rank = dist.get_rank(group)
        rank_size = dist.get_world_size(group)
        if group_name is None:
            group_name = _hccl_comm_name(group, rank)
        ctx = MoonepContext(
            group_name,
            rank=rank,
            rank_size=rank_size,
            workspace_bytes=workspace_bytes,
            group=group,
        )
        check_heap_capacity(ctx, HEAP_BASE + PLAN_REGION_END, "plan 交换区")
        return ctx
    if group_name is None:
        group_name = _hccl_comm_name(dist.group.WORLD, dist.get_rank())
    ctx = MoonepContext(group_name, workspace_bytes=workspace_bytes)
    check_heap_capacity(ctx, HEAP_BASE + PLAN_REGION_END, "plan 交换区")
    return ctx
