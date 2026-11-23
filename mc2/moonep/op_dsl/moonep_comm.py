# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""moonep_comm：MoonEP 跨 rank 数据面的搬运工（UbMem 映射内存）。

本文件负责三件事：host 侧把各 rank 的 workspace 接成一张可达的网、设备侧
把字节从本端搬到对端、以及用 epoch 哨兵告诉对端“这批数据到了”。

- Host 控制面（`connect_moonep_comm`）：每个 rank 各自分配并注册自己的
  workspace（`MemoryDesc`），全组用同一套布局公式，偏移自然对齐。建连时
  优先挑一条能直达所有 peer 的最低层；没有这样的层就逐 peer 诊断，做不到
  就直说 —— 不退而求其次。
- 设备侧搬运（`ub_write`）：GM 到 GM 走不通（`mem_copy` 不支持远端目的），
  所以绕一趟 UB：GM → UB staging 环 → 对端 mapped GM，按字节分块。
- 到达信号：每 (peer, 轮) 最后补一笔 4B epoch 写到对端到达槽（fence +
  MTE3 数据通道）—— 哨兵到了，说明同通道之前的数据都已落地。接收侧用
  `spin_u32_ge` 自旋等本端到达槽 ≥ epoch。

布局约定（workspace 相对偏移，公式唯一真源 moonep_layout）：

    到达槽（per sender rank）：offset(r) = r * NOTIFY_STRIDE
    值 = 本轮 epoch（u32）；epoch 跨算子单调（moonep_context 维护）。
"""

from __future__ import annotations

import atexit
import ctypes
import threading

import cannbotdsl as cbd
import torch
from cannbotdsl import Constexpr, comm, datastruct, dtypes
from cannbotdsl._mlir.dialects import cannbotir
from cannbotdsl.buffer import Buffer
from cannbotdsl.core.ir_transport import unwrap_operand
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc, make_layout, make_pointer, make_tensor, tile_slice
from cannbotdsl.types._numeric import UInt64
from cannbotdsl.types.datastruct import _marshal_datastruct
# --------------------------------------------------------------------------
# 到达槽布局（workspace 内偏移）
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 设备上下文：host 侧打包，kernel 侧按 StructRef 读取
# --------------------------------------------------------------------------
@datastruct(c_name="MoonepCommCtx")
class MoonepCommCtx:
    """设备侧通信上下文（rank/连接地址/本端 workspace 基址）。

    Host 持有 connection 与 workspace 的生命周期；kernel 侧仅持借用视图
    （DeviceCommContext 契约）。
    """

    rankId: dtypes.uint64
    rankSize: dtypes.uint64
    connection: dtypes.uint64  # UbMem 单连接地址
    localWsAddr: dtypes.uint64  # 本端 workspace 基址（到达槽/flags 轮询寻址）


def marshal_ctx(rank_id, rank_size, connection, ws_addr):
    """构造设备上下文字节 → uint8 NPU 张量（kernel 的 StructRef 实参）。"""
    value = MoonepCommCtx(
        rankId=rank_id,
        rankSize=rank_size,
        connection=connection.get_address(),
        localWsAddr=ws_addr,
    )
    # 该 schema 字段由 @datastruct 自动生成，是稳定的序列化入口
    packed = _marshal_datastruct(value, MoonepCommCtx.__cannbotdsl_datastruct_schema__)
    payload = ctypes.string_at(ctypes.addressof(packed), ctypes.sizeof(packed))
    return torch.tensor(list(payload), dtype=torch.uint8).npu()


# --------------------------------------------------------------------------
# UbMem staging 环：两跳搬运的 UB 中转
# --------------------------------------------------------------------------
UB_XFER_CHUNK = 32768  # 分块（字节）——行 224KB→7 块；小行单块
# depth=3 时在飞 96KB，正卡在 UB 预算内；再深就编译不过
UB_XFER_DEPTH = 3


def make_xfer_rb(depth=UB_XFER_DEPTH):
    """UbMem 字节 staging 环（kernel __call__ 期分配，逐笔 next_write）。

    depth 由调用方按其 UB 预算选（如 prefetch 用深环提高 MTE 在飞量；
    reduce 的 UB 被 ub_row/ub_acc 大头占据，维持缺省浅环）。
    """
    return RingBuffer(
        MemLoc.UB, dtypes.uint8, make_layout((UB_XFER_CHUNK,)), depth=depth
    )


def _gm_view(address, count, dtype):
    """绝对地址（u64 运行期值）→ GM Tensor（无分配无拷贝）。"""
    return make_tensor(
        make_pointer(dtype, dtypes.int64(address), MemLoc.GM), make_layout((count,))
    )


def tensor_address(tensor):
    """Tensor → u64 GM 地址（提取 signless IR 后显式补 unsigned 语义）。"""
    return UInt64(
        cannbotir.extract_address(cannbotir.extract_pointer(unwrap_operand(tensor)))
    )


# --------------------------------------------------------------------------
# 设备原语（@jit）；循环携带的状态由调用方用变量传进来
# --------------------------------------------------------------------------
@jit
def spin_u32_ge(ctx, offset, epoch):
    """自旋等待本端 workspace[offset] 追平 epoch（scalar bypass 轮询）。

    offset 为 workspace 相对偏移（到达槽/flags 槽，运行期值）；epoch 为
    运行期 uint32 标量。load_bypass（DCache 旁路）读 + while 轮询；
    vec_busy_wait 是自旋降频指令（避免纯死循环 100% 占核），不是功能重复
    ——删去则编译器可能消除轮询副作用。
    用有符号差值判断是否到达，epoch 跨 2^32 回绕时仍然成立。
    """
    view = _gm_view(
        dtypes.uint64(ctx.localWsAddr) + dtypes.uint64(offset), 1, dtypes.uint32
    )
    v = dtypes.uint32(cbd.scalar.load_bypass(view.ptr(0)))
    while dtypes.int32(v - epoch) < 0:
        cbd.vec_busy_wait(view[0], v, cmp=cbd.CmpOp.EQ)
        v = dtypes.uint32(cbd.scalar.load_bypass(view.ptr(0)))
    return v


@jit
def ub_write(peer, remote_off, src_addr, nbytes, xfer_rb):
    """UbMem 两跳写：GM(src)→UB(staging 环)→对端 mapped GM（字节粒度）。

    nbytes 为编译期常量——整块/尾块编译期展开（tile 计数静态）。当前
    mem_copy 不支持 GM→GM（含远端 dst），故以 UB 环中转；
    depth=2 使 MTE2（GM→UB）与 MTE3（UB→远端）流水。数据落地的完成由
    调用方 fence（``_ub_fence``）或末笔 flag 写（``ub_flag_write``）收敛。
    """
    mapped = dtypes.uint64(peer.get_mapped_addr(0))
    if const_expr(nbytes <= UB_XFER_CHUNK):
        stage = xfer_rb.next_write()
        cbd.mem_copy(
            tile_slice(stage, (nbytes,), (0,)), _gm_view(src_addr, nbytes, dtypes.uint8)
        )
        cbd.mem_copy(
            _gm_view(mapped + dtypes.uint64(remote_off), nbytes, dtypes.uint8),
            tile_slice(stage, (nbytes,), (0,)),
        )
    else:
        full = int(nbytes) // UB_XFER_CHUNK
        tail = int(nbytes) % UB_XFER_CHUNK
        # 运行时循环：IR 恒为一个 loop body（RingBuffer 游标与地址均为
        # 运行期值；小行/大行统一走此路径，避免按块展开导致 IR 随行宽爆炸）。
        for c in cbd.range(full):
            stage = xfer_rb.next_write()
            off = c * dtypes.int64(UB_XFER_CHUNK)
            cbd.mem_copy(
                tile_slice(stage, (UB_XFER_CHUNK,), (0,)),
                _gm_view(src_addr + dtypes.uint64(off), UB_XFER_CHUNK, dtypes.uint8),
            )
            cbd.mem_copy(
                _gm_view(
                    mapped + dtypes.uint64(remote_off) + dtypes.uint64(off),
                    UB_XFER_CHUNK,
                    dtypes.uint8,
                ),
                tile_slice(stage, (UB_XFER_CHUNK,), (0,)),
            )
        if const_expr(tail > 0):
            stage = xfer_rb.next_write()
            off = full * UB_XFER_CHUNK
            cbd.mem_copy(
                tile_slice(stage, (tail,), (0,)),
                _gm_view(src_addr + dtypes.uint64(off), tail, dtypes.uint8),
            )
            cbd.mem_copy(
                _gm_view(
                    mapped + dtypes.uint64(remote_off) + dtypes.uint64(off),
                    tail,
                    dtypes.uint8,
                ),
                tile_slice(stage, (tail,), (0,)),
            )


@jit
def _ub_fence():
    """管同步 + 数据栅栏（MTE 写落地收敛——flag 写前/flush 用）。"""
    cbd.vec_sync_pipe(cbd.PIPE.ALL)
    cbd.scalar.data_barrier()


@jit
def ub_flag_write_snapshot(peer, slot_off, epoch_ub):
    """4B epoch 快照写对端槽——值取 kernel 起始的一次性 UB 快照。

    与 ``ub_flag_write`` 同通道保序（首尾 fence + MTE3 数据通道：flag
    落地即此前全部行已到对端）。快照源保证同一 kernel 内的自旋阈值、
    本地完成标志与远端写值始终同代。
    """
    mapped = dtypes.uint64(peer.get_mapped_addr(0))
    _ub_fence()
    cbd.mem_copy(_gm_view(mapped + dtypes.uint64(slot_off), 1, dtypes.uint32), epoch_ub)
    _ub_fence()


@jit
def ub_flag_write(peer, slot_off, src_addr):
    """4B epoch 写对端槽——经 MTE3 数据通道（与数据写同 pipe 保序）。

    （combine 数值错位根因）：scalar bypass 写与 MTE3 批量写走
    不同通道——64 行批量下 flag 先落地、数据在途，接收侧 kernel 返回后
    torch/MTE 读窗仍见部分陈旧行。改经 UB 暂存 + mem_copy（MTE3）写
    flag——同 pipe 内 MTE3 写按序完成，flag 到达即此前数据全部落地
    （probe_ubmem_full 单笔形态原已验证同通道序，本修把批量形态对齐
    到同一结论）。首尾 fence：数据写落地 + flag 全通道可见。
    """
    mapped = dtypes.uint64(peer.get_mapped_addr(0))
    _ub_fence()
    # 值经 GM 源（epoch 张量视图）→ UB → MTE3 写对端——全程数据通道，
    # 免 UB 标量运行期赋值（@jit 不支持动态标量写 UB 元素）
    ub_flag = Buffer(MemLoc.UB, dtypes.uint32, make_layout((1,)))
    cbd.mem_copy(ub_flag, _gm_view(src_addr, 1, dtypes.uint32))
    cbd.mem_copy(_gm_view(mapped + dtypes.uint64(slot_off), 1, dtypes.uint32), ub_flag)
    _ub_fence()


@jit
def connection_from_address(addr):
    """构造 UbMem 连接借用视图（kernel 侧从 ctx.connection 取地址后解析）。"""
    return comm.UbMemConnection.from_address(addr)


@jit
def xfer_open(ctx, peer_idx):
    """解析 peer 传输句柄（ctx.connection 上 get_peer(peer_idx)）。

    返回 peer handle（``peer.get_mapped_addr`` 直接寻址对端 workspace）。
    """
    conn = comm.UbMemConnection.from_address(dtypes.uint64(ctx.connection))
    return conn.get_peer(peer_idx)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
@jit
def comm_barrier_all(
    ctx,
    rank_size: Constexpr[int],
    flags_off,
    gm_epoch,
    epoch,
):
    """向全部 peer 直推本轮 epoch flag 并自旋收齐（单 block 调用）。

    flags 区：workspace 相对偏移 flags_off 起的 (MAX_R,) int32 槽（旧
    BAR_FLAGS_OFF 布局同款，flag 槽按 sender rank 索引）；flag 数据取
    gm_epoch（[1] int32 张量，host 侧按 epoch 填好）。

    rank_size 为编译期常量，逐 peer range_constexpr 展开：每笔 flag 走
    ub_flag_write（经 MTE3 数据通道，首尾 fence）。返回即：本轮全部
    flag 已落地对端且全部对端 flag 已收齐——头部屏障语义（对端 kernel
    启动即其流序在前的 host 写入已完成）。
    """
    my_rank_u64 = dtypes.uint64(ctx.rankId)
    src_addr = tensor_address(gm_epoch)
    for p in range_constexpr(rank_size):
        if dtypes.uint64(p) != my_rank_u64:
            peer_idx = dtypes.uint32(dtypes.uint64(p))
            if dtypes.uint64(p) > my_rank_u64:
                peer_idx = peer_idx - dtypes.uint32(1)
            peer = xfer_open(ctx, peer_idx)
            ub_flag_write(
                peer,
                dtypes.uint64(flags_off) + my_rank_u64 * dtypes.uint64(4),
                src_addr,
            )
    for p in range_constexpr(rank_size):
        if dtypes.uint64(p) != my_rank_u64:
            spin_u32_ge(
                ctx,
                dtypes.uint64(flags_off) + dtypes.uint64(p) * dtypes.uint64(4),
                epoch,
            )


# --------------------------------------------------------------------------
# Host 控制面：建连（layer 选择 + 校验 + 上下文字节）
# --------------------------------------------------------------------------
_CONNECT_LOCK = threading.Lock()
_REG_LOCK = threading.Lock()
_TAG_SEQ = 0  # 同通信域内连接 tag 唯一化（对称进程序列 ⇒ 全 rank 同步推进）

# 活连接注册表：退出前按逆序断连，免得 GC 乱序回收原生对象导致
# 退出期 SIGSEGV，或残留通信域拖住同卡的下一个进程
_LIVE_CONNECTIONS = []


def _register_connection(conn):
    with _REG_LOCK:
        _LIVE_CONNECTIONS.append(conn)
    return conn


def _unregister_connection(conn):
    with _REG_LOCK:
        try:
            _LIVE_CONNECTIONS.remove(conn)
        except ValueError:
            pass


@atexit.register
def _disconnect_all():
    with _REG_LOCK:
        conns = list(reversed(_LIVE_CONNECTIONS))
        _LIVE_CONNECTIONS.clear()
    for conn in conns:
        try:
            conn.disconnect()
        except Exception:  # noqa: BLE001
            pass


class _TopoNative:
    """x86 面 adapter：native handle 上的 layer 级协议表 + 逐对
    check_reachable（x86 whl 的 Python 层移除旧方法且 get_peer_protocols
    委托的 native 符号缺失——native handle 持完整旧方法面，混装可用）。

    ``handle`` 为 native handle（``domain._native``）；``domain`` 提供
    ``get_rank_id``。
    """

    def __init__(self, domain, handle):
        self._domain = domain
        self._handle = handle

    def protocols(self, layer, peer):
        return [_proto_value(x) for x in self._handle.get_protocols(layer)]

    def reachable(self, layer, peer, proto):
        my_rank = self._domain.get_rank_id()
        return bool(self._handle.check_reachable(my_rank, peer, layer, proto))


class _TopoPeerProto:
    """aarch64 面 adapter：``get_peer_protocols(layer, dst)`` 逐对协议表。

    A5_8P 唯一可用查询面——可达性 = 目标协议出现在
    该 (layer, peer) 的链路协议列表中（无独立 check_reachable 查询）。
    """

    def __init__(self, handle):
        self._handle = handle

    def protocols(self, layer, peer):
        return [_proto_value(x) for x in self._handle.get_peer_protocols(layer, peer)]

    def reachable(self, layer, peer, proto):
        return proto in self.protocols(layer, peer)


def _topo_surface(domain):
    """拓扑查询 API 面解析（两种形态）。

    x86 形态：native handle 持完整旧方法面（get_protocols /
    check_reachable）——Python 层的 get_peer_protocols 委托符号缺失。
    aarch64 形态：两面旧方法均缺（native 亦无 get_protocols），
    py-new ``get_peer_protocols(layer, dst)`` 直接给出 (my_rank, dst) 逐对
    协议表——A5_8P 建连/查询可用。

    返回统一 adapter（``protocols(layer, peer)`` / ``reachable(layer,
    peer, proto)``，proto 为数值）。
    """
    native = getattr(domain, "_native", None)
    if native is not None and hasattr(native, "get_protocols"):
        return _TopoNative(domain, native)
    if hasattr(domain, "get_peer_protocols"):
        return _TopoPeerProto(domain)
    raise comm.CommError(
        "通信域无可用拓扑查询面（get_protocols/get_peer_protocols）："
        "native 与 Python handle 均缺失",
    )


def _proto_value(x):
    """协议对象 → native 数值（x86 native 面可返回裸 int）。"""
    if hasattr(x, "name"):
        return int(getattr(comm._native(), x.name))
    return x


def _full_coverage_layer(domain, peers, proto_enum):
    """全部 peer 均可达目标协议的最低 layer（不存在则 None）。

    kernel 侧紧凑序换算假设单条连接承载全部 peers（peer 索引 = 全组
    升序去掉本 rank 后的位置），因此建连应优先选「单层全覆盖」。16 卡
    UB 拓扑中 layer 0 仅覆盖同 8 卡组、layer 1 覆盖全部 peers——旧
    「逐 peer 取首个含协议层」贪心会拆成 layer 0/1 两组并被误判为
    「分层多组」；本函数直接选出 layer 1 单连接。≤8 卡仍返回 layer 0，
    原有行为不变。
    """
    surface = _topo_surface(domain)
    layers = domain.get_layers()
    if not layers:
        raise ValueError("通信域无可用网络层级")
    proto = _proto_value(proto_enum)
    ordered = sorted(peers)
    for layer in layers:
        if all(
            proto in surface.protocols(layer, p) and surface.reachable(layer, p, proto)
            for p in ordered
        ):
            return layer
    return None


def _group_peers_by_layer(domain, peers, proto_enum):
    """逐 peer 分组：{layer: [peers]}——任意拓扑建链的数据源。

    对每个 peer，在层序中找首个含目标协议且 (my_rank, p) 可达的层。
    同层全互联（4/8 卡）自然聚成单组；跨层级/supernode/异构组网按
    peer 落入不同层组——每组建一条连接（connect 的 peers 序保持升序，
    peer 索引 = 组内位置）。不可达 peer 记 None（调用方决定处理）。
    """
    surface = _topo_surface(domain)
    layers = domain.get_layers()
    if not layers:
        raise ValueError("通信域无可用网络层级")
    proto = _proto_value(proto_enum)
    groups = {}
    for p in sorted(peers):
        for layer in layers:
            if proto not in surface.protocols(layer, p):
                continue
            if surface.reachable(layer, p, proto):
                groups.setdefault(layer, []).append(p)
                break
        else:
            groups.setdefault(None, []).append(p)
    return groups


def connect_moonep_comm(group_name, rank, rank_size, workspace, tag="moonep"):
    """注册 workspace 并建连（仅 UbMem），返回 (connection, ctx_tensor)。

    - workspace：torch uint8 NPU 张量（调用方持有；对称布局约定——全组
      以相同公式各自分配，区域偏移一致）。
    - UbMem（映射内存）：优先选全部 peer 均可达的最低层，以单连接承载
      全部 peers（kernel 紧凑序契约）；不存在全覆盖层时，部分可达或
      跨层多组直接拒绝（多层混合形态留二期）。
    - 通信域分离不变量自然消解：workspace 为本 rank 私有注册内存。
    """

    if not isinstance(workspace, torch.Tensor):
        raise TypeError(f"workspace 须为 torch.Tensor，got {type(workspace).__name__}")
    if workspace.dtype != torch.uint8:
        raise TypeError("workspace 须为 uint8 NPU 张量（布局公式以字节计）")
    ws_addr = int(workspace.data_ptr())
    ws_bytes = int(workspace.numel())
    if ws_addr <= 0 or ws_bytes <= 0:
        raise ValueError("workspace 须为有效 NPU 张量")
    if rank_size < 2:
        raise ValueError("rank_size >= 2 才需要建连（单 rank 无通信）")
    domain = comm.get_comm_by_name(group_name)
    if domain.get_rank_size() != rank_size:
        raise ValueError(f"通信域 rank size {domain.get_rank_size()} != {rank_size}")
    peers = [r for r in range(rank_size) if r != rank]
    memories = [comm.MemoryDesc(ws_addr, ws_bytes)]
    global _TAG_SEQ
    connection = None  # 建连真源（None = 尚未连通）
    with _CONNECT_LOCK:
        # tag 追加进程级序号：同通信域多 context（测试池化等）防重名——
        # 全 rank 以同序构造（集体纪律）⇒ 序号一致；先占用序号再建连，
        # 失败重试不会复用旧 tag
        unique_tag = f"{tag}-{_TAG_SEQ}"
        _TAG_SEQ += 1
        # 建链优先挑能直达所有 peer 的那一层：16 卡的 UB 拓扑里 layer 0
        # 只够同组 8 卡，layer 1 才全覆盖；≤8 卡仍然用 layer 0
        coverage_layer = _full_coverage_layer(domain, peers, comm.UbMemProtocol.UB_MEM)
        if coverage_layer is not None:
            connection = domain.connect(
                peers=peers,
                layer=coverage_layer,
                transport=comm.UbMem(),
                memories=memories,
                tag=unique_tag,
            )
            _register_connection(connection)
        else:
            # 没有全覆盖层就按 peer 分组诊断；这套单连接结构承载不了
            # 多组连接，直接拒绝
            ub_groups = _group_peers_by_layer(domain, peers, comm.UbMemProtocol.UB_MEM)
            fallback = ub_groups.pop(None, [])
            if fallback:
                missing = sorted(fallback)
                raise comm.CommError(
                    f"UbMem 不可达 {len(fallback)}/{len(peers)} peers"
                    f"（不可达：{missing}）——部分可达/多层分组形态当前不支持"
                )
            if len(ub_groups) == 1:
                layer, plist = next(iter(ub_groups.items()))
                if len(plist) < len(peers):
                    # 部分可达：kernel 紧凑序换算假设全 peers——直接拒绝
                    raise comm.CommError(
                        f"UbMem 组仅覆盖 {len(plist)}/{len(peers)} peers"
                        "——部分可达形态当前不支持",
                    )
                connection = domain.connect(
                    peers=plist,
                    layer=layer,
                    transport=comm.UbMem(),
                    memories=memories,
                    tag=unique_tag,
                )
                _register_connection(connection)
            elif len(ub_groups) > 1:
                # 多层 UbMem 且无全覆盖层：单连接 ctx 结构不支持多组
                raise comm.CommError(
                    f"UbMem 分层多组（{len(ub_groups)} 组，且无单层全覆盖）"
                )
            else:
                # 拓扑查询返回空时先把 layer 0 连上试试；失败就报错，不做回退
                try:
                    connection = domain.connect(
                        peers=peers,
                        layer=0,
                        transport=comm.UbMem(),
                        memories=memories,
                        tag=unique_tag,
                    )
                    _register_connection(connection)
                except comm.CommError:
                    pass  # 试连失败——落到下方统一报错
        if connection is None:
            raise comm.CommError(
                "无法建立 UbMem 通信连接——检查 CANN 环境与拓扑"
                "（native 查询面全空且 layer-0 试连失败）",
            )
    ctx_tensor = marshal_ctx(rank, rank_size, connection, ws_addr)
    return connection, ctx_tensor
