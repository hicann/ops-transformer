# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MoonEP workspace 的图纸：区域常量与偏移公式（纯计算，不碰设备）。

布局常量、分区基址和步长都写在这里；kernel 编译期、host 侧视图和容量
计算共用这套公式，全组算出来的偏移自然一致。moonep_context 原样重导出。
"""

from __future__ import annotations

import math

from cannbotdsl import dtypes

# 堆（workspace）内布局（字节偏移，均以 heap_base 为基址，32B 对齐）：
#   [0, HEAP_BASE)               到达槽区（per-sender epoch 哨兵，见 notify_slot_off）
#   [HEAP_BASE, PLAN_REGION_END) 元数据与同步标志区
#   [prefetch_weight_off(), ...) 权重视图（每组包含本地专家行与预取槽）
#   [reduce_stg_off(...), ...)   发布区之后的工作缓冲区
# 同步标志与工作缓冲区使用各自的固定基址。
# 各区域基址由下列函数按运行时形状计算，所有 rank 公式一致 → 结果一致。
NOTIFY_STRIDE = 64  # per-sender 到达槽步长（64B=1 cache line，950PR 定值）
MAX_E = 1024  # v1 支持的专家总数上限
MAX_R = 128  # 组内 rank 上限（128 卡适配；下列定宽区域随本常量放大）
HEAP_BASE = max(
    4096, MAX_R * NOTIFY_STRIDE
)  # 到达槽区上界：MAX_R 个 sender 槽不得侵入堆区
DEM_ALL_OFF = 0  # (MAX_R, MAX_E) int32：各 rank 本地计数交换区（512KB@R=128）
READY_OFF = MAX_R * MAX_E * 4  # (MAX_R,) int32：元数据就绪标志
BAR_FLAGS_OFF = READY_OFF + MAX_R * 4  # (MAX_R,) int32：rank 自旋同步屏障的 epoch flag
LAYOUT_TAIL_BYTES = 4096  # 区域尾部对齐余量（32B 对齐要求下取整页）
# prefetch 行分片完成标志（本 rank 本地，按 (peer, slice) 索引）：
# 每 peer 至多一个常驻分片，占用尾部前 256B；slice-0 等全部分片置位后
# 再给对端发 flag。
PREFETCH_SLICE_FLAGS_OFF = BAR_FLAGS_OFF + MAX_R * 4
PLAN_REGION_END = PREFETCH_SLICE_FLAGS_OFF + LAYOUT_TAIL_BYTES
WEIGHT_DTYPE = dtypes.bfloat16  # 权重发布区缺省 bf16（按构造 weight_dtype 参数化）
WEIGHT_BYTES = WEIGHT_DTYPE.bytes_for(1)  # 权重要素字节数缺省（bf16 = 2；uint8 = 1）
ROWS_PER_EXPERT = 2  # 每专家两行：前 epn 行真实权重，后 epn 行预取槽
# scale 行宽对齐（字节）：mxfp8 的 ue8m0 block scale 每行须为 32B 倍数
# （MVP-A 定档假设——不足时调用方 pad 到 ceil(Sg/32)*32，消费端按 padded 宽解）
SCALE_ALIGN_BYTES = 32


def prefetch_weight_off() -> int:
    """prefetch 权重发布区基址（紧随 plan 交换区）。"""
    return PLAN_REGION_END


def lcm_slot_stride(epn: int, row_bytes) -> int:
    """行宽列表（字节）→ LCM 对齐的 slot 步长（lcm_slot_stride 唯一真源）。

    kernel 编译期（sizes×elem_bytes 折成行字节列表）与 host 视图/容量
    公式（widths/scale_widths 折行字节）都收敛到这里——两侧行字节列表
    相同则步长逐字节一致。
    """
    row_bytes = [int(row_bytes_i) for row_bytes_i in row_bytes]
    if not row_bytes or any(row_bytes_i <= 0 for row_bytes_i in row_bytes):
        # 防御深度：0 行宽会使 LCM 归零后除零（上游 E3/normalize_group_list
        # 已友好拦截，这里兜底把 ZeroDivisionError 变为指向根因的报错）
        raise ValueError(f"行宽列表须非空且为正，实得 {row_bytes}")
    raw = sum(row_bytes) * ROWS_PER_EXPERT * int(epn)
    lcm = row_bytes[0]
    for row_bytes_i in row_bytes[1:]:
        lcm = lcm * row_bytes_i // math.gcd(lcm, row_bytes_i)
    return (raw + lcm - 1) // lcm * lcm


def slot_stride_bytes(epn: int, widths, scale_widths, weight_bytes: int) -> int:
    """单 slot 窗口步长（字节）——LCM 对齐（kernel/host 共用唯一公式）。

    原始步长 = (ΣWg×weight_bytes + ΣSg) × 2×epn；但 push kernel 的槽寻址
    ``dst = group_rows[g] + slot × slot_rows[g] + slot_row`` 以"本组一行字节宽"
    为单位（tile_slice 坐标 = 瓦片序号），要求步长被每个组的行字节数
    整除。scale 组加入后原始步长一般不再满足（如交织 (1024,512)+scale
    (32,16)：原始 12672B 不被 1024 整除）——向上取整到全部行宽的最小公
    倍数。代价 ≤ lcm-1 字节/slot；kernel（编译期 __init__）与本侧
    （视图/容量公式）同式调用，逐字节一致（不一致即寻址错位）。
    """
    row_bytes = [int(w) * int(weight_bytes) for w in widths]
    row_bytes += [int(s) for s in (scale_widths or ())]
    return lcm_slot_stride(epn, row_bytes)


def group_slot_rows(epn: int, row_bytes):
    """校验组前缀和可整除，返回每组起始行号与 slot 行步长。

    kernel 按“组起始行 + slot × 行步长 + 槽内行号”寻址，要求每组前缀和
    和 slot 步长都能被该组行字节数整除；不满足时在这里提前报错。
    """
    row_bytes = tuple(int(row_bytes_i) for row_bytes_i in row_bytes)
    slot_bytes = lcm_slot_stride(epn, row_bytes)
    group_rows, slot_rows = [], []
    base_bytes = 0
    for g, row_bytes_i in enumerate(row_bytes):
        if base_bytes % row_bytes_i or slot_bytes % row_bytes_i:
            raise ValueError(
                f"第 {g} 组行宽 {row_bytes_i}B 无法与前缀 {base_bytes}B / "
                f"slot 步长 {slot_bytes}B 对齐；请调整该组行宽或 epn"
            )
        group_rows.append(base_bytes // row_bytes_i)
        slot_rows.append(slot_bytes // row_bytes_i)
        base_bytes += row_bytes_i * ROWS_PER_EXPERT * int(epn)
    return tuple(group_rows), tuple(slot_rows)


def reduce_stg_off(
    epn: int,
    w_elems: int = 0,
    num_groups: int = 1,
    slots: int = 1,
    *,
    widths=None,
    scale_widths=None,
    weight_bytes: int = WEIGHT_BYTES,
) -> int:
    """权重发布区后的工作缓冲区基址（发布区按 slots×组数预留）。

    每组发布区是完整的 [2*epn, W] 权重视图（真实行 + 预取槽同址），
    组步长必须按整视图 2*epn*W*elem 字节计——按 epn*W 计会使组间重叠一半
    （组 g+1 起点压在组 g 的预取槽区正中）。slots 为多槽窗口池层数
    （槽 s、组 g 的视图基址 = prefetch_weight_off + 槽内 sizes 前缀和）：
    双槽轮转使 GEMM 可直读当前槽视图，下一层 prefetch 写另一槽，免快照拷贝。

    等宽组传 ``w_elems``/``num_groups``；变宽组（如交织型 W13 宽 2W、
    down 宽 W）传 ``widths=(2W, W)``，总宽按各组件宽求和——交织两组与
    等宽三组字节数相同（2W + W = 3W）。``weight_bytes`` 为每组
    权重要素字节宽（bf16 2B / fp8 1B，进编译键）；``scale_widths`` 为
    各 scale 组行宽（字节——ue8m0 定宽 1B，mxfp8 形态传入；scale 组
    跟随同 slot 窗口的权重组之后前缀和排布，None = 无 scale 组）。
    """
    if widths is None:
        widths = (int(w_elems),) * int(num_groups)
    return prefetch_weight_off() + slots * slot_stride_bytes(
        epn, widths, scale_widths, weight_bytes
    )


def grad_slot_off(
    epn: int,
    w_elems: int = 0,
    num_groups: int = 1,
    slots: int = 1,
    *,
    widths=None,
    scale_widths=None,
    weight_bytes: int = WEIGHT_BYTES,
) -> int:
    """pull 协议梯度副本区基址（紧随权重发布区，变宽组/scale 组同 reduce_stg_off）。

    梯度区为 [epn, Wg] × 组副本发布区——每 rank 只发布自己的
    副本行，owner 经远读拉取归并（无 R 维度、无并发写）；区域大小见
    reduce_region_end。
    """
    return reduce_stg_off(
        epn,
        w_elems,
        num_groups,
        slots,
        widths=widths,
        scale_widths=scale_widths,
        weight_bytes=weight_bytes,
    )


def reduce_region_end(
    epn: int,
    w_elems: int,
    num_groups: int = 1,
    slots: int = 1,
    grad_bytes_per_elem: int = 2,
    *,
    widths=None,
    scale_widths=None,
    weight_bytes: int = WEIGHT_BYTES,
) -> int:
    """堆上界（含梯度副本区；变宽组传 widths）。

    - 副本发布区 [epn, Wg] × 组——每 rank 只发布自己的副本行供
      owner 远读；无 R 维度，显存恒定（19.3MB 级）
    - push 形态接收区 [R, epn, Wg] 随 R 线性（32 卡
      617MB），为 pull 选型的动因
    ``scale_widths`` 只扩大权重发布区（scale 随 slot 轮转），不影响梯度区。
    """
    total_w = (
        sum(int(w) for w in widths)
        if widths is not None
        else int(w_elems) * int(num_groups)
    )
    slot_off = grad_slot_off(
        epn,
        w_elems,
        num_groups,
        slots,
        widths=widths,
        scale_widths=scale_widths,
        weight_bytes=weight_bytes,
    )
    return slot_off + total_w * epn * grad_bytes_per_elem + LAYOUT_TAIL_BYTES
