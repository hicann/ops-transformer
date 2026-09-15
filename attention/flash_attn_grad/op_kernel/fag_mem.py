# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""片上内存布局与 buffer-id 分配。

pypto 的 tile 地址与 mutex id 都必须是 trace 期常量，所以布局在
**import 期**一次算完，导出平表给 :mod:`fag_buffers`。

**一块 buffer 只声明一次。** ``arena.buf(名字, 字节数, 槽位数)`` 同时给出地址
和 buffer-id —— arena 持有本侧的 :class:`_MutexPool`，两者不可能对不上，也不
存在「在 A 处列一遍名单、在 B 处再列一遍」。

**mutex id 不由人指定。** 编译器只按「两块 buffer 的 id 是否相交」做分组
（``_call_parser._group_refs_by_mutex_overlap`` 的并查集），不看数值大小，也不
要求一块 buffer 的号连续，所以任何单射编号都等价，手工挑号只会挑出撞号。

本模块是地址与 id 的**唯一出处**：kernel 主体不再出现裸地址或裸 id。

读这个文件的三种方式：

* 想知道某块 buffer 放在哪 —— 看 ``_build_*``，它就是布局说明书；
* 想看完整内存图（地址 + 大小 + 锁）—— :func:`describe_layout`；
* 想知道 id 被谁占了、还剩几个 —— :func:`audit_mutex`。

::

    cd attention/flash_attn_grad/op_kernel
    python -c "import fag_mem; print(fag_mem.describe_layout(192))"
    python -c "import fag_mem; print(fag_mem.audit_mutex(192))"

容量由 ``_Arena`` 守住（L1 512KB / L0A·L0B 各 64KB / L0C 256KB / UB 256KB），
id 上限由 ``_MutexPool`` 守住（每核 32 个），越界都会在 import 期失败。
"""

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_MAXW,
    D_NCHUNKS,
    D_PHYS_W,
    HEAD_DIM_ALIGN,
    NZP_ROWS,
    VECTOR_BASEM,
    VECTOR_BASEN,
)
# --- end generated imports ---


# d_align 的三个分档。所有地址表与 id 表都以它为键。
_TIERS = (64, 128, 192)

_L1_BYTES = 512 * 1024
_L0AB_BYTES = 64 * 1024
_L0C_BYTES = 256 * 1024
_UB_BYTES = 256 * 1024


# ==================================================================
#  buffer-id (mutex) 分配
# ==================================================================
# id 空间是**扁平的 [0, 32)，不按内存空间分**：L1 / L0A / L0B / L0C / UB 共用
# 同一批号。生成代码里就是 get_buf(PIPE_x, id, 0)，除了 pipe 和 id 再没有第三个
# 维度。唯一的分离维度是**核**：cube 与 vector 是两颗物理核，各有一套 32 个
# token，同号互不相干。
#
# CV 共享的 6 块 buffer 声明在两个 section 之外，只有一份 mutex_ids，两侧都要
# 用，所以必须两侧同号。把它们钉在号段最前面，两个池子都从这段之后开始发号，
# 谁也不会踩到 —— 这样 cube 池和 vector 池就能各自独立编号。
_CV_SLOTS = {
    "mm1_res": 2,  # cube FIX 写、vector V 读
    "mm2_res": 2,
    "ds": 1,  # vector 写、cube mm3/mm4 读
    "p": 1,
    # dS^T / P^T 与 ds / p 同址但**另发一个号**：它们是同一块 L1 的 ZN 视图，
    # 有独立游标，读写序由 CV flag(SYNC_V3_TO_C3 等)而不是 mutex 保证。
    "ds_t": 1,
    "p_t": 1,
}


def _build_cv_mutex():
    table, nxt = {}, 0
    for name, n in _CV_SLOTS.items():
        table[name] = list(range(nxt, nxt + n))
        nxt += n
    return table, nxt


_CV_MTX, _CV_RESERVED = _build_cv_mutex()


class _MutexPool:
    """一侧核的 buffer-id 顺序发号器。

    一律顺序发号、绝不复用：id 只要单射就正确，而「同一地址挂了不同号」才是
    真正的 bug（互斥直接消失，且不会当场报错）。别名视图不在这里申请号，它们
    直接引用被别名那块 buffer 的记录，同号是结构保证而非人为约定。
    """

    LIMIT = 32

    def __init__(self, side, tag):
        self._side, self._tag = side, tag
        self._next = _CV_RESERVED
        self.owners = {i: f"{n}(CV)" for n, ids in _CV_MTX.items() for i in ids}

    def take(self, label, count):
        # 每核只有 LIMIT 个 token；排到 LIMIT 之后就该合并生命期不重叠的
        # buffer 或砍槽位。当前用量见 audit_mutex()。
        ids = list(range(self._next, self._next + count))
        self._next += count
        for i in ids:
            self.owners[i] = label
        return ids

    def report(self):
        free = sorted(set(range(self.LIMIT)) - set(self.owners))
        lines = [
            f"{self._side}({self._tag}): 用了 {len(self.owners)}/{self.LIMIT}，"
            f"空闲 {len(free)} 个 -> {free}"
        ]
        for i in sorted(self.owners):
            lines.append(f"   {i:>2}  {self.owners[i]}")
        return "\n".join(lines)


# ==================================================================
#  片上内存的 bump 分配器
# ==================================================================
class _Arena:
    """一块片上内存的顺序铺排。

    只在 import 期执行，绝不进 trace：导出的永远是 int / list[int] 的平表，
    因为 tile 的 ``addrs=`` / ``mutex_ids=`` 只接受 trace 期常量。

    ``capacity`` 是**绝对上限**（含 ``base`` 之前的保留区），越界即断言失败。
    """

    def __init__(self, name, capacity, pool, base=0):
        self._name = name
        self._cap = capacity
        self._pool = pool
        self._cur = base
        self.map = []  # [(标签, 地址, 字节数, mutex id)]

    def buf(self, label, nbytes, slots=1, align=1, cv=None):
        """声明一块 buffer —— **地址和 buffer-id 在这一处一起发**。

        ``cv`` 给出 CV 共享段的名字时用钉死的两侧同号，否则由本侧 pool 发号。
        """
        ids = list(_CV_MTX[cv]) if cv else self._pool.take(label, slots)
        return {"addrs": self._place(label, nbytes, len(ids), align, ids), "mutex": ids}

    def _place(self, label, nbytes, count, align, ids):
        addrs = []
        for i in range(count):
            self._cur = (self._cur + align - 1) & ~(align - 1)
            addrs.append(self._cur)
            self.map.append(
                (f"{label}[{i}]" if count > 1 else label, self._cur, nbytes, ids[i])
            )
            self._cur += nbytes
        # 越过 capacity 就是片上内存放不下了；余量见 describe_layout()。
        return addrs

    def describe(self):
        def size(n):
            return f"{n // 1024}KB" if n >= 1024 else f"{n}B"

        lines = [
            f"{self._name}: 用到 0x{self._cur:06X}"
            f"（{size(self._cur)} / {size(self._cap)}，"
            f"余 {size(self._cap - self._cur)}）"
        ]
        for label, addr, nbytes, mid in self.map:
            lock = f"mutex {mid}" if mid is not None else ""
            lines.append(f"  0x{addr:06X}  {size(nbytes):>5}  {label:<12}{lock}")
        return "\n".join(lines)


def _addrs(table, field):
    """``{档: {字段: buffer}}`` → ``{档: 地址列表}``。"""
    return {w: table[w][field]["addrs"] for w in table}


def _addr(table, field):
    """单槽 buffer 的标量地址表。"""
    return {w: table[w][field]["addrs"][0] for w in table}


def _mutex(table, field):
    """``{档: {字段: buffer}}`` → ``{档: id 列表}``。与 _addrs 同源，不会对不上。"""
    return {w: table[w][field]["mutex"] for w in table}


def _flat(table, field):
    """槽位数与档无关的 buffer 导出成扁平列表。

    vector 侧的 buffer 名单与 d_align 无关，三档发号顺序一致，取哪档都一样；
    真要改动发号顺序，用 audit_mutex() 对一下三档是否仍然对齐。
    """
    return list(table[HEAD_DIM_ALIGN][field]["mutex"])


# ==================================================================
#  Cube 侧：L1 (512KB) + L0A/L0B (各 64KB) + L0C (256KB)
# ==================================================================
def _build_l1(width, pool):
    """铺排一档的 L1，返回 ``{名字: buffer}``。

    Q/dO 是多槽 double buffer：tile_group 填多个 addrs + 多个 mutex_id 后，
    框架会自动插核内同步，使下一轮 load 与本轮 matmul 重叠。槽位数不是 2
    而是 3+：cube 领先 vector 两个 task，mm2 在
    task N 载入 Q_N，而 mm4 要的是 Q_{N-2}；3 槽时 N%3 != (N-2)%3，两者不会
    互相覆写，于是 mm4/mm5 可以直接复用 mm2/mm1 搬好的数据，无需再读 GM。
    """
    ar = _Arena(f"L1 d_align={width}", _L1_BYTES, pool)
    blk = CUBE_BASEM * D_PHYS_W[width] * 2  # Q/K/V/dO 一块：64/128 档 32KB，192 档 48KB
    dsp = CUBE_BASEM * CUBE_BASEN * 2  # dS/P 是 [M,N]，与 D 无关，恒 32KB
    # 64/128 档 4 槽；192 档每块 48KB，512KB 只放得下 3 槽。
    nq = 4 if width <= HEAD_DIM_ALIGN else 3

    q = ar.buf("q", blk, slots=nq)  # q_lag_view 复用这份记录，同址同号
    do = ar.buf("do", blk, slots=nq)  # do_lag_view 同理
    k = ar.buf("k", blk)
    # MM3(dQ = dS@K) 的右矩阵要非转置的 K[S2,D]，必须独占一块 L1：同一地址上
    # 叠加 ZN 的 [D,S2] 与 ND 的 [S2,D] 两种视图能编过，但结果静默错误。
    k_nd = ar.buf("k_nd", blk)
    v = ar.buf("v", blk)
    # dS^T / P^T 不单独占地址，是 ds / p 上的 ZN 视图（见 make_ds_t_l1）。
    ds = ar.buf("ds", dsp, cv="ds")
    p = ar.buf("p", dsp, cv="p")

    return {
        "q": q,
        "do": do,
        "k": k,
        "k_nd": k_nd,
        "v": v,
        "ds": ds,
        "p": p,
        "_arenas": [ar],
    }


# acc / L0A / L0B 的列宽：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个 D 分块，
# 取两者较大值，一块 tile 同时够用。三档都等于 128。
ACC_COLS = {w: max(CUBE_BASEN, D_CHUNK_MAXW[w]) for w in _TIERS}


def _build_l0ab(width, pool):
    """L0A 与 L0B —— 两块独立的 64KB，各开双缓冲，布局相同但各有一套号。

    5 个 matmul 串行复用同一块 L0A/L0B 时，pl.move 填充期间 mac 只能干等
    （实测单缓冲 mte1 占用率 0.309）；双缓冲让下一个 matmul 的
    L1->L0 搬运与当前 matmul 的计算重叠。

    间距必须按 **tile 实际声明的宽度** ACC_COLS 算，而不是 D 分块宽度：
    L0A/L0B 要同时装下 MM3/MM4/MM5 的 [M,N] 左矩阵，故声明成 ACC_COLS 列。
    若按 192 档的分块宽 96 算间距（24KB），两个槽位会重叠（每块实占 32KB）。
    """
    blk = CUBE_BASEM * ACC_COLS[width] * 2
    a = _Arena(f"L0A d_align={width}", _L0AB_BYTES, pool)
    b = _Arena(f"L0B d_align={width}", _L0AB_BYTES, pool)
    # 两块布局相同，地址表一致，故对外只导出 l0a 那份（L0AB_ADDRS）。
    l0a = a.buf("l0a", blk, slots=2)
    l0b = b.buf("l0b", blk, slots=2)
    return {"l0a": l0a, "l0b": l0b, "_arenas": [a, b]}


def _build_l0c(width, pool):
    """L0C：acc 槽位 + dK/dV 各 D 分块的常驻块。

    dK/dV 要在同一 s2 块内跨 s1 轮次累加，s2 切换时才
    fixpipe 出 GM，所以每个 D 分块都得有自己的常驻 L0C —— 不能像 acc 那样复用，
    故一个分块一个号。

    acc 单缓冲时 fixpipe 写出期间下一个 matmul 只能干等（实测 fixpipe 0.458
    而 mac 仅 0.783），故 64/128 档开两槽；192 档预算被 dK/dV 分块吃掉，退回
    单槽。三档都正好用满 256KB。
    """
    ar = _Arena(f"L0C d_align={width}", _L0C_BYTES, pool)
    accBlk = CUBE_BASEM * ACC_COLS[width] * 4  # acc 要装 MM2/MM1 的 [M,s2]
    dkvBlk = CUBE_BASEN * D_CHUNK_MAXW[width] * 4  # dK/dV 只装一个 D 分块
    nChunk = D_NCHUNKS[width]

    acc = ar.buf("acc", accBlk, slots=2 if width <= HEAD_DIM_ALIGN else 1)
    # dv 也按 D 的分块数分块；Dv<=D 时后一块可能整块无效，由运行期宽度判断
    # 跳过（见 dv_chunk_w）。
    dk = ar.buf("dk", dkvBlk, slots=nChunk)
    dv = ar.buf("dv", dkvBlk, slots=nChunk)
    return {"acc": acc, "dk": dk, "dv": dv, "_arenas": [ar]}


# ==================================================================
#  Vector 侧：UB (256KB)
# ==================================================================
# 头部留给 codegen 的 UB[0] tiling 中转：BN2 没有 pre/sync_all，不留这块会和
# cube 的 fixpipe 抢 UB[0]。所有 UB arena 都从它之后开始。
UB_HEAD_RESERVE = 0x1000
# [64,1] fp32 只有 256B，但按 512B 一格排，保证后面的槽位对齐。
_UB_ROW_SLOT = 0x200

# pre/post 走一维元素，不按 [行, D] 切。三档都吃满可用 UB，且每档
# elems 都能被 128 整除，full-tile 的 VF 段数是编译期常量。行区间已由 metadata
# 按 nAiv 均分，块大不再空核。
PRE_ELEMS = {64: 63488, 128: 63488, 192: 63360}  # 单槽 fp32
POST_ELEMS = {64: 20480, 128: 21120, 192: 21120}  # depth-2，fp32 + fp16


def _build_ub(width, pool):
    """一档的 vector 侧全部 UB。

    三张图共用这 256KB，靠执行阶段错开：

    * 主图（y/dy/nzp/sfmg/lse/cast）—— 主循环用，从 0x21000 起；
    * mm1/mm2 结果区 —— CV 共享，cube fixpipe 写、vector 读，从头部起；
    * pre/post —— 与主循环被 sync_all 完全隔开，整块借用头部。

    V1 一趟 VECTOR_BASEM 行、按行 reduce，不再开 prod/tmp。列宽
    ``v1w = 64 if d_align == 64 else dv_align``（y/dy 是 Dv 宽），这里按档取。
    attenmask 复用 y/dy 这两条记录：uint8[64,128]=8KB，而 y/dy 在 v1w>=64
    时至少 8KB，装得下。
    复用的是**同一条记录**，所以「同址必须同号」是结构保证，不靠人守。

    cast 只有 BN2 的 v5/v6 用，但三档都留：它排在最后，不占别人的地方，
    留着可以让 BN2 直接复用这张图（见 BN2_UB_*）。
    """
    main = _Arena(f"UB v1w={width}", _UB_BYTES, pool, base=UB_HEAD_RESERVE + 0x20000)
    y = main.buf("y", VECTOR_BASEM * width * 2)
    dy = main.buf("dy", VECTOR_BASEM * width * 2)
    # NZ+1 行：末行是 bank 错开的填充，见 fag_common.NZP_BLOCK_STRIDE。
    nzp = main.buf("nzp", NZP_ROWS * VECTOR_BASEN * 2)
    sfmg = main.buf("sfmg", _UB_ROW_SLOT, align=_UB_ROW_SLOT)
    lse = main.buf("lse", _UB_ROW_SLOT, slots=2)
    cast = main.buf("cast", VECTOR_BASEM * VECTOR_BASEN * 2)

    shared = _Arena("UB mm1/mm2", _UB_BYTES, pool, base=UB_HEAD_RESERVE)
    resBlk = VECTOR_BASEM * VECTOR_BASEN * 4  # fp32[64,128] = 32KB
    mm1 = shared.buf("mm1_res", resBlk, cv="mm1_res")
    mm2 = shared.buf("mm2_res", resBlk, cv="mm2_res")

    # pre 的 fp16 支（BN2 need_zero）只占一半，按 fp32 算上限。
    pre = _Arena(f"UB pre d_align={width}", _UB_BYTES, pool, base=UB_HEAD_RESERVE)
    zero = pre.buf("pre_zero", PRE_ELEMS[width] * 4)

    # post 的 fp32 输入与 fp16 输出各用一套号，框架据此判定「下一块的 load」与
    # 「本块的 cast/store」无冲突，从而让 MTE2 / V / MTE3 三条流水重叠。
    post = _Arena(f"UB post d_align={width}", _UB_BYTES, pool, base=UB_HEAD_RESERVE)
    n = POST_ELEMS[width]
    f32 = post.buf("post_f32", n * 4, slots=2)
    f16 = post.buf("post_f16", n * 2, slots=2)

    return {
        "y": y,
        "dy": dy,
        "nzp": nzp,
        "sfmg": sfmg,
        "lse": lse,
        "cast": cast,
        "mm1": mm1,
        "mm2": mm2,
        "zero": zero,
        "f32": f32,
        "f16": f16,
        "_arenas": [main, shared, pre, post],
    }


# ==================================================================
#  分档铺排：每档一套 arena，cube / vector 各一个号池
# ==================================================================
_CUBE_POOL = {w: _MutexPool("cube", f"d_align={w}") for w in _TIERS}
_VEC_POOL = {w: _MutexPool("vector", f"d_align={w}") for w in _TIERS}

_L1 = {w: _build_l1(w, _CUBE_POOL[w]) for w in _TIERS}
_L0AB = {w: _build_l0ab(w, _CUBE_POOL[w]) for w in _TIERS}
_L0C = {w: _build_l0c(w, _CUBE_POOL[w]) for w in _TIERS}
_UB = {w: _build_ub(w, _VEC_POOL[w]) for w in _TIERS}


# ---- L1 ----
L1_Q_ADDRS = _addrs(_L1, "q")
L1_Q_MUTEX = _mutex(_L1, "q")
L1_DO_ADDRS = _addrs(_L1, "do")
L1_DO_MUTEX = _mutex(_L1, "do")
L1_K_ADDR = _addr(_L1, "k")
MTX_L1_K = _mutex(_L1, "k")
L1_KND_ADDR = _addr(_L1, "k_nd")
MTX_L1_K_ND = _mutex(_L1, "k_nd")
L1_V_ADDR = _addr(_L1, "v")
MTX_L1_V = _mutex(_L1, "v")
L1_DS_ADDR = _addr(_L1, "ds")
L1_P_ADDR = _addr(_L1, "p")
# CV 共享段的号钉死，三档一致，导出成扁平列表。
MTX_L1_DS = list(_CV_MTX["ds"])
MTX_L1_P = list(_CV_MTX["p"])
MTX_L1_DS_T = list(_CV_MTX["ds_t"])  # 与 ds 同址、另发一号，见 _CV_SLOTS
MTX_L1_P_T = list(_CV_MTX["p_t"])

# ---- L0A / L0B ----
L0AB_ADDRS = _addrs(_L0AB, "l0a")
MTX_L0A = _mutex(_L0AB, "l0a")
MTX_L0B = _mutex(_L0AB, "l0b")

# ---- L0C ----
L0C_ACC_ADDRS = _addrs(_L0C, "acc")
L0C_ACC_MUTEX = _mutex(_L0C, "acc")
# 每个 D 分块单独取一个标量地址/id。不能在 kernel 里写 `TABLE[d_align][-1]`：
# parser 不支持负下标（报 "GetItemExpr index -1 out of bounds"）。
_DK = {w: _L0C[w]["dk"] for w in _TIERS}
_DV = {w: _L0C[w]["dv"] for w in _TIERS}
L0C_DK0_ADDR = {w: _DK[w]["addrs"][0] for w in _TIERS}
L0C_DV0_ADDR = {w: _DV[w]["addrs"][0] for w in _TIERS}
MTX_L0C_DK0 = {w: _DK[w]["mutex"][0] for w in _TIERS}
MTX_L0C_DV0 = {w: _DV[w]["mutex"][0] for w in _TIERS}
# 第二个分块。单块档位下这些 tile 永不被真的使用（调用路径被 trace 期的
# `if D_NCHUNKS > 1` 整块剪掉），故让它别名到第 0 块即可。
L0C_DK1_ADDR = {w: _DK[w]["addrs"][D_NCHUNKS[w] - 1] for w in _TIERS}
L0C_DV1_ADDR = {w: _DV[w]["addrs"][D_NCHUNKS[w] - 1] for w in _TIERS}
MTX_L0C_DK1 = {w: _DK[w]["mutex"][D_NCHUNKS[w] - 1] for w in _TIERS}
MTX_L0C_DV1 = {w: _DV[w]["mutex"][D_NCHUNKS[w] - 1] for w in _TIERS}
# BN2 不切 D，但它只编 64/128 两档，那两档 D_NCHUNKS 本来就是 1，所以直接用
# 上面的第 0 块即可，不需要另一套常量。

# ---- UB ----
# vector 侧的 buffer 名单与档无关，三档发出的号必然一致，故导出扁平列表。
UB_MM1_0, UB_MM1_1 = _UB[HEAD_DIM_ALIGN]["mm1"]["addrs"]
UB_MM2_0, UB_MM2_1 = _UB[HEAD_DIM_ALIGN]["mm2"]["addrs"]
MTX_MM1_RES = list(_CV_MTX["mm1_res"])
MTX_MM2_RES = list(_CV_MTX["mm2_res"])

UB_Y_A = _addr(_UB, "y")
UB_DY_A = _addr(_UB, "dy")
UB_NZP_A = _addr(_UB, "nzp")
UB_SFMG_A = _addr(_UB, "sfmg")
UB_LSE0_A = {w: _UB[w]["lse"]["addrs"][0] for w in _TIERS}
UB_LSE1_A = {w: _UB[w]["lse"]["addrs"][1] for w in _TIERS}
# attenmask 复用 y/dy 的地址与号（vec1 之后 vec2 再用）。
UB_MASK_A = UB_Y_A
UB_MASK_PRE_A = UB_DY_A
MTX_UB_Y = _flat(_UB, "y")  # attenmask 复用
MTX_UB_DY = _flat(_UB, "dy")  # attenmask(pre) 复用
MTX_UB_NZP = _flat(_UB, "nzp")
MTX_UB_SFMG = _flat(_UB, "sfmg")
MTX_UB_LSE = _flat(_UB, "lse")
MTX_UB_CAST = _flat(_UB, "cast")  # 仅 BN2

UB_PRE_ZERO = _UB[HEAD_DIM_ALIGN]["zero"]["addrs"][0]
MTX_UB_ZERO = _flat(_UB, "zero")
POST_F32_ADDRS = _addrs(_UB, "f32")
POST_F16_ADDRS = _addrs(_UB, "f16")
MTX_UB_POST_F32 = _flat(_UB, "f32")
MTX_UB_POST_F16 = _flat(_UB, "f16")


# ==================================================================
#  BN2 模板专用的 UB 槽位
# ==================================================================
# BN2 的 V1 与 GS1S2 共用 softmax_grad_front_row_vf，且只在 d_align 64/128 两档
# 编译（y/dy 最宽就是 [64,128]），所以**整张图直接沿用 v1w=128 那一份** ——
# 同地址、同号、连 cast 都是同一块。每个 tilingkey 只编一个 template，两套用法
# 不会同时活跃。
BN2_V1_ROWS = VECTOR_BASEM
BN2_CAST_W = VECTOR_BASEN
_BN2 = _UB[HEAD_DIM_ALIGN]
BN2_UB_Y = _BN2["y"]["addrs"][0]
BN2_UB_DY = _BN2["dy"]["addrs"][0]
BN2_UB_NZP = _BN2["nzp"]["addrs"][0]
BN2_UB_SFMG = _BN2["sfmg"]["addrs"][0]
BN2_UB_LSE_0, BN2_UB_LSE_1 = _BN2["lse"]["addrs"]
BN2_UB_CAST = _BN2["cast"]["addrs"][0]


# ==================================================================
#  自查
# ==================================================================
def describe_layout(tier=HEAD_DIM_ALIGN):
    """打印一档的完整片上内存图：地址 + 大小 + 持有的 buffer-id。"""
    arenas = []
    for group in (_L1, _L0AB, _L0C, _UB):
        arenas.extend(group[tier]["_arenas"])
    return "\n\n".join(a.describe() for a in arenas)


def audit_mutex(tier=HEAD_DIM_ALIGN):
    """打印一档的 buffer-id 占用：每个号归谁、每侧还剩几个。"""
    return _CUBE_POOL[tier].report() + "\n\n" + _VEC_POOL[tier].report()
