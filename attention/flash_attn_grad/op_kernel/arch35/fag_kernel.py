# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""两个分核模板的入口与编译期分发。

本文件只负责**编排**：建 GM/L1 视图和 ``pl.make_tuple`` 袋子、跑 pre/post、
按 ``deter`` / ``template`` 分发到主循环。真正的 while 在
:mod:`fag_kernel_norm` 与 :mod:`fag_kernel_deter`；计算在
:mod:`fag_block_cube` 与 :mod:`fag_block_vec`，分核映射在 :mod:`fag_schedule`，
片上地址在 :mod:`fag_mem`。

两个模板：

* :func:`flash_attn_grad_preload3` —— ``Tmpl.BN2GS1S2`` 且 D≤192：
  cube 领先 vector 两个 task（``PRELOAD_TIMES=3``），自带 pre/post。
  192 档 dK 走 fp32 workspace（与 256 档同口径）、dV 仍常驻 L0C。
* :func:`flash_attn_grad_preload2` —— 1-ahead。``Tmpl.BN2`` 走
  :mod:`fag_kernel_norm` 的 BN2 循环；GS1S2 再按 ``deter`` 分到
  normal / deterministic。

CV 通信全部使用手动 set/wait cross-core，且正反向成对：

  正向 C->V : SYNC_C2_TO_V2_FLAG[2] / SYNC_C1_TO_V2_FLAG[2]  (mm2/mm1 结果就绪)
  正向 V->C : SYNC_V3_TO_C3_FLAG / SYNC_V4_TO_C5_FLAG        (dS/P 已落 L1)
  反向 C->V : SYNC_C4_TO_V3_FLAG / SYNC_C5_TO_V4_FLAG        (L1 已被 cube 读完)
  反向 V->C : SYNC_V2_TO_C1_FLAG[2] / SYNC_V2_TO_C2_FLAG[2]  (mm1/mm2 UB 已读完)

AIC 侧对每个 V->C flag 用 ``sync_mode=INTRA_BLOCK`` 等一次，覆盖两个
vector 子核各自发来的信号。

当前裁剪范围：BSND/BNSD/TND、FP16/BF16。mask_mode 0=dense，3=causal，
4=band。tilingkey ``swizzle`` 区分线性 skip 无效块与 packed 有效块分核。
``deter`` 只做编译期分发。非 swizzle 核上闭式分核，与现网同套跨轮流水和
L0C 累加，不每轮 ``sync_all``。BN2 支持 mask，禁止 BN2+swizzle，且与
GS1S2 deterministic 完全隔离。
"""

import inspect

import pypto_pro.language as pl

import fag_block_cube
import fag_block_vec
import fag_buffers
import fag_common
import fag_deter
import fag_deter_tnd_dense
import fag_deter_tnd_sparse
import fag_schedule
import fag_nzout
from fag_tiling import FlashAttnGradTilingData, Tmpl
from fag_kernel_deter import (
    _run_preload2_cube_gs_deter,
    _run_preload2_vector_gs_deter,
    _run_preload3_cube_deter,
    _run_preload3_vector_deter,
)
from fag_kernel_norm import (
    _run_preload2_cube_bn2,
    _run_preload2_cube_gs_normal,
    _run_preload2_vector_bn2,
    _run_preload2_vector_gs_normal,
    _run_preload3_cube_normal,
    _run_preload3_vector_normal,
)

# codegen 内联时读的是本模块 globals() 里的裸名，不是 from-import 清单。
# module.func() 会被当成 IR op，所以这里把 helper / 常量绑成裸名。
# 其它 list 不灌进来，避免和必须具名的 ping-pong flag 抢提升。
_PINGPONG_LISTS = frozenset(
    {
        "SYNC_C1_TO_V2_FLAG",
        "SYNC_C2_TO_V2_FLAG",
        "SYNC_V2_TO_C1_FLAG",
        "SYNC_V2_TO_C2_FLAG",
    }
)


def _bind_helpers():
    g = globals()
    for mod in (
        fag_block_cube,
        fag_block_vec,
        fag_buffers,
        fag_common,
        fag_deter,
        fag_deter_tnd_dense,
        fag_deter_tnd_sparse,
        fag_schedule,
        fag_nzout,
    ):
        for name, val in vars(mod).items():
            if name.startswith("_") or inspect.ismodule(val):
                continue
            if isinstance(val, list) and name not in _PINGPONG_LISTS:
                continue
            g[name] = val


_bind_helpers()


def flash_attn_grad_preload3(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # 以下为可选输入。TND 的 cu_seqlens 给物理偏移，seqused 给有效长度。
    # sinks：tilingkey sink=1 时按 head 读标量，算 dsink。
    # attn_mask：mask_mode 3/4 时按 [2048,2048] uint8 压缩矩阵搬 UB。
    # metadata：FAG 区含分核 [start,end) 与 pre/post 行区间（412 int32）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    # 4 个输出。dsink 是 [N1] fp32，仅在传入 sinks 时写出有效结果。
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    # 单块 workspace，内部再切 dq/dk/dv 累加区和 dsink 槽位
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
    deter,
    sink,
):
    # 入口只建视图 / buffer / pre-post，cube/vector 主循环按 deter 编译期分发。
    # coreNum 从内建 API 取而非 tiling：TilingData 无此字段。
    # get_block_num() 是 host SetBlockDim 启动的 cube 数（dense 下等于
    # metadata blockOuter），cube/vector 两侧同一口径，set/wait 才能配对。
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey，是编译期常量 —— 每种分核各生成一份二进制，
    # `if swizzle` 的两路在编译期折叠，运行期没有判断开销。
    cu_q_tensor = pl.make_tensor(cu_seqlens_q, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    cu_kv_tensor = pl.make_tensor(cu_seqlens_kv, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    used_q_tensor = pl.make_tensor(seqused_q, [tiling.b], [1], dtype=pl.DT_INT32)
    used_kv_tensor = pl.make_tensor(seqused_kv, [tiling.b], [1], dtype=pl.DT_INT32)
    # constInfo 内统一计算实际 token 数，区分逻辑序列长度与 GM 视图长度。
    constInfo = set_const_info(tiling, coreNum, cu_q_tensor, cu_kv_tensor)

    # ---- workspace 分区 ----
    # host(GetWorkspaceSizes) 给的是单块连续内存，这里按 fp32 元素切成
    # dq/dk/dv 三段累加区。三段大小必须与 host 的 CalcWorkSpace 逐一对齐。
    # dq 段按 n1(=n2*G) 个 head，dk/dv 段按 n2 —— 与 host 的 dqElems/dkvElems
    # 一致。GQA 下若这里误用 n2，dk/dv 的起始偏移就会算错。
    # TND 按物理 packed token 分段，非 TND 的 T 仍为 B*S；seqused 不改变跨度。
    # host 只在三段之后多留 128 行保护垫，不改这里的基地址。
    wsRowsQ = constInfo.t1Size * constInfo.n1Size
    wsRowsKV = constInfo.t2Size * constInfo.n2Size
    dqWsElems = wsRowsQ * tiling.d
    dkWsElems = wsRowsKV * tiling.d
    dvWsElems = wsRowsKV * tiling.dv
    wsPadCols = pl.max(tiling.d, tiling.dv)
    if nz_out == 1:
        qTiles = (
            nz_workspace_tiles(constInfo.t1Size, tiling.b, tiling.s1) * constInfo.n1Size
        )
        kvTiles = (
            nz_workspace_tiles(constInfo.t2Size, tiling.b, tiling.s2) * constInfo.n2Size
        )
        dqWsElems = qTiles * 128 * constInfo.dAlign
        dkWsElems = kvTiles * 128 * constInfo.dAlign
        dvWsElems = dkWsElems
        wsPadCols = constInfo.dAlign
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dq_workspace = ws_ptr
    # 段宽按各自的 head dim：dq/dk 是 D，dv 是 Dv(可以短于 D)。
    # 必须与 host 的 dqElems/dkElems/dvElems 逐一对齐。
    dk_workspace = pl.addptr(dq_workspace, dqWsElems)
    dv_workspace = pl.addptr(dk_workspace, dkWsElems)
    # 128 行对齐垫之后是 dsink 槽位 [aivCount x N1]，与 host BuildWorkspacePlan 对齐。
    dsink_workspace = pl.addptr(dv_workspace, dvWsElems + 128 * wsPadCols)
    # dsink 槽位之后是 sfmg 前移缓存 [N1*T1] fp32，与 host 的 plan.sfmg 对齐。
    # dv_align>=128 时 pre 阶段全量写入、主循环 V1 只读现成值。
    sfmg_workspace = pl.addptr(dsink_workspace, constInfo.aivCount * constInfo.n1Size)

    # ---- 统一四维视图 [viewD0, S, viewD2, D] ----
    # BSND 与 BNSD 共用同一套 make_tensor / pl.load(order=...) 代码，
    # 布局差异只体现在 viewD0/viewD2 与索引系数上(见 set_run_info 的 idx0/idx2)：
    #   BSND: [B,   S, N, D]，idx = [b,     s, n, 0]
    #   BNSD: [B*N, S, 1, D]，idx = [b*N+n, s, 0, 0]
    # 两者都是紧凑排布 —— 这一点是必须的：make_tensor 对非紧凑 stride 会静默
    # 忽略并按紧凑跨距去读(不报错)，所以不能用「倒置 stride」表达 BNSD。
    # Q 侧(n1 个 head)与 KV 侧(n2 个 head)各一套视图参数
    # 最内轴宽度取各张量**真实的** head dim，不能用分配宽度 d_align：
    # 这是 GM 上的实际排布，Q/K/dq/dk 侧是 D，V/dO/attn_out/dv 侧是 Dv。
    # (片上 tile 才按 d_align 分配、按真实宽度填。)
    dQ = constInfo.dSize
    dV = constInfo.dvSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1ViewSize * d2dQ
    viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2ViewSize * d2dKV
    # Dv 宽的那几个张量另有一套行距
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1ViewSize * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2ViewSize * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=input_dtype,
    )
    tensor_k = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # MM3(dQ = dS@K) 需要「未转置」的 K，而 MM2 需要 K^T。
    # 每个 GM tensor 只会推导出一种 Layout：tensor_k 因 MM2 的转置读被定成
    # Layout::DN，MM3 若共用它，plain 读也会被按 DN 解释，实测等价于拿到 K^T
    # (partial 与 K[s2o]^T 吻合到 1e-4)。故为 MM3 单独建一份同 buffer 的
    # tensor 视图，让它独立推导 layout。
    tensor_k_nt = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # V/dO/attn_out 是 Dv 宽
    tensor_v = pl.make_tensor(
        v,
        [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=input_dtype,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    # softmax_lse : [B,N1,S1] —— 按 Q 的 head 数(n1 = n2*G)存，因为每个 Q head
    # 的 softmax 归一化因子不同，G 个 Q head 不能共享同一份 lse。
    # lse offset = ((b*n2 + n2o)*G + g)*S1。
    n1Size = constInfo.n2Size * constInfo.gSize
    if layout == 1:
        # TND LSE 为 [N1,T1]，loadBase 已是全局 token 行。
        # head 间距必须取实际 T1，不能用 B*maxSeqlenQ。
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [n1Size, constInfo.t1Size],
            [constInfo.t1Size, 1],
            dtype=pl.DT_FP32,
        )
    else:
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [constInfo.bSize, n1Size, constInfo.s1Size],
            [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
            dtype=pl.DT_FP32,
        )
    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask,
            [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1],
            dtype=pl.DT_UINT8,
        )
    else:
        tensor_mask = tensor_lse
    # sfmg 前移缓存：[1, N1*T1] fp32 扁平，行号 = head*T1 + 全局 token 行
    # （与 TND lse 的 [N1,T1] token 主序同构，读侧统一用 loadBaseTnd 口径）。
    # dv_align=64 不前移：占位必须是 rank-2 扁平视图（与非 TND 的
    # tensor_lse [B,N1,S1] 三维不同 rank，pypto 严格校验 load 的 offsets
    # 数与源 rank 一致，直接占位 lse 会在 trace 期报 INVALID_SHAPE）。
    # process_vec1 走内联分支时不解引用本占位。
    if dv_align >= 128:
        sfmgElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            sfmg_workspace, [1, sfmgElems], [sfmgElems, 1], dtype=pl.DT_FP32
        )
    else:
        lseElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            softmax_lse, [1, lseElems], [lseElems, 1], dtype=pl.DT_FP32
        )
    # dq/dk/dv workspace 为 fp32，供 atomicAdd 累加
    # dq/dk 是 D 宽，dv 是 Dv 宽 —— 与 workspace 的分段宽度一致
    if nz_out == 1:
        tensor_dq_ws = pl.make_tensor(
            dq_workspace, [1, dqWsElems], [dqWsElems, 1], dtype=pl.DT_FP32
        )
        tensor_dk_ws = pl.make_tensor(
            dk_workspace, [1, dkWsElems], [dkWsElems, 1], dtype=pl.DT_FP32
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace, [1, dvWsElems], [dvWsElems, 1], dtype=pl.DT_FP32
        )
    else:
        tensor_dq_ws = pl.make_tensor(
            dq_workspace,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dk_ws = pl.make_tensor(
            dk_workspace,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=pl.DT_FP32,
        )
    # ---- pre/post：一维视图 ----
    # InitOutput / Muls+Cast 按实际 T*N*D 个元素线性切（非 TND 的 T=B*S）。
    # dq 侧 head 数是 n1(=n2*G)，dk/dv 侧是 n2。
    # ws 与 out 扁平视图都按真实 d/dv（无 padding）。
    dqRows = constInfo.t1Size * n1Size
    dkvRows = constInfo.t2Size * constInfo.n2Size
    dqElems = dqRows * dQ
    dkElems = dkvRows * dQ
    dvElems = dkvRows * dV
    tensor_dq_ws_flat = pl.make_tensor(
        dq_workspace, [1, dqWsElems], [dqWsElems, 1], dtype=pl.DT_FP32
    )
    tensor_dk_ws_flat = pl.make_tensor(
        dk_workspace, [1, dkWsElems], [dkWsElems, 1], dtype=pl.DT_FP32
    )
    tensor_dv_ws_flat = pl.make_tensor(
        dv_workspace, [1, dvWsElems], [dvWsElems, 1], dtype=pl.DT_FP32
    )
    if nz_out == 1:
        tensor_dq_out = pl.make_tensor(
            dq,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=input_dtype,
        )
    else:
        tensor_dq_out = pl.make_tensor(
            dq, [1, dqElems], [dqElems, 1], dtype=input_dtype
        )
        tensor_dk_out = pl.make_tensor(
            dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
        )
        tensor_dv_out = pl.make_tensor(
            dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
        )
    tensor_sinks = pl.make_tensor(sinks, [n1Size], [1], dtype=pl.DT_FP32)
    dsinkElems = constInfo.aivCount * n1Size
    tensor_dsink_ws = pl.make_tensor(
        dsink_workspace, [1, dsinkElems], [dsinkElems, 1], dtype=pl.DT_FP32
    )
    # dsink : [N1] fp32，post 阶段由 AIV0 固定次序归约后全量写出。
    tensor_dsink_out = pl.make_tensor(dsink, [1, n1Size], [n1Size, 1], dtype=pl.DT_FP32)
    # V1 列宽就是 dv_align。D=256 不进本函数。
    v1w = dv_align

    # cBlockIdx 必须按物理核取：AIV 侧 get_block_idx() 按 vector 子核计数
    # (0..2*coreNum-1)，AIC 侧按物理核计数。两侧口径不一致会算出不同的
    # numBlocks 不同 -> 循环次数不同 -> set/wait 配不上 -> 死锁。
    # 核数直接用 tiling 传入，避免再依赖 get_block_num 的口径。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum

    # 分核区间取自 metadata。swizzle 支仍走公式，原因见 block_range_of。
    meta = pl.make_tensor(metadata, [META_VIEW_LEN], [1], dtype=pl.DT_INT32)
    one64 = pl.const(1, pl.DT_INT64)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX) * one64
    blockStart = (
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx) * one64
    )
    blockEnd = pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx) * one64
    numBlocks = block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd)

    # ---- CV 共享区：mm1/mm2 结果 UB + dS/P 的 L1 及其 ZN 转置视图 ----
    # vector 侧写、cube 侧读，故必须声明在两个 section 之外。
    mm1_res = make_mm1_res()
    mm2_res = make_mm2_res()
    ds_l1 = make_ds_l1()
    p_l1 = make_p_l1()
    ds_t_l1 = make_ds_t_l1()
    p_t_l1 = make_p_t_l1()

    # ==== Pre 阶段：清零 dq/dk/dv workspace ====
    # 三个 workspace 都用 atomicAdd 累加，进主循环前必须为 0。
    # 清完用 sync_all 隔开：主循环的 atomicAdd 必须看到已清零的 GM。
    # dv_align>=128 时整段走"交错清零"路径；是否真正追加 softmaxGrad 前移由
    # 运行时 constInfo.enablePreSfmg 决定（host 按 fasg enablePreSfmg 判据，
    # 见 flash_attn_grad_tiling_plan.cpp 的 DecidePreSfmg）。前移时一次算完
    # 全部行的 rowsum(dO⊙O) 存 ws、主循环 V1 只读；否则主循环 V1 内联重算。
    # 交错 pre：zt 单槽 32KB（[0x1000,0x9000)，见 fag_mem.PRE_ELEMS），与
    # sp 窗口 [0x9000,0x40000) 完全避开（sp 借 mm1 次槽 + mm2 + 主图段，
    # mutex 复用主图 tile 的号，见 fag_mem 的 sfmg_pre 注释），pre 与主循环
    # 由 sync_all 分时。zero 与 sp
    # 不同 mutex，框架不插 wait——清零写(MTE3)与 presfmg 载入(MTE2)/VF
    # 两条流水并行，且分块交替 issue：三段清零按统一 chunk 切成任务流，穿插
    # 在 presfmg 的窗口循环里（每窗 kBatch 个，见 process_presfmg 的
    # _zero_step）——避免整段清零先灌满 MTE3 队列、拖延 sfmg 写的双缓冲
    # 周转（实测中小 S 用例清零占 pre 搬运 1/3~1/2）。
    if nz_out == 1:
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32()
            # NZ 布局按整段 tile 口径清零（覆盖 padding 与 TND gap）。sfmg
            # 前移的交错清零按 ND 行区间寻址，与 NZ tile 布局不兼容——这里
            # 清零段传 0，复用 process_presfmg 的纯计算形态（同 BN2 调用点：
            # zS 全 0 → zTotal=0 → 清零任务零循环），仅保留 y/dO 窗口计算与
            # sfmg 落盘；enablePreSfmg=0 时内部自动跳过计算。
            process_nz_pre(
                tensor_dq_ws_flat, dqWsElems, constInfo.aivCount, zero_ub.current()
            )
            process_nz_pre(
                tensor_dk_ws_flat, dkWsElems, constInfo.aivCount, zero_ub.current()
            )
            process_nz_pre(
                tensor_dv_ws_flat, dvWsElems, constInfo.aivCount, zero_ub.current()
            )
            if sink == 1:
                process_sink_zero(
                    tensor_dsink_ws, constInfo, zero_ub.current(), pl.get_block_idx()
                )
            sp_y_ub = make_sp_y_ub(v1w)
            sp_dy_ub = make_sp_dy_ub(v1w)
            sp_out_ub = make_sp_out_ub(v1w)
            sp_out_flat = make_sp_out_flat(v1w)
            process_presfmg(
                constInfo,
                tensor_y,
                tensor_dy,
                tensor_sfmg_ws,
                sp_y_ub,
                sp_dy_ub,
                sp_out_ub,
                sp_out_flat,
                fag_common.SP_WORK_W[v1w],
                zero_ub.current(),
                tensor_dq_ws_flat,
                tensor_dk_ws_flat,
                tensor_dv_ws_flat,
                0,
                0,
                0,
                0,
                0,
                0,
            )
    elif dv_align >= 128:
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32()
            zt = zero_ub.current()
            # 行区间与三段边界（原 process_pre_from_meta 的读法，见其文档）。
            vBlockIdx = pl.get_block_idx()
            one64 = pl.const(1, pl.DT_INT64)
            dqRowStart = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx) * one64
            )
            dqRowEnd = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx) * one64
            )
            dkvRowStart = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx) * one64
            )
            dkvRowEnd = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx) * one64
            )
            dqNElems = (dqRowEnd - dqRowStart) * dQ
            dkNElems = (dkvRowEnd - dkvRowStart) * dQ
            dvNElems = (dkvRowEnd - dkvRowStart) * dV
            preElems = pl.min(
                PRE_ELEMS[d_align], pl.max(dqNElems, pl.max(dkNElems, dvNElems))
            )
            fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
            if sink == 1:
                process_sink_zero(tensor_dsink_ws, constInfo, zt, pl.get_block_idx())
            sp_y_ub = make_sp_y_ub(v1w)
            sp_dy_ub = make_sp_dy_ub(v1w)
            sp_out_ub = make_sp_out_ub(v1w)
            sp_out_flat = make_sp_out_flat(v1w)
            process_presfmg(
                constInfo,
                tensor_y,
                tensor_dy,
                tensor_sfmg_ws,
                sp_y_ub,
                sp_dy_ub,
                sp_out_ub,
                sp_out_flat,
                fag_common.SP_WORK_W[v1w],
                zt,
                tensor_dq_ws_flat,
                tensor_dk_ws_flat,
                tensor_dv_ws_flat,
                dqRowStart * dQ,
                dqNElems,
                dkvRowStart * dQ,
                dqNElems + dkNElems,
                dkvRowStart * dV,
                dqNElems + dkNElems + dvNElems,
            )
    else:
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32()
            process_pre_from_meta(
                meta,
                fagOff,
                tensor_dq_ws_flat,
                tensor_dk_ws_flat,
                tensor_dv_ws_flat,
                dQ,
                dV,
                zero_ub.current(),
            )
            if sink == 1:
                process_sink_zero(
                    tensor_dsink_ws, constInfo, zero_ub.current(), pl.get_block_idx()
                )
    pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1()
        k_l1 = make_k_l1()
        k_l1_n = make_k_nd_l1()
        v_l1 = make_v_l1()
        do_l1 = make_do_l1()
        left = make_l0a()
        right = make_l0b()
        # 192 档 L0C 的两套布局由工厂内部按 template/d_align 自己选（BN2 用 dK/dV
        # 常驻表，GS1S2 用 acc 双槽表），调用点不传标志位。
        acc = make_l0c_acc()
        q_l1_dk = make_q_lag_view()
        do_l1_dv = make_do_lag_view()
        dk_acc0 = make_dk_l0c_chunk0()
        dv_acc0 = make_dv_l0c_chunk0()
        dk_acc1 = make_dk_l0c_chunk1()
        dv_acc1 = make_dv_l0c_chunk1()
        runInfos = make_run_infos()
        sched = pl.make_tuple(
            meta=meta,
            fagOff=fagOff,
            cBlockIdx=cBlockIdx,
            coreNum=coreNum,
            cuQ=cu_q_tensor,
            cuKv=cu_kv_tensor,
            usedQ=used_q_tensor,
            usedKv=used_kv_tensor,
            blockStart=blockStart,
            blockEnd=blockEnd,
            numBlocks=numBlocks,
        )
        gm = pl.make_tuple(
            tensor_q=tensor_q,
            tensor_k=tensor_k,
            tensor_k_nt=tensor_k_nt,
            tensor_dy=tensor_dy,
            tensor_v=tensor_v,
            tensor_dq_ws=tensor_dq_ws,
            tensor_dk_ws=tensor_dk_ws,
            tensor_dv_ws=tensor_dv_ws,
        )
        buf = pl.make_tuple(
            mm1_res=mm1_res,
            mm2_res=mm2_res,
            q_l1=q_l1,
            k_l1=k_l1,
            k_l1_n=k_l1_n,
            v_l1=v_l1,
            do_l1=do_l1,
            q_l1_dk=q_l1_dk,
            do_l1_dv=do_l1_dv,
            left=left,
            right=right,
            acc=acc,
            dk_acc0=dk_acc0,
            dv_acc0=dv_acc0,
            dk_acc1=dk_acc1,
            dv_acc1=dv_acc1,
            ds_l1=ds_l1,
            ds_t_l1=ds_t_l1,
            p_t_l1=p_t_l1,
        )
        if deter == 1:
            _run_preload3_cube_deter(constInfo, sched, gm, buf, runInfos)
        else:
            _run_preload3_cube_normal(constInfo, sched, gm, buf, runInfos)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = make_y_ub(v1w)
        dy_ub = make_dy_ub(v1w)
        nzp_ub = make_nzp_ub(v1w)
        sfmg_ub = make_sfmg_ub(v1w)
        lse_ub = make_lse_ub(v1w)
        sink_part_ub = make_sink_part_ub(v1w)
        mask_ub = make_mask_view(v1w)
        mask_pre_ub = make_mask_pre_view(v1w)

        runInfos = make_run_infos()
        sched = pl.make_tuple(
            meta=meta,
            fagOff=fagOff,
            cBlockIdx=cBlockIdx,
            coreNum=coreNum,
            cuQ=cu_q_tensor,
            cuKv=cu_kv_tensor,
            usedQ=used_q_tensor,
            usedKv=used_kv_tensor,
            blockStart=blockStart,
            blockEnd=blockEnd,
            numBlocks=numBlocks,
            subIdx=subIdx,
            v1w=v1w,
        )
        gm = pl.make_tuple(
            tensor_y=tensor_y,
            tensor_dy=tensor_dy,
            tensor_lse=tensor_lse,
            tensor_mask=tensor_mask,
            tensor_sinks=tensor_sinks,
            tensor_dsink_ws=tensor_dsink_ws,
            tensor_sfmg_ws=tensor_sfmg_ws,
        )
        buf = pl.make_tuple(
            mm1_res=mm1_res,
            mm2_res=mm2_res,
            y_ub=y_ub,
            dy_ub=dy_ub,
            nzp_ub=nzp_ub,
            sfmg_ub=sfmg_ub,
            lse_ub=lse_ub,
            sink_part_ub=sink_part_ub,
            mask_ub=mask_ub,
            mask_pre_ub=mask_pre_ub,
            ds_l1=ds_l1,
            p_l1=p_l1,
        )
        if deter == 1:
            _run_preload3_vector_deter(constInfo, sched, gm, buf, runInfos, sink)
        else:
            _run_preload3_vector_normal(constInfo, sched, gm, buf, runInfos, sink)

    # ==== Post 阶段：dQ/dK/dV 的 scale + cast fp32->16-bit ====
    # scale 在矩阵乘之后、fp32 结果上施加(而非提前乘进 dS)：先乘会让
    # 16-bit 舍入发生在不同量级上，网络训练中误差会累积。
    # 只有 dQ/dK 乘 scale，dV 不乘（传 1.0，fp32 乘 1.0 精确）。
    # 进 post 前必须 sync_all：要等所有核的 atomicAdd 都落盘。
    pl.system.sync_all()

    with pl.section_vector():
        vBlockIdxP = pl.get_block_idx()
        dqRowStartP = (
            pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdxP) * one64
        )
        dqRowEndP = pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdxP) * one64
        dkvRowStartP = (
            pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdxP) * one64
        )
        dkvRowEndP = (
            pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdxP) * one64
        )
        if nz_out == 1:
            post_f32 = make_nz_post_f32_buffer()
            post_f16 = make_nz_post_f16_buffer()
            process_nz_post(
                constInfo,
                cu_q_tensor,
                used_q_tensor,
                constInfo.sequsedQOptional,
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.n1Size,
                constInfo.s1Size,
                constInfo.coefB0Q,
                constInfo.scaleValue,
            )
            process_nz_post(
                constInfo,
                cu_kv_tensor,
                used_kv_tensor,
                constInfo.sequsedKvOptional,
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.n2Size,
                constInfo.s2Size,
                constInfo.coefB0KV,
                constInfo.scaleValue,
            )
            process_nz_post(
                constInfo,
                cu_kv_tensor,
                used_kv_tensor,
                constInfo.sequsedKvOptional,
                tensor_dv_ws_flat,
                tensor_dv_out,
                post_f32,
                post_f16,
                constInfo.n2Size,
                constInfo.s2Size,
                constInfo.coefB0KV,
                1.0,
            )
        else:
            post_f32 = make_post_f32_buffer()
            post_f16 = make_post_f16_buffer()
            process_post(
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dqRowStartP * dQ,
                (dqRowEndP - dqRowStartP) * dQ,
            )
            process_post(
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dkvRowStartP * dQ,
                (dkvRowEndP - dkvRowStartP) * dQ,
            )
            process_post(
                tensor_dv_ws_flat,
                tensor_dv_out,
                post_f32,
                post_f16,
                1.0,
                dkvRowStartP * dV,
                (dkvRowEndP - dkvRowStartP) * dV,
            )
        if sink == 1:
            if pl.get_block_idx() == 0:
                sink_red_ub = make_sink_red_post_ub(v1w)
                sink_row_ub = make_sink_row_post_ub(v1w)
                process_sink_reduce(
                    constInfo,
                    tensor_dsink_ws,
                    tensor_dsink_out,
                    sink_red_ub.current(),
                    sink_row_ub,
                    constInfo.aivCount,
                )


def flash_attn_grad_preload2(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # TND 变长：与 preload3 同款，循环外创建视图，非 TND 时为 null（不读）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
    deter,
    sink,
):
    """1-ahead 入口：建视图、pre/post，并按 template/deter 分发主循环。

    cube/vector 主循环已拆成 BN2、GS normal、GS deterministic 三套。
    路径按编译期 ``template`` / ``deter`` 分发，不是按 D 档。

    ``template!=Tmpl.BN2``（GS1S2，含 D=192/256/768 与 64/128 两档）：
    store 到 fp32 workspace；进循环前清 workspace，循环后再 scale+cast。
    不含 BN2 的 UB 直出（v5/v6）。
    ``Tmpl.BN2``：一个核独占整个 (b, n2) head；不跑全局 workspace pre/post
    （仅 ``bn2_need_zero`` 清 16-bit out，带 sink 时另清 dsink）。
    ``d_align<=128`` 的结果在 UB 里 Muls+Cast 直出；``d_align>128``（wide）只有
    经 4D fp32 workspace 的那部分由 vector 逐 D 分块读回 cast：dQ 三档都走
    workspace；dK 仅 256/768 走（192 驻留 L0C，flush 时搬 UB 再 cast）；dV 仅
    768 走（192/256 驻留 L0C，flush 时 cube fixpipe 直出）。首写仍用 AtomicNone
    覆盖，所以同样不需要全局清零。

    MultiBlk（is_bn2_multiblk）：核内按 s2 逐块累加，dQ 首有效 s2 覆盖
    workspace、其余 AtomicAdd，vector 在 ``dqLast`` 时从 4D workspace /
    per-core 2D 条带 load 后 Muls+Cast；dK/dV 在 L0C 累加到换 s2 再写回
    （dK 只有 256/768 档直接累加进 workspace；dV 192/256 仍走 L0C）。
    """
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey。BN2 is_valid 保证为 0，与原先写死 0 等价。
    cu_q_tensor = pl.make_tensor(cu_seqlens_q, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    cu_kv_tensor = pl.make_tensor(cu_seqlens_kv, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    used_q_tensor = pl.make_tensor(seqused_q, [tiling.b], [1], dtype=pl.DT_INT32)
    used_kv_tensor = pl.make_tensor(seqused_kv, [tiling.b], [1], dtype=pl.DT_INT32)
    # 与 preload3 共用逻辑长度 / GM 视图长度的初始化约定。
    constInfo = set_const_info(tiling, coreNum, cu_q_tensor, cu_kv_tensor)

    dQ = constInfo.dSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1ViewSize * d2dQ
    if template == Tmpl.BN2:
        # BN2 无 GQA，KV 与 Q 同轴；V/dO 仍按 dSize 建，与现网 BN2 一致。
        dV = constInfo.dSize
        viewD0KV = constInfo.viewD0Q
        viewD2KV = constInfo.viewD2Q
    else:
        dV = constInfo.dvSize
        viewD0KV = constInfo.viewD0KV
        viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2ViewSize * d2dKV
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1ViewSize * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2ViewSize * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=input_dtype,
    )
    tensor_k = pl.make_tensor(
        k,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # MM3 需要未转置的 K，必须与 MM2 的 K^T 视图分开(layout 各自推导)
    tensor_k_nt = pl.make_tensor(
        k,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    tensor_v = pl.make_tensor(
        v,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=input_dtype,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    # BN2 无 GQA（g=1），n1==n2，与原先 [B,n2,S1] 等价。
    n1Size = constInfo.n2Size * constInfo.gSize
    if layout == 1:
        # TND：LSE 为 [N1,T1]，head stride 必须是实际 token 总数。
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [n1Size, constInfo.t1Size],
            [constInfo.t1Size, 1],
            dtype=pl.DT_FP32,
        )
    else:
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [constInfo.bSize, n1Size, constInfo.s1Size],
            [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
            dtype=pl.DT_FP32,
        )

    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    # 与 preload3 / host 共用物理 token 口径；BN2 也用它定位 dsink 槽位。
    wsRowsQ = constInfo.t1Size * constInfo.n1Size
    wsRowsKV = constInfo.t2Size * constInfo.n2Size
    if template == Tmpl.BN2:
        # BN2 直出 16-bit。d_align<=128 的中间结果在 mm1/mm2 UB（MultiBlk 的
        # dQ 走 per-core 2D 条带）；d_align>128 放不下的段改走与 GS1S2 同布局的
        # 4D fp32 workspace：dQ 三档全走，dK 走 256/768，dV 只走 768（192 的
        # dK 与 192/256 的 dV 驻留 L0C）。workspace 只作本核 staging，不做全局
        # pre/post：cube 首写 AtomicNone、vector 内联 cast。
        tensor_dq_out = pl.make_tensor(
            dq,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=input_dtype,
        )
        if d_align > 128:
            # BN2 wide：dq/dk/dv 的 4D fp32 workspace。分段与 GS1S2 一致，
            # host BuildWorkspacePlan 不必改。
            dq_workspace = ws_ptr
            dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
            dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)
            tensor_dq_ws = pl.make_tensor(
                dq_workspace,
                [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
                [s1ndQ, d2dQ, dQ, 1],
                dtype=pl.DT_FP32,
            )
            tensor_dk_ws = pl.make_tensor(
                dk_workspace,
                [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
                [s2ndKV, d2dKV, dQ, 1],
                dtype=pl.DT_FP32,
            )
            tensor_dv_ws = pl.make_tensor(
                dv_workspace,
                [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
                [s2ndKVv, d2dKVv, dV, 1],
                dtype=pl.DT_FP32,
            )
            v1w = 256 if dv_align == 768 else dv_align
        else:
            # 单块 BN2 的 C3 被 tilingkey 剪掉 store，不能把 UB
            # 或 16-bit out 冒充成 fp32 GM 参数。dQ 用 per-core 条带。
            alignS1 = constInfo.s1Outer * CUBE_BASEM
            tensor_dq_ws = pl.make_tensor(
                ws_ptr, [constInfo.coreNum * alignS1, dQ], [dQ, 1], dtype=pl.DT_FP32
            )
    else:
        # GS1S2：三段 fp32 累加区，分段必须与 host CalcWorkSpace 对齐。
        dq_workspace = ws_ptr
        dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
        dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)
        tensor_dq_ws = pl.make_tensor(
            dq_workspace,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dk_ws = pl.make_tensor(
            dk_workspace,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=pl.DT_FP32,
        )
        dqElems = constInfo.t1Size * n1Size * dQ
        dkElems = constInfo.t2Size * constInfo.n2Size * dQ
        dvElems = constInfo.t2Size * constInfo.n2Size * dV
        tensor_dq_ws_flat = pl.make_tensor(
            dq_workspace, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP32
        )
        tensor_dk_ws_flat = pl.make_tensor(
            dk_workspace, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP32
        )
        tensor_dv_ws_flat = pl.make_tensor(
            dv_workspace, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP32
        )
        tensor_dq_out = pl.make_tensor(
            dq, [1, dqElems], [dqElems, 1], dtype=input_dtype
        )
        tensor_dk_out = pl.make_tensor(
            dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
        )
        tensor_dv_out = pl.make_tensor(
            dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
        )
        # V1 列宽：768 档工作 tile 仍是 256，其余等于 dv_align。
        v1w = 256 if dv_align == 768 else dv_align

    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask,
            [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1],
            dtype=pl.DT_UINT8,
        )
    else:
        tensor_mask = tensor_lse

    dsink_workspace = pl.addptr(
        ws_ptr,
        wsRowsQ * tiling.d
        + wsRowsKV * tiling.d
        + wsRowsKV * tiling.dv
        + 128 * pl.max(tiling.d, tiling.dv),
    )
    tensor_sinks = pl.make_tensor(sinks, [constInfo.n1Size], [1], dtype=pl.DT_FP32)
    dsinkElems = constInfo.aivCount * constInfo.n1Size
    tensor_dsink_ws = pl.make_tensor(
        dsink_workspace, [1, dsinkElems], [dsinkElems, 1], dtype=pl.DT_FP32
    )
    tensor_dsink_out = pl.make_tensor(
        dsink, [1, constInfo.n1Size], [constInfo.n1Size, 1], dtype=pl.DT_FP32
    )
    # sfmg 前移缓存：[1, N1*T1] fp32 扁平（head*T1 + 全局 token 行），与
    # host 的 plan.sfmg 对齐。是否真正写/读由运行时 constInfo.enablePreSfmg
    # 决定（host 按 fasg enablePreSfmg 判据裁决）：
    #   GS1S2 + dv_align>=128 —— 可能前移，建 ws 视图；
    #   BN2（含 multiblk）/ dv_align=64 —— 不前移，占位必须是 rank-2 扁平
    #     视图（非 TND 的 tensor_lse 是 [B,N1,S1] 三维，直接占位会因 load
    #     offsets 数与源 rank 不一致在 trace 期报 INVALID_SHAPE），内联
    #     分支不解引用。
    if template != Tmpl.BN2GS1S2 or dv_align < 128:
        lseElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            softmax_lse, [1, lseElems], [lseElems, 1], dtype=pl.DT_FP32
        )
    else:
        sfmgElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            pl.addptr(dsink_workspace, dsinkElems),
            [1, sfmgElems],
            [sfmgElems, 1],
            dtype=pl.DT_FP32,
        )

    # AIV get_block_idx 按 vector 子核计数，必须换算成物理核，否则
    # cube/vector 的 numBlocks 不一致，set/wait 配不上。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum
    meta = pl.make_tensor(metadata, [META_VIEW_LEN], [1], dtype=pl.DT_INT32)
    one64 = pl.const(1, pl.DT_INT64)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX) * one64
    blockStart = (
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx) * one64
    )
    blockEnd = pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx) * one64
    # swizzle==0（含 BN2）时就是 max(end-start, 0)。
    numBlocks = block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd)

    mm1_res = make_mm1_res()
    mm2_res = make_mm2_res()
    ds_l1 = make_ds_l1()
    p_l1 = make_p_l1()
    ds_t_l1 = make_ds_t_l1()
    p_t_l1 = make_p_t_l1()

    if template == Tmpl.BN2:
        # skip_invalid 不写整行/整列无效 tile，out GM 会留脏数据。
        # 三个 out 都清，不分行/列；不清 MultiBlk workspace。
        if bn2_need_zero == 1:
            with pl.section_vector():
                n1Size = constInfo.n2Size * constInfo.gSize
                dqElems = constInfo.t1Size * n1Size * constInfo.dSize
                dkElems = constInfo.t2Size * constInfo.n2Size * constInfo.dSize
                dvElems = constInfo.t2Size * constInfo.n2Size * constInfo.dvSize
                tensor_dq_flat = pl.make_tensor(
                    dq, [1, dqElems], [dqElems, 1], dtype=input_dtype
                )
                tensor_dk_flat = pl.make_tensor(
                    dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
                )
                tensor_dv_flat = pl.make_tensor(
                    dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
                )
                zero_ub = make_pre_zero_buffer_f16()
                process_pre_from_meta(
                    meta,
                    fagOff,
                    tensor_dq_flat,
                    tensor_dk_flat,
                    tensor_dv_flat,
                    constInfo.dSize,
                    constInfo.dvSize,
                    zero_ub.current(),
                )
                if sink == 1:
                    sink_zero_ub = make_pre_zero_buffer_f32()
                    process_sink_zero(
                        tensor_dsink_ws,
                        constInfo,
                        sink_zero_ub.current(),
                        pl.get_block_idx(),
                    )
            pl.system.sync_all()
        else:
            # No output pre-clear and no sink: BN2 needs no pre barrier.
            if sink == 1:
                with pl.section_vector():
                    sink_zero_ub = make_pre_zero_buffer_f32()
                    process_sink_zero(
                        tensor_dsink_ws,
                        constInfo,
                        sink_zero_ub.current(),
                        pl.get_block_idx(),
                    )
                pl.system.sync_all()
        # BN2（含 multiblk）不做 sfmg 前移：对齐 fasg enablePreSfmg 的
        # splitAxis 门（仅 BN2GS1S2 / BN2S2；FAG 无 BN2S2）。BN2 的 rowsum
        # 仍在主循环 V1 内联，多 s2Outer 的重复计算保留，行为与 fasg 一致。
        # 上面的 need_zero/sink 子支各自带 sync_all（或空跑无清零），进主
        # 循环前无需额外 barrier。
    else:
        # GS1S2：三段 fp32 累加区清零；dv_align>=128 走交错 pre，是否追加
        # softmaxGrad 前移由运行时 constInfo.enablePreSfmg 决定（对齐 fasg
        # enablePreSfmg）：为 1 时一次算全 rowsum(dO⊙O) 存 ws、V1 只读，
        # 为 0 时主循环 V1 内联重算（本分支只做清零）。
        # 交错 pre（同 preload3）：zt 32KB 避开 sp 的窗口，无 UB 冒险；
        # zero 与 sp 不同 mutex，框架不插 wait——清零写(MTE3)与 presfmg
        # 载入(MTE2)/VF 并行，且分块交替 issue（见 process_presfmg 的
        # _zero_step）。本函数（preload2）的 GS1S2 租户是 256/768 档——sp
        # 用半窗塞进空洞。
        if dv_align >= 128:
            with pl.section_vector():
                zero_ub = make_pre_zero_buffer_f32()
                zt = zero_ub.current()
                # 行区间与三段边界（同 preload3，见 process_pre_from_meta 文档）。
                vBlockIdx = pl.get_block_idx()
                one64 = pl.const(1, pl.DT_INT64)
                dqRowStart = (
                    pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx) * one64
                )
                dqRowEnd = (
                    pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx) * one64
                )
                dkvRowStart = (
                    pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx)
                    * one64
                )
                dkvRowEnd = (
                    pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx) * one64
                )
                dqNElems = (dqRowEnd - dqRowStart) * dQ
                dkNElems = (dkvRowEnd - dkvRowStart) * dQ
                dvNElems = (dkvRowEnd - dkvRowStart) * dV
                preElems = pl.min(
                    PRE_ELEMS[d_align], pl.max(dqNElems, pl.max(dkNElems, dvNElems))
                )
                fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
                if sink == 1:
                    process_sink_zero(
                        tensor_dsink_ws, constInfo, zt, pl.get_block_idx()
                    )
                sp_y_ub = make_sp_y_ub(v1w)
                sp_dy_ub = make_sp_dy_ub(v1w)
                sp_out_ub = make_sp_out_ub(v1w)
                sp_out_flat = make_sp_out_flat(v1w)
                process_presfmg(
                    constInfo,
                    tensor_y,
                    tensor_dy,
                    tensor_sfmg_ws,
                    sp_y_ub,
                    sp_dy_ub,
                    sp_out_ub,
                    sp_out_flat,
                    fag_common.SP_WORK_W[v1w],
                    zt,
                    tensor_dq_ws_flat,
                    tensor_dk_ws_flat,
                    tensor_dv_ws_flat,
                    dqRowStart * dQ,
                    dqNElems,
                    dkvRowStart * dQ,
                    dqNElems + dkNElems,
                    dkvRowStart * dV,
                    dqNElems + dkNElems + dvNElems,
                )
        else:
            with pl.section_vector():
                zero_ub = make_pre_zero_buffer_f32()
                process_pre_from_meta(
                    meta,
                    fagOff,
                    tensor_dq_ws_flat,
                    tensor_dk_ws_flat,
                    tensor_dv_ws_flat,
                    dQ,
                    dV,
                    zero_ub.current(),
                )
                if sink == 1:
                    process_sink_zero(
                        tensor_dsink_ws,
                        constInfo,
                        zero_ub.current(),
                        pl.get_block_idx(),
                    )
        pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1()
        k_l1 = make_k_l1()
        k_l1_n = make_k_nd_l1()
        v_l1 = make_v_l1()
        do_l1 = make_do_l1()
        left = make_l0a()
        right = make_l0b()
        # 192 档 L0C 的两套布局由工厂内部按 template/d_align 选（BN2 用 dK/dV
        # 常驻 256KB，GS1S2 用 acc 双槽 224KB）；其它档两套指向同一份布局。
        acc = make_l0c_acc()
        if template == Tmpl.BN2:
            # dK/dV 每个 D 分块一个槽（c0 = 第 0 块、c1 = 第 1 块）；64/128/768 只有
            # 一个分块，c1 别名到 c0、trace 期剪掉。L0C 占用（单位 KB，额 256KB）：
            #   64/128   : acc 2×64 + dK 1×64 + dV 1×64 = 256，各自常驻累加
            #   192 BN2  : acc 1×64 + dK 2×48 + dV 2×48 = 256，dK/dV 常驻 L0C
            #   192 GS1S2: acc 2×64 + dV 2×48 = 224，dK 与 acc 同址（mm4 → fp32 ws）
            #   256      : acc 2×64 + dV 2×64 = 256，dK 与 acc 第 0 槽同址（→ fp32 ws）
            #   768      : acc 2×64 + dK 1×64 + dV 1×64 = 256，dK/dV 单 scratch → fp32 ws
            dk_c0 = make_dk_l0c_chunk0()
            dk_c1 = make_dk_l0c_chunk1()
            dv_c0 = make_dv_l0c_chunk0()
            dv_c1 = make_dv_l0c_chunk1()
            # BN2 192档使用，dK 从 L0C 搬到 UB，chunk0 → mm2_res、chunk1 → l0c_load_ub
            l0c_load_ub = make_l0c_load_ub()
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
            )
            if d_align > 128:
                gm = pl.make_tuple(
                    tensor_q=tensor_q,
                    tensor_k=tensor_k,
                    tensor_k_nt=tensor_k_nt,
                    tensor_dy=tensor_dy,
                    tensor_v=tensor_v,
                    tensor_dq_ws=tensor_dq_ws,
                    tensor_dk_ws=tensor_dk_ws,
                    tensor_dv_ws=tensor_dv_ws,
                    tensor_dv_out=tensor_dv_out,
                )
            else:
                gm = pl.make_tuple(
                    tensor_q=tensor_q,
                    tensor_k=tensor_k,
                    tensor_k_nt=tensor_k_nt,
                    tensor_dy=tensor_dy,
                    tensor_v=tensor_v,
                    tensor_dq_ws=tensor_dq_ws,
                    tensor_dv_out=tensor_dv_out,
                )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                q_l1=q_l1,
                k_l1=k_l1,
                k_l1_n=k_l1_n,
                v_l1=v_l1,
                do_l1=do_l1,
                left=left,
                right=right,
                acc=acc,
                dk_c0=dk_c0,
                dk_c1=dk_c1,
                dv_c0=dv_c0,
                dv_c1=dv_c1,
                l0c_load_ub=l0c_load_ub,
                ds_l1=ds_l1,
                ds_t_l1=ds_t_l1,
                p_t_l1=p_t_l1,
            )
            _run_preload2_cube_bn2(constInfo, sched, gm, buf, runInfos)
        elif deter == 1:
            q_l1_dk = make_q_lag_view()
            do_l1_dv = make_do_lag_view()
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
            )
            gm = pl.make_tuple(
                tensor_q=tensor_q,
                tensor_k=tensor_k,
                tensor_k_nt=tensor_k_nt,
                tensor_dy=tensor_dy,
                tensor_v=tensor_v,
                tensor_dq_ws=tensor_dq_ws,
                tensor_dk_ws=tensor_dk_ws,
                tensor_dv_ws=tensor_dv_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                q_l1=q_l1,
                k_l1=k_l1,
                k_l1_n=k_l1_n,
                v_l1=v_l1,
                do_l1=do_l1,
                q_l1_dk=q_l1_dk,
                do_l1_dv=do_l1_dv,
                left=left,
                right=right,
                acc=acc,
                ds_l1=ds_l1,
                ds_t_l1=ds_t_l1,
                p_t_l1=p_t_l1,
            )
            _run_preload2_cube_gs_deter(constInfo, sched, gm, buf, runInfos)
        else:
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
            )
            gm = pl.make_tuple(
                tensor_q=tensor_q,
                tensor_k=tensor_k,
                tensor_k_nt=tensor_k_nt,
                tensor_dy=tensor_dy,
                tensor_v=tensor_v,
                tensor_dq_ws=tensor_dq_ws,
                tensor_dk_ws=tensor_dk_ws,
                tensor_dv_ws=tensor_dv_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                q_l1=q_l1,
                k_l1=k_l1,
                k_l1_n=k_l1_n,
                v_l1=v_l1,
                do_l1=do_l1,
                left=left,
                right=right,
                acc=acc,
                ds_l1=ds_l1,
                ds_t_l1=ds_t_l1,
                p_t_l1=p_t_l1,
            )
            _run_preload2_cube_gs_normal(constInfo, sched, gm, buf, runInfos)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        if template == Tmpl.BN2 and d_align > 128:
            y_ub = make_y_ub(v1w)
            dy_ub = make_dy_ub(v1w)
            nzp_ub = make_nzp_ub(v1w)
            sfmg_ub = make_sfmg_ub(v1w)
            lse_ub = make_lse_ub(v1w)
            mask_ub = make_mask_view(v1w)
            mask_pre_ub = make_mask_pre_view(v1w)
            # gm_ws -> ws_load_ub -> muls_cast_vf -> ws_cast_ub -> gm_out
            ws_load_ub = make_ws_load_f32_ub()
            ws_cast_ub = make_ws_cast_f16_ub()
            # BN2 192档使用，dK 从 L0C 搬到 UB，chunk0 → mm2_res、chunk1 → l0c_load_ub
            l0c_load_ub = make_l0c_load_ub()
            l0c_cast_ub = make_l0c_cast_ub()
            sink_part_ub = make_sink_part_ub(v1w)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
                subIdx=subIdx,
                dv_align=dv_align,
                v1w=v1w,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_dq_out=tensor_dq_out,
                tensor_dk_out=tensor_dk_out,
                tensor_dv_out=tensor_dv_out,
                tensor_dq_ws=tensor_dq_ws,
                tensor_dk_ws=tensor_dk_ws,
                tensor_dv_ws=tensor_dv_ws,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                nzp_ub=nzp_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ws_load_ub=ws_load_ub,
                ws_cast_ub=ws_cast_ub,
                l0c_cast_ub=l0c_cast_ub,
                l0c_load_ub=l0c_load_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_bn2(constInfo, sched, gm, buf, runInfos, sink)
        elif template == Tmpl.BN2:
            # narrow：d==dv 由 host 门 Bn2FamilyFeasible 保证，故 v1w == d_align。
            # 这里一律传 d_align（tilingkey 常量）而不是 v1w：pypto 的定值分析认为
            # v1w 在本分支「可能未定义」（旧代码此分支传的是常量 HEAD_DIM_ALIGN，
            # 所以没暴露过），会报 NameNotFound: __inline_0_v1w。
            # 地址档由各工厂内部按 template/d_align 判成 HEAD_DIM_ALIGN。
            y_ub = make_y_ub(d_align)
            dy_ub = make_dy_ub(d_align)
            nzp_ub = make_nzp_ub(d_align)
            sfmg_ub = make_sfmg_ub(d_align)
            lse_ub = make_lse_ub(d_align)
            mask_ub = make_mask_view(d_align)
            mask_pre_ub = make_mask_pre_view(d_align)
            l0c_cast_ub = make_l0c_cast_ub()
            # 同上：narrow 不实际使用，但 buf 字段要齐（函数顶部无条件 unpack）
            l0c_load_ub = make_l0c_load_ub()
            ws_cast_ub = make_ws_cast_f16_ub()
            sink_part_ub = make_sink_part_ub(d_align)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
                subIdx=subIdx,
                dv_align=dv_align,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_dq_out=tensor_dq_out,
                tensor_dk_out=tensor_dk_out,
                tensor_dq_ws=tensor_dq_ws,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                nzp_ub=nzp_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ws_cast_ub=ws_cast_ub,
                l0c_cast_ub=l0c_cast_ub,
                l0c_load_ub=l0c_load_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_bn2(constInfo, sched, gm, buf, runInfos, sink)
        elif deter == 1:
            y_ub = make_y_ub(v1w)
            dy_ub = make_dy_ub(v1w)
            nzp_ub = make_nzp_ub(v1w)
            sfmg_ub = make_sfmg_ub(v1w)
            lse_ub = make_lse_ub(v1w)
            sink_part_ub = make_sink_part_ub(v1w)
            mask_ub = make_mask_view(v1w)
            mask_pre_ub = make_mask_pre_view(v1w)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                subIdx=subIdx,
                v1w=v1w,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                nzp_ub=nzp_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_gs_deter(constInfo, sched, gm, buf, runInfos, sink)
        else:
            y_ub = make_y_ub(v1w)
            dy_ub = make_dy_ub(v1w)
            nzp_ub = make_nzp_ub(v1w)
            sfmg_ub = make_sfmg_ub(v1w)
            lse_ub = make_lse_ub(v1w)
            sink_part_ub = make_sink_part_ub(v1w)
            mask_ub = make_mask_view(v1w)
            mask_pre_ub = make_mask_pre_view(v1w)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
                subIdx=subIdx,
                v1w=v1w,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                nzp_ub=nzp_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_gs_normal(constInfo, sched, gm, buf, runInfos, sink)

    if template != Tmpl.BN2:
        pl.system.sync_all()
        with pl.section_vector():
            vBlockIdxP = pl.get_block_idx()
            dqRowStartP = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdxP) * one64
            )
            dqRowEndP = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdxP) * one64
            )
            dkvRowStartP = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdxP) * one64
            )
            dkvRowEndP = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdxP) * one64
            )
            post_f32 = make_post_f32_buffer()
            post_f16 = make_post_f16_buffer()
            process_post(
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dqRowStartP * dQ,
                (dqRowEndP - dqRowStartP) * dQ,
            )
            process_post(
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dkvRowStartP * dQ,
                (dkvRowEndP - dkvRowStartP) * dQ,
            )
            process_post(
                tensor_dv_ws_flat,
                tensor_dv_out,
                post_f32,
                post_f16,
                1.0,
                dkvRowStartP * dV,
                (dkvRowEndP - dkvRowStartP) * dV,
            )
            if sink == 1:
                if pl.get_block_idx() == 0:
                    sink_red_ub = make_sink_red_post_ub(v1w)
                    sink_row_ub = make_sink_row_post_ub(v1w)
                    process_sink_reduce(
                        constInfo,
                        tensor_dsink_ws,
                        tensor_dsink_out,
                        sink_red_ub.current(),
                        sink_row_ub,
                        constInfo.aivCount,
                    )
    if template == Tmpl.BN2:
        # BN2 没有全局 workspace post：dQ/dK/dV 都在本核内联 scale/cast 落盘；
        # 只有 dsink 需要一次全局归约。
        if sink == 1:
            pl.system.sync_all()
            with pl.section_vector():
                if pl.get_block_idx() == 0:
                    sink_red_ub = make_sink_red_post_ub(HEAD_DIM_ALIGN)
                    sink_row_ub = make_sink_row_post_ub(HEAD_DIM_ALIGN)
                    process_sink_reduce(
                        constInfo,
                        tensor_dsink_ws,
                        tensor_dsink_out,
                        sink_red_ub.current(),
                        sink_row_ub,
                        constInfo.aivCount,
                    )
