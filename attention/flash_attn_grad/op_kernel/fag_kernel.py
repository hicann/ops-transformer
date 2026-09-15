# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""两个分核模板的主循环。

本文件只负责**编排**：取下一个有效块、推进 CV ping-pong、在阶段之间插
cross-core set/wait、决定谁在哪一拍消费谁的产出。真正的计算全部在
:mod:`fag_block_cube` 与 :mod:`fag_block_vec`，分核映射在 :mod:`fag_schedule`，
片上地址在 :mod:`fag_mem` —— 这一层不该出现裸地址，也不该出现 matmul。

两个模板：

* :func:`flash_attn_grad_bn2gs1s2` —— 通用切分，cube 领先 vector 两个 task
  (``PRELOAD_TIMES=3``)，dQ 走 fp32 workspace + atomicAdd，dK/dV 在 L0C 跨 s1
  累加。
* :func:`flash_attn_grad_bn2` —— 一个核独占一个 (b, n2) head，1-ahead
  ping-pong，无 pre/post。

CV 通信全部使用手动 set/wait cross-core，且正反向成对：

  正向 C->V : SYNC_C2_TO_V2_FLAG[2] / SYNC_C1_TO_V2_FLAG[2]  (mm2/mm1 结果就绪)
  正向 V->C : SYNC_V3_TO_C3_FLAG / SYNC_V4_TO_C5_FLAG        (dS/P 已落 L1)
  反向 C->V : SYNC_C4_TO_V3_FLAG / SYNC_C5_TO_V4_FLAG        (L1 已被 cube 读完)
  反向 V->C : SYNC_V2_TO_C1_FLAG[2] / SYNC_V2_TO_C2_FLAG[2]  (mm1/mm2 UB 已读完)

AIC 侧对每个 V->C flag 用 ``sync_mode=INTRA_BLOCK`` 等一次，覆盖两个
vector 子核各自发来的信号。

以下是这两个模板共同的设计约束，改动前务必读完。

当前裁剪范围：非 TND（BSND/BNSD）、FP16。mask_mode 0=dense，3=causal，
4=band。tilingkey ``swizzle`` 区分线性 skip 无效块与 packed 有效块分核。
BN2 支持 mask，禁止 BN2+swizzle。

dQ/dK/dV 均由 fixpipe 经 GM workspace 做 atomicAdd 累加。dK/dV 的 L0C
驻留累加作为后续优化单独叠加。mask_mode 静态分支：0 恒有效，3=causal
块有效，4=band 区间；线性路径 skip 无效块。主循环是两槽 taskId
ping-pong。

反向同步由 needSyncDkMM 控制：首轮 L1 尚无人读，无需等待。

数据流：
  MM2(Q@K^T)->UB -> V2(softmax->P) -> V4(P cast+ND2NZ->L1)
  MM1(dO@V^T)->UB -> V3(dS=(dP-sfmg)*P cast+ND2NZ->L1)
  -> MM5(P^T@dO->dV) -> MM3(dS@K->dQ) -> MM4(dS^T@Q->dK)
V1 计算 softmaxGradFront sfmg = rowsum(dy * y)，取自前向输出 y。

每个 GM tensor 只推导出一种 Layout：tensor_k 因 MM2 的转置读(order=[3,1])
被定为 Layout::DN，MM3 需要非转置的 K，若共用同一 tensor，plain 读会被按 DN
解释而静默拿到 K^T(dQ 幅度正确但数值全错)。故 MM3 用独立的 tensor_k_nt 视图
指向同一 buffer，各自推导 layout。平台仅支持
ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN，不支持 ND2ZN。

MM4/MM5 的左矩阵需要转置(dS^T / P^T)。L1->L0A 不能随路转置（NZ 版本静默
算错、ZZ 版本无法编译，L0A 要求非 row-major fractal），因此显式走
transpose(UB) -> ND2NZ -> insert 到 [N,M] NZ 的 L1，再常规载入 L0A。
"""

import inspect

import pypto_pro.language as pl

import fag_block_cube
import fag_block_vec
import fag_buffers
import fag_common
import fag_schedule
from fag_tiling import FlashAttnGradTilingData

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
        fag_schedule,
    ):
        for name, val in vars(mod).items():
            if name.startswith("_") or inspect.ismodule(val):
                continue
            if isinstance(val, list) and name not in _PINGPONG_LISTS:
                continue
            g[name] = val


_bind_helpers()


def flash_attn_grad_bn2gs1s2(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # 以下 6 个为可选输入。cu_seqlens/seqused(TND)、sinks 本轮不读。
    # attn_mask：mask_mode 3/4 时按 [2048,2048] uint8 压缩矩阵搬 UB。
    # metadata：FAG 区含分核 [start,end) 与 pre/post 行区间（412 int32）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    # 3 个输出
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    # 单块 workspace，内部再切三份 fp32 累加区（见下方 make_ptr 分区）
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
):
    # coreNum 从内建 API 取而非 tiling：TilingData 无此字段。
    # get_block_num() 是 host SetBlockDim 启动的 cube 数（dense 下等于
    # metadata blockOuter），cube/vector 两侧同一口径，set/wait 才能配对。
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey，是编译期常量 —— 每种分核各生成一份二进制，
    # `if constInfo.enableSwizzle` 的两路在编译期折叠，运行期没有判断开销。
    constInfo = set_const_info(tiling, coreNum, swizzle)

    # ---- workspace 分区 ----
    # host(GetWorkspaceSizes) 给的是单块连续内存，这里按 fp32 元素切成
    # dq/dk/dv 三段累加区。三段大小必须与 host 的 CalcWorkSpace 逐一对齐。
    # dq 段按 n1(=n2*G) 个 head，dk/dv 段按 n2 —— 与 host 的 dqElems/dkvElems
    # 一致。GQA 下若这里误用 n2，dk/dv 的起始偏移就会算错。
    # host 只在三段之后多留一行 128 对齐，不改这里的基地址。
    wsRowsQ = tiling.b * tiling.s1 * tiling.n1
    wsRowsKV = tiling.b * tiling.s2 * tiling.n2
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dq_workspace = ws_ptr
    # 段宽按各自的 head dim：dq/dk 是 D，dv 是 Dv(可以短于 D)。
    # 必须与 host 的 dqElems/dkElems/dvElems 逐一对齐。
    dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
    dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)

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
    s1ndQ = constInfo.s1Size * d2dQ
    viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2Size * d2dKV
    # Dv 宽的那几个张量另有一套行距
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1Size * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2Size * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=pl.DT_FP16,
    )
    tensor_k = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=pl.DT_FP16,
    )
    # MM3(dQ = dS@K) 需要「未转置」的 K，而 MM2 需要 K^T。
    # 每个 GM tensor 只会推导出一种 Layout：tensor_k 因 MM2 的转置读被定成
    # Layout::DN，MM3 若共用它，plain 读也会被按 DN 解释，实测等价于拿到 K^T
    # (partial 与 K[s2o]^T 吻合到 1e-4)。故为 MM3 单独建一份同 buffer 的
    # tensor 视图，让它独立推导 layout。
    tensor_k_nt = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=pl.DT_FP16,
    )
    # V/dO/attn_out 是 Dv 宽
    tensor_v = pl.make_tensor(
        v,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=pl.DT_FP16,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=pl.DT_FP16,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=pl.DT_FP16,
    )
    # softmax_lse : [B,N1,S1] —— 按 Q 的 head 数(n1 = n2*G)存，因为每个 Q head
    # 的 softmax 归一化因子不同，G 个 Q head 不能共享同一份 lse。
    # lse offset = ((b*n2 + n2o)*G + g)*S1。
    n1Size = constInfo.n2Size * constInfo.gSize
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
    # dq/dk/dv workspace 为 fp32，供 atomicAdd 累加
    # dq/dk 是 D 宽，dv 是 Dv 宽 —— 与 workspace 的分段宽度一致
    tensor_dq_ws = pl.make_tensor(
        dq_workspace,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dk_ws = pl.make_tensor(
        dk_workspace,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dv_ws = pl.make_tensor(
        dv_workspace,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=pl.DT_FP32,
    )

    # ---- pre/post：一维视图 ----
    # 清零 / Muls+Cast 都按 B*S*N*D 个元素线性切，不看行宽。
    # dq 侧 head 数是 n1(=n2*G)，dk/dv 侧是 n2。
    dqRows = constInfo.bSize * constInfo.s1Size * n1Size
    dkvRows = constInfo.bSize * constInfo.s2Size * constInfo.n2Size
    dqElems = dqRows * dQ
    dkElems = dkvRows * dQ
    dvElems = dkvRows * dV
    tensor_dq_ws_flat = pl.make_tensor(
        dq_workspace, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP32
    )
    tensor_dk_ws_flat = pl.make_tensor(
        dk_workspace, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP32
    )
    tensor_dv_ws_flat = pl.make_tensor(
        dv_workspace, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP32
    )
    tensor_dq_out = pl.make_tensor(dq, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP16)
    tensor_dk_out = pl.make_tensor(dk, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP16)
    tensor_dv_out = pl.make_tensor(dv, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP16)
    # V1 列宽：64 档收到 64；其它档跟 dv_align（y/dy 是 Dv 宽）。
    v1w = 64 if d_align == 64 else dv_align

    # cBlockIdx 必须按物理核取：AIV 侧 get_block_idx() 按 vector 子核计数
    # (0..2*coreNum-1)，AIC 侧按物理核计数。两侧口径不一致会算出不同的
    # numBlocks 不同 -> 循环次数不同 -> set/wait 配不上 -> 死锁。
    # 核数直接用 tiling 传入，避免再依赖 get_block_num 的口径。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum

    # 分核区间取自 metadata。swizzle 支仍走公式，原因见 block_range_of。
    meta = pl.make_tensor(metadata, [META_VIEW_LEN], [1], dtype=pl.DT_INT32)
    fagOff = pl.astype(pl.getval(meta, META_HEAD_FAG_START_INDEX), pl.DT_INT64)
    blockStart = pl.astype(
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx), pl.DT_INT64
    )
    blockEnd = pl.astype(
        pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx), pl.DT_INT64
    )
    numBlocks = block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd)

    # ---- CV 共享区：mm1/mm2 结果 UB + dS/P 的 L1 及其 ZN 转置视图 ----
    # 对应 kernel_base.h 的「CV核间共享Buffer」，vector 侧写、cube 侧读，
    # 故必须声明在两个 section 之外。
    mm1_res = make_mm1_res()
    mm2_res = make_mm2_res()
    ds_l1 = make_ds_l1()
    p_l1 = make_p_l1()
    ds_t_l1 = make_ds_t_l1()
    p_t_l1 = make_p_t_l1()

    # ==== Pre 阶段：清零 dq/dk/dv workspace ====
    # 三个 workspace 都用 atomicAdd 累加，进主循环前必须为 0。
    # 清完用 sync_all 隔开：主循环的 atomicAdd 必须看到已清零的 GM。
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
    pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1()
        k_l1 = make_k_l1()
        k_l1_n = make_k_nd_l1()
        v_l1 = make_v_l1()
        do_l1 = make_do_l1()
        q_l1_dk = make_q_lag_view()
        do_l1_dv = make_do_lag_view()
        left = make_l0a()
        right = make_l0b()
        acc = make_l0c_acc()
        dk_acc0 = make_dk_l0c_chunk0()
        dv_acc0 = make_dv_l0c_chunk0()
        dk_acc1 = make_dk_l0c_chunk1()
        dv_acc1 = make_dv_l0c_chunk1()

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIC 侧) ----------------
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        nFetched = 0
        # cube 领先 vector 两个 task。gIdx<0 后再排空两轮，与 vector 对齐。
        while True:
            # 无条件轮转，保证与 vector 侧步调一致（末轮不做 mm 也要转）
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            # Q/dO 的 double buffer 无条件轮转，供本轮 mm2/mm1。
            # mm4/mm5 用 q_l1_dk/do_l1_dv —— 同地址、同 mutex_ids 的延后
            # 游标视图，直接复用这里载入的数据(见其声明处说明)。
            q_cur = q_l1.next()
            do_cur = do_l1.next()
            if isLastLoop == 0:
                gIdx = next_work_gidx(
                    constInfo,
                    cBlockIdx,
                    coreNum,
                    blockStart,
                    blockEnd,
                    localIdx,
                    denseCursor,
                    numBlocks,
                )
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    if swizzle == 0:
                        denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % PRELOAD_TIMES]
                    set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
                    nFetched = nFetched + 1
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, PRELOAD_TIMES)

                    # ---- mm2: Q@K^T ----
                    # 反向等待：mm2 只有 2 个 UB 槽位，task N 复用 task N-2 的
                    # 那块，必须等 vector 读完（taskId>=2 才 wait
                    # SYNC_V2_TO_C2_FLAG）。
                    # taskId>=2 才有 N-2 存在；首两轮槽位是干净的，直接发。
                    # 这就是「首轮连发四次 matmul」：taskId 0/1 两轮各发
                    # mm2+mm1，累计 4 次，把两个槽位填满后 vector 才开始消费，
                    # 于是第 N 组 cube 计算与第 N-2 组 vector 计算重叠。
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C2_FLAG[taskId % 2],
                        )
                    iterate_mm_qk(
                        constInfo,
                        curInfo,
                        tensor_q,
                        tensor_k,
                        q_cur,
                        k_l1,
                        left,
                        right,
                        acc,
                        mm2_cur,
                        curInfo.loadKV,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2]
                    )
                    # ---- mm1: dO@V^T ----
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C1_FLAG[taskId % 2],
                        )
                    iterate_mm_dyv(
                        constInfo,
                        curInfo,
                        tensor_dy,
                        tensor_v,
                        do_cur,
                        v_l1,
                        left,
                        right,
                        acc,
                        mm1_cur,
                        curInfo.loadKV,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2]
                    )

            # mm3/mm4/mm5 消费的是 vector 刚产出的 dS/P，对应 task N-2
            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # 正向：等 vector 把 P / dS 写入 L1
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
                )
                # ---- mm5: dV = P^T@dO (L0C 驻留累加) ----
                # dkvAcc=1: 与上一轮同一个 s2 块 -> 在 L0C 上累加
                # prevInfo 对应 task N-2 的 block 下标。
                # 排空阶段不再发新块，用 taskId-2 对应当年那一拍。
                prevIdx = taskId - 2
                # 延后游标在此推进：taskId>1 才走到这里，故它恒落后两拍
                q_prev = q_l1_dk.next()
                do_prev = do_l1_dv.next()
                iterate_mm_pdy(
                    constInfo,
                    prevInfo,
                    p_t_l1.current(),
                    do_prev,
                    left,
                    right,
                    dv_acc0,
                    dv_acc1,
                    prevInfo.dkvAcc,
                )
                # 反向：P 的 L1 已读完，放行 vector 覆写
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG
                )
                # ---- mm3: dQ = dS@K (仍直出 GM，dQ 按 s1 分块无法驻留) ----
                iterate_mm_dsk(
                    constInfo,
                    prevInfo,
                    tensor_k_nt,
                    tensor_dq_ws,
                    prevInfo.loadKV,
                    ds_l1.current(),
                    k_l1_n,
                    left,
                    right,
                    acc,
                )

                # ---- mm4: dK = dS^T@Q (L0C 驻留累加) ----
                iterate_mm_dsq(
                    constInfo,
                    prevInfo,
                    ds_t_l1.current(),
                    q_prev,
                    left,
                    right,
                    dk_acc0,
                    dk_acc1,
                    prevInfo.dkvAcc,
                )

                # s2 块即将切换(或已是最后一轮)时，把 L0C 上的 dK/dV 落 GM。
                # 核可能在列中途结束，末块必须无条件落盘。
                if prevIdx >= nFetched - 1 or prevInfo.flushDkv == 1:
                    flush_dkv(
                        constInfo,
                        prevInfo,
                        tensor_dk_ws,
                        tensor_dv_ws,
                        dk_acc0,
                        dk_acc1,
                        dv_acc0,
                        dv_acc1,
                    )
                # 反向：dS 的 L1 已读完
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG
                )

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 2:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = make_y_ub(v1w)
        dy_ub = make_dy_ub(v1w)
        nzp_ub = make_nzp_ub(v1w)
        sfmg_ub = make_sfmg_ub(v1w)
        lse_ub = make_lse_ub(v1w)
        mask_ub = make_mask_view(v1w)
        mask_pre_ub = make_mask_pre_view(v1w)

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIV 侧) ----------------
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        while True:
            # 无条件轮转，与 cube 侧一一对应
            mm2_res.next()
            mm1_res.next()
            # vector 落后 cube 两个 task：消费 runInfos[(taskId+1)%3] 即 N-2
            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # ---- v1: softmaxGrad sfmg = rowsum(dy*y) ----
                process_vec1(
                    constInfo,
                    prevInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    y_ub.current(),
                    dy_ub.current(),
                    sfmg_ub.current(),
                    v1w,
                )

            if isLastLoop == 0:
                gIdx = next_work_gidx(
                    constInfo,
                    cBlockIdx,
                    coreNum,
                    blockStart,
                    blockEnd,
                    localIdx,
                    denseCursor,
                    numBlocks,
                )
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    if swizzle == 0:
                        denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % PRELOAD_TIMES]
                    set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, PRELOAD_TIMES)
                    lse_ub.next()

            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # 消费 cube 在 task N-2 写入的槽位。mm1/mm2 只有 2 槽，
                # task N 与 N-2 同奇偶 -> 同一块物理槽位，故这里要取
                # current()(与 cube 侧 next() 返回的同一块)，
                # 而不是 lag=1 时用的 previous()
                mm1_t = mm1_res.current()
                mm2_t = mm2_res.current()
                # ---- BN2GS1S2 last vec ----
                # 正向：等 mm2 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2]
                )
                if mask_mode != 0:
                    process_vec2(
                        constInfo,
                        prevInfo,
                        subIdx,
                        tensor_lse,
                        mm2_t,
                        lse_ub.previous(),
                        tensor_mask,
                        mask_ub.current(),
                        mask_pre_ub.current(),
                    )
                else:
                    process_vec2(
                        constInfo,
                        prevInfo,
                        subIdx,
                        tensor_lse,
                        mm2_t,
                        lse_ub.previous(),
                        tensor_lse,
                        mask_ub.current(),
                        mask_pre_ub.current(),
                    )
                # 正向：等 mm1 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2]
                )

                # ---- last vec：P / dS 落 L1 ----
                # 反向：等 cube 读完上一轮 P L1 再覆写。
                # needSyncDkMM = taskId > 2：已经历过一轮完整 CV，L1 才有人读过。
                needSyncDkMM = taskId > 2
                if needSyncDkMM:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                process_vec4(
                    constInfo, prevInfo, subIdx, p_l1.current(), mm2_t, nzp_ub.current()
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V4_TO_C5_FLAG
                )
                # 反向：等 cube 读完上一轮 dS L1
                if needSyncDkMM:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                process_vec3(
                    constInfo,
                    prevInfo,
                    subIdx,
                    ds_l1.current(),
                    mm1_t,
                    mm2_t,
                    sfmg_ub.current(),
                    nzp_ub.current(),
                )
                # 反向：mm1/mm2 的 UB 槽位已读完，放行 cube 覆写。
                # 下标与刚消费的 task N-2 一致，cube 侧 task N 等的正是它
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_V2_TO_C1_FLAG[taskId % 2]
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_V2_TO_C2_FLAG[taskId % 2]
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG
                )

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 2:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1

    # ==== Post 阶段：dQ/dK/dV 的 scale + cast fp32->fp16 ====
    # scale 在矩阵乘之后、fp32 结果上施加(而非提前乘进 dS)：先乘会让
    # fp16 舍入发生在不同量级上，网络训练中误差会累积。
    # 只有 dQ/dK 乘 scale，dV 不乘（传 1.0，fp32 乘 1.0 精确）。
    # 进 post 前必须 sync_all：要等所有核的 atomicAdd 都落盘。
    pl.system.sync_all()

    with pl.section_vector():
        vBlockIdxP = pl.get_block_idx()
        dqRowStartP = pl.astype(
            pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdxP), pl.DT_INT64
        )
        dqRowEndP = pl.astype(
            pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdxP), pl.DT_INT64
        )
        dkvRowStartP = pl.astype(
            pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdxP), pl.DT_INT64
        )
        dkvRowEndP = pl.astype(
            pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdxP), pl.DT_INT64
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


def flash_attn_grad_bn2(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
):
    """BN2 模板：一个核独占整个 (b, n2) head。

    单块（S1/S2<=128）：没有 pre、没有 post、没有 sync_all，也不读
    fp32 workspace。主循环是 1-ahead ping-pong。

    MultiBlk（is_bn2_multiblk）：默认仍无 pre/post/sync_all。dQ：cube 首
    有效 s2 覆盖 workspace，其余 AtomicAdd；vector 只在 isLastS1Outer
    从 GM Muls+Cast。dK/dV 在 L0C 累加到换 s2 再写回。

    bn2_need_zero==1（整行或整列无效 tile）：pre 按 metadata 行区间把
    dq/dk/dv 三个 out 全部清零，再 sync_all。不分行/列。不清 workspace。
    """
    coreNum = pl.get_block_num()
    # BN2 合轴分核，kernel 内不启用 swizzle
    constInfo = set_const_info(tiling, coreNum, 0)

    viewD2 = constInfo.viewD2Q
    dSize = constInfo.dSize
    d2d = viewD2 * dSize
    s1nd = constInfo.s1Size * d2d
    s2nd = constInfo.s2Size * d2d

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
        [s1nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_k = pl.make_tensor(
        k,
        [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
        [s2nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    # MM3 需要未转置的 K，必须与 MM2 的 K^T 视图分开(layout 各自推导)
    tensor_k_nt = pl.make_tensor(
        k,
        [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
        [s2nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_v = pl.make_tensor(
        v,
        [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
        [s2nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
        [s1nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
        [s1nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_lse = pl.make_tensor(
        softmax_lse,
        [constInfo.bSize, constInfo.n2Size, constInfo.s1Size],
        [constInfo.n2Size * constInfo.s1Size, constInfo.s1Size, 1],
        dtype=pl.DT_FP32,
    )
    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask,
            [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1],
            dtype=pl.DT_UINT8,
        )
    # 输出直接就是 fp16：单块路径没有 fp32 workspace，也就没有 pre/post
    tensor_dq_out = pl.make_tensor(
        dq,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2, dSize],
        [s1nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_dk_out = pl.make_tensor(
        dk,
        [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
        [s2nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )
    tensor_dv_out = pl.make_tensor(
        dv,
        [constInfo.viewD0Q, constInfo.s2Size, viewD2, dSize],
        [s2nd, d2d, dSize, 1],
        dtype=pl.DT_FP16,
    )

    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    # workspace 视图两边都建：单块 iterate_mm_dsk_bn2 被 tilingkey 剪掉 store，
    # 不能把 UB 或 fp16 out 冒充成 fp32 GM 参数。
    alignS1 = constInfo.s1Outer * CUBE_BASEM
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    tensor_dq_ws = pl.make_tensor(
        ws_ptr, [constInfo.coreNum * alignS1, dSize], [dSize, 1], dtype=pl.DT_FP32
    )
    coreNum = constInfo.coreNum
    # ---- 分核区间直接取自 metadata（AICPU DoFagBn2DenseSplit 的结果）----
    # BN2 下 s1Outer=s2Outer=1，一个块就是一个 (b,n2) head，与 metadata
    # 的块编号一一对应。host 按同一公式 SetBlockDim(blockOuter)；若仍有
    # 多启的核，拿到 start==end 则 numBlocks=0，安全空转。
    # cube/vector 两侧读的是同一块 GM，故 numBlocks 天然一致(set/wait 才配得上)。
    meta = pl.make_tensor(metadata, [META_VIEW_LEN], [1], dtype=pl.DT_INT32)
    fagOff = pl.astype(pl.getval(meta, META_HEAD_FAG_START_INDEX), pl.DT_INT64)
    blockStart = pl.astype(
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx), pl.DT_INT64
    )
    blockEnd = pl.astype(
        pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx), pl.DT_INT64
    )
    numBlocks = pl.max(blockEnd - blockStart, 0)

    # BN2 只编 d_align 64/128 两档，这两档在 fag_mem 里共用同一张 L1 图，
    # L1_DS_ADDR / L1_P_ADDR 取值与 GS1S2 相同，故共用同一套 CV 共享区。
    mm1_res = make_mm1_res()
    mm2_res = make_mm2_res()
    ds_l1 = make_ds_l1()
    p_l1 = make_p_l1()
    ds_t_l1 = make_ds_t_l1()
    p_t_l1 = make_p_t_l1()

    # skip_invalid 不写整行/整列无效 tile，out GM 会留脏数据。本编译位
    # 下 pre 清 dq/dk/dv out（三个都清，不分行/列），再 sync_all。
    # 行区间与 GS1S2 共用 metadata FagSplitRows，不在核内按元素 ceil 切。
    # 不清 MultiBlk workspace：cube 首写仍是 AtomicNone。
    if bn2_need_zero == 1:
        with pl.section_vector():
            n1Size = constInfo.n2Size * constInfo.gSize
            dqElems = constInfo.bSize * constInfo.s1Size * n1Size * constInfo.dSize
            dkElems = (
                constInfo.bSize * constInfo.s2Size * constInfo.n2Size * constInfo.dSize
            )
            dvElems = (
                constInfo.bSize * constInfo.s2Size * constInfo.n2Size * constInfo.dvSize
            )
            tensor_dq_flat = pl.make_tensor(
                dq, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP16
            )
            tensor_dk_flat = pl.make_tensor(
                dk, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP16
            )
            tensor_dv_flat = pl.make_tensor(
                dv, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP16
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
        pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1()
        k_l1 = make_k_l1()
        k_l1_n = make_k_nd_l1()
        v_l1 = make_v_l1()
        do_l1 = make_do_l1()
        left = make_l0a()
        right = make_l0b()
        acc = make_l0c_acc()
        dk_acc = make_dk_l0c_chunk0()
        dv_acc = make_dv_l0c_chunk0()

        runInfos = make_run_infos()
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        # 1-ahead：gIdx<0 后再排空 1 轮
        while True:
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            q_cur = q_l1.next()
            do_cur = do_l1.next()
            if isLastLoop == 0:
                gIdx = next_work_gidx(
                    constInfo,
                    cBlockIdx,
                    coreNum,
                    blockStart,
                    blockEnd,
                    localIdx,
                    denseCursor,
                    numBlocks,
                )
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % 2]
                    set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, 2)

                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX, event_id=SYNC_DETER_FIX_FLAG
                        )
                    iterate_mm_qk_bn2(
                        constInfo,
                        curInfo,
                        tensor_q,
                        tensor_k,
                        q_cur,
                        k_l1,
                        left,
                        right,
                        acc,
                        mm2_cur,
                        curInfo.loadKV,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2]
                    )

                    iterate_mm_dyv_bn2(
                        constInfo,
                        curInfo,
                        tensor_dy,
                        tensor_v,
                        do_cur,
                        v_l1,
                        left,
                        right,
                        acc,
                        mm1_cur,
                        curInfo.loadKV,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2]
                    )

            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                mm1_prev = mm1_res.previous()
                mm2_prev = mm2_res.previous()
                q_prev = q_l1.previous()
                do_prev = do_l1.previous()
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
                )

                iterate_mm_dsk_bn2(
                    constInfo,
                    prevInfo,
                    tensor_k_nt,
                    ds_l1,
                    k_l1_n,
                    left,
                    right,
                    acc,
                    mm1_prev,
                    tensor_dq_ws,
                    cBlockIdx,
                    prevInfo.loadKV,
                )
                if is_bn2_multiblk == 1:
                    if prevInfo.dqLast == 1:
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=SYNC_C3_TO_V5_FLAG
                        )
                    iterate_mm_dsq_bn2(
                        constInfo,
                        prevInfo,
                        ds_t_l1.current(),
                        q_prev,
                        left,
                        right,
                        dk_acc,
                        prevInfo.dkvAcc,
                    )
                    if prevInfo.flushDkv == 1:
                        pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                        pl.move(
                            mm2_prev,
                            dk_acc.current(),
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                        )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
                        )
                    iterate_mm_pdy_bn2(
                        constInfo,
                        prevInfo,
                        p_t_l1.current(),
                        do_prev,
                        left,
                        right,
                        dv_acc,
                        prevInfo.dkvAcc,
                    )
                    if prevInfo.flushDkv == 1:
                        pl.store(
                            tensor_dv_out,
                            dv_acc.current(),
                            [prevInfo.idx0KV, prevInfo.keyOffset, prevInfo.idx2KV, 0],
                            order=[1, 3],
                            atomic=pl.AtomicType.AtomicNone,
                        )
                else:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C3_TO_V5_FLAG
                    )
                    iterate_mm_dsq_bn2(
                        constInfo,
                        prevInfo,
                        ds_t_l1.current(),
                        q_prev,
                        left,
                        right,
                        dk_acc,
                        0,
                    )
                    pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                    pl.move(
                        mm2_prev,
                        dk_acc.current(),
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
                    )
                    iterate_mm_pdy_bn2(
                        constInfo,
                        prevInfo,
                        p_t_l1.current(),
                        do_prev,
                        left,
                        right,
                        dv_acc,
                        0,
                    )
                    pl.store(
                        tensor_dv_out,
                        dv_acc.current(),
                        [prevInfo.idx0KV, prevInfo.keyOffset, prevInfo.idx2KV, 0],
                        order=[1, 3],
                        atomic=pl.AtomicType.AtomicNone,
                    )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG
                )

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 1:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = make_y_ub_bn2()
        dy_ub = make_dy_ub_bn2()
        nzp_ub = make_nzp_ub_bn2()
        sfmg_ub = make_sfmg_ub_bn2()
        lse_ub = make_lse_ub_bn2()
        mask_ub = make_mask_view_bn2()
        mask_pre_ub = make_mask_pre_view_bn2()
        cast_ub = make_cast_ub_bn2()

        runInfos = make_run_infos()
        taskId = 0
        localIdx = 0
        denseCursor = blockStart
        isLastLoop = 0
        drainDone = 0
        while True:
            mm2_res.next()
            mm1_res.next()
            lse_ub.next()
            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                process_vec1_bn2(
                    constInfo,
                    prevInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    y_ub.current(),
                    dy_ub.current(),
                    sfmg_ub.current(),
                    d_align,
                )

            if isLastLoop == 0:
                gIdx = next_work_gidx(
                    constInfo,
                    cBlockIdx,
                    coreNum,
                    blockStart,
                    blockEnd,
                    localIdx,
                    denseCursor,
                    numBlocks,
                )
                if gIdx < 0:
                    isLastLoop = 1
                else:
                    denseCursor = gIdx + 1
                    curInfo = runInfos[taskId % 2]
                    set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
                    if mask_mode != 0:
                        apply_s2_idx_no_change(runInfos, curInfo, taskId, 2)

            if taskId > 0:
                prevInfo = runInfos[(taskId + 1) % 2]
                mm1_t = mm1_res.previous()
                mm2_t = mm2_res.previous()
                lse_t = lse_ub.previous()
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C2_TO_V2_FLAG[(taskId + 1) % 2]
                )
                if mask_mode != 0:
                    process_vec2(
                        constInfo,
                        prevInfo,
                        subIdx,
                        tensor_lse,
                        mm2_t,
                        lse_t,
                        tensor_mask,
                        mask_ub.current(),
                        mask_pre_ub.current(),
                    )
                else:
                    process_vec2(
                        constInfo,
                        prevInfo,
                        subIdx,
                        tensor_lse,
                        mm2_t,
                        lse_t,
                        tensor_lse,
                        mask_ub.current(),
                        mask_pre_ub.current(),
                    )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C1_TO_V2_FLAG[(taskId + 1) % 2]
                )
                needSyncDkMM = taskId > 1
                if needSyncDkMM:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                process_vec3_bn2(
                    constInfo,
                    prevInfo,
                    subIdx,
                    ds_l1.current(),
                    mm1_t,
                    mm2_t,
                    sfmg_ub.current(),
                    nzp_ub.current(),
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG
                )
                if needSyncDkMM:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                process_vec4(
                    constInfo, prevInfo, subIdx, p_l1.current(), mm2_t, nzp_ub.current()
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V4_TO_C5_FLAG
                )
                if is_bn2_multiblk == 1:
                    if prevInfo.dqLast == 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE2, event_id=SYNC_C3_TO_V5_FLAG
                        )
                        process_dq_multiblk(
                            tensor_dq_out,
                            tensor_dq_ws,
                            mm1_t,
                            cast_ub.current(),
                            constInfo,
                            prevInfo,
                            subIdx,
                            cBlockIdx,
                        )
                    if prevInfo.flushDkv == 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                        )
                        process_muls_cast_dk_bn2(
                            tensor_dk_out,
                            mm2_t,
                            cast_ub.current(),
                            constInfo,
                            prevInfo,
                            subIdx,
                        )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_DETER_FIX_FLAG
                    )
                else:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_C3_TO_V5_FLAG
                    )
                    process_muls_cast_dq_bn2(
                        tensor_dq_out,
                        mm1_t,
                        cast_ub.current(),
                        constInfo,
                        prevInfo,
                        subIdx,
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                    )
                    process_muls_cast_dk_bn2(
                        tensor_dk_out,
                        mm2_t,
                        cast_ub.current(),
                        constInfo,
                        prevInfo,
                        subIdx,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_DETER_FIX_FLAG
                    )

            if isLastLoop == 1:
                drainDone = drainDone + 1
                if drainDone >= 1:
                    break
            taskId = taskId + 1
            localIdx = localIdx + 1
