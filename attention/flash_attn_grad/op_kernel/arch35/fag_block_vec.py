# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vector 侧编排。

阶段编号与 CV 同步 flag 一致::

    V1  sfmg = rowsum(dy * y)
    V2  P    = softmax(S*scale - lse) [+mask]
    V3  dS   = (dP - sfmg) * P -> cast -> L1
    V4  P    -> cast -> L1
    V5/V6    dQ/dK 的 muls(scale) + cast

本文件只做**搬运与编排**：GM->UB 的 load、UB->L1 的 insert、选哪一档 VF；
真正的寄存器级计算在 :mod:`fag_vector_api` 与 :mod:`attenmask`。

pre / post 两个阶段也放在这里：它们同样是纯 vector、同样按 metadata 给的
行区间切，只是被 ``sync_all`` 与主循环完全隔开。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    D_CHUNK_MAXW,
    SINK_REDUCE_BATCH,
    D_NCHUNKS,
    NZP_ROWS,
    SP_ROWS,
    VECTOR_BASEM,
    VECTOR_BASEN,
    VF_LANES,
    VF_LANES_F16,
    chunk_off,
    chunk_valid_d,
)
from fag_mem import POST_ELEMS, PRE_ELEMS
from fag_vector_api import (
    broadcast_sub_mul_cast_vf,
    cast_p_vf,
    dsink_partial_vf,
    dsink_reduce_add_vf,
    fill_zero_f16_vf,
    fill_zero_vf,
    muls_cast_vf,
    softmax_grad_front_row192_vf,
    softmax_grad_front_row256_acc_vf,
    softmax_grad_front_row256_vf,
    softmax_grad_front_row_vf,
)
from fag_schedule import (
    META_FAG_DKV_ROW_ENDS,
    META_FAG_DKV_ROW_STARTS,
    META_FAG_DQ_ROW_ENDS,
    META_FAG_DQ_ROW_STARTS,
)
from fag_tiling import Tmpl
from attenmask import copy_in_atten_mask, muls_sel_simple_softmax_vf
# --- end generated imports ---

# 窗口循环内每窗交错穿插的清零 chunk 上限。大 N 场景（zTotal >> nIter，如
# N32×S4096 时 kBatchRaw≈31）若不封顶，每窗穿插 31 个清零 chunk 严重打断
# MTE3 流水（板测 MTE3 有效带宽 2.8→1.8 TB/s，降 36%）。封顶 2：每窗最多
# 穿插 2 个 chunk 保住 sfmg 载入与清零写的基本重叠，余量全部落排空段——
# sfmg 窗口完成后 MTE3 独占流水，清零 chunk 背靠背发（不穿插 sfmg load/VF），
# 接近 fasg 的顺序大 chunk 清零效率。
_KBATCH_CAP = 2

# ==================================================================
#  UB -> CV 共享 L1
# ==================================================================


def copy_ub2l1(l1_tile, nzp_ub, row_offset):
    """把 NZ+1 缓冲搬进 CV 共享 L1，跳过 bank 错开用的填充块。

    valid_shape 固定设为 [VECTOR_BASEM, N]=[64,128]，框架据此算出源跨距
    65-64=1，正好跳过每个分形列末尾那一个填充 block。

    为什么不按尾块收窄到 halfS1RealSize：两个 vector 子核分别 insert 到
    rowOff=0 与 64，各占 L1 的 64 行、互不重叠，故超出 s1Real 的脏行只落在
    本子核自己那半区内，随后由 matmul 的 valid_shape 与 store 裁掉。

    搬完把 valid_shape 恢复成 [65, N]，让下一轮 VF 仍按 65 行跨距写入。
    """
    pl.set_validshape(nzp_ub, [VECTOR_BASEM, VECTOR_BASEN])
    pl.insert(l1_tile, nzp_ub, [row_offset, 0])
    pl.set_validshape(nzp_ub, [NZP_ROWS, VECTOR_BASEN])


# ==================================================================
#  V1 —— sfmg = rowsum(dy * y)
# ==================================================================


def _vector_load_rows(constInfo, runInfo):
    """Bound y/dy/lse GM reads by the physical Q view, not the used length.

    s1ViewSize is packed T for TND and physical S for BSND/BNSD. Keep the
    tile's physical shape/stride unchanged; rowShift selects the valid rows
    inside this load window, including windows borrowed across TND batches.
    """
    return pl.min(VECTOR_BASEM, pl.max(constInfo.s1ViewSize - runInfo.loadBase, 0))


def _sfmg_tile_compute(
    constInfo,
    tensor_y,
    tensor_dy,
    idx0,
    loadBase,
    idx2,
    loadRows,
    halfRows,
    rowShift,
    y_ub,
    dy_ub,
    sfmg_ub,
    rowStride,
):
    """一个 64 行窗口的 sfmg = rowsum(dy * y)：载 y/dy 后按 dv_align 分档跑 VF。

    process_vec1（主循环内联，非前移档）与 process_presfmg（pre 阶段前移）
    共用同一套 load 与 VF 分档，保证两条路径数值行为一致：
    loadRows 是物理窗口行数（TND 借行含在内），halfRows 是本子核有效行数，
    rowShift 把 VF 的读取跳到窗口内的有效行。
    """
    if dv_align == 768:
        # 768 档：y/dy 按 256 列窗口三趟载入，后两趟累加到同一 sfmg。
        for k in pl.range(0, 3):
            colOff = k * 256
            rem = pl.max(constInfo.dvSize - colOff, 0)
            winW = pl.min(256, rem)
            if rem > 0:
                pl.set_validshape(y_ub, [loadRows, winW])
                pl.set_validshape(dy_ub, [loadRows, winW])
                pl.load(
                    y_ub,
                    tensor_y,
                    [idx0, loadBase, idx2, colOff],
                    order=[1, 3],
                )
                pl.load(
                    dy_ub,
                    tensor_dy,
                    [idx0, loadBase, idx2, colOff],
                    order=[1, 3],
                )
                w0 = pl.min(rem, 128)
                w1 = pl.max(rem - 128, 0)
                even0 = (w0 + 1) // 2
                odd0 = w0 // 2
                even1 = (w1 + 1) // 2
                odd1 = w1 // 2
                if k == 0:
                    softmax_grad_front_row256_vf(
                        sfmg_ub,
                        y_ub,
                        dy_ub,
                        halfRows,
                        rowShift,
                        rowStride,
                        even0,
                        odd0,
                        even1,
                        odd1,
                    )
                else:
                    softmax_grad_front_row256_acc_vf(
                        sfmg_ub,
                        y_ub,
                        dy_ub,
                        halfRows,
                        rowShift,
                        rowStride,
                        even0,
                        odd0,
                        even1,
                        odd1,
                    )
        return
    pl.set_validshape(y_ub, [loadRows, constInfo.dvSize])
    pl.set_validshape(dy_ub, [loadRows, constInfo.dvSize])
    pl.load(y_ub, tensor_y, [idx0, loadBase, idx2, 0], order=[1, 3])
    pl.load(dy_ub, tensor_dy, [idx0, loadBase, idx2, 0], order=[1, 3])
    # 列宽跟 dv_align：256/192 各一档 VF；64 与 128（含 D=192,Dv=128）共用
    # 一次 128-lane load，用 even/odd mask 收真实 Dv。
    if dv_align == 256:
        w0 = pl.min(constInfo.dvSize, 128)
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row256_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            halfRows,
            rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    elif dv_align == 192:
        w0 = pl.min(constInfo.dvSize, 128)
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row192_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            halfRows,
            rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    else:
        softmax_grad_front_row_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            halfRows,
            rowShift,
            rowStride,
            constInfo.dvEvenSize,
            constInfo.dvOddSize,
        )


def process_vec1(
    constInfo,
    runInfo,
    subIdx,
    tensor_y,
    tensor_dy,
    tensor_sfmg_ws,
    y_ub,
    dy_ub,
    sfmg_ub,
    rowStride,
):
    """softmaxGradFront: sfmg = rowsum(dy * y)。一趟 VECTOR_BASEM 行。

    dv_align>=128 走前移（对齐老 FAG 的 preSfmg）：sfmg 一行一值、与 s2
    无关，主循环每个 tile 重算会被重复 s2Outer 遍，大 D 档是 V 侧的主要
    浪费。pre 阶段(process_presfmg)已按 (head, 全局 token 行) 算好全量存
    ws，这里只搬本子核有效行：起点 loadBaseTnd+rowShift 即 rowBase 的
    全局行；尾部不足 64 行的槽位与内联路径一样由 V3 的写满语义吞掉。
    dv_align=64 保持内联（一趟半宽 VF，前移的 GM 往返不划算）。
    ★门必须与写入方逐一对齐（读未初始化的 ws 是灾难）：写侧
    process_presfmg 与读侧共用 constInfo.enablePreSfmg（host 按 fasg
    enablePreSfmg 判据统一裁决，见 flash_attn_grad_tiling_plan.cpp 的
    DecidePreSfmg）。两端同标志是结构保证，不再各自用 template/dv_align
    拼条件。
    """
    if constInfo.enablePreSfmg == 1:
        headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
        rows = runInfo.halfS1RealSize
        if rows > 0:
            # 只搬有效行（rows），并 clamp 确保不超出 ws 的 (head,batch) 段：
            # rowShift 使读取起点 = rowBase（全局行），rows 使终点 = rowBase+rows
            # 不会超出本 batch 的物理范围。剩余 sfmg_ub[rows..63] 为残留，
            # 与内联路径的 VF 写入行为一致（V3 后续按 validshape 裁剪）。
            # validshape/order 与 lse 同一套：[64,1] DN tile 取 [rows,1]，
            # order=[1,0] 把 tile 行轴映到 ws 的扁平 token 轴。
            pl.set_validshape(sfmg_ub, [rows, 1])
            pl.load(
                sfmg_ub,
                tensor_sfmg_ws,
                [0, headQ * constInfo.t1Size + runInfo.loadBaseTnd + runInfo.rowShift],
                order=[1, 0],
            )
            # load 后恢复满形状 [64,1]：V3 的 VF 按线性地址读 64 行。
            pl.set_validshape(sfmg_ub, [VECTOR_BASEM, 1])
    else:
        # 未前移：主循环内联重算，数值与 pre 路径一致（同一套 VF）。
        loadRows = _vector_load_rows(constInfo, runInfo)
        _sfmg_tile_compute(
            constInfo,
            tensor_y,
            tensor_dy,
            runInfo.idx0Q,
            runInfo.loadBase,
            runInfo.idx2Q,
            loadRows,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            y_ub,
            dy_ub,
            sfmg_ub,
            rowStride,
        )


def _zero_step(zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc):
    """issue 下一个统一清零任务（三段逻辑连续编号，运行期条件分发）。

    zc 为全局元素游标（dq/dk/dv 三段拼接的连续下标）：每任务从 zc 起
    清 min(chunk, 段尾−zc) 个，返回值推进实际清零量 n——段尾用运行期
    min 裁 validshape，保证 zc 恰好落在段边界上，下一任务自动进入下一
    段且段内偏移归零，绝不越段（越段会误清 dsink/sfmg 段，清零对它们
    不幂等）。zc 越过 zS3 时 n=0 空转、游标冻结（排空段允许 kBatch 过量）。
    供 process_presfmg 的交错清零：清零任务与窗口 load/VF 交替 issue，
    MTE3 不被整段清零灌满阻塞 sfmg 写。
    """
    chunk = PRE_ELEMS[d_align]
    n = 0
    if zc < zS1:
        n = pl.min(chunk, zS1 - zc)
        pl.set_validshape(zt, [1, n])
        pl.store(zdq, zt, [0, dqStart + zc])
    else:
        if zc < zS2:
            n = pl.min(chunk, zS2 - zc)
            pl.set_validshape(zt, [1, n])
            pl.store(zdk, zt, [0, dkStart + (zc - zS1)])
        else:
            if zc < zS3:
                n = pl.min(chunk, zS3 - zc)
                pl.set_validshape(zt, [1, n])
                pl.store(zdv, zt, [0, dvStart + (zc - zS2)])
    return zc + n


def process_presfmg(
    constInfo,
    tensor_y,
    tensor_dy,
    tensor_sfmg_ws,
    y_ub,
    dy_ub,
    sfmg_ub,
    sfmg_flat,
    rowStride,
    zt,
    zdq,
    zdk,
    zdv,
    dqStart,
    zS1,
    dkStart,
    zS2,
    dvStart,
    zS3,
):
    """softmaxGrad 前移：pre 阶段一次算完全部行的 sfmg 写入 workspace。

    行空间 [N1, T1]（token 主序扁平，与 TND lse 同构）按窗口行块切给全部
    AIV 均分（窗口行数按档取 SP_ROWS）；每块复用 _sfmg_tile_compute（与
    主循环 V1 同一套 VF 分档），结果 store 到 ws 的 head*T1+全局行 处。

    sp_out 与 sp_out_flat 双槽（depth 2）：循环内同步 .next() 乒乓轮转，
    VF(N) 写与 store(N-1) 读并行；sp_y/sp_dy 也各双槽（.next() 乒乓）：
    load(N+1) 写下一槽、VF(N) 读本槽，物理隔离无需跨队列 wait。
    **不能退回 sp_y/sp_dy 单槽（.current() 固定单 tile）**：pypto 对
    同一 tile 的跨队列操作不插 wait，load(N+1) 会覆盖 VF(N) 正在读的
    窗口（上板 sfmg 大面积错值 + 非逐位 + NaN）。

    块内地址分解：
      TND —— 全局 token 行即 packed 物理行，窗口任意切都合法；
      BSND/BNSD —— 按 (head, b, s 块) 切，窗口不跨 batch（s1Size 裁剪），
        视图系数 coefB0Q/coefN0/coefN2 与 set_run_info 同一套。
    ws 行号与 process_vec1 前移分支的读取口径 (loadBaseTnd) 一致——ws 是
    扁平缓存，写侧窗口行数与读侧 64 行窗口无需一致。
    落盘走 sfmg_flat（[1,SP_ROWS] 扁平视图，同址同号）：store 的 order 必须
    升序，[rows,1] DN tile 无法用 order=[1,0] 映到扁平轴，改走 zt/sink
    的恒等映射口径（validshape [1,rows] + 无 order）。
    """
    t1Size = constInfo.t1Size
    spRows = SP_ROWS[dv_align]
    aivIdx = pl.get_block_idx()
    zc = 0
    chunk = PRE_ELEMS[d_align]
    # 任务数三段分计：段尾裁宽后每段末任务步长 < chunk，全局 ceil(zS3/chunk)
    # 会少计 0~2 个任务（每段各可能少 1 个），段尾就清不干净。
    zTotal = (
        (zS1 + chunk - 1) // chunk
        + ((zS2 - zS1) + chunk - 1) // chunk
        + ((zS3 - zS2) + chunk - 1) // chunk
    )
    if layout == 1:
        tBlocks = (t1Size + spRows - 1) // spRows
        totalBlocks = constInfo.n1Size * tBlocks
        perCore = (totalBlocks + constInfo.aivCount - 1) // constInfo.aivCount
        biStart = pl.min(aivIdx * perCore, totalBlocks)
        nIter = pl.max(pl.min(biStart + perCore, totalBlocks) - biStart, 0)
        nIterSafe = pl.max(nIter, 1)
        # kBatch 封顶 _KBATCH_CAP：保住 sfmg 载入与清零写的基本重叠，
        # 余量全部落排空段集中背靠背发（设计依据见 _KBATCH_CAP）。
        kBatch = pl.min((zTotal + nIterSafe - 1) // nIterSafe, _KBATCH_CAP)
        for i in pl.range(0, nIter):
            for _ in pl.range(0, kBatch):
                zc = _zero_step(
                    zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc
                )
            bi = biStart + i
            head = bi // tBlocks
            t0 = (bi - head * tBlocks) * spRows
            rows = pl.min(spRows, t1Size - t0)
            # enablePreSfmg=0 时只保留清零（主循环 V1 走内联重算），
            # sp 载入/VF/store 全部跳过；标志按 launch 恒定，乒乓轮转一致。
            if constInfo.enablePreSfmg == 1:
                y_t = y_ub.next()
                dy_t = dy_ub.next()
                out_t = sfmg_ub.next()
                flat_t = sfmg_flat.next()
                _sfmg_tile_compute(
                    constInfo,
                    tensor_y,
                    tensor_dy,
                    0,
                    t0,
                    head,
                    rows,
                    rows,
                    0,
                    y_t,
                    dy_t,
                    out_t,
                    rowStride,
                )
                # 扁平视图落盘：[1,rows] 恒等映射，只落有效 rows 个元素。
                pl.set_validshape(flat_t, [1, rows])
                pl.store(tensor_sfmg_ws, flat_t, [0, head * t1Size + t0])
    else:
        sBlocks = (constInfo.s1Size + spRows - 1) // spRows
        bsBlocks = constInfo.bSize * sBlocks
        totalBlocks = constInfo.n1Size * bsBlocks
        perCore = (totalBlocks + constInfo.aivCount - 1) // constInfo.aivCount
        biStart = pl.min(aivIdx * perCore, totalBlocks)
        nIter = pl.max(pl.min(biStart + perCore, totalBlocks) - biStart, 0)
        nIterSafe = pl.max(nIter, 1)
        # 同 TND 分支：kBatch 封顶 _KBATCH_CAP，余量落排空段集中发。
        kBatch = pl.min((zTotal + nIterSafe - 1) // nIterSafe, _KBATCH_CAP)
        for i in pl.range(0, nIter):
            for _ in pl.range(0, kBatch):
                zc = _zero_step(
                    zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc
                )
            bi = biStart + i
            head = bi // bsBlocks
            rem = bi - head * bsBlocks
            b = rem // sBlocks
            s = (rem - b * sBlocks) * spRows
            rows = pl.min(spRows, constInfo.s1Size - s)
            # 同 TND 分支：enablePreSfmg=0 时只清零，不搬 y/dO、不算 sfmg。
            if constInfo.enablePreSfmg == 1:
                y_t = y_ub.next()
                dy_t = dy_ub.next()
                out_t = sfmg_ub.next()
                flat_t = sfmg_flat.next()
                _sfmg_tile_compute(
                    constInfo,
                    tensor_y,
                    tensor_dy,
                    b * constInfo.coefB0Q + head * constInfo.coefN0,
                    s,
                    head * constInfo.coefN2,
                    rows,
                    rows,
                    0,
                    y_t,
                    dy_t,
                    out_t,
                    rowStride,
                )
                # 同上：扁平视图 [1,rows] 恒等映射，落有效 rows 个元素。
                pl.set_validshape(flat_t, [1, rows])
                pl.store(
                    tensor_sfmg_ws,
                    flat_t,
                    [0, head * t1Size + b * constInfo.s1Size + s],
                )

    # 排空段：kBatch*nIter 可能 < zTotal（ceil 截断）或 nIter=0（无窗任务
    # 的核全部任务在此）；亦可能 > zTotal（_zero_step 的越界保护空转）。
    # zS3=0（BN2 调用点）zTotal=0 → zTail=0，编译期零循环。
    zTail = pl.max(zTotal - kBatch * nIter, 0)
    for _ in pl.range(0, zTail):
        zc = _zero_step(zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc)


# ==================================================================
#  V2 —— P = simpleSoftmax(S * scale, lse) [+ atten mask]
# ==================================================================


def process_vec2(
    constInfo,
    runInfo,
    subIdx,
    tensor_lse,
    mm2_ub,
    lse_ub,
    tensor_mask,
    mask_ub,
    mask_pre_ub,
):
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    loadRows = _vector_load_rows(constInfo, runInfo)
    pl.set_validshape(lse_ub, [loadRows, 1])
    if layout == 1:
        # TND：lse 是 (N1, T1) token 主序。变长 TND 的 loadBase 已是全局
        # token 行（tokenBaseQ + 块内偏移，由 _set_run_info_tnd_varlen 计算）；
        # 等长 TND 的 loadBase 也是全局坐标（tokenBase = boIdx*s1 已折叠）。
        # 0930 bisect-sink 定案移植。
        rowT = runInfo.loadBase
        pl.load(lse_ub, tensor_lse, [headQ, rowT], order=[1, 0])
    else:
        pl.load(
            lse_ub, tensor_lse, [runInfo.boIdx, headQ, runInfo.loadBase], order=[2, 0]
        )
    if mask_mode != 0:
        copy_in_atten_mask(
            constInfo, runInfo, subIdx, tensor_mask, mask_ub, mask_pre_ub
        )
    muls_sel_simple_softmax_vf(
        mm2_ub,
        lse_ub,
        mask_ub,
        constInfo.scaleValue,
        runInfo.halfS1RealSize,
        runInfo.rowShift,
    )


# ==================================================================
#  V3 —— dS = (dP - sfmg) * P，cast + ND2NZ 落 L1（mm4 读同址 ZN 视图）
# ==================================================================


def process_vec3(constInfo, runInfo, subIdx, ds_l1, mm1_ub, mm2_ub, sfmg_ub, nzp_ub):
    rowOff = subIdx * VECTOR_BASEM
    # nd/nz 不裁剪：列方向的脏数据由 matmul 的 valid_shape 与 store 挡掉。
    # VF 固定写满 VECTOR_BASEM 行：NZ+1 的分形列间距由行数决定，按尾块
    # 缩小会让间距漂移。尾块只在 insert 时用 valid_shape 裁掉。
    broadcast_sub_mul_cast_vf(
        nzp_ub, mm1_ub, mm2_ub, sfmg_ub, constInfo.scaleValue, VECTOR_BASEM
    )
    # 只写一次 L1：mm4 通过 ds_t_l1(同地址的 ZN 视图)直接读到 dS^T，
    # 无需再做 transpose/insert
    copy_ub2l1(ds_l1, nzp_ub, rowOff)


# ==================================================================
#  V4 —— P cast + ND2NZ 落 L1（mm5 读同址 ZN 视图）
# ==================================================================


def process_vec4(constInfo, runInfo, subIdx, p_l1, mm2_ub, nzp_ub):
    rowOff = subIdx * VECTOR_BASEM
    cast_p_vf(nzp_ub, mm2_ub, VECTOR_BASEM)
    # 同 process_vec3：mm5 从 p_t_l1(同地址 ZN 视图)读 P^T
    copy_ub2l1(p_l1, nzp_ub, rowOff)


# ==================================================================
#  Pre —— dq/dk/dv 累加区清零
# ==================================================================


def process_sink_partial(
    constInfo,
    runInfo,
    tensor_slots,
    aiv_idx,
    p_ub,
    dp_ub,
    lse_ub,
    sink_part_ub,
    sink_val,
):
    """本 s2 块的 dsink 贡献，原子加到槽位区本 AIV 行的 headQ 列。"""
    rows = runInfo.halfS1RealSize
    if rows <= 0:
        return
    s2Real = runInfo.s2RealSize
    if s2Real <= 0:
        return
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    loopTimes = (s2Real + VF_LANES - 1) // VF_LANES - 1
    tailCols = s2Real - loopTimes * VF_LANES
    dsink_partial_vf(
        sink_part_ub,
        p_ub,
        dp_ub,
        lse_ub,
        sink_val,
        rows,
        loopTimes,
        tailCols,
        runInfo.rowShift,
    )
    pl.set_validshape(sink_part_ub, [1, 1])
    pl.store(
        tensor_slots,
        sink_part_ub,
        [0, aiv_idx * constInfo.n1Size + headQ],
        atomic=pl.AtomicType.AtomicAdd,
    )
    pl.set_validshape(sink_part_ub, [1, VECTOR_BASEN])


def process_sink_reduce(
    constInfo, tensor_slots, tensor_dsink, acc_ub, row_ub, aiv_count
):
    """AIV0 按行序把各 AIV 的 dsink 部分和加到输出。

    ``row_ub`` 是 SINK_REDUCE_BATCH 个独立的 1 行 tile：每批发 BATCH 次 load
    （偏移与原逐行版完全相同，不依赖任何行距假设），只等 1 次 bar_mte2。
    实测逐行版每次 load 后都排空流水，aivCount=72 时这一项占 sink 固定开销
    13.8us 里的 7.5us。
    """
    row0 = row_ub[0].current()
    row1 = row_ub[1].current()
    row2 = row_ub[2].current()
    row3 = row_ub[3].current()
    n1Size = constInfo.n1Size
    nsegs = VECTOR_BASEN // VF_LANES
    nChunks = (n1Size + VECTOR_BASEN - 1) // VECTOR_BASEN
    nRest = aiv_count - 1
    nBatch = (nRest + SINK_REDUCE_BATCH - 1) // SINK_REDUCE_BATCH
    for c in pl.range(0, nChunks):
        w = pl.min(VECTOR_BASEN, n1Size - c * VECTOR_BASEN)
        pl.set_validshape(acc_ub, [1, w])
        pl.load(acc_ub, tensor_slots, [0, c * VECTOR_BASEN])
        pl.system.bar_mte2()
        for b in pl.range(0, nBatch):
            r0 = 1 + b * SINK_REDUCE_BATCH
            k = pl.min(SINK_REDUCE_BATCH, aiv_count - r0)
            pl.set_validshape(row0, [1, w])
            pl.load(row0, tensor_slots, [0, (r0 + 0) * n1Size + c * VECTOR_BASEN])
            if k > 1:
                pl.set_validshape(row1, [1, w])
                pl.load(row1, tensor_slots, [0, (r0 + 1) * n1Size + c * VECTOR_BASEN])
            if k > 2:
                pl.set_validshape(row2, [1, w])
                pl.load(row2, tensor_slots, [0, (r0 + 2) * n1Size + c * VECTOR_BASEN])
            if k > 3:
                pl.set_validshape(row3, [1, w])
                pl.load(row3, tensor_slots, [0, (r0 + 3) * n1Size + c * VECTOR_BASEN])
            pl.system.bar_mte2()
            dsink_reduce_add_vf(acc_ub, row0, nsegs)
            if k > 1:
                dsink_reduce_add_vf(acc_ub, row1, nsegs)
            if k > 2:
                dsink_reduce_add_vf(acc_ub, row2, nsegs)
            if k > 3:
                dsink_reduce_add_vf(acc_ub, row3, nsegs)
        pl.set_validshape(acc_ub, [1, w])
        pl.store(tensor_dsink, acc_ub, [0, c * VECTOR_BASEN])


def process_sink_zero(tensor_slots, constInfo, zt, aiv_idx):
    """清本 AIV 的 dsink 槽位行，供主循环 atomicAdd。"""
    fill_zero_vf(zt, (constInfo.n1Size + VF_LANES - 1) // VF_LANES)
    process_pre(tensor_slots, zt, aiv_idx * constInfo.n1Size, constInfo.n1Size)


def process_pre(tensor_ws, zt, elemStart, nElems):
    """把一段连续 GM 清零，按一维元素切。

    tensor_ws 是 [1, totalElems] 的扁平视图（B*S*N*D 紧凑）。本核负责
    [elemStart, elemStart+nElems)。UB 按 PRE_ELEMS 切 full tile；尾块用
    trip count 0/1 的第二个 pl.range。
    PyPTO tile 必须是二维，故写成 [1, N]，语义仍是一维元素。
    zt 已由调用方填零，本函数只 store。dtype 跟 zt / tensor_ws（fp32 ws
    或 16-bit out）。
    """
    chunk = PRE_ELEMS[d_align]
    nFull = nElems // chunk
    rem = nElems - nFull * chunk
    off = elemStart
    pl.set_validshape(zt, [1, chunk])
    for _ in pl.range(0, nFull):
        pl.store(tensor_ws, zt, [0, off])
        off = off + chunk
    nTail = (rem + chunk - 1) // chunk
    for _ in pl.range(0, nTail):
        pl.set_validshape(zt, [1, rem])
        pl.store(tensor_ws, zt, [0, off])


def process_pre_from_meta(meta, fagOff, tensor_dq, tensor_dk, tensor_dv, dQ, dV, zt):
    """按 metadata 行区间清 dq/dk/dv。GS1S2 清 fp32 workspace，bn2_need_zero 清 16-bit out。

    AICPU FagSplitRows：dqRows=B*S1*N1、dkvRows=B*S2*N2，按 AIV 均分行。
    元素起点 = rowStart * 行宽。填零 VF 跟 tilingkey：bn2_need_zero==1 走 16-bit。
    """
    vBlockIdx = pl.get_block_idx()
    one64 = pl.const(1, pl.DT_INT64)
    dqRowStart = pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx) * one64
    dqRowEnd = pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx) * one64
    dkvRowStart = pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx) * one64
    dkvRowEnd = pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx) * one64
    dqNElems = (dqRowEnd - dqRowStart) * dQ
    dkvNElems = (dkvRowEnd - dkvRowStart) * dQ
    dvNElems = (dkvRowEnd - dkvRowStart) * dV
    preElems = pl.min(PRE_ELEMS[d_align], pl.max(dqNElems, pl.max(dkvNElems, dvNElems)))
    pl.set_validshape(zt, [1, PRE_ELEMS[d_align]])
    if bn2_need_zero == 1:
        fill_zero_f16_vf(zt, (preElems + VF_LANES_F16 - 1) // VF_LANES_F16)
    else:
        fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
    process_pre(tensor_dq, zt, dqRowStart * dQ, dqNElems)
    process_pre(tensor_dk, zt, dkvRowStart * dQ, dkvNElems)
    process_pre(tensor_dv, zt, dkvRowStart * dV, dvNElems)


# ==================================================================
#  Post —— dQ/dK/dV 的 scale + cast
# ==================================================================


def process_post(tensor_ws, tensor_out, f32_ub, f16_ub, scale, elemStart, nElems):
    """fp32 workspace -> muls(scale) -> cast 16-bit，按一维元素切。

    本核 [elemStart, elemStart+nElems)。full tile 的 VF 段数是编译期常量；
    尾块用 trip count 0/1 的第二个 pl.range。
    """
    chunk = POST_ELEMS[d_align]
    nFull = nElems // chunk
    rem = nElems - nFull * chunk
    nsegsFull = chunk // VECTOR_BASEN
    off = elemStart
    for _ in pl.range(0, nFull):
        ft = f32_ub.next()
        ht = f16_ub.next()
        pl.set_validshape(ft, [1, chunk])
        pl.set_validshape(ht, [1, chunk])
        pl.load(ft, tensor_ws, [0, off])
        muls_cast_vf(ht, ft, scale, nsegsFull)
        pl.store(tensor_out, ht, [0, off])
        off = off + chunk
    nTail = (rem + chunk - 1) // chunk
    nsegsTail = (rem + VECTOR_BASEN - 1) // VECTOR_BASEN
    for _ in pl.range(0, nTail):
        ft = f32_ub.next()
        ht = f16_ub.next()
        pl.set_validshape(ft, [1, rem])
        pl.set_validshape(ht, [1, rem])
        pl.load(ft, tensor_ws, [0, off])
        muls_cast_vf(ht, ft, scale, nsegsTail)
        pl.store(tensor_out, ht, [0, off])


# ==================================================================
#  BN2 模板的 vector 阶段
# ==================================================================


def muls_cast_ub_out(
    tensor_out, src_ub, dst_ub, idx0, row, idx2, rows, scale, dim_size, d_off=0
):
    """UB 里的 fp32 结果 → Muls(scale) + Cast → out GM。

    整块 ``[VECTOR_BASEM, VECTOR_BASEN]`` 一次 cast，再按 ``rows``/``dim_size``
    收窄落盘。BN2 narrow 的 v5/v6 与 192 档 dK 的每个 D 分块（``d_off`` 指定列
    区间）都用这条；行区间由调用点算（``halfS1RealSize`` / ``firstHalfS2RealSize``，
    DualModeSplitM 的切点不能钉 64）。
    """
    if rows <= 0:
        return
    pl.set_validshape(src_ub, [VECTOR_BASEM, VECTOR_BASEN])
    pl.set_validshape(dst_ub, [VECTOR_BASEM, VECTOR_BASEN])
    muls_cast_vf(dst_ub, src_ub, scale, VECTOR_BASEM * VECTOR_BASEN // VF_LANES_F16)
    pl.set_validshape(dst_ub, [rows, dim_size])
    pl.store(
        tensor_out,
        dst_ub,
        [idx0, row, idx2, d_off],
        order=[1, 3],
        atomic=pl.AtomicType.AtomicNone,
    )
    pl.set_validshape(dst_ub, [VECTOR_BASEM, VECTOR_BASEN])


def muls_cast_ws_out(
    tensor_out,
    tensor_ws,
    ws_load_ub,
    ws_cast_ub,
    idx0,
    row,
    idx2,
    rows,
    scale,
    dim_size,
):
    """4D fp32 workspace → 逐 D 分块 load → Muls(scale) + Cast → out GM。

    BN2 wide（d_align>128）：dQ 三档、dK 256/768、dV 768。cube 侧用
    ``dqFirst``/``dkvAcc`` 保证首写 AtomicNone，故这里不需要 pre 清零，也没有
    全局 post。整块 cast（chunk=96/128 列）让 16-bit 行宽达到 192B/256B。
    """
    if rows <= 0:
        return
    tile_w = D_CHUNK_MAXW[d_align]
    nSegs = tile_w * VECTOR_BASEM // VF_LANES_F16
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = chunk_off(i)
        w = chunk_valid_d(dim_size, i)
        if w > 0:
            pl.set_validshape(ws_load_ub, [rows, w])
            pl.set_validshape(ws_cast_ub, [rows, w])
            pl.load(ws_load_ub, tensor_ws, [idx0, row, idx2, dOff], order=[1, 3])
            muls_cast_vf(ws_cast_ub, ws_load_ub, scale, nSegs)
            pl.store(
                tensor_out,
                ws_cast_ub,
                [idx0, row, idx2, dOff],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicNone,
            )
    pl.set_validshape(ws_load_ub, [VECTOR_BASEM, tile_w])
    pl.set_validshape(ws_cast_ub, [VECTOR_BASEM, tile_w])


def process_dq_multiblk(
    tensor_dq, tensor_dq_ws, src_ub, dst_ub, constInfo, runInfo, subIdx, cBlockIdx
):
    """只在本 s1 列的最后一个有效 s2 块（``dqLast``）从 per-core 2D 条带 load。

    累加已由 cube 按 needAtomic 写在 workspace 上。src_ub 是 load 缓冲。
    """
    rows = runInfo.halfS1RealSize
    rowOff = (
        cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx
    ) * CUBE_BASEM + subIdx * VECTOR_BASEM
    pl.set_validshape(src_ub, [VECTOR_BASEM, VECTOR_BASEN])
    if rows > 0:
        pl.set_validshape(src_ub, [rows, constInfo.dSize])
        pl.load(src_ub, tensor_dq_ws, [rowOff, 0])
        pl.set_validshape(src_ub, [VECTOR_BASEM, VECTOR_BASEN])
    muls_cast_ub_out(
        tensor_dq,
        src_ub,
        dst_ub,
        runInfo.idx0Q,
        runInfo.queryOffset + subIdx * VECTOR_BASEM,
        runInfo.idx2Q,
        rows,
        constInfo.scaleValue,
        constInfo.dSize,
    )
