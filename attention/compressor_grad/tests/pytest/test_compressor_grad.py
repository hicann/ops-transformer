#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import random

import os
import sys
import torch
import pytest


def _small_ops_golden(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    ape,
    block_table,
    sps,
    cr,
    coff,
    cache_mode,
    cu_seqlens,
    seqused,
    dc,
    device,
    matmul_mode,
    compute_dtype,
):
    """通路 3：小算子拼接 + 自动反向推导四梯度（loss 只对有效压缩块，mask 过滤）。

    matmul_mode/compute_dtype 决定精度口径（与 compressor_forward 对齐）：
      two:  CPU f32 提升 + 中间量量化（两方 golden）
      same: 输入不提升，bf16 进 bf16 出（三方 B，模拟 kernel 同精度）
      high: float64 全程（三方 C，高精度真值）
    """
    dev = x.device if device == "cpu" else device
    x_in = x.detach().requires_grad_(True)
    wkv_in = wkv.detach().requires_grad_(True)
    wgate_in = wgate.detach().requires_grad_(True)
    ape_in = ape.detach().requires_grad_(True)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(dev)
    ref_cmp, ref_mask, _, _, _, _ = compressor_forward.compressor_forward(
        x_in,
        wkv_in,
        wgate_in,
        kv_state,
        score_state,
        ape_in,
        block_table,
        sps,
        cmp_ratio=cr,
        coff=coff,
        cache_mode=cache_mode,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        return_intermediates=True,
        device=device,
        matmul_mode=matmul_mode,
        compute_dtype=compute_dtype,
    )
    dc = dc.to(dev)
    if ref_cmp.requires_grad:
        # ⚠️ loss 只对有效压缩块：mask 过滤，无效 padding 不附加梯度
        loss = (ref_cmp * dc * ref_mask.float()).sum()
        loss.backward()
        g_dx = x_in.grad.reshape(-1, x_in.shape[-1]) if x_in.dim() == 3 else x_in.grad
        g_dwkv = wkv_in.grad
        g_dwgate = wgate_in.grad
        g_dape = ape_in.grad
    else:
        # 无有效压缩块（totalValid=0）：cmp_kv 为纯 zeros 无梯度图，golden 梯度全 0
        g_dx = (
            torch.zeros_like(x_in).reshape(-1, x_in.shape[-1])
            if x_in.dim() == 3
            else torch.zeros_like(x_in)
        )
        g_dwkv = torch.zeros_like(wkv_in)
        g_dwgate = torch.zeros_like(wgate_in)
        g_dape = torch.zeros_like(ape_in)
    return g_dx, g_dwkv, g_dwgate, g_dape


def run_small_ops_case(case, device, compare_mode=3):
    """通路 3：小算子拼接通路——小算子拼接 + 自动反向推导梯度作 golden，
    算子侧经 pta 反向传播串联正反向（与通路 1 相同 NPU 流程）。

    ⚠️ loss 只对有效压缩块计算（cmp_mask 过滤），无效 padding 不附加梯度。
    golden 全部来自小算子自动反向（不调手写 compressor_grad_golden）：
    compare_mode=2 时用 CPU 小算子 two 口径；compare_mode=3 时
    B=NPU 小算子 same（bf16 进 bf16 出）、C=CPU 小算子 high（float64）。
    """
    p = _parse_case(case, device)
    B, S1, H, D = p["B"], p["S1"], p["H"], p["D"]
    cr, coff = p["cr"], p["coff"]
    layoutId, ioDtype, dataType = p["layoutId"], p["ioDtype"], p["dataType"]
    layout = p["layout"]
    seqSize = p["seqSize"]
    tokenSize = p["tokenSize"]
    sps = case["start_pos"] or [0] * B
    sqs = case["seqused_q"]
    cu = case.get("seqlens_list_q")
    seed = case["manual_seed"]

    if layoutId == 1:
        if cu is not None:
            cuNpu = torch.tensor(cu, dtype=torch.int32)
        elif sqs:
            cuNpu = torch.tensor(
                [0] + torch.tensor(sqs).cumsum(0).tolist(), dtype=torch.int32
            )
        else:
            cuNpu = torch.zeros(B + 1, dtype=torch.int32)
    else:
        cuNpu = None
    totalValid = 0
    for i in range(B):
        sp = sps[i]
        sq = (
            sqs[i]
            if sqs
            else (
                int(cuNpu[i + 1] - cuNpu[i])
                if cuNpu is not None
                else (tokenSize if layoutId == 1 else tokenSize // B)
            )
        )
        cmpLimit = (sp + sq) // cr * cr
        if cmpLimit > sp:
            totalValid += _ceil_div(cmpLimit - sp, cr)

    # ── NPU 侧：pta 反向传播串联正反向（与通路 1 相同的输入构造 + compute）──
    b = _build_inputs(case, p, device, seed, totalValid, cuNpu, sqs, sps)
    # 初始 state 必须在 NPU 前向之前保留副本（_compressor_forward 原地更新 state_cache）：
    # CPU golden 重放与 compare_mode=3 直调 forward 均以初始 state 为输入，与 autograd
    # 前向一致——对齐通路 1/5 run_forward_direct_case 的 state_cpu_initial 处理
    # （coff=2 读 cache 历史场景，防更新后 state 污染读值）
    state_initial_npu = b["state_cache_npu"].clone()
    backend = NPUBackend(device_id=DEVICE_ID)
    npu_result = backend.compute(inputs=b["inputs"])
    torch.npu.synchronize()

    # ── Golden：小算子拼接 + 自动反向推导四梯度（不调手写 compressor_grad_golden）──
    coffD = coff * D
    stateCpu = state_initial_npu.cpu()
    xCpu = b["xNpu"].cpu().view(B, seqSize, H) if layoutId == 0 else b["xNpu"].cpu()
    dcCpu = b["dcNpu"].cpu().float()
    golden_kw = dict(
        x=xCpu,
        wkv=b["wkvInp"].cpu(),
        wgate=b["wgateInp"].cpu(),
        kv_state=stateCpu[..., :coffD],
        score_state=stateCpu[..., coffD:],
        ape=b["ape_npu"].cpu(),
        block_table=b["block_table_npu"].cpu(),
        sps=sps,
        cr=cr,
        coff=coff,
        cache_mode=b["cache_mode"],
        cu_seqlens=cuNpu.cpu() if cuNpu is not None else None,
        seqused=sqs,
        dc=dcCpu.view(B, -1, D) if layoutId == 0 else dcCpu,
    )

    # ── 比对：NPU 四梯度 vs 小算子自动反向四梯度 ──
    d_xNpu = npu_result["d_x"].cpu().float().reshape(-1, H)
    if compare_mode == 2:
        g_dx, g_dwkv, g_dwgate, g_dape = _small_ops_golden(
            device="cpu", matmul_mode="two", compute_dtype=torch.float32, **golden_kw
        )
        p_pct_thd = get_pct_thd(dataType)
        checks = [
            (
                "d_wkv",
                g_dwkv.float(),
                npu_result["d_wkv"].cpu().float(),
                dataType,
                p_pct_thd,
            ),
            (
                "d_wgate",
                g_dwgate.float(),
                npu_result["d_wgate"].cpu().float(),
                dataType,
                p_pct_thd,
            ),
            (
                "d_ape",
                g_dape,
                npu_result["d_ape"].cpu().float(),
                "float32",
                get_pct_thd("float32"),
            ),
            ("d_x", g_dx.float(), d_xNpu, dataType, p_pct_thd),
        ]
        statuses = []
        for name, exp, act, dt, thd in checks:
            r = check_one_output(name, exp, act, dt, True, 1, thd)
            statuses.append(f"{name}={r['status']}")
    else:
        # 三方：B=NPU 小算子 same（bf16 进 bf16 出，模拟 kernel 同精度）、
        #      C=CPU 小算子 high（float64 高精度真值）——均来自自动反向推导
        b_dx, b_dwkv, b_dwgate, b_dape = _small_ops_golden(
            device=device, matmul_mode="same", compute_dtype=torch.float32, **golden_kw
        )
        c_dx, c_dwkv, c_dwgate, c_dape = _small_ops_golden(
            device="cpu", matmul_mode="high", compute_dtype=torch.float64, **golden_kw
        )
        statuses = []
        for name, act, b_out, c_out, dt in [
            (
                "d_wkv",
                npu_result["d_wkv"].cpu().float(),
                b_dwkv.float().cpu(),
                c_dwkv.float().cpu(),
                dataType,
            ),
            (
                "d_wgate",
                npu_result["d_wgate"].cpu().float(),
                b_dwgate.float().cpu(),
                c_dwgate.float().cpu(),
                dataType,
            ),
            (
                "d_ape",
                npu_result["d_ape"].cpu().float(),
                b_dape.float().cpu(),
                c_dape.float().cpu(),
                "float32",
            ),
            ("d_x", d_xNpu, b_dx.float().cpu(), c_dx.float().cpu(), dataType),
        ]:
            r = three_way_report(name, act, b_out, c_out, dt)
            statuses.append(f"{name}={r['status']}")
    status = (
        "ERROR"
        if any("ERROR" in s for s in statuses)
        else "FAIL"
        if any("FAIL" in s for s in statuses)
        else "PASS"
    )
    print_log(f"Grad check: {'  '.join(statuses)}")
    return {"status": status, "detail": "  ".join(statuses)}


# PyPTO kernel 直跑（通路 2）：不经 aclnn/custom 包，运行时 JIT 编译
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "op_kernel")
)
from compressor_grad import compressor_grad, CompressorGradTiling  # noqa: E402

from test_compressor_grad_cases import REDLINE_CASES
from compressor_grad_cpu_golden import compressor_grad_golden
from compressor_grad_npu import NPUBackend
from cann_ops_transformer.ops.compressor import (
    _compressor_forward,
    _compressor_backward,
)
import compressor_forward
from compressor_grad_check import (
    check_one_output,
    check_result,
    print_log,
    get_pct_thd,
    format_case_line,
    three_way_report,
)

DEVICE_ID = 0


def _ceil_div(a, b):
    return (a + b - 1) // b


# ── 输出精度对比开关 ──
ENABLE_DWKV_CHECK = True
ENABLE_DWGATE_CHECK = True
ENABLE_APE_CHECK = True
ENABLE_DX_CHECK = True

# ================================================================
#  Workspace reduction
# ================================================================


def reduce_ape_workspace(workspace_raw, headDim, coff, cmpRatio, roundCnt, totalBlocks):
    """
    Reduce d_ape workspace partials to final d_ape: shape (cmpRatio, coff*headDim).

    Each Vec core writes a tile of shape [coff*cmpRatio, dBaseSize] per round,
    covering its D-slice columns. All slots summed produce the final d_ape.
    """
    coreNum = 64
    dBaseSize = 128 // coff
    groupSize = headDim // 128
    nCorePerGroup = (coreNum // 2) // groupSize
    mPerRoundPerGroup = 128 * 2 // coff // cmpRatio
    mBaseSize = 128 // cmpRatio
    mPerRoundAll = nCorePerGroup * mPerRoundPerGroup

    dbRows = mPerRoundAll * cmpRatio
    ws = workspace_raw[: 2 * dbRows * headDim].view(2 * dbRows, headDim).float()
    d_ape = torch.zeros(cmpRatio, coff * headDim)

    for coreIdx in range(coreNum):
        cubeCoreIdx = coreIdx // 2
        groupIdx = cubeCoreIdx // groupSize
        intraGroupIdx = cubeCoreIdx % groupSize
        nStart = intraGroupIdx * 128 + (coreIdx % coff) * dBaseSize

        for roundIdx in range(roundCnt):
            roundBlocks = min(mPerRoundAll, totalBlocks - roundIdx * mPerRoundAll)
            preDealTcSize = groupIdx * mPerRoundPerGroup
            if coff == 1 and coreIdx % 2 == 1:
                preDealTcSize += mBaseSize
            if preDealTcSize >= roundBlocks:
                continue
            dealTcSize = min(mBaseSize, max(0, roundBlocks - preDealTcSize))
            if dealTcSize <= 0:
                continue

            dbBase = (roundIdx % 2) * dbRows
            localSlot = groupIdx * mPerRoundPerGroup * cmpRatio
            if coff == 1 and coreIdx % 2 == 1:
                localSlot += mBaseSize * cmpRatio

            # [coff*cmpRatio, dBaseSize] at (dbBase + localSlot, nStart)
            tile = ws[
                dbBase + localSlot : dbBase + localSlot + coff * cmpRatio,
                nStart : nStart + dBaseSize,
            ]

            for oid in range(coff):
                row_start = oid * cmpRatio
                col_start = oid * headDim + nStart
                d_ape[:, col_start : col_start + dBaseSize] += tile[
                    row_start : row_start + cmpRatio, :
                ]

    # Note: with 2x db, rounds N and N+2 share the same slot.
    # Phase 3 will consume per-round; standalone test sees only last 2 rounds.

    return d_ape


# ================================================================
#  Test runner
# ================================================================


def _check_block_size(
    case, batchSize, seqSize, layoutId, cmpRatio, coff, cacheMode, cuNpu, sqs
):
    """校验 case 数据的 block_size 字段合法性；不符则终止+告警（不静默替换）。

    cache_mode=1: block_size 为自由参数，需在 [1, 1024]；
    cache_mode=2: block_size 必须等于 min(coff*cmpRatio + S_for_bs - 1, 1024)
                  （S_for_bs: TH 布局 = max(seqused) 或 max(cu_seqlens 差)；BSH 布局 = S1）
    """
    bs = case["block_size"]
    name = case["testcase_name"]
    if cacheMode == 1:
        if not (1 <= bs <= 1024):
            raise SystemExit(
                f"[block_size 校验失败] {name}: cache_mode=1 block_size={bs} "
                f"不在 [1,1024] 范围内，请修正 case 数据后重跑"
            )
        return
    # cache_mode=2
    if layoutId == 1:  # TH
        if sqs:
            S_for_bs = max(sqs)
        elif cuNpu is not None:
            S_for_bs = (
                max(int(cuNpu[i + 1]) - int(cuNpu[i]) for i in range(batchSize))
                if batchSize > 0
                else 0
            )
        else:
            S_for_bs = seqSize
    else:  # BSH
        S_for_bs = seqSize
    need = coff * cmpRatio + S_for_bs - 1
    expect = min(need, 1024)
    if bs != expect:
        raise SystemExit(
            f"[block_size 校验失败] {name}: cache_mode=2 block_size={bs} "
            f"与要求 {expect} 不一致 (coff*cmpRatio+S-1={need}, S_for_bs={S_for_bs})，"
            f"请修正 case 数据后重跑"
        )


def check_forward_outputs(
    npu_result,
    xCpu,
    wkvCpu,
    wgateCpu,
    stateCacheCpu,
    apeCpu,
    blockTableCpu,
    spGolden,
    sqGolden,
    cuGolden,
    cmpRatio,
    coff,
    cacheMode,
    headDim,
    dataType,
    ref=None,
    compare_mode=3,
    device="cpu",
):
    """通路 1/5：正向输出（cmp_kv/softmax_score/kv）与小算子拼接参考比对。

    参考实现：compressor_forward.compressor_forward（纯 PyTorch 小算子拼接）。
    仅比对有效压缩块行（padding 行参考为 0、NPU 不保证写）。
    ref: 可选预计算参考（return_intermediates=True 的 6/8 元组），避免重复计算；
         为 None 时内部调用 compressor_forward。
    compare_mode=3 时：cmp_kv/sm/kv 与三方参考比对（B=NPU 小算子 same，
    C=float64 高精度 high）；state 校验仍由调用方用两方 ref 完成。
    """
    coffD = coff * headDim
    kv_state = stateCacheCpu[..., :coffD]
    score_state = stateCacheCpu[..., coffD:]
    if ref is None:
        ref = compressor_forward.compressor_forward(
            xCpu,
            wkvCpu,
            wgateCpu,
            kv_state,
            score_state,
            apeCpu,
            blockTableCpu,
            [int(v) for v in spGolden] if spGolden is not None else None,
            cmp_ratio=cmpRatio,
            coff=coff,
            cache_mode=cacheMode,
            cu_seqlens=cuGolden,
            seqused=[int(v) for v in sqGolden] if sqGolden is not None else None,
            return_intermediates=True,
        )
    ref_cmp, ref_mask, _, _, ref_sm, ref_kv = ref[:6]

    valid = ref_mask.any(dim=-1)  # (B, blocks) 或 (rows,)

    def gather(t):
        t = (
            t.detach().cpu().float()
        )  # npu 侧输出/参考输入可能带 requires_grad（autograd 图）
        return t[valid]

    fwd_key_map = {
        "cmp_kv": ("cmpkvPct", "cmpkvStatus"),
        "softmax_score": ("smPct", "smStatus"),
        "kv": ("kvPct", "kvStatus"),
    }
    if compare_mode == 3:
        b_fwd = compressor_forward.compressor_forward(
            xCpu,
            wkvCpu,
            wgateCpu,
            kv_state,
            score_state,
            apeCpu,
            blockTableCpu,
            [int(v) for v in spGolden] if spGolden is not None else None,
            cmp_ratio=cmpRatio,
            coff=coff,
            cache_mode=cacheMode,
            cu_seqlens=cuGolden,
            seqused=[int(v) for v in sqGolden] if sqGolden is not None else None,
            return_intermediates=True,
            device=device,
            matmul_mode="same",
        )
        c_fwd = compressor_forward.compressor_forward(
            xCpu,
            wkvCpu,
            wgateCpu,
            kv_state,
            score_state,
            apeCpu,
            blockTableCpu,
            [int(v) for v in spGolden] if spGolden is not None else None,
            cmp_ratio=cmpRatio,
            coff=coff,
            cache_mode=cacheMode,
            cu_seqlens=cuGolden,
            seqused=[int(v) for v in sqGolden] if sqGolden is not None else None,
            return_intermediates=True,
            device="cpu",
            compute_dtype=torch.float64,
            matmul_mode="high",
        )
        results = []
        per = {}
        for name, npu_t, b_t, c_t, dt in (
            ("cmp_kv", npu_result["cmp_kv"], b_fwd[0], c_fwd[0], dataType),
            (
                "softmax_score",
                npu_result["softmax_score"],
                b_fwd[4],
                c_fwd[4],
                "float32",
            ),
            ("kv", npu_result["kv"], b_fwd[5], c_fwd[5], "float32"),
        ):
            r = three_way_report(name, gather(npu_t), gather(b_t), gather(c_t), dt)
            r["name"] = name
            results.append(r)
            pk, sk = fwd_key_map[name]
            per[pk] = r.get("pct", float("nan"))
            per[sk] = r["status"]
        detail = "  ".join(f"{r['name']}={r['status']}" for r in results)
        ok = all(r["status"] == "PASS" for r in results)
        d = {"status": "PASS" if ok else "FAIL", "detail": detail}
        d.update(per)
        return d

    results = []
    per = {}
    for name, ref_t, npu_t, dt in (
        ("cmp_kv", ref_cmp, npu_result["cmp_kv"], dataType),
        ("softmax_score", ref_sm, npu_result["softmax_score"], "float32"),
        ("kv", ref_kv, npu_result["kv"], "float32"),
    ):
        # 与反向 4 输出同一检查函数（check_one_output）：
        # 输出块格式（check <名> / Loop 明细 / Rtol-Atol 汇总 / 错误值打印）完全一致
        r = check_one_output(
            name, gather(ref_t), gather(npu_t), dt, True, 1, get_pct_thd(dt)
        )
        r["name"] = name
        results.append(r)
        pk, sk = fwd_key_map[name]
        per[pk] = r["pct"]
        per[sk] = r["status"]
    detail = "  ".join(f"{r['name']}={r['pct']:.2f}%[{r['status']}]" for r in results)
    ok = all(r["status"] == "PASS" for r in results)
    d = {"status": "PASS" if ok else "FAIL", "detail": detail}
    d.update(per)
    return d


def check_state_outputs(
    npu_state_cache,
    ref_kv_state,
    ref_score_state,
    kv_mask,
    score_mask,
    coff,
    headDim,
    dataType,
    totalValid,
):
    """通路 5：state_cache 4 项校验（与正向 pytest 的 kv_state_update / score_state_update /
    kv_state_origin / score_state_origin 完全一致）。

    npu_state_cache: (block_num, block_size, 2*coff*headDim) float32——直调正向后已原地更新；
                     前一半列 = kv_state（stateIdx=0），后一半列 = score_state（stateIdx=1）。
    ref_kv_state / ref_score_state: 参考（compressor_forward 写回后）的 CPU state。
    kv_mask / score_mask: 参考的写入位置 bool mask（与正向 pytest 的 update_kv/update_score 同语义）。

    校验：
      *update: 写入位置 NPU vs 参考（pct_thd 常规阈值）；
      *origin: 未写位置 NPU vs 参考（= 初始值），pct_thd=0.0 严格相等（kernel 不得误写）。
    """
    coffD = coff * headDim
    kv_half = npu_state_cache[..., :coffD]
    score_half = npu_state_cache[..., coffD:]
    pct_thd = get_pct_thd("float32")
    results = []
    per = {}
    state_key_map = {
        "kv_state_update": ("kvUpdPct", "kvUpdStatus"),
        "score_state_update": ("scoreUpdPct", "scoreUpdStatus"),
        "kv_state_origin": ("kvOrgPct", "kvOrgStatus"),
        "score_state_origin": ("scoreOrgPct", "scoreOrgStatus"),
    }
    for name, exp, act, thd in (
        ("kv_state_update", ref_kv_state[kv_mask], kv_half[kv_mask], pct_thd),
        (
            "score_state_update",
            ref_score_state[score_mask],
            score_half[score_mask],
            pct_thd,
        ),
        ("kv_state_origin", ref_kv_state[~kv_mask], kv_half[~kv_mask], 0.0),
        (
            "score_state_origin",
            ref_score_state[~score_mask],
            score_half[~score_mask],
            0.0,
        ),
    ):
        r = check_one_output(name, exp, act, dataType, True, totalValid, thd)
        results.append((name, r))
        pk, sk = state_key_map[name]
        per[pk] = r["pct"]
        per[sk] = r["status"]
    detail = "  ".join(f"{n}={r['pct']:.2f}%[{r['status']}]" for n, r in results)
    ok = all(r["status"] == "PASS" for _, r in results)
    d = {"status": "PASS" if ok else "FAIL", "detail": detail}
    d.update(per)
    return d


def _prepare_case(case, device):
    """通路 1/4/5 共用：解析 case + 构造 cuNpu/totalValid/sps/sqs/seed。"""
    p = _parse_case(case, device)
    batchSize, layoutId = p["B"], p["layoutId"]
    tokenSize = p["tokenSize"]
    sps = case["start_pos"]
    if sps is None:
        sps = [0] * batchSize
    sqs = case["seqused_q"]
    seqlens_list_q = case["seqlens_list_q"]
    seed = case["manual_seed"]

    # ── 构造 cu_seqlens（TH 布局；BSH 为 None）──
    if layoutId == 1:
        # case 数据统一为"0 开始累加"的累积形式（B+1 项、首项 0），直用不再调整
        if seqlens_list_q is not None:
            cuNpu = torch.tensor(seqlens_list_q, dtype=torch.int32)
        elif sqs is not None:
            cuNpu = torch.tensor(
                [0] + torch.tensor(sqs).cumsum(0).tolist(), dtype=torch.int32
            )
        else:
            cuNpu = torch.zeros(batchSize + 1, dtype=torch.int32)
    else:
        cuNpu = None

    # ── Total valid compression blocks ──
    totalValid = 0
    for i in range(batchSize):
        sp = sps[i] if sps else 0
        if sqs:
            sq = sqs[i]
        elif cuNpu is not None:
            sq = int(cuNpu[i + 1].item() - cuNpu[i].item())
        else:
            sq = tokenSize if layoutId == 1 else tokenSize // batchSize
        cmpLimit = (sp + sq) // p["cr"] * p["cr"]
        if cmpLimit > sp:
            # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（与 kernel 一致）
            totalValid += (cmpLimit - sp + p["cr"] - 1) // p["cr"]
    return p, totalValid, cuNpu, sps, sqs, seed


def run_forward_direct_case(case, p, b, totalValid, device, compare_mode=3):
    """通路 5：单正向——直调 _compressor_forward（不经 autograd）。

    校验：
      - 正向 3 输出：cmp_kv / softmax_score / kv vs compressor_forward 参考（check_forward_outputs）；
        compare_mode=3 时与三方参考比对（B=NPU 小算子 same、C=float64 高精度 high）；
      - state_cache 4 项：kv_state_update / score_state_update / kv_state_origin /
        score_state_origin（check_state_outputs，与正向 pytest 一致；origin 严格相等）。

    返回 (result, intermediates)：intermediates = {cmp_kv, softmax_score, kv} 为
    NPU 真实中间量，供通路 4（直调反向）复用——通路 1 = 通路 5 + 通路 4 拼接。
    """
    batchSize, seqSize, headDim, hiddenSize = p["B"], p["S1"], p["D"], p["H"]
    cmpRatio, coff = p["cr"], p["coff"]
    layoutId, dataType = p["layoutId"], p["dataType"]
    wkvInp, wgateInp = b["wkvInp"], b["wgateInp"]
    state_cache_npu = b["state_cache_npu"]
    ape_npu = b["ape_npu"]
    block_table_npu = b["block_table_npu"]
    cache_mode = b["cache_mode"]
    cuArg = b["cuArg"]
    if cuArg is not None:
        cuArg = cuArg.to(
            device
        )  # _build_inputs 中 cuNpu 在 CPU 构造（golden 用）；直调必须 NPU
    sqNpu = b["inputs"]["seqused"]
    spNpu = b["inputs"]["start_pos"]
    sps = b["inputs"]["start_pos"]
    sqs = b["inputs"]["seqused"]

    # 参考输入：初始 state 必须在直调 forward 之前 clone（forward 原地更新 state_cache）
    state_cpu_initial = state_cache_npu.cpu()

    # ── 直调正向（不经 autograd）；grad_enabled=True 才会写 softmax_score/kv 中间输出 ──
    cmp_kv, softmax_score, kv = _compressor_forward(
        b["inputs"]["x"],
        wkvInp,
        wgateInp,
        state_cache_npu,
        ape_npu,
        state_block_table=block_table_npu,
        cu_seqlens=cuArg,
        seqused=sqNpu,
        start_pos=spNpu,
        cmp_ratio=cmpRatio,
        coff=coff,
        cache_mode=cache_mode,
        grad_enabled=True,
    )
    torch.npu.synchronize()
    npu_state_updated = state_cache_npu.cpu()

    # ── CPU 参考（小算子拼接 + 写回 state + update mask）──
    coffD = coff * headDim
    xCpu = b["xNpu"]
    xCpu = (
        xCpu.cpu().view(batchSize, seqSize, hiddenSize) if layoutId == 0 else xCpu.cpu()
    )
    cuGolden = cuArg.cpu() if cuArg is not None else None
    ref = compressor_forward.compressor_forward(
        xCpu,
        wkvInp.cpu(),
        wgateInp.cpu(),
        state_cpu_initial[..., :coffD],
        state_cpu_initial[..., coffD:],
        ape_npu.cpu(),
        block_table_npu.cpu(),
        [int(v) for v in sps] if sps is not None else None,
        cmp_ratio=cmpRatio,
        coff=coff,
        cache_mode=cache_mode,
        cu_seqlens=cuGolden,
        seqused=[int(v) for v in sqs] if sqs is not None else None,
        return_intermediates=True,
        return_update_mask=True,
    )
    # ref = (cmp_kv, cmp_mask, kv_state_out, score_state_out, sm, kv, kv_mask, score_mask)

    # ── 校验 ──
    npu_result = {"cmp_kv": cmp_kv, "softmax_score": softmax_score, "kv": kv}
    fwd = check_forward_outputs(
        npu_result,
        xCpu,
        wkvInp.cpu(),
        wgateInp.cpu(),
        state_cpu_initial,
        ape_npu.cpu(),
        block_table_npu.cpu(),
        sps,
        sqs,
        cuGolden,
        cmpRatio,
        coff,
        cache_mode,
        headDim,
        dataType,
        ref=ref,
        compare_mode=compare_mode,
        device=device,
    )
    state = check_state_outputs(
        npu_state_updated,
        ref[2],
        ref[3],
        ref[6],
        ref[7],
        coff,
        headDim,
        dataType,
        totalValid,
    )
    print_log(f"Forward check: {fwd['detail']}")
    print_log(f"State check: {state['detail']}")

    result = {
        "fwdStatus": fwd["status"],
        "fwdDetail": fwd["detail"],
        "stateStatus": state["status"],
        "stateDetail": state["detail"],
    }
    for k in ("cmpkvPct", "cmpkvStatus", "smPct", "smStatus", "kvPct", "kvStatus"):
        result[k] = fwd[k]
    for k in (
        "kvUpdPct",
        "kvUpdStatus",
        "scoreUpdPct",
        "scoreUpdStatus",
        "kvOrgPct",
        "kvOrgStatus",
        "scoreOrgPct",
        "scoreOrgStatus",
    ):
        result[k] = state[k]
    statuses = [fwd["status"], state["status"]]
    result["status"] = (
        "ERROR"
        if any(s == "ERROR" for s in statuses)
        else "FAIL"
        if any(s == "FAIL" for s in statuses)
        else "SKIP"
        if all(s == "SKIP" for s in statuses)
        else "PASS"
    )
    return result, {"cmp_kv": cmp_kv, "softmax_score": softmax_score, "kv": kv}


def run_backward_direct_case(case, p, b, totalValid, device, fwd=None, compare_mode=3):
    """通路 4：单反向——直调 _compressor_backward（不经 autograd）。

    中间量（softmax_score/kv）默认来自真实正向：fwd 为 None 时先直调一次
    _compressor_forward（与 golden 一致性原则相同——golden 输入用前向真实输出）；
    通路 1 拼接时由通路 5 传入复用同一份中间量。

    校验反向 4 输出：compare_mode=2 两方（vs compressor_grad_golden two）；
    compare_mode=3 三方（vs NPU 小算子 same + float64 高精度 high）。
    """
    batchSize, seqSize, headDim, hiddenSize = p["B"], p["S1"], p["D"], p["H"]
    cmpRatio, coff = p["cr"], p["coff"]
    layoutId, dataType = p["layoutId"], p["dataType"]
    wkvInp, wgateInp = b["wkvInp"], b["wgateInp"]
    dcNpu = b["dcNpu"]
    inputs = b["inputs"]
    cuArg, sqNpu, spNpu = inputs["cu_seqlens"], inputs["seqused"], inputs["start_pos"]
    if cuArg is not None:
        cuArg = cuArg.to(
            device
        )  # _build_inputs 中 cuNpu 在 CPU 构造（golden 用）；直调必须 NPU

    if fwd is None:
        # 先直调一次真实正向拿中间量（不经 autograd）
        cmp_kv, softmax_score, kv = _compressor_forward(
            inputs["x"],
            wkvInp,
            wgateInp,
            b["state_cache_npu"],
            b["ape_npu"],
            state_block_table=b["block_table_npu"],
            cu_seqlens=cuArg,
            seqused=sqNpu,
            start_pos=spNpu,
            cmp_ratio=cmpRatio,
            coff=coff,
            cache_mode=b["cache_mode"],
            grad_enabled=True,
        )
        torch.npu.synchronize()
    else:
        cmp_kv, softmax_score, kv = fwd["cmp_kv"], fwd["softmax_score"], fwd["kv"]

    # ── 直调反向（不经 autograd）──
    d_x, d_wkv, d_wgate, d_ape = _compressor_backward(
        dcNpu,
        inputs["x"],
        wkvInp,
        wgateInp,
        softmax_score,
        kv,
        cu_seqlens=cuArg,
        seqused=sqNpu,
        start_pos=spNpu,
        cmp_ratio=cmpRatio,
        coff=coff,
    )
    torch.npu.synchronize()

    # ── Golden（一份参数化实现：2=两方 two；3=三方 B same + C high）──
    dcCpu = dcNpu.cpu().float()
    spGolden = spNpu.cpu() if spNpu is not None else None
    sqGolden = sqNpu.cpu() if sqNpu is not None else None
    cuGolden = cuArg.cpu() if cuArg is not None else None
    xCpu = inputs["x"].cpu()
    if layoutId == 0:  # BSH
        xCpu = xCpu.view(batchSize, seqSize, hiddenSize)
    dcCpuView = dcCpu.view(batchSize, -1, headDim) if layoutId == 0 else dcCpu
    smCpu = softmax_score.cpu()
    kvCpu = kv.cpu()
    if compare_mode == 2:
        dXGold, dWkvGold, dWgateGold, apeGold = compressor_grad_golden(
            x=xCpu,
            wkv=wkvInp.cpu().float(),
            wgate=wgateInp.cpu().float(),
            d_cpm_kv=dcCpuView,
            softmax_score=smCpu,
            kv=kvCpu,
            cu_seqlens=cuGolden,
            seqused=sqGolden,
            start_pos=spGolden,
            cmp_ratio=cmpRatio,
            coff=coff,
        )
    else:
        b_ref = compressor_grad_golden(
            x=xCpu,
            wkv=wkvInp.cpu(),
            wgate=wgateInp.cpu(),
            d_cpm_kv=dcCpuView,
            softmax_score=smCpu,
            kv=kvCpu,
            cu_seqlens=cuGolden,
            seqused=sqGolden,
            start_pos=spGolden,
            cmp_ratio=cmpRatio,
            coff=coff,
            device=device,
            matmul_mode="same",
        )
        c_ref = compressor_grad_golden(
            x=xCpu,
            wkv=wkvInp.cpu(),
            wgate=wgateInp.cpu(),
            d_cpm_kv=dcCpuView,
            softmax_score=smCpu,
            kv=kvCpu,
            cu_seqlens=cuGolden,
            seqused=sqGolden,
            start_pos=spGolden,
            cmp_ratio=cmpRatio,
            coff=coff,
            device="cpu",
            compute_dtype=torch.float64,
            matmul_mode="high",
        )

    # ── 反向 4 输出比对 ──
    dWkvNp = d_wkv.cpu().float()
    dWgateNp = d_wgate.cpu().float()
    apeNp = d_ape.cpu().view(cmpRatio, coff * headDim)
    dXNp = d_x.cpu().float()

    print_log("=" * 80)
    print_log(
        f"Start precision check for case '{case['testcase_name']}' (validBlocks={totalValid})"
    )

    if compare_mode == 2:
        p_pct_thd = get_pct_thd(dataType)
        dXGoldF = dXGold.float()
        r_dwkv = check_one_output(
            "d_wkv",
            dWkvGold.float(),
            dWkvNp,
            dataType,
            ENABLE_DWKV_CHECK,
            totalValid,
            p_pct_thd,
        )
        r_dwgate = check_one_output(
            "d_wgate",
            dWgateGold.float(),
            dWgateNp,
            dataType,
            ENABLE_DWGATE_CHECK,
            totalValid,
            p_pct_thd,
        )
        r_ape = check_one_output(
            "d_ape",
            apeGold,
            apeNp,
            "float32",
            ENABLE_APE_CHECK,
            totalValid,
            get_pct_thd("float32"),
        )
        r_dx = check_one_output(
            "d_x", dXGoldF, dXNp, dataType, ENABLE_DX_CHECK, totalValid, p_pct_thd
        )
    else:
        b_dx, b_dwkv, b_dwgate, b_dape = b_ref
        c_dx, c_dwkv, c_dwgate, c_dape = c_ref
        r_dwkv = three_way_report("d_wkv", dWkvNp, b_dwkv, c_dwkv, dataType)
        r_dwgate = three_way_report("d_wgate", dWgateNp, b_dwgate, c_dwgate, dataType)
        r_ape = three_way_report(
            "d_ape",
            apeNp,
            b_dape.view(cmpRatio, coff * headDim),
            c_dape.view(cmpRatio, coff * headDim),
            "float32",
        )
        r_dx = three_way_report("d_x", dXNp, b_dx, c_dx, dataType)

    result = dict(
        dwkvDiff=r_dwkv.get("diff", float("nan")),
        dwkvPct=r_dwkv.get("pct", float("nan")),
        dwkvStatus=r_dwkv["status"],
        dwgateDiff=r_dwgate.get("diff", float("nan")),
        dwgatePct=r_dwgate.get("pct", float("nan")),
        dwgateStatus=r_dwgate["status"],
        apeDiff=r_ape.get("diff", float("nan")),
        apePct=r_ape.get("pct", float("nan")),
        apeStatus=r_ape["status"],
        dxDiff=r_dx.get("diff", float("nan")),
        dxPct=r_dx.get("pct", float("nan")),
        dxStatus=r_dx["status"],
    )
    statuses = [r_dwkv["status"], r_dwgate["status"], r_ape["status"], r_dx["status"]]
    result["status"] = (
        "ERROR"
        if any(s == "ERROR" for s in statuses)
        else "FAIL"
        if any(s == "FAIL" for s in statuses)
        else "SKIP"
        if all(s == "SKIP" for s in statuses)
        else "PASS"
    )
    print_log("Grad check: d_wkv=%s  d_wgate=%s  d_ape=%s  d_x=%s" % tuple(statuses))
    return result


def _parse_case(case, device):
    """解析 case 公共参数（通路 1/2/4/5 共用）。"""
    B, S1, H, D = case["B"], case["S1"], case["hidden_size"], case["D"]
    cr, coff = case["cmp_ratio"], case["coff"]
    layout = case["input_layout"]
    layoutId = 1 if layout == "TH" else 0
    dataType = case["dtype"]
    ioDtype = torch.float16 if dataType == "float16" else torch.bfloat16
    if layoutId == 1:
        tokenSize = int(case["seqlens_list_q"][-1])
        outputRows = min(tokenSize, tokenSize // cr + B)
        seqSize = 0  # 与 host tiling 一致（TH 不设置 seqSize）
    else:
        tokenSize = B * S1
        outputRows = B * _ceil_div(S1, cr)
        seqSize = S1
    return dict(
        B=B,
        S1=S1,
        H=H,
        D=D,
        cr=cr,
        coff=coff,
        layout=layout,
        layoutId=layoutId,
        dataType=dataType,
        ioDtype=ioDtype,
        tokenSize=tokenSize,
        seqSize=seqSize,
        outputRows=outputRows,
    )


def _build_inputs(
    case, p, device, seed, totalValid, cuNpu, sqs, sps, state_init="zeros"
):
    """构造通路 1/3/4/5 共用的 NPU 输入（x/wkv/wgate/state_cache/ape/dc/block_table + 输出占位）。

    返回 compute inputs dict + 附带 tensor（输出占位用 rand 模拟未初始化内存，
    不依赖初始 0——kernel 未写输出时比对必须 FAIL 而非被零掩盖）。
    state_init: "zeros"（通路 1/3/4，反向不依赖 state 内容）或
                "rand"（通路 5 state 校验：与正向 pytest 的 kv_state/score_state
                随机 uniform(-10,10) 初始一致，origin 校验未写位置保持原值）。
    """
    batchSize, seqSize, headDim, hiddenSize = p["B"], p["S1"], p["D"], p["H"]
    cmpRatio, coff = p["cr"], p["coff"]
    layoutId, ioDtype = p["layoutId"], p["ioDtype"]
    layout = p["layout"]
    tokenSize, outputRows = p["tokenSize"], p["outputRows"]
    sqNpu = torch.tensor(sqs, device=device, dtype=torch.int32) if sqs else None
    spNpu = torch.tensor(sps, device=device, dtype=torch.int32) if sps else None

    # ── Workspace sizing（仅注释参考，autograd 路径由 aclnn 分配）──
    scKvRowCount = coff * cmpRatio
    coreNum = 64
    groupSize = headDim // 128
    nCorePerGroup = (coreNum // 2) // groupSize
    mPerRoundPerGroup = 128 * 2 // coff // cmpRatio
    mPerRoundAll = nCorePerGroup * mPerRoundPerGroup
    dbRows = mPerRoundAll * cmpRatio
    apeWsSize = 2 * dbRows * headDim
    dxPerGroup = 256 * hiddenSize
    dxPerRoundAll = 32 * dxPerGroup
    dxWsSize = 2 * dxPerRoundAll * 2
    cvGmPerCore = 128 * coff * (128 // coff)
    cvGmSize = 32 * cvGmPerCore * 2
    wsSize = apeWsSize + dxWsSize + cvGmSize

    # ── Generate random inputs ──
    torch.manual_seed(seed)
    if layoutId == 0:  # BSH: dc 与 cmp_kv 同为 (B, blocks, D)，kernel 按扁平读兼容
        dcNpu = torch.randn(
            batchSize,
            (seqSize + cmpRatio - 1) // cmpRatio,
            headDim,
            device=device,
            dtype=ioDtype,
        )
    else:
        dcNpu = torch.randn(outputRows, headDim, device=device, dtype=ioDtype)

    # ── Output tensors（占位 rand）──
    totalTokens = tokenSize if layoutId == 1 else batchSize * seqSize
    d_xNpu = torch.rand(totalTokens, hiddenSize, device=device, dtype=ioDtype)
    d_wkvNpu = torch.rand(
        coff * headDim, hiddenSize, device=device, dtype=torch.float32
    )
    d_wgateNpu = torch.rand(
        coff * headDim, hiddenSize, device=device, dtype=torch.float32
    )
    dapeWsNpu = torch.rand(wsSize, device=device, dtype=torch.float32)
    dapeOutNpu = torch.rand(
        coff * cmpRatio * headDim, device=device, dtype=torch.float32
    )

    # cu_seqlens argument (None if BSH or noCu)
    cuArg = None if (layoutId == 0 or case.get("noCu")) else cuNpu

    # x / wkv / wgate input tensors
    xNpu = torch.randn(totalTokens, hiddenSize, device=device, dtype=ioDtype)
    wkvInp = torch.randn(coff * headDim, hiddenSize, device=device, dtype=ioDtype)
    wgateInp = torch.randn(coff * headDim, hiddenSize, device=device, dtype=ioDtype)

    # ── state_cache 与 block_table（构造与正向 golden 一致）──
    cache_mode = case.get("cache_mode", 1)
    if cache_mode == 1:
        block_size = case.get("block_size", 128)
        if layoutId == 1:  # TH
            cu_lens = cuNpu.tolist() if cuNpu is not None else [0] * (batchSize + 1)
            if sps:
                S_max = (
                    max(
                        sps[i] + (cu_lens[i + 1] - cu_lens[i]) for i in range(batchSize)
                    )
                    if batchSize > 0
                    else 0
                )
            else:
                S_max = (
                    max((cu_lens[i + 1] - cu_lens[i]) for i in range(batchSize))
                    if batchSize > 0
                    else 0
                )
        else:  # BSH
            S_max = max(sps) + seqSize if sps else seqSize
        max_block_num_per_batch = (S_max + block_size - 1) // block_size
        block_num = batchSize * max_block_num_per_batch
        state_shape = (block_num, block_size, 2 * coff * headDim)
    else:
        # cache_mode=2: block_size 直用 case 数据（合法性由 _check_block_size 校验）
        state_shape = (batchSize, case["block_size"], 2 * coff * headDim)

    _check_block_size(
        case, batchSize, seqSize, layoutId, cmpRatio, coff, cache_mode, cuNpu, sqs
    )

    if state_init == "rand":
        # 与正向 pytest 的 kv_state/score_state 初始一致：uniform(-10, 10)
        state_cache_npu = torch.empty(
            state_shape, device=device, dtype=torch.float32
        ).uniform_(-10, 10)
    else:
        state_cache_npu = torch.zeros(state_shape, device=device, dtype=torch.float32)
    ape_npu = torch.randn(
        torch.Size([cmpRatio, coff * headDim]), device=device, dtype=torch.float32
    )

    block_table_npu = None
    if cache_mode == 1:
        block_table = torch.zeros(batchSize, max_block_num_per_batch, dtype=torch.int32)
        if batchSize > 0 and totalValid > 0:
            next_block_id = 1
            for i in range(batchSize):
                sp = sps[i] if sps else 0
                if sqs:
                    sq = sqs[i]
                elif cuNpu is not None:
                    sq = int(cuNpu[i + 1].item() - cuNpu[i].item())
                else:
                    sq = tokenSize if layoutId == 1 else tokenSize // batchSize
                end_pos = sq
                # 读取范围 (与 golden compressor_golden.py:572-583 一致)
                cur_start = sp // cmpRatio * cmpRatio - cmpRatio
                cur_end = sp // cmpRatio * cmpRatio + cmpRatio
                if sp % cmpRatio == 0:
                    cur_end = sp
                cur_end = min(cur_end, sp + sq)
                for j in range(
                    max(cur_start // block_size, 0), (cur_end - 1) // block_size + 1
                ):
                    if next_block_id < block_num:
                        block_table[i][j] = next_block_id
                        next_block_id = next_block_id + 1
                # 写入范围 (与 golden compressor_golden.py:589-600 一致)
                next_start = (sp + end_pos) // cmpRatio * cmpRatio - cmpRatio
                next_end = (sp + end_pos) // cmpRatio * cmpRatio + cmpRatio
                if (sp + end_pos) % cmpRatio == 0:
                    next_end = sp + end_pos
                next_end = min(next_end, sp + end_pos)
                for j in range(
                    max(next_start // block_size, 0), (next_end - 1) // block_size + 1
                ):
                    if next_block_id < block_num and block_table[i][j] == 0:
                        block_table[i][j] = next_block_id
                        next_block_id = next_block_id + 1
        block_table_npu = block_table.to(device)
    else:
        block_table_npu = (
            torch.tensor(
                random.sample(list(range(batchSize)), batchSize),
                device=device,
                dtype=torch.int32,
            )
            if batchSize > 0
            else torch.zeros(0, device=device, dtype=torch.int32)
        )

    x_npu_input = xNpu.view(batchSize, seqSize, hiddenSize) if layoutId == 0 else xNpu
    inputs = {
        "x": x_npu_input,
        "wkv": wkvInp,
        "wgate": wgateInp,
        "state_cache": state_cache_npu,
        "ape": ape_npu,
        "d_cpm_kv": dcNpu,
        "cu_seqlens": cuArg,
        "seqused": sqNpu,
        "start_pos": spNpu,
        "block_table": block_table_npu,
        "cmp_ratio": cmpRatio,
        "coff": coff,
        "cache_mode": cache_mode,
        "input_layout": layout,
    }
    return dict(
        inputs=inputs,
        xNpu=xNpu,
        wkvInp=wkvInp,
        wgateInp=wgateInp,
        dcNpu=dcNpu,
        state_cache_npu=state_cache_npu,
        ape_npu=ape_npu,
        block_table_npu=block_table_npu,
        cuArg=cuArg,
        d_xNpu=d_xNpu,
        d_wkvNpu=d_wkvNpu,
        d_wgateNpu=d_wgateNpu,
        dapeOutNpu=dapeOutNpu,
        totalTokens=totalTokens,
        cache_mode=cache_mode,
        scKvRowCount=scKvRowCount,
    )


def run_case(case, device, compare_mode=3):
    """通路 1：正反向全链路 = 通路 5（直调正向）+ 通路 4（直调反向）拼接。

    共享同一份输入构造（_build_inputs, state 随机初始）与 NPU 真实中间量
    （softmax_score/kv），校验正向 3 输出 + state_cache 4 项 + 反向 4 输出。
    """
    name = case["testcase_name"]
    p, totalValid, cuNpu, sps, sqs, seed = _prepare_case(case, device)
    b = _build_inputs(
        case, p, device, seed, totalValid, cuNpu, sqs, sps, state_init="rand"
    )

    # 通路 5：直调正向 + 校验（cmp_kv/sm/kv + state 4 项），返回 NPU 中间量
    fwd_result, fwd_inter = run_forward_direct_case(
        case, p, b, totalValid, device, compare_mode=compare_mode
    )
    # 通路 4：直调反向（复用同一份中间量）+ 校验（反向 4 输出）
    bwd_result = run_backward_direct_case(
        case, p, b, totalValid, device, fwd=fwd_inter, compare_mode=compare_mode
    )

    result = dict(
        name=name,
        totalValid=totalValid,
        coff=p["coff"],
        cmpRatio=p["cr"],
        headDim=p["D"],
        batchSize=p["B"],
    )
    result.update(fwd_result)
    result.update(bwd_result)

    # ── Overall status ──
    # 任一 FAIL → FAIL；任一 ERROR → ERROR
    # 注：totalValid=0 不再 trivial PASS——四输出必须真实写 0（golden 全 0），
    #      kernel 未写输出（垃圾）时校验必须 FAIL（见 _zero_outputs）
    statuses = [
        result["dwkvStatus"],
        result["dwgateStatus"],
        result["apeStatus"],
        result["dxStatus"],
        result["fwdStatus"],
        result["stateStatus"],
    ]
    if any(s == "ERROR" for s in statuses):
        result["status"] = "ERROR"
    elif any(s == "FAIL" for s in statuses):
        result["status"] = "FAIL"
    elif all(s == "SKIP" for s in statuses):
        result["status"] = "SKIP"
    else:
        result["status"] = "PASS"

    return result


def run_backward_case(case, device, compare_mode=3):
    """通路 2：单反向——随机构造中间量 → PyPTO 直跑 kernel → golden 比对。

    不经 aclnn/custom 包（kernel 运行时 JIT 编译）；TilingData 29 字段
    派生逻辑与 op_host/compressor_grad_tiling.cpp 一致。
    """
    p = _parse_case(case, device)
    B, H, D = p["B"], p["H"], p["D"]
    cr, coff = p["cr"], p["coff"]
    layoutId, ioDtype = p["layoutId"], p["ioDtype"]
    tokenSize, seqSize, outputRows = p["tokenSize"], p["seqSize"], p["outputRows"]
    sqs, sps = case.get("seqused_q"), case.get("start_pos")

    if layoutId == 1:
        cuNpu = torch.tensor(case["seqlens_list_q"], device=device, dtype=torch.int32)
    else:
        cuNpu = None
    sqNpu = torch.tensor(sqs, device=device, dtype=torch.int32) if sqs else None
    spNpu = torch.tensor(sps, device=device, dtype=torch.int32) if sps else None

    # ── TilingData（派生与 op_host/compressor_grad_tiling.cpp 一致）──
    cubeCoreNum, coreNum = 32, 64
    coffCoef = 2 // coff
    totalHeadDim = coff * D
    cmpSize = coff * cr * D
    cmpKvBatchStride = _ceil_div(seqSize, cr)
    xRows = tokenSize if layoutId == 1 else B * seqSize
    groupSize = D // 128
    groupNum = cubeCoreNum // groupSize
    dealScNum = 128 // cr
    groupDealScNum = dealScNum * coffCoef
    totalScNumPerRound = groupNum * groupDealScNum
    groupRowStride = groupDealScNum * cr + (coff - 1) * cr
    dbRowCnt = groupNum * groupRowStride
    tiling = CompressorGradTiling(
        batch_size=B,
        token_size=tokenSize,
        seq_size=seqSize,
        cmp_ratio=cr,
        hidden_size=H,
        head_dim=D,
        cube_core_num=cubeCoreNum,
        core_num=coreNum,
        total_head_dim=totalHeadDim,
        cmp_row_cnt=coff * cr,
        cmp_size=cmpSize,
        cmp_kv_batch_stride=cmpKvBatchStride,
        cmp_kv_rows=outputRows,
        x_rows=xRows,
        group_size=groupSize,
        group_num=groupNum,
        group_deal_sc_num=groupDealScNum,
        deal_sc_num=dealScNum,
        total_sc_num_per_round=totalScNumPerRound,
        db_row_cnt=dbRowCnt,
        group_row_stride=groupRowStride,
        coff_coef=coffCoef,
        cube_m_base_size=128 * coffCoef,
        d_deal_size=128 // coff,
        m_deal_size=128 * coff,
        dape_ws_size=groupNum * cmpSize * coffCoef,
        d_x_ws_size=2 * cubeCoreNum * 256 * H,
        d_w_weight_ws_size=groupNum * totalHeadDim * H,
        x_ws_size=2 * groupNum * groupRowStride * H * groupSize,
        d_x_cache_ws_size=2 * cr * H,
    )

    # ── 随机构造输入/中间量（不依赖正向输出）──
    torch.manual_seed(case["manual_seed"])
    xNpu = torch.randn(xRows, H, device=device, dtype=ioDtype)
    wkvInp = torch.randn(coff * D, H, device=device, dtype=ioDtype)
    wgateInp = torch.randn(coff * D, H, device=device, dtype=ioDtype)
    dcNpu = torch.randn(outputRows, D, device=device, dtype=ioDtype)
    kvNpu = torch.randn(outputRows, coff * cr, D, device=device, dtype=torch.float32)
    smNpu = torch.softmax(torch.randn(outputRows, coff * cr, D, device=device), dim=1)

    # ── 输出与 workspace（精确大小，FP32 元素数）──
    # ⚠️ 全部用随机值初始化：kernel 未写区域将暴露为垃圾而非被 0 掩盖
    # （曾因 workspace=zeros 掩盖 d_ape 未写槽位读入问题）
    dX = torch.rand(xRows, H, device=device, dtype=ioDtype)
    dWkv = torch.rand(coff * D, H, device=device, dtype=ioDtype)
    dWgate = torch.rand(coff * D, H, device=device, dtype=ioDtype)
    dApe = torch.rand(cr * coff * D, device=device, dtype=torch.float32)
    wsSize = (
        tiling.dape_ws_size
        + tiling.d_x_ws_size
        + 2 * tiling.d_w_weight_ws_size
        + tiling.x_ws_size
        + tiling.d_x_cache_ws_size
    )
    workspace = torch.rand(wsSize, device=device, dtype=torch.float32)

    # ── PyPTO 直跑（不经 aclnn/custom 包）──
    compressor_grad[
        None,
        cubeCoreNum,
        {
            "Coff": coff,
            "Layout": layoutId,
            "DataType": 0 if ioDtype == torch.bfloat16 else 1,
        },
    ](
        xNpu,
        wkvInp,
        wgateInp,
        dcNpu,
        smNpu,
        kvNpu,
        cuNpu,
        sqNpu,
        spNpu,
        dX,
        dWkv,
        dWgate,
        dApe,
        workspace,
        tiling,
    )
    torch.npu.synchronize()

    # ── golden（BSH 输入需 3 维；dc/sm/kv 需 (B, blocks, ...) 维）──
    xCpu = xNpu.cpu().view(B, seqSize, H) if layoutId == 0 else xNpu.cpu()
    smCpu = smNpu.cpu().view(B, -1, coff * cr, D) if layoutId == 0 else smNpu.cpu()
    kvCpu = kvNpu.cpu().view(B, -1, coff * cr, D) if layoutId == 0 else kvNpu.cpu()
    dcCpu = dcNpu.cpu().view(B, -1, D) if layoutId == 0 else dcNpu.cpu()
    golden_kw = dict(
        x=xCpu,
        wkv=wkvInp.cpu(),
        wgate=wgateInp.cpu(),
        d_cpm_kv=dcCpu,
        softmax_score=smCpu,
        kv=kvCpu,
        cu_seqlens=cuNpu.cpu() if cuNpu is not None else None,
        seqused=sqNpu.cpu() if sqNpu is not None else None,
        start_pos=spNpu.cpu() if spNpu is not None else None,
        cmp_ratio=cr,
        coff=coff,
    )
    if compare_mode == 2:
        g_dx, g_dwkv, g_dwgate, g_dape = compressor_grad_golden(**golden_kw)
        p_pct_thd = get_pct_thd(p["dataType"])
        checks = [
            ("d_wkv", g_dwkv.float(), dWkv.cpu().float(), p["dataType"], p_pct_thd),
            (
                "d_wgate",
                g_dwgate.float(),
                dWgate.cpu().float(),
                p["dataType"],
                p_pct_thd,
            ),
            ("d_ape", g_dape, dApe.cpu().float(), "float32", get_pct_thd("float32")),
            ("d_x", g_dx.float(), dX.cpu().float(), p["dataType"], p_pct_thd),
        ]
        statuses = []
        for name, exp, act, dt, thd in checks:
            r = check_one_output(name, exp, act, dt, True, 1, thd)
            statuses.append(f"{name}={r['status']}")
    else:
        b_ref = compressor_grad_golden(**golden_kw, device=device, matmul_mode="same")
        c_ref = compressor_grad_golden(
            **golden_kw, device="cpu", compute_dtype=torch.float64, matmul_mode="high"
        )
        b_dx, b_dwkv, b_dwgate, b_dape = b_ref
        c_dx, c_dwkv, c_dwgate, c_dape = c_ref
        statuses = []
        for name, act, b_out, c_out, dt in [
            ("d_wkv", dWkv.cpu().float(), b_dwkv, c_dwkv, p["dataType"]),
            ("d_wgate", dWgate.cpu().float(), b_dwgate, c_dwgate, p["dataType"]),
            ("d_ape", dApe.cpu().float().view(cr, coff * D), b_dape, c_dape, "float32"),
            ("d_x", dX.cpu().float(), b_dx, c_dx, p["dataType"]),
        ]:
            r = three_way_report(name, act, b_out, c_out, dt)
            statuses.append(f"{name}={r['status']}")
    status = (
        "ERROR"
        if any("ERROR" in s for s in statuses)
        else "FAIL"
        if any("FAIL" in s for s in statuses)
        else "PASS"
    )
    print_log(f"Grad check: {'  '.join(statuses)}")
    return {"status": status, "detail": "  ".join(statuses)}


def _pathway(request):
    """当前验证通路（--pathway 入参）。"""
    return request.config.getoption("--pathway")


def _compare_mode(request):
    """当前精度比对模式（--compare-mode: 2=两方 3=三方）。"""
    return request.config.getoption("--compare-mode")


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad(case, request):
    """通路 1：autograd 正反向全链路 + 反向四输出校验
    + 正向输出校验（cmp_kv/softmax_score/kv vs compressor_forward 参考）。

    运行：pytest test_compressor_grad.py -k <case名子串> -q
    """
    if _pathway(request) != 1:
        pytest.skip(f"pathway={_pathway(request)}")
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    result = run_case(case, device, _compare_mode(request))
    assert result["status"] == "PASS", format_case_line(result)


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad_backward(case, request):
    """通路 2：单反向（PyPTO 直跑，随机构造中间量，不经 custom 包）。

    运行：pytest test_compressor_grad.py --pathway 2 -k <case名子串> -q
    """
    if _pathway(request) != 2:
        pytest.skip(f"pathway={_pathway(request)}")
    result = run_backward_case(case, f"npu:{DEVICE_ID}", _compare_mode(request))
    assert result["status"] == "PASS", result["detail"]


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad_small_ops(case, request):
    """通路 3：小算子拼接 golden + pta 串联正反向（loss 只对有效压缩块）。

    运行：pytest test_compressor_grad.py --pathway 3 -k <case名子串> -q
    """
    if _pathway(request) != 3:
        pytest.skip(f"pathway={_pathway(request)}")
    result = run_small_ops_case(case, f"npu:{DEVICE_ID}", _compare_mode(request))
    assert result["status"] == "PASS", result["detail"]


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad_backward_direct(case, request):
    """通路 4：单反向——通过正向接口内部函数 _compressor_backward 直接调用
    （不经 autograd，经 custom 包/aclnn）；中间量来自真实正向（先直调一次
    _compressor_forward，与通路 1 的 golden 一致性原则相同）。

    运行：pytest test_compressor_grad.py --pathway 4 -k <case名子串> -q
    """
    if _pathway(request) != 4:
        pytest.skip(f"pathway={_pathway(request)}")
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    p, totalValid, cuNpu, sps, sqs, seed = _prepare_case(case, device)
    b = _build_inputs(
        case, p, device, seed, totalValid, cuNpu, sqs, sps, state_init="rand"
    )
    result = run_backward_direct_case(
        case, p, b, totalValid, device, compare_mode=_compare_mode(request)
    )
    assert result["status"] == "PASS", format_case_line(result)


@pytest.mark.parametrize(
    "case",
    [c for c in REDLINE_CASES if c.get("enable", True)],
    ids=[c["testcase_name"] for c in REDLINE_CASES if c.get("enable", True)],
)
def test_compressor_grad_forward_direct(case, request):
    """通路 5：单正向——通过正向接口内部函数 _compressor_forward 直接调用
    （不经 autograd，经 custom 包/aclnn），校验正向 3 输出
    （cmp_kv/softmax_score/kv）+ state_cache 4 项
    （kv_state_update/score_state_update/kv_state_origin/score_state_origin，
    与正向算子 pytest 一致）。

    运行：pytest test_compressor_grad.py --pathway 5 -k <case名子串> -q
    """
    if _pathway(request) != 5:
        pytest.skip(f"pathway={_pathway(request)}")
    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    p, totalValid, cuNpu, sps, sqs, seed = _prepare_case(case, device)
    b = _build_inputs(
        case, p, device, seed, totalValid, cuNpu, sqs, sps, state_init="rand"
    )
    result, _ = run_forward_direct_case(
        case, p, b, totalValid, device, compare_mode=_compare_mode(request)
    )
    assert result["status"] == "PASS", format_case_line(result)
