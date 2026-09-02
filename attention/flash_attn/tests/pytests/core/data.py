# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""声明式张量生成 + 按文档参数组构造参数。"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 参数组构造 — 按 flash_attn.md 的 6 组参数规范化
# ============================================================


def build_flash_attn_params(
    p: dict, device: torch.device, inputs: dict
) -> Tuple[dict, dict, dict]:
    """从 normalized params 按参数组构造 metadata / kernel kwargs。

    Returns: (meta_kwargs, kernel_kwargs, out_layout)
    """
    layout_q = p.get("layout_q", p.get("input_layout", "BNSD"))
    layout_kv = p.get("layout_kv", layout_q)
    layout_out = p.get("layout_out", layout_q)
    d = p["D"]

    # === 公共参数组 ===
    scale = p.get("scale", 1 / (d**0.5))

    # === Mask 参数组 ===
    mask_mode = p.get("mask_mode", 0)
    wl = wr = -1
    attn_mask = None
    is_causal = False
    if mask_mode == 3:
        is_causal = True
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
        attn_mask = mask.to(dtype=torch.int8, device=device)
    elif mask_mode == 4:
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
        attn_mask = mask.to(dtype=torch.int8, device=device)
        wl = p.get("win_left", 2147483647)
        wr = p.get("win_right", 2147483647)
    elif mask_mode != 0:
        wl = p.get("win_left", 2147483647)
        wr = p.get("win_right", 2147483647)

    # === SeqLens 参数组 ===
    def _u32(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(torch.int32, device=device).contiguous()
        return torch.tensor(x, dtype=torch.int32, device=device).contiguous()

    cu_q = _u32(p.get("cu_seqlens_q"))
    cu_kv = _u32(p.get("cu_seqlens_kv"))
    sq = _u32(p.get("seqused_q", p.get("actual_seq_qlen")))
    skv = _u32(p.get("seqused_kv", p.get("actual_seq_kvlen")))

    def _resolve_max_seqlen(explicit, used_seq, cu_seqlens, fallback):
        if explicit is not None:
            return int(explicit)
        if used_seq is not None:
            return int(used_seq.max())
        if cu_seqlens is not None:
            return int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        return fallback

    # 从 tensor shape 推导 metadata 参数
    q = inputs["q"]
    k = inputs["k"]
    if layout_q == "BNSD":
        bs, nqh, s_q, hd = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        nkh, s_kv = k.shape[1], k.shape[2]
    elif layout_q == "BSND":
        bs, nqh, s_q, hd = q.shape[0], q.shape[2], q.shape[1], q.shape[3]
        nkh, s_kv = k.shape[2], k.shape[1]
    elif layout_q == "TND":
        bs = (cu_q.shape[0] - 1) if cu_q is not None else p.get("B", 1)
        nqh, hd = q.shape[1], q.shape[2]
        nkh = k.shape[1]
        s_q = p.get("S1", 1)
        s_kv = p.get("S2", p.get("S1", 1))
    else:
        bs = q.shape[0]
        nqh = p["N1"]
        nkh = p.get("N2", nqh)
        hd = d
        s_q = q.shape[1]
        s_kv = k.shape[1]

    msq = _resolve_max_seqlen(p.get("max_seqlen_q"), sq, cu_q, s_q)
    msk = _resolve_max_seqlen(p.get("max_seqlen_kv"), skv, cu_kv, s_kv)

    # === Paged Attention 参数组 ===
    bt = None
    if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        bt = inputs.get("block_table")
        if bt is not None:
            bt = bt.to(device)
            nkh = p.get("N2", nqh)
            hd = d
            msk = _resolve_max_seqlen(p.get("max_seqlen_kv"), skv, cu_kv, s_kv)

    # === metadata kwargs ===
    meta_kwargs = dict(
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        seqused_q=sq,
        seqused_kv=skv,
        num_heads_q=nqh,
        num_heads_kv=nkh,
        head_dim=hd,
        batch_size=bs,
        max_seqlen_q=msq,
        max_seqlen_kv=msk,
        mask_mode=mask_mode,
        win_left=wl,
        win_right=wr,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
    )

    # === kernel kwargs ===
    kernel_kwargs = dict(
        softmax_scale=scale,
        mask_mode=mask_mode,
        win_left=wl,
        win_right=wr,
        block_table=bt,
        sinks=None,
        attn_mask=attn_mask,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        seqused_q=sq,
        seqused_kv=skv,
        max_seqlen_q=msq,
        max_seqlen_kv=msk,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=p.get("return_softmax_lse", False),
    )

    return meta_kwargs, kernel_kwargs, layout_out


class InputSpec:
    """声明式张量数据生成器。

    用法:
        spec = (InputSpec("flash_attn")
            .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D"})
            .tensor("k", layout={"BNSD": "B,N2,S2,D"})
        )
        inputs = spec.generate(params, device, layout="BNSD")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._items: List[dict] = []

    # ── 核心 API ──

    def tensor(
        self,
        name: str,
        *,
        shape: str = "",
        layout: dict = None,
        dtype=None,
        method: str = "randn",
        when: Callable = None,
        layout_override: str = None,  # 覆盖全局 layout，如 k/v 用 layout_kv
        range_key: str = None,  # 从 params[range_key] 读取 (lo, hi) 覆盖 low/high
        **kwargs,
    ) -> "InputSpec":
        """添加一个张量定义。

        Args:
            name:   张量名，如 "q", "k", "v", "block_table"
            shape:  默认 shape 表达式，如 "B,N1,S1,D"
            layout: 按 layout 的 shape 映射，如 {"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D"}
            dtype:  torch.dtype，默认从 params["Dtype"] 推断
            method: "randn" | "uniform" | "fixed" | "zeros" | "custom"
            when:   fn(params) → bool，条件生成（None = 总是生成）
            layout_override: 覆盖全局 layout，如 k/v 应该用 layout_kv
            range_key: params 中的 key 名（如 "q_range"），其值 (lo, hi) 覆盖 low/high
        """
        self._items.append(
            dict(
                name=name,
                shape=shape,
                layout=layout or {},
                dtype=dtype,
                method=method,
                when=when,
                layout_override=layout_override,
                range_key=range_key,
                kwargs=kwargs,
            )
        )
        return self

    # ── 生成入口 ──

    def generate(
        self,
        params: dict,
        device,
        layout: str = "BNSD",
        layout_kv: str = None,
        seed: int = 42,
    ) -> Dict[str, torch.Tensor]:
        """在指定 device 上生成所有张量。"""
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        tensors = {}
        for it in self._items:
            if it["when"] and not it["when"](params):
                continue
            # layout_override: 字符串 → 从 params 取值; None → 用全局 layout
            lo = it.get("layout_override")
            if lo:
                eff_layout = params.get(lo, layout)
            else:
                eff_layout = layout
            shape = self._resolve(it, params, eff_layout)
            if not shape:
                continue
            dt = it["dtype"] or _infer_dtype(params.get("Dtype", "fp16"))
            kw = dict(it["kwargs"])
            rk = it.get("range_key")
            if rk and rk in params:
                lo, hi = params[rk]
                kw["low"], kw["high"] = float(lo), float(hi)
            t = _generate(it["method"], shape, dt, device, rng, **kw)
            tensors[it["name"]] = t
        return tensors

    # ── shape 解析 ──

    def _resolve(self, item, params, layout):
        shape_str = item["layout"].get(layout, item["shape"])
        if not shape_str:
            return None
        dims = []
        for token in shape_str.split(","):
            token = token.strip()
            if token in params:
                dims.append(params[token])
            elif token == "_V":
                dims.append(params.get("DV", params["D"]))
            elif token == "total_s1":
                cu = params.get("cu_seqlens_q")
                dims.append(cu[-1] if cu else params.get("S1", 1))
            elif token == "total_s2":
                cu = params.get("cu_seqlens_kv")
                dims.append(cu[-1] if cu else params.get("S2", params.get("S1", 1)))
            elif token == "max_blocks":
                sv = params.get("seqused_kv", [params.get("S2", 1)])
                bs = params.get("block_size", 128)
                sm = max(sv) if isinstance(sv, list) else sv
                dims.append((sm + bs - 1) // bs if sm > 0 else 1)
            elif token == "B+1":
                cu = params.get("cu_seqlens_q")
                dims.append(len(cu) if cu else params.get("B", 1) + 1)
            elif token == "num_blocks":
                dims.append(params.get("num_blocks", 1))
            else:
                dims.append(int(token))
        return tuple(dims)


# ── helpers ──


def _infer_dtype(ds: str) -> torch.dtype:
    return torch.float16 if ds == "fp16" else torch.bfloat16


def _generate(method, shape, dtype, device, rng, **kw):
    if method == "randn":
        return torch.randn(shape, dtype=dtype, device=device, generator=rng)
    if method == "uniform":
        lo, hi = kw.get("low", 0.0), kw.get("high", 1.0)
        t = torch.rand(shape, dtype=dtype, device=device, generator=rng)
        return t * (hi - lo) + lo
    if method == "fixed":
        return torch.full(shape, kw.get("value", 10.0), dtype=dtype, device=device)
    if method == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if method == "custom":
        return kw["fn"](shape, dtype, device, rng)
    raise ValueError(f"Unknown method: {method}")


# ── 预定义 recipe ──

flash_attn_inputs = (
    InputSpec("flash_attn")
    # q — 按 layout_q 生成
    .tensor(
        "q",
        layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "TND": "total_s1,N1,D"},
        method="uniform",
        low=-5.0,
        high=5.0,
        range_key="q_range",
        when=lambda p: "q_range" in p,
    )
    .tensor(
        "q",
        layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "TND": "total_s1,N1,D"},
        when=lambda p: "q_range" not in p,
    )
    # k/v — 按 layout_kv 生成（PA 时生成 page 格式）
    .tensor(
        "k",
        layout={
            "BNSD": "B,N2,S2,D",
            "BSND": "B,S2,N2,D",
            "TND": "total_s2,N2,D",
            "PA_BBND": "num_blocks,block_size,N2,D",
            "PA_BNBD": "num_blocks,N2,block_size,D",
            "PA_NZ": "num_blocks,N2,D_nz_sub,block_size,nz_blk_elem",
        },
        method="uniform",
        low=-5.0,
        high=5.0,
        range_key="k_range",
        when=lambda p: "k_range" in p,
        layout_override="layout_kv",
    )
    .tensor(
        "k",
        layout={
            "BNSD": "B,N2,S2,D",
            "BSND": "B,S2,N2,D",
            "TND": "total_s2,N2,D",
            "PA_BBND": "num_blocks,block_size,N2,D",
            "PA_BNBD": "num_blocks,N2,block_size,D",
            "PA_NZ": "num_blocks,N2,D_nz_sub,block_size,nz_blk_elem",
        },
        when=lambda p: "k_range" not in p,
        layout_override="layout_kv",
    )
    .tensor(
        "v",
        layout={
            "BNSD": "B,N2,S2,_V",
            "BSND": "B,S2,N2,_V",
            "TND": "total_s2,N2,_V",
            "PA_BBND": "num_blocks,block_size,N2,_V",
            "PA_BNBD": "num_blocks,N2,block_size,_V",
            "PA_NZ": "num_blocks,N2,DV_nz_sub,block_size,nz_blk_elem",
        },
        method="uniform",
        low=-5.0,
        high=5.0,
        range_key="v_range",
        when=lambda p: "v_range" in p,
        layout_override="layout_kv",
    )
    .tensor(
        "v",
        layout={
            "BNSD": "B,N2,S2,_V",
            "BSND": "B,S2,N2,_V",
            "TND": "total_s2,N2,_V",
            "PA_BBND": "num_blocks,block_size,N2,_V",
            "PA_BNBD": "num_blocks,N2,block_size,_V",
            "PA_NZ": "num_blocks,N2,DV_nz_sub,block_size,nz_blk_elem",
        },
        when=lambda p: "v_range" not in p,
        layout_override="layout_kv",
    )
)
