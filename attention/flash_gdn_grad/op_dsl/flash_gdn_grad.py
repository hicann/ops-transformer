# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent CANNBotDSL FlashGDN backward entry points."""

import math
from dataclasses import dataclass, replace
from functools import wraps

import torch
from cannbotdsl import aot

from .flash_gdn_grad_metadata import FlashGDNMetadata, resolve_metadata
from .flash_gdn_grad_persistent import flash_gdn_grad_npu as run_persistent
from .flash_gdn_grad_recompute import (
    CHUNK_SIZE,
    NORM_EPS,
    ChunkCache,
    compile_staged_providers,
    add,
    build_cache,
    mul,
    scale,
    load_cache,
    normalize_qk,
    packed_to_planes,
    reduce_last,
    restore_normalized_grad,
)
from .flash_gdn_grad_staged import run_staged


def _no_grad(function):
    """Defer Torch's grad context until execution so OpKit can import the module."""

    @wraps(function)
    def run(*args, **kwargs):
        with torch.no_grad():
            return function(*args, **kwargs)

    return run


# --- Forward-cache tensor ABI ---------------------------------------------
#: ``(ABI 参数名, ChunkCache 字段名)``：与 flash_kda_grad 同名的缓存项沿用 KDA 的名字
#: （``Aqk``/``Akk``/``v_new``/``chunk_states``/``beta_sigmoid``），GDN 独有的项
#: 沿用内核字段名。
CACHE_ABI_FIELDS = (
    ("Aqk", "aqk"),
    ("Akk", "inverse"),
    ("w", None),  # WY Key：与 KDA 保持一致，反向不读取
    ("u", None),  # WY Value：与 KDA 保持一致，反向不读取
    ("v_new", "corrected"),
    ("qg", "qg"),
    ("kg", "kg"),
    ("g_cumsum", "gate_step"),  # chunk 内累计 log gate
    ("q_rstd", "q_rstd"),
    ("k_rstd", "k_rstd"),
    ("beta_sigmoid", "beta"),
    ("chunk_states", "h_in"),
)


#: ABI 参数名，按 ``CACHE_ABI_FIELDS`` 的顺序。
CACHE_ABI_NAMES = tuple(name for name, _ in CACHE_ABI_FIELDS)


def _cache_planes(plan, hv):
    """Per group ``(rows, chunks)``: plane rows and chunk count of the cache."""
    layout = []
    for length, ids in plan.groups:
        chunks = 0 if length == 0 else (length + 63) // 64
        layout.append((len(ids) * hv, chunks))
    return tuple(layout)


def _pack_forward_cache(groups):
    """内部工具：把 ``ChunkCache`` 列表摊平成缓存 ABI 张量。

    反向算子本身**不生产**前向缓存（那是前向算子的职责，与 ``flash_kda_grad`` 一致），
    这里仅供单测/框架在构造缓存张量时复用。
    """
    groups = [g for g in groups if g is not None]
    if not groups:
        raise ValueError("forward cache is empty")
    packed = {}
    for abi, field in CACHE_ABI_FIELDS:
        if field is None:
            # ABI 保留但反向不读的槽位（w/u）：用零张量占位
            ref = groups[0].corrected
            packed[abi] = torch.zeros(
                (sum(int(g.corrected.size(0)) for g in groups), 1),
                dtype=torch.float32,
                device=ref.device,
            )
            continue
        parts = [getattr(g, field) for g in groups]
        if field == "gate_step":
            parts = [torch.cumsum(t, dim=1) for t in parts]  # ABI 存 cumsum
        packed[abi] = torch.cat(parts)
    return packed


def _slice_cache_tensors(plan, hv, cache, require_saved=False):
    """按调度把缓存 ABI 张量切成每组片段。

    ``Aqk``/``Akk`` 两种模式都必须给出（与 flash_kda_grad 一致，反向不重算）；
    其余 8 项只在 ``disable_recompute=True`` 时必需（``w``/``u`` 反向不读）。
    """
    # Aqk/Akk 可选：不传就由反向重算；保存缓存模式下要给出其余 10 项
    saved = tuple(name for name in CACHE_ABI_NAMES if name not in ("Aqk", "Akk"))
    must = saved if require_saved else ()
    missing = [name for name in must if cache.get(name) is None]
    if missing:
        raise ValueError(f"the cache ABI is missing {missing}")
    anchor = (
        "Aqk"
        if cache.get("Aqk") is not None
        else ("Akk" if cache.get("Akk") is not None else None)
    )
    rows_total = None if anchor is None else cache[anchor].size(0)
    for name in must:
        if rows_total is not None and cache[name].size(0) != rows_total:
            raise ValueError(f"cache tensor {name} does not match the cache row count")
    groups, offset = [], 0
    for rows, chunks in _cache_planes(plan, hv):
        if chunks == 0:
            groups.append(None)
            continue
        span = rows * chunks
        sl = slice(offset, offset + span)
        groups.append(
            {
                name: (
                    None if cache.get(name) is None else cache[name][sl].contiguous()
                )
                for name in CACHE_ABI_NAMES
            }
        )
        offset += span
    if rows_total is not None and offset != rows_total:
        raise ValueError("cache tensors do not match this schedule")
    return tuple(groups)


def _check_gate_flags(use_gate_in_kernel, allow_neg_eigval, use_beta_sigmoid_in_kernel):
    """与 flash_kda_grad 相同的开关约束。"""
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires activated beta")


def _storage_shapes(q, k, v, g, beta, metadata):
    if q.shape != k.shape or v.ndim != q.ndim:
        raise ValueError("expected matching q/k and a V tensor with the same rank")
    axis = 2 if metadata.layout == "BSND" else 1
    heads, hv = q.size(axis), v.size(axis)
    if min(heads, hv, q.size(-1), v.size(-1)) < 1 or hv % heads:
        raise ValueError("dimensions must be positive and HV a multiple of H")
    if (
        q.shape[:axis] + q.shape[axis + 1 : -1]
        != v.shape[:axis] + v.shape[axis + 1 : -1]
    ):
        raise ValueError("Q and V must agree on batch and sequence dimensions")
    if g.shape != v.shape[:-1] or beta.shape != g.shape:
        raise ValueError("expected one gate and beta per value head and token")
    return heads, hv, (len(metadata.lengths), hv, q.size(-1), v.size(-1))


def _validate(q, k, v, g, beta, h0, a, bias, metadata, scale, do=None, dht=None):
    heads, hv, state_shape = _storage_shapes(q, k, v, g, beta, metadata)
    if h0 is not None and h0.shape != state_shape:
        raise ValueError("initial_state must be [B,HV,K,V]")
    if do is not None and (do.shape != v.shape or dht.shape != state_shape):
        raise ValueError("expected do shaped like V and dht [B,HV,K,V]")
    if a is not None and a.shape != (hv,):
        raise ValueError("A_log must be [HV]")
    if bias is not None and (a is None or bias.shape != (hv,)):
        raise ValueError("dt_bias must be [HV] and requires A_log")
    for tensor in (q, k, v, g, beta, h0, a, bias, do, dht):
        if tensor is not None and (q.device.type != "npu" or tensor.device != q.device):
            raise ValueError("all tensors must live on the same NPU")
    # dtype 契约与 flash_kda_grad 一致：
    #   q/k/v/do = BF16；g/beta = BF16 或 FP32；状态与 gate 参数 = FP32
    for name, tensor in (("q", q), ("k", k), ("v", v), ("do", do)):
        if tensor is not None and tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BFLOAT16")
    for name, tensor in (("g", g), ("beta", beta)):
        if tensor.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name} must be BFLOAT16 or FLOAT32")
    for name, tensor in (
        ("initial_state", h0),
        ("dht", dht),
        ("A_log", a),
        ("dt_bias", bias),
    ):
        if tensor is not None and tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be FLOAT32")
    scale = q.size(-1) ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return heads, hv, state_shape, scale


def _to_token_major(tensor, layout):
    return tensor.transpose(1, 2) if layout == "BNSD" else tensor


def _gather(tensor, metadata, ids, length):
    if metadata.layout == "TND":
        return torch.stack(
            tuple(tensor[metadata.starts[i] : metadata.starts[i] + length] for i in ids)
        )
    return torch.stack(tuple(tensor[i, :length] for i in ids))


def _planes(tensor, metadata, ids, length, heads=None):
    result = _gather(tensor, metadata, ids, length).float().transpose(1, 2)
    if result.ndim == 3:
        result = result.unsqueeze(-1)
    if heads is not None and heads != result.size(1):
        result = result.repeat_interleave(heads // result.size(1), dim=1)
    return result.flatten(0, 1).contiguous()


def _state(tensor, ids):
    return torch.stack(tuple(tensor[i] for i in ids)).float().flatten(0, 1)


def _zeros(ref, shape):
    return torch.zeros(shape, dtype=torch.float32, device=ref.device)


def prepare_forward_akk(
    q, k, v, g, beta, state, a, bias, plan, layout, scale, fused, l2norm
):
    """Build the packed forward Akk tensor for tests and framework adapters.

    ``plan`` describes sequence groups for ``layout``. Q/K normalization and
    beta activation follow the same options as the backward operator.
    """
    hv = v.shape[1] if layout == "BNSD" else v.shape[-2]
    qt, kt, vt, gt, bt = (_to_token_major(t, layout) for t in (q, k, v, g, beta))
    groups = []
    for length, ids in plan.groups:
        if length == 0:
            groups.append(None)
            continue
        qh, kh = (_planes(t, plan, ids, length, hv) for t in (qt, kt))
        qh, kh, q_rstd, k_rstd = normalize_qk(qh, kh, l2norm)
        vh, gh, bh = (_planes(t, plan, ids, length) for t in (vt, gt, bt))
        h0 = (
            _state(state, ids)
            if state is not None
            else _zeros(qh, (len(ids) * hv, qh.size(-1), vh.size(-1)))
        )
        groups.append(
            build_cache(
                qh,
                kh,
                vh,
                gh,
                bh,
                h0,
                a,
                bias,
                hv,
                qh.size(-1) ** -0.5 if scale is None else float(scale),
                fused,
                q_rstd=q_rstd,
                k_rstd=k_rstd,
            )
        )
    if all(item is None for item in groups):
        return q.new_empty((0, 64, 64), dtype=torch.float32)
    return _pack_forward_cache(tuple(groups))["Akk"]


def _check_cache_tensors(cache, enabled):
    """Validate the tensor-form cache: completeness and mutual consistency."""
    if not enabled:
        return
    missing = [name for name in CACHE_ABI_NAMES if cache.get(name) is None]
    if missing:
        raise ValueError(
            f"the cache ABI needs all {len(CACHE_ABI_NAMES)} tensors; missing {missing}"
        )
    rows = cache[CACHE_ABI_NAMES[0]].size(0)
    for name in CACHE_ABI_NAMES:
        if cache[name].size(0) != rows:
            raise ValueError(f"cache tensor {name} does not match the cache row count")


def _head_grad(tensor, batches, heads, hv):
    _, length, width = tensor.shape
    ratio = hv // heads
    if ratio == 1:
        return tensor.reshape(batches, heads, length, width)
    values = tensor.reshape(batches, heads, ratio, length, width).permute(0, 1, 3, 4, 2)
    return reduce_last(values.reshape(batches * heads, length * width, ratio)).reshape(
        batches, heads, length, width
    )


def _parameter_grad(tensor, batches, hv, length):
    values = (
        tensor.reshape(batches, hv, length)
        .transpose(0, 1)
        .reshape(1, hv, batches * length)
    )
    return reduce_last(values)


def _restore(rows, metadata, like):
    if metadata.layout == "TND":
        return torch.cat(rows).to(like.dtype)
    full_length = metadata.storage[1]
    padded = tuple(
        torch.nn.functional.pad(t, (0, 0) * (t.ndim - 1) + (0, full_length - t.size(0)))
        for t in rows
    )
    out = torch.stack(padded)
    return _to_token_major(out, metadata.layout).contiguous().to(like.dtype)


@_no_grad
def flash_gdn_grad_npu(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    Aqk=None,
    Akk=None,
    metadata=None,
    *,
    initial_state=None,
    chunk_states=None,
    w=None,
    u=None,
    v_new=None,
    qg=None,
    kg=None,
    g_cumsum=None,
    q_rstd=None,
    k_rstd=None,
    beta_sigmoid=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    strategy="staged",
    layout="TND",
):
    """Return dq, dk, dv, dh0, dg, dbeta, dA_log, dBias for chunk GDN."""
    if not isinstance(do, torch.Tensor) or not isinstance(dht, torch.Tensor):
        raise ValueError(
            "do and dht are required tensors; use zeros for absent gradients"
        )
    if Akk is None:
        raise ValueError("Akk is required (only Aqk may be omitted)")
    plan = resolve_metadata(q, layout, metadata, cu_seqlens, seqused)
    heads, hv, state_shape, scale = _validate(
        q, k, v, g, beta, initial_state, A_log, dt_bias, plan, scale, do, dht
    )
    if strategy not in ("staged", "persistent"):
        raise ValueError("strategy must be staged or persistent")
    cache_tensors = dict(
        Aqk=Aqk,
        Akk=Akk,
        w=w,
        u=u,
        v_new=v_new,
        qg=qg,
        kg=kg,
        g_cumsum=g_cumsum,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_sigmoid,
        chunk_states=chunk_states,
    )
    saved = (v_new, qg, kg, g_cumsum, q_rstd, k_rstd, beta_sigmoid, chunk_states, w, u)
    if any(item is not None for item in saved):
        # 给出了保存项 → 走缓存路径（Aqk/Akk 本来就必传）
        disable_recompute = True
    _check_cache_tensors(cache_tensors, disable_recompute)
    if disable_recompute and strategy != "staged":
        raise ValueError("saved-cache backward requires staged strategy")
    cache_groups = _slice_cache_tensors(
        plan, hv, cache_tensors, require_saved=disable_recompute
    )
    q, k, v, g, beta, do = (_to_token_major(t, layout) for t in (q, k, v, g, beta, do))
    if (
        strategy == "persistent"
        and layout != "TND"
        and plan.lengths == (plan.storage[1],) * plan.storage[0]
    ):
        dq, dk, dv, dh, dg, db, da, dbias = run_persistent(
            q,
            k,
            v,
            g,
            beta,
            do,
            dht,
            initial_state,
            A_log,
            dt_bias,
            scale=scale,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        )
        return (
            *(_to_token_major(t, layout).contiguous() for t in (dq, dk, dv)),
            dh,
            _to_token_major(dg, layout).contiguous(),
            _to_token_major(db, layout).contiguous(),
            da,
            dbias,
        )
    h0 = _zeros(q, state_shape) if initial_state is None else initial_state
    _check_gate_flags(use_gate_in_kernel, allow_neg_eigval, use_beta_sigmoid_in_kernel)
    beta_inverse_factor = 0.5 if allow_neg_eigval else 1.0
    if not use_gate_in_kernel:
        # 非融合：g 已经是 log gate，A_log/dt_bias 不参与计算也不产生梯度
        a, bias = None, None
        A_log, dt_bias = None, None
    else:
        a = None if A_log is None else A_log.float().contiguous()
        bias = None if dt_bias is None else dt_bias.float().contiguous()
    out = [[None] * state_shape[0] for _ in range(6)]
    da = _zeros(q, (1, hv, 1))
    dbias = _zeros(q, (1, hv, 1))
    for group, (length, ids) in enumerate(plan.groups):
        batches = len(ids)
        if length == 0:
            for i in ids:
                out[0][i] = _zeros(q, (0, heads, q.size(-1)))
                out[1][i] = out[0][i]
                out[2][i] = _zeros(q, (0, hv, v.size(-1)))
                out[3][i] = _zeros(q, (0, hv))
                out[4][i] = out[3][i]
                out[5][i] = dht[i].float()
            continue
        if strategy == "persistent":
            inputs = tuple(
                _gather(t, plan, ids, length) for t in (q, k, v, g, beta, do)
            )
            state = _state(h0, ids).reshape(batches, *state_shape[1:])
            final_gradient = _state(dht, ids).reshape_as(state)
            dq, dk, dv, dh, dg, db, d_a, d_bias = run_persistent(
                *inputs,
                final_gradient,
                state,
                a,
                bias,
                scale=scale,
                use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            )
            if d_a is not None:
                da = add(da, d_a.float().reshape(1, hv, 1))
            if d_bias is not None:
                dbias = add(dbias, d_bias.float().reshape(1, hv, 1))
            for pos, i in enumerate(ids):
                for column, value in zip(out, (dq, dk, dv, dg, db, dh)):
                    column[i] = value[pos]
            continue
        qh, kh = (_planes(t, plan, ids, length, hv) for t in (q, k))
        vh, doh = (_planes(t, plan, ids, length) for t in (v, do))
        part = cache_groups[group]
        rows = len(ids) * hv
        gh, bh = (_planes(t, plan, ids, length) for t in (g, beta))
        if disable_recompute:
            # ABI 里的 qg/kg 就是前向处理后的 Q/K，直接使用，不再二次归一化
            qh = packed_to_planes(part["qg"], rows, length)
            kh = packed_to_planes(part["kg"], rows, length)
            q_rstd = packed_to_planes(part["q_rstd"], rows, length)
            k_rstd = packed_to_planes(part["k_rstd"], rows, length)
            cache = load_cache(
                qh,
                kh,
                gh,
                bh,
                a,
                bias,
                hv,
                use_beta_sigmoid_in_kernel,
                part["Aqk"],
                part["Akk"],
                part["v_new"],
                part["chunk_states"],
                # g_cumsum 保持打包形状 [rows*chunks, C, 1]，load_cache 内部按块使用
                part["g_cumsum"],
                q_rstd,
                k_rstd,
                beta_inverse_factor=beta_inverse_factor,
            )
            if part["Aqk"] is None:  # Aqk 可省：用 qk·decay 现算
                cache = replace(cache, aqk=mul(scale(cache.qk, scale), cache.decay))
        else:
            qh, kh, q_rstd, k_rstd = normalize_qk(qh, kh, use_qk_l2norm_in_kernel)
            cache = build_cache(
                qh,
                kh,
                vh,
                gh,
                bh,
                _state(h0, ids),
                a,
                bias,
                hv,
                scale,
                use_beta_sigmoid_in_kernel,
                q_rstd=q_rstd,
                k_rstd=k_rstd,
                beta_inverse_factor=beta_inverse_factor,
            )
            # Akk 必传、直接用；Aqk 可选，没给就用核内重算结果
            cache = replace(
                cache,
                inverse=part["Akk"],
                aqk=part["Aqk"] if part["Aqk"] is not None else cache.aqk,
            )
        dq, dk, dv, dg, db, d_a, dh = run_staged(
            qh, kh, vh, doh, _state(dht, ids), cache, scale
        )
        if use_qk_l2norm_in_kernel:
            # dq/dk 是对归一化后 q/k 的梯度，按 KDA 的链式法则还原到原始输入
            dq = restore_normalized_grad(dq, qh, q_rstd)
            dk = restore_normalized_grad(dk, kh, k_rstd)
        if a is not None:
            da = add(da, _parameter_grad(d_a, batches, hv, length))
        if bias is not None:
            dbias = add(dbias, _parameter_grad(dg, batches, hv, length))
        group_out = (
            _head_grad(dq, batches, heads, hv).transpose(1, 2),
            _head_grad(dk, batches, heads, hv).transpose(1, 2),
            dv.reshape(batches, hv, length, -1).transpose(1, 2),
            dg.reshape(batches, hv, length).transpose(1, 2),
            db.reshape(batches, hv, length).transpose(1, 2),
            dh.reshape(batches, *state_shape[1:]),
        )
        for pos, i in enumerate(ids):
            for column, value in zip(out, group_out):
                column[i] = value[pos]
    dq, dk, dv, dg, db = (
        _restore(rows, plan, like) for rows, like in zip(out[:5], (q, k, v, g, beta))
    )
    return (
        dq,
        dk,
        dv,
        torch.stack(out[5]).contiguous(),
        dg,
        db,
        None if A_log is None else da.reshape(hv).to(A_log.dtype),
        None if dt_bias is None else dbias.reshape(hv).to(dt_bias.dtype),
    )


@aot.export("flash_gdn_grad")
def export_flash_gdn_grad():
    """Publish the shape-dynamic Cube and Vector providers used by staged GDN."""

    compile_staged_providers()
