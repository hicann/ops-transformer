# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashGDN chunk reconstruction and its Cube and Vector providers."""

from dataclasses import dataclass
from functools import lru_cache

import cannbotdsl as dsl
import torch
from cannbotdsl import Dim, TensorSpec, dtypes, make_layout
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.cube import set_fp32_mode
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc, tile_slice

TILE = 32


class _Cube:
    @kernel
    def compute(self, left, right_t, out, batch_size, m_tiles, n_tiles, k_tiles):
        set_fp32_mode()
        layout = make_layout((TILE, TILE))
        a_l1 = RingBuffer(MemLoc.L1, dtypes.float32, layout, depth=2, data_format="nz")
        b_l1 = RingBuffer(MemLoc.L1, dtypes.float32, layout, depth=2, data_format="nz")
        a_l0 = RingBuffer(MemLoc.L0A, dtypes.float32, layout, depth=2)
        b_l0 = RingBuffer(MemLoc.L0B, dtypes.float32, layout, depth=2)
        c_l0 = RingBuffer(MemLoc.L0C, dtypes.float32, layout, depth=2)
        for job in range(
            get_block_idx(), batch_size * m_tiles * n_tiles, get_block_num()
        ):
            batch = job // (m_tiles * n_tiles)
            row = job // n_tiles % m_tiles
            col = job % n_tiles
            acc = c_l0.next_write()
            for inner in range(k_tiles):
                mem_copy(
                    a_l1.next_write(),
                    tile_slice(left[batch, None, None], (TILE, TILE), (row, inner)),
                )
                mem_copy(
                    b_l1.next_write(),
                    tile_slice(right_t[batch, None, None], (TILE, TILE), (col, inner)),
                )
                mem_copy(a_l0.next_write(), a_l1.next_read())
                mem_copy(b_l0.next_write(), b_l1.next_read())
                acc = matmul(acc, a_l0.next_read(), b_l0.next_read(), init=inner == 0)
            mem_copy(
                tile_slice(out[batch, None, None], (TILE, TILE), (row, col)),
                c_l0.next_read(),
            )

    @host
    def run(self, left, right_t, out, blocks, batch_size, m_tiles, n_tiles, k_tiles):
        self.compute[blocks](left, right_t, out, batch_size, m_tiles, n_tiles, k_tiles)


def _compile_cube(device):
    batch = Dim("batch")
    m, n, k = (Dim(name, min=TILE, multiple_of=TILE) for name in ("m", "n", "k"))
    return dsl.compile(
        _Cube().run,
        TensorSpec((batch, m, k), dtypes.float32),
        TensorSpec((batch, n, k), dtypes.float32),
        TensorSpec((batch, m, n), dtypes.float32),
        *((dtypes.int64,) * 5),
    )


@lru_cache(maxsize=None)
def _get_cube_program(device):
    return _compile_cube(device)


def cube_matmul(left, right):
    batch, m, k = left.shape
    _, _, n = right.shape
    pm = (m + TILE - 1) // TILE * TILE
    pn = (n + TILE - 1) // TILE * TILE
    pk = (k + TILE - 1) // TILE * TILE
    right_t = right.transpose(-1, -2).contiguous()
    left = left.contiguous()
    if (pm, pk) != (m, k):
        left = torch.nn.functional.pad(left, (0, pk - k, 0, pm - m))
    if (pn, pk) != (n, k):
        right_t = torch.nn.functional.pad(right_t, (0, pk - k, 0, pn - n))
    out = torch.empty((batch, pm, pn), device=left.device, dtype=torch.float32)
    with torch.npu.device(left.device):
        program = _get_cube_program(left.device.index)
        stream = torch.npu.current_stream()
        for tensor in (left, right_t, out):
            tensor.record_stream(stream)
        m_tiles, n_tiles, k_tiles = pm // TILE, pn // TILE, pk // TILE
        blocks = min(32, batch * m_tiles * n_tiles)
        program(left, right_t, out, blocks, batch, m_tiles, n_tiles, k_tiles)
    return out[:, :m, :n].contiguous()


THREADS = 128
_OP = {"add": 0, "sub": 1, "mul": 2, "scale": 3, "exp": 4}
_SCAN = {"sum": 0, "prefix": 1, "suffix": 2}


@jit
def load(tensor, index):
    return simt.l2cache_load(tensor.ptr((index,)))


@jit
def store(tensor, index, value):
    simt.l2cache_store(tensor.ptr((index,)), value)


@jit
def sigmoid(x):
    z = simt.exp(-simt.abs(x))
    result = z / (1.0 + z)
    if x >= 0.0:
        result = 1.0 / (1.0 + z)
    return result


def _blocks(count):
    return min(32, (count + THREADS - 1) // THREADS)


def _launch(entry, tensors, scalars):
    device = tensors[0].device
    with torch.npu.device(device):
        program = _get_vector_program(device.index, entry)
        stream = torch.npu.current_stream()
        args = tuple(t.contiguous().view(-1) for t in tensors)
        for tensor in args:
            tensor.record_stream(stream)
        program(*args, *scalars)


def _empty(like, shape):
    return torch.empty(shape, dtype=torch.float32, device=like.device)


class _Epilogue:
    @kernel
    def map_kernel(
        self, a, b, out, blocks, count, rows, cols, ba, ma, na, bb, mb, nb, op, scale
    ):
        block = get_block_idx()
        with vf(mode="simt", thread=THREADS):
            start = dtypes.int64(block) * THREADS + dtypes.int64(simt.thread_idx()[0])
            for pos in range(start, count, blocks * THREADS):
                batch = pos // (rows * cols)
                row = pos // cols % rows
                col = pos % cols
                x = load(a, ((batch % ba) * ma + row % ma) * na + col % na)
                result = x * scale
                if op == 4:
                    result = simt.exp(x)
                elif op != 3:
                    y = load(b, ((batch % bb) * mb + row % mb) * nb + col % nb)
                    if op == 0:
                        result = x + y
                    elif op == 1:
                        result = x - y
                    else:
                        result = x * y
                store(out, pos, result)

    @host
    def map_run(
        self, a, b, out, blocks, count, rows, cols, ba, ma, na, bb, mb, nb, op, scale
    ):
        self.map_kernel[blocks](
            a, b, out, blocks, count, rows, cols, ba, ma, na, bb, mb, nb, op, scale
        )

    @kernel
    def reduce_kernel(self, a, out, blocks, rows, cols, scan):
        block = get_block_idx()
        with vf(mode="simt", thread=THREADS):
            start = dtypes.int64(block) * THREADS + dtypes.int64(simt.thread_idx()[0])
            for row in range(start, rows, blocks * THREADS):
                value = dtypes.float32(0.0)
                for col in range(cols):
                    offset = col
                    if scan == 2:
                        offset = cols - 1 - col
                    value = value + load(a, row * cols + offset)
                    if scan != 0:
                        store(out, row * cols + offset, value)
                if scan == 0:
                    store(out, row, value)

    @host
    def reduce_run(self, a, out, blocks, rows, cols, scan):
        self.reduce_kernel[blocks](a, out, blocks, rows, cols, scan)

    @kernel
    def gate_kernel(
        self,
        g,
        b,
        a,
        bias,
        step,
        beta,
        gate_derivative,
        beta_derivative,
        blocks,
        count,
        tokens,
        valid,
        heads,
        fused,
        has_bias,
        beta_sigmoid,
    ):
        block = get_block_idx()
        with vf(mode="simt", thread=THREADS):
            start = dtypes.int64(block) * THREADS + dtypes.int64(simt.thread_idx()[0])
            for i in range(start, count, blocks * THREADS):
                gate_value = dtypes.float32(0.0)
                beta_value = dtypes.float32(0.0)
                dg = dtypes.float32(0.0)
                db = dtypes.float32(0.0)
                if i % tokens < valid:
                    gate_value = load(g, i)
                    beta_value = load(b, i)
                    dg = dtypes.float32(1.0)
                    db = dtypes.float32(1.0)
                    if fused != 0:
                        head = i // tokens % heads
                        if has_bias != 0:
                            gate_value = gate_value + load(bias, head)
                        rate = -simt.exp(load(a, head))
                        dg = rate * sigmoid(gate_value)
                        positive = simt.max(gate_value, dtypes.float32(0.0))
                        gate_value = rate * (
                            positive + simt.log1p(simt.exp(-simt.abs(gate_value)))
                        )
                    if beta_sigmoid != 0:
                        beta_value = sigmoid(beta_value)
                        db = beta_value * (1.0 - beta_value)
                store(step, i, gate_value)
                store(beta, i, beta_value)
                store(gate_derivative, i, dg)
                store(beta_derivative, i, db)

    @host
    def gate_run(
        self,
        g,
        b,
        a,
        bias,
        step,
        beta,
        gate_derivative,
        beta_derivative,
        blocks,
        count,
        tokens,
        valid,
        heads,
        fused,
        has_bias,
        beta_sigmoid,
    ):
        self.gate_kernel[blocks](
            g,
            b,
            a,
            bias,
            step,
            beta,
            gate_derivative,
            beta_derivative,
            blocks,
            count,
            tokens,
            valid,
            heads,
            fused,
            has_bias,
            beta_sigmoid,
        )

    @kernel
    def causal_kernel(self, prefix, decay, strict, identity, blocks, count, n):
        block = get_block_idx()
        with vf(mode="simt", thread=THREADS):
            start = dtypes.int64(block) * THREADS + dtypes.int64(simt.thread_idx()[0])
            for i in range(start, count, blocks * THREADS):
                batch = i // (n * n)
                row = i // n % n
                col = i % n
                d = dtypes.float32(0.0)
                s = dtypes.float32(0.0)
                eye = dtypes.float32(0.0)
                if row > col:
                    d = simt.exp(
                        load(prefix, batch * n + row) - load(prefix, batch * n + col)
                    )
                    s = d
                if row == col:
                    d = dtypes.float32(1.0)
                    eye = d
                store(decay, i, d)
                store(strict, i, s)
                store(identity, i, eye)

    @host
    def causal_run(self, prefix, decay, strict, identity, blocks, count, n):
        self.causal_kernel[blocks](prefix, decay, strict, identity, blocks, count, n)


def _compile_vector(device, entry):
    instance = _Epilogue()
    kernels = {
        "map": (instance.map_run, 3, 11, (dtypes.float32,)),
        "reduce": (instance.reduce_run, 2, 4, ()),
        "gate": (instance.gate_run, 8, 8, ()),
        "causal": (instance.causal_run, 4, 3, ()),
    }
    kernel, tensors, integers, extra = kernels[entry]
    specs = [
        TensorSpec((Dim(f"tensor_{i}"),), dtypes.float32, stride=(1,))
        for i in range(tensors)
    ]
    return dsl.compile(kernel, *specs, *((dtypes.int64,) * integers), *extra)


def compile_staged_providers():
    """Compile all shape-dynamic providers required by the staged AOT export."""
    _compile_cube(None)
    for entry in ("map", "reduce", "gate", "causal"):
        _compile_vector(None, entry)


@lru_cache(maxsize=None)
def _get_vector_program(device, entry):
    return _compile_vector(device, entry)


def element(op, a, b=None, *, scale=1.0):
    b = a if b is None else b
    ashape, bshape = tuple(a.shape), tuple(b.shape)
    shape = torch.broadcast_shapes(ashape, bshape)
    out = _empty(a, shape)
    count = out.numel()
    _launch(
        "map",
        (a, b, out),
        (_blocks(count), count, *shape[1:], *ashape, *bshape, _OP[op], scale),
    )
    return out


def reduce_last(a, scan="sum"):
    batch, rows, cols = a.shape
    out = _empty(a, (batch, rows, 1) if scan == "sum" else a.shape)
    _launch(
        "reduce", (a, out), (_blocks(batch * rows), batch * rows, cols, _SCAN[scan])
    )
    return out


def gate_beta(g, beta, a, bias, valid, heads, beta_sigmoid):
    groups, tokens, _ = g.shape
    outputs = tuple(_empty(g, g.shape) for _ in range(4))
    placeholder = _empty(g, (heads,))
    args = (
        g,
        beta,
        placeholder if a is None else a,
        placeholder if bias is None else bias,
        *outputs,
    )
    count = groups * tokens
    _launch(
        "gate",
        args,
        (
            _blocks(count),
            count,
            tokens,
            valid,
            heads,
            int(a is not None),
            int(bias is not None),
            int(beta_sigmoid),
        ),
    )
    return outputs


def causal(prefix):
    groups, size, _ = prefix.shape
    outputs = tuple(_empty(prefix, (groups, size, size)) for _ in range(3))
    count = groups * size * size
    _launch("causal", (prefix, *outputs), (_blocks(count), count, size))
    return outputs


#: L2 归一化 eps，与 flash_kda_grad 保持一致。
NORM_EPS = 1e-6


def normalize_qk(q, k, enabled, dtype=None):
    """L2-normalize q/k the way the KDA kernel does.

    ``rstd = rsqrt(sum(x^2) + NORM_EPS)`` is kept in FP32; the normalized value
    is rounded through BF16 (the kernel's storage) and cast back to the compute
    dtype. Returns ``(q_used, k_used, q_rstd, k_rstd)``; with ``enabled=False``
    the inputs pass through untouched and the inverse norms are ones.
    """
    if not enabled:
        ones = torch.ones(*q.shape[:-1], 1, dtype=q.dtype, device=q.device)
        return q, k, ones, torch.ones(*k.shape[:-1], 1, dtype=k.dtype, device=k.device)
    compute = q.dtype if dtype is None else dtype

    def _one(tensor):
        rstd = torch.rsqrt(tensor.square().sum(-1, keepdim=True) + NORM_EPS)
        low = (tensor * rstd).to(torch.bfloat16)
        return low.to(compute), rstd.to(torch.float32)

    q_used, q_rstd = _one(q)
    k_used, k_rstd = _one(k)
    return q_used, k_used, q_rstd, k_rstd


def restore_normalized_grad(grad, value, rstd):
    """Chain rule through ``y = x / ||x||`` for one of q/k."""
    dot = (value * grad).sum(-1, keepdim=True)
    return (grad - value * dot) * rstd


CHUNK_SIZE = 64


def add(a, b):
    return element("add", a, b)


def sub(a, b):
    return element("sub", a, b)


def mul(a, b):
    return element("mul", a, b)


def scale(a, value):
    return element("scale", a, scale=value)


def exp(a):
    return element("exp", a)


def transpose(a):
    return a.transpose(-1, -2)


def pack(a, chunks):
    groups, length, width = a.shape
    if length != chunks * CHUNK_SIZE:
        a = torch.nn.functional.pad(a, (0, 0, 0, chunks * CHUNK_SIZE - length))
    return (
        a.reshape(groups, chunks, CHUNK_SIZE, width)
        .transpose(0, 1)
        .flatten(0, 1)
        .contiguous()
    )


def unpack(a, groups, length):
    return (
        a.reshape(-1, groups, CHUNK_SIZE, a.size(-1))
        .transpose(0, 1)
        .flatten(1, 2)[:, :length]
    )


@dataclass(frozen=True)
class ChunkCache:
    beta: torch.Tensor
    beta_derivative: torch.Tensor
    gate_step: torch.Tensor
    gate_derivative: torch.Tensor
    e: torch.Tensor
    r: torch.Tensor
    c: torch.Tensor
    decay: torch.Tensor
    strict_decay: torch.Tensor
    qk: torch.Tensor
    kk: torch.Tensor
    aqk: torch.Tensor
    inverse: torch.Tensor
    h_in: torch.Tensor
    corrected: torch.Tensor
    #: Q/K 的 L2 逆范数（FP32，[rows*chunks, C, 1]）；未启用归一化时全 1。
    q_rstd: torch.Tensor
    k_rstd: torch.Tensor
    #: 前向处理后的 Q/K（缓存路径从 ABI 的 ``qg``/``kg`` 直接带入，重算路径为归一化结果）。
    qg: torch.Tensor = None
    kg: torch.Tensor = None


def load_cache(
    qg,
    kg,
    g,
    beta,
    a,
    bias,
    hv,
    activate,
    aqk,
    inverse,
    corrected,
    h_in,
    g_cumsum,
    q_rstd,
    k_rstd,
    beta_inverse_factor=1.0,
):
    """Rebuild ``ChunkCache`` from the cache-ABI tensors of ``flash_kda_grad``.

    Only the expensive quantities are taken from the cache (``Aqk``/``Akk``/``v_new``/
    ``chunk_states``/``qg``/``kg``/``g_cumsum``/``q_rstd``/``k_rstd``/``beta_sigmoid``);
    the derived fields are recomputed with the same formulas ``build_cache`` uses.
    """
    groups, tokens, _ = qg.shape
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded = chunks * CHUNK_SIZE
    steps, betas, gate_grad, beta_grad = gate_beta(
        torch.nn.functional.pad(g, (0, 0, 0, padded - tokens)),
        torch.nn.functional.pad(beta, (0, 0, 0, padded - tokens)),
        a,
        bias,
        tokens,
        hv,
        activate,
    )
    if beta_inverse_factor != 1.0:
        beta_grad = betas * (1.0 - betas * beta_inverse_factor)
    steps, betas, gate_grad, beta_grad = (
        pack(t, chunks) for t in (steps, betas, gate_grad, beta_grad)
    )
    # g_cumsum 就是 chunk 内的 prefix，直接用来还原 e/r/c/decay，避免再累加一次
    prefix = g_cumsum
    decay, strict_decay, _identity = causal(prefix)
    e = exp(prefix)
    r = exp(sub(prefix[:, -1:], prefix))
    c = e[:, -1:].contiguous()
    qg, kg = pack(qg, chunks), pack(kg, chunks)
    qk, kk = cube_matmul(qg, transpose(kg)), cube_matmul(kg, transpose(kg))
    return ChunkCache(
        betas,
        beta_grad,
        steps,
        gate_grad,
        e,
        r,
        c,
        decay,
        strict_decay,
        qk,
        kk,
        aqk,
        inverse,
        h_in,
        corrected,
        q_rstd,
        k_rstd,
        qg,
        kg,
    )


def build_cache(
    q,
    k,
    v,
    g,
    beta,
    h0,
    a,
    bias,
    hv,
    scale_value,
    activate,
    q_rstd=None,
    k_rstd=None,
    beta_inverse_factor=1.0,
):
    groups, tokens, _ = q.shape
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded = chunks * CHUNK_SIZE
    steps, betas, gate_grad, beta_grad = gate_beta(
        torch.nn.functional.pad(g, (0, 0, 0, padded - tokens)),
        torch.nn.functional.pad(beta, (0, 0, 0, padded - tokens)),
        a,
        bias,
        tokens,
        hv,
        activate,
    )
    if beta_inverse_factor != 1.0:
        # allow_neg_eigval：beta 允许落在 (0,2)，dβ/draw = β(1 − 0.5β)
        beta_grad = betas * (1.0 - betas * beta_inverse_factor)
    steps, betas, gate_grad, beta_grad = (
        pack(t, chunks) for t in (steps, betas, gate_grad, beta_grad)
    )
    q, k, v = (pack(t, chunks) for t in (q, k, v))
    prefix = transpose(reduce_last(transpose(steps), "prefix"))
    decay, strict_decay, identity = causal(prefix)
    e = exp(prefix)
    r = exp(sub(prefix[:, -1:], prefix))
    c = e[:, -1:].contiguous()
    qk, kk = cube_matmul(q, transpose(k)), cube_matmul(k, transpose(k))
    aqk = mul(scale(qk, scale_value), decay)

    # L is strict lower triangular, so the series for (I + L)^-1 terminates.
    # Squaring -L doubles the number of terms included at each step.
    power = scale(mul(betas, mul(kk, strict_decay)), -1.0)
    inverse = add(identity, power)
    terms = 2
    while terms < CHUNK_SIZE:
        power = cube_matmul(power, power)
        inverse = add(inverse, cube_matmul(power, inverse))
        terms *= 2

    state = h0
    h_in, corrected = [], []
    for start in range(0, chunks * groups, groups):
        sl = slice(start, start + groups)
        h_in.append(state)
        k_chunk = k[sl]
        residual = sub(v[sl], mul(e[sl], cube_matmul(k_chunk, state)))
        x = cube_matmul(inverse[sl], mul(betas[sl], residual))
        corrected.append(x)
        if start + groups < chunks * groups:
            state = add(
                mul(c[sl], state), cube_matmul(transpose(k_chunk), mul(r[sl], x))
            )

    if q_rstd is None or k_rstd is None:
        ones = torch.ones_like(steps)
        q_rstd = ones if q_rstd is None else q_rstd
        k_rstd = ones if k_rstd is None else k_rstd
    elif q_rstd.shape != steps.shape:
        # 入口给的是 [rows, tokens, 1]，按与 gate 相同的打包方式对齐
        q_rstd = pack(q_rstd, chunks)
        k_rstd = pack(k_rstd, chunks)
    return ChunkCache(
        betas,
        beta_grad,
        steps,
        gate_grad,
        e,
        r,
        c,
        decay,
        strict_decay,
        qk,
        kk,
        aqk,
        inverse,
        torch.cat(h_in),
        torch.cat(corrected),
        q_rstd,
        k_rstd,
        q,
        k,
    )


def packed_to_planes(tensor, rows, tokens):
    """``[rows*chunks, C, w]`` 的打包张量还原成 ``[rows, tokens, w]``。"""
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    return (
        tensor.reshape(chunks, rows, CHUNK_SIZE, tensor.size(-1))
        .transpose(0, 1)
        .flatten(1, 2)[:, :tokens]
    )
