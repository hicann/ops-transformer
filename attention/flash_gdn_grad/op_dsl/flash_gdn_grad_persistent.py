# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone FlashGDN backward using CANNBotDSL SIMT on the Ascend AIV."""

import math
from functools import lru_cache

import cannbotdsl as dsl
import torch
from cannbotdsl import TensorSpec, dtypes
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import simt
from cannbotdsl.ops.arch import get_block_idx


CHUNK_SIZE = 64
THREADS = 128


@jit
def _read(tensor, index):
    return simt.l2cache_load(tensor.ptr((index,)))


@jit
def _write(tensor, index, value):
    simt.l2cache_store(tensor.ptr((index,)), value)


@jit
def _add(tensor, index, value):
    simt.atomic_add(tensor.ptr((index,)), value)


@jit
def _sigmoid(x):
    z = simt.exp(-simt.abs(x))
    value = z / (1.0 + z)
    if x >= 0.0:
        value = 1.0 / (1.0 + z)
    return value


@jit
def _gate(g, a_log, bias, token, head, fused, has_bias):
    raw = _read(g, token)
    shifted = raw
    rate = dtypes.float32(0.0)
    step = raw
    if fused:
        if has_bias:
            shifted = raw + _read(bias, head)
        rate = -simt.exp(_read(a_log, head))
        magnitude = simt.exp(-simt.abs(shifted))
        softplus = simt.log(1.0 + magnitude)
        # Preserve small positive softplus values when 1 + magnitude rounds.
        if magnitude < 0.0001:
            softplus = magnitude
        if shifted > 0.0:
            softplus = softplus + shifted
        step = rate * softplus
    return step, shifted, rate


@kernel
class _FlashGDNBackward:
    def __init__(self, config):
        self.config = config

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        h0,
        a_log,
        bias,
        work,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dh,
        da,
        dbias,
    ):
        (
            batch,
            tokens,
            heads,
            hv,
            key_dim,
            value_dim,
            scale,
            fused,
            has_bias,
            activate,
            blocks,
        ) = self.config
        chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
        matrix = key_dim * value_dim
        block = get_block_idx()
        with vf(mode="simt", thread=THREADS):
            first = dtypes.int64(block) * THREADS + dtypes.int64(simt.thread_idx()[0])
            for lane in range(first, batch * hv * value_dim, THREADS * blocks):
                owner = lane // value_dim
                col = lane % value_dim
                head = owner % hv
                batch_id = owner // hv
                q_head = head // (hv // heads)
                h_offset = owner * matrix + col
                checkpoint = owner * (chunks + CHUNK_SIZE + 1) * matrix + col
                scratch = checkpoint + chunks * matrix

                for row in range(key_dim):
                    off = row * value_dim
                    _write(work, checkpoint + off, _read(h0, h_offset + off))
                    _write(dh, h_offset + off, _read(dht, h_offset + off))

                # Save chunk boundaries, then reconstruct one chunk at a time
                # in reverse. Each lane owns one value column of the state.
                for phase in range(2):
                    count = chunks - 1
                    if phase == 1:
                        count = chunks
                    for index in range(count):
                        chunk = index
                        if phase == 1:
                            chunk = chunks - 1 - index
                        start = chunk * CHUNK_SIZE
                        length = tokens - start
                        if length > CHUNK_SIZE:
                            length = CHUNK_SIZE
                        for row in range(key_dim):
                            off = row * value_dim
                            _write(
                                work,
                                scratch + off,
                                _read(work, checkpoint + chunk * matrix + off),
                            )

                        for pos in range(length):
                            token = batch_id * tokens + start + pos
                            scalar = token * hv + head
                            qk = (token * heads + q_head) * key_dim
                            val = scalar * value_dim + col
                            prev = scratch + pos * matrix
                            step, shifted, rate = _gate(
                                g, a_log, bias, scalar, head, fused, has_bias
                            )
                            decay = simt.exp(step)
                            b = _read(beta, scalar)
                            if activate:
                                b = _sigmoid(b)
                            prediction = dtypes.float32(0.0)
                            for row in range(key_dim):
                                prediction = prediction + _read(k, qk + row) * _read(
                                    work, prev + row * value_dim
                                )
                            correction = b * (_read(v, val) - decay * prediction)
                            for row in range(key_dim):
                                off = row * value_dim
                                state = (
                                    decay * _read(work, prev + off)
                                    + _read(k, qk + row) * correction
                                )
                                _write(work, prev + matrix + off, state)

                        if phase == 0:
                            for row in range(key_dim):
                                off = row * value_dim
                                _write(
                                    work,
                                    checkpoint + (chunk + 1) * matrix + off,
                                    _read(work, scratch + length * matrix + off),
                                )
                        else:
                            for reverse in range(length):
                                pos = length - 1 - reverse
                                token = batch_id * tokens + start + pos
                                scalar = token * hv + head
                                qk = (token * heads + q_head) * key_dim
                                val = scalar * value_dim + col
                                prev = scratch + pos * matrix
                                step, shifted, rate = _gate(
                                    g, a_log, bias, scalar, head, fused, has_bias
                                )
                                decay = simt.exp(step)
                                b = _read(beta, scalar)
                                if activate:
                                    b = _sigmoid(b)
                                upstream = _read(do, val) * scale
                                prediction = dtypes.float32(0.0)
                                d_correction = dtypes.float32(0.0)
                                for row in range(key_dim):
                                    off = row * value_dim
                                    key = _read(k, qk + row)
                                    state_grad = _read(
                                        dh, h_offset + off
                                    ) + upstream * _read(q, qk + row)
                                    prediction = prediction + key * _read(
                                        work, prev + off
                                    )
                                    d_correction = d_correction + key * state_grad
                                residual = _read(v, val) - decay * prediction
                                correction = b * residual
                                d_residual = b * d_correction
                                _write(dv, val, d_residual)
                                gate_grad = dtypes.float32(0.0)
                                for row in range(key_dim):
                                    off = row * value_dim
                                    state = _read(work, prev + off)
                                    state_grad = _read(
                                        dh, h_offset + off
                                    ) + upstream * _read(q, qk + row)
                                    prev_grad = decay * (
                                        state_grad - _read(k, qk + row) * d_residual
                                    )
                                    gate_grad = gate_grad + prev_grad * state
                                    _write(dh, h_offset + off, prev_grad)
                                    _add(
                                        dq,
                                        qk + row,
                                        upstream * _read(work, prev + matrix + off),
                                    )
                                    _add(
                                        dk,
                                        qk + row,
                                        correction * state_grad
                                        - decay * state * d_residual,
                                    )
                                b_grad = residual * d_correction
                                if activate:
                                    b_grad = b_grad * b * (1.0 - b)
                                _add(dbeta, scalar, b_grad)
                                if fused:
                                    _add(da, head, gate_grad * step)
                                    gate_grad = gate_grad * rate * _sigmoid(shifted)
                                    if has_bias:
                                        _add(dbias, head, gate_grad)
                                _add(dg, scalar, gate_grad)


class _Launcher:
    def __init__(self, config):
        self.config = config

    @host
    def run(
        self,
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        h0,
        a_log,
        bias,
        work,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dh,
        da,
        dbias,
    ):
        _FlashGDNBackward(self.config)[self.config[-1]](
            q,
            k,
            v,
            g,
            beta,
            do,
            dht,
            h0,
            a_log,
            bias,
            work,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dh,
            da,
            dbias,
        )


@lru_cache(maxsize=None)
def _compile(device_index, config, sizes):
    specs = [TensorSpec((size,), dtypes.float32, stride=(1,)) for size in sizes]
    return dsl.compile(_Launcher(config).run, *specs)


@torch.no_grad()
def flash_gdn_grad_npu(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    *,
    scale=None,
    use_beta_sigmoid_in_kernel=True,
):
    """BSND GDN backward with FP32 accumulation and optional head-wise gates.

    q/k are [B,T,H,K], v/do are [B,T,HV,V], g/beta are [B,T,HV].
    States are [B,HV,K,V]; A_log and dt_bias are [HV]. Without A_log,
    g is already the per-token log decay. Returns dq, dk, dv, dh0, dg,
    dbeta, dA_log, dBias. An absent parameter has a None gradient.
    """
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or q.shape[:2] != v.shape[:2]:
        raise ValueError("expected q/k [B,T,H,K] and v [B,T,HV,V]")
    batch, tokens, heads, key_dim = q.shape
    hv, value_dim = v.shape[2:]
    if min(batch, heads, hv, key_dim, value_dim) < 1 or hv % heads:
        raise ValueError("dimensions must be positive and HV must be a multiple of H")
    if g.shape != v.shape[:3] or beta.shape != g.shape or do.shape != v.shape:
        raise ValueError("expected g/beta [B,T,HV] and do shaped like v")
    state_shape = (batch, hv, key_dim, value_dim)
    if dht.shape != state_shape or (
        initial_state is not None and initial_state.shape != state_shape
    ):
        raise ValueError("expected dht and initial_state [B,HV,K,V]")
    if A_log is not None and A_log.shape != (hv,):
        raise ValueError("A_log must be [HV]")
    if dt_bias is not None and (A_log is None or dt_bias.shape != (hv,)):
        raise ValueError("dt_bias must be [HV] and requires A_log")
    for tensor in (q, k, v, g, beta, do, dht, initial_state, A_log, dt_bias):
        if tensor is not None and (
            q.device.type != "npu"
            or tensor.device != q.device
            or tensor.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("expected FP32 or BF16 tensors on one NPU")
    scale = key_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")

    def zeros(shape):
        return torch.zeros(shape, device=q.device, dtype=torch.float32)

    dq, dk, dv = zeros(q.shape), zeros(k.shape), zeros(v.shape)
    dg, dbeta = zeros(g.shape), zeros(beta.shape)
    da, dbias = zeros((hv,)), zeros((hv,))
    dh = dht.float().contiguous()
    if tokens:
        chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
        h0 = zeros(state_shape) if initial_state is None else initial_state
        dh = torch.empty(state_shape, device=q.device, dtype=torch.float32)
        work = torch.empty(
            (batch * hv * (chunks + CHUNK_SIZE + 1) * key_dim * value_dim,),
            device=q.device,
            dtype=torch.float32,
        )
        inputs = (
            q,
            k,
            v,
            g,
            beta,
            do,
            dht,
            h0,
            da if A_log is None else A_log,
            dbias if dt_bias is None else dt_bias,
        )
        args = [tensor.float().contiguous().view(-1) for tensor in inputs]
        args.extend(
            tensor.view(-1) for tensor in (work, dq, dk, dv, dg, dbeta, dh, da, dbias)
        )
        blocks = min(32, (batch * hv * value_dim + THREADS - 1) // THREADS)
        config = (
            batch,
            tokens,
            heads,
            hv,
            key_dim,
            value_dim,
            scale,
            A_log is not None,
            dt_bias is not None,
            use_beta_sigmoid_in_kernel,
            blocks,
        )
        with torch.npu.device(q.device):
            program = _compile(
                q.device.index, config, tuple(tensor.numel() for tensor in args)
            )
            stream = torch.npu.current_stream()
            for tensor in args:
                tensor.record_stream(stream)
            program(*args)
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dh,
        dg.to(g.dtype),
        dbeta.to(beta.dtype),
        da.to(A_log.dtype) if A_log is not None else None,
        dbias.to(dt_bias.dtype) if dt_bias is not None else None,
    )
