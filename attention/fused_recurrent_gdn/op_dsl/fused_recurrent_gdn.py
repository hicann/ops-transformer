# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Native-layout GDN stages with event-ordered Cube/Vector double buffering.

State ABI: [state_rows, Nv, Dk, Dv].  A = alpha*(I-beta*k*k.T),
S = A@S + beta*k*v.T, out = scale*S.T@q.
"""

from functools import lru_cache
from threading import Lock

_inflight = []
_inflight_lock = Lock()
import cannbotdsl as cbd
from cannbotdsl import (
    Tensor,
    MemLoc,
    dim3,
    dtypes,
    host,
    kernel,
    make_layout,
    simt,
    vf,
)
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice


def fused_recurrent_gdn(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    *,
    scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    layout="TND",
):
    """GDN public implementation entry; parameter names follow the Torch schema."""
    import torch
    from ..fused_recurrent_gdn import _validate, _sequence_plan

    batch, _, nv, dk, dv = _validate(
        q,
        k,
        v,
        beta,
        g,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        layout,
    )
    spans, reads, writes = _sequence_plan(
        q,
        batch,
        layout,
        cu_seqlens,
        seqused,
        initial_state,
        ssm_state_indices,
        num_accepted_tokens,
    )

    state_dtype = torch.float32 if initial_state is None else initial_state.dtype
    state_shape = (batch, nv, dk, dv)
    if batch == 0:
        out = torch.empty_like(v, memory_format=torch.contiguous_format)
        final_state = (
            v.new_empty(state_shape, dtype=state_dtype) if output_final_state else None
        )
        return out, final_state

    pool_mode = ssm_state_indices is not None
    state = (
        torch.zeros(state_shape, dtype=state_dtype, device=q.device)
        if initial_state is None
        else initial_state.contiguous()
    )
    out = torch.zeros_like(v, memory_format=torch.contiguous_format)
    final_state = torch.empty(state_shape, dtype=state_dtype, device=q.device)
    zeros = torch.zeros((nv,), dtype=torch.float32, device=q.device)
    read_scale = torch.full(
        (1,), dk**-0.5 if scale is None else scale, dtype=torch.float32, device=q.device
    )
    launch_recurrent(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        beta.contiguous(),
        g.contiguous(),
        state,
        zeros if A_log is None else A_log.contiguous(),
        zeros if dt_bias is None else dt_bias.contiguous(),
        spans,
        reads,
        writes,
        read_scale,
        out,
        final_state,
        layout=layout,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        pool_mode=pool_mode,
    )
    if pool_mode and state is not initial_state:
        initial_state.copy_(state)
    return out, final_state if output_final_state else None


@jit
def _storage_row(layout, batch, head, token, extent, heads):
    if const_expr(layout == "TND"):
        return token * heads + head
    if const_expr(layout == "NTD"):
        return head * extent + token
    if const_expr(layout == "BSND"):
        return (batch * extent + token) * heads + head
    return (batch * heads + head) * extent + token


class ActivationTransitionStage:
    """VF produces a complete token's independent transition and write term."""

    def __init__(
        self,
        layout,
        dk,
        dv,
        nq,
        nv,
        batch,
        use_beta_sigmoid_in_kernel,
        use_gate_in_kernel,
        use_qk_l2norm_in_kernel,
        allow_neg_eigval,
    ):
        self.layout, self.dk, self.dv = layout, dk, dv
        self.nq, self.nv, self.batch = nq, nv, batch
        storage_batches = batch if layout in ("BSND", "BNSD") else 1
        self.q_rows = storage_batches * nq
        self.v_rows = storage_batches * nv
        self.kp, self.vp = (dk + 31) // 32 * 32, (dv + 63) // 64 * 64
        self.use_beta_sigmoid_in_kernel, self.use_gate_in_kernel = (
            use_beta_sigmoid_in_kernel,
            use_gate_in_kernel,
        )
        self.use_qk_l2norm_in_kernel, self.allow_neg_eigval = (
            use_qk_l2norm_in_kernel,
            allow_neg_eigval,
        )

    @kernel
    def compute(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        g: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        plan: Tensor,
        ah: Tensor,
        al: Tensor,
        write: Tensor,
        query: Tensor,
    ):
        with vf(mode="simt"):
            b = simt.block_idx()[0] // self.nv
            h = simt.block_idx()[0] % self.nv
            qh = h // (self.nv // self.nq)
            token = plan[b, 0]
            extent = plan[b, 2]
            q_row = _storage_row(self.layout, b, qh, token, extent, self.nq)
            v_row = _storage_row(self.layout, b, h, token, extent, self.nv)
            q_outer, q_inner = q_row // self.q_rows, q_row % self.q_rows
            v_outer, v_inner = v_row // self.v_rows, v_row % self.v_rows
            active = token >= 0
            for row_tile in range((self.kp + 127) // 128):
                i = row_tile * 128 + simt.thread_idx()[0]
                if i < self.kp:
                    decay = 1.0
                    coefficient = 0.0
                    ki = 0.0
                    qi = 0.0
                    kn = 1.0
                    qn = 1.0
                    if active:
                        gate = g[v_outer, v_inner]
                        c = beta[v_outer, v_inner]
                        if const_expr(self.use_gate_in_kernel):
                            x = simt.cast(gate, dtype=dtypes.float32) + dt_bias[h]
                            sp = 0.0
                            if x > 0.0:
                                sp = x + simt.log(1.0 + simt.exp(-x))
                            else:
                                sp = simt.log(1.0 + simt.exp(x))
                            gate_f = -simt.exp(A_log[h]) * sp
                        else:
                            gate_f = gate
                        decay = simt.exp(gate_f)
                        if const_expr(self.use_beta_sigmoid_in_kernel):
                            x = simt.cast(c, dtype=dtypes.float32)
                            if x >= 0.0:
                                coefficient = 1.0 / (1.0 + simt.exp(-x))
                            else:
                                ex = simt.exp(x)
                                coefficient = ex / (1.0 + ex)
                            if const_expr(self.allow_neg_eigval):
                                coefficient = coefficient * 2.0
                        else:
                            coefficient = c
                        if const_expr(self.use_qk_l2norm_in_kernel):
                            ks = 0.0
                            qs = 0.0
                            for j in range(self.dk):
                                kj = simt.cast(
                                    k[q_outer, q_inner, j], dtype=dtypes.float32
                                )
                                qj = simt.cast(
                                    q[q_outer, q_inner, j], dtype=dtypes.float32
                                )
                                ks = ks + kj * kj
                                qs = qs + qj * qj
                            kn = simt.rsqrt(ks + 1e-6)
                            qn = simt.rsqrt(qs + 1e-6)
                        if i < self.dk:
                            ki = (
                                simt.cast(k[q_outer, q_inner, i], dtype=dtypes.float32)
                                * kn
                            )
                            qi = (
                                simt.cast(q[q_outer, q_inner, i], dtype=dtypes.float32)
                                * qn
                            )
                    query[b, h, i] = qi
                    for j in range(self.kp):
                        kj = 0.0
                        if active and j < self.dk:
                            kj = (
                                simt.cast(k[q_outer, q_inner, j], dtype=dtypes.float32)
                                * kn
                            )
                        a = -decay * coefficient * ki * kj
                        if i == j:
                            a = a + decay
                        hi = simt.cast(a, dtype=dtypes.bfloat16)
                        ah[b, h, i, j] = hi
                        al[b, h, i, j] = simt.cast(
                            a - simt.cast(hi, dtype=dtypes.float32),
                            dtype=dtypes.bfloat16,
                        )
                    for j in range(self.vp):
                        value = 0.0
                        if active and j < self.dv:
                            value = simt.cast(
                                v[v_outer, v_inner, j], dtype=dtypes.float32
                            )
                        write[b, h, i, j] = coefficient * ki * value

    @host
    def run(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        g: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        plan: Tensor,
        ah: Tensor,
        al: Tensor,
        write: Tensor,
        query: Tensor,
    ):
        self.compute[dim3(self.batch * self.nv), dim3(128)](
            q, k, v, beta, g, A_log, dt_bias, plan, ah, al, write, query
        )


class StateLoadStage:
    def __init__(self, dk, dv, nv, batch):
        self.dk, self.dv, self.nv, self.batch = dk, dv, nv, batch
        self.kp, self.vp = (dk + 31) // 32 * 32, (dv + 63) // 64 * 64

    @kernel
    def compute(
        self,
        initial_state: Tensor,
        state_read_slots: Tensor,
        state: Tensor,
        bh: Tensor,
        bl: Tensor,
        final_state: Tensor,
    ):
        with vf(mode="simt"):
            b = simt.block_idx()[0] // self.nv
            h = simt.block_idx()[0] % self.nv
            for tile in range((self.kp + 127) // 128):
                i = tile * 128 + simt.thread_idx()[0]
                if i < self.kp:
                    for j in range(self.vp):
                        x = 0.0
                        if i < self.dk and j < self.dv:
                            x = initial_state[state_read_slots[b], h, i, j]
                            if const_expr(initial_state.dtype == dtypes.bfloat16):
                                x = simt.cast(x, dtype=dtypes.float32)
                            if const_expr(final_state.dtype == dtypes.bfloat16):
                                final_state[b, h, i, j] = simt.cast(
                                    x, dtype=dtypes.bfloat16
                                )
                            else:
                                final_state[b, h, i, j] = x
                        state[b, h, i, j] = x
                        hi = simt.cast(x, dtype=dtypes.bfloat16)
                        bh[b, h, j, i] = hi
                        bl[b, h, j, i] = simt.cast(
                            x - simt.cast(hi, dtype=dtypes.float32),
                            dtype=dtypes.bfloat16,
                        )

    @host
    def run(
        self,
        initial_state: Tensor,
        state_read_slots: Tensor,
        state: Tensor,
        bh: Tensor,
        bl: Tensor,
        final_state: Tensor,
    ):
        self.compute[dim3(self.batch * self.nv), dim3(128)](
            initial_state, state_read_slots, state, bh, bl, final_state
        )


class StateTransitionStage:
    """Cube A@S, four BF16 split products accumulated into one FP32 tile.

    L1/L0 depth-two rings overlap panel loads with MMAD. B is stored [Dv,Dk]
    for the explicit A[M,K] B[N,K]^T Cube convention.
    """

    def __init__(self, kp, vp, nv, batch):
        self.kp, self.vp, self.nv, self.batch = kp, vp, nv, batch
        self.tasks = batch * nv * (kp // 32) * (vp // 64)
        self.cores = min(32, self.tasks)

    @kernel
    def compute(self, ah: Tensor, al: Tensor, bh: Tensor, bl: Tensor, product: Tensor):
        a1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((32, 32)), depth=2)
        b1 = RingBuffer(MemLoc.L1, dtypes.bfloat16, make_layout((64, 32)), depth=2)
        a0 = RingBuffer(MemLoc.L0A, dtypes.bfloat16, make_layout((32, 32)), depth=2)
        b0 = RingBuffer(MemLoc.L0B, dtypes.bfloat16, make_layout((64, 32)), depth=2)
        c0 = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((32, 64)), depth=2)
        for task in range(get_block_idx(), self.tasks, get_block_num()):
            n = task % (self.vp // 64)
            m = task // (self.vp // 64) % (self.kp // 32)
            h = task // ((self.vp // 64) * (self.kp // 32)) % self.nv
            b = task // ((self.vp // 64) * (self.kp // 32) * self.nv)
            acc = c0.next_write()
            for panel in range(self.kp // 32):
                for part in range(4):
                    aa = a1.next_write()
                    bb = b1.next_write()
                    if part < 2:
                        mem_copy(
                            aa, tile_slice(ah[b, h, None, None], (32, 32), (m, panel))
                        )
                    else:
                        mem_copy(
                            aa, tile_slice(al[b, h, None, None], (32, 32), (m, panel))
                        )
                    if part % 2 == 0:
                        mem_copy(
                            bb, tile_slice(bh[b, h, None, None], (64, 32), (n, panel))
                        )
                    else:
                        mem_copy(
                            bb, tile_slice(bl[b, h, None, None], (64, 32), (n, panel))
                        )
                    la = a0.next_write()
                    lb = b0.next_write()
                    mem_copy(la, aa)
                    mem_copy(lb, bb)
                    matmul(acc, la, lb, init=panel == 0 and part == 0)
            mem_copy(tile_slice(product[b, h, None, None], (32, 64), (m, n)), acc)

    @host
    def run(self, ah: Tensor, al: Tensor, bh: Tensor, bl: Tensor, product: Tensor):
        self.compute[self.cores](ah, al, bh, bl, product)


class StateUpdateStage(StateLoadStage):
    def __init__(self, dk, dv, nv, batch, pool):
        super().__init__(dk, dv, nv, batch)
        self.pool = pool

    @kernel
    def update(
        self,
        product: Tensor,
        write: Tensor,
        previous: Tensor,
        plan: Tensor,
        state: Tensor,
        bh: Tensor,
        bl: Tensor,
        initial_state: Tensor,
        final_state: Tensor,
    ):
        with vf(mode="simt"):
            b = simt.block_idx()[0] // self.nv
            h = simt.block_idx()[0] % self.nv
            active = plan[b, 0] >= 0
            for tile in range((self.kp + 127) // 128):
                i = tile * 128 + simt.thread_idx()[0]
                if i < self.kp:
                    for j in range(self.vp):
                        x = previous[b, h, i, j]
                        if active:
                            x = product[b, h, i, j] + write[b, h, i, j]
                        state[b, h, i, j] = x
                        hi = simt.cast(x, dtype=dtypes.bfloat16)
                        bh[b, h, j, i] = hi
                        bl[b, h, j, i] = simt.cast(
                            x - simt.cast(hi, dtype=dtypes.float32),
                            dtype=dtypes.bfloat16,
                        )
                        if i < self.dk and j < self.dv:
                            if const_expr(final_state.dtype == dtypes.bfloat16):
                                final_state[b, h, i, j] = simt.cast(
                                    x, dtype=dtypes.bfloat16
                                )
                            else:
                                final_state[b, h, i, j] = x
                            if const_expr(self.pool):
                                if active:
                                    if const_expr(
                                        initial_state.dtype == dtypes.bfloat16
                                    ):
                                        initial_state[plan[b, 1], h, i, j] = simt.cast(
                                            x, dtype=dtypes.bfloat16
                                        )
                                    else:
                                        initial_state[plan[b, 1], h, i, j] = x

    @host
    def run_update(
        self,
        product: Tensor,
        write: Tensor,
        previous: Tensor,
        plan: Tensor,
        state: Tensor,
        bh: Tensor,
        bl: Tensor,
        initial_state: Tensor,
        final_state: Tensor,
    ):
        self.update[dim3(self.batch * self.nv), dim3(128)](
            product, write, previous, plan, state, bh, bl, initial_state, final_state
        )


class ReadoutStage(StateLoadStage):
    def __init__(self, dk, dv, nv, batch, layout):
        super().__init__(dk, dv, nv, batch)
        self.layout = layout
        self.v_rows = nv * (batch if layout in ("BSND", "BNSD") else 1)

    @kernel
    def read(
        self, state: Tensor, query: Tensor, plan: Tensor, scale: Tensor, out: Tensor
    ):
        with vf(mode="simt"):
            b = simt.block_idx()[0] // self.nv
            h = simt.block_idx()[0] % self.nv
            token = plan[b, 0]
            if token >= 0:
                extent = plan[b, 2]
                row = _storage_row(self.layout, b, h, token, extent, self.nv)
                outer, inner = row // self.v_rows, row % self.v_rows
                for tile in range((self.dv + 127) // 128):
                    j = tile * 128 + simt.thread_idx()[0]
                    if j < self.dv:
                        value = 0.0
                        for i in range(self.dk):
                            value = value + state[b, h, i, j] * query[b, h, i]
                        y = simt.cast(value * scale[0], dtype=dtypes.bfloat16)
                        out[outer, inner, j] = y

    @host
    def run_read(
        self, state: Tensor, query: Tensor, plan: Tensor, scale: Tensor, out: Tensor
    ):
        self.read[dim3(self.batch * self.nv), dim3(128)](state, query, plan, scale, out)


def _stage_metadata(kind, args):
    # Prepare/read views have a leading token axis and fixed inner strides.
    token_operands = range(5) if kind == "prepare" else (4,) if kind == "read" else ()
    metadata = []
    for index, tensor in enumerate(args):
        shape = list(tensor.shape)
        if index in token_operands:
            shape[0] = None
        metadata.append((tuple(shape), str(tensor.dtype).split(".")[-1]))
    return tuple(metadata)


@lru_cache(maxsize=128)
def _compile_stage(
    kind,
    layout,
    dk,
    dv,
    nq,
    nv,
    batch,
    use_beta_sigmoid_in_kernel,
    use_gate_in_kernel,
    use_qk_l2norm_in_kernel,
    allow_neg_eigval,
    pool,
    metadata,
):
    tokens = cbd.Dim("tokens", min=1) if kind in ("prepare", "read") else None
    specs = [
        cbd.TensorSpec(
            tuple(tokens if extent is None else extent for extent in shape),
            getattr(dtypes, dtype),
        )
        for shape, dtype in metadata
    ]
    if kind == "prepare":
        program = ActivationTransitionStage(
            layout,
            dk,
            dv,
            nq,
            nv,
            batch,
            use_beta_sigmoid_in_kernel,
            use_gate_in_kernel,
            use_qk_l2norm_in_kernel,
            allow_neg_eigval,
        ).run
    elif kind == "load":
        program = StateLoadStage(dk, dv, nv, batch).run
    elif kind == "cube":
        program = StateTransitionStage(
            (dk + 31) // 32 * 32, (dv + 63) // 64 * 64, nv, batch
        ).run
    elif kind == "update":
        program = StateUpdateStage(dk, dv, nv, batch, pool).run_update
    else:
        program = ReadoutStage(dk, dv, nv, batch, layout).run_read
    return cbd.compile(program, *specs)


def launch_recurrent(
    q,
    k,
    v,
    beta,
    g,
    initial_state,
    A_log,
    dt_bias,
    sequence_spans,
    state_read_slots,
    state_write_slots,
    scale,
    out,
    final_state,
    *,
    layout,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    use_qk_l2norm_in_kernel,
    allow_neg_eigval,
    pool_mode,
):
    """Launch native-layout stages with event-ordered double buffers.

    No NPU execution graph is constructed. Stage data and compiled handles
    remain alive until the caller stream completes this invocation.
    """
    import torch

    dk, dv = initial_state.shape[-2:]
    batch, nv = final_state.shape[:2]
    nq = q.shape[{"TND": 1, "NTD": 0, "BSND": 2, "BNSD": 1}[layout]]
    kp, vp = (dk + 31) // 32 * 32, (dv + 63) // 64 * 64
    config = (
        layout,
        dk,
        dv,
        nq,
        nv,
        batch,
        use_beta_sigmoid_in_kernel,
        use_gate_in_kernel,
        use_qk_l2norm_in_kernel,
        allow_neg_eigval,
        pool_mode,
    )

    def allocate(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device=q.device)

    states = [allocate((batch, nv, kp, vp)) for _ in range(2)]
    bh = [allocate((batch, nv, vp, kp), torch.bfloat16) for _ in range(2)]
    bl = [torch.empty_like(x) for x in bh]
    ah = [allocate((batch, nv, kp, kp), torch.bfloat16) for _ in range(2)]
    al = [torch.empty_like(x) for x in ah]
    terms = [allocate((batch, nv, kp, vp)) for _ in range(2)]
    queries = [allocate((batch, nv, kp)) for _ in range(2)]
    product = allocate((batch, nv, kp, vp))
    steps = max((end - start for start, end in sequence_spans), default=0)
    token_extent = q.shape[{"TND": 0, "NTD": 1, "BSND": 1, "BNSD": 2}[layout]]
    plans = []
    for step in range(steps):
        row = []
        for b, (start, end) in enumerate(sequence_spans):
            token = start + step
            active = token < end
            native_token = token if layout in ("TND", "NTD") else step
            row.append(
                [
                    native_token if active else -1,
                    state_write_slots[token] if active else 0,
                    token_extent,
                ]
            )
        plans.append(torch.tensor(row, dtype=torch.int32, device=q.device))
    read_tensor = torch.tensor(state_read_slots, dtype=torch.int32, device=q.device)

    def compile_for(kind, args):
        metadata = _stage_metadata(kind, args)
        return _compile_stage(kind, *config, metadata)

    load_args = (initial_state, read_tensor, states[0], bh[0], bl[0], final_state)
    load = compile_for("load", load_args)
    if steps == 0:
        load(*load_args)
        return
    # Reinterpret contiguous storage without permuting or copying it. The
    # kernels compute the original layout's linear row offsets explicitly.
    q = q.view(token_extent, -1, dk)
    k = k.view(token_extent, -1, dk)
    v = v.view(token_extent, -1, dv)
    beta = beta.view(token_extent, -1)
    g = g.view(token_extent, -1)
    out = out.view(token_extent, -1, dv)
    prep_args = (
        q,
        k,
        v,
        beta,
        g,
        A_log,
        dt_bias,
        plans[0],
        ah[0],
        al[0],
        terms[0],
        queries[0],
    )
    cube_args = (ah[0], al[0], bh[0], bl[0], product)
    update_args = (
        product,
        terms[0],
        states[0],
        plans[0],
        states[1],
        bh[1],
        bl[1],
        initial_state,
        final_state,
    )
    read_args = (states[1], queries[0], plans[0], scale, out)
    prepare = compile_for("prepare", prep_args)
    cube = compile_for("cube", cube_args)
    update = compile_for("update", update_args)
    read = compile_for("read", read_args)
    with torch.npu.device(q.device):
        caller = torch.npu.current_stream()
        compute = caller
        prep_stream = torch.npu.Stream(device=q.device)
        read_stream = torch.npu.Stream(device=q.device)
        ready_input = torch.npu.Event()
        ready_input.record(compute)
        prep_stream.wait_event(ready_input)
        read_stream.wait_event(ready_input)
        load(*load_args)
        prepared = [None, None]
        slot_free = [None, None]
        state_read_done = [None, None]

        def enqueue_prepare(t):
            slot = t % 2
            with torch.npu.stream(prep_stream):
                if slot_free[slot] is not None:
                    prep_stream.wait_event(slot_free[slot])
                prepare(
                    q,
                    k,
                    v,
                    beta,
                    g,
                    A_log,
                    dt_bias,
                    plans[t],
                    ah[slot],
                    al[slot],
                    terms[slot],
                    queries[slot],
                )
                prepared[slot] = torch.npu.Event()
                prepared[slot].record(prep_stream)

        enqueue_prepare(0)
        for t in range(steps):
            slot = t % 2
            previous = t % 2
            target = (t + 1) % 2
            compute.wait_event(prepared[slot])
            cube(ah[slot], al[slot], bh[previous], bl[previous], product)
            # Preparation depends only on token inputs, not on recurrent state.
            if t + 1 < steps:
                enqueue_prepare(t + 1)
            if state_read_done[target] is not None:
                compute.wait_event(state_read_done[target])
            update(
                product,
                terms[slot],
                states[previous],
                plans[t],
                states[target],
                bh[target],
                bl[target],
                initial_state,
                final_state,
            )
            updated = torch.npu.Event()
            updated.record(compute)
            with torch.npu.stream(read_stream):
                read_stream.wait_event(updated)
                read(states[target], queries[slot], plans[t], scale, out)
                consumed = torch.npu.Event()
                consumed.record(read_stream)
                slot_free[slot] = consumed
                state_read_done[target] = consumed
        for event in slot_free:
            if event is not None:
                compute.wait_event(event)
        # Keep asynchronous stage allocations alive until this invocation completes.
        complete = torch.npu.Event()
        complete.record(caller)
        owner = (
            load,
            prepare,
            cube,
            update,
            read,
            states,
            bh,
            bl,
            ah,
            al,
            terms,
            queries,
            product,
            plans,
            read_tensor,
            q,
            k,
            v,
            beta,
            g,
            initial_state,
            A_log,
            dt_bias,
            scale,
            out,
            final_state,
        )
        with _inflight_lock:
            _inflight[:] = [
                (event, item) for event, item in _inflight if not event.query()
            ]
            _inflight.append((complete, owner))
