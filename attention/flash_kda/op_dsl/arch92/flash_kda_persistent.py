# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, Tuple

import torch
from cannbotdsl import GroupSync, Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.scalar import data_barrier
from cannbotdsl.ops.sync import cube_sync_all, vec_sync_all
from cannbotdsl.tensor import tile_slice

from . import flash_kda as fwd
from . import flash_kda_staged as staged
from .flash_kda_validation import CHUNK_SIZE, SUPPORTED_HEAD_DIM
from .flash_kda_metadata import (
    HEADER_WORDS,
    TEMPLATE_PERSISTENT,
    CHUNK_TASK_BATCH,
    CHUNK_TASK_STORAGE_BATCH,
    CHUNK_TASK_TOKEN_START,
    CHUNK_TASK_LENGTH,
    CHUNK_TASK_VALUE_HEAD,
    CHUNK_TASK_NEXT,
)


# Stage2 owns a complete value head on this pair. A full DV tile avoids
# reloading the same Q/W/K/Aqk operands four times for 32-column subtiles.
# State accumulation and chunk/pair completion boundaries remain unchanged.
PERSISTENT_DV_BASE = SUPPORTED_HEAD_DIM
# Four private slots per pair, independent of the final group length.
PERSISTENT_CHUNK_GROUP = 4


@jit
def _persistent_pair_boundary(pair_sync, phase):
    # Complete GM writes before both sides publish and wait on this phase.
    cube_sync_all()
    vec_sync_all()
    producer_ticket = pair_sync.acquire(phase)
    data_barrier(scope="gm")
    pair_sync.commit(producer_ticket)
    data_barrier(scope="gm")
    consumer_ticket = pair_sync.wait(phase)
    data_barrier(scope="gm")
    pair_sync.release(consumer_ticket)
    data_barrier(scope="gm")
    # The Buffer-ID pass requires completion immediately before rewind.
    cube_sync_all()
    vec_sync_all()
    staged._rewind_buffers()


@jit
def run_persistent_flash_kda_fwd_arch92(
    gm_K: Tensor,
    gm_V: Tensor,
    gm_Q: Tensor,
    gm_beta: Tensor,
    gm_g: Tensor,
    gm_qk_exchange: Tensor,
    gm_A_log: Tensor,
    gm_dt_bias: Tensor,
    gm_kg: Tensor,
    gm_gamma: Tensor,
    gm_q_decayed: Tensor,
    gm_u: Tensor,
    gm_saved_w: Tensor,
    gm_qg: Tensor,
    gm_g_cumsum: Tensor,
    gm_v_new: Tensor,
    gm_h: Tensor,
    gm_O: Tensor,
    gm_state_in: Tensor,
    gm_state_out: Tensor,
    gm_identity16: Tensor,
    gm_q_norm: Tensor,
    gm_k_norm: Tensor,
    gm_q_rstd: Tensor,
    gm_k_rstd: Tensor,
    gm_beta_activated: Tensor,
    gm_Aqk: Tensor,
    gm_Akk: Tensor,
    gm_metadata: Tensor,
    scale_value: float,
    lower_bound: float,
    use_qk_l2norm_in_kernel: int,
    use_gate_in_kernel: int,
    use_beta_sigmoid_in_kernel: int,
    allow_neg_eigval: int,
    has_lower_bound: int,
    state_is_compact: int,
    save_qg: int,
    save_g_cumsum: int,
    save_intermediate_state: int,
    save_matrices: int,
    save_final_state: int,
):
    """A6 persistent task chains: each Cube/Vector pair owns a complete sequence/head.

    Stage resources are reused at group boundaries. Synchronization here is
    local to the pair: empty cores must not participate in another core's chain.
    """
    block = get_block_idx()
    cores = get_block_num()
    heads = gm_V.shape[1]
    # GroupSync uses the first 128 bytes of this pair's 1024-byte GM slot.
    pair_workspace = tile_slice(gm_qk_exchange, (1, 2, 2, CHUNK_SIZE), (block, 0, 0, 0))
    pair_sync = GroupSync(pair_workspace, depth=1, producers=2, consumers=2)
    phase = dtypes.int64(0)
    task = gm_metadata[HEADER_WORDS + block]
    while task != 0:
        batch = gm_metadata[task + CHUNK_TASK_BATCH]
        storage = gm_metadata[task + CHUNK_TASK_STORAGE_BATCH]
        start = gm_metadata[task + CHUNK_TASK_TOKEN_START]
        length = dtypes.int64(gm_metadata[task + CHUNK_TASK_LENGTH])
        head = gm_metadata[task + CHUNK_TASK_VALUE_HEAD]
        stop = start + length
        chunks = (length + 63) // 64
        chunk_base = 0
        if state_is_compact != 0:
            chunk_base = start // 64 + batch
        k = gm_K[storage, None, start:stop, None]
        v = gm_V[storage, None, start:stop, None]
        q = gm_Q[storage, None, start:stop, None]
        beta = gm_beta[storage, None, start:stop, None]
        gate = gm_g[storage, None, start:stop, None]
        kg = gm_kg[storage, None, start:stop, None]
        u = gm_u[storage, None, start:stop, None]
        w = gm_saved_w[storage, None, start:stop, None]
        qg = gm_qg[storage, None, start:stop, None]
        v_new = gm_v_new[storage, None, start:stop, None]
        out = gm_O[storage, None, start:stop, None]
        mqk = gm_Aqk[storage : storage + 1, None, start:stop, None]
        akk = gm_Akk[storage : storage + 1, None, start:stop, None]
        gate_batch, gate_start, gate_stop = storage, start, stop
        if save_g_cumsum == 0:
            gate_batch, gate_start, gate_stop = 0, 0, 1
        gate_cache = gm_g_cumsum[
            gate_batch : gate_batch + 1, None, gate_start:gate_stop, None
        ]
        for group_start in range(0, chunks, PERSISTENT_CHUNK_GROUP):
            group_count = PERSISTENT_CHUNK_GROUP
            if group_start + group_count > chunks:
                group_count = chunks - group_start
            rows = length - (chunks - 1) * CHUNK_SIZE
            # Stage1 publishes all group operands before Stage2 consumes them.
            # The second pair boundary completes consumers before these slots
            # are reused. The stride stays four even for a partial final group.
            workspace_base = (block - head) * PERSISTENT_CHUNK_GROUP
            staged._run_stage1_body(
                group_start + group_count == chunks,
                rows,
                k,
                v,
                q,
                beta,
                gate,
                gm_qk_exchange,
                gm_A_log,
                gm_dt_bias,
                gm_q_decayed,
                kg,
                gm_gamma,
                u,
                w,
                qg,
                gate_cache,
                gm_identity16,
                gm_q_norm[storage, None, start:stop, None],
                gm_k_norm[storage, None, start:stop, None],
                gm_q_rstd[storage, None, start:stop, None],
                gm_k_rstd[storage, None, start:stop, None],
                gm_beta_activated[storage, None, start:stop, None],
                mqk,
                akk,
                workspace_base,
                heads,
                chunks,
                length,
                0,
                group_start,
                group_count,
                PERSISTENT_CHUNK_GROUP,
                cores,
                head * group_count,
                (head + 1) * group_count,
                scale_value,
                lower_bound,
                use_qk_l2norm_in_kernel,
                use_gate_in_kernel,
                use_beta_sigmoid_in_kernel,
                allow_neg_eigval,
                has_lower_bound,
                save_qg,
                save_g_cumsum,
                save_matrices,
            )
            _persistent_pair_boundary(pair_sync, phase)
            phase += 1
            resources = staged._StageTwoResources(PERSISTENT_DV_BASE)
            staged._run_stage2_body(
                kg,
                gm_gamma,
                gm_q_decayed,
                w,
                u,
                v_new,
                gm_h,
                out,
                gm_state_in,
                gm_state_out,
                mqk,
                workspace_task_base=workspace_base,
                dv_base=PERSISTENT_DV_BASE,
                head_base=head,
                head_count=1,
                chunks_total=chunks,
                seq_len=length,
                group_start=group_start,
                group_count=group_count,
                eff_group=PERSISTENT_CHUNK_GROUP,
                logical_core_num=cores,
                item_start=0,
                item_end=SUPPORTED_HEAD_DIM // PERSISTENT_DV_BASE,
                resources=resources,
                state_batch=batch,
                state_batch_idx=0
                if state_is_compact != 0 and save_intermediate_state != 0
                else batch,
                state_chunk_base=chunk_base,
                save_final_state=save_final_state,
                save_intermediate_state=save_intermediate_state,
                save_v_new=save_qg,
            )
            _persistent_pair_boundary(pair_sync, phase)
            phase += 1
        task = gm_metadata[task + CHUNK_TASK_NEXT]
        # Complete both local pipelines after the task load, before the loop backedge.
        cube_sync_all()
        vec_sync_all()
    cube_sync_all()
    vec_sync_all()


def persistent_flash_kda_fwd_arch92(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Optional[torch.Tensor], ...]:
    return fwd._flash_kda_fwd_arch92_impl(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
        internal_template=TEMPLATE_PERSISTENT,
    )


@export("persistent_flash_kda_fwd_arch92")
def export_persistent_flash_kda_fwd_arch92():
    """收集直接模板编译请求，不启动 Provider。"""
    try:
        fwd._get_compiled_kernel(TEMPLATE_PERSISTENT)
    finally:
        fwd.clear_caches()


_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_flash_kda_fwd_arch92", "FRAGMENT")
_GRAPH_LIBRARY.define(
    "persistent_flash_kda_fwd_arch92(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "Tensor? metadata, Tensor? initial_state=None, Tensor? A_log=None, Tensor? dt_bias=None, "
    "Tensor? cu_seqlens=None, int[]? cu_seqlens_cpu=None, Tensor? seqused=None, *, "
    f"Scalar? scale={SUPPORTED_HEAD_DIM ** (-0.5)}, "
    "Scalar output_final_state=False, Scalar return_intermediate_states=False, "
    "Scalar use_qk_l2norm_in_kernel=True, Scalar use_gate_in_kernel=True, "
    "Scalar use_beta_sigmoid_in_kernel=True, Scalar allow_neg_eigval=False, "
    'Scalar disable_recompute=False, Scalar? lower_bound=None, str layout="TND") '
    "-> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, "
    "Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)"
)


@torch.library.impl(_GRAPH_LIBRARY, "persistent_flash_kda_fwd_arch92", "Meta")
def _persistent_flash_kda_fwd_arch92_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Optional[torch.Tensor], ...]:
    return fwd._flash_kda_fwd_arch92_meta(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
    )


@torch.library.impl(_GRAPH_LIBRARY, "persistent_flash_kda_fwd_arch92", "PrivateUse1")
def _persistent_flash_kda_fwd_arch92_privateuse1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Optional[torch.Tensor], ...]:
    return persistent_flash_kda_fwd_arch92(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
    )


persistent_flash_kda_fwd_arch92_op = (
    torch.ops.cannbotdsl_flash_kda_fwd_arch92.persistent_flash_kda_fwd_arch92
)


@torch._dynamo.substitute_in_graph(
    persistent_flash_kda_fwd_arch92, can_constant_fold_through=False
)
def _persistent_flash_kda_fwd_arch92_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = SUPPORTED_HEAD_DIM ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Optional[torch.Tensor], ...]:
    return persistent_flash_kda_fwd_arch92_op(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        seqused=seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        lower_bound=lower_bound,
        layout=layout,
    )
