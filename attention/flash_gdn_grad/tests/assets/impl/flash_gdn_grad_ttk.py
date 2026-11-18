# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTK adapter for ``flash_gdn_grad``: the ``flash_gdn_grad_ttk.npu`` callable.

``api_name`` in the case CSV resolves here. Tensors arrive in CSV order::

    q, k, v, g, beta, do, dht, h0, A_log, dt_bias, cu_seqlens, seqused

``None`` entries are absent optional inputs. Outputs are returned in the
operator order ``(dq, dk, dv, dh0, dg, dbeta, dA_log, ddt_bias)``.
"""

import sys
from pathlib import Path
from typing import Optional

from torch import Tensor

import flash_gdn_grad_ttk_golden as golden_impl

import torch

__all__ = ["npu"]

_OPERATOR_ROOT = Path(__file__).resolve().parents[3]


def _load_operator():
    """Import the operator kernels from this repository's ``op_dsl`` package."""
    root = str(_OPERATOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from op_dsl import flash_gdn_grad as operator
    from op_dsl.flash_gdn_grad_metadata import flash_gdn_grad_metadata

    return operator, flash_gdn_grad_metadata


def _synth_akk(
    q,
    k,
    v,
    g,
    beta,
    dht,
    h0,
    a_log,
    dt_bias,
    metadata,
    layout,
    scale,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    use_qk_l2norm_in_kernel,
):
    """TTK 侧造必需的 Akk（前向算子职责）；用例里 Akk 传 None 时使用。"""
    operator, _ = _load_operator()
    a = None if not use_gate_in_kernel or a_log is None else a_log.float().contiguous()
    bias = None if a is None or dt_bias is None else dt_bias.float().contiguous()
    return operator.prepare_forward_akk(
        q,
        k,
        v,
        g,
        beta,
        h0,
        a,
        bias,
        metadata,
        layout,
        scale,
        use_beta_sigmoid_in_kernel,
        use_qk_l2norm_in_kernel,
    )


def npu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    dht: Tensor,
    Akk: Tensor,
    Aqk: Optional[Tensor] = None,
    h0: Optional[Tensor] = None,
    a_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    layout: str = "BSND",
    scale: Optional[float] = None,
    use_beta_sigmoid_in_kernel: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    strategy: str = "staged",
    disable_recompute: bool = False,
    chunk_states: Optional[Tensor] = None,
    w: Optional[Tensor] = None,
    u: Optional[Tensor] = None,
    v_new: Optional[Tensor] = None,
    qg: Optional[Tensor] = None,
    kg: Optional[Tensor] = None,
    g_cumsum: Optional[Tensor] = None,
    q_rstd: Optional[Tensor] = None,
    k_rstd: Optional[Tensor] = None,
    beta_sigmoid: Optional[Tensor] = None,
):
    """Run ``flash_gdn_grad`` on NPU and return the eight gradients."""
    if q.device.type != "npu":
        # Host replay: TTK calls the API on CPU to build the golden.
        return golden_impl.golden(
            q,
            k,
            v,
            g,
            beta,
            do,
            dht,
            h0,
            a_log,
            dt_bias,
            layout=layout,
            scale=scale,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
        )

    operator, build_metadata = _load_operator()
    metadata = build_metadata(q, layout=layout, cu_seqlens=cu_seqlens, seqused=seqused)
    if Akk is None:
        Akk = _synth_akk(
            q,
            k,
            v,
            g,
            beta,
            dht,
            h0,
            a_log,
            dt_bias,
            metadata,
            layout,
            scale,
            use_gate_in_kernel,
            use_beta_sigmoid_in_kernel,
            use_qk_l2norm_in_kernel,
        )
    outputs = operator.flash_gdn_grad_npu(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk=Aqk,
        Akk=Akk,
        metadata=metadata,
        initial_state=h0,
        chunk_states=chunk_states,
        w=w,
        u=u,
        v_new=v_new,
        qg=qg,
        kg=kg,
        g_cumsum=g_cumsum,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_sigmoid,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        layout=layout,
        scale=scale,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        strategy=strategy,
        disable_recompute=disable_recompute,
    )
    torch.npu.synchronize()
    return tuple(outputs)
