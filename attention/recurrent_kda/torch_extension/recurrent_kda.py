# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class RecurrentKdaOpBuilder(OpBuilder):
    def __init__(self):
        super(RecurrentKdaOpBuilder, self).__init__(
            "recurrent_kda", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/recurrent_kda.cpp"]

    def schema(self) -> str:
        """PyTorch operator schema."""
        return (
            "recurrent_kda("
            "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor(a!) initial_state, "
            "*, "
            "Tensor? cu_seqlens=None, "
            "Tensor? ssm_state_indices=None, "
            "Tensor? A_log=None, "
            "Tensor? dt_bias=None, "
            "Tensor? num_accepted_tokens=None, "
            'str layout="BSND", '
            "float? scale=None, "
            "bool? output_final_state=False, "
            "bool? inplace_final_state=True, "
            "bool? use_qk_l2norm_in_kernel=False, "
            "bool? use_gate_in_kernel=False, "
            "bool? use_beta_sigmoid_in_kernel=False, "
            "bool? allow_neg_eigval=False, "
            "bool? safe_gate=False, "
            "float? lower_bound=None, "
            "bool? state_v_first=False"
            ") -> (Tensor, Tensor?)"
        )

    def register_meta(self):
        """Register Meta implementation for shape and dtype inference."""

        @impl(get_as_library(), self.name, "Meta")
        def recurrent_kda_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            *,
            cu_seqlens: Optional[torch.Tensor] = None,
            ssm_state_indices: Optional[torch.Tensor] = None,
            A_log: Optional[torch.Tensor] = None,
            dt_bias: Optional[torch.Tensor] = None,
            num_accepted_tokens: Optional[torch.Tensor] = None,
            layout: str = "BSND",
            scale: Optional[float] = None,
            output_final_state: Optional[bool] = False,
            inplace_final_state: Optional[bool] = True,
            use_qk_l2norm_in_kernel: Optional[bool] = False,
            use_gate_in_kernel: Optional[bool] = False,
            use_beta_sigmoid_in_kernel: Optional[bool] = False,
            allow_neg_eigval: Optional[bool] = False,
            safe_gate: Optional[bool] = False,
            lower_bound: Optional[float] = None,
            state_v_first: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            del (
                q,
                k,
                g,
                beta,
                cu_seqlens,
                ssm_state_indices,
                A_log,
                dt_bias,
                num_accepted_tokens,
                layout,
                scale,
                inplace_final_state,
                use_qk_l2norm_in_kernel,
                use_gate_in_kernel,
                use_beta_sigmoid_in_kernel,
                allow_neg_eigval,
                safe_gate,
                lower_bound,
                state_v_first,
            )
            out = torch.empty_like(v)
            final_state = (
                torch.empty_like(initial_state) if output_final_state else None
            )
            return out, final_state


_recurrent_kda_op_builder = RecurrentKdaOpBuilder()
_recurrent_kda_op_builder._ensure_initialized()


@impl(get_as_library(), _recurrent_kda_op_builder.name, "PrivateUse1")
def _recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: Optional[float] = None,
    output_final_state: Optional[bool] = False,
    inplace_final_state: Optional[bool] = True,
    use_qk_l2norm_in_kernel: Optional[bool] = False,
    use_gate_in_kernel: Optional[bool] = False,
    use_beta_sigmoid_in_kernel: Optional[bool] = False,
    allow_neg_eigval: Optional[bool] = False,
    safe_gate: Optional[bool] = False,
    lower_bound: Optional[float] = None,
    state_v_first: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    op_module = _recurrent_kda_op_builder.load()
    return op_module.recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        A_log,
        dt_bias,
        num_accepted_tokens,
        layout,
        scale,
        output_final_state,
        inplace_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        lower_bound,
        state_v_first,
    )


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    layout: str = "BSND",
    scale: Optional[float] = None,
    output_final_state: Optional[bool] = False,
    inplace_final_state: Optional[bool] = True,
    use_qk_l2norm_in_kernel: Optional[bool] = False,
    use_gate_in_kernel: Optional[bool] = False,
    use_beta_sigmoid_in_kernel: Optional[bool] = False,
    allow_neg_eigval: Optional[bool] = False,
    safe_gate: Optional[bool] = False,
    lower_bound: Optional[float] = None,
    state_v_first: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Execute recurrent KDA and optionally return the final recurrent state.

    The initial_state input is mutable when inplace_final_state is true.
    The second return value is None when output_final_state is false.
    """
    return torch.ops.cann_ops_transformer.recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        A_log=A_log,
        dt_bias=dt_bias,
        num_accepted_tokens=num_accepted_tokens,
        layout=layout,
        scale=scale,
        output_final_state=output_final_state,
        inplace_final_state=inplace_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
    )
