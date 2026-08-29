# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional
import torch
import torch_npu
from enum import IntEnum
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MaskMode(IntEnum):
    ALL = 0
    CAUSAL = 3
    WINDOW = 4


class QuantMode(IntEnum):
    HIF8_PER_TENSOR = 0
    FP4_E2M1 = 1
    HIF4 = 2


class QuantFlashAttnGradOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashAttnGradOpBuilder, self).__init__(
            "quant_flash_attn_grad", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_flash_attn_grad.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature.

        Note: 17 inputs (11 required + 6 optional) + 9 attrs + 4 outputs (dq, dk, dv, dsink).
        Output dtype is always BF16 per excel1 spec.
        Attr order MUST match C++ quant_flash_attn_grad.cpp signature:
            quant_mode, softmax_scale, mask_mode, win_left, win_right,
            max_seqlen_q, max_seqlen_kv, layout_q, layout_k
        (C++ param named max_mode at position 3 is treated as mask_mode here; will be fixed in C++ side.)
        """

        return (
            "quant_flash_attn_grad("
            "Tensor q, Tensor k, Tensor v, Tensor dout, Tensor attn_out, "
            "Tensor q_descale, Tensor k_descale, Tensor v_descale, Tensor do_descale, "
            "Tensor p_scale, Tensor ds_scale, Tensor softmax_lse, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, "
            "int quant_mode=0, float softmax_scale=1.0, int mask_mode=0, "
            "int win_left=-1, int win_right=-1, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="BSND", str layout_kv="BSND")'
            " -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.

        Output shapes per excel1 (user decision: dk/dv same as dq, not mathematically strict):
          dq/dk/dv: same shape as q (layout-dependent)
          dsink: (nheads_q,)
        All outputs are BF16.
        """

        @impl(get_as_library(), self.name, "Meta")
        def quant_flash_attn_grad_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dout: torch.Tensor,
            attn_out: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            v_descale: torch.Tensor,
            do_descale: torch.Tensor,
            p_scale: torch.Tensor,
            ds_scale: torch.Tensor,
            softmax_lse: torch.Tensor,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            quant_mode: QuantMode = QuantMode.HIF8_PER_TENSOR,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[MaskMode] = MaskMode.ALL,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
        ):
            if layout_q == "TND":
                t_q, n_q, d = q.size(0), q.size(1), q.size(2)
                # excel1 规定 dk/dv 与 dq 同形 (用户决策保持原文，不按数学定义修正)
                dq_size = (t_q, n_q, d)
                dk_size = (t_q, n_q, d)
                dv_size = (t_q, n_q, d)
            elif layout_q == "BSND":
                b, s_q, s_k, n_q, d = (
                    q.size(0),
                    q.size(1),
                    k.size(2),
                    q.size(2),
                    q.size(3),
                )
                dq_size = (b, s_q, n_q, d)
                dk_size = (b, s_k, n_q, d)
                dv_size = (b, s_k, n_q, d)
            elif layout_q == "BNSD":
                b, n_q, s_q, s_k, d = (
                    q.size(0),
                    q.size(1),
                    q.size(2),
                    k.size(2),
                    q.size(3),
                )
                dq_size = (b, n_q, s_q, d)
                dk_size = (b, s_k, s_q, d)
                dv_size = (b, s_k, s_q, d)
            else:
                torch._check(
                    False,
                    lambda: "Unsupported layout_q: "
                    + str(layout_q)
                    + ", supported: BSND / BNSD / TND",
                )

            dsink_size = (n_q,)

            return (
                torch.empty(dq_size, dtype=torch.bfloat16, device="meta"),
                torch.empty(dk_size, dtype=torch.bfloat16, device="meta"),
                torch.empty(dv_size, dtype=torch.bfloat16, device="meta"),
                torch.empty(dsink_size, dtype=torch.bfloat16, device="meta"),
            )


# Instantiate the builder
quant_flash_attn_grad_op_builder = QuantFlashAttnGradOpBuilder()
quant_flash_attn_grad_op_builder._ensure_initialized()


@impl(get_as_library(), quant_flash_attn_grad_op_builder.name, "PrivateUse1")
def quant_flash_attn_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    attn_out: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    do_descale: torch.Tensor,
    p_scale: torch.Tensor,
    ds_scale: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    quant_mode: QuantMode = QuantMode.HIF8_PER_TENSOR,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[MaskMode] = MaskMode.ALL,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.

    QuantFlashAttnGrad反向梯度算子:
        输入: q/k/v/dout (UINT8, 量化), attn_out (BF16), 6个 descale/scale (FP32),
              softmax_lse (FP32), cu_seqlens/seqused/sinks/attn_mask/metadata (可选)
        输出: dq/dk/dv/dsink (BF16)
    """
    quant_mode = quant_mode.value if isinstance(quant_mode, IntEnum) else quant_mode
    mask_mode = mask_mode.value if isinstance(mask_mode, IntEnum) else mask_mode
    op_module = quant_flash_attn_grad_op_builder.load()  # compiles/loads the .so file
    return op_module.quant_flash_attn_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        q_descale,
        k_descale,
        v_descale,
        do_descale,
        p_scale,
        ds_scale,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        quant_mode,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
    )


quant_flash_attn_grad.MaskMode = MaskMode
quant_flash_attn_grad.QuantMode = QuantMode
