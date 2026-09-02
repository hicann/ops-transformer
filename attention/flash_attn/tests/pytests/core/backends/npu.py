# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU 后端 — 按 flash_attn.md 参数组调用，支持 eager / graph 两种模式。"""

import numpy as np
import torch
from typing import Any, Dict

from .base import Backend

try:
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    _HAS_TORCHAIR = True
except ImportError:
    _HAS_TORCHAIR = False


class FlashAttnGraphNetwork(torch.nn.Module):
    """torch.compile 图模式包装器。"""

    def forward(
        self,
        q,
        k,
        v,
        block_table,
        attn_mask,
        metadata,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        return_softmax_lse,
    ):
        out, lse_out = torch.ops.cann_ops_transformer.flash_attn(
            q,
            k,
            v,
            block_table=block_table,
            sinks=None,
            attn_mask=attn_mask,
            metadata=metadata,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=return_softmax_lse,
        )
        return out, lse_out


_AIC_CORE_NUM = 36
_AIV_CORE_NUM = 72
_FA_META_SIZE = 16
_FD_META_SIZE = 16
_HEADER_SIZE = 16

_FA_FIELDS = [
    "BN2_START",
    "M_START",
    "S2_START",
    "BN2_END",
    "M_END",
    "S2_END",
    "FIRST_FD_WS_IDX",
]
_FD_FIELDS = ["BN2_IDX", "M_IDX", "WS_IDX", "WS_NUM", "M_START", "M_NUM"]


def print_metadata(metadata):
    arr = metadata.cpu().contiguous().to(torch.int32).numpy().astype(np.uint32)

    section_num = int(arr[0])
    is_fd = int(arr[1])
    m_base_size = int(arr[2])
    s2_base_size = int(arr[3])

    SEP = "=" * 78
    DASH = "-" * 78

    print(f"\n{SEP}")
    print("  [Metadata Header]")
    print(f"  sectionNum  = {section_num}")
    print(f"  isFD        = {is_fd}")
    print(f"  mBaseSize   = {m_base_size}")
    print(f"  s2BaseSize  = {s2_base_size}")

    fa_base = _HEADER_SIZE
    fd_base = _HEADER_SIZE + section_num * _AIC_CORE_NUM * _FA_META_SIZE

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(
            f"  [Section {sec}] FA Metadata — AIC cores  "
            f"(36 cores × 16 slots, 7 fields used)"
        )
        print(DASH)
        cw = 15
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FA_FIELDS)
        print(hdr)
        print(
            f"  {'':<7}"
            + "".join(f"{'[' + str(i) + ']':>{cw}}" for i in range(len(_FA_FIELDS)))
        )
        print(f"  {'─' * 7}" + "─" * (cw * len(_FA_FIELDS)))
        for core in range(_AIC_CORE_NUM):
            off = fa_base + sec * _AIC_CORE_NUM * _FA_META_SIZE + core * _FA_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FA_FIELDS))]
            note = "  (inactive)" if all(v == 0 for v in vals) else ""
            print(f"  AIC{core:02d}  " + "".join(f"{v:>{cw}}" for v in vals) + note)

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(
            f"  [Section {sec}] FD Metadata — active AIV cores only  "
            f"(M_NUM > 0 shown, 72 cores × 16 slots, 6 fields used)"
        )
        print(DASH)
        cw = 12
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FD_FIELDS)
        print(hdr)
        print(
            f"  {'':<7}"
            + "".join(f"{'[' + str(i) + ']':>{cw}}" for i in range(len(_FD_FIELDS)))
        )
        print(f"  {'─' * 7}" + "─" * (cw * len(_FD_FIELDS)))
        active = 0
        for core in range(_AIV_CORE_NUM):
            off = fd_base + sec * _AIV_CORE_NUM * _FD_META_SIZE + core * _FD_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FD_FIELDS))]
            if vals[5] == 0:
                continue
            print(f"  AIV{core:02d} " + "".join(f"{v:>{cw}}" for v in vals))
            active += 1
        if active == 0:
            print("  (no active FD cores)")

    print(f"{SEP}\n")


class NPUBackend(Backend):
    name = "npu"

    def __init__(
        self, device_id: int = 0, meta_only: bool = False, graph_mode: bool = False
    ):
        self._device = torch.device(f"npu:{device_id}")
        self._meta_only = meta_only
        self._graph_mode = graph_mode
        self._graph_net = None
        if _HAS_NPU:
            torch.npu.set_device(device_id)
        if graph_mode and not _HAS_TORCHAIR:
            raise ImportError("graph_mode 需要 torchair，请先安装")

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_NPU

    def clear_cache(self):
        if _HAS_NPU:
            torch.npu.empty_cache()

    def _get_graph_net(self):
        if self._graph_net is None:
            torch._dynamo.reset()
            net = FlashAttnGraphNetwork().npu()
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = (
                "./"
            )
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True
            config.experimental_config.topology_sorting_strategy = "StableRDFS"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            self._graph_net = torch.compile(
                net, fullgraph=False, backend=npu_backend, dynamic=False
            )
        return self._graph_net

    def compute(
        self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        from core.data import build_flash_attn_params

        meta_kwargs, kernel_kwargs, _ = build_flash_attn_params(
            params, self._device, inputs
        )

        metadata = flash_attn_metadata(**meta_kwargs)

        if self._meta_only:
            print_metadata(metadata)
            bs = inputs["q"].shape[0]
            n1 = inputs["q"].shape[1] if inputs["q"].dim() >= 3 else params.get("N1", 1)
            s1 = inputs["q"].shape[2] if inputs["q"].dim() >= 4 else params.get("S1", 1)
            d = params["D"]
            dummy_out = torch.zeros(
                bs, n1, s1, d, dtype=inputs["q"].dtype, device="cpu"
            )
            return {"out": dummy_out, "lse": None}

        if self._graph_mode:
            net = self._get_graph_net()
            out, lse = net(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                kernel_kwargs.get("block_table"),
                kernel_kwargs.get("attn_mask"),
                metadata,
                kernel_kwargs.get("cu_seqlens_q"),
                kernel_kwargs.get("cu_seqlens_kv"),
                kernel_kwargs.get("seqused_q"),
                kernel_kwargs.get("seqused_kv"),
                kernel_kwargs.get("softmax_scale", 1.0),
                kernel_kwargs.get("mask_mode", 0),
                kernel_kwargs.get("win_left", -1),
                kernel_kwargs.get("win_right", -1),
                kernel_kwargs.get("max_seqlen_q", -1),
                kernel_kwargs.get("max_seqlen_kv", -1),
                kernel_kwargs.get("layout_q", "BNSD"),
                kernel_kwargs.get("layout_kv", "BNSD"),
                kernel_kwargs.get("layout_out", "BNSD"),
                kernel_kwargs.get("return_softmax_lse", False),
            )
        else:
            out, lse = flash_attn(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                metadata=metadata,
                **kernel_kwargs,
            )
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
