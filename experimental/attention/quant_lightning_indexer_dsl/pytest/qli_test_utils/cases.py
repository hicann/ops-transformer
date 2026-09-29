# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import torch


BATCH_SIZES = (1, 2, 4, 8, 16, 32)
SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 131072)
CMP_RATIOS = (1, 2)
STORAGE_LAYOUTS = ("contiguous", "axis0_noncontiguous")
PREFILL_QUERY_LENGTH = 8192

PAGE_SIZE = 128
PACKED_D = 64
N1 = 32
N2 = 1
TOPK = 512
CANDIDATE_TOPK = 2048
CANDIDATE_BLOCK_SIZE = 8


@dataclass(frozen=True)
class QliCase:
    scenario: Literal["decode", "prefill"]
    batch: int
    s2: int
    cmp_ratio: int
    storage_layout: Literal[
        "contiguous",
        "key_axis0_noncontiguous",
        "scale_axis0_noncontiguous",
        "axis0_noncontiguous",
    ]

    @property
    def query_length(self) -> int:
        return 1 if self.scenario == "decode" else PREFILL_QUERY_LENGTH

    @property
    def id(self) -> str:
        geometry = (
            f"{self.scenario}-B{self.batch}-S1_{self.query_length // 1024}K"
            f"-S2_{self.s2 // 1024}K-R{self.cmp_ratio}"
            if self.scenario == "prefill"
            else f"decode-B{self.batch}-S2_{self.s2 // 1024}K-R{self.cmp_ratio}"
        )
        layout = {
            "contiguous": "CONTIG",
            "key_axis0_noncontiguous": "K-NC0",
            "scale_axis0_noncontiguous": "KS-NC0",
            "axis0_noncontiguous": "NC0",
        }[self.storage_layout]
        return f"{geometry}-{layout}"


@dataclass
class QliInputs:
    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    q_descale: torch.Tensor
    k_descale: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_q: torch.Tensor
    seqused_k: torch.Tensor
    cmp_residual_k: torch.Tensor | None
    block_table: torch.Tensor


def precision_cases(scenario: Literal["decode", "prefill"]) -> tuple[QliCase, ...]:
    """Return the requested 6×7×2 matrix for both PA storage modes."""
    return tuple(
        QliCase(scenario, batch, s2, ratio, storage_layout)
        for batch, s2, ratio, storage_layout in product(
            BATCH_SIZES, SEQUENCE_LENGTHS, CMP_RATIOS, STORAGE_LAYOUTS
        )
    )


def _actual_k_lengths(batch: int, s2: int) -> torch.Tensor:
    tails = (0, 1, 7, 9, 31, 63, 95, 127)
    return torch.tensor(
        [s2 - tails[index % len(tails)] for index in range(batch)],
        dtype=torch.int32,
    )


def _with_axis0_gap(tensor: torch.Tensor) -> torch.Tensor:
    """Keep logical values while making only PA axis 0 non-contiguous."""
    strides = list(tensor.stride())
    strides[0] *= 2
    result = torch.empty_strided(
        tensor.shape,
        tuple(strides),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result.copy_(tensor)
    return result


def make_inputs(case: QliCase) -> QliInputs:
    """Create deterministic CPU tensors using only the public QLI ABI.

    Physical PA pages are shared across batches but each block-table row uses
    a different cyclic logical order.  This keeps 128K prefill inputs bounded
    while still exercising device-side page translation for every batch.
    """
    generator = torch.Generator(device="cpu").manual_seed(
        case.batch * 1_000_000 + case.query_length * 100 + case.s2 + case.cmp_ratio
    )
    # PA storage is page based even when the logical S2 is only a short tail.
    # Keep one complete physical page so Cube always receives its stable
    # 128-token N tile; seqused_k carries the exact logical token count.
    pages_per_batch = (case.s2 + PAGE_SIZE - 1) // PAGE_SIZE
    total_q = case.batch * case.query_length

    q = torch.randint(
        0,
        256,
        (total_q, N1, PACKED_D),
        generator=generator,
        dtype=torch.uint8,
    )
    k = torch.randint(
        0,
        256,
        (pages_per_batch, PAGE_SIZE, N2, PACKED_D),
        generator=generator,
        dtype=torch.uint8,
    )
    w = torch.rand((total_q, N1), generator=generator, dtype=torch.float32)
    q_descale = torch.randint(
        125,
        130,
        (total_q, N1, 2, 2),
        generator=generator,
        dtype=torch.uint8,
    )
    k_descale = torch.randint(
        125,
        130,
        (pages_per_batch, PAGE_SIZE, N2, 2, 2),
        generator=generator,
        dtype=torch.uint8,
    )
    if case.storage_layout in (
        "key_axis0_noncontiguous",
        "axis0_noncontiguous",
    ):
        k = _with_axis0_gap(k)
    if case.storage_layout in (
        "scale_axis0_noncontiguous",
        "axis0_noncontiguous",
    ):
        k_descale = _with_axis0_gap(k_descale)

    page_ids = torch.arange(pages_per_batch, dtype=torch.int32)
    block_table = torch.stack(
        [torch.roll(page_ids, shifts=batch) for batch in range(case.batch)]
    )
    cu_seqlens_q = torch.arange(0, total_q + 1, case.query_length, dtype=torch.int32)
    cmp_residual_k = None
    if case.cmp_ratio != 1:
        cmp_residual_k = torch.tensor(
            [batch % case.cmp_ratio for batch in range(case.batch)],
            dtype=torch.int32,
        )
    return QliInputs(
        q=q,
        k=k,
        w=w,
        q_descale=q_descale,
        k_descale=k_descale,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=torch.full((case.batch,), case.query_length, dtype=torch.int32),
        seqused_k=_actual_k_lengths(case.batch, case.s2),
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
    )


def operator_kwargs(case: QliCase, inputs: QliInputs) -> dict[str, object]:
    """Build the shared CPU/NPU call arguments from the public contract."""
    return {
        "cu_seqlens_q": inputs.cu_seqlens_q,
        "seqused_q": inputs.seqused_q,
        "seqused_k": inputs.seqused_k,
        "cmp_residual_k": inputs.cmp_residual_k,
        "block_table": inputs.block_table,
        "topk": TOPK,
        "quant_mode": 1,
        "max_seqlen_q": case.query_length,
        "mask_mode": 3,
        "cmp_ratio": case.cmp_ratio,
        "layout_q": "TND",
        "layout_k": "PA_BBND",
        "return_value": True,
        "candidate_topk_blocks": CANDIDATE_TOPK,
        "candidate_block_size": CANDIDATE_BLOCK_SIZE,
    }


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "CANDIDATE_TOPK",
    "QliCase",
    "QliInputs",
    "STORAGE_LAYOUTS",
    "TOPK",
    "make_inputs",
    "operator_kwargs",
    "precision_cases",
]
