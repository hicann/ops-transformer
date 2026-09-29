# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

from typing import Optional

import torch


_FP4_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


def pack_mxfp4(values: torch.Tensor) -> torch.Tensor:
    """Quantize test data to packed E2M1, with two values in each byte."""
    values = values.to(torch.float32)
    if values.shape[-1] % 2:
        raise ValueError("MXFP4 packing requires an even last dimension")
    levels = _FP4_VALUES[:8]
    codes = (values.abs().unsqueeze(-1) - levels).abs().argmin(-1)
    codes = codes.to(torch.uint8) + (values < 0).to(torch.uint8) * 8
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


class GeneralizedQSLI:
    """Self-contained QSLI reference following the ASC QLI golden structure."""

    GROUP_SIZE = 32
    CANDIDATE_BLOCK_SIZE = 8
    CANDIDATE_CAPACITY = 2048

    def __init__(
        self,
        *,
        topk: int,
        candidate_block_size: int,
        quant_mode: int,
        max_seqlen_q: int,
        mask_mode: int,
        cmp_ratio: int,
        layout_q: str,
        layout_k: str,
        return_value: bool,
    ):
        if quant_mode != 1:
            raise ValueError("QSLI golden supports quant_mode=1 only")
        if candidate_block_size != self.CANDIDATE_BLOCK_SIZE:
            raise ValueError("candidate_block_size must be 8")
        if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
            raise ValueError("QSLI golden requires TND queries and PA_BBND or TND keys")
        if mask_mode not in (0, 3):
            raise ValueError("mask_mode must be 0 or 3")
        if not 1 <= cmp_ratio <= 128:
            raise ValueError("cmp_ratio must be in [1,128]")
        self.topk = int(topk)
        self.max_seqlen_q = int(max_seqlen_q)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.return_value = bool(return_value)
        self.layout_k = layout_k

    @staticmethod
    def _unpack_mxfp4(packed: torch.Tensor) -> torch.Tensor:
        data = packed.contiguous().view(torch.uint8)
        pairs = torch.stack((data & 0x0F, data >> 4), dim=-1)
        return _FP4_VALUES[pairs.long()].reshape(*data.shape[:-1], data.shape[-1] * 2)

    @staticmethod
    def _decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
        raw = scale.contiguous().view(torch.uint8)
        exponent = raw.to(torch.int16).to(torch.float32) - 127.0
        decoded = torch.pow(torch.tensor(2.0), exponent)
        return torch.where(raw == 255, torch.full_like(decoded, float("nan")), decoded)

    @classmethod
    def _decode_q(cls, q: torch.Tensor, descale_q: torch.Tensor) -> torch.Tensor:
        values = cls._unpack_mxfp4(q)
        logical_d = int(values.shape[-1])
        expected = (*q.shape[:2], (logical_d + 63) // 64, 2)
        if tuple(descale_q.shape) != expected:
            raise ValueError("descale_q must have shape (T1,N1,D/64,2)")
        groups = (logical_d + cls.GROUP_SIZE - 1) // cls.GROUP_SIZE
        scale = cls._decode_e8m0(descale_q).reshape(*q.shape[:2], -1)
        scale = scale[..., :groups].repeat_interleave(cls.GROUP_SIZE, dim=-1)[
            ..., :logical_d
        ]
        return values * scale

    @classmethod
    def _decode_k(cls, k: torch.Tensor, descale_k: torch.Tensor) -> torch.Tensor:
        values = cls._unpack_mxfp4(k)
        scale = cls._decode_e8m0(descale_k).reshape(*k.shape[:-1], 4)
        scale = scale.repeat_interleave(cls.GROUP_SIZE, dim=-1)
        return values * scale

    @staticmethod
    def _round_bf16(value: torch.Tensor) -> torch.Tensor:
        bits = value.contiguous().view(torch.int32)
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        mask = torch.tensor(-65536, dtype=torch.int32, device=value.device)
        return (rounded & mask).view(torch.float32)

    @classmethod
    def _weighted_reduce(
        cls, qk_relu: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Match the BF16 destination boundary after every ASC G update."""
        qk = cls._round_bf16(qk_relu)
        weight = cls._round_bf16(weights.to(torch.float32))
        score = torch.zeros(
            (qk.shape[0], qk.shape[3], qk.shape[2]), dtype=torch.float32
        )
        for group in range(cls.GROUP_SIZE):
            contribution = (
                weight[:, group :: cls.GROUP_SIZE]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(torch.float64)
            )
            product = qk[:, group :: cls.GROUP_SIZE].to(torch.float64)
            update = (contribution * product).sum(dim=1).permute(0, 2, 1)
            score = cls._round_bf16(
                (score.to(torch.float64) + update).to(torch.float32)
            )
        return score

    def _page_plan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        cmp_residual_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
    ) -> tuple[list[int], list[int], list[int], list[list[int]], list[int]]:
        total_q = int(q.shape[0])
        if cu_seqlens_q is None:
            cu = [0, total_q]
        else:
            cu = [int(x) for x in cu_seqlens_q.reshape(-1).tolist()]
        if cu[0] != 0 or cu[-1] != total_q:
            raise ValueError("cu_seqlens_q must cover T1")
        q_lengths = [end - begin for begin, end in zip(cu, cu[1:])]
        if any(length < 0 for length in q_lengths):
            raise ValueError("cu_seqlens_q must be nondecreasing")
        used_q = (
            q_lengths
            if seqused_q is None
            else [int(x) for x in seqused_q.reshape(-1).tolist()]
        )
        if len(used_q) != len(q_lengths) or any(
            value < 0 or value > q_lengths[index] for index, value in enumerate(used_q)
        ):
            raise ValueError("invalid seqused_q")
        if self.max_seqlen_q != -1 and max(q_lengths, default=0) > self.max_seqlen_q:
            raise ValueError("max_seqlen_q does not cover every sequence")

        batch = len(q_lengths)
        if self.layout_k == "TND":
            if cu_seqlens_k is None:
                raise ValueError("TND K requires cu_seqlens_k")
            cu_k = [int(x) for x in cu_seqlens_k.reshape(-1).tolist()]
            if len(cu_k) != batch + 1:
                raise ValueError("cu_seqlens_k must have B+1 entries")
            pages = [[cu_k[index]] for index in range(batch)]
            capacity = [cu_k[index + 1] - cu_k[index] for index in range(batch)]
        elif block_table is None:
            pages = [list(range(int(k.shape[0]))) for _ in range(batch)]
        else:
            if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
                raise ValueError("block_table must have shape (B,max_blocks)")
            pages = []
            for row in block_table.tolist():
                valid = [int(page) for page in row if int(page) != -1]
                if not valid:
                    raise ValueError("every batch requires a physical K page")
                pages.append(valid)
        if self.layout_k != "TND":
            capacity = [len(row) * int(k.shape[1]) for row in pages]
        used_k = (
            capacity
            if seqused_k is None
            else [int(x) for x in seqused_k.reshape(-1).tolist()]
        )
        if len(used_k) != batch or any(
            value < 0 or value > capacity[index] for index, value in enumerate(used_k)
        ):
            raise ValueError("invalid seqused_k")

        if self.mask_mode == 3 and self.cmp_ratio != 1:
            if cmp_residual_k is None:
                raise ValueError("cmp_residual_k is required")
            residual = [int(x) for x in cmp_residual_k.reshape(-1).tolist()]
        else:
            if cmp_residual_k is not None:
                raise ValueError("cmp_residual_k is not used by this mode")
            residual = [0] * batch
        return cu, used_q, used_k, pages, residual

    @classmethod
    def _gather_k(
        cls,
        decoded_k: torch.Tensor,
        pages: list[int],
        logical_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if decoded_k.ndim == 3:
            return decoded_k[pages[0] + logical_tokens, 0]
        logical_page = torch.div(
            logical_tokens, int(decoded_k.shape[1]), rounding_mode="floor"
        )
        page_offset = logical_tokens % int(decoded_k.shape[1])
        page_tensor = torch.tensor(pages, dtype=torch.long)
        physical_page = page_tensor.index_select(0, logical_page)
        return decoded_k[physical_page, page_offset, 0]

    def _score_row(
        self,
        decoded_query: torch.Tensor,
        decoded_k: torch.Tensor,
        weight: torch.Tensor,
        blocks: torch.Tensor,
        pages: list[int],
        used_k: int,
        used_q: int,
        local_q: int,
        residual: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_in_block = torch.arange(self.CANDIDATE_BLOCK_SIZE)
        logical_tokens = (
            blocks.to(torch.long).unsqueeze(1) * self.CANDIDATE_BLOCK_SIZE
            + token_in_block
        ).reshape(-1)
        logical_tokens = logical_tokens[
            (logical_tokens >= 0) & (logical_tokens < used_k)
        ]
        if logical_tokens.numel() == 0:
            return logical_tokens, torch.empty(0, dtype=torch.float32)
        gathered_k = self._gather_k(decoded_k, pages, logical_tokens)
        qk = (
            torch.einsum("nd,smd->nsm", decoded_query, gathered_k.unsqueeze(1))
            .unsqueeze(0)
            .clamp_min_(0)
        )
        score = self._weighted_reduce(qk, weight.unsqueeze(0))[0, 0]
        if self.mask_mode == 3:
            original_k = used_k * self.cmp_ratio + residual
            boundary = (original_k - used_q + local_q + 1) // self.cmp_ratio
            causal_valid = logical_tokens < boundary
            logical_tokens = logical_tokens[causal_valid]
            score = score[causal_valid]
        return logical_tokens, score

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        descale_q: torch.Tensor,
        descale_k: torch.Tensor,
        candidate_block_indices: torch.Tensor,
        candidate_block_length: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        cmp_residual_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        output_idx_offset: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if tuple(q.shape[1:]) != (32, 64):
            raise ValueError("q must have shape (T1,32,64)")
        expected_k = (1, 64) if self.layout_k == "TND" else (k.shape[1], 1, 64)
        if tuple(k.shape[1:]) != expected_k:
            raise ValueError(f"k must have inner shape {expected_k}")
        if candidate_block_indices.shape[-1] != self.CANDIDATE_CAPACITY:
            raise ValueError("candidate capacity must be 2048")
        cu, used_q, used_k, pages, residual = self._page_plan(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
        )
        decoded_q = self._decode_q(q, descale_q)
        decoded_k = self._decode_k(k, descale_k)
        candidate_length = candidate_block_length.reshape(-1).tolist()
        offsets = (
            torch.zeros((q.shape[0], 1), dtype=torch.int32)
            if output_idx_offset is None
            else output_idx_offset.reshape(q.shape[0], 1).to(torch.int32)
        )

        sparse_indices = torch.full((q.shape[0], 1, self.topk), -1, dtype=torch.int32)
        sparse_values = torch.zeros((q.shape[0], 1, self.topk), dtype=torch.bfloat16)
        max_q = max((end - begin for begin, end in zip(cu, cu[1:])), default=0)
        max_k = max(used_k, default=0)
        topk_value = torch.full(
            (len(used_q), 1, max_q, max_k), -float("inf"), dtype=torch.float32
        )

        for batch, begin in enumerate(cu[:-1]):
            for local_q in range(used_q[batch]):
                row = begin + local_q
                block_count = int(candidate_length[row])
                blocks = candidate_block_indices[row, 0, :block_count].to(torch.long)
                logical_tokens, score = self._score_row(
                    decoded_q[row],
                    decoded_k,
                    w[row],
                    blocks,
                    pages[batch],
                    used_k[batch],
                    used_q[batch],
                    local_q,
                    residual[batch],
                )
                if logical_tokens.numel() == 0:
                    continue
                topk_value[batch, 0, local_q, logical_tokens] = score

                valid_topk = min(self.topk, int(score.numel()))
                values, compact_pos = torch.topk(
                    score, valid_topk, dim=-1, largest=True, sorted=False
                )
                selected = logical_tokens.index_select(0, compact_pos)
                sparse_indices[row, 0, :valid_topk] = (
                    selected.to(torch.int32) + offsets[row, 0]
                )
                sparse_values[row, 0, :valid_topk] = values.to(torch.bfloat16)

        return {
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "_topk_value": topk_value,
            "_cu_seqlens_q": torch.tensor(cu, dtype=torch.int32),
            "_seqused_k": torch.tensor(used_k, dtype=torch.int32),
            "_output_idx_offset": offsets,
        }

    def forward_selected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        descale_q: torch.Tensor,
        descale_k: torch.Tensor,
        candidate_block_indices: torch.Tensor,
        candidate_block_length: torch.Tensor,
        selected_rows: list[int],
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        cmp_residual_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        output_idx_offset: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate selected Prefill rows while preserving full TND positions."""
        cu, used_q, used_k, pages, residual = self._page_plan(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
        )
        selected = torch.tensor(selected_rows, dtype=torch.long)
        decoded_q = self._decode_q(
            q.index_select(0, selected), descale_q.index_select(0, selected)
        )
        decoded_k = self._decode_k(k, descale_k)
        offsets = (
            torch.zeros((q.shape[0], 1), dtype=torch.int32)
            if output_idx_offset is None
            else output_idx_offset.reshape(q.shape[0], 1).to(torch.int32)
        )
        result_indices = torch.full(
            (len(selected_rows), 1, self.topk), -1, dtype=torch.int32
        )
        result_values = torch.zeros(
            (len(selected_rows), 1, self.topk), dtype=torch.bfloat16
        )
        topk_value = torch.full(
            (len(selected_rows), 1, 1, max(used_k)),
            -float("inf"),
            dtype=torch.float32,
        )
        selected_used_k = []
        for output_row, query_row in enumerate(selected_rows):
            batch = next(
                index
                for index in range(len(used_q))
                if cu[index] <= query_row < cu[index + 1]
            )
            local_q = query_row - cu[batch]
            selected_used_k.append(used_k[batch])
            if local_q >= used_q[batch]:
                continue
            block_count = int(candidate_block_length[query_row, 0])
            blocks = candidate_block_indices[query_row, 0, :block_count]
            logical_tokens, score = self._score_row(
                decoded_q[output_row],
                decoded_k,
                w[query_row],
                blocks,
                pages[batch],
                used_k[batch],
                used_q[batch],
                local_q,
                residual[batch],
            )
            if logical_tokens.numel() == 0:
                continue
            topk_value[output_row, 0, 0, logical_tokens] = score
            valid_topk = min(self.topk, int(score.numel()))
            values, compact_pos = torch.topk(
                score, valid_topk, dim=-1, largest=True, sorted=False
            )
            selected_tokens = logical_tokens.index_select(0, compact_pos)
            result_indices[output_row, 0, :valid_topk] = (
                selected_tokens.to(torch.int32) + offsets[query_row, 0]
            )
            result_values[output_row, 0, :valid_topk] = values.to(torch.bfloat16)
        return {
            "sparse_indices": result_indices,
            "sparse_values": result_values,
            "_topk_value": topk_value,
            "_cu_seqlens_q": torch.arange(len(selected_rows) + 1, dtype=torch.int32),
            "_seqused_k": torch.tensor(selected_used_k, dtype=torch.int32),
            "_output_idx_offset": offsets.index_select(0, selected),
        }


def pack_candidate_k(k: torch.Tensor, descale_k: torch.Tensor) -> torch.Tensor:
    """Fixture-only packing: each eight-token block is K[512] + scale[32]."""
    pages, page_size = k.shape[:2]
    return torch.cat(
        (
            k.view(torch.uint8).reshape(pages, page_size // 8, 512),
            descale_k.view(torch.uint8).reshape(pages, page_size // 8, 32),
        ),
        dim=-1,
    )


def unpack_candidate_k(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the public packed layout independently on CPU for golden."""
    if packed.dtype != torch.uint8 or packed.ndim != 3 or packed.shape[-1] != 544:
        raise ValueError("packed K must be uint8 [pages, PA/8, 544]")
    pages, blocks = packed.shape[:2]
    return (
        packed[..., :512].reshape(pages, blocks * 8, 1, 64),
        packed[..., 512:].reshape(pages, blocks * 8, 1, 2, 2),
    )


def quant_sparse_lightning_indexer_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    descale_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
    return_value: bool = False,
) -> dict[str, torch.Tensor]:
    if metadata is not None:
        raise ValueError("metadata is reserved and must be None")
    if layout_k == "PA_BBND":
        k, descale_k = unpack_candidate_k(k)
    elif descale_k is None:
        raise ValueError("TND K requires descale_k")
    golden = GeneralizedQSLI(
        topk=topk,
        candidate_block_size=candidate_block_size,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
    )
    return golden.forward(
        q,
        k,
        w,
        descale_q,
        descale_k,
        candidate_block_indices,
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
    )


def quant_sparse_lightning_indexer_golden_selected(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    selected_rows: list[int],
    *,
    cu_seqlens_q: torch.Tensor,
    seqused_q: torch.Tensor,
    seqused_k: torch.Tensor,
    cmp_residual_k: Optional[torch.Tensor],
    block_table: torch.Tensor,
    output_idx_offset: torch.Tensor,
    descale_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
    return_value: bool = False,
) -> dict[str, torch.Tensor]:
    """Prefill golden for selected rows, retaining their full-sequence mask."""
    if layout_k == "PA_BBND":
        k, descale_k = unpack_candidate_k(k)
    elif descale_k is None:
        raise ValueError("TND K requires descale_k")
    golden = GeneralizedQSLI(
        topk=topk,
        candidate_block_size=candidate_block_size,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
    )
    return golden.forward_selected(
        q,
        k,
        w,
        descale_q,
        descale_k,
        candidate_block_indices,
        candidate_block_length,
        selected_rows,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
    )
