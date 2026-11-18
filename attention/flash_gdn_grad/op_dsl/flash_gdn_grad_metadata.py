# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""FlashGDN sequence schedule produced on AICPU for NPU inputs."""

import tempfile
import threading
from dataclasses import dataclass
from itertools import pairwise

import torch
from cannbotdsl.aicpu import I64, aicpu_kernel, current_raw_stream, gm_in, gm_out
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

_PADDED = 0
_LENGTHS = 1
_OFFSETS = 2
_OK = 0
_INVALID = 1
_HEADER_WORDS = 1
_ROW_WORDS = 2
_COMPILED = None
_COMPILE_LOCK = threading.Lock()


class _MetadataArgs:
    sequence: gm_in(I64)
    output: gm_out(I64)
    batch: I64
    tokens: I64
    mode: I64


@aicpu_kernel
def _metadata_kernel(a: _MetadataArgs):
    a.output[0] = _INVALID
    if a.mode == _OFFSETS:  # noqa: SIM102
        if a.sequence[0] != 0 or a.sequence[a.batch] != a.tokens:
            return _OK
    # Keep the explicit lower bound used by the AICPU DSL.
    for index in range(0, a.batch):  # noqa: PIE808
        start = 0
        end = a.tokens
        if a.mode == _OFFSETS:
            start = a.sequence[index]
            end = a.sequence[index + 1]
        if a.mode == _LENGTHS:
            end = a.sequence[index]
        if start < 0 or end < start or end > a.tokens:
            return _OK
        row = _HEADER_WORDS + index * _ROW_WORDS
        a.output[row] = start
        a.output[row + 1] = end - start
    a.output[0] = _OK
    return _OK


def _compiled_kernel():
    global _COMPILED
    with _COMPILE_LOCK:
        if _COMPILED is None:
            _COMPILED = compile_aicpu_kernel(
                _metadata_kernel,
                workdir=tempfile.mkdtemp(prefix="flash_gdn_grad_metadata_"),
                launch_mode="interface",
            )
        return _COMPILED


@dataclass(frozen=True)
class FlashGDNMetadata:
    layout: str
    storage: tuple
    lengths: tuple
    starts: tuple
    groups: tuple


def _assemble(layout, storage, lengths, starts):
    groups = {}
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append(index)
    return FlashGDNMetadata(
        layout,
        storage,
        lengths,
        starts,
        tuple((length, tuple(ids)) for length, ids in groups.items()),
    )


def _schedule_args(q, layout, cu_seqlens, seqused):
    sequence = cu_seqlens if layout == "TND" else seqused
    if sequence is not None and (
        not isinstance(sequence, torch.Tensor)
        or sequence.ndim != 1
        or sequence.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("sequence metadata must be a one-dimensional integer tensor")
    if layout == "TND":
        if (
            q.ndim != 3
            or seqused is not None
            or sequence is None
            or sequence.numel() < 2
        ):
            raise ValueError(
                "TND expects [T,H,K] and cu_seqlens with at least two offsets"
            )
        return sequence, _OFFSETS, sequence.numel() - 1, q.size(0), (q.size(0),)
    if layout not in ("BSND", "BNSD") or q.ndim != 4 or cu_seqlens is not None:
        raise ValueError("padded layout must be BSND or BNSD with optional seqused")
    batch, tokens = q.size(0), q.size(1 if layout == "BSND" else 2)
    if batch < 1 or (sequence is not None and sequence.numel() != batch):
        raise ValueError(
            "expected a positive batch and one seqused length per sequence"
        )
    return (
        sequence,
        _LENGTHS if sequence is not None else _PADDED,
        batch,
        tokens,
        (batch, tokens),
    )


def _device_rows(q, sequence, mode, batch, tokens):
    with torch.npu.device(q.device):
        stream = torch.npu.current_stream(q.device)
        if sequence is not None:
            sequence = sequence.to(device=q.device, dtype=torch.int64).contiguous()
            sequence.record_stream(stream)
        output = torch.empty(
            _HEADER_WORDS + batch * _ROW_WORDS, dtype=torch.int64, device=q.device
        )
        output.record_stream(stream)
        _compiled_kernel().launch(
            current_raw_stream(q.device.index),
            sequence=0 if sequence is None else sequence.data_ptr(),
            output=output.data_ptr(),
            batch=batch,
            tokens=tokens,
            mode=mode,
        )
        words = output.cpu().tolist()
    if words[0] != _OK:
        raise ValueError("invalid sequence lengths or packed offsets")
    return tuple(words[_HEADER_WORDS::_ROW_WORDS]), tuple(
        words[_HEADER_WORDS + 1 :: _ROW_WORDS]
    )


def _host_rows(sequence, mode, batch, tokens):
    if mode == _PADDED:
        return (0,) * batch, (tokens,) * batch
    values = tuple(sequence.tolist())
    if mode == _OFFSETS:
        if values[0] != 0 or values[-1] != tokens:
            raise ValueError("cu_seqlens must start at zero and end at T")
        lengths = tuple(end - start for start, end in pairwise(values))
        if min(lengths) < 0:
            raise ValueError("cu_seqlens must be nondecreasing")
        return values[:-1], lengths
    if min(values) < 0 or max(values) > tokens:
        raise ValueError("seqused must contain B lengths in [0,T]")
    return (0,) * batch, values


def flash_gdn_grad_metadata(q, *, layout="TND", cu_seqlens=None, seqused=None):
    """Build a reusable host schedule from the AICPU sequence table on NPU."""
    sequence, mode, batch, tokens, storage = _schedule_args(
        q, layout, cu_seqlens, seqused
    )
    if q.device.type == "npu":
        starts, lengths = _device_rows(q, sequence, mode, batch, tokens)
    else:
        starts, lengths = _host_rows(sequence, mode, batch, tokens)
    return _assemble(layout, storage, lengths, starts)


def resolve_metadata(q, layout, metadata, cu_seqlens, seqused):
    if metadata is None or cu_seqlens is not None or seqused is not None:
        actual = flash_gdn_grad_metadata(
            q, layout=layout, cu_seqlens=cu_seqlens, seqused=seqused
        )
        if metadata is not None and metadata != actual:
            raise ValueError("explicit lengths do not match metadata")
        return actual
    if not isinstance(metadata, FlashGDNMetadata) or metadata.layout != layout:
        raise ValueError("metadata layout does not match")
    if layout == "TND" and q.ndim == 3:
        offsets = (*metadata.starts, q.size(0))
        expected = tuple(b - a for a, b in pairwise(offsets))
        valid = (
            metadata.storage == (q.size(0),)
            and bool(metadata.starts)
            and metadata.starts[0] == 0
            and expected == metadata.lengths
            and min(expected) >= 0
        )
    elif layout in ("BSND", "BNSD") and q.ndim == 4:
        batch, tokens = q.size(0), q.size(1 if layout == "BSND" else 2)
        valid = (
            metadata.storage == (batch, tokens)
            and batch > 0
            and len(metadata.lengths) == batch
            and metadata.starts == (0,) * batch
            and min(metadata.lengths) >= 0
            and max(metadata.lengths) <= tokens
        )
    else:
        valid = False
    if not valid or metadata != _assemble(
        layout, metadata.storage, metadata.lengths, metadata.starts
    ):
        raise ValueError("metadata shape or schedule does not match q")
    return metadata
