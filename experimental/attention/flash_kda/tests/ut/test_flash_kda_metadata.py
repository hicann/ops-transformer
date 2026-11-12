# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone scheduling and ABI tests for FlashKDA training-forward metadata."""

from types import SimpleNamespace

import pytest
import torch
from experimental.attention.flash_kda.op_dsl import (
    flash_kda_metadata as metadata_module,
)


def _prefix_sum(lengths):
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    return prefix


def _assert_partition(ranges, task_num):
    cursor = 0
    unused = False
    for start, end in ranges:
        if (start, end) == (0, 0):
            unused = True
            continue
        assert not unused
        assert start == cursor
        assert start < end <= task_num
        cursor = end
    assert cursor == task_num


def _assert_schedule(decoded, lengths, value_heads, packed):
    assert decoded["status"] == metadata_module.STATUS_OK
    assert decoded["batch"] == len(lengths)
    assert decoded["value_heads"] == value_heads
    assert decoded["workspace_slots"] == metadata_module.WORKSPACE_SLOTS
    expected_chunks = [(length + 63) // 64 for length in lengths]
    next_chunk = [0] * len(lengths)
    starts = _prefix_sum(lengths)[:-1] if packed else [0] * len(lengths)
    for round_index, record in enumerate(decoded["stage12_rounds"]):
        assert record["stage12_round_idx"] == round_index
        active = [
            batch
            for batch, chunks in enumerate(expected_chunks)
            if next_chunk[batch] < chunks
        ]
        assert record["batch_idx"] == active
        assert record["active_batch_num"] == len(active)
        assert record["storage_batch_idx"] == ([0] * len(active) if packed else active)
        assert record["token_start"] == [starts[batch] for batch in active]
        assert record["valid_seq_len"] == [lengths[batch] for batch in active]
        assert record["stage1_task_prefix"][0] == 0
        assert record["stage2_task_prefix"][0] == 0
        for entry, batch in enumerate(active):
            chunk_start = record["chunk_start_per_group"][entry]
            chunk_num = record["chunk_num_per_group"][entry]
            split_num = record["dv_splits_num"][entry]
            assert chunk_start == next_chunk[batch]
            assert 0 < chunk_num <= record["group_num"]
            assert chunk_start + chunk_num <= expected_chunks[batch]
            assert split_num in (1, 2, 4, 8)
            assert (
                record["stage1_task_prefix"][entry + 1]
                - record["stage1_task_prefix"][entry]
            ) == value_heads * chunk_num
            assert (
                record["stage2_task_prefix"][entry + 1]
                - record["stage2_task_prefix"][entry]
            ) == value_heads * split_num
            next_chunk[batch] += chunk_num
        assert record["stage1_task_prefix"][-1] <= metadata_module.WORKSPACE_SLOTS
        _assert_partition(
            record["stage1_core_ranges"], record["stage1_task_prefix"][-1]
        )
        _assert_partition(
            record["stage2_core_ranges"], record["stage2_task_prefix"][-1]
        )
        for stage in (1, 2):
            unused_ranges = record[f"stage{stage}_core_ranges"][decoded["core_num"] :]
            assert all(pair == (0, 0) for pair in unused_ranges)
    assert next_chunk == expected_chunks


def test_metadata_capacity_and_record_words_cover_round_boundaries():
    assert metadata_module.round_record_words(1) == 142
    assert metadata_module.round_record_words(3) == 160
    assert (
        metadata_module.metadata_capacity_upper_bound(
            2, 65, is_packed=False, value_heads=1
        )
        == 584
    )
    assert (
        metadata_module.metadata_capacity_upper_bound(
            2, 65, is_packed=True, value_heads=1
        )
        == 441
    )
    assert metadata_module.metadata_capacity_upper_bound(2, 65, is_packed=False) == 1197
    with pytest.raises(ValueError, match="positive"):
        metadata_module.round_record_words(0)


def test_decode_metadata_does_not_require_compact_tnd_chunk_offsets():
    lengths = (63, 937, 65)
    record_start = metadata_module.HEADER_WORDS + 2
    record_end = record_start + metadata_module.round_record_words(1)
    words = [0] * record_end
    words[metadata_module.HEADER_MAGIC] = metadata_module.MAGIC
    words[metadata_module.HEADER_ABI_VERSION] = metadata_module.ABI_VERSION
    words[metadata_module.HEADER_STATUS] = metadata_module.STATUS_OK
    words[metadata_module.HEADER_TEMPLATE] = metadata_module.TEMPLATE_STAGED
    words[metadata_module.HEADER_CORE_NUM] = 1
    words[metadata_module.HEADER_STAGE12_ROUND_NUM] = 1
    words[metadata_module.HEADER_BATCH] = len(lengths)
    words[metadata_module.HEADER_VALUE_HEADS] = 1
    words[metadata_module.HEADER_HAS_CU] = 1
    words[metadata_module.HEADER_WORKSPACE_SLOTS] = metadata_module.WORKSPACE_SLOTS
    words[metadata_module.HEADER_WORDS] = record_start
    words[metadata_module.HEADER_WORDS + 1] = record_end
    words[record_start] = 0
    words[record_start + 1] = 1
    cursor = record_start + 2
    ranges = [(0, 1)] + [(0, 0)] * (metadata_module.MAX_AIC_CORES - 1)
    for values in (
        [0],
        [0],
        [0],
        [63],
        [1],
        [0],
        [1],
        [0, 1],
        ranges,
        [1],
        [0, 1],
        ranges,
    ):
        if values and isinstance(values[0], tuple):
            for start, end in values:
                words[cursor : cursor + 2] = [start, end]
                cursor += 2
        else:
            words[cursor : cursor + len(values)] = values
            cursor += len(values)
    decoded = metadata_module.decode_metadata(words)
    assert "cu_chunk_offsets" not in decoded
    assert decoded["used_words"] == record_end


def test_decode_metadata_rejects_invalid_header():
    with pytest.raises(ValueError, match="shorter"):
        metadata_module.decode_metadata([0])
    with pytest.raises(ValueError, match="magic"):
        metadata_module.decode_metadata([0] * metadata_module.HEADER_WORDS)


def test_device_core_count_uses_current_stream_and_caps_at_32(monkeypatch):
    stream = object()
    observed = {}

    def current_stream(device_index):
        observed["device_index"] = device_index
        return stream

    def platform_info(*, stream):
        observed["stream"] = stream
        return SimpleNamespace(cube_core_num=48)

    monkeypatch.setattr(
        metadata_module.torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0, current_stream=current_stream),
        raising=False,
    )
    monkeypatch.setattr(metadata_module, "get_platform_info", platform_info)
    ref = SimpleNamespace(device=SimpleNamespace(type="npu", index=2))
    assert metadata_module._device_block_num(ref) == 32
    assert observed == {"device_index": 2, "stream": stream}


def test_forward_metadata_rejects_invalid_public_inputs():
    attrs = dict(batch_size=1, nheads_qk=1, nheads_v=2, seqlens=65, layout="BNSD")
    for overrides, message in [
        ({"layout": "BAD"}, "layout expected BNSD/BSND/TND"),
        ({"batch_size": torch.tensor(1)}, "batch_size must be a positive integer"),
        ({"headdim_qk": 64}, "headdim_qk/headdim_v must be 128"),
        ({"nheads_qk": 2, "nheads_v": 3}, "Nv % Nqk"),
        ({"nheads_v": 97}, "Nv <= 96"),
        ({"cu_seqlens": torch.tensor([0, 65])}, "cu_seqlens dtype"),
        ({"cu_seqlens": torch.zeros((1, 2), dtype=torch.int32)}, "cu_seqlens rank"),
        ({"cu_seqlens": torch.zeros(3, dtype=torch.int32)}, "cu_seqlens shape"),
        ({"cu_seqlens": torch.zeros(4, dtype=torch.int32)[::2]}, "contiguous"),
        ({"layout": "TND"}, "required for TND"),
    ]:
        with pytest.raises(ValueError, match=message):
            metadata_module.flash_kda_metadata(**(attrs | overrides))
    del attrs["seqlens"]
    with pytest.raises(TypeError, match="seqlens"):
        metadata_module.flash_kda_metadata(**attrs)


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,lengths,value_heads,with_cu",
    [
        pytest.param("BNSD", (65, 3), 2, True, id="bnsd-tail"),
        pytest.param("BSND", (129, 129), 2, False, id="bsnd-dense"),
        pytest.param("TND", (1, 63, 65), 3, True, id="tnd-packed-tails"),
        pytest.param("BNSD", (8192,), 32, False, id="bnsd-short-group"),
        pytest.param("TND", (1, 63, 285 * 64 + 1), 3, True, id="tnd-multiround"),
    ],
)
def test_forward_metadata_schedule(layout, lengths, value_heads, with_cu):
    pytest.importorskip("torch_npu")
    batch = len(lengths)
    packed = layout == "TND"
    storage = sum(lengths) if packed else max(lengths)
    offsets = (
        torch.tensor(_prefix_sum(lengths), dtype=torch.int32, device="npu")
        if with_cu
        else None
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        metadata = metadata_module._build_flash_kda_metadata(
            batch_size=batch,
            nheads_qk=value_heads,
            nheads_v=value_heads,
            cu_seqlens=offsets,
            seqlens=storage,
            layout=layout,
            enable_grad=True,
            template="staged",
        )
    stream.synchronize()
    assert metadata.dtype == torch.int32 and metadata.is_contiguous()
    assert metadata.device == torch.device("npu", torch.npu.current_device())
    decoded = metadata_module.decode_metadata(metadata.cpu().tolist())
    assert decoded["template"] == "staged"
    assert 0 < decoded["used_words"] <= metadata.numel()
    _assert_schedule(decoded, lengths, value_heads, packed)
    if value_heads == 32 and metadata_module._device_block_num() == 32:
        assert [record["group_num"] for record in decoded["stage12_rounds"]] == [
            13
        ] * 9 + [11]
    if storage > 285 * 64:
        assert decoded["stage12_round_num"] > 1


_INVALID_LENGTHS = [
    ("TND", [1, 2], metadata_module.STATUS_BAD_CU_START, "must be 0"),
    ("TND", [0, 3], metadata_module.STATUS_BAD_PACKED_END, "cu_seqlens end expected"),
    ("BNSD", [0, 3], metadata_module.STATUS_BAD_PADDED_LENGTH, "exceeds S"),
    ("BNSD", [0, 0], metadata_module.STATUS_BAD_LENGTH, "must be positive"),
]


@pytest.mark.npu
@pytest.mark.parametrize("layout,offsets,status,message", _INVALID_LENGTHS)
def test_forward_metadata_reports_invalid_lengths_through_public_api(
    layout,
    offsets,
    status,
    message,
):
    pytest.importorskip("torch_npu")
    attrs = dict(batch_size=1, nheads_qk=1, nheads_v=1, seqlens=2, layout=layout)
    prefix = torch.tensor(offsets, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match=message):
        metadata_module.flash_kda_metadata(prefix, **attrs)
    valid_prefix = torch.tensor([0, 2], dtype=torch.int32, device="npu")
    valid_metadata = metadata_module.flash_kda_metadata(valid_prefix, **attrs)
    assert (
        metadata_module.decode_metadata(valid_metadata.cpu().tolist())["status"]
        == metadata_module.STATUS_OK
    )


@pytest.mark.npu
@pytest.mark.parametrize("layout,offsets,status,message", _INVALID_LENGTHS)
def test_forward_metadata_reports_invalid_lengths_without_faulting_stream(
    layout,
    offsets,
    status,
    message,
):
    pytest.importorskip("torch_npu")
    prefix = torch.tensor(offsets, dtype=torch.int32, device="npu")
    capacity = metadata_module.metadata_capacity_upper_bound(
        1,
        2,
        is_packed=layout == "TND",
        value_heads=1,
    )
    metadata = torch.empty(capacity, dtype=torch.int32, device="npu")
    # Exercise the device ABI directly as well as the public asynchronous entry.
    metadata_module._get_aicpu_kernel().launch(
        metadata_module.current_raw_stream(prefix.get_device()),
        cu_seqlens=prefix.data_ptr(),
        metadata=metadata.data_ptr(),
        metadata_word_capacity=capacity,
        batch=1,
        physical_batches=1,
        storage_length=2,
        value_heads=1,
        has_cu=1,
        is_packed=int(layout == "TND"),
        core_num=metadata_module._device_block_num(prefix),
        workspace_slots=metadata_module.WORKSPACE_SLOTS,
        max_chunk_group=8,
        template_kind=metadata_module.TEMPLATE_STAGED,
    )
    torch.npu.synchronize()
    assert metadata_module.decode_metadata(metadata.cpu().tolist())["status"] == status


def teardown_module():
    metadata_module.clear_metadata_caches()
