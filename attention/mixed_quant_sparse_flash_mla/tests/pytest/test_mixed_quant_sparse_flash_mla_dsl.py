# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch/DSL regression with library metadata and an independent CPU golden."""

from importlib import import_module
import logging
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

OPS = import_module(os.environ.get("MQSMLA_PACKAGE", "cann_ops_transformer"))


_BASE = dict(
    q_lengths=[1, 1],
    s_ori=257,
    s_cmp=193,
    k_ori=128,
    k_cmp=129,
    block_size_ori=128,
    block_size_cmp=128,
    has_cmp=True,
    kv_axis0_noncontiguous=False,
    return_softmax_lse=True,
)
TEST_PARAMS = {
    "mixed_decode": dict(_BASE),
    "library_fd_split": dict(
        _BASE, q_lengths=[1], s_ori=4096, s_cmp=4096, k_ori=4096, k_cmp=4096
    ),
    "mixed_pair_gaps": dict(_BASE, mixed_pair_gaps=True),
    "ori_only": dict(_BASE, has_cmp=False, return_softmax_lse=False),
    "empty_and_tail": dict(_BASE, q_lengths=[3, 0, 5], vary_lengths=True),
    # Entire padding subtiles must reuse valid addresses, including when the
    # address table contains only one padded 16-column block.
    "single_token_k": dict(_BASE, q_lengths=[4], k_ori=1, k_cmp=1, vary_lengths=True),
    "narrow_k_tail": dict(_BASE, q_lengths=[4], k_ori=17, k_cmp=65, vary_lengths=True),
    "padded_pages": dict(
        _BASE,
        q_lengths=[2, 3],
        kv_axis0_noncontiguous=True,
        block_size_ori=64,
        block_size_cmp=32,
    ),
    "cmp_only_rows": dict(_BASE, q_lengths=[3], zero_ori=True, vary_lengths=True),
    "omit_ori_length": dict(_BASE, omit_ori_topk_length=True),
    "omit_cmp_length": dict(_BASE, omit_cmp_topk_length=True),
    "omit_both_lengths": dict(
        _BASE, omit_ori_topk_length=True, omit_cmp_topk_length=True
    ),
    "multirow_42": dict(_BASE, q_lengths=[42], vary_lengths=True),
    "multirow_64": dict(_BASE, q_lengths=[64], vary_lengths=True),
    "multi_batch_72": dict(_BASE, q_lengths=[6] * 12, s_cmp=769, k_cmp=512),
}
ENABLED_PARAMS = [
    "mixed_decode",
    "mixed_pair_gaps",
    "ori_only",
    "empty_and_tail",
    "padded_pages",
]

LOGGER = logging.getLogger(__name__)

D, N1 = 512, 64
KV_ROW_BYTES_ORI, KV_ROW_BYTES_CMP, KV_PAGE_PADDING_BYTES = 544, 320, 64


@pytest.fixture(scope="module", autouse=True)
def _cpu_reference_threads():
    # Tiny per-row matmuls are slower with the server's large default pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def _selected_cases():
    selection = os.environ.get("MQSMLA_CASES", "").strip()
    names = (
        list(TEST_PARAMS)
        if selection == "all"
        else [name.strip() for name in selection.split(",")]
        if selection
        else ENABLED_PARAMS
    )
    unknown = set(names) - TEST_PARAMS.keys()
    if unknown:
        raise ValueError(f"Unknown MQSMLA_CASES: {sorted(unknown)}")
    return names


def _make_kv(batch, tokens, block_size, fp4, generator):
    """Encode logical KV, then shuffle physical pages independently of the golden.

    Features are nope[448], rope[64], followed by BF16 group scales.
    FP4 stores the earlier feature in the low nibble. The CPU reference keeps
    dequantized logical rows, so it never shares the kernel's page addressing.
    """
    pages = (tokens + block_size - 1) // block_size
    shape = (batch, pages * block_size, D)
    group = 16 if fp4 else 32
    scales = (
        0.125 + torch.rand((*shape[:-1], D // group), generator=generator) * 0.375
    ).bfloat16()
    if fp4:
        codes = torch.randint(0, 16, shape, generator=generator, dtype=torch.uint8)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        values = lut[codes.long()]
        encoded = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        values = (torch.randn(shape, generator=generator) * 2).to(torch.float8_e4m3fn)
        encoded = values.view(torch.uint8)
        values = values.float()
    logical = (values * scales.float().repeat_interleave(group, dim=-1)).bfloat16()
    packed = torch.cat((encoded, scales.view(torch.uint8)), dim=-1)
    table = (
        torch.randperm(batch * pages, generator=generator).reshape(batch, pages).int()
    )
    physical = torch.empty_like(packed.reshape(batch * pages, block_size, -1))
    physical[table.long().flatten()] = packed.reshape_as(physical)
    return {
        "phys": {"row": physical},
        "num_phys_blocks": batch * pages,
        "block_size": block_size,
        "block_table": table,
    }, logical


def gen_data(params):
    """Generate a bounded deterministic case without files or external test code."""
    p = dict(params)
    heads = p.get("heads", N1)
    generator = torch.Generator().manual_seed(p.get("seed", 42))
    spans = p["q_lengths"]
    batch, rows = len(spans), sum(spans)
    has_cmp = p.get("has_cmp", True)
    q = (torch.randn((rows, heads, D), generator=generator) * 0.25).bfloat16()
    sinks = (
        torch.full((heads,), float("-inf"))
        if p.get("omit_sinks")
        else torch.randn(heads, generator=generator)
    )
    cu = torch.tensor([0, *np.cumsum(spans).tolist()], dtype=torch.int32)
    case = {
        "q": q,
        "sinks": sinks,
        "cu_seqlens_q": cu,
        "seqused_q": torch.tensor(spans, dtype=torch.int32),
        "b": batch,
        "t1": rows,
        "quant_mode": 4,
        "mode": "ORI_CMP_SPARSE" if has_cmp else "ORI_SPARSE",
        "scale": D**-0.5,
    }
    logical = {}
    for side, fp4 in (("ori", False), ("cmp", True)):
        width = p[f"k_{side}"]
        pa, logical[side] = _make_kv(
            batch, p[f"s_{side}"], p[f"block_size_{side}"], fp4, generator
        )
        case["pa" if side == "ori" else "pa_cmp"] = pa
        indices = torch.randint(
            p[f"s_{side}"], (rows, width), generator=generator, dtype=torch.int32
        )
        if p.get("mixed_pair_gaps", False):
            # One duplicate pair among seven distinct pairs in every full
            # subtile exercises local fallback without disabling its neighbors.
            for start in range(0, width - 15, 16):
                indices[:, start : start + 16] = torch.arange(16, dtype=torch.int32)
                indices[:, start + 7] = indices[:, start + 6]
        lengths = torch.full((rows,), width, dtype=torch.int32)
        if p.get("vary_lengths", False):
            lengths = torch.tensor(
                ([0, 1, min(127, width), width] * ((rows + 3) // 4))[:rows],
                dtype=torch.int32,
            )
        if (side == "cmp" and not has_cmp) or (
            side == "ori" and p.get("zero_ori", False)
        ):
            lengths.zero_()
        if p.get(f"omit_{side}_topk_length", False):
            lengths.fill_(width)
        # Invalid unused indices catch reads beyond each valid prefix.
        indices[torch.arange(width)[None, :] >= lengths[:, None]] = -1
        case[f"idx_{side}"] = indices
        case[f"len_{side}"] = lengths
    output = torch.zeros_like(q)
    lse = torch.empty((1, rows, heads), dtype=torch.float32)
    for b in range(batch):
        for row in range(int(cu[b]), int(cu[b + 1])):
            sides = [
                logical[side][
                    b, case[f"idx_{side}"][row, : int(case[f"len_{side}"][row])].long()
                ]
                for side in ("ori", "cmp")
            ]
            # Dense FP64 logsumexp independently checks LSE and sink semantics.
            kv = torch.cat(sides).double()
            logits = q[row].double() @ kv.T * case["scale"]
            all_logits = torch.cat((logits, sinks.double()[:, None]), dim=1)
            lse[0, row] = torch.logsumexp(all_logits, dim=1).float()
            # Match sparse_flash_mla's CPU reference arithmetic: online softmax
            # over 128-token tiles, BF16 probabilities for PV, FP32 accumulation.
            # Each side starts a new tile sequence, including its own tail.
            row_max = sinks.clone()
            row_sum = torch.ones(heads)
            accumulator = torch.zeros(heads, D)
            for values in sides:
                for start in range(0, len(values), 128):
                    tile = values[start : start + 128].float()
                    scores = q[row].float() @ tile.T * case["scale"]
                    new_max = torch.maximum(row_max, scores.max(dim=1).values)
                    correction = torch.exp(row_max - new_max)
                    weights = torch.exp(scores - new_max[:, None])
                    row_sum = row_sum * correction + weights.sum(dim=1)
                    accumulator = (
                        accumulator * correction[:, None]
                        + weights.bfloat16().float() @ tile
                    )
                    row_max = new_max
            output[row] = (accumulator / row_sum[:, None]).bfloat16()
    return p, case, output.reshape(-1, D), lse


def check_result(expect, npu_result, chunk_elements=1 << 20):
    """与原BF16比较规则一致，跨块累计全局合格率及前10000000个错误的最大相对误差。"""
    expected = expect.reshape(-1)
    actual = npu_result.reshape(-1)
    total = actual.numel()
    if total != expected.numel():
        LOGGER.error("Error, the sizes of NPU output and benchmark differ")
        return "Failed", 0.0
    errors = 0
    sampled_errors = 0
    max_error = 0.0
    for start in range(0, total, chunk_elements):
        real = actual[start : start + chunk_elements].cpu().float().numpy()
        golden = expected[start : start + chunk_elements].cpu().float().numpy()
        close = np.isclose(real, golden, rtol=0.0078125, atol=0.0001, equal_nan=True)
        bad = np.flatnonzero(~close)
        errors += bad.size
        if bad.size and sampled_errors < 10000000:
            take = min(bad.size, 10000000 - sampled_errors)
            # 保持原比较器的float32运算顺序与max规则，包括NaN行为。
            diff_abs = abs(golden - real)
            b1 = np.maximum(np.abs(real), np.abs(golden))
            b2 = float((1.0 / (1 << 14)) / 0.005)
            b = np.add(np.maximum(b1, b2), 10e-10)
            relative = (diff_abs / (b + 10e-10))[bad[:take]]
            for value in relative:
                if sampled_errors == 0 or value > max_error:
                    max_error = value
                sampled_errors += 1
    percent = float(total - errors) / float(total) * 100.0
    result = "Pass" if percent >= (1 - 0.005) * 100.0 else "Failed"
    if sampled_errors and max_error >= 10:
        result = "Failed"
    LOGGER.info(
        "BF16 chunked compare: count=%s, errors=%s, pass=%.6f%%, "
        "rtol=0.0078125, atol=0.0001, max_relative=%s, result=%s",
        total,
        errors,
        percent,
        max_error,
        result,
    )
    return result, percent


def _pa_tensor(pa, row_bytes, *, axis0_noncontiguous=False):
    """先搬完整含 padding 的 storage，再在 NPU 上切视图，保留真实页距。"""
    row = pa["phys"]["row"]
    shape = (pa["num_phys_blocks"], pa["block_size"], 1, row_bytes)
    if not axis0_noncontiguous:
        return row.reshape(shape).npu()
    page_bytes = pa["block_size"] * row_bytes
    stride0 = page_bytes + KV_PAGE_PADDING_BYTES
    strides = (stride0, row_bytes, row_bytes, 1)
    # 0xA5 毒化页间无用数据；golden 保留原始逻辑 KV，不读取这份 storage。
    backing = torch.full((shape[0] * stride0,), 0xA5, dtype=torch.uint8)
    cpu_view = backing.as_strided(shape, strides)
    cpu_view.copy_(row.reshape(shape))
    result = backing.npu().as_strided(shape, strides)
    assert result.stride() == strides and result.storage_offset() == 0
    if shape[0] > 1:
        assert not result.is_contiguous()
    return result


def _library_metadata(kwargs, api=OPS):
    """Use the existing metadata operator without changing its mode contract."""
    has_cmp = kwargs.get("cmp_kv") is not None
    names = (
        "cu_seqlens_q",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "layout_q",
        "layout_kv",
    )
    metadata_args = {name: kwargs[name] for name in names if name in kwargs}
    metadata_args["batch_size"] = kwargs["cu_seqlens_q"].numel() - 1
    if kwargs["layout_q"] == "BSND":
        metadata_args.update(
            batch_size=kwargs["q"].shape[0], max_seqlen_q=kwargs["q"].shape[1]
        )
    for side in ("ori", "cmp") if has_cmp else ("ori",):
        indices = kwargs[f"{side}_sparse_indices"]
        topk = indices.shape[-1]
        lengths = kwargs.get(f"{side}_topk_length")
        if lengths is None:
            lengths = torch.full(
                indices.shape[:-1], topk, dtype=torch.int32, device=indices.device
            )
        metadata_args[f"{side}_topk"] = topk
        metadata_args[f"{side}_topk_length"] = lengths
        if kwargs["layout_kv"] == "BSND":
            metadata_args[f"max_seqlen_{side}_kv"] = kwargs[f"{side}_kv"].shape[1]
    # The library metadata API supports modes 1/2/3. Mode 1 supplies the
    # shared FA/FD scheduling format; only attention dispatch uses mode 4.
    return api.mixed_quant_sparse_flash_mla_metadata(
        kwargs["q"].shape[-2], 1, D, 1, has_cmp_kv=has_cmp, **metadata_args
    )


def _call_public(
    kwargs,
    cu_q_cpu,
    *,
    omit_ori_length=False,
    omit_cmp_length=False,
    api=OPS,
):
    """Exercise the public metadata producer and the mode-4 attention route."""
    kwargs = dict(kwargs)
    if omit_ori_length:
        kwargs.pop("ori_topk_length")
    if omit_cmp_length:
        kwargs.pop("cmp_topk_length")
    kwargs["metadata"] = _library_metadata(kwargs, api)
    assert kwargs["metadata"].shape == (2048,)
    frontend = import_module(
        f"{api.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder,
        "load",
        side_effect=AssertionError(
            "Expected DSL routing, but the ASC builder was loaded"
        ),
    ):
        out, lse = api.mixed_quant_sparse_flash_mla(**kwargs)
    output, lse = out.cpu().reshape(-1, D), lse.cpu()
    assert torch.isfinite(output).all()
    if kwargs.get("return_softmax_lse", False):
        assert torch.isfinite(lse).all()
    else:
        assert lse.shape == (0,) and lse.dtype == torch.float32
    return output, lse


def run_prepared(
    p,
    case,
    expected,
    expected_lse,
    api=OPS,
):
    """Execute locally generated inputs against the independent CPU reference."""
    mode = case["mode"]
    quant_mode = case["quant_mode"]
    pa_ori = case["pa"]
    has_ori = mode != "CMP_SPARSE"
    has_cmp = mode != "ORI_SPARSE"
    ori_kv = (
        _pa_tensor(
            pa_ori, KV_ROW_BYTES_ORI, axis0_noncontiguous=p["kv_axis0_noncontiguous"]
        )
        if has_ori
        else None
    )
    pa_cmp = case["pa_cmp"]
    kwargs = {
        "q": case["q"].npu(),
        "ori_kv": ori_kv,
        "cmp_kv": _pa_tensor(
            pa_cmp, KV_ROW_BYTES_CMP, axis0_noncontiguous=p["kv_axis0_noncontiguous"]
        )
        if has_cmp
        else None,
        "ori_sparse_indices": (
            case["idx_ori"].reshape(case["t1"], 1, -1).npu() if has_ori else None
        ),
        "cmp_sparse_indices": case["idx_cmp"].reshape(case["t1"], 1, -1).npu()
        if has_cmp
        else None,
        "ori_block_table": pa_ori["block_table"].npu() if has_ori else None,
        "cmp_block_table": pa_cmp["block_table"].npu() if has_cmp else None,
        "cu_seqlens_q": case["cu_seqlens_q"].npu(),
        "seqused_ori_kv": None,
        "seqused_cmp_kv": None,
        "ori_topk_length": (
            case["len_ori"].reshape(case["t1"], 1).npu() if has_ori else None
        ),
        "cmp_topk_length": (
            case["len_cmp"].reshape(case["t1"], 1).npu() if has_cmp else None
        ),
        "sinks": case["sinks"].npu(),
        "metadata": None,
        "quant_mode": quant_mode,
        "softmax_scale": case["scale"],
        "rope_head_dim": 64,
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "return_softmax_lse": p.get("return_softmax_lse", False),
    }
    for side in ("ori", "cmp"):
        kv = kwargs[f"{side}_kv"]
        if kv is not None:
            LOGGER.info(
                "%s_kv shape=%s stride=%s storage_offset=%s",
                side,
                tuple(kv.shape),
                kv.stride(),
                kv.storage_offset(),
            )
    actual, actual_lse = _call_public(
        kwargs,
        case["cu_seqlens_q"],
        omit_ori_length=p.get("omit_ori_topk_length", False),
        omit_cmp_length=p.get("omit_cmp_topk_length", False),
        api=api,
    )
    if kwargs["return_softmax_lse"]:
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    if has_ori:
        empty_rows = case["len_ori"].reshape(-1) == 0
        if has_cmp:
            empty_rows &= case["len_cmp"].reshape(-1) == 0
    else:
        empty_rows = case["len_cmp"].reshape(-1) == 0
    if bool(empty_rows.any()):
        empty_out = actual.reshape(case["t1"], N1, D)[empty_rows]
        assert torch.count_nonzero(empty_out) == 0, (
            "empty queries must produce exact zero"
        )
        if kwargs["return_softmax_lse"]:
            torch.testing.assert_close(
                actual_lse[0, empty_rows],
                case["sinks"].expand(int(empty_rows.sum()), N1),
                rtol=0,
                atol=0,
            )
    result, percent = check_result(expected, actual)
    assert result == "Pass", f"strict BF16 comparator failed: {percent:.6f}%"
    # 大prefill输出可达16GiB，误差统计不要同时物化三份全量FP32输出。
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    max_abs = 0.0
    for start in range(0, actual_flat.numel(), 1 << 20):
        diff = (
            actual_flat[start : start + (1 << 20)].float()
            - expected_flat[start : start + (1 << 20)].float()
        ).abs()
        chunk_max = diff.max().item()
        max_abs = chunk_max if chunk_max != chunk_max else max(max_abs, chunk_max)
    LOGGER.info(
        "[mqsmla test_case] NPU %s quant%s B=%s T1=%s OK: max_abs=%.6g",
        mode,
        quant_mode,
        case["b"],
        case["t1"],
        max_abs,
    )


@pytest.mark.parametrize("case_name", _selected_cases())
def test_mixed_quant_sparse_flash_mla(case_name):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    run_prepared(*gen_data(TEST_PARAMS[case_name]), api=OPS)


def test_dsl_meta_output_shapes():
    q = torch.empty((3, 64, 512), dtype=torch.bfloat16, device="meta")
    for has_lse in (False, True):
        out, lse = OPS.mixed_quant_sparse_flash_mla(
            q=q,
            quant_mode=4,
            ori_kv=torch.empty((1, 128, 1, 544), dtype=torch.uint8, device="meta"),
            layout_q="TND",
            layout_kv="PA_BBND",
            return_softmax_lse=has_lse,
        )
        assert out.shape == q.shape and out.dtype == q.dtype
        assert lse.shape == ((1, q.shape[0], q.shape[1]) if has_lse else (0,))
        assert lse.dtype == torch.float32


def test_dsl_uses_current_npu_stream():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        run_prepared(*gen_data(TEST_PARAMS["mixed_decode"]), api=OPS)
    stream.synchronize()


def test_dsl_rejects_invalid_packing():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    device = torch.device("npu", torch.npu.current_device())
    kwargs = dict(
        q=torch.empty((1, 64, 512), device=device, dtype=torch.bfloat16),
        ori_kv=torch.empty((1, 128, 1, 543), device=device, dtype=torch.uint8),
        ori_sparse_indices=torch.empty((1, 1, 128), device=device, dtype=torch.int32),
        ori_block_table=torch.empty((1, 1), device=device, dtype=torch.int32),
        cu_seqlens_q=torch.empty(2, device=device, dtype=torch.int32),
        metadata=torch.zeros(2048, device=device, dtype=torch.int32),
        sinks=torch.empty(64, device=device),
        softmax_scale=512**-0.5,
        quant_mode=4,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    frontend = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder,
        "load",
        side_effect=AssertionError(
            "Expected DSL validation, but the ASC builder was loaded"
        ),
    ):
        with pytest.raises(ValueError, match="ori_kv"):
            OPS.mixed_quant_sparse_flash_mla(**kwargs)


@pytest.mark.parametrize("quant_mode", (1, 2, 3))
def test_existing_modes_keep_asc_route(quant_mode):
    frontend = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    from types import SimpleNamespace

    sentinel = object()
    main_args, metadata_args = [], []
    fake = SimpleNamespace(
        mixed_quant_sparse_flash_mla=lambda *args: main_args.append(args) or sentinel,
        mixed_quant_sparse_flash_mla_metadata=lambda *args: metadata_args.append(args)
        or sentinel,
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder, "load", return_value=fake
    ):
        assert (
            frontend.mixed_quant_sparse_flash_mla(torch.empty(1), quant_mode)
            is sentinel
        )
        assert (
            frontend.mixed_quant_sparse_flash_mla_metadata(64, 1, 512, quant_mode)
            is sentinel
        )
    assert main_args[0][1] == quant_mode and metadata_args[0][3] == quant_mode


@pytest.mark.parametrize("layout_q", ("TND", "BSND"))
@pytest.mark.parametrize("layout_kv", ("PA_BBND", "TND", "BSND"))
@pytest.mark.parametrize("has_cmp", (False, True))
def test_dsl_layouts(layout_q, layout_kv, has_cmp):
    _check_layout(layout_q, layout_kv, has_cmp=has_cmp)


@pytest.mark.parametrize("heads", (1, 17, 32, 33, 63, 64))
def test_dsl_heads_and_tnd_padding(heads):
    _check_layout("TND", "PA_BBND", heads=heads, padding=True, has_sinks=False)


def test_dsl_bsnd_padding():
    _check_layout("BSND", "BSND", padding=True)


def _check_layout(
    layout_q, layout_kv, *, has_cmp=True, heads=64, padding=False, has_sinks=True
):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    p, case, expected, expected_lse = gen_data(
        dict(
            _BASE,
            q_lengths=[2, 2],
            has_cmp=has_cmp,
            heads=heads,
            omit_sinks=not has_sinks,
        )
    )
    q = case["q"]
    expected = expected.view(case["t1"], heads, D)
    kwargs = dict(
        q=q.npu(),
        quant_mode=4,
        layout_q=layout_q,
        layout_kv=layout_kv,
        cu_seqlens_q=case["cu_seqlens_q"].npu(),
        return_softmax_lse=True,
        softmax_scale=case["scale"],
        sinks=case["sinks"][:heads].npu() if has_sinks else None,
    )
    for side in ("ori", "cmp") if has_cmp else ("ori",):
        pa = case["pa" if side == "ori" else "pa_cmp"]
        row_bytes = KV_ROW_BYTES_ORI if side == "ori" else KV_ROW_BYTES_CMP
        kwargs[f"{side}_sparse_indices"] = (
            case[f"idx_{side}"].view(case["t1"], 1, -1).npu()
        )
        kwargs[f"{side}_topk_length"] = case[f"len_{side}"].view(case["t1"], 1).npu()
        if layout_kv == "PA_BBND":
            kwargs[f"{side}_kv"] = _pa_tensor(pa, row_bytes)
            kwargs[f"{side}_block_table"] = pa["block_table"].npu()
        else:
            blocks = pa["phys"]["row"][pa["block_table"].long().flatten()]
            dense = blocks.reshape(case["b"], -1, 1, row_bytes)
            if layout_kv == "TND":
                span = dense.shape[1]
                dense = dense.flatten(0, 1)
                kwargs[f"cu_seqlens_{side}_kv"] = (
                    torch.arange(case["b"] + 1, dtype=torch.int32).mul_(span).npu()
                )
            kwargs[f"{side}_kv"] = dense.npu()
    if padding:
        kwargs["seqused_q"] = torch.tensor([1, 0], dtype=torch.int32, device="npu")
        expected[[1, 2, 3]] = 0
        # PR 323 preserves the existing BSND empty-row LSE contract;
        # only physical TND padding clears LSE to zero.
        expected_lse[:, [1, 2, 3]] = 0 if layout_q == "TND" else case["sinks"][:heads]
    if layout_q == "BSND":
        kwargs["q"] = kwargs["q"].view(2, 2, heads, D)
        for name in (
            "ori_sparse_indices",
            "cmp_sparse_indices",
            "ori_topk_length",
            "cmp_topk_length",
        ):
            if name in kwargs:
                kwargs[name] = kwargs[name].view(2, 2, *kwargs[name].shape[1:])
    kwargs["metadata"] = _library_metadata(kwargs)
    frontend = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder,
        "load",
        side_effect=AssertionError("Mode 4 must bypass the ASC builder"),
    ):
        actual, lse = OPS.mixed_quant_sparse_flash_mla(**kwargs)
    assert actual.shape == kwargs["q"].shape
    assert lse.shape == ((1, 4, heads) if layout_q == "TND" else (2, 1, 2, heads))
    actual = actual.cpu().reshape_as(expected)
    torch.testing.assert_close(
        lse.cpu().reshape_as(expected_lse), expected_lse, rtol=1e-5, atol=1e-5
    )
    result, percent = check_result(expected.reshape(-1, D), actual.reshape(-1, D))
    assert result == "Pass", f"strict BF16 comparator failed: {percent:.6f}%"


def test_dsl_bsnd_meta_output_shapes():
    q = torch.empty((2, 3, 17, D), dtype=torch.bfloat16, device="meta")
    kv = torch.empty((2, 128, 1, KV_ROW_BYTES_ORI), dtype=torch.uint8, device="meta")
    for want_lse in (False, True):
        out, lse = OPS.mixed_quant_sparse_flash_mla(
            q, 4, ori_kv=kv, return_softmax_lse=want_lse
        )
        assert out.shape == q.shape and out.dtype == q.dtype
        assert lse.shape == ((2, 1, 3, 17) if want_lse else (0,))
        assert lse.dtype == torch.float32


def test_dsl_metadata_fake_tensor():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        cu = torch.empty((3,), dtype=torch.int32, device="npu")
        metadata = OPS.mixed_quant_sparse_flash_mla_metadata(
            17, 1, D, 1, cu_seqlens_q=cu, layout_q="TND", has_cmp_kv=False, ori_topk=128
        )
        assert metadata.shape == (2048,) and metadata.dtype == torch.int32
        assert metadata.device == cu.device


@pytest.mark.parametrize(
    "extra",
    (
        {"cmp_ratio": 4},
        {"ori_mask_mode": 3},
        {"topk_value_mode": 0},
        {"key_dtype": 1},
        {"value_dtype": 1},
        {"rope_head_dim": 32},
    ),
)
def test_dsl_rejects_unsupported_modes(extra):
    pytest.importorskip("torch_npu")
    frontend = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder,
        "load",
        side_effect=AssertionError("Mode 4 must bypass the ASC builder"),
    ):
        with pytest.raises(NotImplementedError):
            OPS.mixed_quant_sparse_flash_mla(
                torch.empty((1, 64, D), dtype=torch.bfloat16, device="npu"), 4, **extra
            )


def test_dsl_bsnd_library_metadata():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    frontend = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    lengths = torch.full((2, 1, 1), 128, dtype=torch.int32, device="npu")
    metadata = OPS.mixed_quant_sparse_flash_mla_metadata(
        64,
        1,
        D,
        1,
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_ori_kv=128,
        ori_topk=128,
        ori_topk_length=lengths,
        has_cmp_kv=False,
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder,
        "load",
        side_effect=AssertionError("Mode 4 must bypass the ASC builder"),
    ):
        q = torch.zeros((2, 1, 64, D), dtype=torch.bfloat16, device="npu")
        kv = torch.zeros((2, 128, 1, KV_ROW_BYTES_ORI), dtype=torch.uint8, device="npu")
        indices = (
            torch.arange(128, dtype=torch.int32, device="npu")
            .view(1, 1, 1, 128)
            .repeat(2, 1, 1, 1)
        )
        out, lse = OPS.mixed_quant_sparse_flash_mla(
            q,
            4,
            ori_kv=kv,
            ori_sparse_indices=indices,
            metadata=metadata,
            return_softmax_lse=True,
        )
    assert metadata.shape == (2048,) and torch.count_nonzero(out).item() == 0
    import math

    torch.testing.assert_close(
        lse.cpu(), torch.full((2, 1, 1, 64), math.log(128)), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("layout_q,layout_kv", (("TND", "PA"), ("tnd", "pa_bbnd")))
def test_dsl_meta_layout_aliases(layout_q, layout_kv):
    q = torch.empty((3, 17, D), dtype=torch.bfloat16, device="meta")
    kv = torch.empty((2, 128, 1, KV_ROW_BYTES_ORI), dtype=torch.uint8, device="meta")
    out, lse = OPS.mixed_quant_sparse_flash_mla(
        q, 4, ori_kv=kv, layout_q=layout_q, layout_kv=layout_kv, return_softmax_lse=True
    )
    assert out.shape == q.shape and lse.shape == (1, 3, 17)
