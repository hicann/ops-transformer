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

from mixed_quant_sparse_flash_mla_golden import (
    gen_qm4_kv,
    generate_cpu_golden,
    restore_qm4_data,
)

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


def _make_qm4_kv(batch, tokens, block_size, fp4, generator):
    """Encode logical KV, then shuffle physical pages independently of the golden.

    Features are nope[448], rope[64], followed by BF16 group scales.
    FP4 stores the earlier feature in the low nibble. The CPU reference keeps
    dequantized logical rows, so it never shares the kernel's page addressing.
    """
    pages = (tokens + block_size - 1) // block_size
    packed, logical = gen_qm4_kv((batch, pages * block_size, D), fp4, generator)
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
    """Build a deterministic pytest case and calculate its CPU reference."""
    p = dict(params)
    p.pop("quant_mode", None)
    heads = p.get("heads", N1)
    generator = torch.Generator().manual_seed(p.get("seed", 42))
    spans = p["q_lengths"]
    batch, rows = len(spans), sum(spans)
    has_ori = p.get("has_ori", True)
    has_cmp = p.get("has_cmp", True)
    mode = "ORI_CMP_SPARSE" if has_cmp else "ORI_SPARSE"
    if not has_ori:
        mode = "CMP_SPARSE"
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
        "mode": mode,
        "scale": D**-0.5,
    }
    logical = {}
    for side, fp4 in (("ori", False), ("cmp", True)):
        if side == "ori" and not has_ori:
            continue
        width = p[f"k_{side}"]
        pa, logical[side] = _make_qm4_kv(
            batch, p[f"s_{side}"], p[f"block_size_{side}"], fp4, generator
        )
        case["pa" if side == "ori" else "pa_cmp"] = pa
        indices = torch.randint(
            p[f"s_{side}"], (rows, width), generator=generator, dtype=torch.int32
        )
        if p.get("unique_indices", False):
            if width > p[f"s_{side}"]:
                raise ValueError("unique TopK width cannot exceed logical KV length")
            indices = torch.stack(
                [
                    torch.randperm(p[f"s_{side}"], generator=generator)[:width]
                    for _ in range(rows)
                ]
            ).int()
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
        if (
            p.get("all_empty", False)
            or (side == "cmp" and not has_cmp)
            or (side == "ori" and p.get("zero_ori", False))
        ):
            lengths.zero_()
        if "active_rows" in p:
            inactive = torch.ones(rows, dtype=torch.bool)
            inactive[p["active_rows"]] = False
            lengths[inactive] = 0
        if p.get(f"omit_{side}_topk_length", False):
            lengths.fill_(width)
        # Invalid unused indices catch reads beyond each valid prefix.
        indices[torch.arange(width)[None, :] >= lengths[:, None]] = -1
        case[f"idx_{side}"] = indices
        case[f"len_{side}"] = lengths
    op_input = {
        "q": q,
        "sinks": None if p.get("omit_sinks") else sinks,
        "cu_seqlens_q": cu,
        "seqused_q": case["seqused_q"],
    }
    for side in ("ori", "cmp"):
        if not (has_ori if side == "ori" else has_cmp):
            continue
        pa = case["pa" if side == "ori" else "pa_cmp"]
        op_input[side + "_kv"] = pa["phys"]["row"].unsqueeze(2)
        op_input[side + "_block_table"] = pa["block_table"]
        op_input[side + "_sparse_indices"] = case["idx_" + side].unsqueeze(1)
        op_input[side + "_topk_length"] = case["len_" + side].unsqueeze(1)
    data = restore_qm4_data(
        op_input,
        dict(
            p,
            layout_q="TND",
            layout_kv="PA_BBND",
            softmax_scale=case["scale"],
            return_softmax_lse=True,
        ),
        logical_kv=logical,
    )
    expected, expected_lse = generate_cpu_golden(data)
    return p, case, expected.reshape(-1, D), expected_lse


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
    has_ori = kwargs.get("ori_kv") is not None
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
    for side in ("ori", "cmp"):
        if kwargs.get(f"{side}_kv") is None:
            continue
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
        kwargs["q"].shape[-2],
        1,
        D,
        1,
        has_ori_kv=has_ori,
        has_cmp_kv=has_cmp,
        **metadata_args,
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
    pa_ori = case.get("pa")
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
        "cmp_ratio": p.get("cmp_ratio", 1),
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


LAYOUTS = (("BSND", "BSND"), ("TND", "TND"), ("BSND", "PA_BBND"), ("TND", "PA_BBND"))


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("has_cmp", (False, True))
def test_dsl_layouts(layout_q, layout_kv, has_cmp):
    _check_layout(layout_q, layout_kv, has_cmp=has_cmp)


@pytest.mark.parametrize(
    "layout_q,layout_kv,heads",
    (
        ("TND", "PA_BBND", 64),
        ("BSND", "PA_BBND", 17),
        ("TND", "TND", 17),
        ("BSND", "BSND", 33),
    ),
)
def test_dsl_cmp_only_layouts(layout_q, layout_kv, heads):
    _check_layout(
        layout_q,
        layout_kv,
        has_ori=False,
        heads=heads,
        spans=[5, 5],
        used=[4, 3],
        axis0_noncontiguous=layout_kv == "PA_BBND",
        address_mode="off",
        data_params=dict(vary_lengths=True),
    )


def test_dsl_cmp_only_fd():
    _check_layout(
        "TND",
        "PA_BBND",
        has_ori=False,
        require_fd=True,
        data_params=dict(s_cmp=389, k_cmp=384, unique_indices=True),
    )


def test_dsl_cmp_only_omit_length_and_sinks():
    _check_layout(
        "TND",
        "PA_BBND",
        has_ori=False,
        heads=17,
        has_sinks=False,
        sink_form="omitted",
        omit_lengths=True,
        return_lse=False,
    )


def test_dsl_cmp_only_empty_without_sinks():
    _check_layout(
        "TND",
        "PA_BBND",
        has_ori=False,
        has_sinks=False,
        all_empty=True,
    )


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_dsl_cmp_only_meta_lse(layout_q, layout_kv):
    q_shape = (6, 17, D) if layout_q == "TND" else (2, 3, 17, D)
    kv_shape = (
        (6, 1, KV_ROW_BYTES_CMP)
        if layout_kv == "TND"
        else (2, 128, 1, KV_ROW_BYTES_CMP)
    )
    q = torch.empty(q_shape, dtype=torch.bfloat16, device="meta")
    cmp_kv = torch.empty(kv_shape, dtype=torch.uint8, device="meta")
    out, lse = OPS.mixed_quant_sparse_flash_mla(
        q,
        4,
        cmp_kv=cmp_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=True,
    )
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == ((1, 6, 17) if layout_q == "TND" else (2, 1, 3, 17))
    assert lse.dtype == torch.float32


@pytest.mark.parametrize(
    "input_name",
    (
        "ori_sparse_indices",
        "ori_block_table",
        "ori_topk_length",
        "cu_seqlens_ori_kv",
        "seqused_ori_kv",
    ),
)
def test_dsl_cmp_only_rejects_orphan_ori_inputs(input_name):
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    kwargs = {input_name: torch.ones(1, dtype=torch.int32)}
    with pytest.raises(ValueError, match="ORI inputs require ori_kv"):
        dsl._validate_inputs(
            torch.empty((1, 64, D), dtype=torch.bfloat16),
            cmp_kv=torch.empty((1, 128, 1, KV_ROW_BYTES_CMP), dtype=torch.uint8),
            cmp_sparse_indices=torch.zeros((1, 1, 1), dtype=torch.int32),
            cmp_block_table=torch.zeros((1, 1), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            metadata=torch.zeros(2048, dtype=torch.int32),
            quant_mode=4,
            **kwargs,
        )


@pytest.mark.parametrize("heads", (1, 17, 32, 33, 63, 64))
def test_dsl_heads_and_tnd_padding(heads):
    _check_layout("TND", "PA_BBND", heads=heads, padding=True, has_sinks=False)


@pytest.mark.parametrize("heads", range(1, 65))
def test_dsl_every_head_count(heads):
    _check_layout("TND", "PA_BBND", heads=heads)


@pytest.mark.parametrize("heads", (2, 33, 64))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_fd_prefix_precision(heads, has_sinks):
    # Small valid prefixes expose BF16 P's dependence on the running max.
    # Require actual library FD splits, including all three head buckets.
    _check_layout(
        "TND",
        "PA_BBND",
        heads=heads,
        has_sinks=has_sinks,
        spans=[2, 1, 2],
        require_fd=True,
        address_mode="local" if heads == 64 else "pre",
        data_params=dict(
            seed=323, s_cmp=521, k_cmp=512, vary_lengths=True, unique_indices=True
        ),
    )


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("has_cmp", (False, True))
def test_dsl_no_sink(layout_q, layout_kv, has_cmp):
    _check_layout(layout_q, layout_kv, heads=17, has_cmp=has_cmp, has_sinks=False)


@pytest.mark.parametrize("layout_kv", ("TND", "PA_BBND"))
@pytest.mark.parametrize("heads", (1, 17, 32, 33, 63, 64))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_physical_tnd_padding(layout_kv, heads, has_sinks):
    _check_layout(
        "TND",
        layout_kv,
        heads=heads,
        has_sinks=has_sinks,
        spans=[4, 0, 3, 5],
        used=[2, 0, 0, 3],
    )


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_all_empty_rows(layout_q, layout_kv, has_sinks):
    _check_layout(layout_q, layout_kv, heads=17, has_sinks=has_sinks, all_empty=True)


@pytest.mark.parametrize("layout_kv", ("TND", "PA_BBND"))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_all_tnd_padding(layout_kv, has_sinks):
    _check_layout(
        "TND",
        layout_kv,
        heads=17,
        has_sinks=has_sinks,
        spans=[0, 4, 0, 3],
        used=[0, 0, 0, 0],
    )


@pytest.mark.parametrize("layout_kv", ("TND", "PA_BBND"))
@pytest.mark.parametrize("has_cmp", (False, True))
def test_dsl_padding_without_topk_lengths(layout_kv, has_cmp):
    _check_layout(
        "TND",
        layout_kv,
        heads=17,
        has_cmp=has_cmp,
        has_sinks=False,
        spans=[4, 0, 3, 5],
        used=[2, 0, 0, 3],
        omit_lengths=True,
    )


@pytest.mark.parametrize("layout_q", ("BSND", "TND"))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_pa_axis0_stride(layout_q, has_sinks):
    _check_layout(
        layout_q, "PA_BBND", heads=33, has_sinks=has_sinks, axis0_noncontiguous=True
    )


def test_dsl_bsnd_padding():
    _check_layout("BSND", "BSND", padding=True)


@pytest.mark.parametrize("layout_q", ("BSND", "TND"))
@pytest.mark.parametrize(
    "scenario",
    ("mixed", "ori_only", "empty", "omit_lengths", "no_sinks", "empty_no_sinks"),
)
def test_dsl_batched_pa(layout_q, scenario):
    omit_lengths = scenario == "omit_lengths"
    _check_layout(
        layout_q,
        "PA_BBND",
        has_cmp=scenario != "ori_only",
        has_sinks=scenario not in ("no_sinks", "empty_no_sinks"),
        all_empty=scenario in ("empty", "empty_no_sinks"),
        spans=[65, 65] if layout_q == "BSND" else [65, 0, 64],
        omit_lengths=omit_lengths,
        address_mode="pipe",
        data_params=dict(
            s_cmp=521,
            k_cmp=512,
            vary_lengths=not omit_lengths,
            omit_ori_topk_length=omit_lengths,
            omit_cmp_topk_length=omit_lengths,
        ),
    )


@pytest.mark.parametrize("boundary", ("padding", "heads", "axis0"))
def test_dsl_batched_pa_fallback(boundary):
    _check_layout(
        "TND",
        "PA_BBND",
        heads=33 if boundary == "heads" else 64,
        padding=boundary == "padding",
        axis0_noncontiguous=boundary == "axis0",
        address_mode="local" if boundary == "padding" else "pre",
        data_params=dict(s_cmp=521, k_cmp=512, vary_lengths=True),
    )


@pytest.mark.parametrize("topk", (1, 17, 129, 544, 1024))
@pytest.mark.parametrize("block_size", (64, 128, 256))
def test_dsl_local_pa_tails(topk, block_size):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[3, 0, 4],
        used=[2, 0, 3],
        address_mode="local",
        data_params=dict(
            k_cmp=topk,
            s_cmp=1031,
            block_size_ori=block_size,
            block_size_cmp=block_size,
            vary_lengths=True,
        ),
    )


def test_dsl_local_pa_fd_boundary():
    # A split query has a private address-table row on each participating core.
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[1],
        require_fd=True,
        address_mode="local",
        data_params=dict(s_cmp=1031, k_cmp=1024),
    )


def test_dsl_local_pa_multiple_groups():
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[737],
        has_sinks=False,
        address_mode="local",
        data_params=dict(s_cmp=743, k_cmp=640, vary_lengths=True),
    )


@pytest.mark.parametrize(
    "pages,query_rows,page_padding,expected",
    (
        (65536, 1024, 0, "pipe_rows"),
        (65536, 128, 0, "local"),
        (65536, 1024, 64, "off"),
        (1 << 24, 1024, 0, "off"),
    ),
)
def test_dsl_large_pa_address_dispatch(pages, query_rows, page_padding, expected):
    # Meta tensors exercise the 2-GiB byte boundary and INT32 row bound without
    # allocating the corresponding physical pool. Numerical coverage is above.
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    kv = torch.empty_strided(
        (pages, 128, 1, 320),
        (128 * 320 + page_padding, 320, 320, 1),
        dtype=torch.uint8,
        device="meta",
    )
    ori = torch.empty((256, 128, 1, 544), dtype=torch.uint8, device="meta")
    bt = torch.empty((256, 256), dtype=torch.int32, device="meta")
    idx = torch.empty((query_rows, 1, 128), dtype=torch.int32, device="meta")
    idx_cmp = torch.empty((query_rows, 1, 512), dtype=torch.int32, device="meta")
    assert dsl._select_addr_mode(ori, bt, idx, kv, bt, idx_cmp) == expected


@pytest.mark.parametrize("topk", (300, 384))
def test_dsl_large_pa_pipe_rows_fd_boundary(topk):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[65, 0, 65],
        used=[63, 0, 64],
        require_fd=True,
        address_mode="pipe_rows",
        high_offset_cmp=True,
        data_params=dict(s_cmp=521, k_cmp=topk, vary_lengths=True, block_size_cmp=256),
    )


@pytest.mark.parametrize("table_width", (4096, 8192))
@pytest.mark.parametrize("has_cmp", (False, True))
def test_dsl_wide_pa_prefill(table_width, has_cmp):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[257],
        has_cmp=has_cmp,
        has_sinks=False,
        address_mode="pipe",
        pa_table_width=table_width,
        data_params=dict(s_cmp=521, k_cmp=512, vary_lengths=True),
    )


@pytest.mark.parametrize("block_size", (64, 256))
@pytest.mark.parametrize("topk", (64, 128, 384))
def test_dsl_generic_pipe(block_size, topk):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[65, 0, 65],
        has_sinks=False,
        require_fd=block_size == 64 and topk == 128,
        address_mode="pipe",
        data_params=dict(
            k_ori=64 if topk == 64 else 128,
            k_cmp=topk,
            s_cmp=521,
            block_size_ori=block_size,
            block_size_cmp=256 if block_size == 64 else 64,
            vary_lengths=True,
        ),
    )


@pytest.mark.parametrize("topk", (127, 300, 500))
def test_dsl_generic_pipe_unaligned_topk(topk):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[67, 0, 66],
        used=[64, 0, 64],
        address_mode="pipe",
        data_params=dict(k_ori=127, k_cmp=topk, s_cmp=521, vary_lengths=True),
    )


@pytest.mark.parametrize("block_size", (64, 256))
def test_dsl_generic_pipe_ori_only(block_size):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[257],
        has_cmp=False,
        has_sinks=False,
        omit_lengths=True,
        address_mode="pipe",
        data_params=dict(k_ori=64, block_size_ori=block_size),
    )


def test_dsl_batched_pa_prefill():
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[129],
        has_sinks=False,
        address_mode="pipe",
        data_params=dict(s_cmp=521, k_cmp=512, vary_lengths=True),
    )


@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_batched_pa_fd_prefix(has_sinks):
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[129],
        has_sinks=has_sinks,
        require_fd=True,
        address_mode="pipe",
        data_params=dict(
            seed=323,
            s_cmp=521,
            k_cmp=512,
            unique_indices=True,
            active_rows=[128],
        ),
    )


def _split_test_row_metadata(kwargs, case):
    """Exercise FD even on SDKs whose planner disables it for this device.

    Keep the library's serialized layout and all query/KV inputs. Two FA
    cores partition one row at tile 1, then one FD task reduces its two
    partials. This fixture changes neither the golden nor its tolerances.
    """
    tiles = torch.zeros(case["t1"], dtype=torch.int32)
    for side in ("ori", "cmp"):
        if kwargs.get(f"{side}_kv") is not None:
            tiles += (case[f"len_{side}"].reshape(-1) + 127) // 128
    cu = case["cu_seqlens_q"].tolist()
    used = kwargs.get("seqused_q")
    if used is not None:
        for batch, count in enumerate(used.cpu().tolist()):
            tiles[cu[batch] + count : cu[batch + 1]] = 0
    candidates = (tiles > 1).nonzero().flatten()
    assert candidates.numel(), "FD precision fixture requires a multi-tile row"
    row = int(candidates[0])
    batch = next(b for b in range(len(cu) - 1) if cu[b] <= row < cu[b + 1])
    local_row = row - cu[batch]
    metadata = torch.zeros(2048, dtype=torch.int32)
    # Core 0 ends after the first tile of the selected row, in slot 0.
    metadata[:9] = torch.tensor([1, 0, 0, 0, batch, local_row, 1, 0, 0])
    # Core 1 continues that row in slot 1 and owns the remaining full rows.
    metadata[9:18] = torch.tensor([1, batch, local_row, 1, len(cu) - 1, 0, 0, 1, 0])
    metadata[720:728] = torch.tensor(
        [1, batch, local_row, 0, 2, 0, kwargs["q"].shape[-2], 0]
    )
    return metadata.to(kwargs["q"].device)


def _check_layout(
    layout_q,
    layout_kv,
    *,
    has_ori=True,
    has_cmp=True,
    heads=64,
    padding=False,
    has_sinks=True,
    sink_form="none",
    spans=None,
    used=None,
    all_empty=False,
    axis0_noncontiguous=False,
    omit_lengths=False,
    omit_length_sides=(),
    require_fd=False,
    address_mode=None,
    data_params=None,
    pa_table_width=None,
    high_offset_cmp=False,
    return_lse=True,
):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    params = dict(
        _BASE,
        q_lengths=spans if spans is not None else [2, 2],
        has_ori=has_ori,
        has_cmp=has_cmp,
        heads=heads,
        omit_sinks=not has_sinks,
        all_empty=all_empty,
    )
    params.update(data_params or {})
    sides = (("ori",) if has_ori else ()) + (("cmp",) if has_cmp else ())
    for side in sides:
        if omit_lengths or side in omit_length_sides:
            params[f"omit_{side}_topk_length"] = True
    p, case, expected, expected_lse = gen_data(params)
    q = case["q"]
    batch, rows = case["b"], case["t1"]
    expected = expected.view(case["t1"], heads, D)
    kwargs = dict(
        q=q.npu(),
        quant_mode=4,
        layout_q=layout_q,
        layout_kv=layout_kv,
        cu_seqlens_q=case["cu_seqlens_q"].npu(),
        return_softmax_lse=return_lse,
        softmax_scale=case["scale"],
        sinks=case["sinks"][:heads].npu() if has_sinks else None,
    )
    for side in sides:
        pa = case["pa" if side == "ori" else "pa_cmp"]
        row_bytes = KV_ROW_BYTES_ORI if side == "ori" else KV_ROW_BYTES_CMP
        kwargs[f"{side}_sparse_indices"] = (
            case[f"idx_{side}"].view(case["t1"], 1, -1).npu()
        )
        kwargs[f"{side}_topk_length"] = case[f"len_{side}"].view(case["t1"], 1).npu()
        if layout_kv == "PA_BBND":
            kwargs[f"{side}_kv"] = _pa_tensor(
                pa, row_bytes, axis0_noncontiguous=axis0_noncontiguous
            )
            table = pa["block_table"]
            if pa_table_width is not None:
                assert pa_table_width >= table.shape[1]
                table = torch.cat(
                    (
                        table,
                        torch.zeros(
                            (batch, pa_table_width - table.shape[1]), dtype=table.dtype
                        ),
                    ),
                    dim=1,
                )
            kwargs[f"{side}_block_table"] = table.npu()
            if side == "cmp" and high_offset_cmp:
                # Initialize only the pages the case references, all above 2 GiB.
                kv = kwargs["cmp_kv"]
                page_offset = (1 << 31) // kv.stride(0) + 1
                pool = kv.new_empty((page_offset + kv.shape[0], *kv.shape[1:]))
                pool[page_offset:].copy_(kv)
                kwargs["cmp_kv"] = pool
                kwargs["cmp_block_table"] += page_offset
        else:
            blocks = pa["phys"]["row"][pa["block_table"].long().flatten()]
            dense = blocks.reshape(case["b"], -1, 1, row_bytes)
            if layout_kv == "TND":
                # Independent physical spans with poisoned KV padding ensure
                # addressing uses cu_seqlens rather than a uniform batch pitch.
                chunks, ends = [], [0]
                for b in range(batch):
                    chunks.append(
                        torch.cat(
                            (
                                dense[b, : p[f"s_{side}"]],
                                torch.full(
                                    (7 * b, 1, row_bytes), 0xA5, dtype=torch.uint8
                                ),
                            )
                        )
                    )
                    ends.append(ends[-1] + chunks[-1].shape[0])
                dense = torch.cat(chunks)
                kwargs[f"cu_seqlens_{side}_kv"] = torch.tensor(
                    ends, dtype=torch.int32
                ).npu()
                kwargs[f"seqused_{side}_kv"] = torch.full(
                    (batch,), p[f"s_{side}"], dtype=torch.int32, device="npu"
                )
            kwargs[f"{side}_kv"] = dense.npu()
    if padding or used is not None:
        used = used if used is not None else [1, 0]
        kwargs["seqused_q"] = torch.tensor(used, dtype=torch.int32, device="npu")
        padded = torch.zeros(rows, dtype=torch.bool)
        for b, actual in enumerate(used):
            padded[
                int(case["cu_seqlens_q"][b]) + actual : int(case["cu_seqlens_q"][b + 1])
            ] = True
        expected[padded] = 0
        # PR 323 preserves the existing BSND empty-row LSE contract;
        # only physical TND padding clears LSE to zero.
        expected_lse[:, padded] = 0 if layout_q == "TND" else case["sinks"][:heads]
        # No query or sparse-index load is allowed for physical padding.
        poison_rows = padded.npu()
        kwargs["q"][poison_rows] = float("nan")
        for side in sides:
            kwargs[f"{side}_sparse_indices"][poison_rows] = 0x7FFFFFFF
    if layout_q == "BSND":
        seq = p["q_lengths"][0]
        kwargs["q"] = kwargs["q"].view(batch, seq, heads, D)
        for name in (
            "ori_sparse_indices",
            "cmp_sparse_indices",
            "ori_topk_length",
            "cmp_topk_length",
        ):
            if name in kwargs:
                kwargs[name] = kwargs[name].view(batch, seq, *kwargs[name].shape[1:])
    for side in sides:
        if omit_lengths or side in omit_length_sides:
            kwargs.pop(f"{side}_topk_length", None)
    if not has_sinks and sink_form == "omitted":
        kwargs.pop("sinks")
    kwargs["metadata"] = _library_metadata(kwargs)
    if require_fd:
        metadata = kwargs["metadata"].cpu()
        vector_cores = int(torch.npu.get_device_properties().vector_core_num)
        fd = metadata[80 * 9 : 80 * 9 + vector_cores * 8].view(-1, 8)
        if not bool(((fd[:, 0] != 0) & (fd[:, 4] > 1)).any()):
            kwargs["metadata"] = _split_test_row_metadata(kwargs, case)
            fd = kwargs["metadata"].cpu()[720 : 720 + vector_cores * 8].view(-1, 8)
        assert bool(((fd[:, 0] != 0) & (fd[:, 4] > 1)).any()), (
            "precision regression must exercise cross-core FD reduction"
        )
    if layout_q == "BSND":
        kwargs.pop("cu_seqlens_q")
    frontend = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.mixed_quant_sparse_flash_mla"
    )
    with patch.object(
        frontend.mixed_quant_sparse_flash_mla_op_builder,
        "load",
        side_effect=AssertionError("Mode 4 must bypass the ASC builder"),
    ):
        # Poison the caller-owned outputs so unassigned metadata rows cannot
        # pass accidentally through freshly allocated zero-filled pages.
        dsl = import_module(
            f"{frontend.__package__}.op_dsl.mixed_quant_sparse_flash_mla"
        )
        if address_mode is not None:
            assert (
                dsl._select_addr_mode(
                    kwargs.get("ori_kv"),
                    kwargs.get("ori_block_table"),
                    kwargs["ori_sparse_indices"].view(rows, 1, -1) if has_ori else None,
                    kwargs.get("cmp_kv"),
                    kwargs.get("cmp_block_table"),
                    kwargs["cmp_sparse_indices"].view(rows, 1, -1) if has_cmp else None,
                    layout_kv,
                    heads,
                )
                == address_mode
            )
        lse_shape = (
            ((1, rows, heads) if layout_q == "TND" else (batch, 1, seq, heads))
            if return_lse
            else (0,)
        )
        if has_ori:
            out = torch.full_like(kwargs["q"], float("nan"))
            lse = torch.full(lse_shape, float("nan"), dtype=torch.float32, device="npu")
            dsl._mixed_quant_sparse_flash_mla_out(**kwargs, out=out, lse=lse)
            actual = out
        else:
            actual, lse = OPS.mixed_quant_sparse_flash_mla(**kwargs)
    assert actual.shape == kwargs["q"].shape
    assert lse.shape == lse_shape
    actual = actual.cpu().reshape_as(expected)
    if not has_ori:
        empty = case["len_cmp"].reshape(-1) == 0
        if padding or used is not None:
            empty |= padded
        assert torch.count_nonzero(actual[empty]) == 0
    if return_lse:
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


@pytest.mark.parametrize("layout_q,layout_kv", (("BSND", "TND"), ("TND", "BSND")))
def test_dsl_rejects_mismatched_dense_layouts(layout_q, layout_kv):
    # The tensor-only validator must reject this before compilation/device work.
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    with pytest.raises(ValueError, match="non-PA KV requires"):
        dsl._validate_inputs(
            torch.empty((1, 1, D), dtype=torch.bfloat16),
            ori_kv=torch.empty(1, dtype=torch.uint8),
            quant_mode=4,
            layout_q=layout_q,
            layout_kv=layout_kv,
        )


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


@pytest.mark.parametrize(
    "has_ori,has_cmp,cmp_ratio",
    (
        (True, True, 0),
        (True, True, -7),
        (True, True, 48),
        (True, True, 1 << 40),
        (True, False, 2),
        (False, True, 129),
    ),
)
def test_dsl_ignores_cmp_ratio_for_explicit_indices(has_ori, has_cmp, cmp_ratio):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    torch.npu.set_device(int(os.environ.get("MQSMLA_DEVICE", "0")))
    params = dict(
        _BASE,
        heads=17,
        q_lengths=[2, 1],
        has_ori=has_ori,
        has_cmp=has_cmp,
        cmp_ratio=cmp_ratio,
    )
    run_prepared(*gen_data(params), api=OPS)


@pytest.mark.parametrize(
    "overrides,message",
    (
        ({"softmax_scale": float("nan")}, "softmax_scale must be finite as float32"),
        ({"softmax_scale": float("inf")}, "softmax_scale must be finite as float32"),
        ({"softmax_scale": -float("inf")}, "softmax_scale must be finite as float32"),
        ({"softmax_scale": 3.5e38}, "softmax_scale must be finite as float32"),
        (
            {"cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32)},
            "only supported for TND Q",
        ),
        ({"q": torch.empty((), dtype=torch.bfloat16)}, "q must have rank 4"),
        (
            {"metadata": torch.zeros((1, 2048), dtype=torch.int32)},
            "metadata must be a 1-D int32",
        ),
    ),
)
def test_dsl_rejects_invalid_common_inputs(overrides, message):
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    kwargs = dict(
        q=torch.empty((1, 1, 17, D), dtype=torch.bfloat16),
        quant_mode=4,
        ori_kv=torch.empty((1, 128, 1, KV_ROW_BYTES_ORI), dtype=torch.uint8),
        cmp_kv=torch.empty((1, 128, 1, KV_ROW_BYTES_CMP), dtype=torch.uint8),
        ori_sparse_indices=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
        cmp_sparse_indices=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
        metadata=torch.zeros(2048, dtype=torch.int32),
    )
    kwargs.update(overrides)
    with (
        patch.object(
            dsl,
            "construct_workspace",
            side_effect=AssertionError("launch preparation started"),
        ),
        pytest.raises(ValueError, match=message),
    ):
        dsl.mixed_quant_sparse_flash_mla(**kwargs)


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


@pytest.mark.parametrize("layout_kv", ("TND", "PA_BBND"))
@pytest.mark.parametrize("has_cmp", (False, True))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_tnd_padding_reuses_variant(layout_kv, has_cmp, has_sinks):
    """Changing effective Q lengths must not select another AOT/tiling key."""
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    variant = dsl._get_compiled_variant
    keys = []

    def record(*args, **kwargs):
        keys.append((args, tuple(sorted(kwargs.items()))))
        return variant(*args, **kwargs)

    with patch.object(dsl, "_get_compiled_variant", side_effect=record):
        for used in (None, [33, 0, 20], [0, 0, 0]):
            _check_layout(
                "TND",
                layout_kv,
                heads=64,
                spans=[65, 0, 68],
                used=used,
                has_cmp=has_cmp,
                has_sinks=has_sinks,
                data_params=dict(k_cmp=512, vary_lengths=True),
                address_mode="pipe" if layout_kv == "PA_BBND" else "off",
            )
    assert len(keys) == 3 and keys[0] == keys[1] == keys[2]


@pytest.mark.parametrize("heads", (33, 64))
@pytest.mark.parametrize("has_cmp", (False, True))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_padded_strided_pa_prefill(heads, has_cmp, has_sinks):
    _check_layout(
        "TND",
        "PA_BBND",
        heads=heads,
        spans=[67, 0, 69],
        used=[31, 0, 40],
        axis0_noncontiguous=True,
        has_cmp=has_cmp,
        has_sinks=has_sinks,
        data_params=dict(s_cmp=521, k_cmp=512, vary_lengths=True),
        address_mode="pipe" if heads == 64 else "pre",
    )


@pytest.mark.parametrize("has_cmp", (False, True))
def test_dsl_pa_prefill_sparse_order(has_cmp):
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    # Level 3 retains the caller's sparse order instead of pair-sorted DMA.
    with patch.object(dsl, "_batch_consistency_enabled", return_value=True):
        _check_layout(
            "TND",
            "PA_BBND",
            spans=[129],
            has_cmp=has_cmp,
            data_params=dict(s_cmp=521, k_cmp=512, vary_lengths=True),
            address_mode="pipe",
        )


@pytest.mark.parametrize("used", (31, 0))
def test_dsl_long_padded_pa_query_count(used):
    # Exercise AIC query counting overlapped with the strided address prepass,
    # including a launch with no active query and no sink contribution.
    _check_layout(
        "TND",
        "PA_BBND",
        spans=[129],
        used=[used],
        axis0_noncontiguous=True,
        has_sinks=False,
        data_params=dict(s_cmp=521, k_cmp=512, vary_lengths=True),
        address_mode="pipe",
    )


def _check_length_variant_reuse(layout_q, layout_kv, has_cmp, has_sinks, pipe=False):
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    variant = dsl._get_compiled_variant
    keys = []

    def record(*args, **kwargs):
        keys.append((args, tuple(sorted(kwargs.items()))))
        return variant(*args, **kwargs)

    omissions = ((), ("ori",), ("cmp",), ("ori", "cmp")) if has_cmp else ((), ("ori",))
    with patch.object(dsl, "_get_compiled_variant", side_effect=record):
        for sides in omissions:
            _check_layout(
                layout_q,
                layout_kv,
                heads=64 if pipe else 17,
                has_cmp=has_cmp,
                has_sinks=has_sinks,
                spans=[129] if pipe else [2, 2],
                omit_length_sides=sides,
                data_params=dict(k_cmp=512 if pipe else 129, vary_lengths=True),
                address_mode="pipe" if pipe else None,
            )
    assert len(keys) == len(omissions) and all(key == keys[0] for key in keys)


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("has_cmp", (False, True))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_length_inputs_reuse_variant(layout_q, layout_kv, has_cmp, has_sinks):
    _check_length_variant_reuse(layout_q, layout_kv, has_cmp, has_sinks)


@pytest.mark.parametrize("has_cmp", (False, True))
@pytest.mark.parametrize("has_sinks", (False, True))
def test_dsl_pipe_length_inputs_reuse_variant(has_cmp, has_sinks):
    _check_length_variant_reuse("TND", "PA_BBND", has_cmp, has_sinks, pipe=True)


def _check_sink_variant_reuse(layout_q, layout_kv, has_cmp, omit_lengths, pipe=False):
    dsl = import_module(
        f"{OPS.__name__}.ops.attention.mixed_quant_sparse_flash_mla.op_dsl.mixed_quant_sparse_flash_mla"
    )
    variant = dsl._get_compiled_variant
    keys = []

    def record(*args, **kwargs):
        keys.append((args, tuple(sorted(kwargs.items()))))
        return variant(*args, **kwargs)

    with patch.object(dsl, "_get_compiled_variant", side_effect=record):
        for has_sinks, form in ((True, "none"), (False, "none"), (False, "omitted")):
            _check_layout(
                layout_q,
                layout_kv,
                heads=64 if pipe else 17,
                has_cmp=has_cmp,
                has_sinks=has_sinks,
                sink_form=form,
                spans=[129] if pipe else [2, 2],
                omit_lengths=omit_lengths,
                data_params=dict(k_cmp=512 if pipe else 129, vary_lengths=True),
                address_mode="pipe" if pipe else None,
            )
    assert len(keys) == 3 and keys[0] == keys[1] == keys[2]


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("has_cmp", (False, True))
@pytest.mark.parametrize("omit_lengths", (False, True))
def test_dsl_sink_inputs_reuse_variant(layout_q, layout_kv, has_cmp, omit_lengths):
    _check_sink_variant_reuse(layout_q, layout_kv, has_cmp, omit_lengths)


@pytest.mark.parametrize("has_cmp", (False, True))
@pytest.mark.parametrize("omit_lengths", (False, True))
def test_dsl_pipe_sink_inputs_reuse_variant(has_cmp, omit_lengths):
    _check_sink_variant_reuse("TND", "PA_BBND", has_cmp, omit_lengths, pipe=True)
