# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import torch

from mqsmla_golden import (
    C,
    make_inputs,
    mqsmla_cpu_benchmark,
    pack_pa_layout,
)
from result_compare_method import check_result


MODES = ("ORI_SPARSE", "ORI_CMP_SPARSE", "CMP_SPARSE")

DEFAULTS = {
    "quant_mode": 1,
    "B": 1,
    "S1": 2,
    "cu_seqlens_q": None,
    "seqused_q": None,
    "S2": 8192,
    "S2C": 2048,
    "K1": 128,
    "K2": 512,
    "omit_ori_topk_length": False,
    "omit_cmp_topk_length": False,
    "ori_topk_length": None,
    "cmp_topk_length": None,
    "ori_kv_topk_mode": "fullK",
    "cmp_kv_topk_mode": "fullK",
    "block_size1": 64,
    "block_size2": 64,
    "seed": 7,
    "dist": "norm",
    "kv_axis0_noncontiguous": False,
}


def fill_defaults(params):
    p = dict(DEFAULTS)
    p.update(params)
    return p


def build_case(params):
    """Generate one complete multi-batch case; CPU/NPU execution is separate."""
    p = fill_defaults(params)
    mode = p["template_run_mode"]
    if mode not in MODES:
        raise ValueError(f"template_run_mode must be one of {MODES}, got {mode!r}")
    _, ref = make_inputs(
        scenario=getattr(C.Scenario, mode),
        quant_mode=int(p["quant_mode"]),
        batch_size=int(p["B"]),
        seqlen_q=p["S1"],
        seqlen_kv=p["S2"],
        cmp_seqlen=p["S2C"],
        ori_topk=int(p["K1"]),
        cmp_topk=int(p["K2"]),
        seed=int(p["seed"]),
        dist=p["dist"],
        cu_seqlens_q=p["cu_seqlens_q"],
        seqused_q=p["seqused_q"],
        ori_topk_length=p["ori_topk_length"],
        cmp_topk_length=p["cmp_topk_length"],
        ori_kv_topk_mode=p["ori_kv_topk_mode"],
        cmp_kv_topk_mode=p["cmp_kv_topk_mode"],
    )
    for side in ("ori", "cmp"):
        if (
            p.get(f"omit_{side}_topk_length")
            and ref.get(f"{side}_topk_length") is not None
        ):
            width = ref[f"{side}_sparse_indices"].shape[-1]
            if bool((ref[f"{side}_topk_length"] != width).any()):
                raise ValueError(
                    f"omitting {side}_topk_length requires all K indices to be valid"
                )
    if p.get("softmax_scale") is not None:
        ref["softmax_scale"] = float(p["softmax_scale"])
    pa = pack_pa_layout(
        ref,
        block_size=int(p["block_size1"]),
        cmp_block_size=int(p["block_size2"]),
        seed=int(p["seed"]) + 97,
    )
    case = {
        "mode": mode,
        "quant_mode": int(p["quant_mode"]),
        "b": ref["B"],
        "t1": ref["T1"],
        "n1": 64,
        "d": C.D,
        "q": ref["q"],
        "sinks": ref["sinks"],
        "scale": ref["softmax_scale"],
        "cu_seqlens_q": ref["cu_seqlens_q"],
        "seqused_q": ref["seqused_q"],
        "pa": pa.get("ori"),
        "pa_cmp": pa.get("cmp"),
        "idx_ori": ref.get("ori_sparse_indices"),
        "len_ori": ref.get("ori_topk_length"),
        "idx_cmp": ref.get("cmp_sparse_indices"),
        "len_cmp": ref.get("cmp_topk_length"),
    }
    return case, ref


def _npu_available():
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def prepare_case(params):
    """Prepare CPU inputs and golden once; does not require an NPU or wheel."""
    p = fill_defaults(params)
    case, ref = build_case(p)
    expected, expected_lse = mqsmla_cpu_benchmark(ref, return_softmax_lse=True)
    return p, case, expected.reshape(-1, C.D), expected_lse


def run_params(params, *, metadata_backend="aicpu", verify_metadata_backends=False):
    run_prepared(
        *prepare_case(params),
        metadata_backend=metadata_backend,
        verify_metadata_backends=verify_metadata_backends,
    )


def run_prepared(
    p,
    case,
    expected,
    expected_lse,
    *,
    metadata_backend="aicpu",
    verify_metadata_backends=False,
):
    """Execute prepared inputs; never regenerate inputs or golden."""
    mode = case["mode"]
    quant_mode = case["quant_mode"]
    if not _npu_available():
        raise RuntimeError("torch_npu/device unavailable")
    pa_ori = case["pa"]
    has_ori = mode != "CMP_SPARSE"
    has_cmp = mode != "ORI_SPARSE"
    ori_kv = (
        _pa_tensor(
            pa_ori, C.KV_ROW_BYTES_ORI, axis0_noncontiguous=p["kv_axis0_noncontiguous"]
        )
        if has_ori
        else None
    )
    pa_cmp = case["pa_cmp"]
    kwargs = {
        "q": case["q"].npu(),
        "ori_kv": ori_kv,
        "cmp_kv": _pa_tensor(
            pa_cmp, C.KV_ROW_BYTES_CMP, axis0_noncontiguous=p["kv_axis0_noncontiguous"]
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
        "seqused_q": case["seqused_q"].npu()
        if case.get("seqused_q") is not None
        else None,
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
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "return_softmax_lse": p.get("return_softmax_lse", False),
    }
    for side in ("ori", "cmp"):
        kv = kwargs[f"{side}_kv"]
        if kv is not None:
            print(
                f"{side}_kv shape={tuple(kv.shape)} stride={kv.stride()} "
                f"storage_offset={kv.storage_offset()}"
            )
    actual, actual_lse = _call_public(
        kwargs,
        mode,
        case["t1"],
        omit_ori_length=p.get("omit_ori_topk_length", False),
        omit_cmp_length=p.get("omit_cmp_topk_length", False),
        metadata_backend=metadata_backend,
        verify_metadata_backends=verify_metadata_backends,
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
        empty_out = actual.reshape(case["t1"], C.N1, C.D)[empty_rows]
        assert torch.count_nonzero(empty_out) == 0, (
            "empty queries must produce exact zero"
        )
        if kwargs["return_softmax_lse"]:
            torch.testing.assert_close(
                actual_lse[0, empty_rows],
                case["sinks"].expand(int(empty_rows.sum()), C.N1),
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
    print(
        f"[mqsmla test_case] NPU {mode} quant{quant_mode} B={case['b']} "
        f"T1={case['t1']} OK: max_abs={max_abs:.6g}"
    )


def _pa_tensor(pa, row_bytes, *, axis0_noncontiguous=False):
    """先搬完整含 padding 的 storage，再在 NPU 上切视图，保留真实页距。"""
    row = pa["phys"]["row"]
    shape = (pa["num_phys_blocks"], pa["block_size"], 1, row_bytes)
    if not axis0_noncontiguous:
        return row.reshape(shape).npu()
    page_bytes = pa["block_size"] * row_bytes
    stride0 = page_bytes + C.KV_PAGE_PADDING_BYTES
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


def _get_metadata_cube_core_num(device):
    import acl

    with torch.npu.device(device):
        blocks, status = acl.rt.get_device_info(torch.npu.current_device(), 102)
    if status != 0:
        raise RuntimeError(f"ACL Cube core count query failed: {status}")
    return int(blocks)


def _make_metadata_python(ori_lengths, cmp_lengths, cu_seqlens_q, device, *, has_cmp):
    """Compute the active wheel's FA/FD policy on CPU, without launching AICPU."""
    from metadata_python import build_metadata
    import ops.mixed_quant_sparse_flash_mla_metadata as policy

    blocks = _get_metadata_cube_core_num(device)
    metadata = build_metadata(
        ori_lengths.detach().cpu().reshape(-1).tolist(),
        cmp_lengths.detach().cpu().reshape(-1).tolist(),
        None if cu_seqlens_q is None else cu_seqlens_q.detach().cpu().tolist(),
        blocks,
        has_cmp=has_cmp,
        support_fd=policy.SUPPORT_FD,
        selective_fd_72=policy.SELECTIVE_FD_72,
        fd_min_saved_tiles=policy.FD_MIN_SAVED_TILES,
    )
    print(
        f"[mqsmla metadata] backend=python rows={ori_lengths.shape[0]} "
        f"blocks={blocks} fd_vectors={metadata[C.FD_USED_VEC_NUM_WORD]} "
        f"async={metadata[C.FD_ASYNC_MODE_WORD]}"
    )
    return torch.tensor(metadata, dtype=torch.int32)


def _call_public(
    kwargs,
    mode,
    t1,
    *,
    omit_ori_length=False,
    omit_cmp_length=False,
    metadata_backend="aicpu",
    verify_metadata_backends=False,
):
    """Run metadata and attention through the installed wheel's torch.ops."""
    # Explicit import surfaces DSL registration failures instead of testing another implementation.
    import cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl  # noqa: F401

    mixed_quant_sparse_flash_mla = (
        torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla
    )
    mixed_quant_sparse_flash_mla_metadata = (
        torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata
    )

    kwargs = dict(kwargs)
    if kwargs.get("metadata") is None:
        # Metadata always requires both length tensors; an absent side (cmp in
        # ORI_SPARSE, ori in CMP_SPARSE) passes a zero-length tensor.
        ori_lengths = kwargs["ori_topk_length"]
        if ori_lengths is None:
            ori_lengths = torch.zeros((t1, 1), dtype=torch.int32).npu()
        cmp_lengths = kwargs["cmp_topk_length"]
        if cmp_lengths is None:
            cmp_lengths = torch.zeros((t1, 1), dtype=torch.int32).npu()
        if metadata_backend == "python":
            kwargs["metadata"] = _make_metadata_python(
                ori_lengths,
                cmp_lengths,
                kwargs["cu_seqlens_q"],
                kwargs["q"].device,
                has_cmp=(mode != "ORI_SPARSE"),
            ).to(kwargs["q"].device)
        else:
            aicpu_metadata = mixed_quant_sparse_flash_mla_metadata(
                ori_lengths,
                cmp_lengths,
                cu_seqlens_q=kwargs["cu_seqlens_q"],
                num_heads_q=C.N1,
                num_heads_kv=1,
                head_dim=C.D,
                quant_mode=kwargs["quant_mode"],
                has_cmp_kv=(mode != "ORI_SPARSE"),
            )
            kwargs["metadata"] = aicpu_metadata
            if verify_metadata_backends:
                python_metadata = _make_metadata_python(
                    ori_lengths,
                    cmp_lengths,
                    kwargs["cu_seqlens_q"],
                    kwargs["q"].device,
                    has_cmp=(mode != "ORI_SPARSE"),
                )
                torch.testing.assert_close(
                    python_metadata[: C.FD_USED_VEC_NUM_WORD + 1],
                    aicpu_metadata.cpu()[: C.FD_USED_VEC_NUM_WORD + 1],
                    rtol=0,
                    atol=0,
                )
                # Uniform AICPU plans leave tail padding undefined. Selective FD
                # initializes and consumes readiness flags and async mode as well.
                if python_metadata[C.FD_ASYNC_MODE_WORD] == 1:
                    torch.testing.assert_close(
                        python_metadata[
                            C.FD_USED_VEC_NUM_WORD + 1 : C.FD_ASYNC_MODE_WORD + 1
                        ],
                        aicpu_metadata.cpu()[
                            C.FD_USED_VEC_NUM_WORD + 1 : C.FD_ASYNC_MODE_WORD + 1
                        ],
                        rtol=0,
                        atol=0,
                    )
                print("[mqsmla metadata] aicpu/python partition match")
    # Metadata keeps its required length inputs; attention can omit either side.
    if omit_ori_length:
        kwargs.pop("ori_topk_length")
    if omit_cmp_length:
        kwargs.pop("cmp_topk_length")
    want_lse = kwargs.get("return_softmax_lse", False)
    out, lse = mixed_quant_sparse_flash_mla(**kwargs)
    output = out.cpu().reshape(-1, C.D)
    lse = lse.cpu()
    if want_lse:
        assert torch.isfinite(lse).all()
    else:
        assert lse.shape == (0,) and lse.dtype == torch.float32
    return output, lse
