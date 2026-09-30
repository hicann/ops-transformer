#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import concurrent.futures


from itertools import product
from pathlib import Path
import pytest
import torch
from common import golden_cache, test_runner
from common import quant_flash_attn_golden as golden

MAIN = ("k", "v", "dequant_scale_k", "v_descale")
SHAPES = (("BnNBsD", 64), ("BnBsND", 72), ("PA_NZ", 128))
ROUTES = (("TND", 64, 3), ("N2TGD", 1, 0), ("N2TGD", 64, 3))
CASES = list(product(SHAPES, ROUTES, (0, 1, 17, 32, 63), (0.5, 1.0)))


def strided_copy(tensor, axes):
    source = tensor.cpu()
    if source.element_size() == 1:
        source = source.view(torch.uint8)
    shape = list(source.shape)
    selection = [slice(None)] * source.ndim
    for axis in axes:
        shape[axis] *= 2
        selection[axis] = slice(1, None, 2)
    storage = torch.zeros(shape, dtype=source.dtype)
    storage[tuple(selection)].copy_(source)
    result = storage.npu()[tuple(selection)]
    if result.dtype != tensor.dtype:
        result = result.view(tensor.dtype)
    assert not result.is_contiguous()
    assert torch.equal(result.cpu().view(torch.uint8), tensor.cpu().view(torch.uint8))
    return result


def prepare_case(spec, cache_dir):
    (layout, dim), (q_layout, q_len, mask), tail, p_scale = spec
    kv_lens = [256 + tail, 128 + (tail + 16) % 64]
    name = f"NC_{layout}_D{dim}_{q_layout}_Q{q_len}_M{mask}_T{tail}_P{p_scale}"
    params = dict(
        name=name,
        B=2,
        N_q=4,
        N_kv=2,
        D=dim,
        cu_seqlens_q=[0, q_len, 2 * q_len],
        cu_seqlens_kv=None,
        seqused_q=[q_len] * 2,
        seqused_kv=kv_lens,
        max_seqlen_q=q_len,
        max_seqlen_kv=max(kv_lens),
        enable_pa=True,
        kv_cache_layout=layout,
        block_size=128,
        mask_mode=mask,
        q_scale_layout=q_layout,
        p_scale=p_scale,
        enable_lse=True,
        enable_v_tail=True,
        is_contiguous=True,
        use_fp64_golden=False,
        use_fp64_compare=False,
    )
    cache = Path(cache_dir or Path(__file__).parent / "common/noncontiguous_cache")
    cache.mkdir(parents=True, exist_ok=True)
    if not all(
        (cache / f"{name}_{suffix}.pt").exists() for suffix in ("input", "cpu_output")
    ):
        torch.manual_seed(42)
        test_runner.execute_test(params, {"gen", "cpu"}, str(cache))
    return params, cache


@pytest.mark.parametrize("spec", CASES)
def test_vtail_noncontiguous(spec, cache_dir, monkeypatch):
    params, cache = prepare_case(spec, cache_dir)
    original = golden.prepare_npu_inputs
    active = {"keys": (), "axes": (0,), "zero": False}

    def prepare(*args, **kwargs):
        inputs = original(*args, **kwargs)
        for key in active["keys"]:
            tensor = inputs[key]
            inputs[key] = (
                tensor[:1].expand_as(tensor)
                if active["zero"]
                else strided_copy(tensor, active["axes"])
            )
        return inputs

    monkeypatch.setattr(golden, "prepare_npu_inputs", prepare)
    variants = [
        ("contiguous", (), (0,)),
        ("main_axis0", MAIN, (0,)),
        ("tail_axis0", ("v_tail",), (0,)),
        ("both_axis0", MAIN + ("v_tail",), (0,)),
    ]
    if params["kv_cache_layout"] != "BnBsND":
        variants += [
            (label, keys, (0, 1))
            for label, keys in (
                ("main_axis01", MAIN),
                ("tail_axis01", ("v_tail",)),
                ("both_axis01", MAIN + ("v_tail",)),
            )
        ]
    baseline = None
    for label, keys, axes in variants:
        active.update(keys=keys, axes=axes)
        result, lse = test_runner.execute_test(params, {"npu", "compare"}, str(cache))
        test_runner.check_results(result, lse)
        actual = golden_cache.load_npu_output(params["name"], cache_dir=str(cache))
        if baseline is None:
            baseline = tuple(t.clone() for t in actual)
        else:
            assert all(torch.equal(a, b) for a, b in zip(actual, baseline)), label
        print(f"NONCONTIG_PASS {params['name']} {label}")
    active.update(keys=("v_tail",), axes=(-1,))
    with pytest.raises(RuntimeError, match="v_tail only supports non-contiguous axes"):
        test_runner.execute_test(params, {"npu", "compare"}, str(cache))
    if params["kv_cache_layout"] == "BnBsND":
        active.update(axes=(1,))
        with pytest.raises(
            RuntimeError, match="v_tail only supports non-contiguous axes"
        ):
            test_runner.execute_test(params, {"npu", "compare"}, str(cache))
    active.update(axes=(0,), zero=True)
    with pytest.raises(RuntimeError, match="v_tail strides must be positive"):
        test_runner.execute_test(params, {"npu", "compare"}, str(cache))
