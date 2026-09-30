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
"""Regression tests for MXFP8 TTK scale and overflow handling."""

import ast
import importlib.util
import inspect
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


@pytest.fixture
def golden(monkeypatch):
    path = Path(__file__).parents[1] / "assets" / "quant_flash_attn_golden.py"
    spec = importlib.util.spec_from_file_location("ttk_mxfp8_golden_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {"D": 64, "N_q": 1, "N_kv": 1, "SPARSE_MODE": 3}.items():
        monkeypatch.setattr(module, name, value, raising=False)
    return module


@pytest.mark.parametrize("kv_len", [33, 319, 513])
@pytest.mark.parametrize("p_scale", [0.5, 1.0, 2.0, 184.2, 397.639])
@pytest.mark.parametrize("tail_len", [0, 17])
def test_lse_matches_logsumexp(golden, p_scale, tail_len, kv_len):
    generator = torch.Generator().manual_seed(714)
    q = (torch.randn(1, 1, 8, 64, generator=generator) * 0.2).to(torch.float8_e4m3fn)
    k = (torch.randn(1, 1, kv_len, 64, generator=generator) * 0.2).to(
        torch.float8_e4m3fn
    )
    v = torch.randn(1, 1, kv_len, 64, generator=generator).to(torch.float8_e4m3fn)
    tail = (
        torch.randn(1, 1, 17, 64, generator=generator).bfloat16() if tail_len else None
    )
    _, lse = golden.cpu_mxfp8_golden(
        q,
        k,
        v,
        torch.ones(1, 1, 8, 2),
        torch.ones(1, 1, kv_len, 2),
        torch.ones(1, 1, (kv_len + 31) // 32, 64),
        p_scale,
        [8],
        [kv_len],
        v_tail_bnsd=tail,
        tail_lens=[tail_len],
    )
    scores = q.float() @ k.float().transpose(-1, -2) / 8
    mask = torch.arange(kv_len)[None, :] > kv_len - 8 + torch.arange(8)[:, None]
    expected = torch.logsumexp(scores.masked_fill(mask, float("-inf")), dim=-1)
    torch.testing.assert_close(lse.squeeze(-1), expected, rtol=0, atol=1e-6)


def test_grouped_qk_keeps_extreme_scores_finite(golden):
    q = torch.full((1, 1, 2, 128), -256.0)
    k = torch.full((1, 1, 3, 128), -256.0)
    dq = torch.full_like(q, 2.0**-18)
    dk = torch.full_like(k, 2.0**120)
    actual = golden._compute_s_block(q, k, dq, dk, 0.088388348)
    expected = (q.double() * dq.double()) @ (k.double() * dk.double()).transpose(-1, -2)
    expected = (expected * 0.088388348).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0)


def test_extreme_softmax_uses_kernel_rounding(golden):
    scores = torch.tensor([1.970337850073242e36, 1.8809596092029503e36]).reshape(
        1, 1, 2, 1
    )
    maximum, total, probabilities = golden._online_softmax_update(
        scores,
        torch.zeros_like(scores, dtype=torch.bool),
        torch.full_like(scores, torch.finfo(torch.float32).min),
        None,
        None,
        torch.tensor([184.2]).log(),
    )
    lse = maximum + total.log()
    assert torch.isposinf(lse[0, 0, 0, 0])
    assert torch.isnan(probabilities[0, 0, 0, 0])
    assert lse[0, 0, 1, 0] == scores[0, 0, 1, 0]
    assert probabilities[0, 0, 1, 0] == 1


@pytest.mark.parametrize("with_tail", [False, True])
def test_aclgraph_forward_passes_optional_tail_inputs(with_tail):
    path = (
        Path(__file__).parent
        / "qfa_mxfp8_test"
        / "common"
        / "quant_flash_attn_golden.py"
    )
    tree = ast.parse(path.read_text())
    network = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Network"
    )
    namespace = {
        "nn": torch.nn,
        "quant_flash_attn_metadata": lambda **kwargs: None,
        "quant_flash_attn": lambda **kwargs: (kwargs, None),
        "_get_npu_fa_kwargs": lambda: {},
    }
    exec(
        compile(ast.Module(body=[network], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    model = namespace["Network"]()
    arguments = {name: None for name in inspect.signature(model.forward).parameters}
    arguments["q"] = SimpleNamespace(shape=(1, 1, 64))
    tail_names = ("v_tail", "block_table_tail", "seqused_v_tail")
    if with_tail:
        arguments.update({name: object() for name in tail_names})
    actual, _ = model.forward(**arguments)
    for name in tail_names:
        assert actual[name] is arguments[name]


@pytest.mark.parametrize("kv_len", [31, 319, 335])
@pytest.mark.parametrize("p_scale", [0.5, 2.0])
def test_sectioned_tail_matches_uniform_attention(golden, kv_len, p_scale):
    q_len, dim = 8, 64
    tail_len = kv_len % 64
    main_len = kv_len - tail_len
    chunks = [(i, i + 1) for i in range((kv_len + 255) // 256)]
    section_info = (64, 256, [[chunks]], 1, False)
    out, lse = golden._cpu_mxfp8_golden_sectioned(
        torch.zeros(1, 1, q_len, dim).to(torch.float8_e4m3fn),
        torch.ones(1, 1, kv_len, dim).to(torch.float8_e4m3fn),
        torch.ones(1, 1, kv_len, dim).to(torch.float8_e4m3fn),
        torch.ones(1, 1, q_len, 2),
        torch.ones(1, 1, kv_len, 2),
        torch.full((1, 1, (kv_len + 31) // 32, dim), 2.0),
        p_scale,
        [q_len],
        [kv_len],
        section_info,
        v_tail_bnsd=torch.full((1, 1, tail_len, dim), 6.0).bfloat16(),
        tail_lens=[tail_len],
    )
    counts = torch.arange(kv_len - q_len + 1, kv_len + 1).float()
    tail_counts = (counts - main_len).clamp(min=0)
    expected = (2 * (counts - tail_counts) + 6 * tail_counts) / counts
    torch.testing.assert_close(
        out[0, 0], expected[:, None].expand(-1, dim), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(lse[0, 0, :, 0], counts.log(), rtol=0, atol=1e-6)
