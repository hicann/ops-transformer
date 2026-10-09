#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import math
import torch
import itertools
import importlib

_PYT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_testcase_files(testset=None):
    tc_dir = os.path.join(_PYT_DIR, "testcase")
    if not os.path.isdir(tc_dir):
        raise FileNotFoundError(f"testcase directory not found: {tc_dir}")
    all_files = sorted(
        f for f in os.listdir(tc_dir) if f.endswith(".py") and not f.startswith("_")
    )
    if testset:
        matched = [f for f in all_files if testset in f]
        if not matched:
            raise ValueError(f"No testcase files matched testset='{testset}'")
        all_files = matched
    return [os.path.splitext(f)[0] for f in all_files]


def expand_cases(module_names):
    all_cases = {}
    for mod_name in module_names:
        mod = importlib.import_module(f"testcase.{mod_name}")
        for case_name, case_dict in mod.TestCases.items():
            keys = list(case_dict.keys())
            values = [
                case_dict[k] if isinstance(case_dict[k], list) else [case_dict[k]]
                for k in keys
            ]
            combos = list(itertools.product(*values))
            if len(combos) == 1:
                all_cases[f"{mod_name}/{case_name}"] = dict(zip(keys, combos[0]))
            else:
                for i, combo in enumerate(combos):
                    all_cases[f"{mod_name}/{case_name}_{i}"] = dict(zip(keys, combo))
    return all_cases


def normalize_params(raw):
    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    layout_kv = c.get("layout_kv", layout_q)
    c["layout_q"] = layout_q
    c["layout_kv"] = layout_kv
    c.setdefault("layout_kv", layout_kv)
    c.setdefault("layout_attn_out", layout_q)
    c.setdefault("win_left", -1)
    c.setdefault("win_right", -1)
    c.setdefault("mask_mode", 0)
    c.setdefault("quant_compute_mode", c.get("quant_compute_mode"))
    c.setdefault("q_dtype", c.get("q_dtype", "bf16"))
    c.setdefault(
        "kv_dtype", "fp4_e2m1" if c.get("quant_compute_mode", 1) != 2 else "fp4_e1m2"
    )

    from core.gen_data import resolve_kv_scale_dtype

    kv_dtype = c["kv_dtype"]
    q_dtype = c["q_dtype"]
    quant_compute_mode = c["quant_compute_mode"]
    c.setdefault("kv_scale_dtype", resolve_kv_scale_dtype(quant_compute_mode, q_dtype))

    d = c["D"]
    c.setdefault("softmax_scale", 1.0 / (d**0.5))

    b = c.get("B")
    s1 = c.get("S1")
    s2 = c.get("S2")
    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None or c.get(key) == []:
            c.pop(key, None)

    if layout_q == "TND":
        if "seqused_q" in c:
            c.setdefault(
                "cu_seqlens_q",
                torch.tensor(
                    [0] + [sum(c["seqused_q"][:i]) for i in range(len(c["seqused_q"]))],
                    dtype=torch.int32,
                ),
            )
        else:
            c.setdefault(
                "cu_seqlens_q",
                torch.tensor([i * s1 for i in range(b + 1)], dtype=torch.int32),
            )
    if layout_kv == "TND":
        if "seqused_kv" in c:
            c.setdefault(
                "cu_seqlens_kv",
                torch.tensor(
                    [0]
                    + [sum(c["seqused_kv"][:i]) for i in range(len(c["seqused_kv"]))],
                    dtype=torch.int32,
                ),
            )
        else:
            c.setdefault(
                "cu_seqlens_kv",
                torch.tensor([i * s2 for i in range(b + 1)], dtype=torch.int32),
            )
    if "PA" in layout_kv:
        c.setdefault("block_size", 128)
        if "seqused_kv" not in c:
            c["seqused_kv"] = torch.tensor([s2] * b, dtype=torch.int32)
        if isinstance(c.get("block_table"), list):
            bt_raw = c["block_table"]
            c["block_table"] = torch.tensor(bt_raw, dtype=torch.int32)
            c["block_num"] = max(
                int(max(max(row) for row in bt_raw)) + 1, c.get("block_num", 0)
            )
        else:
            seqused_kv = torch.tensor(c["seqused_kv"], dtype=torch.int32)
            block_size = c["block_size"]
            min_block_num = sum([math.ceil(x / block_size) for x in seqused_kv])
            c["block_num"] = max(min_block_num, c.get("block_num", 0))

    # Resolve max_seqlen_q / max_seqlen_kv defaults (-1 means auto from seqused/s2)
    if c.get("max_seqlen_q", -1) == -1:
        if "seqused_q" in c and c["seqused_q"] is not None:
            sq = c["seqused_q"]
            c["max_seqlen_q"] = int(
                max(sq.tolist() if isinstance(sq, torch.Tensor) else sq)
            )
        else:
            c["max_seqlen_q"] = s1
    if c.get("max_seqlen_kv", -1) == -1:
        if "seqused_kv" in c and c["seqused_kv"] is not None:
            skv = c["seqused_kv"]
            c["max_seqlen_kv"] = int(
                max(skv.tolist() if isinstance(skv, torch.Tensor) else skv)
            )
        else:
            c["max_seqlen_kv"] = s2
    return c


def resolve_case_ids(case_id_arg, all_cases):
    if case_id_arg == "all":
        return sorted(all_cases.keys())
    ids = [x.strip() for x in case_id_arg.split(",")]
    result = []
    missing = []
    for cid in ids:
        matches = [
            k
            for k in all_cases
            if k.endswith(f"/{cid}")
            or k == cid
            or k.rsplit("/", 1)[-1].startswith(f"{cid}_")
        ]
        if matches:
            result.extend(matches)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] cases not found: {missing}")
    return result


def get_data_dir(case_name, testset="testset1"):
    testcase = case_name.split("/")[0]
    base = os.path.join(_PYT_DIR, "data", testset)
    safe_name = case_name.replace("/", "_")
    case_dir = os.path.join(base, safe_name)
    os.makedirs(case_dir, exist_ok=True)
    return case_dir
