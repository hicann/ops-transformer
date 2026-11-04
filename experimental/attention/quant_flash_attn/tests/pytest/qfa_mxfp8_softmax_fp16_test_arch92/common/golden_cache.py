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

import logging
import os

import torch

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "golden_cache"
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _path(case_name, suffix, cache_dir=None):
    d = cache_dir or DEFAULT_CACHE_DIR
    _ensure_dir(d)
    return os.path.join(d, f"{case_name}_{suffix}.pt")


def save_input(case_name, data_dict, cache_dir=None):
    path = _path(case_name, "input", cache_dir)
    torch.save(data_dict, path)
    logger.info("[CACHE] save input → %s", path)


def load_input(case_name, cache_dir=None):
    path = _path(case_name, "input", cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached input: {path}")
    data = torch.load(path, weights_only=False)
    logger.info("[CACHE] load input ← %s", path)
    return (
        data["q_fp8"],
        data["k_fp8"],
        data["v_fp8"],
        data["dequant_scale_q"],
        data["dequant_scale_k"],
        data["v_descale"],
        data["p_scale"],
        data.get("qr_bf16"),
        data.get("kr_bf16"),
        data.get("block_table_torch"),
    )


def has_input(case_name, cache_dir=None):
    return os.path.exists(_path(case_name, "input", cache_dir))


def save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=None):
    path = _path(case_name, "cpu_output", cache_dir)
    torch.save({"cpu_out": cpu_out, "cpu_lse": cpu_lse}, path)
    logger.info("[CACHE] save CPU output → %s", path)


def load_cpu_output(case_name, cache_dir=None):
    path = _path(case_name, "cpu_output", cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached CPU output: {path}")
    data = torch.load(path, weights_only=False)
    logger.info("[CACHE] load CPU output ← %s", path)
    return data["cpu_out"], data["cpu_lse"]


def has_cpu_output(case_name, cache_dir=None):
    return os.path.exists(_path(case_name, "cpu_output", cache_dir))


def save_npu_output(case_name, atten_out, lse_out, cache_dir=None):
    path = _path(case_name, "npu_output", cache_dir)
    torch.save({"atten_out": atten_out, "lse_out": lse_out}, path)
    logger.info("[CACHE] save NPU output → %s", path)


def load_npu_output(case_name, cache_dir=None):
    path = _path(case_name, "npu_output", cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached NPU output: {path}")
    data = torch.load(path, weights_only=False)
    logger.info("[CACHE] load NPU output ← %s", path)
    return data["atten_out"], data["lse_out"]


def has_npu_output(case_name, cache_dir=None):
    return os.path.exists(_path(case_name, "npu_output", cache_dir))


def build_input_dict(
    q_fp8,
    k_fp8,
    v_fp8,
    dequant_scale_q,
    dequant_scale_k,
    v_descale,
    p_scale,
    qr_bf16,
    kr_bf16,
    block_table_torch,
):
    return {
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "v_fp8": v_fp8,
        "dequant_scale_q": dequant_scale_q,
        "dequant_scale_k": dequant_scale_k,
        "v_descale": v_descale,
        "p_scale": p_scale,
        "qr_bf16": qr_bf16,
        "kr_bf16": kr_bf16,
        "block_table_torch": block_table_torch,
    }
