# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from test_constants import D, N1, KV_PAGE_PADDING_BYTES


FORMAT = "mqsmla-dsl-input-golden-v2"
KV_LAYOUT = "nope448-rope64-scale"


def case_path(directory, name, index, params):
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:120] or "case"

    def tensor_values(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        raise TypeError(f"Unsupported saved parameter type: {type(value).__name__}")

    digest = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=tensor_values).encode()
    ).hexdigest()[:16]
    return Path(directory).expanduser().resolve() / f"{index:04d}_{slug}_{digest}.pt"


def save_case(path, prepared):
    params, case, golden, golden_lse = prepared
    payload = dict(
        format=FORMAT,
        kv_layout=KV_LAYOUT,
        params=params,
        case=case,
        golden=golden,
        golden_lse=golden_lse,
        page_padding_bytes=KV_PAGE_PADDING_BYTES,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed/interrupted write must not leave a partial .pt file for replay.
    fd, temporary = tempfile.mkstemp(prefix=".mqsmla-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"[mqsmla save] {path}")


def list_cases(directory):
    path = Path(directory).expanduser().resolve()
    files = sorted(path.glob("*.pt")) if path.is_dir() else []
    if not files:
        raise ValueError(f"No saved .pt cases in {path}; run batch_save first")
    return files


def load_case(path):
    """Load only tensors/basic values, on CPU, with a checked format version."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(
                f"expected format {FORMAT}; regenerate old data with batch_save"
            )
        if data.get("kv_layout") != KV_LAYOUT:
            raise ValueError(
                f"expected KV layout {KV_LAYOUT}; regenerate with batch_save"
            )
        params, case = data["params"], data["case"]
        golden, lse = data["golden"], data["golden_lse"]
        if data["page_padding_bytes"] != KV_PAGE_PADDING_BYTES:
            raise ValueError("saved KV padding does not match the test framework")
        if case["sinks"].shape != (N1,) or case["sinks"].dtype != torch.float32:
            raise ValueError(
                "saved sinks must be FP32 [N1]; regenerate old batch_save data"
            )
        t1 = case["t1"]
        if case["q"].shape != (t1, N1, D) or case["q"].dtype != torch.bfloat16:
            raise ValueError("invalid saved q shape/dtype")
        if golden.shape != (t1 * N1, D) or golden.dtype != torch.bfloat16:
            raise ValueError("invalid golden shape/dtype")
        if lse.shape != (1, t1, N1) or lse.dtype != torch.float32:
            raise ValueError("invalid golden LSE shape/dtype")
        for key in ("kv_axis0_noncontiguous", "template_run_mode"):
            if key not in params:
                raise ValueError(f"missing saved parameter {key}")
        if params["template_run_mode"] != case["mode"]:
            raise ValueError("saved mode mismatch")
        return params, case, golden, lse
    except Exception as exc:
        raise ValueError(f"Cannot load MQSMLA case {path}: {exc}") from exc
