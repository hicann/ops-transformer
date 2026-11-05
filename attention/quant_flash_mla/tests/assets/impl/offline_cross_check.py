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


"""Portable offline references for QFMLA's custom cross-check.

Keep auxiliary files outside TTK's per-case directories: its manual-data writer
replaces those directories and its loader rejects unrecognized files in them.
Only arrays and JSON are stored; loading never uses pickle or invokes an NPU.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

logger = logging.getLogger(__name__)
_DIRECTORY = "_qfmla_cross_check"
_VERSION = 1
_DEFAULTS = {
    "quant_mode": 1,
    "softmax_scale": 1.0 / (576**0.5),
    "mask_mode": 0,
    "max_seqlen_q": -1,
    "max_seqlen_kv": -1,
    "head_dim_v": 512,
    "layout_q": "TND",
    "layout_kv": "PA_BNBD",
    "layout_out": "TND",
    "return_softmax_lse": False,
}


def _switches():
    try:
        from ttk.switches import get_global_storage
    except ImportError:
        from ttk.utilities.container_utils import get_global_storage
    return get_global_storage()


def _signature(inputs, attributes):
    # Metadata is generated on the target machine and deliberately excluded.
    # All nine data inputs, including actual sequence/page-table contents, bind
    # these references to the dataset rather than just a repeated testcase name.
    if inputs is None or len(inputs) < 9:
        raise ValueError("offline cross_check requires all nine data input slots")
    digest = hashlib.sha256()
    for value in inputs[:9]:
        if value is None:
            digest.update(b"none;")
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            shape, itemsize = list(value.shape), value.element_size()
            payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        else:
            value = np.ascontiguousarray(value)
            shape, itemsize = list(value.shape), value.dtype.itemsize
            payload = value.tobytes()
        digest.update(json.dumps([shape, itemsize]).encode("utf-8"))
        digest.update(payload)
    attrs = {}
    for name, default in _DEFAULTS.items():
        value = attributes.get(name, default)
        if value is None:
            value = default
        attrs[name] = type(default)(value)
    digest.update(json.dumps(attrs, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _path(root, testcase_name):
    token = hashlib.sha256(testcase_name.encode("utf-8")).hexdigest()
    return Path(root).expanduser().resolve() / _DIRECTORY / (token + ".npz")


def save(testcase_name, inputs, attributes, bench, reference):
    """Save during CPU --no-prof --dump golden manual-data preparation only."""
    switches = _switches()
    dump = getattr(switches, "dump_config", None)
    roots = tuple(getattr(switches, "manual_data_dirs", ()) or ())
    if not (
        getattr(switches, "no_prof", False)
        and getattr(dump, "dump_golden", False)
        and roots
    ):
        return
    if len(roots) != 1:
        raise ValueError(
            "offline cross_check preparation needs one manual-data directory"
        )
    if bench is None or reference is None or len(bench) != len(reference):
        raise ValueError("offline cross_check preparation requires bench and reference")
    arrays = {}
    manifest = {
        "version": _VERSION,
        "testcase_name": testcase_name,
        "signature": _signature(inputs, attributes),
        "slots": {},
    }
    for role, values in (("bench", bench), ("reference", reference)):
        manifest["slots"][role] = [value is not None for value in values]
        for index, value in enumerate(values):
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu()
                # np has no native torch.bfloat16; FP32 preserves its exact values.
                if value.dtype == torch.bfloat16:
                    value = value.float()
                value = value.numpy()
            arrays[f"{role}_{index}"] = np.asarray(value)
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    target = _path(roots[0], testcase_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".npz", delete=False
        ) as f:
            temporary = Path(f.name)
            np.savez(f, **arrays)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    logger.info(
        "[%s] saved offline cross_check bench/reference: %s", testcase_name, target
    )


def load(compare_context):
    """Restore references for replay, rejecting another case/input/attribute set."""
    switches = _switches()
    roots = tuple(getattr(switches, "manual_data_dirs", ()) or ())
    if not roots or getattr(switches, "no_prof", False):
        return None
    if compare_context is None or not getattr(compare_context, "testcase_name", None):
        raise ValueError(
            "offline cross_check needs TTK compare_context with testcase_name"
        )
    from ttk.core_modules.common.dump.manual_data import case_directory_name

    name = compare_context.testcase_name
    # Select the same first case directory as TTK's manual-data loader.
    root = next(
        (
            root
            for root in roots
            if (Path(root).expanduser() / case_directory_name(name)).is_dir()
        ),
        None,
    )
    if root is None:
        return None
    source = _path(root, name)
    if not source.is_file():
        raise ValueError(
            f"offline cross_check references missing: {source}; regenerate CPU data "
            f"with --compare cross_check --no-prof --dump in,golden, and copy "
            f"the entire manual-data root including {_DIRECTORY}"
        )
    expected = _signature(compare_context.input_tensors, compare_context.attributes)
    with np.load(source, allow_pickle=False) as saved:
        manifest = json.loads(saved["manifest"].tobytes().decode("utf-8"))
        if manifest.get("version") != _VERSION or manifest.get("testcase_name") != name:
            raise ValueError(f"incompatible offline cross_check manifest: {source}")
        if manifest.get("signature") != expected:
            raise ValueError(
                f"offline cross_check input/attribute mismatch for {name}; regenerate dataset"
            )
        values = tuple(
            [
                saved[f"{role}_{index}"].copy() if present else None
                for index, present in enumerate(manifest["slots"][role])
            ]
            for role in ("bench", "reference")
        )
    logger.info("[%s] loaded offline cross_check bench/reference: %s", name, source)
    return values
