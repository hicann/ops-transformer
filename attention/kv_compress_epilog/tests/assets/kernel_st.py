#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel ST plugin: shared golden plus archived deterministic slot patterns."""

import importlib.util
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("kce_shared_golden", _HERE / "golden.py")
_GOLDEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GOLDEN)
with (_HERE / "kernel_st_slots.json").open(encoding="utf-8") as _file:
    _SLOTS = json.load(_file)


def _make_big_slots(spec):
    """Preserve the archived big50 normal/negative/out-of-range slot sequence."""
    tokens, rows, block_size, profile = spec
    slots = np.linspace(0, rows - 1, tokens).astype(np.int64)
    slots[:2] = -1
    slots[2] = rows + 5
    if tokens > 3:
        slots[3] = -1
    used = set(int(value) for value in slots)
    if profile == 2 and tokens > 4:
        candidates = [
            0,
            block_size - 1,
            block_size,
            2 * block_size - 1,
            (rows // block_size - 2) * block_size,
            (rows // block_size - 2) * block_size + block_size - 1,
            rows - block_size,
            rows - 1,
        ]
        index = 4
        for value in candidates:
            if index >= tokens:
                break
            if 0 <= value < rows and value not in used:
                slots[index] = value
                used.add(value)
                index += 1
    return slots


class KvCompressEpilogTestSpec:
    golden = staticmethod(_GOLDEN.KvCompressEpilogTestSpec.golden)
    third_party = _GOLDEN.KvCompressEpilogTestSpec.third_party
    tolerance = _GOLDEN.KvCompressEpilogTestSpec.tolerance

    @staticmethod
    def customize_inputs(cache, x, slot_mapping, **kwargs):
        name = kwargs.get("testcase_name")
        if name not in _SLOTS:
            raise ValueError("Kernel ST slot specification missing for %r" % name)
        spec = _SLOTS[name]
        slots = (
            np.asarray(spec["slots"], dtype=np.int64)
            if "slots" in spec
            else _make_big_slots(spec["big"])
        )
        if tuple(slot_mapping.shape) != slots.shape:
            raise ValueError("Kernel ST slot shape mismatch for %s" % name)
        if hasattr(slot_mapping, "copy_"):
            import torch

            slot_mapping.copy_(
                torch.as_tensor(
                    slots, dtype=slot_mapping.dtype, device=slot_mapping.device
                )
            )
        else:
            slot_mapping[...] = slots.astype(slot_mapping.dtype)
        return cache, x, slot_mapping


__spec__ = {"kv_compress_epilog": "KvCompressEpilogTestSpec"}
