# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pure geometry for MQSMLA quant_mode=4 tables; no tensor generation."""

import math


def describe_mixed_quant_sparse_flash_mla(params):
    """Expand mode-4 geometry; API attributes still belong in the attributes cell."""

    def integer(name, alias=None, default=None):
        value = params.get(name, params.get(alias, default))
        if alias in params and name in params and params[alias] != value:
            raise ValueError(f"conflicting {name}/{alias} values")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def flag(name, default=True):
        value = params.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False")
        return value

    allowed = {
        "B",
        "S",
        "S1",
        "S2",
        "S3",
        "T1",
        "T2",
        "T3",
        "N",
        "N1",
        "N2",
        "D",
        "K1",
        "K",
        "block_size1",
        "block_size2",
        "num_blocks1",
        "num_blocks2",
        "layout_q",
        "layout_kv",
        "quant_mode",
        "has_cmp",
        "use_sinks",
        "use_ori_topk_length",
        "use_cmp_topk_length",
        "use_seqused_q",
        "use_seqused_ori_kv",
        "use_seqused_cmp_kv",
        "metadata_shape",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"unsupported geometry columns: {sorted(unknown)}; "
            "runtime attributes and *_values must be in attributes"
        )
    if params.get("quant_mode", 4) != 4:
        raise ValueError(
            "parameter-table describe currently supports MQSMLA quant_mode=4 only"
        )
    b, s, n, d = (integer("B"), integer("S1", "S"), integer("N1", "N"), integer("D"))
    if d != 512 or n > 64 or integer("N2", default=1) != 1:
        raise ValueError("MQSMLA quant_mode=4 requires D=512, N1 in 1..64 and N2=1")
    lq, lk = params.get("layout_q", "BSND"), params.get("layout_kv", "BSND")
    if (lq, lk) not in (
        ("TND", "TND"),
        ("BSND", "BSND"),
        ("TND", "PA_BBND"),
        ("BSND", "PA_BBND"),
    ):
        raise ValueError("unsupported MQSMLA quant_mode=4 metadata layout combination")
    for name, applicable in (
        ("T1", lq == "TND"),
        ("T2", lk == "TND"),
        ("T3", lk == "TND"),
        ("num_blocks1", lk == "PA_BBND"),
        ("num_blocks2", lk == "PA_BBND"),
    ):
        if name in params and not applicable:
            raise ValueError(f"{name} does not apply to the declared layout")
    has_cmp = flag("has_cmp")
    q = (b, s, n, d) if lq == "BSND" else (integer("T1", default=b * s), n, d)
    shapes = [q] + [None] * 17
    types = ["bfloat16"] + [None] * 17

    def slot(index, shape, dtype="int32"):
        shapes[index], types[index] = shape, dtype

    for side, index, width in (("ori", 1, 544), ("cmp", 2, 320)):
        if side == "cmp" and not has_cmp:
            continue
        seq = integer("S2" if side == "ori" else "S3", default=params.get("S2"))
        topk = integer("K1" if side == "ori" else "K")
        if lk == "PA_BBND":
            block = integer(
                "block_size1" if side == "ori" else "block_size2", default=128
            )
            pages = math.ceil(seq / block)
            blocks = integer(
                "num_blocks1" if side == "ori" else "num_blocks2", default=b * pages
            )
            slot(index, (blocks, block, 1, width), "uint8")
            slot(index + 4, (b, pages))
        else:
            total = integer("T2" if side == "ori" else "T3", default=b * seq)
            kv = (b, seq, 1, width) if lk == "BSND" else (total, 1, width)
            slot(index, kv, "uint8")
            if lk == "TND":
                slot(index + 7, (b + 1,))
        slot(index + 2, (*q[:-2], 1, topk))
        if flag(f"use_seqused_{side}_kv"):
            slot(index + 10, (b,))
        if flag(f"use_{side}_topk_length"):
            slot(index + 13, (*q[:-2], 1))
    if lq == "TND":
        slot(7, (b + 1,))
    if flag("use_seqused_q"):
        slot(10, (b,))
    if flag("use_sinks"):
        slot(16, (n,), "float32")
    metadata_shape = params.get("metadata_shape", (-1,))
    if (
        not isinstance(metadata_shape, (tuple, list))
        or not metadata_shape
        or any(type(dim) is not int or dim < -1 for dim in metadata_shape)
    ):
        raise ValueError(
            "metadata_shape must be a nonempty tuple of integer dimensions >= -1"
        )
    slot(17, tuple(metadata_shape))
    return {"shapes": tuple(shapes), "dtypes": tuple(types)}
