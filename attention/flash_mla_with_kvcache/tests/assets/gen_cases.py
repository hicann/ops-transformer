# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Generate 50 supported-range TTK cases (or the 16-case --suite smoke)."""

import argparse
import ast
import csv
import itertools
import math
from pathlib import Path


HEADER = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
]
API = "flash_mla_with_kvcache_ttk_ops.flash_mla_with_kvcache_ttk"
DTYPES = ("float16", "bfloat16")
COVERAGE_HEADER = HEADER + [
    "tensor_storage_shapes",
    "tensor_view_strides",
    "tensor_view_offsets",
    "remark",
]


def contiguous_stride(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def coverage_row(
    name,
    dtype,
    kv_lengths,
    q_lengths,
    *,
    layout="PA_BBND",
    mask=0,
    lse=True,
    no_seqused=False,
    omit_max=(),
    nc_axes=(),
    page_order="sequential",
    scale=576**-0.5,
    num_heads_q=96,
):
    """Small valid TND/NTD case; strides are element strides, not byte strides."""
    assert len(kv_lengths) == len(q_lengths) and all(
        x > 0 for x in kv_lengths + q_lengths
    )
    assert all(q <= kv for q, kv in zip(q_lengths, kv_lengths))
    batch = len(q_lengths)
    counts = [(length + 127) // 128 for length in kv_lengths]
    pages = sum(counts)
    kshape = (pages, 128, 1, 576) if layout == "PA_BBND" else (pages, 1, 36, 128, 16)
    assert num_heads_q in (64, 96)
    shapes = (
        (sum(q_lengths), num_heads_q, 576),
        kshape,
        (batch, max(counts)),
        (batch,),
        (batch + 1,),
        None if no_seqused else (batch,),
        (2048, 2048) if mask else None,
        (0,),
    )
    dtypes = (
        dtype,
        dtype,
        "int32",
        "int32",
        "int32",
        None if no_seqused else "int32",
        "int8" if mask else None,
        "int32",
    )
    attrs = dict(
        head_dim_v=512,
        softmax_scale=scale,
        mask_mode=mask,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv=layout,
        layout_out="NTD",
        return_softmax_lse=lse,
        cache_seqlens_values=kv_lengths,
        cu_seqlens_q_values=[0] + list(itertools.accumulate(q_lengths)),
        page_order=page_order,
    )
    if not no_seqused:
        attrs["seqused_q_values"] = q_lengths
    for key in omit_max:
        attrs.pop(key)
    ranges = tuple(
        (-1, 1) if i < 2 else (0, 0) if shape is not None else (None, None)
        for i, shape in enumerate(shapes)
    )
    row = dict(
        zip(
            HEADER,
            (
                f"mla_cov_{name}_{dtype}",
                API,
                repr(shapes),
                repr(dtypes),
                "",
                repr(attrs),
                "",
                "",
                repr(ranges),
                "((0.005, 2.5e-05),)",
                "1e-8",
            ),
        )
    )
    row["remark"] = name
    if nc_axes:
        assert set(nc_axes) <= ({0} if layout == "PA_BBND" else {0, 1})
        strides = [contiguous_stride(s) if s is not None else None for s in shapes]
        offsets = [0 if s is not None else None for s in shapes]
        storage = list(shapes)
        kv_stride = list(strides[1])
        for axis in nc_axes:
            kv_stride[axis] *= 2
        strides[1] = tuple(kv_stride)
        # Nonzero storage offset also checks TensorV2 view propagation.
        offsets[1] = 16
        storage[1] = (16 + 1 + sum((s - 1) * t for s, t in zip(kshape, kv_stride)),)
        row.update(
            tensor_storage_shapes=repr(tuple(storage)),
            tensor_view_strides=repr(tuple(strides)),
            tensor_view_offsets=repr(tuple(offsets)),
        )
    return row


def coverage_cases():
    # 16: all currently reachable dtype/layout/mask/LSE template combinations.
    rows = list(cases())
    for row in rows:
        row["remark"] = "template_matrix"
    # 20: S2 tile=112, page=128, two-tile=224; test both sides and exact boundaries.
    for index, length in enumerate((1, 111, 112, 113, 127, 128, 129, 223, 224, 225)):
        for dtype in DTYPES:
            rows.append(
                coverage_row(
                    f"n64_kv{length}",
                    dtype,
                    [length],
                    [1],
                    num_heads_q=64,
                    layout="PA_NZ" if index % 2 else "PA_BBND",
                    mask=3 if index % 2 else 0,
                    page_order="reverse",
                )
            )
    # 14: optional inputs/default attrs, stride, task pipeline and FD candidates.
    specials = [
        ("no_seqused", "float16", [257, 129], [2, 1], dict(no_seqused=True)),
        (
            "no_seqused_nz",
            "bfloat16",
            [257, 129],
            [2, 1],
            dict(no_seqused=True, layout="PA_NZ", mask=3),
        ),
        (
            "omit_max_both",
            "float16",
            [129],
            [3],
            dict(omit_max=("max_seqlen_q", "max_seqlen_kv")),
        ),
        (
            "omit_max_q",
            "bfloat16",
            [129],
            [3],
            dict(omit_max=("max_seqlen_q",), layout="PA_NZ"),
        ),
        (
            "stride_bbnd0",
            "float16",
            [257, 129],
            [3, 1],
            dict(nc_axes=(0,), page_order="reverse"),
        ),
        (
            "stride_nz0",
            "bfloat16",
            [257, 129],
            [3, 1],
            dict(nc_axes=(0,), layout="PA_NZ", mask=3),
        ),
        (
            "stride_nz1_degenerate",
            "float16",
            [257, 129],
            [3, 1],
            dict(nc_axes=(1,), layout="PA_NZ"),
        ),
        (
            "stride_nz01",
            "bfloat16",
            [257, 129],
            [3, 1],
            dict(nc_axes=(0, 1), layout="PA_NZ", mask=3),
        ),
        ("n64_q3_section_policy", "float16", [337], [3], dict(mask=3, num_heads_q=64)),
        (
            "n64_q17_pipeline",
            "bfloat16",
            [257],
            [17],
            dict(layout="PA_NZ", mask=3, num_heads_q=64),
        ),
        ("n64_q33_pipeline", "float16", [257], [33], dict(scale=0.125, num_heads_q=64)),
        (
            "decode_long_tail",
            "float16",
            [8193],
            [1],
            dict(layout="PA_NZ", page_order="reverse"),
        ),
        ("decode_long_aligned", "bfloat16", [4096], [2], dict(mask=3)),
        (
            "ragged4",
            "float16",
            [1, 112, 129, 337],
            [1, 2, 3, 1],
            dict(
                layout="PA_NZ",
                mask=3,
                omit_max=("max_seqlen_kv",),
                page_order="reverse",
            ),
        ),
    ]
    for name, dtype, kv, q, options in specials:
        rows.append(coverage_row(name, dtype, kv, q, **options))
    assert len(rows) == 50 and len({r["testcase_name"] for r in rows}) == 50
    # Bound input storage independently of host golden intermediates/workspace.
    for row in rows:
        shapes = ast.literal_eval(
            row.get("tensor_storage_shapes") or row["tensor_view_shapes"]
        )
        dtypes = ast.literal_eval(row["tensor_dtypes"])
        sizes = {"float16": 2, "bfloat16": 2, "int32": 4, "int8": 1}
        assert (
            sum(
                math.prod(s) * sizes[d] for s, d in zip(shapes, dtypes) if s is not None
            )
            < 16 * 1024**2
        )
    return rows


def cases():
    for dtype, layout_kv, mask, return_lse in itertools.product(
        ("float16", "bfloat16"),
        ("PA_BBND", "PA_NZ"),
        (0, 3),
        (False, True),
    ):
        # Variable lengths exercise partial pages and per-batch Q lengths.
        layout_q = "TND"
        lengths = [257, 129]
        qshape = (3, 96, 576)
        kshape = {"PA_BBND": (5, 128, 1, 576), "PA_NZ": (5, 1, 36, 128, 16)}[layout_kv]
        shapes = (
            qshape,
            kshape,
            (2, 3),
            (2,),
            (3,) if layout_q == "TND" else None,
            (2,),
            (2048, 2048) if mask == 3 else None,
            (0,),
        )
        dtypes = (
            dtype,
            dtype,
            "int32",
            "int32",
            "int32" if layout_q == "TND" else None,
            "int32",
            "int8" if mask == 3 else None,
            "int32",
        )
        attrs = dict(
            head_dim_v=512,
            softmax_scale=576**-0.5,
            mask_mode=mask,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out="NTD",
            return_softmax_lse=return_lse,
            cache_seqlens_values=lengths,
            seqused_q_values=[2, 1],
        )
        if layout_q == "TND":
            attrs["cu_seqlens_q_values"] = [0, 2, 3]
        ranges = tuple(
            (-1, 1) if i < 2 else (0, 0) if shape is not None else (None, None)
            for i, shape in enumerate(shapes)
        )
        yield dict(
            zip(
                HEADER,
                (
                    f"mla_{dtype}_{layout_q}_{layout_kv}_mask{mask}_lse{int(return_lse)}",
                    API,
                    repr(shapes),
                    repr(dtypes),
                    "",
                    repr(attrs),
                    "",
                    "",
                    repr(ranges),
                    "((0.005, 2.5e-05),)",
                    "1e-8",
                ),
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--suite", choices=("smoke", "coverage"), default="coverage")
    args = parser.parse_args()
    requested = args.output or Path(
        f"testcase/flash_mla_with_kvcache_e2e_{args.suite}.csv"
    )
    output = requested if requested.is_absolute() else Path(__file__).parent / requested
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=COVERAGE_HEADER if args.suite == "coverage" else HEADER,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(coverage_cases() if args.suite == "coverage" else cases())


if __name__ == "__main__":
    main()
