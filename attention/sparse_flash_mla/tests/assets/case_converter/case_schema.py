# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Literal parsing and TTK tensor descriptors for this operator."""

import ast
import math

OPERATOR = "sparse_flash_mla"
INPUT_COUNT = 18
E2E_API = "torch.ops.cann_ops_transformer.sparse_flash_mla"
ACLNN_API = "aclnnSparseFlashMla"
API_ALIASES = ("torch.ops.cann_ops_transformer.sparse_flash_mla",)
METADATA_INDEX = 17
ATTRIBUTES = (
    ("softmaxScale", "softmax_scale", 1.0),
    ("cmpRatio", "cmp_ratio", 1),
    ("oriMaskMode", "ori_mask_mode", 0),
    ("cmpMaskMode", "cmp_mask_mode", 0),
    ("oriWinLeft", "ori_win_left", -1),
    ("oriWinRight", "ori_win_right", -1),
    ("layoutQOptional", "layout_q", "BSND"),
    ("layoutKvOptional", "layout_kv", "BSND"),
    ("topkValueMode", "topk_value_mode", 1),
    ("returnSoftmaxLse", "return_softmax_lse", False),
)
FIELDS = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_storage_shapes",
    "tensor_view_strides",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "is_enabled",
)
OPTIONAL_FIELDS = (
    "network_name",
    "priority",
    "remark",
    "soc_series",
    "golden_api",
    "device_ids",
    "batch_axis",
    "batch_slice_info",
    "batch_seed",
    "manual_tensor_binaries",
    "manual_golden_binaries",
    "dump_file_prefix",
    "tensor_formats",
    "tensor_view_offsets",
    "inplace_input_indexes",
)
DTYPES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "float": "float32",
    "hif8": "uint8",
    "hifp8": "uint8",
    "hifloat8": "uint8",
    "fp8_e4m3fn": "float8_e4m3fn",
    "fp8_e8m0": "uint8",
    "float8_e8m0fnu": "uint8",
    "float4_e2m1fn_x2": "uint8",
    "fp4_e2m1": "uint8",
    "hif4": "uint8",
}


class LiteralNumbers(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id.lower() in ("inf", "nan"):
            return ast.copy_location(ast.Constant(float(node.id)), node)
        return node


def literal(value):
    """Parse literals and INF/NAN tokens without executing workbook expressions."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in ("", "none", "null"):
        return None
    try:
        return ast.literal_eval(LiteralNumbers().visit(ast.parse(text, mode="eval")))
    except (ValueError, SyntaxError):
        return text


def cell(row, *names, default=None):
    supplied = [
        (name, literal(row[name]))
        for name in names
        if name in row and literal(row[name]) is not None
    ]
    if len(supplied) > 1 and any(
        (value != supplied[0][1] for _, value in supplied[1:])
    ):
        raise ValueError(f"conflicting columns: {[name for name, _ in supplied]}")
    return supplied[0][1] if supplied else default


def required(row, *names):
    value = cell(row, *names)
    if value is None:
        raise ValueError(f"missing required column {names[0]!r}")
    return value


def integer(value, name, minimum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
        if not isinstance(value, (int, str)) and value != number:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


def number(row, name, *aliases, default=None, minimum=None):
    value = cell(row, name, *aliases, default=default)
    if value is None:
        raise ValueError(f"missing required column {name!r}")
    return integer(value, name, minimum)


def boolean(value):
    if value in (True, False, 0, 1):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "y", "1"):
            return True
        if value.lower() in ("false", "no", "n", "0"):
            return False
    raise ValueError(f"expected a boolean, got {value!r}")


def dtype(value):
    value = str(value).lower().removeprefix("torch.")
    value = DTYPES.get(value, value)
    supported = {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int32",
        "int64",
        "bool",
        "float8_e4m3fn",
        "hifloat8",
        "float8_e8m0",
        "float4_e2m1",
    }
    if value not in supported:
        raise ValueError(f"unsupported dtype: {value!r}")
    return value


def vector(row, name, *aliases):
    value = cell(row, name, name + "_values", *aliases)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (tuple, list)):
        value = [value]
    return [integer(item, name, 0) for item in value]


def data_range(row, name, default):
    value = cell(row, name, default=default)
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    result = tuple((float(item) for item in value))
    if math.isnan(result[0]) or math.isnan(result[1]):
        if not all((math.isnan(item) for item in result)):
            raise ValueError(f"{name}: NaN must be specified for both endpoints")
    elif result[0] > result[1]:
        raise ValueError(f"{name} must be an ordered range")
    return result


def prefix(lengths):
    result = [0]
    for value in lengths:
        result.append(result[-1] + value)
    return result


def lengths(prefix_values, name, batch):
    if len(prefix_values) != batch + 1 or prefix_values[0] != 0:
        raise ValueError(f"{name} must start with zero and contain B+1 values")
    result = [right - left for left, right in zip(prefix_values, prefix_values[1:])]
    if min(result, default=0) < 0:
        raise ValueError(f"{name} must be nondecreasing")
    return result


def sequence_lengths(row, name, batch, capacity, prefix_name=None):
    values = vector(row, name)
    cumulative = vector(row, prefix_name) if prefix_name else None
    if cumulative is not None:
        capacities = lengths(cumulative, prefix_name, batch)
    else:
        capacities = [capacity] * batch
    if values is None:
        values = capacities
    elif len(values) == 1:
        values *= batch
    if len(values) != batch or any(
        (value > bound for value, bound in zip(values, capacities))
    ):
        raise ValueError(f"{name}: lengths exceed per-batch capacities")
    return (values, cumulative or prefix(capacities))


def slot(shape, type_name, value_range):
    if shape is None:
        return (None, None, None)
    return (tuple(shape), type_name, value_range)


def tensor_fields(tensors, storages=None, strides=None):
    shapes, dtypes, ranges = zip(*tensors)
    return {
        "tensor_view_shapes": repr(shapes),
        "tensor_dtypes": repr(dtypes),
        "input_data_ranges": repr(ranges),
        "tensor_storage_shapes": repr(tuple(storages)) if storages is not None else "",
        "tensor_view_strides": repr(tuple(strides)) if strides is not None else "",
    }


def normalize_attributes(attributes):
    values = dict(attributes)
    for target, source, default in ATTRIBUTES:
        if source not in values:
            if default is None:
                raise ValueError(f"missing required API attribute {source!r}")
            values[source] = default
        if values[source] is None:
            raise ValueError(f"API attribute {source!r} must not be None")
        if target != source and target in values and (values[target] != values[source]):
            raise ValueError(f"conflicting API attributes {source!r}/{target!r}")
    return values


def validate_tensors(record):
    values = {
        field: literal(record.get(field))
        for field in (
            "tensor_view_shapes",
            "tensor_dtypes",
            "input_data_ranges",
            "tensor_storage_shapes",
            "tensor_view_strides",
        )
    }
    shapes = values["tensor_view_shapes"]
    dtypes = values["tensor_dtypes"]
    if (
        not isinstance(shapes, (tuple, list))
        or not isinstance(dtypes, (tuple, list))
        or len(shapes) != INPUT_COUNT
        or (len(dtypes) != INPUT_COUNT)
    ):
        raise ValueError(f"expected {INPUT_COUNT} input tensor shapes and dtypes")
    if shapes[0] is None:
        raise ValueError("query tensor must be present")
    for index, (shape, type_name) in enumerate(zip(shapes, dtypes)):
        if (shape is None) != (type_name is None):
            raise ValueError(
                f"tensor slot {index}: shape/dtype must both be present or absent"
            )
        if shape is not None:
            if not isinstance(shape, (tuple, list)):
                raise ValueError(f"tensor slot {index}: shape must be a tuple/list")
            if any(
                (isinstance(dim, bool) or not isinstance(dim, int) for dim in shape)
            ):
                raise ValueError(
                    f"tensor slot {index}: shape dimensions must be integers"
                )
            if tuple(shape) != (-1,) and any(
                (integer(dim, "shape dimension", 0) != dim for dim in shape)
            ):
                raise ValueError(f"tensor slot {index}: invalid shape")
            if tuple(shape) == (-1,) and index != METADATA_INDEX:
                raise ValueError("only the metadata input supports a dynamic shape")
            dtype(type_name)
    for field, sequence in values.items():
        if sequence is None:
            continue
        if not isinstance(sequence, (tuple, list)) or len(sequence) != INPUT_COUNT:
            raise ValueError(f"{field}: expected {INPUT_COUNT} slots")
        if field == "input_data_ranges":
            for index, value_range in enumerate(sequence):
                if value_range is None:
                    continue
                if (
                    shapes[index] is None
                    or not isinstance(value_range, (tuple, list))
                    or len(value_range) != 2
                ):
                    raise ValueError(
                        f"input_data_ranges slot {index}: expected a range for a present tensor"
                    )
                if all((item is not None for item in value_range)):
                    data_range({field: value_range}, field, (0, 1))
        if (
            field in ("tensor_storage_shapes", "tensor_view_strides")
            and METADATA_INDEX is not None
            and (tuple(shapes[METADATA_INDEX] or ()) == (-1,))
            and (sequence[METADATA_INDEX] is not None)
        ):
            raise ValueError(
                "dynamic metadata must not specify storage shape or stride"
            )
        if field in ("tensor_storage_shapes", "tensor_view_strides"):
            for index, entry in enumerate(sequence):
                if entry is None:
                    continue
                if shapes[index] is None or not isinstance(entry, (tuple, list)):
                    raise ValueError(
                        f"{field} slot {index}: invalid optional tensor descriptor"
                    )
                if field == "tensor_view_strides" and len(entry) != len(shapes[index]):
                    raise ValueError(
                        f"{field} slot {index}: stride rank must match tensor rank"
                    )
                for dim in entry:
                    integer(dim, field, 0)
    for field in ("tensor_formats", "tensor_view_offsets"):
        sequence = literal(record.get(field))
        if sequence is not None and (
            not isinstance(sequence, (tuple, list)) or len(sequence) != INPUT_COUNT
        ):
            raise ValueError(f"{field}: expected {INPUT_COUNT} slots")
    return values


def convert_case(raw, mode):
    """Convert one parameter or E2E descriptor row, retaining explicit options."""
    if mode not in ("e2e", "aclnn"):
        raise ValueError(f"unsupported mode: {mode}")
    case = str(
        required(
            raw,
            "testcase_name",
            "Testcase_Name",
            "Testcase_Prefix",
            "case_name",
            "tc_prefix",
        )
    )
    if "tensor_view_shapes" in raw:
        record = {
            field: str(raw[field]) if raw.get(field) is not None else ""
            for field in FIELDS
            if field in raw
        }
        if raw.get("output_tensor_indexes") not in (None, ""):
            raise ValueError(
                "source must be an E2E descriptor or pytest parameters, not ACLNN output"
            )
        api = cell(raw, "api_name", default=E2E_API)
        if api not in API_ALIASES:
            raise ValueError(f"API {api!r} does not belong to {OPERATOR}")
        attributes = literal(raw.get("attributes")) or {}
    else:
        record, attributes = parameter_case(raw)
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be a dictionary literal")
    for index in range(1, 10):
        extra = literal(raw.get(f"attributes{index}"))
        if extra is not None:
            if not isinstance(extra, dict):
                raise ValueError(f"attributes{index} must be a dictionary literal")
            attributes.update(extra)
    for name in ("precision_tolerances", "absolute_precision"):
        if raw.get(name) is not None:
            record[name] = str(raw[name])
    for name in OPTIONAL_FIELDS:
        if raw.get(name) not in (None, ""):
            record[name] = raw[name]
    record["testcase_name"] = case
    record["is_enabled"] = str(boolean(cell(raw, "is_enabled", default=True)))
    record["api_name"] = E2E_API
    record["attributes"] = repr(normalize_attributes(attributes))
    record.setdefault("precision_tolerances", "")
    record.setdefault("absolute_precision", "")
    validate_tensors(record)
    if mode == "aclnn":
        record = aclnn_case(record)
    return record


def aclnn_case(record):
    """Append API outputs and translate input storage and scalar conventions."""
    record = dict(record)
    values = validate_tensors(record)
    attributes = literal(record["attributes"])
    shapes = tuple(values["tensor_view_shapes"])
    types = tuple(values["tensor_dtypes"])
    output_shapes, output_types = output_contract(shapes, types, attributes)
    shapes, types, values = aclnn_inputs(shapes, types, attributes, values)
    for field, added in (
        ("tensor_view_shapes", output_shapes),
        ("tensor_dtypes", output_types),
        ("input_data_ranges", (None,) * len(output_shapes)),
        ("tensor_storage_shapes", output_shapes),
        ("tensor_view_strides", (None,) * len(output_shapes)),
    ):
        current = (
            shapes
            if field == "tensor_view_shapes"
            else types
            if field == "tensor_dtypes"
            else values[field]
        )
        record[field] = (
            repr(tuple(current) + tuple(added)) if current is not None else ""
        )
    for field, added in (("tensor_formats", "ND"), ("tensor_view_offsets", 0)):
        current = literal(record.get(field))
        if current is not None:
            record[field] = repr(tuple(current) + (added,) * len(output_shapes))
    for target, source, _ in ATTRIBUTES:
        if target != source:
            attributes[target] = attributes.pop(source)
    # Tensor values use a separate name so ACLNN positional inputs cannot collide.
    for name in ("ori_topk_length", "cmp_topk_length"):
        if name in attributes:
            attributes[name + "_values"] = attributes.pop(name)
    record["api_name"] = ACLNN_API
    record["attributes"] = repr(attributes)
    record["output_tensor_indexes"] = repr(
        tuple(range(INPUT_COUNT, INPUT_COUNT + len(output_shapes)))
    )
    return record


def active_lengths(row, name, prefix_name, batch, capacity):
    values = vector(row, name)
    cu = vector(row, prefix_name)
    if (
        values is None
        and cu is None
        and (cell(row, "actlen_mode", default="full") != "full")
    ):
        raise ValueError(f"{name}: random or partial active lengths must be explicit")
    values, cu = sequence_lengths(row, name, batch, capacity, prefix_name)
    return (values, cu)


def parameter_case(row):
    """Derive this MLA operator's tensors from deterministic parameter values."""
    b, s1, s2, n1, n2, d = (
        number(row, name, minimum=1) for name in ("B", "S1", "S2", "N1", "N2", "D")
    )
    lq, lk = (str(required(row, "layout_q")), str(required(row, "layout_kv")))
    if lq not in ("BSND", "TND") or lk not in ("BSND", "TND", "PA_BBND"):
        raise ValueError("unsupported query/KV layout")
    k = number(row, "K", default=0, minimum=0)
    mode = cell(row, "template_run_mode", "template_mode", "template")
    if mode is None:
        mode = "CSA" if k else "HCA" if cell(row, "cmp_ratio") is not None else "SWA"
    mode = {
        "ORI_SPARSE_MODE": "ORI_SPARSE",
        "ORI_CMP_SPARSE_MODE": "ORI_CMP_SPARSE",
    }.get(mode, mode)
    if mode not in ("SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"):
        raise ValueError(f"unsupported template mode: {mode!r}")
    has_cmp = mode not in ("SWA", "ORI_SPARSE")
    ori_sparse = mode in ("ORI_SPARSE", "ORI_CMP_SPARSE")
    cmp_sparse = mode in ("CSA", "ORI_CMP_SPARSE")
    ratio = number(row, "cmp_ratio", default=1, minimum=1) if has_cmp else 1
    k1 = number(row, "K1", default=0, minimum=0)
    if ori_sparse and k1 <= 0 or (cmp_sparse and k <= 0):
        raise ValueError("sparse templates require positive K1/K")
    if (
        boolean(cell(row, "random_seq", default=False))
        and vector(row, "cu_seqlens_q") is None
    ):
        raise ValueError("random_seq requires explicit cumulative sequence lengths")
    qlen, cuq = active_lengths(row, "seqused_q", "cu_seqlens_q", b, s1)
    olen, cuo = active_lengths(row, "seqused_ori_kv", "cu_seqlens_ori_kv", b, s2)
    equal = boolean(cell(row, "S1EQS2", "qk_equal_len", default=False))
    if equal:
        if vector(row, "seqused_q") is None and vector(row, "cu_seqlens_q") is None:
            if max(olen) > s1:
                raise ValueError("equal-length mode exceeds the query capacity")
            qlen = list(olen)
            cuq = list(cuo) if lq == "TND" else prefix([s1] * b)
        elif (
            vector(row, "seqused_ori_kv") is None
            and vector(row, "cu_seqlens_ori_kv") is None
        ):
            if max(qlen) > s2:
                raise ValueError("equal-length mode exceeds the ORI capacity")
            olen = list(qlen)
            cuo = list(cuq) if lk == "TND" else prefix([s2] * b)
        if qlen != olen:
            raise ValueError(
                "equal-length mode requires equal query and ORI active lengths"
            )
    clen = cuc = residual = None
    if has_cmp:
        clen, cuc = active_lengths(
            row, "seqused_cmp_kv", "cu_seqlens_cmp_kv", b, s2 // ratio
        )
        if (
            vector(row, "seqused_cmp_kv") is None
            and vector(row, "cu_seqlens_cmp_kv") is None
        ):
            clen = [value // ratio for value in olen]
            cuc = prefix(
                [value // ratio for value in lengths(cuo, "cu_seqlens_ori_kv", b)]
            )
        residual = (
            vector(row, "cmp_residual_kv")
            if number(row, "cmp_mask_mode", default=3 if ratio != 1 else 0) != 0
            and ratio != 1
            else None
        )
        if (
            residual is None
            and number(row, "cmp_mask_mode", default=3 if ratio != 1 else 0) != 0
            and (ratio != 1)
        ):
            residual = [value % ratio for value in olen]
        if residual is not None:
            if len(residual) == 1:
                residual *= b
            if len(residual) != b or max(residual) >= ratio:
                raise ValueError(
                    "cmp_residual_kv must contain B values less than cmp_ratio"
                )
    t1 = number(row, "T1", default=cuq[-1], minimum=1)
    if lq == "TND" and t1 != cuq[-1]:
        raise ValueError(
            "T1 must match cu_seqlens_q; provide explicit lengths for nonuniform TND capacity"
        )
    for name, cumulative in (("T2", cuo), ("T3", cuc)):
        if (
            lk == "TND"
            and cumulative is not None
            and (cell(row, name) is not None)
            and (number(row, name) != cumulative[-1])
        ):
            raise ValueError(
                f"{name} must match the corresponding cumulative KV lengths"
            )
    block1 = number(row, "block_size1", default=128, minimum=1)
    block2 = number(row, "block_size2", default=block1, minimum=1)
    need1 = sum((math.ceil(value / block1) for value in olen))
    need2 = sum((math.ceil(value / block2) for value in clen)) if has_cmp else 0
    blocks1 = number(row, "block_num1", default=need1, minimum=0)
    blocks2 = number(row, "block_num2", default=need2, minimum=0)
    if lk == "PA_BBND" and (blocks1 < need1 or blocks2 < need2):
        raise ValueError("paged KV block capacity is smaller than the active lengths")
    qtype = dtype(cell(row, "q_type", default="bfloat16"))
    otype = dtype(cell(row, "ori_kv_type", default=qtype))
    ctype = dtype(cell(row, "cmp_kv_type", default=otype))
    width = d
    shared_blocks = None
    qshape = (b, s1, n1, d) if lq == "BSND" else (t1, n1, d)

    def kv_shape(cumulative, block_count, block_size):
        if lk == "PA_BBND":
            return (
                shared_blocks if shared_blocks is not None else block_count,
                block_size,
                n2,
                width,
            )
        if lk == "TND":
            return (cumulative[-1], n2, width)
        return (b, max(lengths(cumulative, "KV prefix", b)), n2, width)

    oshape = kv_shape(cuo, blocks1, block1)
    cshape = kv_shape(cuc, blocks2, block2) if has_cmp else None
    osparse = qshape[:-2] + (n2, k1) if ori_sparse else None
    csparse = qshape[:-2] + (n2, k) if cmp_sparse else None
    obt = (
        (b, math.ceil(max(lengths(cuo, "ORI prefix", b)) / block1))
        if lk == "PA_BBND"
        else None
    )
    cbt = (
        (b, math.ceil(max(lengths(cuc, "CMP prefix", b)) / block2))
        if lk == "PA_BBND" and has_cmp
        else None
    )
    omask = number(row, "ori_mask_mode", default=4)
    cmask = (
        number(row, "cmp_mask_mode", default=3 if ratio != 1 else 0) if has_cmp else 0
    )
    ranges = [
        data_range(row, name, (-10, 10))
        for name in ("q_datarange", "ori_kv_datarange", "cmp_kv_datarange")
    ]
    tensors = [
        slot(qshape, qtype, ranges[0]),
        slot(oshape, otype, ranges[1]),
        slot(cshape, ctype, ranges[2]),
    ]
    tensors += [
        slot(osparse, "int32", (0, max(olen) - 1)),
        slot(csparse, "int32", (0, max(clen or [1]) - 1)),
        slot(obt, "int32", (0, oshape[0] - 1)),
        slot(cbt, "int32", (0, cshape[0] - 1) if cshape else None),
    ]
    vectors = (
        ("cu_seqlens_q", cuq if lq == "TND" else None),
        ("cu_seqlens_ori_kv", cuo if lk == "TND" else None),
        ("cu_seqlens_cmp_kv", cuc if lk == "TND" else None),
        ("seqused_q", qlen),
        ("seqused_ori_kv", None if ori_sparse else olen),
        ("seqused_cmp_kv", None if ori_sparse else clen),
        ("cmp_residual_kv", residual),
    )
    attributes = {
        "layout_q": lq,
        "layout_kv": lk,
        "S1": s1,
        "S2": s2,
        "softmax_scale": float(cell(row, "softmax_scale", default=1.0)),
        "cmp_ratio": ratio,
        "ori_mask_mode": omask,
        "cmp_mask_mode": cmask,
        "ori_win_left": number(row, "ori_win_left", default=-1),
        "ori_win_right": number(row, "ori_win_right", default=-1),
        "template_run_mode": mode,
        "has_ori_kv": True,
        "has_cmp_kv": has_cmp,
        "return_softmax_lse": boolean(cell(row, "return_softmax_lse", default=False)),
        "topk_value_mode": number(row, "topk_value_mode", default=1),
    }
    if lk == "PA_BBND":
        attributes.update(
            block_size1=block1,
            block_size2=block2,
            block_num1=blocks1,
            block_num2=blocks2,
        )
    for name, cumulative in (
        ("cu_seqlens_q", cuq),
        ("cu_seqlens_ori_kv", cuo),
        ("cu_seqlens_cmp_kv", cuc),
    ):
        if cumulative is not None:
            attributes[name + "_values"] = cumulative
    for name, values in vectors:
        tensors.append(
            slot(
                (len(values),) if values is not None else None,
                "int32",
                (0, max(values or [0])),
            )
        )
        if values is not None:
            attributes[name + "_values"] = values
    for branch, sparse, mask, seq, count in (
        ("ori", ori_sparse, omask, olen, k1),
        ("cmp", cmp_sparse, cmask, clen, k),
    ):
        topk_mode = cell(
            row, branch + "_kv_topk_mode", default="fullK" if mask == 0 else "no"
        )
        if isinstance(topk_mode, str) and topk_mode.lower() == "fullk":
            topk_mode = "fullK"
        if topk_mode not in ("no", "fullK", "random"):
            raise ValueError(f"unsupported {branch}_kv_topk_mode")
        attributes[branch + "_kv_topk_mode"] = topk_mode
        override = vector(row, branch + "_topk_length")
        present = sparse and (override is not None or (mask == 0 and topk_mode != "no"))
        shape = qshape[:-2] + (n2,) if present else None
        tensors.append(slot(shape, "int32", (0, max(seq or [0]))))
        if override is not None:
            if not sparse:
                raise ValueError(f"{branch}_topk_length requires sparse indices")
            size = math.prod(shape)
            if len(override) == 1:
                override *= size
            if len(override) != size or max(override, default=0) > count:
                raise ValueError(
                    f"{branch}_topk_length must match query/head shape and lie in [0, K]"
                )
            attributes[branch + "_topk_length_values"] = override
        if sparse:
            attributes["metadata_" + branch + "_topk"] = count
    sinks = boolean(cell(row, "use_sinks", default=True))
    tensors.append(slot((n1,) if sinks else None, "float32", (-1, 1)))
    tensors.append(slot((-1,), "int32", (0, 0)))
    for name in (
        "ori_sparse_indices_mode",
        "cmp_sparse_indices_mode",
        "actlen_mode",
        "random_seq",
        "S1EQS2",
        "key_dtype",
        "value_dtype",
    ):
        if cell(row, name) is not None:
            attributes[name] = cell(row, name)
    storages = strides = None
    if lk == "PA_BBND":
        storages = [entry[0] for entry in tensors]
        strides = [None] * len(tensors)
        for index, block_size in ((1, block1), (2, block2)):
            if tensors[index][0] is not None:
                block_count = tensors[index][0][0]
                stride = (block_size + 10) * n2 * d
                storages[index], strides[index] = (
                    (block_count, stride),
                    (stride, n2 * d, d, 1),
                )
        storages[-1] = None
    record = tensor_fields(tensors, storages, strides)
    record["absolute_precision"] = repr((0.02, 0.0))
    record["precision_tolerances"] = repr(((0.0078125, 0.005), (0.0, 0.0)))
    return (record, attributes)


def output_contract(shapes, types, attributes):
    """Describe attention and optional LSE outputs in the extension's axis order."""
    q = shapes[0]
    kv = shapes[1] if shapes[1] is not None else shapes[2]
    lq = attributes["layout_q"]
    if (
        q is None
        or kv is None
        or lq not in ("BSND", "TND")
        or (len(q) != (4 if lq == "BSND" else 3))
    ):
        raise ValueError("query/KV rank does not match its layout")
    n2 = kv[-2]
    if n2 <= 0 or q[-2] % n2:
        raise ValueError("query heads must be divisible by positive KV heads")
    lse = (q[0], n2, q[1], q[-2] // n2) if lq == "BSND" else (n2, q[0], q[-2] // n2)
    if not boolean(attributes["return_softmax_lse"]):
        lse = (0,)
    output_dtype = types[0]
    return ((tuple(q), lse), (output_dtype, "float32"))


def aclnn_inputs(shapes, types, attributes, values):
    return (shapes, types, values)
