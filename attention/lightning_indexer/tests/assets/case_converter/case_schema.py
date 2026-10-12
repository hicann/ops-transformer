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

OPERATOR = "lightning_indexer"
INPUT_COUNT = 6
E2E_API = "torch_npu.npu_lightning_indexer"
ACLNN_API = "aclnnLightningIndexer"
API_ALIASES = ("torch_npu.npu_lightning_indexer",)
METADATA_INDEX = None
ATTRIBUTES = (
    ("layoutQueryOptional", "layout_query", "BSND"),
    ("layoutKeyOptional", "layout_key", "BSND"),
    ("sparseCount", "sparse_count", 2048),
    ("sparseMode", "sparse_mode", 3),
    ("preTokens", "pre_tokens", 9223372036854775807),
    ("nextTokens", "next_tokens", 9223372036854775807),
    ("returnValues", "return_value", False),
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


def lengths(prefix_values, name, batch):
    if len(prefix_values) != batch + 1 or prefix_values[0] != 0:
        raise ValueError(f"{name} must start with zero and contain B+1 values")
    result = [right - left for left, right in zip(prefix_values, prefix_values[1:])]
    if min(result, default=0) < 0:
        raise ValueError(f"{name} must be nondecreasing")
    return result


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
    record["api_name"] = ACLNN_API
    record["attributes"] = repr(attributes)
    record["output_tensor_indexes"] = repr(
        tuple(range(INPUT_COUNT, INPUT_COUNT + len(output_shapes)))
    )
    return record


def parameter_case(row):
    """Describe indexer inputs from scalar parameters and explicit lengths."""
    b = number(row, "batch_size", minimum=1)
    qs, ks, qn, kn, hd = (
        number(row, name, minimum=1)
        for name in ("q_seq", "k_seq", "q_head_num", "k_head_num", "head_dim")
    )
    lq = str(required(row, "layout_q", "layout_query"))
    lk = str(required(row, "layout_k", "layout_key"))
    paged_layout = "PA_BSND"
    if lq not in ("BSND", "TND") or lk not in ("BSND", "TND", paged_layout):
        raise ValueError("unsupported query/key layout")
    qtype = dtype(required(row, "qk_dtype"))
    weight_type = dtype(cell(row, "weight_dtype", default="float32"))
    block = number(row, "block_size", default=128, minimum=1)
    block_num = number(row, "block_num", default=b * math.ceil(ks / block), minimum=1)
    topk = number(row, "topk", "sparse_count", default=2048, minimum=1)
    mask = number(row, "mask_mode", "sparse_mode", default=3)
    qr, kr, wr = (
        data_range(row, name, (-1, 1))
        for name in ("query_datarange", "key_datarange", "weights_datarange")
    )
    aq = vector(row, "act_seq_q", "actual_seq_q")
    ak = vector(row, "act_seq_k", "actual_seq_k")
    if aq is None and lq == "BSND":
        aq = [qs] * b
    if ak is None and lk == "BSND":
        ak = [ks] * b
    if aq is None or ak is None or len(aq) != b or (len(ak) != b):
        raise ValueError("act_seq_q/act_seq_k must explicitly contain B values")
    qt = number(row, "q_t_size", default=aq[-1], minimum=0)
    kt = number(row, "k_t_size", default=ak[-1], minimum=0)
    for name, sequence, layout, capacity, tokens in (
        ("act_seq_q", aq, lq, qs, qt),
        ("act_seq_k", ak, lk, ks, kt),
    ):
        if layout == "TND":
            lengths([0] + sequence, name, b)
            if sequence[-1] != tokens:
                raise ValueError(
                    f"{name}: final cumulative length must match the TND token count"
                )
        elif max(sequence) > capacity:
            raise ValueError(f"{name}: active lengths exceed sequence capacity")
    qshape = (b, qs, qn, hd) if lq == "BSND" else (qt, qn, hd)
    wshape = qshape[:-1]
    bt = None
    if lk == "BSND":
        kshape = (b, ks, kn, hd)
    elif lk == "TND":
        kshape = (kt, kn, hd)
    else:
        kshape = (block_num, block, kn, hd)
        required_blocks = sum((math.ceil(value / block) for value in ak))
        if block_num < required_blocks:
            raise ValueError("block_num is too small for active key lengths")
        bt = (b, math.ceil(max(ak) / block))
    tensors = [
        slot(qshape, qtype, qr),
        slot(kshape, qtype, kr),
        slot(wshape, weight_type, wr),
    ]
    attributes = {
        "layout_query": lq,
        "layout_key": lk,
        "sparse_count": topk,
        "sparse_mode": mask,
    }
    actual_dtype = dtype(cell(row, "actual_seq_dtype", default="int32"))
    tensors += [
        slot(
            (len(aq),)
            if vector(row, "act_seq_q", "actual_seq_q") is not None
            else None,
            actual_dtype,
            (0, max(aq)),
        ),
        slot(
            (len(ak),)
            if vector(row, "act_seq_k", "actual_seq_k") is not None
            else None,
            actual_dtype,
            (0, max(ak)),
        ),
    ]
    attributes.update(
        actual_seq_lengths_query_values=aq, actual_seq_lengths_key_values=ak
    )
    attributes["return_value"] = boolean(cell(row, "return_value", default=False))
    parameters = dict(row)
    parameters.update(
        batch_size=b,
        q_seq=qs,
        k_seq=ks,
        q_t_size=qt,
        k_t_size=kt,
        q_head_num=qn,
        k_head_num=kn,
        head_dim=hd,
        qk_dtype=qtype,
        weight_dtype=weight_type,
        actual_seq_dtype=actual_dtype,
        act_seq_q=aq,
        act_seq_k=ak,
        layout_query=lq,
        layout_key=lk,
        sparse_count=topk,
        sparse_mode=mask,
        block_size=block,
        block_num=block_num,
        query_datarange=qr,
        key_datarange=kr,
        weights_datarange=wr,
    )
    parameters["return_value"] = attributes["return_value"]
    excluded = {
        "testcase_name",
        "Testcase_Name",
        "case_name",
        "tc_prefix",
        "is_enabled",
        "result",
    } | set(OPTIONAL_FIELDS)
    attributes.update(
        {
            "pytest_" + name: literal(value)
            for name, value in parameters.items()
            if name not in excluded
        }
    )
    tensors.append(slot(bt, "int32", (0, block_num - 1)))
    if bt is not None:
        attributes.update(block_size=block, block_num=block_num)
    storages, strides = (None, None)
    result = tensor_fields(tensors, storages, strides)
    result["absolute_precision"] = repr((0.01, 0.01))
    return (result, attributes)


def output_contract(shapes, types, attributes):
    """Describe index tensors and the optional values tensor."""
    q, key = shapes[:2]
    layout = attributes["layout_query"]
    if layout not in ("BSND", "TND") or len(q) != (4 if layout == "BSND" else 3):
        raise ValueError("query rank does not match its layout")
    count = integer(attributes["sparse_count"], "topk", 1)
    shape = tuple(q[:-2]) + (key[-2], count)
    values = shape if boolean(attributes["return_value"]) else (0,)
    value_type = types[0]
    return ((shape, values), ("int32", value_type))


def aclnn_inputs(shapes, types, attributes, values):
    types = list(types)
    return (shapes, tuple(types), values)
