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

OPERATOR = "sparse_flash_attention"
INPUT_COUNT = 10
E2E_API = "torch.ops.cann_ops_transformer.sparse_flash_attention"
ACLNN_API = "aclnnSparseFlashAttentionV2"
API_ALIASES = (
    "torch.ops.cann_ops_transformer.sparse_flash_attention",
    "torch_npu.npu_sparse_flash_attention",
)
METADATA_INDEX = None
ATTRIBUTES = (
    ("scaleValue", "scale_value", None),
    ("sparseBlockSizeOptional", "sparse_block_size", 1),
    ("layoutQueryOptional", "layout_query", "BSND"),
    ("layoutKvOptional", "layout_kv", "BSND"),
    ("sparseMode", "sparse_mode", 3),
    ("preTokens", "pre_tokens", 9223372036854775807),
    ("nextTokens", "next_tokens", 9223372036854775807),
    ("attentionMode", "attention_mode", 0),
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
    record = normalize_source(record, attributes)
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


def normalize_source(record, attributes):
    shapes = literal(record.get("tensor_view_shapes"))
    if isinstance(shapes, (tuple, list)) and len(shapes) == INPUT_COUNT - 1:
        if boolean(attributes.get("pytest_use_sinks", False)):
            raise ValueError(
                "sinks are enabled but the sinks input descriptor is absent"
            )
        for field in (
            "tensor_view_shapes",
            "tensor_dtypes",
            "input_data_ranges",
            "tensor_storage_shapes",
            "tensor_view_strides",
            "tensor_formats",
            "tensor_view_offsets",
        ):
            values = literal(record.get(field))
            if values is not None:
                if (
                    not isinstance(values, (tuple, list))
                    or len(values) != INPUT_COUNT - 1
                ):
                    raise ValueError(
                        f"{field}: legacy table must contain nine input slots"
                    )
                record[field] = repr(tuple(values) + (None,))
    return record


def actual_sequence(row, name, batch, upper, layout):
    values = vector(row, name)
    if values is None and layout == "BSND":
        values = [upper] * batch
    if values is None or len(values) != batch:
        raise ValueError(f"{name} must explicitly contain B values")
    sizes = (
        [right - left for left, right in zip([0] + values[:-1], values)]
        if layout == "TND"
        else values
    )
    if min(sizes) < 0 or max(sizes) > upper:
        raise ValueError(f"{name}: invalid lengths for {layout}")
    return values


def parameter_case(row):
    """Describe the final attention API inputs, including packed caches and sinks."""
    b, s1, s2, n1, n2, d = (
        number(row, name, minimum=1) for name in ("B", "S1", "S2", "N1", "N2", "D")
    )
    k = number(row, "K", minimum=1)
    lq, lk = (str(required(row, "layout_query")), str(required(row, "layout_kv")))
    if lq not in ("BSND", "TND") or lk not in ("BSND", "TND", "PA_BSND"):
        raise ValueError("unsupported query/KV layout")
    if lk in ("BSND", "TND") and lq != lk:
        raise ValueError(
            "parameter workbooks require matching nonpaged query/KV layouts; use explicit descriptors for mixed layouts"
        )
    aq = actual_sequence(row, "actual_seq_q", b, s1, lq)
    ak = actual_sequence(row, "actual_seq_kv", b, s2, lk)
    q_prefix = (b, s1, n1) if lq == "BSND" else (aq[-1], n1)
    k_prefix = (b, s2, n2) if lk == "BSND" else (ak[-1], n2)
    qtype = dtype(required(row, "q_type"))
    ktype = dtype(cell(row, "kv_type", default=qtype))
    rope = number(row, "rope_head_dim", default=64, minimum=0)
    block = number(row, "block_size", default=256, minimum=1)
    sparse_block = number(row, "sparse_block_size", default=1, minimum=1)
    if k % sparse_block:
        raise ValueError("K must be divisible by sparse_block_size")
    bt = None
    blocks = 0
    if lk == "PA_BSND":
        needed = sum((math.ceil(value / block) for value in ak))
        blocks = number(row, "block_num", default=needed, minimum=1)
        if blocks < needed:
            raise ValueError("block_num is too small for the active KV lengths")
        k_prefix = (blocks, block, n2)
        bt = (b, math.ceil(s2 / block))
    qr = data_range(row, "range_query", (-10, 100))
    kr = data_range(row, "range_key", (5, 100))
    attributes = {
        "scale_value": float(required(row, "scale_value")),
        "layout_query": lq,
        "layout_kv": lk,
        "sparse_block_size": sparse_block,
        "sparse_mode": number(row, "sparse_mode", default=3),
        "attention_mode": number(row, "attention_mode", default=0),
        "rope_head_dim": rope,
        "actual_seq_lengths_query_values": aq,
        "actual_seq_lengths_kv_values": ak,
    }
    key_shape = k_prefix + (d,)
    tensors = [
        slot(q_prefix + (d,), qtype, qr),
        slot(key_shape, ktype, kr),
        slot(key_shape, ktype, kr),
        slot(q_prefix[:-1] + (n2, k // sparse_block), "int32", (-10, 10)),
        slot(bt, "int32", (0, 1)),
        slot(
            (len(aq),) if vector(row, "actual_seq_q") is not None else None,
            "int32",
            (0, max(aq)),
        ),
        slot(
            (len(ak),) if vector(row, "actual_seq_kv") is not None else None,
            "int32",
            (0, max(ak)),
        ),
        slot(q_prefix + (rope,), qtype, data_range(row, "range_query_rope", (-10, 10))),
        slot(k_prefix + (rope,), ktype, data_range(row, "range_key_rope", (-10, 10))),
    ]
    sinks = boolean(cell(row, "use_sinks", default=False))
    tensors.append(
        slot(
            (n1,) if sinks else None,
            "float32",
            data_range(row, "range_sinks", (-0.2, 0.2)),
        )
    )
    attributes["return_softmax_lse"] = boolean(
        cell(row, "return_softmax_lse", default=False)
    )
    for key in ("pre_tokens", "next_tokens"):
        if cell(row, key) is not None:
            attributes[key] = number(row, key)
    if bt is not None:
        attributes.update(block_size=block, block_num=blocks)
    for name, layout, values in (("T1", lq, aq), ("T2", lk, ak)):
        if (
            layout == "TND"
            and cell(row, name) is not None
            and (number(row, name) != values[-1])
        ):
            raise ValueError(
                f"{name} must match the corresponding actual-sequence endpoint"
            )
    parameters = dict(row)
    parameters.update(
        Testcase_Prefix=str(
            required(
                row,
                "Testcase_Prefix",
                "testcase_name",
                "Testcase_Name",
                "case_name",
                "tc_prefix",
            )
        ),
        B=b,
        S1=s1,
        S2=s2,
        N1=n1,
        N2=n2,
        D=d,
        K=k,
        layout_query=lq,
        layout_kv=lk,
        q_type=qtype,
        kv_type="hifloat8"
        if str(cell(row, "kv_type", default="")).lower().removeprefix("torch.")
        in ("hif8", "hifp8", "hifloat8")
        else ktype,
        scale_value=attributes["scale_value"],
        sparse_block_size=sparse_block,
        rope_head_dim=rope,
        sparse_mode=attributes["sparse_mode"],
        attention_mode=attributes["attention_mode"],
        block_size=block,
        block_num=blocks,
        actual_seq_q=aq,
        actual_seq_kv=ak,
        T1=aq[-1] if lq == "TND" else b * s1,
        T2=ak[-1] if lk == "TND" else b * s2,
        range_query=qr,
        range_key=kr,
    )
    parameters["return_softmax_lse"] = attributes["return_softmax_lse"]
    parameters["use_sinks"] = sinks
    excluded = {
        "testcase_name",
        "Testcase_Name",
        "case_name",
        "tc_prefix",
        "is_enabled",
        "precision_tolerances",
        "absolute_precision",
        "result",
    } | set(OPTIONAL_FIELDS)
    attributes.update(
        {
            "pytest_" + name: literal(value)
            for name, value in parameters.items()
            if name not in excluded
        }
    )
    record = tensor_fields(tensors)
    record["absolute_precision"] = repr((0.0001,) * 3)
    record["precision_tolerances"] = repr(((0.0078125, 0.005),) * 3)
    return (record, attributes)


def output_contract(shapes, types, attributes):
    """Describe attention and optional head-first TND softmax outputs."""
    q = shapes[0]
    layout = attributes["layout_query"]
    if layout not in ("BSND", "TND") or len(q) != (4 if layout == "BSND" else 3):
        raise ValueError("query rank does not match layout_query")
    heads = shapes[1][-2]
    if heads <= 0 or q[-2] % heads:
        raise ValueError("query heads must be divisible by positive KV heads")
    output = tuple(q)
    if output[-1] <= 0:
        raise ValueError("query dimension must exceed rope_head_dim")
    lse = (
        (q[0], heads, q[1], q[-2] // heads)
        if layout == "BSND"
        else (heads, q[0], q[-2] // heads)
    )
    if not boolean(attributes["return_softmax_lse"]):
        lse = (0,)
    return ((output, lse, lse), (types[0], "float32", "float32"))


def aclnn_inputs(shapes, types, attributes, values):
    return (shapes, types, values)
