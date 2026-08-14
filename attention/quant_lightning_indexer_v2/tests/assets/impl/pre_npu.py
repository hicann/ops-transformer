#!/usr/bin/python
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

"""QuantLightningIndexer V2 metadata stage between input restoration and the main API."""

import csv
import fcntl
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ttk.test_spec import PreNpuResult, TtkContext


METADATA_INDEX = 12
OPERATOR = "quant_lightning_indexer_v2"
METADATA_INPUTS_FILE = "metadata_inputs.pt"
METADATA_INPUTS_VERSION = 1
METADATA_INPUTS_STATE_KEY = f"{OPERATOR}.metadata_inputs"
QUANT_MODE_MXFP8 = 3
QUANT_MODE_MXFP4 = 5
METADATA_RESULT_FIELDS = (
    "recorded_at",
    "testcase_name",
    "api_name",
    "operator",
    "case_type",
    "status",
    "metadata_only",
    "profiling_enabled",
    "repeat_count",
    "elapsed_us",
    "kernel_count",
    "kernel_names",
    "profile_result_path",
    "error_type",
    "error",
)

NON_INPUT_CSV_FIELDS = {
    "is_enabled",
    "precision_tolerances",
    "absolute_precision",
    "metadata_only",
}


def csv_contract_digest(context):
    """Bind persisted metadata inputs to the CSV contract that generated them."""
    fields = getattr(context, "csv_fields", None)
    if not fields:
        return None
    contract = {
        name: value
        for name, value in fields.items()
        if name not in NON_INPUT_CSV_FIELDS
    }
    attributes = getattr(context, "attributes", None)
    if attributes is not None:
        attributes = dict(attributes)
        attributes.pop("metadata_only", None)
        contract["attributes"] = attributes
    serialized = json.dumps(
        contract,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def clone_metadata_value(value):
    """Copy only the small pytest metadata arguments into a CPU-only payload."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: clone_metadata_value(item) for name, item in value.items()}
    if isinstance(value, list):
        return [clone_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_metadata_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(
        "unsupported QuantLightningIndexer V2 metadata input type: "
        f"{type(value).__name__}"
    )


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def metadata_inputs_path(context):
    case_dir = None if context is None else context.manual_case_dir
    return None if case_dir is None else Path(case_dir) / METADATA_INPUTS_FILE


def persist_metadata_inputs(testcase_name, metadata_input, context):
    """Persist this operator's compact metadata API arguments for replay."""
    if context is None or testcase_name is None:
        return
    if not isinstance(metadata_input, dict):
        raise ValueError("QuantLightningIndexer V2 pytest data lacks metadata_input")
    name = str(testcase_name)
    payload = {
        "version": METADATA_INPUTS_VERSION,
        "operator": OPERATOR,
        "testcase_name": name,
        "csv_contract": csv_contract_digest(context),
        "metadata_input": clone_metadata_value(metadata_input),
    }
    context.state[METADATA_INPUTS_STATE_KEY] = payload
    path = metadata_inputs_path(context)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(
            f"QuantLightningIndexer V2 metadata state is not a regular file: {path}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_metadata_inputs(context):
    payload = context.state.get(METADATA_INPUTS_STATE_KEY)
    path = metadata_inputs_path(context)
    if payload is None and path is not None:
        if path.is_symlink():
            raise ValueError(
                f"QuantLightningIndexer V2 metadata state is not a regular file: {path}"
            )
        if path.exists():
            if not path.is_file():
                raise ValueError(
                    f"QuantLightningIndexer V2 metadata state is not a regular file: {path}"
                )
            payload = load_pt(path)
    if payload is None:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != METADATA_INPUTS_VERSION
        or payload.get("operator") != OPERATOR
        or payload.get("testcase_name") != context.testcase_name
    ):
        raise ValueError(
            "QuantLightningIndexer V2 metadata state has an incompatible schema"
        )
    if payload.get("csv_contract") != csv_contract_digest(context):
        raise ValueError(
            "QuantLightningIndexer V2 metadata state does not match the current CSV contract"
        )
    metadata_input = payload.get("metadata_input")
    if not isinstance(metadata_input, dict):
        raise ValueError("QuantLightningIndexer V2 metadata state lacks metadata_input")
    context.state[METADATA_INPUTS_STATE_KEY] = payload
    return metadata_input


def metadata_arguments(context):
    """Return the exact metadata arguments prepared by pytest input generation."""
    metadata_input = load_metadata_inputs(context)
    if metadata_input is None:
        raise RuntimeError(
            "QuantLightningIndexer V2 requires input-stage metadata inputs before pre-NPU"
        )
    return metadata_input


def parse_metadata_only_option(kwargs):
    """Keep absent, explicit false, and explicit true as separate states."""
    if "metadata_only" not in kwargs:
        return None
    value = kwargs["metadata_only"]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError("metadata_only must be a boolean")


def is_materialized(metadata):
    if torch.is_tensor(metadata):
        return bool(torch.count_nonzero(metadata).item())
    return bool(np.count_nonzero(metadata))


def restore_main_api_mx_dtypes(tensors, quant_mode):
    """Restore logical MX dtypes on the tensors consumed by the installed API."""
    quant_mode = int(quant_mode)
    if quant_mode == QUANT_MODE_MXFP8:
        qk_dtype = getattr(torch, "float8_e4m3fn", None)
    elif quant_mode == QUANT_MODE_MXFP4:
        qk_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    else:
        return
    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if qk_dtype is None or scale_dtype is None:
        raise RuntimeError("current PyTorch does not provide the requested MX dtype")
    if len(tensors) < 5:
        raise ValueError("QLI_V2 MX input requires query, key, and two scale tensors")

    for index, dtype in (
        (0, qk_dtype),
        (1, qk_dtype),
        (3, scale_dtype),
        (4, scale_dtype),
    ):
        tensor = tensors[index]
        if not torch.is_tensor(tensor):
            raise TypeError(f"QLI_V2 MX tensor slot {index} must be a torch.Tensor")
        if tensor.dtype == dtype:
            continue
        if tensor.dtype != torch.uint8:
            raise TypeError(
                f"QLI_V2 MX tensor slot {index} must use uint8 packed storage, "
                f"got {tensor.dtype}"
            )
        tensor.data = tensor.data.view(dtype)


def to_npu(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.npu()
    return torch.as_tensor(value).npu()


def copy_metadata(target, source):
    source = source.detach().cpu()
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"QLI_V2 metadata shape mismatch: placeholder={tuple(target.shape)}, "
            f"generated={tuple(source.shape)}"
        )
    if torch.is_tensor(target):
        target.copy_(source.to(dtype=target.dtype, device=target.device))
    else:
        np.copyto(target, source.numpy().astype(target.dtype, copy=False))


def persist_metadata(context, metadata):
    """Publish generated metadata into replay input storage when it exists."""
    case_dir = context.manual_case_dir
    file_format = context.manual_data_format
    if case_dir is None or file_format is None:
        return
    matches = tuple(case_dir.glob(f"input_{METADATA_INDEX}_*.{file_format}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"QLI_V2 expected one stored metadata input, found {len(matches)} in {case_dir}"
        )
    path = matches[0]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    array = (
        metadata.detach().cpu().contiguous().numpy()
        if torch.is_tensor(metadata)
        else np.asarray(metadata)
    )
    if file_format == "bin":
        array.tofile(temporary)
    elif file_format == "npy":
        with temporary.open("wb") as file:
            np.save(file, array, allow_pickle=False)
    elif file_format == "pt":
        torch.save(torch.from_numpy(np.array(array, copy=True)), temporary)
    else:
        raise ValueError(f"QLI_V2 cannot persist manual-data format {file_format!r}")
    temporary.replace(path)


def run_torch(context, target, arguments):
    generated = None

    def operation():
        nonlocal generated
        generated = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            arguments["num_heads_q"],
            arguments["num_heads_k"],
            arguments["head_dim"],
            arguments["topk"],
            arguments["quant_mode"],
            cu_seqlens_q=to_npu(arguments["cu_seqlens_q"]),
            cu_seqlens_k=to_npu(arguments["cu_seqlens_k"]),
            seqused_q=to_npu(arguments["seqused_q"]),
            seqused_k=to_npu(arguments["seqused_k"]),
            cmp_residual_k=to_npu(arguments["cmp_residual_k"]),
            batch_size=arguments["batch_size"],
            max_seqlen_q=arguments["max_seqlen_q"],
            max_seqlen_k=arguments["max_seqlen_k"],
            layout_q=arguments["layout_q"],
            layout_k=arguments["layout_k"],
            mask_mode=arguments["mask_mode"],
            cmp_ratio=arguments["cmp_ratio"],
        )

    profile = context.run_profiled("metadata", operation)
    copy_metadata(target, generated)
    return profile


def run_aclnn(context, metadata, arguments):
    def operation():
        context.run_aclnn(
            "aclnnQuantLightningIndexerV2Metadata",
            tensors={
                "cuSeqlensQOptional": arguments["cu_seqlens_q"],
                "cuSeqlensKOptional": arguments["cu_seqlens_k"],
                "sequsedQOptional": arguments["seqused_q"],
                "sequsedKOptional": arguments["seqused_k"],
                "cmpResidualKOptional": arguments["cmp_residual_k"],
                "metadata": metadata,
            },
            attributes={
                "numHeadsQ": arguments["num_heads_q"],
                "numHeadsK": arguments["num_heads_k"],
                "headDim": arguments["head_dim"],
                "topk": arguments["topk"],
                "quantMode": arguments["quant_mode"],
                "batchSize": arguments["batch_size"],
                "maxSeqlenQ": arguments["max_seqlen_q"],
                "maxSeqlenK": arguments["max_seqlen_k"],
                "layoutQOptional": arguments["layout_q"],
                "layoutKOptional": arguments["layout_k"],
                "maskMode": arguments["mask_mode"],
                "cmpRatio": arguments["cmp_ratio"],
            },
            output_names=("metadata",),
        )

    return context.run_profiled("metadata", operation)


def metadata_result_path(context):
    output_name = context.options.get("output_file_name")
    if output_name:
        output_path = Path(output_name).expanduser().resolve()
    else:
        input_files = context.options.get("input_files") or ()
        if input_files:
            output_path = Path(input_files[0]).expanduser().resolve()
        else:
            output_path = Path(context.options["root_path"]) / f"{OPERATOR}_result.csv"

    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_name(f"{output_path.name}.csv")
    stem = output_path.stem
    if stem.endswith("_result"):
        stem = stem[: -len("_result")]
    path = output_path.with_name(f"{stem}_metadata_result.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_metadata_result(context, status, metadata_only, profile=None, error=None):
    """Append one metadata execution result without changing TTK's main result."""
    path = metadata_result_path(context)
    kernels = tuple(profile.kernels) if profile is not None else ()
    profiling_enabled = bool(context.options.get("task_prof", False))
    repeat_count = (
        max(int(context.options.get("run_time", 1) or 1), 1) if profiling_enabled else 1
    )
    row = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "testcase_name": context.testcase_name,
        "api_name": context.api_name,
        "operator": OPERATOR,
        "case_type": context.case_type,
        "status": status,
        "metadata_only": metadata_only,
        "profiling_enabled": (
            profile.enabled if profile is not None else profiling_enabled
        ),
        "repeat_count": profile.repeat_count if profile is not None else repeat_count,
        "elapsed_us": profile.elapsed_us if profile is not None else "",
        "kernel_count": len(kernels) if profile is not None else "",
        "kernel_names": ";".join(kernel.name for kernel in kernels),
        "profile_result_path": (
            str(profile.result_path)
            if profile is not None and profile.result_path is not None
            else ""
        ),
        "error_type": type(error).__name__ if error is not None else "",
        "error": str(error) if error is not None else "",
    }

    with path.open("a+", newline="", encoding="utf-8") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            file.seek(0)
            header = next(csv.reader(file), None)
            if header is not None and tuple(header) != METADATA_RESULT_FIELDS:
                raise RuntimeError(
                    f"QLI_V2 metadata result header does not match: {path}"
                )
            file.seek(0, os.SEEK_END)
            writer = csv.DictWriter(file, fieldnames=METADATA_RESULT_FIELDS)
            if header is None:
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    logging.info("[%s] metadata result: %s", context.testcase_name, path)
    return path


def run(*args, context: TtkContext = None, **kwargs):
    """Generate or reuse metadata according to the three-state stop attribute."""
    del args
    if context is None:
        raise RuntimeError(
            "QLI_V2 metadata stage requires an explicitly injected context"
        )
    stop_after_stage = parse_metadata_only_option(kwargs)
    profile = None
    try:
        tensors = tuple(context.input_tensors or ())
        if len(tensors) <= METADATA_INDEX or tensors[METADATA_INDEX] is None:
            raise ValueError("QLI_V2 CSV must reserve tensor slot 12 for metadata")
        metadata = tensors[METADATA_INDEX]
        if stop_after_stage is None and is_materialized(metadata):
            if context.case_type == "e2e":
                restore_main_api_mx_dtypes(
                    tensors, context.attributes.get("quant_mode", 1)
                )
            return None

        arguments = metadata_arguments(context)
        if context.case_type == "aclnn":
            profile = run_aclnn(context, metadata, arguments)
        elif context.case_type == "e2e":
            profile = run_torch(context, metadata, arguments)
        else:
            raise ValueError(
                f"QLI_V2 pre-NPU does not support case type {context.case_type!r}"
            )
        # Explicit false refreshes only this invocation; prepare/metadata-only
        # flows are the paths that publish metadata back to replay storage.
        if stop_after_stage is not False:
            persist_metadata(context, metadata)
        if stop_after_stage is not True and context.case_type == "e2e":
            restore_main_api_mx_dtypes(tensors, context.attributes.get("quant_mode", 1))
    except Exception as error:
        if stop_after_stage is not None:
            try:
                write_metadata_result(
                    context, "ERROR", stop_after_stage, profile, error
                )
            except Exception:
                logging.exception("Failed to write QLI_V2 metadata error result")
        raise

    result_path = None
    if stop_after_stage is not None:
        result_path = write_metadata_result(context, "PASS", stop_after_stage, profile)
    if stop_after_stage is True:
        return PreNpuResult(stop=True, reason=f"METADATA_SUCCESS: {result_path}")
    return None
