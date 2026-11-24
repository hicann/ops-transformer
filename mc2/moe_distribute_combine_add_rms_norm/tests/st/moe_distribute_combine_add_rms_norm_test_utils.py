# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Operator-local MoE references and paired ACLNN calls."""

import ctypes
import os

import torch
import torch.distributed as dist

from atk.tasks.backends.lib_interface.acl_wrapper import get_opp_lib_path, nnopbase
from atk.tasks.run_modes import register as mode_register
from atk.tasks.run_modes.base_run_mode import BaseRunMode
from itertools import product

OPTION_NAMES = (
    "x1_int4",
    "x2_int4",
    "x2_nz",
    "transpose_x2",
    "output_dtype",
    "use_bias",
    "use_x3",
    "use_x1_scale",
    "use_antiquant_offset",
    "use_comm_quant",
    "use_smooth",
    "comm_output",
    "matmul_output",
    "shared_mm",
    "packed_scale",
    "input_distribution",
    "deterministic",
)
OPTION_VALUES = (
    (False, True),
    (False, True),
    (False, True),
    (False, True),
    (None, "fp16", "bf16"),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (False, True),
    (None, "rank_shards", "moe_uniform", "moe_balanced", "eplb_fixed", "eplb_random"),
    (None, True),
)
DEFAULT_OPTIONS = tuple(values[0] for values in OPTION_VALUES)
OPTIONS = list(product(*OPTION_VALUES))


@mode_register.register("mc2_moe_fullmesh")
class Fullmesh(BaseRunMode):
    def push(self):
        super().push()
        self.set_env("HCCL_INTRA_PCIE_ENABLE", "0")
        self.set_env("HCCL_INTRA_ROCE_ENABLE", "1")


def case_options(kwargs):
    data = {"quantMode": 0, "commAlg": None, "commQuantMode": 0, **kwargs}
    options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
    return data, options


def token_order(tokens, balanced):
    key = (
        tokens[:, 0].float() * 65536 + tokens[:, 1].float()
        if balanced
        else tokens[:, 0]
    )
    return torch.argsort(key)


def reference(input_data, task_result, cpu, make_inputs, routes, kind):
    data, options = case_options(input_data.kwargs)
    info = task_result.dist_task_info
    rank, world = info.rank, info.world_size
    case_id = task_result.case_config.id
    balanced = options["input_distribution"] == "moe_balanced"
    x, expert_ids, weights = make_inputs(data, options, rank, world, case_id)
    if kind == "dispatch":
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x)
        selected = []
        for source, part in enumerate(gathered):
            ids = routes(
                x.shape[0], world, expert_ids.shape[1], case_id, source, balanced
            )
            selected.append(part[(ids == rank).any(dim=1).to(part.device)])
        tokens = torch.cat(selected)
        tokens = tokens[token_order(tokens, balanced)]
        counts = torch.tensor([tokens.shape[0]], dtype=torch.int64, device=x.device)
        if data["quantMode"] == 2:
            # Current smooth inputs are ones. Codes and scales remain independent of the target.
            scales = tokens.float().abs().amax(dim=1) / 127
            quant = (
                torch.round(tokens.float() / scales[:, None])
                .clamp(-128, 127)
                .to(torch.int8)
            )
            return [quant.double() if cpu else quant, counts, scales]
        result = [tokens.double() if cpu else tokens, counts]
        if x.dtype == torch.int32:
            result.append(torch.ones(tokens.shape[0], device=x.device))
        return result

    dtype = torch.float64 if cpu else torch.float32
    combined = x.to(dtype) * weights.to(dtype).sum(dim=1, keepdim=True)
    if kind == "combine":
        return [combined if cpu else combined.to(x.dtype)]
    summed = combined[:, None, :] + data["residual"].to(dtype)
    rstd = torch.rsqrt((summed * summed).mean(dim=-1, keepdim=True) + 1e-6)
    normalized = summed * rstd
    if not cpu:
        normalized = normalized.to(x.dtype).to(dtype)
    y = normalized * data["gamma"].to(dtype)
    return (
        [y, rstd, summed] if cpu else [y.to(x.dtype), rstd.float(), summed.to(x.dtype)]
    )


def tensor(executor, value):
    package = executor.torch_tensor_to_acl(value.contiguous())
    executor.packages.append(package)
    return package


def zeros(executor, shape, dtype):
    return tensor(executor, torch.zeros(shape, dtype=dtype))


def init_arguments(executor, input_data, make_inputs, routes, kind, variant, target):
    data, options = case_options(input_data.kwargs)
    info = executor.task_result.dist_task_info
    world, rank = info.world_size, info.rank
    case_id = executor.task_result.case_config.id
    balanced = options["input_distribution"] == "moe_balanced"
    platform = os.environ["MC2_PLATFORM"]
    x, expert_ids, weights = make_inputs(data, options, rank, world, case_id)
    bs, hidden = x.shape
    topk = expert_ids.shape[1]
    executor.packages = []
    executor.kind, executor.variant = kind, variant
    executor.balanced = balanced
    executor.valid_tokens = sum(
        int((routes(bs, world, topk, case_id, source, balanced) == rank).sum())
        for source in range(world)
    )
    executor.quant = data["quantMode"]
    executor.int32 = x.dtype == torch.int32
    hierarchy = data["commAlg"] == "hierarchy"
    capacity = bs * world
    null = ctypes.c_void_p()
    i64, string = ctypes.c_int64, ctypes.c_char_p
    suffix = {1: "", 2: "V2", 3: "V5"}[variant]
    executor.dispatch_api = "aclnnMoeDistributeDispatch" + suffix
    executor.combine_api = (
        "aclnnMoeDistributeCombine" + suffix if kind != "arn" else target
    )
    x_arg = tensor(executor, x)
    ids_arg = tensor(executor, expert_ids)
    weights_arg = tensor(executor, weights)
    smooth = (
        tensor(executor, torch.ones((bs,) if executor.int32 else (world, hidden)))
        if options["use_smooth"]
        else null
    )
    expert_scales = weights_arg if hierarchy else null
    expanded = zeros(
        executor, (capacity, hidden), torch.int8 if executor.quant else x.dtype
    )
    dynamic = zeros(executor, (capacity,), torch.float32)
    assist = zeros(
        executor, (bs * topk if variant == 1 else capacity * 128,), torch.int32
    )
    tokens = zeros(executor, (1,), torch.int64)
    extended_counts = hierarchy or (platform == "A2" and variant == 1)
    count_size = world + (bs * world * 2 * topk * world // 8 if extended_counts else 0)
    counts = zeros(executor, (count_size,), torch.int32)
    tp_counts = (
        zeros(executor, (1,), torch.int32) if platform == "A2" or variant == 3 else null
    )
    # Preserve each frozen scenario's native optional-output descriptor contract.
    scale_size = capacity if hierarchy or (variant == 1 and not balanced) else 0
    scales = zeros(executor, (scale_size,), torch.float32)
    executor.dispatch_outputs = [
        expanded,
        dynamic,
        assist,
        tokens,
        counts,
        tp_counts,
        scales,
    ]
    payload_dtype = torch.float16 if executor.int32 else x.dtype
    payload = (
        zeros(executor, (capacity, hidden), payload_dtype)
        if executor.quant or executor.int32
        else expanded
    )
    executor.payload_dtype = payload_dtype
    combine_tp_counts = tp_counts if balanced else null
    shared = 1 if variant == 1 else 0
    mode = string() if data["commAlg"] is None else string(data["commAlg"].encode())
    quant_comm = data["commQuantMode"]
    if variant == 3:
        from cann_ops_transformer.ops import MoeDistributeBuffer

        # A3 KFC uses the default HCCL window sized before process-group creation.
        executor.comm_buffer = MoeDistributeBuffer(dist.group.WORLD)
        context = tensor(executor, executor.comm_buffer.context)
        buffer_bytes = executor.comm_buffer.ccl_buffer_size
        dispatch_args = [
            context,
            x_arg,
            ids_arg,
            smooth,
            null,
            expert_scales,
            null,
            null,
        ]
        dispatch_args += [
            i64(v)
            for v in (
                world,
                rank,
                world,
                buffer_bytes,
                1,
                0,
                0,
                shared,
                0,
                executor.quant,
                bs * world,
                1,
            )
        ]
        # V5's no-algorithm value is an empty string, including the uniform scenario.
        mode = string((data["commAlg"] or "").encode())
        dispatch_args += [mode, i64(0), i64(0), i64(0)]
        combine_inputs = [
            context,
            payload,
            ids_arg,
            assist,
            counts,
            weights_arg,
            combine_tp_counts,
            null,
            null,
            null,
            null,
            scales if hierarchy else null,
        ] + [null] * 7
        combine_attrs = [
            i64(v)
            for v in (
                world,
                rank,
                world,
                buffer_bytes,
                1,
                0,
                0,
                shared,
                0,
                bs * world,
                0,
                quant_comm,
                0,
            )
        ]
        combine_attrs += [mode, i64(0), i64(0), i64(0)]
    else:
        group = string(
            dist.group.WORLD._get_backend(torch.device("npu"))
            .get_hccl_comm_name(rank)
            .encode()
        )
        dispatch_args = [x_arg, ids_arg, smooth, null, expert_scales, group]
        dispatch_args += [i64(v) for v in (world, rank, world)] + [string(b"")]
        dispatch_args += [
            i64(v) for v in (1, 0, 0, shared, 0, executor.quant, bs * world, 1)
        ]
        if variant == 2:
            dispatch_args.append(mode)
        combine_inputs = [
            payload,
            ids_arg,
            assist,
            counts,
            weights_arg,
            combine_tp_counts,
            null,
            null,
            null,
            null,
            scales if hierarchy else null,
        ]
        if variant == 2:
            combine_inputs.append(null)
        combine_attrs = [group] + [i64(v) for v in (world, rank, world)] + [string(b"")]
        combine_attrs += [
            i64(v) for v in (1, 0, 0, shared, 0, bs * world, 0, quant_comm, 0)
        ]
        if variant == 2:
            combine_attrs.append(mode)
    dispatch_args += executor.dispatch_outputs
    if kind == "arn":
        combine_inputs = [
            payload,
            ids_arg,
            assist,
            counts,
            weights_arg,
            tensor(executor, data["residual"]),
            tensor(executor, data["gamma"]),
            combine_tp_counts,
            null,
            null,
            null,
            null,
            scales if hierarchy else null,
            null,
        ]
        combine_attrs[-2] = i64(1)
        combine_attrs.append(ctypes.c_float(1e-6))
        executor.combine_outputs = [
            zeros(executor, (bs, 1, hidden), x.dtype),
            zeros(executor, (bs, 1, 1), torch.float32),
            zeros(executor, (bs, 1, hidden), x.dtype),
        ]
    else:
        executor.combine_outputs = [zeros(executor, (bs, hidden), payload_dtype)]
    executor.dispatch_args = dispatch_args
    executor.combine_args = combine_inputs + combine_attrs + executor.combine_outputs
    target_args = (
        executor.dispatch_args if kind == "dispatch" else executor.combine_args
    )
    outputs = [expanded, tokens] if kind == "dispatch" else executor.combine_outputs
    if kind == "dispatch" and (executor.quant or executor.int32):
        outputs.append(dynamic)
    target_ids = {id(arg) for arg in target_args}
    executor.extra_packages = [
        package for package in executor.packages if id(package) not in target_ids
    ]
    torch.npu.synchronize()
    return target_args, outputs


def partner_call(api, arguments):
    library = ctypes.CDLL(get_opp_lib_path(api))
    args = [
        ctypes.cast(arg.tensor, ctypes.c_void_p) if hasattr(arg, "tensor") else arg
        for arg in arguments
    ]
    workspace_size = ctypes.c_uint64()
    op_executor = ctypes.c_void_p()
    query = getattr(library, api + "GetWorkspaceSize")
    query.argtypes = [type(arg) for arg in args] + [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    query.restype = ctypes.c_int
    rc = query(*args, ctypes.byref(workspace_size), ctypes.byref(op_executor))
    if rc:
        raise RuntimeError(f"{api}GetWorkspaceSize failed: {rc}")
    workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device="npu")
    launch = getattr(library, api)
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    rc = launch(
        workspace.data_ptr(),
        workspace_size,
        op_executor,
        torch.npu.current_stream().npu_stream,
    )
    if rc:
        raise RuntimeError(f"{api} failed: {rc}")
    torch.npu.synchronize()


def call(executor):
    if executor.kind != "dispatch":
        partner_call(executor.dispatch_api, executor.dispatch_args)
    executor.backend.aclnn_x_get_workspace_size()
    executor.backend.aclnn_x()
    torch.npu.synchronize()
    if executor.kind == "dispatch":
        if executor.quant or executor.int32:
            payload = executor.acl_tensor_to_torch(executor.dispatch_outputs[0]).float()
            if executor.quant:
                payload *= executor.acl_tensor_to_torch(executor.dispatch_outputs[1])[
                    :, None
                ]
            replacement = tensor(executor, payload.to(executor.payload_dtype))
            executor.extra_packages.append(replacement)
            executor.combine_args[1 if executor.variant == 3 else 0] = replacement
        partner_call(executor.combine_api, executor.combine_args)


def outputs(executor):
    if executor.kind == "dispatch":
        tokens = executor.acl_tensor_to_torch(executor.dispatch_outputs[0])[
            : executor.valid_tokens
        ]
        indices = token_order(tokens, executor.balanced)
        result = [
            tokens[indices],
            executor.acl_tensor_to_torch(executor.dispatch_outputs[3]),
        ]
        if executor.quant or executor.int32:
            result.append(
                executor.acl_tensor_to_torch(executor.dispatch_outputs[1])[
                    : executor.valid_tokens
                ][indices]
            )
    else:
        result = [
            executor.acl_tensor_to_torch(package)
            for package in executor.combine_outputs
        ]
    for package in executor.extra_packages:
        nnopbase.aclrt_destroy_arg(package)
    executor.extra_packages = []
    return result
