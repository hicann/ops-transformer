# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""One designed CSV catalogue and deterministic PA/TH inputs. No runtime grid expansion."""

import ast
import csv
import functools
import hashlib
import json
import os
from pathlib import Path

import torch

_CASE_DIR = Path(__file__).resolve().parents[1] / "ttk"
_CASE_SUITES = ("interception", "redline", "generalization", "stc", "performance")


def _row_configuration(row):
    return json.loads(ast.literal_eval(row["attributes"])["case_config"])


def selected_architecture():
    arch = os.environ.get("KDA_ARCH", "arch35")
    if arch not in ("arch35", "arch92"):
        raise ValueError("KDA_ARCH must be arch35 or arch92")
    return arch


def _catalogue_path(suite):
    return _CASE_DIR / selected_architecture() / f"ttk_cases_{suite}.csv"


@functools.lru_cache(None)
def catalogue():
    rows = []
    for suite in _CASE_SUITES:
        path = _catalogue_path(suite)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: _row_configuration(row)["ordinal"])
    cases = {}
    aliases = {}
    for row in rows:
        c = _row_configuration(row)
        cid = row["testcase_name"]
        if cid in cases or c["case_id"] != cid:
            raise ValueError("Duplicate or inconsistent designed case ID: " + cid)
        cases[cid] = c
        for old in c["aliases"]:
            if old in aliases and aliases[old] != cid:
                raise ValueError("Ambiguous old case ID: " + old)
            aliases[old] = cid
    if not cases:
        raise ValueError("Designed case CSV is empty")
    return cases, aliases


def get(case_id):
    cases, _ = catalogue()
    if case_id not in cases:
        raise KeyError("Not a canonical designed case ID: " + case_id)
    return dict(cases[case_id])


def resolve_id(case_id):
    cases, aliases = catalogue()
    if case_id in cases:
        return case_id
    if case_id in aliases:
        return aliases[case_id]
    raise KeyError(case_id)


def select(suite="all", ids=None):
    cases, _ = catalogue()
    rows = list(cases.values())
    if ids is not None:
        wanted = {resolve_id(cid) for cid in ids}
        rows = [c for c in rows if c["case_id"] in wanted]
    if suite in ("all", "mega"):
        return [dict(c) for c in rows]
    if suite == "smoke":
        suite = "typical"
    if suite == "readme":
        suite = "readme_grid"
    if suite == "interception":
        return [dict(c) for c in rows if c["expected"] == "REJECT_ATOMIC"]
    return [dict(c) for c in rows if suite == c["category"] or suite in c["coverage"]]


def write_csv(path, rows, *, for_ttk=False):
    # Export the full TTK rows so every tensor declaration survives selection.
    selected = {c["case_id"] for c in rows}
    records = []
    for suite in _CASE_SUITES:
        with (_catalogue_path(suite)).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            records.extend(row for row in reader if row["testcase_name"] in selected)
    records.sort(key=lambda row: _row_configuration(row)["ordinal"])
    if {row["testcase_name"] for row in records} != selected:
        raise ValueError("Selection contains an unknown case")
    if for_ttk:
        entrypoints = {
            "PASS_NUMERIC": "kda_ttk_ops.verify_csv",
            "REJECT_ATOMIC": "kda_ttk_ops.reject",
            "PASS_CAPTURE": "kda_ttk_ops.capture",
        }
        for row in records:
            row["api_name"] = entrypoints[_row_configuration(row)["expected"]]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


@functools.lru_cache(None)
def load_operator():
    if __package__ == "assets":
        from _mega_kda_paths import load_sample
    else:
        from .._mega_kda_paths import load_sample
    return load_sample("mega_recurrent_kda/mega_recurrent_kda.py")


# Deterministic tensor construction shared by pytest and TTK.

REPLAY_POOLS = ("replay_u", "replay_k", "replay_decay")
PA_POOLS = ("conv_state", "checkpoint_state", *REPLAY_POOLS)


def _replay_pool(backing, config, name):
    return backing[:: config.get(name + "_step", 1)]


NAMES = [
    "hidden_states",
    "qkv_projection_weight",
    "decay_projection_a_weight",
    "decay_projection_b_weight",
    "beta_projection_weight",
    "output_gate_projection_weight",
    "output_norm_weight",
    "output_projection_weight",
    "conv1d_weight",
    "conv_state",
    "checkpoint_state",
    "replay_u",
    "replay_k",
    "replay_decay",
    "state_indices",
    "conv_num_accepted_tokens",
    "A_log",
    "dt_bias",
]
WEIGHTS = (1, 2, 3, 4, 5, 7)


def _make(config=None, *, device="cpu", poison=False):
    c = dict(B=2, N=6, S=8, layout="BSH", seed=3, conv_step=2, checkpoint_step=3)
    c.update(config or {})
    b, n, s = c["B"], c["N"], c["S"]
    p = n * 128
    lengths = c.get("lengths", [s] * b)
    capacity = c.get("capacity", max(lengths, default=max(s, 1)))
    blocks = c.get("blocks", b + 2)
    length = c.get("L", max(max(lengths, default=1) + 2, 3))
    width = c.get("H", 7168)
    g = torch.Generator().manual_seed(c["seed"])

    def rand(shape, dt=torch.bfloat16, scale=0.02):
        if device == "meta":
            return torch.empty(shape, dtype=dt, device="meta")
        return (torch.randn(shape, generator=g) * scale).to(dt)

    xshape = (sum(lengths), width) if c["layout"] == "TH" else (b, s, width)
    tensors = [
        rand(xshape),
        rand((3 * p, width)),
        rand((128, width)),
        rand((p, 128)),
        rand((n, width)),
        rand((p, width)),
        rand((128,), torch.float32, 0.2),
        rand((width, p)),
        rand((4, 3 * p)),
    ]
    conv_backing = rand((blocks * c["conv_step"], length, 3 * p))
    conv = conv_backing[:: c["conv_step"]]
    cp_backing = rand(
        (blocks * c["checkpoint_step"], n, 128, 128), torch.float32, 0.002
    )
    cp = cp_backing[:: c["checkpoint_step"]]
    # 按 PA 块数分配记录池。常规精度用例将未写入区域初始化为零。
    fill = float("nan") if poison else float(c.get("records_fill", 0.0))
    records = [
        _replay_pool(
            torch.full(
                (blocks * c.get(name + "_step", 1), n, capacity, 128),
                fill,
                dtype=dt,
                device=device,
            ),
            c,
            name,
        )
        for name, dt in zip(REPLAY_POOLS, (torch.float32, torch.float16, torch.float32))
    ]
    indices = torch.tensor(
        c.get("state_indices", list(range(b - 1, -1, -1))),
        dtype=torch.int32,
        device=device,
    )
    counts = torch.tensor(
        c.get("conv_accepted", [0] * b), dtype=torch.int32, device=device
    )
    tensors += [
        conv,
        cp,
        *records,
        indices,
        counts,
        rand((n,), torch.float32, 0.1),
        rand((n, 128), torch.float32, 0.1),
    ]
    if device == "meta":
        cu = (
            torch.empty(b + 1, dtype=torch.int32, device="meta")
            if c["layout"] == "TH"
            else None
        )
    else:
        cu = (
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            )
            if c["layout"] == "TH"
            else None
        )
    args = dict(zip(NAMES, tensors))
    args.update(scale=128**-0.5, lower_bound=-5.0, rms_norm_eps=1e-5, cu_seqlens=cu)
    if c.get("zero"):
        for name in NAMES[:11] + ["A_log", "dt_bias"]:
            args[name].zero_()
        args["output_norm_weight"].fill_(1.0)
    for key in ("scale", "lower_bound", "rms_norm_eps"):
        if key in c:
            args[key] = float(c[key])
    if cu is not None and c.get("cu_dtype") == "int64":
        args["cu_seqlens"] = cu.long()
    form = c.get("conv_weight_layout", "flat")
    if form != "flat":
        grouped = args["conv1d_weight"].reshape(4, 3, n, 128).permute(1, 2, 0, 3)
        args["conv1d_weight"] = (
            grouped if form == "per_head" else grouped[:, 0]
        ).contiguous()
    return args, dict(conv=conv_backing, checkpoint=cp_backing), c


def to_npu(args):
    import torch_npu

    result = {}
    for name, value in args.items():
        if not isinstance(value, torch.Tensor):
            result[name] = value
            continue
        if name in PA_POOLS:
            # Preserve genuine PA gaps; this is allocation/copy, not arithmetic.
            actual = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device="npu"
            )
            if value.numel() > 0:
                span = (value.shape[0] - 1) * value.stride(0) + value[0].numel()
                actual.as_strided((span,), (1,)).fill_(43.0)
            actual.copy_(value)
        else:
            actual = value.npu()
        if name in [NAMES[i] for i in WEIGHTS]:
            actual = torch_npu.npu_format_cast(actual, 29)
        result[name] = actual
    return result


def as_positional(args):
    return [args[name] for name in NAMES]


def make(config=None, *, device="cpu", poison=False):
    assert device in ("cpu", "meta"), "Use to_npu to preserve PA strides and NZ weights"
    with torch.device(device):
        return _make(config, device=device, poison=poison)


def digest(tensor):
    if tensor.device.type in ("npu", "privateuseone"):
        import torch_npu

        if torch_npu.get_npu_format(tensor) == 29:
            tensor = torch_npu.npu_format_cast(tensor, 2)
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def manifest(args):
    result = {}
    for name, t in args.items():
        if not isinstance(t, torch.Tensor):
            continue
        entry = dict(
            shape=list(t.shape),
            stride=list(t.stride()),
            dtype=str(t.dtype),
            sha256=digest(t),
        )
        if t.device.type in ("npu", "privateuseone"):
            import torch_npu

            entry["npu_format"] = int(torch_npu.get_npu_format(t))
        result[name] = entry
    return result


def mutate(args, kind, details=None):
    """Mutations describe the new contract, independent of historical messages."""
    if kind == "generic":
        return _generic_mutation(args, details)
    if kind == "accepted_oob":
        args["num_accepted_tokens"][0] = args["replay_u"].shape[2] + 1
    elif kind == "missing_cu":
        args["cu_seqlens"] = None
    elif kind == "extra_cu":
        args["cu_seqlens"] = torch.arange(
            args["state_indices"].numel() + 1,
            dtype=torch.int32,
            device=args["hidden_states"].device,
        )
    elif kind == "cu_start":
        args["cu_seqlens"][0] = 1
    elif kind == "cu_end":
        args["cu_seqlens"][-1] -= 1
    elif kind == "cu_empty":
        args["cu_seqlens"][1] = args["cu_seqlens"][0]
    elif kind == "cu_descending":
        args["cu_seqlens"][1] = -1
    elif kind == "cu_dtype":
        args["cu_seqlens"] = args["cu_seqlens"].float()
    elif kind == "cu_shape":
        args["cu_seqlens"] = args["cu_seqlens"].reshape(1, -1)
    elif kind == "negative_index":
        args["state_indices"][0] = -1
    elif kind == "index_oob":
        args["state_indices"][0] = args["checkpoint_state"].shape[0]
    elif kind == "duplicate_index":
        args["state_indices"][1] = args["state_indices"][0]
    elif kind == "count_negative":
        args[
            "conv_num_accepted_tokens"
            if "conv_num_accepted_tokens" in args
            else "num_accepted_tokens"
        ][0] = -1
    elif kind == "count_oob":
        args["conv_num_accepted_tokens"][0] = args["conv_state"].shape[1] - 1
    elif kind == "u_fp16":
        args["replay_u"] = args["replay_u"].half()
    elif kind == "record_layout":
        args["replay_k"] = args["replay_k"].transpose(1, 2).contiguous()
    elif kind == "checkpoint_shape":
        args["checkpoint_state"] = args["checkpoint_state"][:-1]
    elif kind == "hidden_dtype":
        args["hidden_states"] = args["hidden_states"].half()
    elif kind == "inner_stride":
        args["conv_state"] = args["conv_state"].transpose(1, 2)
    elif kind == "epsilon":
        args["rms_norm_eps"] = 1e-40
    elif kind == "epsilon_zero":
        args["rms_norm_eps"] = 0.0
    elif kind == "lower_bound":
        args["lower_bound"] = 1.0
    elif kind == "scale":
        args["scale"] = float("nan")
    elif kind == "missing_k":
        args.pop("replay_k")
    elif kind == "missing_decay":
        args.pop("replay_decay")
    elif kind == "weight_nd":
        import torch_npu

        args["qkv_projection_weight"] = torch_npu.npu_format_cast(
            args["qkv_projection_weight"], 2
        )
    elif kind == "pa_nz":
        import torch_npu

        args["conv_state"] = torch_npu.npu_format_cast(args["conv_state"], 29)
    elif kind is not None:
        raise KeyError(kind)
    return args


def _generic_mutation(args, d):
    op = d["op"]
    key = d.get("target")
    t = args.get(key)
    if op == "dtype":
        args[key] = t.to(getattr(torch, d["dtype"]))
    elif op == "scalar":
        args[key] = d["value"]
    elif op == "bad_shape":
        shape = list(t.shape)
        if key == "hidden_states":
            shape = [t.numel()]
        elif key in (
            "state_indices",
            "num_accepted_tokens",
            "conv_num_accepted_tokens",
        ):
            shape = [1, *shape]
        else:
            shape[-1] = max(1, shape[-1] - 1)
        args[key] = torch.zeros(shape, dtype=t.dtype, device=t.device)
    elif op == "slice0":
        args[key] = t[:-1]
    elif op == "transpose":
        args[key] = t.transpose(*d["dims"]).contiguous()
    elif op == "noncontiguous":
        shape = list(t.shape)
        shape[-1] *= 2
        base = torch.empty(shape, dtype=t.dtype, device=t.device)
        view = base[..., ::2]
        view.copy_(t)
        args[key] = view
    elif op == "axis0_overlap":
        stride = list(t.stride())
        stride[0] = t[0].numel() - 1
        args[key] = t.as_strided(t.shape, stride)
    elif op == "requires_grad":
        args[key] = t.detach().requires_grad_(True)
    elif op == "cpu_device":
        args[key] = t.cpu()
    elif op == "parameter":
        args[key] = torch.nn.Parameter(t, requires_grad=False)
    elif op == "record_batch_mismatch":
        args[key] = t[:-1]
    elif op == "accepted_value":
        args["num_accepted_tokens"][0] = d["value"]
    elif op in ("format_nd", "format_nz"):
        import torch_npu

        args[key] = torch_npu.npu_format_cast(t, 2 if op == "format_nd" else 29)
    elif op == "alias":
        other = d["other"]
        x = args[other]
        if t.shape == x.shape and t.is_contiguous():
            args[key] = x
        else:
            span = (t.shape[0] - 1) * t.stride(0) + t[0].numel()
            base = torch.zeros(max(span, x.numel()), dtype=t.dtype, device=t.device)
            args[key] = base.as_strided(t.shape, t.stride())
            args[other] = base[: x.numel()].view(x.shape)
    else:
        raise KeyError(op)
    return args


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export only canonical designed cases; no runtime expansions"
    )
    parser.add_argument("--suite", default="all")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--for-ttk",
        action="store_true",
        help="Export a temporary CSV with TTK plugin entrypoints",
    )
    ns = parser.parse_args()
    rows = select(ns.suite, ids=ns.ids)
    if not rows:
        parser.error("Selection contains no designed cases")
    write_csv(ns.output, rows, for_ttk=ns.for_ttk)
    print(f"Exported {len(rows)} designed cases to {ns.output}")


if __name__ == "__main__":
    _main()
