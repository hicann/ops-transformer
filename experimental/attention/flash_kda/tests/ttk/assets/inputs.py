# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""KDA input generation and explicit invalid-input fixtures for TTK."""

import json
import re
import torch

INPUT_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "metadata",
    "initial_state",
    "A_log",
    "dt_bias",
    "cu_seqlens",
    "seqused",
)
OPERATOR_DEFAULTS = {
    "use_qk_l2norm_in_kernel": True,
    "use_gate_in_kernel": True,
    "use_beta_sigmoid_in_kernel": True,
    "allow_neg_eigval": False,
    "disable_recompute": False,
    "output_final_state": False,
    "return_intermediate_states": False,
    "scale": 128 ** (-0.5),
    "lower_bound": None,
}


def configuration(inputs, attributes):
    """Use TTK's supplied tensors/attributes; never reopen a CSV or inspect its ID."""
    q, _, v, g, beta, _, state = inputs[:7]
    layout = attributes.get("layout", "TND")
    packed = layout == "TND"
    batch = state.shape[0] if packed else q.shape[0]
    length = q.shape[0] if packed else q.shape[2 if layout == "BNSD" else 1]
    head_axis = 1 if layout in ("BNSD", "TND") else 2
    test = json.loads(attributes.get("test_config", "{}"))
    kind = test.get("kind", "accuracy")
    omitted = set(attributes.get("omitted_attributes", "").split(","))
    operator = {
        key: attributes.get(key, default)
        for key, default in OPERATOR_DEFAULTS.items()
        if key not in omitted
    }
    c = dict(
        id=attributes.get("testcase_name", ""),
        layout=layout,
        omit_layout="layout" in omitted,
        batch_size=batch,
        nheads_qk=q.shape[head_axis],
        nheads_v=v.shape[head_axis],
        seqlens=length,
        g_dtype=str(g.dtype).removeprefix("torch."),
        beta_dtype=str(beta.dtype).removeprefix("torch."),
        kwargs=operator,
        inference=attributes.get("inference", False),
        no_grad=attributes.get("no_grad", False),
        initial_state_mode=attributes.get("initial_state_mode", "none"),
        dt_bias_mode=attributes.get("dt_bias_mode", "none"),
        seed=attributes.get("seed", 101),
        distribution=attributes.get("distribution", 1.0),
        cu_seqlens_value=attributes.get("cu_seqlens_value"),
        seqused_value=attributes.get("seqused_value"),
        valid_lengths=list(attributes.get("valid_lengths") or [length] * batch),
        test=test,
        suite=(
            "whitebox"
            if kind
            in (
                "sizes",
                "capacity",
                "length_contents",
                "producer_result",
                "runtime_error",
                "buffer",
                "out_stride",
            )
            else kind
        ),
        pending=test.get("pending"),
        ignored=test.get("ignored", {}),
    )
    for key in ("raw_g", "tail"):
        if attributes.get(key) is not None:
            c[key] = attributes[key]
    return c


def effective(c):
    return dict(OPERATOR_DEFAULTS, **c["kwargs"])


def to_layout(x, layout, lengths):
    if layout == "TND":
        return torch.cat(
            [x[b, :, :n].transpose(0, 1) for b, n in enumerate(lengths)], 0
        ).contiguous()
    if layout == "BSND":
        return x.transpose(1, 2).contiguous()
    return x.contiguous()


def sequences(x, c, *, norm=False):
    """Slice only lengths recorded in the test recipe; never read device length tensors."""
    if c["layout"] == "TND":
        pos = 0
        for n in c["valid_lengths"]:
            yield x[pos : pos + n].transpose(0, 1).unsqueeze(0)
            pos += n
    else:
        y = x if norm or c["layout"] == "BNSD" else x.transpose(1, 2)
        for b, n in enumerate(c["valid_lengths"]):
            yield y[b : b + 1, :, :n]


def make(c, input_ranges=()):
    ranges = dict(zip(INPUT_NAMES, input_ranges or ()))
    b, nq, nv = c["batch_size"], c["nheads_qk"], c["nheads_v"]
    lengths = c["valid_lengths"]
    s = max(lengths) if c["layout"] == "TND" else c["seqlens"]
    gen = torch.Generator().manual_seed(c["seed"])

    def sample(name, shape, default):
        bounds = ranges.get(name)
        if bounds is not None and None not in bounds:
            return torch.empty(shape, dtype=torch.float32).uniform_(
                *bounds, generator=gen
            )
        return default(shape)

    def rand(shape):
        return (torch.rand(shape, generator=gen) * 2 - 1) * c["distribution"]

    def small_rand(shape):
        return torch.rand(shape, generator=gen) * 0.2 - 0.1

    q, k = [sample(name, (b, nq, s, 128), rand).bfloat16() for name in ("q", "k")]
    v = sample("v", (b, nv, s, 128), rand).bfloat16()
    g = sample("g", (b, nv, s, 128), rand).to(getattr(torch, c["g_dtype"]))
    beta = sample("beta", (b, nv, s), rand).to(getattr(torch, c["beta_dtype"]))
    state_shape = (b, nv, 128, 128)
    state = (
        sample("initial_state", state_shape, small_rand)
        if c["initial_state_mode"] == "random"
        else torch.zeros(state_shape)
    )
    a = sample("A_log", (nv,), lambda shape: torch.linspace(-0.25, 0.25, shape[0]))
    if c["dt_bias_mode"] == "normal":
        # Reproduce the same bias independently of other input RNG consumption.
        bias_gen = torch.Generator(device="cpu").manual_seed(c["seed"])
        bias = torch.randn(nv, 128, generator=bias_gen, dtype=torch.float32)
    else:
        bias = (
            sample("dt_bias", (nv, 128), small_rand)
            if c["dt_bias_mode"] == "random"
            else torch.zeros(nv, 128)
        )
    f = effective(c)
    if "raw_g" in c:
        g.fill_(c["raw_g"])
    if not f["use_qk_l2norm_in_kernel"]:
        q = (
            q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        k = (
            k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        ).bfloat16()
    if not f["use_gate_in_kernel"]:
        logits = g.float() + bias[None, :, None, :]
        decay = a.exp()[None, :, None, None]
        bound = f["lower_bound"]
        gate = (
            -decay * torch.nn.functional.softplus(logits)
            if bound is None
            else bound * torch.sigmoid(decay * logits)
        )
        g = gate.to(g.dtype)
    if not f["use_beta_sigmoid_in_kernel"]:
        beta = beta.float().sigmoid().to(beta.dtype)
    if "tail" in c:
        value = float(c["tail"])
        for x in (q, k, v, g, beta):
            x[0, :, lengths[0] :] = value
    values = [to_layout(x, c["layout"], lengths) for x in (q, k, v, g, beta)]
    cu = c["cu_seqlens_value"] or [0] + list(torch.tensor(lengths).cumsum(0).tolist())
    used = c["seqused_value"] or lengths
    inputs = values + [
        None,
        state,
        a,
        bias,
        torch.tensor(cu, dtype=torch.int32),
        torch.tensor(used, dtype=torch.int32),
    ]
    # Generate the same ABI4 tensor used by a real caller, before DUT execution.
    from kda_ttk_ops import mode, produce

    with mode(c):
        inputs[5] = produce(inputs, c).cpu()
    return inputs


def customize_inputs(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    **kwargs,
):
    inputs = [
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
    ]
    c = configuration(inputs, kwargs)
    values = make(c, kwargs.get("input_ranges"))
    for name, target, value in zip(INPUT_NAMES, inputs, values):
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(
                f"{name}: CSV {target.shape}/{target.dtype} differs from input generator {value.shape}/{value.dtype}"
            )
        target.copy_(value)
    return tuple(inputs)


def noncontiguous(x):
    backing = torch.empty((*x.shape, 2), dtype=x.dtype, device=x.device)
    view = backing[..., 0]
    view.copy_(x)
    if view.is_contiguous():
        raise AssertionError("noncontiguous fixture construction failed")
    return view


def mutate(kw, c):
    test = c["test"]
    recipe = test["mutation"]
    op, name, value = recipe
    reasons = {
        "value": [
            "must",
            "expected",
            "required",
            "invalid",
            "requires",
            "bool",
            "Tensor",
            "finite",
            "支持",
            "类型",
            "区间",
        ],
        "missing": ["required", "missing", "None", "缺少"],
        "rank": ["rank", "dimension", "维"],
        "dtype": ["dtype", "type", "BF16", "类型"],
        "axis": ["shape", "dimension", "head", "维", "形状"],
        "shape": ["shape", "rank", "header", "short", "维", "形状"],
        "noncontiguous": ["contiguous", "stride", "连续"],
        "all_cpu": ["device", "NPU", "设备"],
        "empty": ["positive", "head", "size", "length", "正", "尺寸"],
        "gqa": ["head", "GQA", "divis", "头"],
        "unsqueeze": ["rank", "dimension", "维"],
    }
    if "mutation_base" in test:
        key, shape, dtype = test["mutation_base"]
        if key != "metadata" and (
            kw.get(key) is None or key in ("A_log", "dt_bias", "seqused")
        ):
            kw[key] = torch.zeros(
                shape, dtype=getattr(torch, dtype), device=kw["q"].device
            )
    if op == "missing":
        if value == "omit":
            kw.pop(name, None)
        else:
            kw[name] = None
    elif op == "value":
        kw[name] = (
            float(value)
            if name == "lower_bound" and value in ("NaN", "+Inf", "-Inf")
            else value
        )
    elif op == "rank":
        kw[name] = kw[name].unsqueeze(0) if value > 0 else kw[name][0]
    elif op == "unsqueeze":
        kw[name] = kw[name].unsqueeze(value)
    elif op == "dtype":
        kw[name] = kw[name].to(getattr(torch, value))
    elif op in ("axis", "shape"):
        shape = list(kw[name].shape)
        if op == "axis":
            shape[value[0]] = value[1]
        else:
            shape = value
        kw[name] = torch.zeros(shape, dtype=kw[name].dtype, device=kw[name].device)
        if test.get("resize_dependents"):
            kw["g"] = torch.zeros_like(kw["v"], dtype=torch.float32)
            kw["beta"] = torch.zeros(
                kw["v"].shape[:-1], dtype=torch.float32, device=kw["v"].device
            )
    elif op == "noncontiguous":
        kw[name] = noncontiguous(kw[name])
    elif op == "all_cpu":
        kw = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
    elif op in ("empty", "gqa"):
        b, nq, nv, s = c["batch_size"], c["nheads_qk"], c["nheads_v"], c["seqlens"]
        if op == "gqa":
            nq, nv = value
        elif name == "b":
            b = 0
        elif name == "nq":
            nq = 0
        elif name == "nv":
            nv = 0
        else:
            s = 0
        for key, heads in [("q", nq), ("k", nq), ("v", nv), ("g", nv), ("beta", nv)]:
            shape = (s, heads, 128) if c["layout"] == "TND" else (b, heads, s, 128)
            if key == "beta":
                shape = shape[:-1]
            kw[key] = torch.zeros(shape, dtype=kw[key].dtype, device=kw[key].device)
        kw["initial_state"] = torch.zeros((b, nv, 128, 128), device=kw["q"].device)
    else:
        raise AssertionError(f"unimplemented mutation {recipe}")
    objects = [re.escape(str(name))]
    if op in ("empty", "gqa"):
        objects = ["batch", "head", "Nqk", "Nv", "length", "size", "storage", "GQA"]
    if op == "all_cpu":
        objects = ["device", "NPU"]
    objects = test.get("error_objects", objects)
    return kw, objects, test.get("error_patterns", reasons[op])


# ABI 4 field layout. Fixtures are independent of the DUT metadata decoder.
HEADER_FIELDS = (
    "magic",
    "abi",
    "status",
    "cores",
    "rounds",
    "batch",
    "heads",
    "has_cu",
    "slots",
    "template",
    "reserved",
)
CORE_SLOTS = 32


def rebuild_record(record, heads):
    for stage in (1, 2):
        increments = record["count"] if stage == 1 else record["splits"]
        prefix = [0]
        for count in increments:
            prefix.append(prefix[-1] + count * heads)
        record[f"p{stage}"] = prefix
        record[f"r{stage}"] = [0, prefix[-1]] + [0] * (2 * CORE_SLOTS - 2)


def metadata_baseline(c, cores):
    lengths = c["valid_lengths"]
    packed = c["layout"] == "TND"
    chunks = [(n + 63) // 64 for n in lengths]
    starts = [sum(lengths[:b]) if packed else 0 for b in range(len(lengths))]
    multi_round = c["test"].get("schedule") == "multi_round"
    records = []
    for chunk in range(max(chunks) if multi_round else 1):
        batches = [b for b, count in enumerate(chunks) if count > chunk]
        r = dict(
            batch=batches,
            physical=[0 if packed else b for b in batches],
            token=[starts[b] for b in batches],
            length=[lengths[b] for b in batches],
            group=1 if multi_round else max(chunks),
            chunk=[chunk] * len(batches),
            count=[1 if multi_round else chunks[b] for b in batches],
            splits=[1] * len(batches),
        )
        rebuild_record(r, c["nheads_v"])
        records.append(r)
    header = [
        0x4B444132,
        4,
        0,
        cores,
        len(records),
        c["batch_size"],
        c["nheads_v"],
        1,
        861,
        1,
        0,
    ]
    return header, records


def encode_metadata(header, records):
    header = header[:]
    header[HEADER_FIELDS.index("rounds")] = len(records)
    data = []
    offsets = [len(header) + len(records) + 1]
    for index, record in enumerate(records):
        row = [record.get("index", index), len(record["batch"])]
        for key in ("batch", "physical", "token", "length"):
            row += record[key]
        row += [record["group"]]
        for key in ("chunk", "count", "p1", "r1", "splits", "p2", "r2"):
            row += record[key]
        data += row
        offsets.append(offsets[-1] + len(row))
    return header + offsets + data


def mutate_metadata(c, cores):
    header, records = metadata_baseline(c, cores)
    test = c["test"]
    change = test["change"]
    record = records[test.get("round", 0)]
    if change == "header":
        value = cores + 1 if test["value"] == "cores+1" else test["value"]
        header[HEADER_FIELDS.index(test["field"])] = value
        if test.get("rebuild"):
            for r in records:
                rebuild_record(r, value)
    elif change == "record":
        if test.get("entry") is None:
            record[test["field"]] = test["value"]
        else:
            record[test["field"]][test["entry"]] = test["value"]
        if test.get("rebuild"):
            rebuild_record(record, c["nheads_v"])
    elif change == "prefix":
        prefix = record[f"p{test['stage']}"]
        error = test["error"]
        if error == "start":
            prefix[0] = 1
        elif error == "decreasing":
            prefix[1] = -1
        elif error == "increment":
            prefix[1] += 1
        elif error == "total":
            prefix[-1] += 1
            record[f"r{test['stage']}"][1] = prefix[-1]
        else:
            raise ValueError(f"unknown prefix mutation: {error}")
    elif change == "core_range":
        ranges = record[f"r{test['stage']}"]
        total = record[f"p{test['stage']}"][-1]
        error = test["error"]
        if error == "gap":
            ranges[0] = 1
        elif error == "overflow":
            ranges[1] = total + 1
        elif error == "overlap":
            if cores < 2:
                raise RuntimeError(
                    "NOT_APPLICABLE: overlapping ranges require at least two cores"
                )
            ranges[2:4] = [0, 1]
        elif error == "inactive":
            if cores >= CORE_SLOTS:
                raise RuntimeError(
                    "NOT_APPLICABLE: no inactive core slot on this device"
                )
            ranges[2 * cores : 2 * cores + 2] = [0, 1]
        else:
            raise ValueError(f"unknown core range mutation: {error}")
    elif change == "drop_last_round":
        records.pop()
    elif change != "words":
        raise ValueError(f"unknown metadata mutation: {change}")
    words = encode_metadata(header, records)
    for index, value in test.get("edits", []):
        words[index] = {
            "size": len(words),
            "size+1": len(words) + 1,
            "size-1": len(words) - 1,
        }.get(value, value)
    if test.get("truncate") is not None:
        words = words[: test["truncate"]]
    return words
