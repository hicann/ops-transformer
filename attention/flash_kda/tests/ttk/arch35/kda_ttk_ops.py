# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTK adapter for unified FlashKDA training and inference contracts.

The CSV supplies test parameters. Case names are used only in diagnostics.
Additional DUT calls use the same official TTK comparator as the main outputs.
No input-file lookups or numerical DUT fallback are used here.
"""

import ast
import copy
import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import torch

import kda_runtime as runtime
from kda_cases import attributes
from torch._subclasses.fake_tensor import FakeTensorMode

ARCH = "arch35"

if importlib.util.find_spec("ttk.core_modules.comparison") is None:
    from ttk.core_modules.common.comparison.comparison import compare as ttk_compare
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance
else:
    from ttk.core_modules.comparison.comparison import compare as ttk_compare
    from ttk.core_modules.comparison.resolve import resolve_tolerance


@lru_cache(None)
def load_asset(name):
    """Load by path, as in QBSA, to avoid collisions with other operators' assets."""
    folder = Path(__file__).resolve().parent
    if name == "performance":
        folder = folder.parent
    path = folder / "assets" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_flash_kda_ttk_{ARCH}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def input_tensor(inputs, name):
    return inputs[load_asset("inputs").INPUT_NAMES.index(name)]


def dut_modules():
    if runtime.selection()[1] != ARCH:
        raise RuntimeError(f"{ARCH} adapter does not match selected architecture")
    modules = runtime.modules()
    producer = modules[1]
    if (producer.MAGIC, producer.ABI_VERSION, producer.MAX_AIC_CORES) != load_asset(
        "inputs"
    ).PROFILE:
        raise RuntimeError(f"{ARCH} DUT metadata ABI mismatch")
    return modules


def dut_call(kwargs):
    return runtime.public_api().flash_kda(**kwargs)


def rejection_target(kwargs, config):
    if config["test"].get("python_validation"):
        if runtime.selection()[0] != "source":
            raise RuntimeError("strict Python type validation requires KDA_DUT=source")
        return dut_modules()[0].flash_kda(**kwargs)
    return dut_call(kwargs)


def effective(c):
    return load_asset("inputs").effective(c)


def make(c):
    return load_asset("inputs").make(c)


def native_metadata(value, config):
    size = load_asset("inputs").capacity(config)
    if value.numel() < size:
        raise ValueError("transport metadata is shorter than native capacity")
    return value[:size]


def reference(inputs, c, kind, *, forward=False):
    golden = load_asset("golden")
    return golden.select_reference(
        golden.reference(inputs, c, kind), c, forward=forward
    )


def produce(inputs, c, *, template=None):
    """Use the scalar metadata interface specified in the detailed design."""
    _, producer, _ = dut_modules()
    cu = (
        input_tensor(inputs, "cu_seqlens").to("npu")
        if c["cu_seqlens_value"] is not None
        else None
    )
    build = runtime.public_api().flash_kda_metadata
    options = {}
    if template is not None:
        build = getattr(producer, "_build_flash_kda_fwd_metadata")
        options["template"] = template
    return build(
        cu_seqlens=cu,
        seqused=input_tensor(inputs, "seqused").to("npu")
        if c["seqused_value"] is not None
        else None,
        batch_size=c["batch_size"],
        nheads_qk=c["nheads_qk"],
        nheads_v=c["nheads_v"],
        seqlens=c["seqlens"],
        headdim_qk=128,
        headdim_v=128,
        enable_grad=not torch.is_inference_mode_enabled(),
        layout=c["layout"],
        **options,
    )


class BoundaryViolation(RuntimeError):
    pass


event = runtime.event


@contextmanager
def mode(c):
    with torch.inference_mode(c["inference"]):
        with (
            torch.no_grad()
            if c["inference"] or c.get("no_grad")
            else torch.enable_grad()
        ):
            if torch.is_inference_mode_enabled() != c["inference"]:
                raise AssertionError("inference mode query mismatch")
            yield


def call_kwargs(inputs, c, meta=None):
    kw = dict(zip(load_asset("inputs").INPUT_NAMES, inputs))
    kw.update(c["kwargs"])
    if not c["omit_layout"]:
        kw["layout"] = c["layout"]
    if c["initial_state_mode"] == "omit":
        kw.pop("initial_state")
    elif c["initial_state_mode"] == "none":
        kw["initial_state"] = None
    if not effective(c)["use_gate_in_kernel"]:
        kw.pop("A_log")
        kw.pop("dt_bias")
    elif c["dt_bias_mode"] == "none":
        kw["dt_bias"] = None
    kw["cu_seqlens"] = kw["cu_seqlens"] if c["cu_seqlens_value"] is not None else None
    kw["seqused"] = kw["seqused"] if c["seqused_value"] is not None else None
    kw.update(c.get("ignored", {}))
    if meta is not None:
        kw["metadata"] = meta
    return kw


def unchanged(before, after):
    for name, a, b in zip(load_asset("inputs").INPUT_NAMES, before, after):
        if not torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),
            b.detach().cpu().contiguous().view(torch.uint8),
        ):
            raise AssertionError(f"input modified: {name}")


def invoke(inputs, c, md=None, *, meta=False, forward=False, call=None):
    with mode(c):
        if md is None:
            md = input_tensor(inputs, "metadata")
        if (
            not meta
            and c["test"].get("relation", {}).get("overrides", {}).get("inference")
            is not None
        ):
            _, producer, _ = dut_modules()
            schedule = producer.decode_metadata(md.detach().cpu().tolist())
            event(
                c,
                status="schedule_observed",
                inference=c["inference"],
                template=schedule["template"],
                rounds=schedule["stage12_round_num"],
            )
        snapshot = [x.clone() for x in inputs] if not meta else None
        md_snapshot = md.clone() if not meta else None
        kw = call_kwargs(inputs, c, md)
        required = [kw.pop(name) for name in ("q", "k", "v", "g", "beta", "metadata")]
        main, _, _ = dut_modules()
        public = runtime.public_api()
        if call is None:
            call = public.flash_kda_fwd if forward else public.flash_kda
        entry = "flash_kda_fwd" if forward else "flash_kda"
        out = call(*required, **kw)
        load_asset("compare").check(out, inputs, c, meta=meta, forward=forward)
        event(
            c, status="return_protocol_checked", entry=entry, slots=len(out), meta=meta
        )
        if not meta:
            unchanged(snapshot, inputs)
            if not torch.equal(md_snapshot, md):
                raise AssertionError("input modified: metadata")
            event(c, status="inputs_unchanged", entry=entry, standard="binary_equal")
        return out, md


@contextmanager
def guard(*, static=False, allocation=True, casts=True):
    main, _, _ = dut_modules()
    seen = []

    def reject(name):
        def wrapped(*args, **kwargs):
            seen.append(name)
            raise BoundaryViolation("forbidden boundary reached: " + name)

        return wrapped

    with ExitStack() as stack:
        for name in ("_get_compiled_kernel",):
            if not hasattr(main, name):
                raise RuntimeError(f"whitebox binding missing: {name}")
            stack.enter_context(patch.object(main, name, reject(name)))
        if static:
            for name in ("cpu", "tolist", "item"):
                stack.enter_context(
                    patch.object(torch.Tensor, name, reject("Tensor." + name))
                )
            if casts:
                for name in ("to", "contiguous", "float", "half", "bfloat16"):
                    stack.enter_context(
                        patch.object(torch.Tensor, name, reject("Tensor." + name))
                    )
        if allocation:
            for name in (
                "empty",
                "empty_like",
                "empty_strided",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
                "full",
                "full_like",
                "tensor",
                "as_tensor",
            ):
                stack.enter_context(patch.object(torch, name, reject("torch." + name)))
            for name in ("new_empty", "new_zeros", "new_ones", "new_full"):
                stack.enter_context(
                    patch.object(torch.Tensor, name, reject("Tensor." + name))
                )
        yield seen


def reject_call(func, objects, reasons, c, *, expected_error=ValueError):
    """检查算子以参数异常拒绝非法调用，断言失败直接交给 TTK。"""
    try:
        func()
    except (expected_error, RuntimeError) as exc:
        message = str(exc)
        # torch.ops validates its schema before entering Python; it reports
        # argument binding/type errors as RuntimeError, not Python TypeError.
        schema_error = isinstance(exc, RuntimeError) and re.search(
            r"Expected a value of type|missing value for argument|Unknown keyword argument|Expected.*for argument.*found type",
            message,
            re.I,
        )
        if not isinstance(exc, expected_error) and not schema_error:
            raise
        if not any(re.search(p, message, re.I) for p in objects):
            raise AssertionError(f"wrong error object: {message}") from exc
        if not any(re.search(p, message, re.I) for p in reasons):
            raise AssertionError(f"wrong rejection reason: {message}") from exc
        event(
            c,
            status="rejected",
            exception_type=type(exc).__name__,
            message=message,
            type_acceptance="torch_schema" if schema_error else expected_error.__name__,
        )
    else:
        raise AssertionError("invalid input was accepted")


def compare_l2(actual, cpu_chunk, npu_chunk, *, slots=None):
    """Official L2 for calls made inside a functional or recovery test."""

    if slots is None:
        slots = load_asset("compare").SLOTS
    if not len(actual) == len(cpu_chunk) == len(npu_chunk) == len(slots):
        raise AssertionError("L2 requires matching outputs from both CPU references")
    dtypes = [x.dtype for x in actual]
    tolerance = load_asset("golden").FlashKdaTestSpec.tolerance
    standards = resolve_tolerance(tolerance, None, None, dtypes, None)
    reports = []
    for i, (slot, a, g, b) in enumerate(zip(slots, actual, cpu_chunk, npu_chunk)):
        # NumPy cannot expose torch.bfloat16; resolve thresholds before conversion.
        a, g, b = [x.detach().cpu().float() for x in (a, g, b)]
        valid = a.shape == g.shape == b.shape and all(
            torch.isfinite(x).all().item() for x in (a, g, b)
        )
        if not valid:
            precision, log, passed, metrics = (
                "FAIL",
                "shape or finite-value violation",
                False,
                {},
            )
        else:
            precision, log, passed, metrics = ttk_compare(
                [a.numpy()],
                [g.numpy()],
                [dtypes[i]],
                standards=[standards[i]],
                third_parties=[b.numpy()],
            )
            if not a.numel():
                precision = "NOT_APPLICABLE"
        reports.append(
            {
                "slot": slot,
                "dtype": str(dtypes[i]),
                "standard": "cross_check",
                "level": "L2",
                "baseline": "npu_chunk_cpu_simulation",
                "pass": passed,
                "precision": precision,
                "error_info": f"L2 slot={slot}: {metrics.get(0, {}).get('reason', '')}\n{log.strip()}",
                "metrics": metrics.get(0, {}),
            }
        )
    return reports


def check_numerical_contracts(actual, references, c, *, forward=False):
    reports = []
    comparison = load_asset("compare")
    slots = comparison.numerical_slots(c, forward=forward)
    entry = "flash_kda_fwd" if forward else "flash_kda"
    for kind, expected in references.items():
        for report in comparison.compare_contracts(actual, expected, kind, slots=slots):
            event(c, status="numerical_contract_check", entry=entry, **report)
            reports.append(report)
    return reports


def assert_golden(inputs, c, out):
    actual = load_asset("compare").pack(out, c)
    references = {
        kind: reference(inputs, c, kind) for kind in ("cpu_chunk", "npu_chunk")
    }
    reports = compare_l2(actual, references["cpu_chunk"], references["npu_chunk"])
    for report in reports:
        event(c, status="l2_reference_check", **report)
    reports.extend(check_numerical_contracts(actual, references, c))
    failures = [report["error_info"] for report in reports if not report["pass"]]
    if failures:
        raise AssertionError("Numerical checks failed:\n" + "\n".join(failures))
    return actual


def negative(inputs, c):
    bc = c
    # Establish that the exact baseline is executable before judging any rejection.
    good, md = invoke(inputs, bc)
    kw, objects, reasons = load_asset("inputs").mutate(call_kwargs(inputs, bc, md), c)
    snapshot = [x.clone() for x in inputs]
    md_snapshot = md.clone()
    mutation = c["test"]["mutation"]
    missing_argument = mutation[0] == "missing" and mutation[2] == "omit"
    with mode(c), guard(static=True):
        reject_call(
            lambda: rejection_target(kw, c),
            objects,
            reasons,
            c,
            expected_error=TypeError if missing_argument else ValueError,
        )
        if "omit_metadata_arg" in c["test"]:
            _, producer, _ = dut_modules()
            field = c["test"]["omit_metadata_arg"]
            params = {
                key: c[key]
                for key in ("batch_size", "nheads_qk", "nheads_v", "seqlens", "layout")
            }
            params.update(
                cu_seqlens=input_tensor(inputs, "cu_seqlens"),
                enable_grad=not c["inference"],
            )
            del params[field]
            reject_call(
                lambda: runtime.public_api().flash_kda_metadata(**params),
                [field],
                ["missing", "required"],
                c,
                expected_error=TypeError if field == "seqlens" else ValueError,
            )
    unchanged(snapshot, inputs)
    if not torch.equal(md_snapshot, md):
        raise AssertionError("negative call changed metadata")
    recovered, _ = invoke(inputs, bc, md)
    assert_golden(inputs, bc, recovered)
    event(c, status="recovered", boundary="before readback/allocation/compile/launch")


def check_length_contents(inputs, c):
    test = c["test"]
    with mode(c):
        md = produce(inputs, c)
    index = load_asset("inputs").INPUT_NAMES.index(test["field"])
    bad = list(inputs)
    bad[index] = inputs[index].new_tensor(test["values"])
    kw = call_kwargs(bad, c, md)
    with mode(c), guard(static=False):
        for call in (lambda: produce(bad, c), lambda: dut_call(kw)):
            try:
                call()
            except ValueError as exc:
                if test["message"] not in str(exc):
                    raise AssertionError(f"unexpected rejection: {exc}") from exc
            else:
                raise AssertionError(f"accepted invalid {test['field']}")
    recovered, _ = invoke(inputs, c, md)
    assert_golden(inputs, c, recovered)


@contextmanager
def metadata_case_stream(ref, test):
    """非活动核槽用例使用独立的限核流，其他用例沿用调用流。"""
    if test.get("error") != "inactive":
        yield
        return
    main, _, _ = dut_modules()
    slots = load_asset("inputs").CORE_SLOTS
    if main._device_block_num(ref) < slots:
        yield
        return
    device = ref.device
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        # 留一个空槽，在真实调度记录中验证该槽不能持有任务。
        torch.npu.set_stream_limit(stream, cube_num=slots - 1)
        try:
            yield
        finally:
            try:
                stream.synchronize()
            finally:
                torch.npu.reset_stream_limit(stream)


def metadata_negative(inputs, c):
    bc = c
    main, _, _ = dut_modules()
    cores = main._device_block_num(input_tensor(inputs, "q"))
    h, rs = load_asset("inputs").metadata_baseline(c, cores)
    md = torch.tensor(
        load_asset("inputs").encode_metadata(h, rs),
        dtype=torch.int32,
        device=input_tensor(inputs, "q").device,
    )
    good, _ = invoke(inputs, bc, md)
    assert_golden(inputs, bc, good)
    damaged = torch.tensor(
        load_asset("inputs").mutate_metadata(c, cores),
        dtype=torch.int32,
        device=input_tensor(inputs, "q").device,
    )
    patterns = c["test"]["error_patterns"]
    with mode(c), guard(static=False):
        reject_call(
            lambda: dut_call(call_kwargs(inputs, bc, damaged)),
            patterns,
            patterns,
            c,
        )
    recovered, _ = invoke(inputs, bc, md)
    assert_golden(inputs, bc, recovered)
    event(c, status="recovered", boundary="before workspace/compile/AICore")


def meta_case(inputs, c):
    bc = c
    tensors = [torch.empty_like(x, device="meta") for x in inputs]
    md = tensors[5]
    kw = call_kwargs(tensors, bc, md)
    if c["suite"] == "meta_negative":
        kw, objects, reasons = load_asset("inputs").mutate(kw, c)
    with mode(c), guard(static=True, allocation=False, casts=False):
        if c["test"].get("fake"):
            with FakeTensorMode() as fake:
                fkw = {
                    k: fake.from_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in kw.items()
                }
                reject_call(
                    lambda: rejection_target(fkw, c),
                    objects,
                    reasons,
                    c,
                )
        elif c["suite"] == "meta_negative":
            reject_call(lambda: rejection_target(kw, c), objects, reasons, c)
        else:
            out = dut_call(kw)
            load_asset("compare").check(out, tensors, bc, meta=True)
    if c["suite"] == "meta_negative":
        with mode(c), guard(static=True, allocation=False, casts=False):
            recovered = dut_call(call_kwargs(tensors, bc, md))
            load_asset("compare").check(recovered, tensors, bc, meta=True)
    event(c, status="static_checked", boundary="no readback/compile/launch")


def accuracy(inputs, c):
    # invoke checks the exact return protocol and input immutability for each call.
    out, md = invoke(inputs, c)
    forward, _ = invoke(inputs, c, md, forward=True)
    comparison = load_asset("compare")
    values = comparison.pack(out, c)
    forward_values = comparison.pack(forward, c, forward=True)
    forward_references = {
        kind: reference(inputs, c, kind, forward=True)
        for kind in ("cpu_chunk", "npu_chunk")
    }
    references = {
        kind: expected[: len(comparison.SLOTS)]
        for kind, expected in forward_references.items()
    }

    # Both calls must meet the original official L2 standard independently.
    # Keep per-entry metrics in the evidence; TTK also compares forward_values.
    reports = []
    for entry, actual, expected, full_forward in (
        ("flash_kda", values, references, False),
        ("flash_kda_fwd", forward_values, forward_references, True),
    ):
        entry_reports = compare_l2(
            actual,
            expected["cpu_chunk"],
            expected["npu_chunk"],
            slots=comparison.numerical_slots(c, forward=full_forward),
        )
        for report in entry_reports:
            event(c, status="l2_reference_check", entry=entry, **report)
        reports.extend(entry_reports)
        reports.extend(
            check_numerical_contracts(actual, expected, c, forward=full_forward)
        )

    contracts_pass = all(report["pass"] for report in reports)
    for slot, actual, expected in zip(comparison.SLOTS, values, forward_values):
        passed = actual.dtype == expected.dtype and torch.equal(actual, expected)
        event(
            c,
            status="entry_consistency_checked",
            slot=slot,
            standard="binary_equal",
            **{"pass": passed},
        )
        contracts_pass = contracts_pass and passed
    try:
        check_relations(inputs, c, out, md)
    except AssertionError as exc:
        contracts_pass = False
        event(c, status="relation_failed", message=str(exc))
    # Inference: three forward slots; training: all sixteen forward slots.
    # The final bool requires BOTH entries' precision and all functional checks.
    # Never replace the actual numeric results with only a success flag.
    return forward_values + [
        torch.tensor([contracts_pass], dtype=torch.bool, device=out[0].device)
    ]


def same_main(a, b, ca, cb):
    left = load_asset("compare").pack(a, ca)
    right = load_asset("compare").pack(b, cb)
    for i in (0, 1):
        if left[i].numel() and right[i].numel():
            if not torch.equal(left[i], right[i]):
                raise AssertionError(f"exact relation mismatch at slot {i}")
            event(
                ca,
                status="relation_checked",
                slot=i,
                standard="exact",
            )


def check_relations(inputs, c, out, md):
    test = c["test"].get("relation")
    if not test:
        return
    other = copy.deepcopy(c)
    other.update(test.get("overrides", {}))
    other["kwargs"].update(test.get("operator_overrides", {}))

    def run(cc, xx=None, metadata=None, compare=True):
        if xx is None:
            xx = [x.to(input_tensor(inputs, "q").device) for x in make(cc)]
        elif metadata is None:
            with mode(cc):
                metadata = produce(xx, cc)
        result, _ = invoke(xx, cc, metadata)
        assert_golden(xx, cc, result)
        if compare and test.get("exact", False):
            same_main(out, result, c, cc)
        return result

    kind = test["kind"]
    if kind == "equivalent":
        run(other, inputs if test.get("reuse_inputs") else None)
    elif kind == "reconfigure":
        run(other, compare=test.get("compare_main", False))
    elif kind == "state_handoff":
        sections, last_state, start = [], None, 0
        for length in test["sections"]:
            cc = copy.deepcopy(c)
            cc.update(
                seqlens=length,
                valid_lengths=[length] * c["batch_size"],
                cu_seqlens_value=None,
                seqused_value=None,
            )
            xx = [x.clone() for x in inputs]
            axis = 2 if c["layout"] == "BNSD" else 1
            for i in range(5):
                token_axis = 1 if i in (3, 4) else axis
                xx[i] = xx[i].narrow(token_axis, start, length).contiguous()
            if last_state is not None:
                xx[6] = last_state
                cc["initial_state_mode"] = "random"
            result = run(cc, xx, compare=False)
            sections.append(result[0])
            last_state = result[1]
            start += length
        combined = list(out)
        combined[0], combined[1] = torch.cat(sections, 1), last_state
        assert_golden(inputs, c, combined)
        event(
            c,
            status="relation_checked",
            relation="state_handoff",
            standard="cross_check",
            level="L2",
        )
    elif kind == "recovery":
        bad = dict(c, test=test)
        kw, objects, reasons = load_asset("inputs").mutate(
            call_kwargs(inputs, c, md), bad
        )
        with mode(c), guard(static=True):
            reject_call(lambda: dut_call(kw), objects, reasons, c)
        result = run(other, inputs, md, compare=False)
        same_main(out, result, c, other)
    elif kind == "repeat":
        for _ in range(test["count"] - 1):
            result = run(other, inputs, md, compare=False)
            same_main(out, result, c, other)
        other["seed"] = test["changed_seed"]
        changed = [x.to(input_tensor(inputs, "q").device) for x in make(other)]
        changed[3:] = inputs[3:]
        run(other, changed, md, compare=False)
    else:
        raise ValueError(f"unknown relation check: {kind}")


def check_capacity(inputs, c):
    """Check scalar capacity bounds without allocating tensors of that size."""
    _, producer, _ = dut_modules()
    reached = []

    class StaticPassed(Exception):
        pass

    def stop(*args, **kwargs):
        reached.append(True)
        raise StaticPassed()

    length = c["test"].get("seqlens_by_arch", {}).get(ARCH, c["test"]["seqlens"])

    def produce_capacity():
        return producer.flash_kda_metadata(
            batch_size=1,
            nheads_qk=1,
            nheads_v=1,
            seqlens=length,
            headdim_qk=128,
            headdim_v=128,
            enable_grad=not torch.is_inference_mode_enabled(),
            layout="BNSD",
        )

    with patch.object(producer, "_device_block_num", stop), guard(static=True):
        if c["test"]["valid"]:
            try:
                produce_capacity()
            except StaticPassed:
                pass
            else:
                raise AssertionError(
                    "legal capacity did not reach device-query boundary"
                )
            if not reached:
                raise AssertionError("capacity check did not execute")
        else:
            reject_call(
                produce_capacity,
                ["capacity"],
                ["INT32", "range", "overflow"],
                c,
            )
    produce(inputs, c)


def check_producer_result(inputs, c):
    _, producer, _ = dut_modules()
    target = {}
    original_empty = torch.empty
    real_get = producer._get_aicpu_kernel

    def empty(*args, **kwargs):
        value = original_empty(*args, **kwargs)
        if value.dtype == torch.int32 and value.ndim == 1:
            target["metadata"] = value
        return value

    class CorruptProducer:
        def launch(self, *args, **kwargs):
            real_get().launch(*args, **kwargs)
            target["metadata"][c["test"]["word"]] = c["test"]["value"]

    with (
        patch.object(torch, "empty", empty),
        patch.object(producer, "_get_aicpu_kernel", lambda: CorruptProducer()),
    ):
        reject_call(
            lambda: produce(inputs, c),
            ["metadata", "status"],
            ["status", "invalid", "offset"],
            c,
        )
    produce(inputs, c)


def check_runtime_error(inputs, c, md):
    main, _, _ = dut_modules()
    point = c["test"]["point"]
    injected = RuntimeError(f"injected {point} failure")
    hit = []

    def fail(*args, **kwargs):
        hit.append(point)
        raise injected

    with ExitStack() as stack:
        if point == "device_query":
            stack.enter_context(patch.object(main, "_device_block_num", fail))
        elif point == "allocation":
            stack.enter_context(patch.object(torch, "empty", fail))
        elif point == "compile":
            stack.enter_context(patch.object(main, "_get_compiled_kernel", fail))
        elif point == "launch":
            stack.enter_context(
                patch.object(main, "_get_compiled_kernel", lambda *args, **kwargs: fail)
            )
        elif point == "readback":
            cpu = torch.Tensor.cpu
            ptr = md.data_ptr()

            def readback(x, *args, **kwargs):
                if x.data_ptr() == ptr:
                    fail()
                return cpu(x, *args, **kwargs)

            stack.enter_context(patch.object(torch.Tensor, "cpu", readback))
        else:
            raise ValueError(f"unknown runtime failure point: {point}")
        with mode(c):
            try:
                dut_call(call_kwargs(inputs, c, md))
            except RuntimeError as exc:
                if exc is not injected:
                    raise AssertionError(
                        "underlying RuntimeError was replaced"
                    ) from exc
            else:
                raise AssertionError("injected failure not propagated")
    if not hit:
        raise AssertionError("requested failure hook was not reached")
    event(c, status="propagated", point=point, message=str(injected))


def check_buffer(inputs, c, md):
    main, _, _ = dut_modules()
    validate = main.validate_buffer
    get_compiled = main._get_compiled_kernel
    test = c["test"]
    kernel = main.FlashKDAFwd.run_flash_kda
    signature = inspect.signature(kernel)
    hit, launched, replacement = [], [], {}
    snapshot = [x.clone() for x in inputs]
    md_snapshot = md.clone()

    def intercept(name, value, *args, **kwargs):
        if name != test["buffer"]:
            return validate(name, value, *args, **kwargs)
        hit.append(name)
        change = test["change"]
        if change == "missing":
            value = None
        elif change == "shape":
            value = torch.zeros(test["value"], dtype=value.dtype, device=value.device)
        elif change == "dtype":
            value = value.to(getattr(torch, test["value"]))
        elif change == "stride":
            # The caller checks strides after validate_buffer returns. Mutate the
            # actual Tensor so those checks see the injected layout as well.
            value.set_(load_asset("inputs").noncontiguous(value))
        elif change == "legal_view":
            offset = test["storage_offset"]
            storage = torch.empty(
                value.numel() + offset, dtype=value.dtype, device=value.device
            )
            replacement["original"] = value.detach()
            view = storage.as_strided(value.shape, value.stride(), offset)
            view.copy_(value)
            value.set_(view)
            replacement["view"] = value
            replacement["storage"] = value.untyped_storage().data_ptr()
            replacement["stride"] = value.stride()
        elif change == "capacity":
            replacement["storage"] = value.untyped_storage().data_ptr()
        else:
            raise ValueError(f"unknown buffer mutation: {change}")
        event(
            c,
            status="buffer_injected",
            buffer=name,
            change=change,
            hook="validate_buffer",
            whitebox=True,
        )
        return validate(name, value, *args, **kwargs)

    def observe_compile(*args, **kwargs):
        kernel = get_compiled(*args, **kwargs)

        def launch(*args, **kwargs):
            # Inspect the actual kernel arguments by name; never fix ABI indexes.
            bound = signature.bind(None, *args, **kwargs)
            value = bound.arguments[test["argument"]]
            if value is not replacement.get("view"):
                raise AssertionError("injected buffer was not passed to the kernel")
            if value.storage_offset() != test["storage_offset"]:
                raise AssertionError("injected storage_offset changed before launch")
            if (
                value.untyped_storage().data_ptr() != replacement["storage"]
                or value.stride() != replacement["stride"]
            ):
                raise AssertionError("injected storage or stride changed before launch")
            result = kernel(*args, **kwargs)
            launched.append(test["buffer"])
            event(
                c,
                status="buffer_launched",
                buffer=test["buffer"],
                shape=list(value.shape),
                stride=list(value.stride()),
                storage_offset=value.storage_offset(),
                whitebox=True,
            )
            # Public output slices still refer to the original allocation.
            replacement["original"].copy_(value)
            value.set_(replacement["original"])
            return result

        return launch

    with ExitStack() as stack:
        stack.enter_context(patch.object(main, "validate_buffer", intercept))
        if test["change"] == "legal_view":
            stack.enter_context(
                patch.object(main, "_get_compiled_kernel", observe_compile)
            )
        else:
            stack.enter_context(guard(static=False, allocation=False))
        if test["change"] == "capacity":
            nbytes = torch.UntypedStorage.nbytes

            def capacity(storage):
                if storage.data_ptr() == replacement.get("storage"):
                    return test["nbytes"]
                return nbytes(storage)

            stack.enter_context(patch.object(torch.UntypedStorage, "nbytes", capacity))
        with mode(c):
            if test["change"] == "legal_view":
                out = dut_call(call_kwargs(inputs, c, md))
                load_asset("compare").check(out, inputs, c)
                assert_golden(inputs, c, out)
            else:
                reject_call(
                    lambda: dut_call(call_kwargs(inputs, c, md)),
                    test["error_objects"],
                    test["error_patterns"],
                    c,
                )
    if len(hit) != 1:
        raise AssertionError(
            f"buffer validation injection count expected 1, got {len(hit)}"
        )
    if test["change"] == "legal_view" and len(launched) != 1:
        raise AssertionError("legal buffer view did not execute exactly once")
    unchanged(snapshot, inputs)
    if not torch.equal(md_snapshot, md):
        raise AssertionError("buffer injection changed metadata")
    event(
        c,
        status="injected",
        hook="validate_buffer",
        buffer=test["buffer"],
        boundary="real kernel" if launched else "before compile/launch",
        whitebox=True,
    )


def check_output_stride(inputs, c):
    with mode(c):
        metadata = input_tensor(inputs, "metadata")
        kwargs = call_kwargs(inputs, c, metadata)
        actual = dut_call(kwargs)[0]
        meta_kwargs = {
            name: value.to("meta") if isinstance(value, torch.Tensor) else value
            for name, value in kwargs.items()
        }
        meta = dut_call(meta_kwargs)[0]
        expected = tuple(c["test"]["out_stride"])
        event(
            c,
            status="out_stride",
            expected=expected,
            npu=actual.stride(),
            meta=meta.stride(),
        )
        v = input_tensor(inputs, "v")
        expected_shape = v.transpose(1, 2).shape if c["layout"] == "BNSD" else v.shape
        for label, out in (("NPU", actual), ("Meta", meta)):
            if (
                out.shape != expected_shape
                or out.dtype != input_tensor(inputs, "v").dtype
            ):
                raise AssertionError(
                    f"{label} out shape/dtype: {out.shape}/{out.dtype}"
                )
            if out.stride() != expected:
                raise AssertionError(
                    f"{label} out stride {out.stride()}, expected {expected}"
                )


def execute_whitebox(inputs, c):
    kind = c["test"]["kind"]
    if kind == "out_stride":
        check_output_stride(inputs, c)
        return
    if kind == "sizes":
        _, _, validation = dut_modules()
        if c["test"]["valid"]:
            validation.validate_sizes(*c["test"]["sizes"])
        else:
            reject_call(
                lambda: validation.validate_sizes(*c["test"]["sizes"]),
                ["head", "Nv"],
                ["96", "head"],
                c,
            )
        validation.validate_sizes(*c["test"]["recovery_sizes"])
        event(c, status="pure_spec_checked", no_device_precision_claim=True)
        return
    if kind == "length_contents":
        check_length_contents(inputs, c)
        return
    if kind == "capacity":
        check_capacity(inputs, c)
        return
    if kind == "producer_result":
        check_producer_result(inputs, c)
        return
    _, md = invoke(inputs, c)
    if kind == "runtime_error":
        check_runtime_error(inputs, c, md)
    elif kind == "buffer":
        check_buffer(inputs, c, md)
    else:
        raise ValueError(f"unknown whitebox check: {kind}")
    recovered, _ = invoke(inputs, c, md)
    assert_golden(inputs, c, recovered)
    event(c, status="recovered")


_CACHE = {}


def execute_performance(inputs, c):
    return load_asset("performance").execute(sys.modules[__name__], inputs, c, _CACHE)


def flash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = 128 ** (-0.5),
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
    cu_seqlens_cpu: Optional[List[int]] = None,
    # Test attributes below are consumed here, never forwarded to the DUT.
    inference: bool = False,
    no_grad: bool = False,
    valid_lengths: Optional[List[int]] = None,
    initial_state_mode: str = "none",
    dt_bias_mode: str = "none",
    cu_seqlens_value: Optional[List[int]] = None,
    seqused_value: Optional[List[int]] = None,
    seed: int = 101,
    distribution: float = 1.0,
    raw_g: Optional[float] = None,
    tail: Optional[float] = None,
    omitted_attributes: str = "",
    test_config: str = "{}",
    testcase_name: str = "",
):
    attributes = locals()
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
    c = load_asset("inputs").configuration(inputs, attributes)
    if c["test"].get("requires_source") and runtime.selection()[0] != "source":
        raise RuntimeError("whitebox case requires KDA_DUT=source")
    inputs[5] = native_metadata(inputs[5], c)
    event(
        c,
        status="selected",
        execution="eager",
        adapter_path=__file__,
        assets_path=str(Path(__file__).resolve().parent / "assets"),
        device=str(q.device),
        device_name=torch.npu.get_device_name(q.device)
        if q.device.type == "npu"
        else q.device.type,
        **runtime.describe(),
    )
    if c["suite"] != "performance":
        hashes = {
            name: hashlib.sha256(
                value.detach().cpu().contiguous().view(torch.uint8).numpy()
            ).hexdigest()
            for name, value in zip(load_asset("inputs").INPUT_NAMES, inputs)
        }
        event(
            c,
            status="input_fingerprints",
            sha256=hashes,
            inference=c["inference"],
            kwargs=c["kwargs"],
        )
    if c["pending"]:
        raise RuntimeError("PENDING_EXPECTATION: " + c["pending"])
    if c["suite"] in ("accuracy", "performance") and q.device.type == "cpu":
        raise RuntimeError(
            "NPU DUT execution required; CPU is for input/golden preparation only"
        )
    try:
        if c["suite"] == "accuracy":
            return accuracy(inputs, c)
        if c["suite"] == "negative":
            negative(inputs, c)
        elif c["suite"] == "metadata":
            with metadata_case_stream(q, c["test"]):
                metadata_negative(inputs, c)
        elif c["suite"] in ("meta", "meta_negative"):
            meta_case(inputs, c)
        elif c["suite"] == "whitebox":
            execute_whitebox(inputs, c)
        elif c["suite"] == "performance":
            return execute_performance(inputs, c)
        else:
            raise ValueError(f"unknown test kind: {c['suite']}")
        # Success is reached only after the contract checks and recovery execute.
        return [torch.ones(1, dtype=torch.bool, device=q.device)]
    except Exception as exc:
        event(
            c,
            status="failed_or_unavailable",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        raise


def prepare(row):
    """Prepare benchmark inputs using the TTK input recipe."""
    shapes = ast.literal_eval(row["tensor_view_shapes"])
    dtypes = ast.literal_eval(row["tensor_dtypes"])
    specs = [
        torch.empty(s, dtype=getattr(torch, d), device="meta")
        for s, d in zip(shapes, dtypes)
    ]
    config = load_asset("inputs").configuration(specs, attributes(row))
    ranges = ast.literal_eval(row.get("input_data_ranges") or "()")
    config["input_ranges"] = ranges
    values = make(config)
    for name, value, shape, dtype in zip(
        load_asset("inputs").INPUT_NAMES, values, shapes, dtypes
    ):
        if tuple(value.shape) != tuple(shape) or value.dtype != getattr(torch, dtype):
            raise ValueError(f"{name}: input generator differs from shared CSV")
    return config, values
