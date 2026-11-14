# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime compatibility checks and explicit stream quotas for PA/TH TTK."""

import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

import torch

__all__ = []

_QUOTA_POLICY = {
    "arch35": ("dav-3510", 2, (8, 16, 24, 28, 32)),
    "arch92": ("dav-9201", 1, (8, 16, 24, 32, 64)),
}


def _assert_environment(require_npu: bool) -> None:
    """Check NPU availability and record the installed runtime versions."""
    if require_npu:
        import torch_npu
        import cannbotdsl

        if not torch.npu.is_available():
            raise RuntimeError(
                "torch.npu.is_available() is False; NPU backend required"
            )
        logging.info(
            "[mkda-ttk] env python=%s torch=%s torch_npu=%s cannbotdsl=%s npu_count=%d",
            sys.version.split()[0],
            torch.__version__,
            torch_npu.__version__,
            cannbotdsl.__version__,
            torch.npu.device_count(),
        )
    else:
        logging.info(
            "[mkda-ttk] CPU chain path: python=%s torch=%s",
            sys.version.split()[0],
            torch.__version__,
        )


def _available_core_quotas():
    """Quotas supported by both the physical device and the current stream."""
    import cannbotdsl  # noqa: PLC0415

    device_id = torch.npu.current_device()
    hardware = torch.npu.get_device_properties(device_id)
    stream = torch.npu.current_stream(device_id)
    platform = cannbotdsl.get_platform_info(stream=stream)
    arch, ratio, quotas = _QUOTA_POLICY[fixtures.selected_architecture()]
    if platform.npu_arch != arch:
        raise RuntimeError(
            f"TTK architecture {arch} does not match device {platform.npu_arch}"
        )
    cube = min(int(hardware.cube_core_num), int(platform.cube_core_num))
    vector = min(int(hardware.vector_core_num), int(platform.vector_core_num))
    return tuple(core for core in quotas if core <= cube and ratio * core <= vector)


def _call_with_quota(sample, tensors, scalars, cores: int, *, guard_case=False):
    """Invoke the sample inside a stream whose AIC/AIV quota matches the pytest semantics.

    ``cores == 0`` means "max available" (pytest ``max``/``auto``); every invocation
    sets the stream quota explicitly, exactly like
    ``test_precision._check_precision`` does.
    """
    _, ratio, _ = _QUOTA_POLICY[fixtures.selected_architecture()]
    available = _available_core_quotas()
    if not available:
        raise RuntimeError(
            f"device/stream exposes no supported AIC quota (needs >= 8 AIC and {8 * ratio} AIV)"
        )
    quota = available[-1] if not cores else int(cores)
    if guard_case:
        hardware = torch.npu.get_device_properties(torch.npu.current_device())
        if quota > hardware.cube_core_num or ratio * quota > hardware.vector_core_num:
            raise RuntimeError(
                "Requested negative stream quota unavailable on physical device"
            )
    elif quota not in available:
        raise ValueError(
            f"requested cores={quota} is not in the device/stream supported set {available}; "
            "regenerate cases with cores=0 for max/auto or use a supported quota"
        )

    stream = torch.npu.Stream()
    torch.npu.synchronize()
    torch.npu.set_stream_limit(stream, cube_num=quota, vector_num=ratio * quota)
    try:
        with torch.npu.stream(stream):
            out = sample(*tensors, *scalars)
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        torch.npu.reset_stream_limit(stream)
    return out


# Real-input adapters and strict TTK specification.
"""TTK adapter builds real PA tensors deterministically and records their hashes.

The scalar marker only selects TTK's device. Actual inputs are generated from
the same frozen case on NPU/golden/peer; references never consume captured NPU
values. Direct inplace/alias contracts are additionally covered by pytest.
"""
if __package__:
    from .assets import inputs as cases
    from .assets import inputs as fixtures
    from .assets import golden
else:
    from assets import inputs as cases
    from assets import inputs as fixtures
    from assets import golden


def _case_configuration(case_id, case_config):
    config = json.loads(case_config) if case_config is not None else cases.get(case_id)
    arch = config.get("architecture", "arch35")
    if arch != fixtures.selected_architecture():
        raise ValueError(f"Use KDA_ARCH={arch} for case {config['case_id']}")
    return config


def prepare(case_id, case_config=None):
    c = _case_configuration(case_id, case_config)
    a, _, _ = fixtures.make(c)
    return a, c


def save(kind, case_id, value):
    if not os.environ.get("MKDA_RUN_DIR"):
        return
    root = Path(os.environ["MKDA_RUN_DIR"]).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("Keep validation evidence outside repository")
    dest = root / kind
    dest.mkdir(parents=True, exist_ok=True)
    (dest / (case_id + ".json")).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8"
    )


def _span(t):
    if t.ndim and t.shape[0] and t.numel():
        length = 1 + sum(
            (extent - 1) * stride for extent, stride in zip(t.shape, t.stride())
        )
        return t.as_strided((length,), (1,))
    return t.flatten()


def _pool_storage(t):
    import torch_npu

    return _span(t) if torch_npu.get_npu_format(t) in (0, 2) else t


def _snapshot(a):
    pools = tuple(k for k in fixtures.PA_POOLS if k in a)
    return {
        k: (_pool_storage(v) if k in pools else v).detach().cpu().clone()
        for k, v in a.items()
        if isinstance(v, torch.Tensor) and (k in pools)
    }


def _unchanged(actual, before):
    # Compare storage bits: unchanged NaN payloads and signed zero must also survive.
    assert fixtures.digest(actual) == fixtures.digest(before), (
        "Unexpected write outside the designed output region"
    )


def _pool_untouched(t, before, changed):
    after = _span(t).detach().cpu()
    stride = t.stride(0)
    payload = t[0].numel()
    for slot in range(t.shape[0]):
        start = slot * stride
        rows = changed.get(slot, 0)
        offset = rows * t[0, 0].numel() if rows else 0
        _unchanged(
            after[start + offset : start + payload],
            before[start + offset : start + payload],
        )
        if slot < t.shape[0] - 1:
            _unchanged(
                after[start + payload : start + stride],
                before[start + payload : start + stride],
            )


def _check_mega(a, c, before, result, module):
    assert (
        len(result) == 5
        and result[0].shape == a["hidden_states"].shape
        and result[0].dtype == torch.bfloat16
    )
    for actual, key in zip(result[1:], ("conv_state", *fixtures.REPLAY_POOLS)):
        assert actual is a[key], "InplaceOutput must return the original input Tensor"
    _unchanged(_span(a["checkpoint_state"]), before["checkpoint_state"])
    lengths = c["lengths"]
    slots = c["state_indices"]
    selected = dict(zip(slots, lengths))
    _pool_untouched(
        a["conv_state"],
        before["conv_state"],
        {slot: length + 2 for slot, length in selected.items()},
    )
    for key in fixtures.REPLAY_POOLS:
        tensor = a[key]
        after = _span(tensor).detach().cpu()
        current = after.as_strided(tensor.shape, tensor.stride())
        original = before[key].as_strided(tensor.shape, tensor.stride())
        stride = tensor.stride(0)
        payload = tensor[0].numel()
        for slot in range(tensor.shape[0]):
            length = selected.get(slot, 0)
            if length:
                assert torch.isfinite(current[slot, :, :length]).all(), (
                    key + " produced NaN/Inf"
                )
            _unchanged(current[slot, :, length:], original[slot, :, length:])
            if slot < tensor.shape[0] - 1:
                _unchanged(
                    after[slot * stride + payload : (slot + 1) * stride],
                    before[key][slot * stride + payload : (slot + 1) * stride],
                )
    assert torch.isfinite(result[0]).all(), "out produced NaN/Inf"
    tp = 96 // c["N"]
    sequence = lengths[0]
    optimized = (
        fixtures.selected_architecture() == "arch35"
        and sequence in (4, 8)
        and all(n == sequence for n in lengths)
        and c["B"] % tp == 0
        and 1 <= c["B"] // tp <= 32
    )
    assert module._last_backend == (
        f"optimized_s{sequence}" if optimized else "variable_length"
    ), "Unexpected execution branch"


def _device_mega(a, c):
    _assert_environment(True)
    module = fixtures.load_operator()
    device = fixtures.to_npu(a)
    save(
        "input_manifests",
        c["case_id"],
        dict(
            case_id=c["case_id"],
            configuration=c,
            inputs=fixtures.manifest(a),
            actual_inputs=fixtures.manifest(device),
        ),
    )
    before = _snapshot(device)
    result = _call_with_quota(
        lambda: module.mega_recurrent_kda(**device), (), (), c["cores"]
    )
    _check_mega(device, c, before, result, module)
    save(
        "invariant_rows",
        c["case_id"],
        dict(
            status="PASS",
            checks=[
                "shape_dtype",
                "input_aliases",
                "checkpoint_readonly",
                "pool_gaps_unselected_slots",
                "conv_tail",
                "record_tails",
                "finite_outputs",
                "execution_branch",
            ],
        ),
    )
    return result, device


def verify(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type in ("npu", "privateuseone"), (
        "Verify requires an NPU marker"
    )
    arguments, config = prepare(case_id)
    return _device_mega(arguments, config)[0]


def verify_hp(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", "FP64 golden requires a CPU marker"
    arguments, _ = prepare(case_id)
    return golden.run_fp64(**arguments)


def verify_peer(marker: torch.Tensor, *, case_id: str):
    assert marker.device.type == "cpu", (
        "Original third-party golden must execute on CPU"
    )
    a, c = prepare(case_id)
    before = fixtures.manifest(a)
    result = golden.run_original(**a)
    assert all(t.device.type == "cpu" for t in result)
    after = fixtures.manifest(a)
    assert after == before, "Original golden adapter modified caller inputs"
    save(
        "peer_manifests",
        case_id,
        dict(
            case_id=case_id,
            configuration=c,
            reference="specified_original_cpu_golden",
            inputs=before,
            actual_inputs=after,
            output_devices=[str(t.device) for t in result],
        ),
    )
    return result


def expected(marker: torch.Tensor, *, case_id: str = None, case_config: str = None):
    return torch.ones(3, dtype=torch.float32, device="cpu")


def reject(marker: torch.Tensor, *, case_id: str = None, case_config: str = None):
    if marker.device.type not in ("npu", "privateuseone"):
        return expected(marker, case_id=case_id)
    _assert_environment(True)
    c = _case_configuration(case_id, case_config)
    case_id = c["case_id"]
    assert c["expected"] == "REJECT_ATOMIC"
    hardware = torch.npu.get_device_properties(torch.npu.current_device())
    _, ratio, _ = _QUOTA_POLICY[fixtures.selected_architecture()]
    if (
        c["cores"] > hardware.cube_core_num
        or ratio * c["cores"] > hardware.vector_core_num
    ):
        raise RuntimeError(
            "Required designed stream quota is unavailable; environment blocked, not an operator contract result"
        )
    a, _ = prepare(case_id, case_config)
    device = fixtures.to_npu(a)
    fixtures.mutate(device, c["mutation"], c.get("mutation_args"))
    save(
        "input_manifests",
        case_id,
        dict(
            case_id=case_id,
            configuration=c,
            actual_inputs=fixtures.manifest(device),
            check_type="CONTRACT_NOT_NUMERICAL",
        ),
    )
    watched = _snapshot(device)
    caught = None
    trace = None
    target = fixtures.load_operator().mega_recurrent_kda

    def invoke():
        return target(**device)

    try:
        _call_with_quota(
            invoke,
            (),
            (),
            c["cores"],
            guard_case=c.get("validation_layer") == "STREAM_CORE_GUARD",
        )
    except Exception as error:
        caught = error
        trace = traceback.format_exc()
    atomic = True
    for key, value in watched.items():
        try:
            _unchanged(_pool_storage(device[key]), value)
        except AssertionError:
            atomic = False
    flags = [
        caught is not None and type(caught).__name__ in c["exception_types"],
        caught is not None
        and all(re.search(p, str(caught)) is not None for p in c["match_all"]),
        atomic,
    ]
    record = dict(
        case_id=case_id,
        configuration=c,
        check_type="CONTRACT_NOT_NUMERICAL",
        actual_exception=type(caught).__name__ if caught else None,
        actual_message=str(caught),
        flags=flags,
        status="PASS" if all(flags) else "FAIL",
        traceback=trace,
    )
    save("contract_rows", case_id, record)
    assert all(flags), json.dumps(record, ensure_ascii=False)
    return torch.tensor(flags, dtype=torch.float32, device=marker.device)


def capture(marker: torch.Tensor, *, case_id: str = None, case_config: str = None):
    if marker.device.type not in ("npu", "privateuseone"):
        return expected(marker, case_id=case_id)
    _assert_environment(True)
    c = _case_configuration(case_id, case_config)
    case_id = c["case_id"]
    cpu, _ = prepare(case_id, case_config)
    device = fixtures.to_npu(cpu)
    target = fixtures.load_operator().mega_recurrent_kda
    writable = ("conv_state", *fixtures.REPLAY_POOLS)

    def check():
        spans = {name: _span(device[name]) for name in writable}
        baseline = {name: span.clone() for name, span in spans.items()}
        eager = target(**device)
        eager_values = [t.clone() for t in eager]
        device["state_indices"].zero_()
        try:
            target(**device)
        except ValueError as error:
            assert "distinct" in str(error)
        else:
            raise AssertionError("Duplicate indices must be rejected in eager mode")
        stream = torch.npu.current_stream()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            assert torch.npu.is_current_stream_capturing()
            captured = target(**device)
        device["state_indices"].copy_(cpu["state_indices"])
        for name, span in spans.items():
            span.copy_(baseline[name])
        graph.replay()
        stream.synchronize()
        for actual, reference in zip(captured, eager_values, strict=True):
            torch.testing.assert_close(
                actual, reference, rtol=0, atol=0, equal_nan=True
            )
        graph.reset()

    _call_with_quota(check, (), (), c["cores"])
    save(
        "capture_rows",
        case_id,
        dict(
            status="PASS",
            checks=[
                "duplicate_eager_rejected",
                "duplicate_placeholder_captured",
                "legal_indices_replayed",
            ],
        ),
    )
    return expected(marker, case_id=case_id).to(marker.device)


def run_pytest_case(case_id, reference_verify):
    c = cases.get(case_id)
    marker = torch.empty(1, device="npu")
    if c["expected"] == "REJECT_ATOMIC":
        return reject(marker, case_id=case_id)
    if c["expected"] == "PASS_CAPTURE":
        return capture(marker, case_id=case_id)
    a, _ = prepare(case_id)
    wanted = reference_verify(**a)
    actual = verify(marker, case_id=case_id)
    for i, (value, reference) in enumerate(zip(actual, wanted, strict=True)):
        atol = (
            5e-5
            if i == 0 and all(n == 8 for n in c["lengths"]) and c["B"] == 96 // c["N"]
            else 0.005
        )
        torch.testing.assert_close(
            value.cpu(), reference, atol=atol, rtol=0.005, equal_nan=True
        )
    save(
        "pytest_rows", case_id, dict(status="PASS", framework="pytest", configuration=c)
    )


"""Unchanged strict cross_check/L2, including FP32 U under the new ABI."""
__spec__ = {
    "kda_ttk_ops.verify": "VerifySpec",
    "kda_ttk_ops.reject": "RejectSpec",
    "kda_ttk_ops.capture": "RejectSpec",
    "kda_ttk_ops.verify_csv": "CsvVerifySpec",
}


class Base:
    tolerance = {
        dtype: {"standard": "cross_check", "level": "L2"}
        for dtype in ("bfloat16", "float16", "float32")
    }

    @staticmethod
    def describe(params):
        cases.get(params["case_id"])
        return dict(shapes=((1,),), dtypes=("float32",))


class VerifySpec(Base):
    golden = staticmethod(verify_hp)
    third_party = {"torch": "kda_ttk_ops.verify_peer"}


class RejectSpec:
    tolerance = Base.tolerance
    golden = staticmethod(expected)
    third_party = {"torch": "kda_ttk_ops.expected"}


# Observe an invalid internal-format tensor without changing the operator input.
def _snapshot(a):
    import torch_npu

    pools = tuple(k for k in fixtures.PA_POOLS if k in a)
    watched = {}
    for key, value in a.items():
        if not isinstance(value, torch.Tensor) or key not in pools:
            continue
        observed = _pool_storage(value) if key in pools else value
        if observed.device.type in ("npu", "privateuseone"):
            fmt = torch_npu.get_npu_format(observed)
            if fmt not in (0, 2):
                observed = torch_npu.npu_format_cast(observed, 2)
                assert torch_npu.get_npu_format(value) == fmt, (
                    "Observer changed the input storage format"
                )
        watched[key] = observed.detach().cpu().clone()
    return watched


# Preserve the historical transpose view and its PA stride; do not materialize it.
_original_mutate_for_views = fixtures.mutate


def _historical_view_mutation(args, kind, details=None):
    if kind == "generic" and details and details.get("op") == "transpose":
        key = details["target"]
        args[key] = args[key].transpose(*details["dims"])
        return args
    return _original_mutate_for_views(args, kind, details)


fixtures.mutate = _historical_view_mutation

# Compatibility only: build the same negative inputs through supported ND operations.
_portable_mutation_before = fixtures.mutate


def _portable_negative_mutation(args, kind, details=None):
    import torch_npu

    d = details or {}
    key = d.get("target")
    t = args.get(key)
    op = d.get("op")
    if (
        kind == "generic"
        and isinstance(t, torch.Tensor)
        and t.device.type in ("npu", "privateuseone")
    ):
        fmt = int(torch_npu.get_npu_format(t))
        if op in ("dtype", "noncontiguous") and fmt == 29:
            nd = torch_npu.npu_format_cast(t, 2)
            if op == "dtype":
                result = torch_npu.npu_format_cast(
                    nd.to(getattr(torch, d["dtype"])), fmt
                )
                assert (
                    result.dtype == getattr(torch, d["dtype"])
                    and result.shape == t.shape
                    and int(torch_npu.get_npu_format(result)) == fmt
                )
                assert fixtures.digest(result) == fixtures.digest(
                    nd.cpu().to(getattr(torch, d["dtype"]))
                )
            else:
                shape = list(t.shape)
                shape[-1] *= 2
                base = torch.empty(shape, dtype=t.dtype, device=t.device)
                result = base[..., ::2]
                result.copy_(nd)
                assert (
                    result.shape == t.shape
                    and result.dtype == t.dtype
                    and not result.is_contiguous()
                    and fixtures.digest(result) == fixtures.digest(t)
                )
            args[key] = result
            save(
                "mutation_compat_rows",
                os.environ.get("MKDA_ACTIVE_CASE", "unknown"),
                dict(
                    op=op,
                    target=key,
                    source_format=fmt,
                    result_format=int(torch_npu.get_npu_format(result)),
                    shape=list(result.shape),
                    dtype=str(result.dtype),
                    stride=list(result.stride()),
                ),
            )
            return args
        if op == "format_nz":
            nd = torch_npu.npu_format_cast(t, 2)
            result = torch_npu.npu_format_cast(nd, 29)
            assert (
                result.dtype == t.dtype
                and result.shape == t.shape
                and int(torch_npu.get_npu_format(result)) == 29
                and fixtures.digest(result) == fixtures.digest(t)
            )
            args[key] = result
            save(
                "mutation_compat_rows",
                os.environ.get("MKDA_ACTIVE_CASE", "unknown"),
                dict(
                    op=op,
                    target=key,
                    source_format=fmt,
                    result_format=29,
                    shape=list(result.shape),
                    dtype=str(result.dtype),
                ),
            )
            return args
    return _portable_mutation_before(args, kind, details)


fixtures.mutate = _portable_negative_mutation

_portable_transfer_before = fixtures.to_npu


def _portable_empty_transfer(args):
    empty = {
        k: v
        for k, v in args.items()
        if k in fixtures.PA_POOLS and isinstance(v, torch.Tensor) and v.numel() == 0
    }
    if not empty:
        return _portable_transfer_before(args)
    result = _portable_transfer_before(
        {k: v for k, v in args.items() if k not in empty}
    )
    for k, v in empty.items():
        result[k] = torch.empty_strided(
            v.shape, v.stride(), dtype=v.dtype, device="npu"
        )
        assert (
            result[k].shape == v.shape
            and result[k].stride() == v.stride()
            and result[k].numel() == 0
        )
    return result


fixtures.to_npu = _portable_empty_transfer

_negative_before_observation = reject


def reject(marker: torch.Tensor, *, case_id: str = None, case_config: str = None):
    config = _case_configuration(case_id, case_config)
    os.environ["MKDA_ACTIVE_CASE"] = config["case_id"]
    return _negative_before_observation(
        marker, case_id=case_id, case_config=case_config
    )


# Rank-four negative-format views use a real two-dimensional NZ allocation.
_rank4_digest_before = fixtures.digest


def _read_rank4_nz(t):
    import torch_npu

    assert t.ndim == 4 and int(torch_npu.get_npu_format(t)) == 29
    return torch_npu.npu_format_cast(t.view(-1, t.shape[-1]), 2).cpu().reshape(t.shape)


def _rank4_safe_digest(t):
    import torch_npu

    if (
        t.device.type in ("npu", "privateuseone")
        and t.ndim == 4
        and int(torch_npu.get_npu_format(t)) == 29
    ):
        return _rank4_digest_before(_read_rank4_nz(t))
    return _rank4_digest_before(t)


fixtures.digest = _rank4_safe_digest

_rank4_snapshot_before = _snapshot


def _snapshot(args):
    import torch_npu

    result = {}
    wanted = fixtures.PA_POOLS
    for key, t in args.items():
        if key not in wanted or not isinstance(t, torch.Tensor):
            continue
        if (
            t.ndim == 4
            and t.device.type in ("npu", "privateuseone")
            and int(torch_npu.get_npu_format(t)) == 29
        ):
            result[key] = _read_rank4_nz(t).clone()
        else:
            result.update(_rank4_snapshot_before({key: t}))
    return result


_rank4_mutation_before = fixtures.mutate


def _rank4_negative_format(args, kind, details=None):
    import torch_npu

    d = details or {}
    key = d.get("target")
    t = args.get(key)
    if (
        kind == "generic"
        and d.get("op") == "format_nz"
        and isinstance(t, torch.Tensor)
        and t.ndim == 4
    ):
        assert t.device.type in ("npu", "privateuseone")
        expected = fixtures.digest(t)
        matrix = t.reshape(-1, t.shape[-1])
        result = torch_npu.npu_format_cast(matrix, 29).view(t.shape)
        assert (
            result.shape == t.shape
            and result.dtype == t.dtype
            and int(torch_npu.get_npu_format(result)) == 29
        )
        assert fixtures.digest(result) == expected
        args[key] = result
        save(
            "mutation_compat_rows",
            os.environ["MKDA_ACTIVE_CASE"],
            dict(
                op="format_nz",
                target=key,
                result_format=29,
                shape=list(result.shape),
                dtype=str(result.dtype),
                allocation_shape=list(matrix.shape),
                logical_values_preserved=True,
            ),
        )
        return args
    return _rank4_mutation_before(args, kind, details)


fixtures.mutate = _rank4_negative_format


def _csv_arguments(tensors, config):
    args = dict(zip((*fixtures.NAMES, "cu_seqlens"), tensors, strict=True))
    args.update(
        scale=config["scale"],
        lower_bound=config["lower_bound"],
        rms_norm_eps=config["rms_norm_eps"],
    )
    return args


def _customize_csv_inputs(*tensors, case_config, **_):
    """Populate framework-allocated views with the original seed and PA data."""
    config = _case_configuration(None, case_config)
    expected, _, _ = fixtures.make(config)
    for name, target in zip((*fixtures.NAMES, "cu_seqlens"), tensors, strict=True):
        source = expected[name]
        if source is None:
            if target is not None:
                raise ValueError(f"{name}: expected an absent tensor slot")
            continue
        if (
            target is None
            or target.shape != source.shape
            or target.dtype != source.dtype
            or target.stride() != source.stride()
        ):
            raise ValueError(
                f"{name}: CSV tensor declaration disagrees with case {config['case_id']}"
            )
        if name in fixtures.PA_POOLS and not target.is_contiguous():
            _span(target).fill_(43.0)
        target.copy_(source)


def verify_csv(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    case_config: str,
):
    import torch_npu

    config = _case_configuration(None, case_config)
    tensors = (
        hidden_states,
        qkv_projection_weight,
        decay_projection_a_weight,
        decay_projection_b_weight,
        beta_projection_weight,
        output_gate_projection_weight,
        output_norm_weight,
        output_projection_weight,
        conv1d_weight,
        conv_state,
        checkpoint_state,
        replay_u,
        replay_k,
        replay_decay,
        state_indices,
        conv_num_accepted_tokens,
        A_log,
        dt_bias,
        cu_seqlens,
    )
    _assert_environment(True)
    device = _csv_arguments(tensors, config)
    for index in fixtures.WEIGHTS:
        key = fixtures.NAMES[index]
        if torch_npu.get_npu_format(device[key]) != 29:
            raise ValueError(
                f"{key}: expected FRACTAL_NZ from the CSV tensor_formats declaration"
            )
    # Each invocation starts from pristine writable pools, even if TTK repeats it.
    # Report the five returned tensors once; inplace_input_indexes would append duplicates.
    for key in ("conv_state", *fixtures.REPLAY_POOLS):
        tensor = device[key]
        device[key] = _span(tensor).clone().as_strided(tensor.shape, tensor.stride())
    module = fixtures.load_operator()
    before = _snapshot(device)
    result = _call_with_quota(
        lambda: module.mega_recurrent_kda(**device), (), (), config["cores"]
    )
    _check_mega(device, config, before, result, module)
    return result


def verify_csv_hp(*tensors, case_config, **_):
    config = _case_configuration(None, case_config)
    return golden.run_fp64(**_csv_arguments(tensors, config))


def verify_csv_peer(*tensors, case_config):
    config = _case_configuration(None, case_config)
    args = _csv_arguments(tensors, config)
    before = fixtures.manifest(args)
    outputs = golden.run_original(**args)
    assert all(t.device.type == "cpu" for t in outputs)
    assert fixtures.manifest(args) == before
    return outputs


class CsvVerifySpec:
    tolerance = Base.tolerance
    golden = staticmethod(verify_csv_hp)
    customize_inputs = staticmethod(_customize_csv_inputs)
    third_party = {"torch": "kda_ttk_ops.verify_csv_peer"}
