# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Architecture and mocked DSL integration checks; no Torch/CANN installation needed."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path


_OPERATOR = Path(__file__).resolve().parents[2]
_MODULES = {
    "flash_kda_grad",
    "flash_kda_grad_metadata",
    "flash_kda_grad_persistent",
    "flash_kda_grad_staged",
    "flash_kda_grad_state",
    "flash_kda_grad_chunk",
    "flash_kda_grad_causal",
    "flash_kda_grad_outputs",
    "flash_kda_grad_recompute",
    "flash_kda_grad_buffer",
    "flash_kda_grad_check",
    "flash_kda_grad_kernel",
}


def _trees():
    return {
        p.stem: ast.parse(p.read_text(encoding="utf-8"))
        for p in (_OPERATOR / "op_dsl").glob("*.py")
    }


class ArchitectureTest(unittest.TestCase):
    def test_module_contract(self):
        self.assertEqual(set(_trees()), _MODULES)

    def test_chunk_calculations_have_separate_owners(self):
        trees = _trees()
        expected = {
            "flash_kda_grad_chunk": {
                "ChunkBackward",
                "ChunkGradientMatmul",
                "WyOperandVector",
                "WyResultVector",
            },
            "flash_kda_grad_causal": {
                "InverseMaskVector",
                "CausalOperandVector",
                "CausalResultVector",
                "DirectCausalDeltaVector",
            },
            "flash_kda_grad_outputs": {
                "ChunkGradientVector",
                "BaseGradientVector",
                "ValueBetaEpilogueVector",
                "BetaCombineVector",
                "SingleChunkGateSensitivity",
            },
        }
        for module, classes in expected.items():
            actual = {n.name for n in trees[module].body if isinstance(n, ast.ClassDef)}
            self.assertEqual(actual, classes)

    def test_host_guards_and_device_entries_have_clear_owners(self):
        trees = _trees()
        checks = {getattr(n, "name", None) for n in trees["flash_kda_grad_check"].body}
        self.assertIn("_validate_input_structure", checks)
        self.assertIn("_validate_saved_cache_contract", checks)
        forbidden = {
            "wait_metadata_ready",
            "metadata_host_info",
            "flash_kda_grad_metadata",
        }
        for n in ast.walk(trees["flash_kda_grad_check"]):
            if isinstance(n, ast.Call):
                name = (
                    n.func.id
                    if isinstance(n.func, ast.Name)
                    else n.func.attr
                    if isinstance(n.func, ast.Attribute)
                    else None
                )
                self.assertNotIn(name, forbidden)
        entries = []
        for module, tree in trees.items():
            for n in tree.body:
                if isinstance(n, ast.ClassDef) and any(
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "kernel"
                    for d in n.decorator_list
                ):
                    entries.append((module, n.name))
        self.assertEqual(entries, [("flash_kda_grad_kernel", "FlashKDAGradKernel")])
        for suffix in ("persistent", "staged"):
            calls = {
                n.func.id
                for n in ast.walk(trees["flash_kda_grad_" + suffix])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            self.assertNotIn("_initialize_backward_gradients", calls)

    def test_causal_consumes_neutral_workspace(self):
        trees = _trees()
        vector = next(
            n
            for n in trees["flash_kda_grad_outputs"].body
            if isinstance(n, ast.ClassDef) and n.name == "ChunkGradientVector"
        )
        self.assertEqual([ast.unparse(b) for b in vector.bases], ["ChunkWorkspace"])
        self.assertNotIn("__init__", {getattr(n, "name", None) for n in vector.body})
        for n in ast.walk(trees["flash_kda_grad_causal"]):
            if isinstance(n, ast.ImportFrom):
                self.assertNotEqual(n.module, "flash_kda_grad_outputs")
        buffers = {
            getattr(n, "name", None) for n in trees["flash_kda_grad_buffer"].body
        }
        self.assertIn("ChunkWorkspace", buffers)
        self.assertIn("_gradient_slot", buffers)

    def test_semantic_calculation_names(self):
        for tree in _trees().values():
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    self.assertFalse(node.name.startswith(("StageOne", "StageTwo")))

    def test_templates_keep_math_in_shared_implementations(self):
        mathematical_calls = {
            "matmul",
            "vadd",
            "vsub",
            "vmul",
            "vmuls",
            "vexp",
            "vcast",
            "vf",
        }
        for suffix in ("persistent", "staged"):
            tree = _trees()["flash_kda_grad_" + suffix]
            calls = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertFalse(calls & mathematical_calls)
            self.assertIn("_finish_masked_track_chunk", calls)

    def test_eager_module_dependencies_have_no_cycles(self):
        graph = {name: set() for name in _MODULES}

        def imports(statements):
            for node in statements:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    yield node
                elif isinstance(node, ast.If):
                    yield from imports(node.body)
                    yield from imports(node.orelse)
                elif isinstance(node, ast.Try):
                    yield from imports(node.body)
                    for handler in node.handlers:
                        yield from imports(handler.body)

        for name, tree in _trees().items():
            for node in imports(tree.body):
                if isinstance(node, ast.ImportFrom) and node.module in _MODULES:
                    graph[name].add(node.module)
                elif isinstance(node, ast.ImportFrom) and node.module is None:
                    graph[name].update(a.name for a in node.names if a.name in _MODULES)
                elif isinstance(node, ast.Import):
                    graph[name].update(a.name for a in node.names if a.name in _MODULES)

        def visit(name, active, complete):
            self.assertNotIn(name, active, f"eager import cycle at {name}")
            if name in complete:
                return
            active.add(name)
            for dependency in graph[name]:
                visit(dependency, active, complete)
            active.remove(name)
            complete.add(name)

        complete = set()
        for name in graph:
            visit(name, set(), complete)
        self.assertNotIn("flash_kda_grad_recompute", graph["flash_kda_grad_buffer"])

    def test_both_schedules_share_state_update(self):
        tree = _trees()["flash_kda_grad_state"]
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        for name in ("PersistentStateBackward", "StagedStateBackward"):
            self.assertEqual(
                [base.id for base in classes[name].bases], ["_StateUpdate"]
            )
            own = {
                node.name
                for node in classes[name].body
                if isinstance(node, ast.FunctionDef)
            }
            self.assertFalse(own & {"prepare_state_update", "finish_state_update"})

    def test_mocked_imports_routes_and_aot(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--mock-dsl"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("All four host routes", result.stdout)


def _check_host_input_guards(checks):
    case = unittest.TestCase()
    assert checks._validate_scale(None) is None
    assert checks._validate_scale("0.125") == 0.125
    for scale in (float("nan"), float("inf"), object(), "bad"):
        with case.assertRaisesRegex(ValueError, "scale must be a finite number"):
            checks._validate_scale(scale)
    with case.assertRaisesRegex(ValueError, "do must be a Tensor"):
        checks._validate_input_structure(checks.BackwardInputs(*([None] * 21)))
    with case.assertRaisesRegex(ValueError, "layout must be TND"):
        checks._validate_state_backward_inputs(
            aqk=None,
            qg=None,
            kg=None,
            w=None,
            v_new=None,
            g_cumsum=None,
            chunk_states=None,
            layout="bad",
            metadata=None,
        )
    with case.assertRaisesRegex(ValueError, "effective Q/K heads"):
        checks._validate_chunk_heads(2, 3)
    checks._validate_forward_outputs((None,) * 16)
    with case.assertRaisesRegex(ValueError, "training forward must return 16"):
        checks._validate_forward_outputs((None,) * 15)
    print("Pure input checks reject invalid arguments without metadata execution.")


def _check_shared_cube_buffers(buffers):
    import types
    from unittest.mock import patch

    calls = []

    class Ring:
        def __init__(self, mem_loc, dtype, layout, **kw):
            calls.append((mem_loc, dtype, layout, kw))

        def next_write(self):
            return object()

    original = buffers._CUBE_SCRATCH_SCOPE.get()
    with patch.multiple(
        buffers,
        RingBuffer=Ring,
        make_layout=tuple,
        MemLoc=types.SimpleNamespace(L0A="L0A", L0B="L0B", L0C="L0C"),
    ):
        with buffers._SharedCubeScope():
            assert list(buffers._CUBE_SCRATCH_SCOPE.get()) == ["L0A", "L0B", "L0C"]
        assert buffers._CUBE_SCRATCH_SCOPE.get() is original
    assert [(loc, layout, kw) for loc, _, layout, kw in calls] == [
        (
            "L0A",
            (128, 128),
            {"depth": 1, "data_format": "nz", "physical_layout": (8, 8, 16, 16)},
        ),
        (
            "L0B",
            (128, 128),
            {"depth": 1, "data_format": "nz", "physical_layout": (8, 8, 16, 16)},
        ),
        (
            "L0C",
            (128, 256),
            {"depth": 1, "data_format": "nz", "physical_layout": (16, 8, 16, 16)},
        ),
    ]
    assert [dtype for _, dtype, _, _ in calls] == [
        buffers.dtypes.bfloat16,
        buffers.dtypes.bfloat16,
        buffers.dtypes.float32,
    ]
    print("Shared L0 allocation order, dtype, shape and NZ layout passed.")


def _check_device_entry_routes(kernels):
    import inspect
    from contextlib import nullcontext
    from unittest.mock import patch

    for staged in (False, True):
        for recompute in (False, True):
            for full, masked, cube in (
                (True, False, False),
                (False, True, False),
                (True, True, True),
            ):
                entry = kernels.FlashKDAGradKernel()
                arguments = {
                    n: object() for n in inspect.signature(entry.__call__).parameters
                }
                arguments.update(
                    staged=staged,
                    recompute=recompute,
                    use_full_chunks=full,
                    needs_masked_chunks=masked,
                    cube_causal=cube,
                )
                calls = []

                def rebuild(use_staged):
                    assert use_staged == (staged or masked or not full)
                    return lambda *args: calls.append(("recompute", args))

                def backward(use_staged):
                    assert use_staged == staged
                    return lambda *args: calls.append(("backward", args))

                with patch.multiple(
                    kernels,
                    _SharedCubeScope=nullcontext,
                    const_expr=bool,
                    _initialize_backward_gradients=lambda *args: calls.append(
                        ("initialize", args)
                    ),
                    _finish_deterministic_gate_gradients=lambda *args: calls.append(
                        ("finish", args)
                    ),
                    _recompute_implementation=rebuild,
                    _backward_implementation=backward,
                    _sync_cube_vector_handoff=lambda *args, **kw: calls.append(
                        ("sync", args, kw)
                    ),
                ):
                    entry(**arguments)
                assert [c[0] for c in calls] == (
                    ["initialize", "recompute", "sync", "backward", "finish"]
                    if recompute
                    else ["initialize", "backward", "finish"]
                )
                names = inspect.signature(
                    kernels.run_persistent_flash_kda_grad
                ).parameters
                expected = tuple(arguments[n] for n in names if n != "recomputed") + (
                    recompute,
                )
                assert calls[-2][1] == expected
                assert calls[0][1][-1] is arguments["has_dht"]
                if recompute:
                    assert len(calls[1][1]) == 26
                    assert calls[1][1][-2] is arguments["gm_owner_schedule"]
                    assert calls[1][1][-1] is arguments["round_u_pre"]
                    reset = full and masked and cube and not staged
                    assert calls[2] == (
                        "sync",
                        (2,),
                        {"grid": True, **({"reset_sync_id": True} if reset else {})},
                    )
    print(
        "One device entry preserves all cache/schedule routes, arguments and synchronization."
    )


def _check_execution_plans(host, dtypes):
    import inspect

    assert (
        inspect.signature(host._metadata.backward_chunk_paths)
        .parameters["staged"]
        .default
        is False
    )
    import types
    from unittest.mock import patch

    cases = [
        (64, 32, 32, 32, True, -0.1, False, (True, False, False, True, 32)),
        (65, 32, 32, 32, True, -0.1, False, (True, True, False, True, 32)),
        (65, 32, 32, 32, True, -0.1, True, (False, True, True, False, 96)),
        (64, 2, 2, 32, True, -0.1, False, (False, True, True, False, 4)),
        (64, 1, 2, 1, True, -0.1, False, (False, True, False, False, 1)),
        (0, 32, 32, 32, True, -0.1, False, (True, True, False, True, 32)),
        (64, 32, 32, 32, True, None, False, (False, True, False, False, 32)),
        (64, 32, 32, 32, False, -0.1, False, (False, True, False, False, 32)),
        (64, 32, 32, 32, True, -2.0, False, (True, False, False, False, 32)),
    ]
    with patch.object(
        host._metadata, "metadata_host_info", lambda metadata: metadata.info
    ):
        for length, qh, vh, cores, gate, bound, recompute, expected in cases:
            header = [0] * 8
            header[host._metadata.HEADER_VALUE_HEADS] = vh
            meta = types.SimpleNamespace(
                info=types.SimpleNamespace(
                    header=tuple(header),
                    sequences=((0, length, (length + 63) // 64, 0),),
                )
            )
            plan = host.make_backward_plan(
                meta,
                qh,
                vh,
                cores,
                use_gate=gate,
                lower_bound=bound,
                recompute=recompute,
            )
            assert (
                plan.use_full_chunks,
                plan.needs_masked_chunks,
                plan.staged,
                plan.cube_causal,
                plan.gradient_slots,
            ) == expected
            assert plan.template == ("staged" if plan.staged else "persistent")
    raw = (0.125, 1, 0.5, 2, 1, 1, 1, 1, 0.0, True, 0, True, False, False, True)
    config = host.BackwardConfig.from_abi(raw)
    assert tuple(config) == raw
    assert config.scalar_specs(dtypes) == tuple(
        v
        if i in (9, 11, 12, 13, 14)
        else dtypes.float32
        if i in (0, 2, 8)
        else dtypes.int64
        for i, v in enumerate(raw)
    )
    assert config.runtime_values() == tuple(
        float(v) if i in (0, 2, 8) else int(v)
        for i, v in enumerate(raw)
        if i not in (9, 11, 12, 13, 14)
    )
    print(
        "Execution plans cover full/tail/empty/GVA/cache modes and preserve the scalar ABI."
    )


def _check_saved_input_contracts(g, checks, torch):
    import math
    import types
    from unittest.mock import patch

    class Tensor(torch.Tensor):
        def __init__(self, shape, dtype, device):
            self.shape, self.dtype, self.device = tuple(shape), dtype, device
            self.ndim = len(self.shape)

        def numel(self):
            return math.prod(self.shape)

        def is_contiguous(self):
            return True

        def stride(self, axis):
            return 1

    case = unittest.TestCase()
    device = types.SimpleNamespace(type="npu", index=0)
    waits = []
    snapshots = []

    def snapshot(metadata):
        snapshots.append(metadata)
        return metadata.info

    with (
        patch.object(g, "wait_metadata_ready", lambda m: waits.append(m)),
        patch.object(g._metadata, "metadata_host_info", snapshot),
    ):
        tested = 0
        for layout in ("TND", "BNSD", "BSND"):
            for length in (0, 17, 64, 65):
                for saved in (False, True):
                    for g_dtype in (torch.bfloat16, torch.float32):
                        for beta_dtype in (torch.bfloat16, torch.float32):

                            def tensor(shape, dtype=torch.bfloat16):
                                return Tensor(shape, dtype, device)

                            chunks = (length + 63) // 64
                            qs = (
                                (length, 1, 128)
                                if layout == "TND"
                                else (1, 1, length, 128)
                                if layout == "BNSD"
                                else (1, length, 1, 128)
                            )
                            vs = (
                                (length, 2, 128)
                                if layout == "TND"
                                else (1, 2, length, 128)
                                if layout == "BNSD"
                                else (1, length, 2, 128)
                            )
                            caches = (
                                (length, 2, 128)
                                if layout == "TND"
                                else (1, 2, length, 128)
                            )
                            norms = (length, 1) if layout == "TND" else (1, 1, length)
                            matrices = (
                                (2, length, 64)
                                if layout == "TND"
                                else (1, 2, length, 64)
                            )
                            states = (
                                (chunks + 1, 2, 128, 128)
                                if layout == "TND"
                                else (1, 2, chunks, 128, 128)
                            )
                            meta = tensor((8 + 7 * 2,), torch.int32)
                            header = [0] * 8
                            m = g._metadata
                            header[m.HEADER_MAGIC] = m.MAGIC
                            header[m.HEADER_ABI_VERSION] = m.ABI_VERSION
                            header[m.HEADER_STATUS] = m.STATUS_OK
                            header[m.HEADER_BATCH] = 1
                            header[m.HEADER_VALUE_HEADS] = 2
                            header[m.HEADER_STORAGE_LENGTH] = length
                            header[m.HEADER_CHUNK_COUNT] = chunks
                            header[m.HEADER_FLAGS] = 3 if layout == "TND" else 0
                            meta.info = types.SimpleNamespace(
                                header=tuple(header),
                                sequences=((0, length, chunks, 0),),
                            )
                            args = dict(
                                q=tensor(qs),
                                k=tensor(qs),
                                v=tensor(vs),
                                g=tensor(vs, g_dtype),
                                beta=tensor(vs[:-1], beta_dtype),
                                do=tensor(vs),
                                dht=tensor((1, 2, 128, 128), torch.float32),
                                Aqk=tensor(matrices),
                                Akk=tensor(matrices),
                                q_rstd=tensor(norms, torch.float32),
                                k_rstd=tensor(norms, torch.float32),
                                beta_sigmoid=tensor(caches[:-1], torch.float32),
                                A_log=tensor((2,), torch.float32),
                                dt_bias=tensor((2, 128), torch.float32),
                                metadata=meta,
                                layout=layout,
                                disable_recompute=saved,
                                initial_state=tensor((1, 2, 128, 128), torch.float32),
                            )
                            for name in ("w", "u", "v_new", "qg", "kg"):
                                args[name] = tensor(caches) if saved else None
                            args["g_cumsum"] = (
                                tensor(caches, torch.float32) if saved else None
                            )
                            args["chunk_states"] = (
                                tensor(states, torch.bfloat16) if saved else None
                            )
                            waits.clear()
                            snapshots.clear()
                            g._validate_backward_inputs(**args)
                            assert waits == snapshots == [meta]
                            tested += 1
        waits.clear()
        snapshots.clear()
        with case.assertRaisesRegex(ValueError, "do must be a Tensor"):
            g._validate_backward_inputs(**dict(args, do=None))
        assert not waits and not snapshots
        with case.assertRaisesRegex(ValueError, "metadata must use the backward int32"):
            g._validate_backward_inputs(
                **dict(args, metadata=tensor((22,), torch.bfloat16))
            )
        assert not waits and not snapshots
        bad_header = list(meta.info.header)
        bad_header[0] = 0
        meta.info = types.SimpleNamespace(header=tuple(bad_header))
        with case.assertRaisesRegex(ValueError, "backward metadata ABI does not match"):
            g._validate_backward_inputs(**args)
        assert waits == snapshots == [meta]
    assert tested == 96
    print(
        "96 saved/recompute input contracts cover all layouts, tails, empty inputs and dtype combinations; rejection order passed."
    )


def _run_mock_dsl_checks():
    import ast
    import importlib
    import os
    import sys
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    folder = root / "op_dsl"

    class Symbol:
        def __init__(self, key=""):
            self.key = key

        def __getattr__(self, name):
            return Symbol((self.key, name))

        def __getitem__(self, key):
            return Symbol((self.key, key))

        def __call__(self, *args, **kwargs):
            return Symbol((self.key, args))

        def __mul__(self, other):
            return Symbol((self.key, "*", other))

        def __rmul__(self, other):
            return Symbol((other, "*", self.key))

        def __eq__(self, other):
            return isinstance(other, Symbol) and self.key == other.key

    def module(name):
        obj = types.ModuleType(name)
        obj.__path__ = []
        obj.__getattr__ = lambda prop: Symbol((name, prop))
        sys.modules[name] = obj
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, obj)
        return obj

    def identity(fn):
        return fn

    def jit(fn):
        fn.fn = fn
        return fn

    names = (
        "torch",
        "cannbotdsl",
        "cannbotdsl.lang",
        "cannbotdsl.lang.jit",
        "cannbotdsl.lang.vf",
        "cannbotdsl.lang.host",
        "cannbotdsl.lang.kernel",
        "cannbotdsl.lang.constexpr",
        "cannbotdsl.lang.control_flow",
        "cannbotdsl.ops",
        "cannbotdsl.ops.reg",
        "cannbotdsl.ops.arch",
        "cannbotdsl.ops.matmul",
        "cannbotdsl.ops.memcpy",
        "cannbotdsl.ops.sync",
        "cannbotdsl.ring_buffer",
        "cannbotdsl.buffer",
        "cannbotdsl.tensor",
        "cannbotdsl.aicpu",
        "cannbotdsl.aicpu.toolchain",
        "cannbotdsl.aot",
    )
    for name in names:
        module(name)
    sys.modules["torch"].Tensor = type("Tensor", (), {})
    dsl = sys.modules["cannbotdsl"]
    dsl.ProfileSpec = lambda **kw: kw
    dsl.Dim = lambda *args, **kw: Symbol(args)
    dsl.TensorSpec = lambda shape, dtype, **kw: types.SimpleNamespace(
        shape=shape, dtype=dtype, **kw
    )
    sys.modules["cannbotdsl.lang.jit"].jit = jit
    sys.modules["cannbotdsl.lang.host"].host = jit
    sys.modules["cannbotdsl.lang.kernel"].kernel = lambda **kw: identity
    sys.modules["cannbotdsl.aicpu"].aicpu_kernel = identity

    def compile_cache(fn):
        def collect(*args):
            fn(*args).close()

        collect.fn = fn
        return collect

    sys.modules["cannbotdsl.aot"].compile_cache = compile_cache
    sys.modules["cannbotdsl.aot"].export = lambda name: identity
    package = module("local_kdag")
    package.__path__ = [str(folder)]
    sys.path.insert(0, str(folder))
    for prefix in ("local_kdag.", ""):
        os.environ.pop("CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD", None)
        importlib.import_module(prefix + "flash_kda_grad_causal")
        assert prefix + "flash_kda_grad_outputs" not in sys.modules
        for p in sorted(folder.glob("*.py")):
            importlib.import_module(prefix + p.stem)
        g = importlib.import_module(prefix + "flash_kda_grad")
        _check_host_input_guards(
            importlib.import_module(prefix + "flash_kda_grad_check")
        )
        _check_shared_cube_buffers(
            importlib.import_module(prefix + "flash_kda_grad_buffer")
        )
        _check_saved_input_contracts(
            g,
            importlib.import_module(prefix + "flash_kda_grad_check"),
            sys.modules["torch"],
        )
        _check_device_entry_routes(
            importlib.import_module(prefix + "flash_kda_grad_kernel")
        )
        kernels = importlib.import_module(prefix + "flash_kda_grad_kernel")
        assert kernels._backward_implementation(False).__module__.endswith(
            "_persistent"
        )
        assert kernels._backward_implementation(True).__module__.endswith("_staged")
        _check_execution_plans(g, dsl.dtypes)
        assert kernels._recompute_implementation(False).__name__ == "recompute_tracks"
        assert (
            kernels._recompute_implementation(True).__name__
            == "recompute_staged_tracks"
        )
        saved = list(g._backward_variants())
        expected = [("backward", v) for v in saved] + [
            ("recompute", v) for v in saved if v[4] or not v[2]
        ]
        marker = [(kind == "recompute", *v) for kind, v in expected]
        assert len(saved) == 36
        assert sum(kind == "recompute" for kind, _ in expected) == 18
        assert len(set(expected)) == 54
        assert list(g._export_variant_calls()) == marker
        launchers = {"backward": g.FlashKDAGrad, "recompute": g._RecomputeBackward}
        for kind, v in expected:
            request = (
                g.backward_dynamic_compile_request
                if kind == "backward"
                else g._recompute_dynamic_compile_request
            )
            import inspect

            specs, scalars = request(*v)
            assert len(specs) + len(scalars) == len(
                inspect.signature(launchers[kind]().run).parameters
            )
        for full, masked, staged in (
            (False, True, False),
            (True, False, False),
            (True, True, False),
            (False, True, True),
        ):
            for recompute in (False, True):
                want = (
                    (False, True, True)
                    if recompute and masked and not staged
                    else (full, masked, staged)
                )
                assert (
                    g._resolve_chunk_template(full, masked, staged, recompute=recompute)
                    == want
                )

        def check_shards(count):
            seen = []
            for idx in range(count):
                close_count = []
                calls = []

                def compile_mock(fn, *args):
                    calls.append((fn.__qualname__, args))
                    return types.SimpleNamespace(close=lambda: close_count.append(True))

                dsl.compile = compile_mock
                os.environ["CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD"] = f"{idx}/{count}"
                g.export_flash_kda_grad()
                assert len(calls) == len(close_count) == len(expected[idx::count])
                for (name, args), (kind, v) in zip(
                    calls, expected[idx::count], strict=True
                ):
                    request = (
                        g.backward_dynamic_compile_request
                        if kind == "backward"
                        else g._recompute_dynamic_compile_request
                    )
                    assert name == launchers[kind].run.__qualname__
                    specs, scalars = request(*v)
                    assert args == (*specs, *scalars)
                seen.extend(range(idx, len(expected), count))
            assert sorted(seen) == list(range(54))

        for n in (1, 2, 7, 8, 54):
            check_shards(n)
        print(
            prefix or "flat",
            ": imports, 54 unique AOT requests, launch ABI and shard coverage passed",
        )
    test = ast.parse(
        (root / "tests/ut/test_flash_kda_grad_ut.py").read_text(encoding="utf-8")
    )
    names = {
        "test_vf_branches_are_compile_time_only",
        "test_gate_reduction_assigns_one_half_per_aiv",
        "test_fused_gate_reduction_keeps_chunk_sums_in_ub",
        "test_head_reduction_assigns_disjoint_token_rows_to_both_aivs",
        "test_fused_phases_publish_caches_for_the_selected_owners",
    }
    import inspect

    env = dict(
        integration=g,
        kernel_module=kernels,
        buffer_module=importlib.import_module("flash_kda_grad_buffer"),
        chunk_module=importlib.import_module("flash_kda_grad_chunk"),
        outputs_module=importlib.import_module("flash_kda_grad_outputs"),
        causal_module=importlib.import_module("flash_kda_grad_causal"),
        ast=ast,
        Path=Path,
        inspect=inspect,
    )
    for fn in test.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in names:
            fn.decorator_list = []
            exec(
                compile(ast.Module(body=[fn], type_ignores=[]), "<whitebox>", "exec"),
                env,
            )
            env[fn.name]()
            print(fn.name, "passed")

    from unittest.mock import patch

    fn = next(
        n
        for n in test.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_host_launches_one_bound_template"
    )
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<host_route>", "exec"), env)

    class Patch:
        def __init__(self):
            self.patches = []

        def setattr(self, *args):
            item = patch.object(*args)
            item.start()
            self.patches.append(item)

        def close(self):
            while self.patches:
                self.patches.pop().stop()

    for staged in (False, True):
        for recompute in (False, True):
            p = Patch()
            try:
                env[fn.name](p, staged, recompute)
            finally:
                p.close()
    print("All four host routes bind one template with the expected device ABI.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--mock-dsl"]:
        _run_mock_dsl_checks()
    else:
        unittest.main()
