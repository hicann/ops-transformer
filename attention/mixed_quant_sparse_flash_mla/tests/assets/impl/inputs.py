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

"""Input customization for MixedQuantSparseFlashMla TTK cases."""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch


class NumpyBatchRandomContext:
    """Make declared batch slices deterministic before quantized KV packing."""

    SEED_MODULUS = (1 << 63) - 1

    def __init__(self, q, ori_kv, cmp_kv, kwargs):
        self.layout_q = kwargs.get("layout_q", "BSND")
        self.layout_kv = kwargs.get("layout_kv", "BSND")
        self.q_prefix = (
            self.build_prefix(kwargs, "cu_seqlens_q", int(q.shape[0]), True)
            if self.layout_q == "TND"
            else None
        )
        self.batch_size = (
            len(self.q_prefix) - 1 if self.q_prefix is not None else int(q.shape[0])
        )
        self.relations = self.parse_relations(q, kwargs)
        self.batch_relations = self.map_batch_relations()
        self.validate_relation_contract(kwargs)
        self.extent_selectors = {}
        self.register_extent(self.batch_size, self.batch_relations, "logical batch")
        if self.q_prefix is not None:
            self.register_prefix(self.q_prefix, "q")
        if self.layout_kv == "TND":
            if ori_kv is not None:
                self.register_prefix(
                    self.build_prefix(
                        kwargs, "cu_seqlens_ori_kv", int(ori_kv.shape[0]), True
                    ),
                    "ori_kv",
                )
            if cmp_kv is not None:
                self.register_prefix(
                    self.build_prefix(
                        kwargs, "cu_seqlens_cmp_kv", int(cmp_kv.shape[0]), True
                    ),
                    "cmp_kv",
                )
        self.base_seed = self.relations[0][1]
        self.call_index = 0
        self.original_numpy_uniform = None
        self.original_python_uniform = None

    @classmethod
    def from_case(cls, q, ori_kv, cmp_kv, kwargs):
        fields = tuple(kwargs.get(name) for name in (
            "batch_axis", "batch_slice_info", "batch_seed"
        ))
        if all(field is None for field in fields):
            return None
        if any(field is None for field in fields):
            raise ValueError(
                "batch_axis, batch_slice_info and batch_seed must be set together"
            )
        layout_q = kwargs.get("layout_q", "BSND")
        layout_kv = kwargs.get("layout_kv", "BSND")
        if layout_q not in ("BSND", "TND"):
            raise ValueError(f"MQSMLA batch consistency does not support layout_q={layout_q!r}")
        if layout_kv not in ("BSND", "TND", "PA_BBND"):
            raise ValueError(f"MQSMLA batch consistency does not support layout_kv={layout_kv!r}")
        return cls(q, ori_kv, cmp_kv, kwargs)

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @classmethod
    def build_prefix(cls, kwargs, name, expected_total, required):
        value = cls.list_value(kwargs, name)
        if value is None:
            if required:
                raise ValueError(f"MQSMLA batch consistency requires explicit {name}_values")
            return None
        if len(value) < 2 or value[0] != 0 or value[-1] != expected_total:
            raise ValueError(
                f"MQSMLA {name}_values must start at 0 and end at {expected_total}: {value!r}"
            )
        if any(right <= left for left, right in zip(value, value[1:])):
            raise ValueError(
                f"MQSMLA {name}_values must be strictly increasing for batch relations"
            )
        return value

    def parse_relations(self, q, kwargs):
        batch_axis = kwargs["batch_axis"]
        batch_slices = kwargs["batch_slice_info"]
        batch_seed = kwargs["batch_seed"]
        if not batch_axis or batch_axis[0] != (0,):
            raise ValueError("MQSMLA batch consistency requires q batch_axis=((0,), ...)")
        if not batch_slices or batch_slices[0] is None or batch_seed[0] is None:
            raise ValueError("MQSMLA batch consistency requires slices and seeds on q")
        if any(slices is not None for slices in batch_slices[1:]):
            raise ValueError("MQSMLA batch consistency relations must be declared on q only")

        relations = list(zip(batch_slices[0][0], batch_seed[0][0]))
        if not relations:
            raise ValueError("MQSMLA batch consistency requires at least one q slice")
        if len({seed for _, seed in relations}) != 1:
            raise ValueError("MQSMLA batch consistency supports one relation seed per case")
        for slice_value, _ in relations:
            start, stop, step = slice_value
            if step != 1 or start < 0 or start >= stop or stop > int(q.shape[0]):
                raise ValueError(
                    "MQSMLA batch consistency requires in-range, contiguous q axis slices"
                )
        return [(tuple(int(item) for item in value), int(seed)) for value, seed in relations]

    def map_batch_relations(self):
        if self.q_prefix is None:
            return [((start, stop, step), seed) for (start, stop, step), seed in self.relations]
        boundary_to_batch = {offset: index for index, offset in enumerate(self.q_prefix)}
        mapped = []
        for (start, stop, step), seed in self.relations:
            if start not in boundary_to_batch or stop not in boundary_to_batch:
                raise ValueError(
                    "MQSMLA TND batch slice must align with complete cu_seqlens_q intervals: "
                    f"{(start, stop, step)!r}"
                )
            mapped.append(
                ((boundary_to_batch[start], boundary_to_batch[stop], step), seed)
            )
        return mapped

    def validate_relation_contract(self, kwargs):
        reference_slice = self.batch_relations[0][0]
        reference_count = reference_slice[1] - reference_slice[0]
        vector_names = (
            "seqused_q",
            "seqused_ori_kv",
            "seqused_cmp_kv",
            "cmp_residual_kv",
        )
        vectors = {}
        for name in vector_names:
            value = self.list_value(kwargs, name)
            if value is not None:
                vectors[name] = value
        prefixes = []
        for name, value in (
            ("cu_seqlens_q", self.q_prefix),
            ("cu_seqlens_ori_kv", self.list_value(kwargs, "cu_seqlens_ori_kv")),
            ("cu_seqlens_cmp_kv", self.list_value(kwargs, "cu_seqlens_cmp_kv")),
        ):
            if value is not None:
                prefixes.append((name, value))
        for name, value in vectors.items():
            if len(value) != self.batch_size:
                raise ValueError(f"MQSMLA {name}_values length must equal B={self.batch_size}")
        for name, value in prefixes:
            if len(value) != self.batch_size + 1:
                raise ValueError(f"MQSMLA {name}_values length must equal B + 1")

        def relation_signature(batch_slice):
            start, stop, _ = batch_slice
            signature = []
            for _name, value in prefixes:
                signature.append(tuple(
                    value[index + 1] - value[index] for index in range(start, stop)
                ))
            for value in vectors.values():
                signature.append(tuple(value[start:stop]))
            return tuple(signature)

        reference_signature = relation_signature(reference_slice)
        for batch_slice, _ in self.batch_relations[1:]:
            if batch_slice[1] - batch_slice[0] != reference_count:
                raise ValueError("MQSMLA relation slices must contain the same logical batch count")
            if relation_signature(batch_slice) != reference_signature:
                raise ValueError(
                    "MQSMLA relation slices require identical q/KV lengths and residual values"
                )

    def register_extent(self, extent, relations, source):
        selectors = tuple(value for value, _ in relations)
        existing = self.extent_selectors.get(int(extent))
        if existing is not None and existing[0] != selectors:
            raise ValueError(
                f"MQSMLA cannot distinguish generated {source} extent {extent} from "
                f"{existing[1]} with different relation slices"
            )
        self.extent_selectors[int(extent)] = (selectors, source)

    def register_prefix(self, prefix, source):
        selectors = []
        for (start, stop, _), _seed in self.batch_relations:
            selectors.append((prefix[start], prefix[stop], 1))
        self.register_extent(prefix[-1], tuple((value, 0) for value in selectors), source)

    def validate_params(self, params):
        if params.get("template_run_mode") not in ("SWA", "HCA"):
            raise ValueError("MQSMLA batch consistency currently excludes sparse modes")
        if int(params.get("quant_mode")) != 1:
            raise ValueError("MQSMLA batch consistency currently supports quant_mode=1 only")

    @classmethod
    def derive_seed(cls, seed, call_index):
        return (int(seed) + (call_index + 1) * 1000003) % cls.SEED_MODULUS

    def numpy_uniform(self, low=0.0, high=1.0, size=None):
        call_index = self.call_index
        self.call_index += 1
        rng = np.random.default_rng(self.derive_seed(self.base_seed, call_index))
        value = rng.uniform(low, high, size)
        if size is None:
            return value
        value = np.asarray(value)
        if value.ndim < 2:
            return value
        extent = self.extent_selectors.get(int(value.shape[0]))
        if extent is None:
            return value
        selectors = extent[0]
        for slice_value, (_batch_slice, seed) in zip(selectors, self.batch_relations):
            selector = (slice(*slice_value),) + (slice(None),) * (value.ndim - 1)
            piece = value[selector]
            piece_rng = np.random.default_rng(self.derive_seed(seed, call_index))
            value[selector] = piece_rng.uniform(low, high, piece.shape)
        return value

    def python_uniform(self, low, high):
        call_index = self.call_index
        self.call_index += 1
        rng = random.Random(self.derive_seed(self.base_seed, call_index))
        return rng.uniform(low, high)

    def __enter__(self):
        self.original_numpy_uniform = np.random.uniform
        self.original_python_uniform = random.uniform
        np.random.uniform = self.numpy_uniform
        random.uniform = self.python_uniform
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.uniform = self.original_numpy_uniform
        random.uniform = self.original_python_uniform
        return False


class MixedQuantSparseFlashMlaInputAdapter:
    """Translate a TTK case to pytest parameters and reuse pytest generation."""

    PYTEST_BLOCK_SIZE = 64
    PYTEST_TILE_SIZE = 64
    TEMPLATE_MODES = frozenset(("SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"))

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load MixedQuantSparseFlashMla module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_modules = {}

    @staticmethod
    def load_golden_store():
        name = "mqsmla_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise MixedQuantSparseFlashMlaInputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    def load_pytest_module(self, stem, filename):
        if stem in self.pytest_modules:
            return self.pytest_modules[stem]

        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        module_path = pytest_dir / filename
        name = f"mqsmla_pytest_{stem}"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.pytest_modules[stem] = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error(f"pytest {stem}", module_path, exc) from exc
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[i + 1]) - int(value[i]) for i in range(len(value) - 1)]

    @staticmethod
    def data_range(input_ranges, index):
        if not input_ranges or index >= len(input_ranges):
            return None
        value = input_ranges[index]
        if value is None or any(item is None for item in value):
            return None
        return list(value)

    @classmethod
    def select_template_mode(cls, kwargs):
        mode = kwargs.get("template_run_mode") or kwargs.get("template_mode")
        if mode is None:
            return None
        mode = str(mode).strip()
        if mode not in cls.TEMPLATE_MODES:
            raise ValueError(f"unsupported explicit template_run_mode: {mode!r}")
        return mode

    def build_case_params(self, q, ori_kv, cmp_kv, ori_sparse_indices,
                          cmp_sparse_indices,
                          layout_q, layout_kv, kwargs):
        is_batch_case = kwargs.get("batch_axis") is not None
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_ori = self.list_value(kwargs, "cu_seqlens_ori_kv")
        cu_cmp = self.list_value(kwargs, "cu_seqlens_cmp_kv")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_ori = self.list_value(kwargs, "seqused_ori_kv")
        seq_cmp = self.list_value(kwargs, "seqused_cmp_kv")
        residual = self.list_value(kwargs, "cmp_residual_kv")

        if layout_q == "BSND":
            batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
        else:
            _, q_heads, head_dim = [int(x) for x in q.shape]
            batch_size = len(seq_q or self.prefix_lengths(cu_q))
            q_seq = max(self.prefix_lengths(cu_q) or seq_q or [int(q.shape[0])])
        if kwargs.get("S1") is not None:
            q_seq = int(kwargs["S1"])

        if layout_kv == "BSND":
            _, kv_seq, kv_heads, _ = [int(x) for x in ori_kv.shape]
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
            if is_batch_case and block_size1 is None:
                block_size1 = self.PYTEST_BLOCK_SIZE
        elif layout_kv == "TND":
            _, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [int(ori_kv.shape[0])])
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
            if is_batch_case and block_size1 is None:
                block_size1 = self.PYTEST_BLOCK_SIZE
        else:
            shape_block_num, shape_block_size, kv_heads, _ = [int(x) for x in ori_kv.shape]
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
            block_num1 = shape_block_num if block_num1 is None else int(block_num1)
            block_size1 = shape_block_size if block_size1 is None else int(block_size1)
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [block_size1])
        if kwargs.get("S2") is not None:
            kv_seq = int(kwargs["S2"])

        metadata_cmp_topk = kwargs.get("metadata_cmp_topk")
        if cmp_kv is None:
            block_num2 = kwargs.get("block_num2")
            block_size2 = kwargs.get("block_size2")
            if is_batch_case and block_size2 is None:
                block_size2 = block_size1
        elif layout_kv == "PA_BBND":
            block_num2 = kwargs.get("block_num2")
            block_size2 = kwargs.get("block_size2")
            block_num2 = int(cmp_kv.shape[0]) if block_num2 is None else int(block_num2)
            block_size2 = int(cmp_kv.shape[1]) if block_size2 is None else int(block_size2)
        else:
            block_num2 = kwargs.get("block_num2")
            block_size2 = kwargs.get("block_size2")
            if is_batch_case and block_size2 is None:
                block_size2 = block_size1

        mode = self.select_template_mode(kwargs)

        params = dict(kwargs)
        input_ranges = kwargs.get("input_ranges") or ()
        data_ranges = {
            "q_datarange": self.data_range(input_ranges, 0),
            "ori_kv_datarange": self.data_range(input_ranges, 1),
            "cmp_kv_datarange": self.data_range(input_ranges, 2),
        }
        params.update({
            "Testcase_Name": kwargs.get("testcase_name"),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "q_type": q.dtype,
            "ori_kv_type": ori_kv.dtype,
            "cmp_kv_type": cmp_kv.dtype if cmp_kv is not None else None,
            "B": batch_size,
            "S1": q_seq,
            "S2": kv_seq,
            "N1": q_heads,
            "N2": kv_heads,
            "D": head_dim,
            "K1": int(ori_sparse_indices.shape[-1]) if ori_sparse_indices is not None else None,
            "K": (
                metadata_cmp_topk
                if metadata_cmp_topk is not None
                else (
                    int(cmp_sparse_indices.shape[-1])
                    if cmp_sparse_indices is not None
                    else None
                )
            ),
            "block_num1": block_num1,
            "block_num2": block_num2,
            "block_size1": block_size1,
            "block_size2": block_size2,
            "seqused_q": seq_q,
            "cu_seqlens_q": cu_q,
            "seqused_ori_kv": seq_ori,
            "seqused_cmp_kv": seq_cmp,
            "cu_seqlens_ori_kv": cu_ori,
            "cu_seqlens_cmp_kv": cu_cmp,
            "cmp_residual_kv": residual,
        })
        if is_batch_case and params.get("tile_size") is None:
            # The batch CSV has no target API field for this pytest packing geometry.
            params["tile_size"] = self.PYTEST_TILE_SIZE
        if mode is not None:
            params["template_run_mode"] = mode
        params.update({name: value for name, value in data_ranges.items() if value is not None})
        return params

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            if src is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest generator produced a tensor"
                )
            return
        if src is None:
            raise ValueError(f"{name} is present in CSV but pytest generator returned None")
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    def generate_case(self, params, batch_random=None):
        pytest_utils = self.load_pytest_module("utils", "utils.py")
        pytest_check = self.load_pytest_module("check", "check_valid_param.py")
        pytest_golden = self.load_pytest_module(
            "golden", "mixed_quant_sparse_flash_mla_golden.py"
        )
        filled = pytest_utils.fill_none_params(params)
        for name in ("q_datarange", "ori_kv_datarange", "cmp_kv_datarange"):
            if params.get(name) is not None:
                filled[name] = params[name]
        if filled.get("Testcase_Name") is None:
            filled["Testcase_Name"] = params.get("testcase_name")
        pytest_check.check_valid_param(filled)
        if batch_random is None:
            data = pytest_golden.generate_and_save_testdata(filled, save_pt=False)
        else:
            batch_random.validate_params(filled)
            with batch_random:
                data = pytest_golden.generate_and_save_testdata(filled, save_pt=False)
        testcase_name = params.get("testcase_name") or filled.get("Testcase_Name")
        self.load_golden_store().CASE_DATA.put(testcase_name, data)
        return data


INPUT_ADAPTER = MixedQuantSparseFlashMlaInputAdapter()


def generate_mixed_quant_sparse_flash_mla_inputs(q, *, ori_kv=None, cmp_kv=None,
                                                 ori_sparse_indices=None, cmp_sparse_indices=None,
                                                 ori_block_table=None, cmp_block_table=None,
                                                 cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                                 cu_seqlens_cmp_kv=None, seqused_q=None,
                                                 seqused_ori_kv=None, seqused_cmp_kv=None,
                                                 cmp_residual_kv=None, ori_topk_length=None,
                                                 cmp_topk_length=None, sinks=None, **kwargs):
    """Reuse the pytest parameter validation and input processing for a TTK case."""
    batch_random = NumpyBatchRandomContext.from_case(q, ori_kv, cmp_kv, kwargs)
    params = INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        kwargs.get("layout_q"),
        kwargs.get("layout_kv"),
        kwargs,
    )
    data = INPUT_ADAPTER.generate_case(params, batch_random)
    op_input = data["op_input"]
    for name, tensor in (
        ("q", q),
        ("ori_kv", ori_kv),
        ("cmp_kv", cmp_kv),
        ("ori_sparse_indices", ori_sparse_indices),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_block_table", ori_block_table),
        ("cmp_block_table", cmp_block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_ori_kv", cu_seqlens_ori_kv),
        ("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv),
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("cmp_residual_kv", cmp_residual_kv),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("sinks", sinks),
    ):
        INPUT_ADAPTER.copy_tensor(tensor, op_input.get(name), name)
