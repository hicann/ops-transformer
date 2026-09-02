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

import hashlib
import importlib.util
import random
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch


class NumpyBatchRandomContext:
    """Map logical B/S relations to MQSMLA pytest random tensors."""

    SEED_MODULUS = (1 << 63) - 1

    def __init__(self, q, ori_kv, cmp_kv, kwargs):
        self.layout_q = kwargs.get("layout_q", "BSND")
        self.layout_kv = kwargs.get("layout_kv", "BSND")
        self.q_shape = tuple(int(value) for value in q.shape)
        self.has_cmp_kv = cmp_kv is not None
        self.q_prefix = (
            self.build_prefix(kwargs, "cu_seqlens_q", int(q.shape[0]), True)
            if self.layout_q == "TND"
            else None
        )
        self.batch_size = (
            len(self.q_prefix) - 1 if self.q_prefix is not None else int(q.shape[0])
        )
        self.q_lengths = (
            self.prefix_lengths(self.q_prefix)
            if self.q_prefix is not None
            else [int(q.shape[1])] * self.batch_size
        )
        self.effective_q_lengths = (
            self.list_value(kwargs, "seqused_q") or self.q_lengths
        )
        self.relations = self.parse_relations(kwargs)
        self.batch_relations = [
            (batch_slice, seed) for batch_slice, _sequence_slice, seed in self.relations
        ]
        self.query_selectors = self.map_query_relations()
        self.validate_relation_contract(kwargs)
        self.extent_selectors = {}
        self.register_extent(self.batch_size, self.batch_relations, "logical batch")
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
        self.relation_seed = self.relations[0][2]
        self.base_seed = self.case_seed(kwargs.get("testcase_name"), self.relation_seed)
        self.call_index = 0
        self.randperm_call_index = 0
        self.randperm_batch_offsets = None
        self.randperm_relation_keys = {}
        self.original_numpy_uniform = None
        self.original_python_uniform = None
        self.original_torch_randperm = None

    @classmethod
    def case_seed(cls, testcase_name, fallback):
        """Give each case an independent background while keeping relation seed explicit."""
        if not testcase_name:
            return int(fallback)
        digest = hashlib.sha256(str(testcase_name).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % cls.SEED_MODULUS

    @classmethod
    def from_case(cls, q, ori_kv, cmp_kv, kwargs):
        fields = tuple(
            kwargs.get(name)
            for name in ("batch_axis", "batch_slice_info", "batch_seed")
        )
        if all(field is None for field in fields):
            return None
        if any(field is None for field in fields):
            raise ValueError(
                "batch_axis, batch_slice_info and batch_seed must be set together"
            )
        layout_q = kwargs.get("layout_q", "BSND")
        layout_kv = kwargs.get("layout_kv", "BSND")
        if layout_q not in ("BSND", "TND"):
            raise ValueError(
                f"MQSMLA batch consistency does not support layout_q={layout_q!r}"
            )
        if layout_kv not in ("BSND", "TND", "PA_BBND"):
            raise ValueError(
                f"MQSMLA batch consistency does not support layout_kv={layout_kv!r}"
            )
        return cls(q, ori_kv, cmp_kv, kwargs)

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(prefix):
        return [right - left for left, right in zip(prefix, prefix[1:])]

    @classmethod
    def build_prefix(cls, kwargs, name, expected_total, required):
        value = cls.list_value(kwargs, name)
        if value is None:
            if required:
                raise ValueError(
                    f"MQSMLA batch consistency requires explicit {name}_values"
                )
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

    @staticmethod
    def parse_slice(value, extent, label):
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError(f"MQSMLA invalid {label} slice: {value!r}")
        if not all(isinstance(item, int) for item in value):
            raise ValueError(f"MQSMLA {label} slice must contain integers: {value!r}")
        start, stop, step = (int(item) for item in value)
        if step != 1 or start < 0 or start >= stop or stop > extent:
            raise ValueError(
                f"MQSMLA {label} slice must be in-range, non-empty and contiguous: {value!r}"
            )
        return start, stop, step

    def parse_relations(self, kwargs):
        batch_axis = kwargs["batch_axis"]
        batch_slices = kwargs["batch_slice_info"]
        batch_seed = kwargs["batch_seed"]
        if not batch_axis or tuple(batch_axis[0]) not in ((0,), (0, 1)):
            raise ValueError(
                "MQSMLA q batch_axis must use logical BSND axes (0,) or (0, 1)"
            )
        if not batch_slices or batch_slices[0] is None or batch_seed[0] is None:
            raise ValueError("MQSMLA batch consistency requires slices and seeds on q")
        if not (len(batch_axis) == len(batch_slices) == len(batch_seed)):
            raise ValueError("MQSMLA batch metadata top-level counts differ")
        if any(value is not None for value in batch_slices[1:]):
            raise ValueError(
                "MQSMLA batch consistency relations must be declared on q only"
            )
        if any(value is not None for value in batch_seed[1:]):
            raise ValueError(
                "MQSMLA batch consistency seeds must be declared on q only"
            )

        axes = tuple(batch_axis[0])
        axis_slices = batch_slices[0]
        axis_seeds = batch_seed[0]
        if len(axis_slices) != len(axes) or len(axis_seeds) != len(axes):
            raise ValueError("MQSMLA q slice/seed groups must match q logical axes")
        sample_count = len(axis_slices[0])
        if sample_count == 0:
            raise ValueError("MQSMLA batch consistency requires at least one q slice")
        if any(len(values) != sample_count for values in (*axis_slices, *axis_seeds)):
            raise ValueError("MQSMLA q axis groups must contain the same sample count")

        relations = []
        for index in range(sample_count):
            batch_slice = self.parse_slice(
                axis_slices[0][index], self.batch_size, "logical B"
            )
            seed = axis_seeds[0][index]
            if not isinstance(seed, int):
                raise ValueError("MQSMLA batch seed must be an integer")
            sequence_slice = None
            if axes == (0, 1):
                if axis_seeds[1][index] != seed:
                    raise ValueError(
                        "MQSMLA logical B and S slices must use the same seed"
                    )
                if batch_slice[1] - batch_slice[0] != 1:
                    raise ValueError(
                        "MQSMLA logical (B,S) relation requires one B per sample"
                    )
                batch_index = batch_slice[0]
                sequence_slice = self.parse_slice(
                    axis_slices[1][index],
                    self.effective_q_lengths[batch_index],
                    "logical S",
                )
            relations.append((batch_slice, sequence_slice, int(seed)))
        if len({seed for _batch, _sequence, seed in relations}) != 1:
            raise ValueError(
                "MQSMLA batch consistency supports one relation seed per case"
            )
        return relations

    def map_query_relations(self):
        selectors = []
        for batch_slice, sequence_slice, _seed in self.relations:
            batch_start, batch_stop, _ = batch_slice
            if self.layout_q == "BSND":
                selector = [batch_slice]
                if sequence_slice is not None:
                    selector.append(sequence_slice)
            elif sequence_slice is None:
                selector = [(self.q_prefix[batch_start], self.q_prefix[batch_stop], 1)]
            else:
                sequence_start, sequence_stop, _ = sequence_slice
                selector = [
                    (
                        self.q_prefix[batch_start] + sequence_start,
                        self.q_prefix[batch_start] + sequence_stop,
                        1,
                    )
                ]
            selectors.append(tuple(selector))
        return selectors

    def validate_disjoint_relations(self):
        """Reject relation samples that select the same logical q positions."""
        for index, (batch_slice, sequence_slice, _seed) in enumerate(self.relations):
            for candidate_batch, candidate_sequence, _candidate_seed in self.relations[
                index + 1 :
            ]:
                if (
                    batch_slice[1] <= candidate_batch[0]
                    or candidate_batch[1] <= batch_slice[0]
                ):
                    continue
                if sequence_slice is None:
                    raise ValueError(
                        "MQSMLA relation samples must not overlap logical B positions"
                    )
                if candidate_sequence is None or not (
                    sequence_slice[1] <= candidate_sequence[0]
                    or candidate_sequence[1] <= sequence_slice[0]
                ):
                    raise ValueError(
                        "MQSMLA relation samples must not overlap logical q positions"
                    )

    def validate_relation_contract(self, kwargs):
        self.validate_disjoint_relations()
        reference_slice = self.batch_relations[0][0]
        reference_count = reference_slice[1] - reference_slice[0]
        reference_sequence = self.relations[0][1]
        reference_sequence_count = (
            reference_sequence[1] - reference_sequence[0]
            if reference_sequence is not None
            else None
        )
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
                raise ValueError(
                    f"MQSMLA {name}_values length must equal B={self.batch_size}"
                )
        for name, value in prefixes:
            if len(value) != self.batch_size + 1:
                raise ValueError(f"MQSMLA {name}_values length must equal B + 1")

        def relation_signature(batch_slice):
            start, stop, _ = batch_slice
            signature = []
            for _name, value in prefixes:
                signature.append(
                    tuple(
                        value[index + 1] - value[index] for index in range(start, stop)
                    )
                )
            for value in vectors.values():
                signature.append(tuple(value[start:stop]))
            return tuple(signature)

        reference_signature = relation_signature(reference_slice)
        for relation, (batch_slice, _seed) in zip(
            self.relations[1:], self.batch_relations[1:]
        ):
            if batch_slice[1] - batch_slice[0] != reference_count:
                raise ValueError(
                    "MQSMLA relation slices must contain the same logical batch count"
                )
            sequence_slice = relation[1]
            sequence_count = (
                sequence_slice[1] - sequence_slice[0]
                if sequence_slice is not None
                else None
            )
            if sequence_count != reference_sequence_count:
                raise ValueError(
                    "MQSMLA relation slices must contain the same logical S count"
                )
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
        self.register_extent(
            prefix[-1], tuple((value, 0) for value in selectors), source
        )

    def validate_params(self, params):
        mode = params.get("template_run_mode")
        if mode not in ("SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"):
            raise ValueError(f"MQSMLA batch consistency does not support mode {mode!r}")
        if int(params.get("quant_mode")) not in (1, 2):
            raise ValueError("MQSMLA batch consistency supports quant_mode 1 and 2")
        if mode not in ("CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"):
            self.validate_sequence_masks(params)
            return
        for name in ("ori_sparse_indices_mode", "cmp_sparse_indices_mode"):
            value = params.get(name)
            if value is not None and value != "full":
                raise ValueError(
                    "MQSMLA batch consistency requires full sparse indices"
                )
        for name in ("ori_kv_topk_mode", "cmp_kv_topk_mode"):
            value = params.get(name)
            if value is not None and value not in ("full", "fullK", "no"):
                raise ValueError(
                    "MQSMLA batch consistency excludes random topk lengths"
                )
        self.validate_sequence_masks(params)
        self.configure_randperm(params)

    def validate_sequence_masks(self, params):
        sequence_slices = [relation[1] for relation in self.relations]
        if sequence_slices[0] is None or len(set(sequence_slices)) == 1:
            return
        mask_names = ["ori_mask_mode"]
        if self.has_cmp_kv:
            mask_names.append("cmp_mask_mode")
        invalid = {
            name: params.get(name)
            for name in mask_names
            if params.get(name) not in (None, 0)
        }
        if invalid:
            raise ValueError(
                "MQSMLA logical S slices at different positions require no-mask mode: "
                f"{invalid!r}"
            )

    def configure_randperm(self, params):
        sequence_lengths = params.get("seqused_q")
        if torch.is_tensor(sequence_lengths):
            sequence_lengths = sequence_lengths.detach().cpu().reshape(-1).tolist()
        if sequence_lengths is None:
            prefix = params.get("cu_seqlens_q")
            if torch.is_tensor(prefix):
                prefix = prefix.detach().cpu().reshape(-1).tolist()
            if prefix is not None:
                sequence_lengths = [
                    prefix[index + 1] - prefix[index]
                    for index in range(len(prefix) - 1)
                ]
            else:
                sequence_lengths = [int(params["S1"])] * self.batch_size
        if len(sequence_lengths) != self.batch_size:
            raise ValueError("MQSMLA sparse batch q lengths must contain B values")

        calls_per_batch = [
            int(length) * int(params["N2"]) for length in sequence_lengths
        ]
        offsets = [0]
        for count in calls_per_batch:
            offsets.append(offsets[-1] + count)
        if offsets[-1] <= 0:
            raise ValueError("MQSMLA sparse batch requires a non-empty q relation")
        self.randperm_batch_offsets = offsets
        n2 = int(params["N2"])
        for batch_slice, sequence_slice, seed in self.relations:
            batch_start, batch_stop, _ = batch_slice
            for relative_batch, batch_index in enumerate(
                range(batch_start, batch_stop)
            ):
                token_start = 0 if sequence_slice is None else sequence_slice[0]
                token_stop = (
                    int(sequence_lengths[batch_index])
                    if sequence_slice is None
                    else sequence_slice[1]
                )
                for token_index in range(token_start, token_stop):
                    for head_index in range(n2):
                        local_index = token_index * n2 + head_index
                        key = (batch_index, local_index)
                        relation_key = (
                            seed,
                            relative_batch,
                            token_index - token_start,
                            head_index,
                        )
                        existing = self.randperm_relation_keys.get(key)
                        if existing is not None and existing != relation_key:
                            raise ValueError(
                                "MQSMLA sparse relation slices overlap ambiguously"
                            )
                        self.randperm_relation_keys[key] = relation_key

    @classmethod
    def derive_seed(cls, seed, call_index):
        return (int(seed) + (call_index + 1) * 1000003) % cls.SEED_MODULUS

    def numpy_uniform(self, low=0.0, high=1.0, size=None):
        call_index = self.call_index
        self.call_index += 1
        requested_rank = (
            0 if size is None else 1 if isinstance(size, int) else len(size)
        )
        seed = self.base_seed if requested_rank >= 2 else self.relation_seed
        rng = np.random.default_rng(self.derive_seed(seed, call_index))
        value = rng.uniform(low, high, size)
        if size is None:
            return value
        value = np.asarray(value)
        if value.ndim < 2:
            return value
        if tuple(int(item) for item in value.shape) == self.q_shape:
            selectors = self.query_selectors
        else:
            extent = self.extent_selectors.get(int(value.shape[0]))
            selectors = None if extent is None else tuple((item,) for item in extent[0])
        if selectors is None:
            return value
        for selector_values, (_batch_slice, _sequence_slice, seed) in zip(
            selectors, self.relations
        ):
            selector = tuple(slice(*item) for item in selector_values)
            selector += (slice(None),) * (value.ndim - len(selector_values))
            piece = value[selector]
            piece_rng = np.random.default_rng(self.derive_seed(seed, call_index))
            value[selector] = piece_rng.uniform(low, high, piece.shape)
        return value

    def python_uniform(self, low, high):
        call_index = self.call_index
        self.call_index += 1
        rng = random.Random(self.derive_seed(self.relation_seed, call_index))
        return rng.uniform(low, high)

    def torch_randperm(self, n, **kwargs):
        if self.randperm_batch_offsets is None:
            return self.original_torch_randperm(n, **kwargs)
        call_index = self.randperm_call_index
        self.randperm_call_index += 1
        total_calls = self.randperm_batch_offsets[-1]
        cycle_index, cycle_offset = divmod(call_index, total_calls)
        batch_index = bisect_right(self.randperm_batch_offsets, cycle_offset) - 1
        local_index = cycle_offset - self.randperm_batch_offsets[batch_index]
        relation = self.randperm_relation_keys.get((batch_index, local_index))
        if relation is None:
            seed = self.base_seed
            seed_index = 3000017 + call_index
        else:
            seed, relative_batch, relative_token, head_index = relation
            seed_index = (
                2000003
                + cycle_index * 1000000007
                + relative_batch * 1000003
                + relative_token * 1009
                + head_index
            )
        call_kwargs = dict(kwargs)
        device = call_kwargs.get("device") or "cpu"
        generator = torch.Generator(device=device)
        generator.manual_seed(self.derive_seed(seed, seed_index))
        call_kwargs["generator"] = generator
        return self.original_torch_randperm(n, **call_kwargs)

    def normalize_block_tables(self, data):
        if self.layout_kv != "PA_BBND":
            return
        op_input = data.get("op_input") or data.get("input", {})
        reference = self.relations[0][0]
        for name in ("ori_block_table", "cmp_block_table"):
            table = op_input.get(name)
            if table is None:
                continue
            for batch_slice, _sequence_slice, _seed in self.relations[1:]:
                for offset in range(reference[1] - reference[0]):
                    source = reference[0] + offset
                    target = batch_slice[0] + offset
                    table[target].copy_(table[source])

    def __enter__(self):
        self.original_numpy_uniform = np.random.uniform
        self.original_python_uniform = random.uniform
        self.original_torch_randperm = torch.randperm
        np.random.uniform = self.numpy_uniform
        random.uniform = self.python_uniform
        torch.randperm = self.torch_randperm
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.uniform = self.original_numpy_uniform
        random.uniform = self.original_python_uniform
        torch.randperm = self.original_torch_randperm
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

    @staticmethod
    def load_metadata_protocol():
        name = "mqsmla_ttk_metadata_protocol"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("metadata_protocol.py")
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
                "assets metadata protocol", path, exc
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

    def build_case_params(
        self,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        layout_q,
        layout_kv,
        kwargs,
    ):
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
            kv_seq = max(
                self.prefix_lengths(cu_ori) or seq_ori or [int(ori_kv.shape[0])]
            )
            block_num1 = kwargs.get("block_num1")
            block_size1 = kwargs.get("block_size1")
            if is_batch_case and block_size1 is None:
                block_size1 = self.PYTEST_BLOCK_SIZE
        else:
            shape_block_num, shape_block_size, kv_heads, _ = [
                int(x) for x in ori_kv.shape
            ]
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
            block_size2 = (
                int(cmp_kv.shape[1]) if block_size2 is None else int(block_size2)
            )
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
        params.update(
            {
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
                "K1": int(ori_sparse_indices.shape[-1])
                if ori_sparse_indices is not None
                else None,
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
            }
        )
        if is_batch_case and params.get("tile_size") is None:
            # The batch CSV has no target API field for this pytest packing geometry.
            params["tile_size"] = self.PYTEST_TILE_SIZE
        if mode is not None:
            params["template_run_mode"] = mode
        params.update(
            {name: value for name, value in data_ranges.items() if value is not None}
        )
        return params

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            if src is not None:
                import logging

                logging.warning(
                    f"{name} is absent from CSV but pytest generator produced a tensor. Skipping copy."
                )
            return
        if src is None:
            raise ValueError(
                f"{name} is present in CSV but pytest generator returned None"
            )
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
            data = pytest_golden.gen_data(filled, generate_golden=False)
        else:
            batch_random.validate_params(filled)
            with batch_random:
                data = pytest_golden.gen_data(filled, generate_golden=False)
            batch_random.normalize_block_tables(data)
        return data


INPUT_ADAPTER = MixedQuantSparseFlashMlaInputAdapter()


def zero_metadata(metadata):
    if metadata is None:
        return
    if torch.is_tensor(metadata):
        metadata.zero_()
    else:
        metadata[...] = 0


def generate_mixed_quant_sparse_flash_mla_inputs(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode=None,
    rope_head_dim=None,
    softmax_scale=None,
    cmp_ratio=None,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    key_dtype=None,
    value_dtype=None,
    **kwargs,
):
    """Populate pytest-derived inputs and leave metadata for npu_preprocess."""
    params = dict(kwargs)
    params.update(
        {
            "quant_mode": quant_mode,
            "rope_head_dim": rope_head_dim,
            "softmax_scale": softmax_scale,
            "cmp_ratio": cmp_ratio,
            "ori_mask_mode": ori_mask_mode,
            "cmp_mask_mode": cmp_mask_mode,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "topk_value_mode": topk_value_mode,
            "return_softmax_lse": return_softmax_lse,
            "key_dtype": key_dtype,
            "value_dtype": value_dtype,
        }
    )
    batch_random = NumpyBatchRandomContext.from_case(q, ori_kv, cmp_kv, params)
    params = INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        layout_q,
        layout_kv,
        params,
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
    zero_metadata(metadata)
    case_data = INPUT_ADAPTER.load_golden_store().CASE_DATA
    testcase_name = params.get("testcase_name")
    case_data.put(testcase_name, data)
    INPUT_ADAPTER.load_metadata_protocol().save_metadata_inputs(
        "mixed_quant_sparse_flash_mla", testcase_name, data.get("metadata_input")
    )
    return data


def generate_aclnn_mixed_quant_sparse_flash_mla_inputs(
    q,
    ori_kv,
    cmp_kv,
    ori_sparse_indices,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    cu_seqlens_q,
    cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv,
    seqused_q,
    seqused_ori_kv,
    seqused_cmp_kv,
    cmp_residual_kv,
    ori_topk_length,
    cmp_topk_length,
    sinks,
    metadata,
    quant_mode,
    rope_head_dim,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
    topk_value_mode,
    return_softmax_lse,
    attn_out,
    softmax_lse_out,
    **kwargs,
):
    """Map the ACLNN C signature to the canonical pytest input adapter."""
    del attn_out, softmax_lse_out
    return generate_mixed_quant_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        rope_head_dim=rope_head_dim,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=topk_value_mode,
        return_softmax_lse=return_softmax_lse,
        **kwargs,
    )
