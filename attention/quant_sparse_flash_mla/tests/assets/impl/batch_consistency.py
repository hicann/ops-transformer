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

"""Batch-consistency input randomization for QuantSparseFlashMla TTK cases."""

import hashlib
import random
from bisect import bisect_right

import torch


class QuantSparseFlashMlaBatchRandomContext:
    """Map logical B/S relations to QSMLA pytest random tensors."""

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
        if self.q_prefix is not None:
            self.register_extent(
                self.q_prefix[-1],
                [
                    (selector[0], relation[2])
                    for selector, relation in zip(self.query_selectors, self.relations)
                ],
                "q",
            )
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
        self.original_torch_rand = None
        self.original_torch_randperm = None
        self.original_python_uniform = None

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
        if any(field is None for field in fields):
            # Level-3 deterministic execution does not imply a batch relation.
            return None
        layout_q = kwargs.get("layout_q", "BSND")
        layout_kv = kwargs.get("layout_kv", "BSND")
        if layout_q not in ("BSND", "TND"):
            raise ValueError(
                f"QSMLA batch consistency does not support layout_q={layout_q!r}"
            )
        if layout_kv not in ("BSND", "TND", "PA_BBND"):
            raise ValueError(
                f"QSMLA batch consistency does not support layout_kv={layout_kv!r}"
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
                    f"QSMLA batch consistency requires explicit {name}_values"
                )
            return None
        if len(value) < 2 or value[0] != 0 or value[-1] != expected_total:
            raise ValueError(
                f"QSMLA {name}_values must start at 0 and end at {expected_total}: {value!r}"
            )
        if any(right <= left for left, right in zip(value, value[1:])):
            raise ValueError(
                f"QSMLA {name}_values must be strictly increasing for batch relations"
            )
        return value

    @staticmethod
    def parse_slice(value, extent, label):
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError(f"QSMLA invalid {label} slice: {value!r}")
        if not all(isinstance(item, int) for item in value):
            raise ValueError(f"QSMLA {label} slice must contain integers: {value!r}")
        start, stop, step = (int(item) for item in value)
        if step != 1 or start < 0 or start >= stop or stop > extent:
            raise ValueError(
                f"QSMLA {label} slice must be in-range, non-empty and contiguous: {value!r}"
            )
        return start, stop, step

    def parse_relations(self, kwargs):
        batch_axis = kwargs["batch_axis"]
        batch_slices = kwargs["batch_slice_info"]
        batch_seed = kwargs["batch_seed"]
        if not batch_axis or tuple(batch_axis[0]) not in ((0,), (0, 1)):
            raise ValueError(
                "QSMLA q batch_axis must use logical BSND axes (0,) or (0, 1)"
            )
        if not batch_slices or batch_slices[0] is None or batch_seed[0] is None:
            raise ValueError("QSMLA batch consistency requires slices and seeds on q")
        if not (len(batch_axis) == len(batch_slices) == len(batch_seed)):
            raise ValueError("QSMLA batch metadata top-level counts differ")
        if any(value is not None for value in batch_slices[1:]):
            raise ValueError(
                "QSMLA batch consistency relations must be declared on q only"
            )
        if any(value is not None for value in batch_seed[1:]):
            raise ValueError("QSMLA batch consistency seeds must be declared on q only")

        axes = tuple(batch_axis[0])
        axis_slices = batch_slices[0]
        axis_seeds = batch_seed[0]
        if len(axis_slices) != len(axes) or len(axis_seeds) != len(axes):
            raise ValueError("QSMLA q slice/seed groups must match q logical axes")
        sample_count = len(axis_slices[0])
        if sample_count == 0:
            raise ValueError("QSMLA batch consistency requires at least one q slice")
        if any(len(values) != sample_count for values in (*axis_slices, *axis_seeds)):
            raise ValueError("QSMLA q axis groups must contain the same sample count")

        relations = []
        for index in range(sample_count):
            batch_slice = self.parse_slice(
                axis_slices[0][index], self.batch_size, "logical B"
            )
            seed = axis_seeds[0][index]
            if not isinstance(seed, int):
                raise ValueError("QSMLA batch seed must be an integer")
            sequence_slice = None
            if axes == (0, 1):
                if axis_seeds[1][index] != seed:
                    raise ValueError(
                        "QSMLA logical B and S slices must use the same seed"
                    )
                if batch_slice[1] - batch_slice[0] != 1:
                    raise ValueError(
                        "QSMLA logical (B,S) relation requires one B per sample"
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
                "QSMLA batch consistency supports one relation seed per case"
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
                        "QSMLA relation samples must not overlap logical B positions"
                    )
                if candidate_sequence is None or not (
                    sequence_slice[1] <= candidate_sequence[0]
                    or candidate_sequence[1] <= sequence_slice[0]
                ):
                    raise ValueError(
                        "QSMLA relation samples must not overlap logical q positions"
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
                    f"QSMLA {name}_values length must equal B={self.batch_size}"
                )
        for name, value in prefixes:
            if len(value) != self.batch_size + 1:
                raise ValueError(f"QSMLA {name}_values length must equal B + 1")

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
                    "QSMLA relation slices must contain the same logical batch count"
                )
            sequence_slice = relation[1]
            sequence_count = (
                sequence_slice[1] - sequence_slice[0]
                if sequence_slice is not None
                else None
            )
            if sequence_count != reference_sequence_count:
                raise ValueError(
                    "QSMLA relation slices must contain the same logical S count"
                )
            if relation_signature(batch_slice) != reference_signature:
                raise ValueError(
                    "QSMLA relation slices require identical q/KV lengths and residual values"
                )

    def register_extent(self, extent, relations, source):
        selectors = tuple(value for value, _ in relations)
        existing = self.extent_selectors.get(int(extent))
        if existing is not None and existing[0] != selectors:
            raise ValueError(
                f"QSMLA cannot distinguish generated {source} extent {extent} from "
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
            raise ValueError(f"QSMLA batch consistency does not support mode {mode!r}")
        if int(params.get("quant_mode")) != 1:
            raise ValueError(
                "QSMLA batch consistency currently supports quant_mode=1 only"
            )
        for name in ("ori_sparse_indices_mode", "cmp_sparse_indices_mode"):
            value = params.get(name)
            if value is not None and value != "full":
                raise ValueError(
                    "QSMLA batch consistency currently requires full sparse indices"
                )
        for name in ("ori_kv_topk_mode", "cmp_kv_topk_mode"):
            value = params.get(name)
            if value is not None and value not in ("fullK", "no"):
                raise ValueError(
                    "QSMLA batch consistency currently excludes random topk lengths"
                )

        self.validate_sequence_masks(params)

        if mode in ("CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"):
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
                "QSMLA logical S slices at different positions require no-mask mode: "
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
            raise ValueError("QSMLA sparse batch q lengths must contain B values")

        calls_per_batch = [
            int(length) * int(params["N2"]) for length in sequence_lengths
        ]
        offsets = [0]
        for count in calls_per_batch:
            offsets.append(offsets[-1] + count)
        if offsets[-1] <= 0:
            raise ValueError("QSMLA sparse batch requires a non-empty q relation")
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
                                "QSMLA sparse relation slices overlap ambiguously"
                            )
                        self.randperm_relation_keys[key] = relation_key

    @classmethod
    def derive_seed(cls, seed, call_index):
        return (int(seed) + (call_index + 1) * 1000003) % cls.SEED_MODULUS

    @staticmethod
    def create_generator(seed, device):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def torch_rand(self, *size, **kwargs):
        call_index = self.call_index
        self.call_index += 1
        call_kwargs = dict(kwargs)
        device = call_kwargs.get("device") or "cpu"
        requested_rank = (
            len(size[0])
            if len(size) == 1 and isinstance(size[0], (tuple, list))
            else len(size)
        )
        seed = self.base_seed if requested_rank >= 2 else self.relation_seed
        call_kwargs["generator"] = self.create_generator(
            self.derive_seed(seed, call_index), device
        )
        value = self.original_torch_rand(*size, **call_kwargs)
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
            piece_generator = self.create_generator(
                self.derive_seed(seed, call_index), value.device
            )
            value[selector] = self.original_torch_rand(
                tuple(piece.shape),
                generator=piece_generator,
                dtype=value.dtype,
                device=value.device,
            )
        return value

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
        call_kwargs["generator"] = self.create_generator(
            self.derive_seed(seed, seed_index), device
        )
        return self.original_torch_randperm(n, **call_kwargs)

    def normalize_block_tables(self, data):
        if self.layout_kv != "PA_BBND":
            return
        op_input = data.get("op_input", {})
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

    def python_uniform(self, low, high):
        call_index = self.call_index
        self.call_index += 1
        rng = random.Random(self.derive_seed(self.relation_seed, call_index))
        return rng.uniform(low, high)

    def __enter__(self):
        self.original_torch_rand = torch.rand
        self.original_torch_randperm = torch.randperm
        self.original_python_uniform = random.uniform
        torch.rand = self.torch_rand
        torch.randperm = self.torch_randperm
        random.uniform = self.python_uniform
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.rand = self.original_torch_rand
        torch.randperm = self.original_torch_randperm
        random.uniform = self.original_python_uniform
        return False
