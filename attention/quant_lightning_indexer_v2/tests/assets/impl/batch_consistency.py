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

"""Small batch-consistency protocol shared by LI_V2 and QLI_V2 assets."""

import ast
import hashlib
import math
import random
from numbers import Integral

import numpy as np
import torch


HIFLOAT8_QUANT_MODE = 4
MXFP8_QUANT_MODE = 3
MXFP4_QUANT_MODE = 5
SUPPORTED_QUANT_MODES = (1, 2, 3, 4, 5)
MXFP4_DECODE_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)


class CaseRandomContext:
    """Give batch cases distinct backgrounds without changing normal cases."""

    def __init__(self, attributes):
        fields = tuple(
            attributes.get(name)
            for name in ("batch_axis", "batch_slice_info", "batch_seed")
        )
        self.enabled = all(value is not None for value in fields)
        self.testcase_name = attributes.get("testcase_name", "")
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None

    def __enter__(self):
        if not self.enabled:
            return self
        digest = hashlib.sha256(str(self.testcase_name).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % ((1 << 32) - 1)
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            random.setstate(self.python_state)
            np.random.set_state(self.numpy_state)
            torch.random.set_rng_state(self.torch_state)
        return False


class BatchRelationProtocol:
    """Parse the q-only logical B/S relation contract used by both indexers."""

    def __init__(self, operator_name):
        self.operator_name = operator_name

    @staticmethod
    def relation_slices_overlap(first, second):
        """Return whether two relation samples select the same q output region."""
        first_axes, first_slices = first[0], first[1]
        second_axes, second_slices = second[0], second[1]
        if first_axes != second_axes:
            return False
        first_batch = first_slices[0]
        second_batch = second_slices[0]
        if not BatchRelationProtocol.ranges_overlap(first_batch, second_batch):
            return False
        if first_axes == (0,):
            return True
        first_sequence = first_slices[1]
        second_sequence = second_slices[1]
        return BatchRelationProtocol.ranges_overlap(first_sequence, second_sequence)

    @staticmethod
    def ranges_overlap(first, second):
        """Return whether two positive-step slices share an integer position."""
        first_values = range(*first)
        second_values = range(*second)
        if len(first_values) > len(second_values):
            first_values, second_values = second_values, first_values
        return any(value in second_values for value in first_values)

    def validate_disjoint_relations(self, relations):
        """Reject duplicate or overlapping samples that would self-compare."""
        for index, relation in enumerate(relations):
            for candidate in relations[index + 1 :]:
                if self.relation_slices_overlap(relation, candidate):
                    raise ValueError(
                        f"{self.operator_name} relation samples must not overlap"
                    )

    def parse(self, batch_axis, batch_slice_info, batch_seed):
        fields = (batch_axis, batch_slice_info, batch_seed)
        if any(value is None for value in fields):
            return None
        if not (len(batch_axis) == len(batch_slice_info) == len(batch_seed)):
            raise ValueError(f"{self.operator_name} batch metadata counts differ")
        if not batch_axis or tuple(batch_axis[0]) not in ((0,), (0, 1)):
            raise ValueError(
                f"{self.operator_name} supports q logical axes (0,) or (0, 1)"
            )
        if any(value is not None for value in batch_axis[1:]):
            raise ValueError(f"{self.operator_name} supports q relations only")
        if batch_slice_info[0] is None or batch_seed[0] is None:
            raise ValueError(f"{self.operator_name} requires q slices and q seeds")
        if any(value is not None for value in batch_slice_info[1:]):
            raise ValueError(f"{self.operator_name} supports q relations only")
        if any(value is not None for value in batch_seed[1:]):
            raise ValueError(f"{self.operator_name} supports q relation seeds only")

        axes = tuple(batch_axis[0])
        axis_slices = batch_slice_info[0]
        axis_seeds = batch_seed[0]
        if len(axis_slices) != len(axes) or len(axis_seeds) != len(axes):
            raise ValueError(f"{self.operator_name} q groups do not match q axes")
        sample_count = len(axis_slices[0])
        if not sample_count or any(
            len(values) != sample_count for values in (*axis_slices, *axis_seeds)
        ):
            raise ValueError(
                f"{self.operator_name} q sample counts differ or are empty"
            )

        relations = []
        for sample_index in range(sample_count):
            slices = []
            relation_seed = None
            for axis_group, axis in enumerate(axes):
                value = axis_slices[axis_group][sample_index]
                if not isinstance(value, (tuple, list)) or len(value) != 3:
                    raise ValueError(
                        f"{self.operator_name} invalid q axis {axis} slice: {value!r}"
                    )
                if not all(isinstance(item, Integral) for item in value):
                    raise ValueError(
                        f"{self.operator_name} slices must contain integers"
                    )
                start, stop, step = (int(item) for item in value)
                if step <= 0 or start < 0 or start >= stop:
                    raise ValueError(
                        f"{self.operator_name} slices must be non-empty with positive step"
                    )
                seed = axis_seeds[axis_group][sample_index]
                if not isinstance(seed, Integral):
                    raise ValueError(
                        f"{self.operator_name} batch seed must be an integer"
                    )
                seed = int(seed)
                if relation_seed is not None and seed != relation_seed:
                    raise ValueError(
                        f"{self.operator_name} logical B and S must use the same seed"
                    )
                relation_seed = seed
                slices.append((start, stop, step))
            if axes == (0, 1) and len(range(*slices[0])) != 1:
                raise ValueError(
                    f"{self.operator_name} logical (B,S) requires one B per sample"
                )
            relations.append((axes, tuple(slices), relation_seed))
        self.validate_disjoint_relations(relations)
        return relations


class IndexerBatchInputNormalizer:
    """Materialize equal logical inputs for declared LI/QLI relations."""

    def __init__(
        self,
        data,
        attributes,
        operator_name,
        quantized,
        hifloat8_encoder=None,
    ):
        self.data = data
        self.attributes = attributes
        self.operator_name = operator_name
        self.quantized = quantized
        self.quant_mode = int(attributes.get("quant_mode", 1)) if quantized else None
        self.hifloat8_encoder = hifloat8_encoder
        self.layout_q = attributes.get(
            "layout_q", attributes.get("layout_query", "BSND")
        )
        self.layout_k = attributes.get("layout_k", attributes.get("layout_key", "BSND"))
        self.query = data["query"]
        self.key = data["key"]
        self.weights = data["weights"]
        self.query_scale = data.get("query_dequant_scale")
        self.key_scale = data.get("key_dequant_scale")
        self.offset = data.get("output_idx_offset")
        self.block_table = data.get("block_table")
        self.q_prefix = self.tensor_values(
            data.get("cu_seqlens_query", data.get("cu_seqlens_q"))
        )
        self.k_prefix = self.tensor_values(
            data.get("cu_seqlens_key", data.get("cu_seqlens_k"))
        )
        self.batch_size = self.resolve_batch_size()
        self.q_lengths = self.resolve_lengths("q")
        self.k_lengths = self.resolve_lengths("k")
        self.residual = self.resolve_vector("cmp_residual_k", 0)
        self.assigned_blocks = {}
        self.input_ranges = self.resolve_input_ranges()

    @staticmethod
    def tensor_values(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    def resolve_batch_size(self):
        if self.layout_q == "BSND":
            return int(self.query.shape[0])
        if self.layout_q == "TND" and self.q_prefix is not None:
            return len(self.q_prefix) - 1
        raise ValueError(
            f"{self.operator_name} batch consistency requires BSND or explicit TND prefix"
        )

    def resolve_vector(self, name, default):
        value = self.attributes.get(f"{name}_values")
        if value is None:
            value = self.data.get(name)
        value = self.tensor_values(value)
        if value is None:
            return [default] * self.batch_size
        if len(value) != self.batch_size:
            raise ValueError(
                f"{self.operator_name} {name} length must equal B={self.batch_size}"
            )
        return value

    def resolve_input_ranges(self):
        """Read the exact ranges normalized by the reused pytest generator."""
        params = self.data.get("params")
        if not isinstance(params, (tuple, list)) or len(params) not in (32, 33):
            raise ValueError(f"{self.operator_name} pytest params are unavailable")
        range_start = 25 if len(params) == 33 else 24
        ranges = {
            0: params[range_start],
            10: params[range_start + 1],
            1: params[range_start + 2],
            2: params[range_start + 3],
            11: params[range_start + 4],
            3: params[-1],
        }
        for slot, value in tuple(ranges.items()):
            if value is None:
                ranges.pop(slot)
                continue
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if not isinstance(value, (tuple, list)) or len(value) < 2:
                raise ValueError(
                    f"{self.operator_name} input range for relation slot {slot} is invalid"
                )
            ranges[slot] = (float(value[0]), float(value[1]))
        return ranges

    def resolve_lengths(self, target):
        prefix = self.q_prefix if target == "q" else self.k_prefix
        tensor = self.query if target == "q" else self.key
        layout = self.layout_q if target == "q" else self.layout_k
        if prefix is not None:
            if (
                len(prefix) != self.batch_size + 1
                or prefix[0] != 0
                or prefix[-1] != int(tensor.shape[0])
                or any(right < left for left, right in zip(prefix, prefix[1:]))
            ):
                raise ValueError(
                    f"{self.operator_name} {target} prefix must non-decreasingly span its tensor"
                )
            lengths = [right - left for left, right in zip(prefix, prefix[1:])]
        else:
            if layout == "BSND":
                lengths = [int(tensor.shape[1])] * self.batch_size
            else:
                lengths = self.resolve_vector(f"seqused_{target}", 0)
        actual = self.attributes.get(f"seqused_{target}_values")
        if actual is not None:
            actual = [int(item) for item in actual]
            if len(actual) != self.batch_size:
                raise ValueError(
                    f"{self.operator_name} seqused_{target} length must equal B"
                )
            if any(length < 0 for length in actual):
                raise ValueError(
                    f"{self.operator_name} seqused_{target} must be non-negative"
                )
            if layout in ("BSND", "TND") and any(
                actual_length > physical_length
                for actual_length, physical_length in zip(actual, lengths)
            ):
                raise ValueError(
                    f"{self.operator_name} seqused_{target} exceeds its tensor extent"
                )
            return actual
        return lengths

    @staticmethod
    def derived_seed(seed, relative_batch, slot):
        value = f"{int(seed)}:{int(relative_batch)}:{int(slot)}".encode("ascii")
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (
            (1 << 63) - 1
        )

    @staticmethod
    def e8m0_code_range(data_range):
        lower, upper = data_range
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("E8M0 scale range must be finite")
        if lower <= 0 or lower > upper:
            raise ValueError("E8M0 scale range must satisfy 0 < min <= max")
        low_code = max(0, math.ceil(math.log2(lower)) + 127)
        high_code = min(254, math.floor(math.log2(upper)) + 127)
        while low_code <= 254 and math.ldexp(1.0, low_code - 127) < lower:
            low_code += 1
        while high_code >= 0 and math.ldexp(1.0, high_code - 127) > upper:
            high_code -= 1
        if low_code > high_code:
            raise ValueError("E8M0 scale range contains no representable value")
        return low_code, high_code

    def random_tensor(self, shape, template, seed, relative_batch, slot):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.derived_seed(seed, relative_batch, slot))
        dtype = template.dtype
        if slot not in self.input_ranges:
            raise ValueError(
                f"{self.operator_name} input range for relation slot {slot} is required"
            )
        lower, upper = self.input_ranges[slot]
        if lower > upper:
            raise ValueError(
                f"{self.operator_name} input range must satisfy min <= max"
            )
        if dtype == torch.bool:
            value = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
        elif self.quant_mode == MXFP4_QUANT_MODE and slot in (0, 10):
            valid_codes = torch.nonzero(
                (MXFP4_DECODE_VALUES >= lower) & (MXFP4_DECODE_VALUES <= upper),
                as_tuple=False,
            ).flatten()
            if valid_codes.numel() == 0:
                raise ValueError("MXFP4 input range contains no representable value")
            logical_shape = (*shape[:-1], shape[-1] * 2)
            code_indexes = torch.randint(
                valid_codes.numel(),
                logical_shape,
                generator=generator,
                dtype=torch.int64,
            )
            codes = valid_codes[code_indexes].to(torch.uint8)
            packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
            return packed.view(dtype)
        elif self.quant_mode in (MXFP8_QUANT_MODE, MXFP4_QUANT_MODE) and slot in (
            2,
            11,
        ):
            low_code, high_code = self.e8m0_code_range((lower, upper))
            raw = torch.randint(
                low_code,
                high_code + 1,
                shape,
                generator=generator,
                dtype=torch.uint8,
            )
            return raw.view(dtype)
        elif self.quant_mode == HIFLOAT8_QUANT_MODE and slot in (0, 10):
            if self.hifloat8_encoder is None:
                raise ValueError(
                    f"{self.operator_name} HIFLOAT8 encoder is unavailable"
                )
            value = torch.rand(shape, generator=generator, dtype=torch.float32)
            value = value * (upper - lower) + lower
            return self.hifloat8_encoder(value, round_mode="hybrid", over_mode=True)
        elif dtype.is_floating_point:
            value = torch.rand(shape, generator=generator, dtype=torch.float32)
            value = value * (upper - lower) + lower
        else:
            low = int(np.ceil(lower))
            high = int(np.floor(upper))
            if low > high:
                raise ValueError(
                    f"{self.operator_name} integer input range has no representable value"
                )
            value = torch.randint(
                low, high + 1, shape, generator=generator, dtype=torch.int64
            )
        return value.to(dtype=dtype)

    @staticmethod
    def decode_mxfp4(value):
        packed = value.view(torch.uint8)
        decode_values = MXFP4_DECODE_VALUES.to(device=value.device)
        low = decode_values[(packed & 0x0F).to(torch.long)]
        high = decode_values[(packed >> 4).to(torch.long)]
        return torch.stack((low, high), dim=-1).reshape(
            *value.shape[:-1], value.shape[-1] * 2
        )

    @staticmethod
    def copy_selection(tensor, selector, value):
        if tensor is None:
            return
        source = value
        if torch.is_tensor(source) and "float4" in str(source.dtype):
            if "float4" in str(tensor.dtype):
                target = tensor[selector].view(torch.uint8)
                target.copy_(source.view(torch.uint8).to(device=tensor.device))
                return
            source = IndexerBatchInputNormalizer.decode_mxfp4(source)
        tensor[selector].copy_(source.to(dtype=tensor.dtype, device=tensor.device))

    def query_selector(self, batch_index, sequence_slice):
        if self.layout_q == "BSND":
            start, stop, step = (0, self.q_lengths[batch_index], 1)
            if sequence_slice is not None:
                start, stop, step = sequence_slice
            return (batch_index, slice(start, stop, step)), len(
                range(start, stop, step)
            )
        token_start = self.q_prefix[batch_index]
        token_stop = self.q_prefix[batch_index + 1]
        step = 1
        if sequence_slice is not None:
            token_start += sequence_slice[0]
            token_stop = self.q_prefix[batch_index] + sequence_slice[1]
            step = sequence_slice[2]
        return (slice(token_start, token_stop, step),), len(
            range(token_start, token_stop, step)
        )

    def query_capacity(self, batch_index):
        """Return the physical Q span represented by one logical batch."""
        if self.layout_q == "BSND":
            return int(self.query.shape[1])
        return self.q_prefix[batch_index + 1] - self.q_prefix[batch_index]

    def query_comparison_length(self, batch_index):
        """Match the output span selected by the phase-two comparator."""
        if self.layout_q == "TND":
            return self.query_capacity(batch_index)
        return self.q_lengths[batch_index]

    def validate_relations(self, relations):
        mask_mode = int(
            self.attributes.get("mask_mode", self.attributes.get("sparse_mode", 0))
        )
        grouped_signatures = {}
        occupied = []
        for axes, slices, seed in relations:
            batch_start, batch_stop, batch_step = slices[0]
            if batch_stop > self.batch_size:
                raise ValueError(
                    f"{self.operator_name} logical B slice exceeds B={self.batch_size}"
                )
            batch_indices = range(batch_start, batch_stop, batch_step)
            sequence_slice = slices[1] if axes == (0, 1) else None
            if sequence_slice is not None and mask_mode != 0:
                raise ValueError(
                    f"{self.operator_name} shifted logical S relations require mask_mode=0"
                )
            if (
                sequence_slice is None
                and len(
                    {self.query_capacity(batch_index) for batch_index in batch_indices}
                )
                != 1
            ):
                raise ValueError(
                    f"{self.operator_name} one B-only relation requires equal q output spans"
                )
            signature = []
            for batch_index in batch_indices:
                selector, q_count = self.query_selector(batch_index, sequence_slice)
                if (
                    sequence_slice is not None
                    and sequence_slice[1] > self.q_lengths[batch_index]
                ):
                    raise ValueError(
                        f"{self.operator_name} logical S slice exceeds effective q length"
                    )
                effective_q_count = (
                    q_count
                    if sequence_slice is not None
                    else self.q_lengths[batch_index]
                )
                if effective_q_count == 0:
                    raise ValueError(
                        f"{self.operator_name} relation selects no q output elements"
                    )
                occupied.append((selector, seed))
                signature.append(
                    (
                        q_count
                        if sequence_slice is not None
                        else self.query_comparison_length(batch_index),
                        effective_q_count,
                        self.k_lengths[batch_index],
                        self.residual[batch_index],
                    )
                )
            relation_size = tuple(len(range(*value)) for value in slices)
            key = (axes, seed, relation_size)
            value = tuple(signature)
            previous = grouped_signatures.setdefault(key, value)
            if previous != value:
                raise ValueError(
                    f"{self.operator_name} relation requires equal q output spans, "
                    "effective q lengths, K lengths and residuals"
                )

        for index, (left, left_seed) in enumerate(occupied):
            for right, right_seed in occupied[index + 1 :]:
                if left_seed == right_seed or len(left) != len(right):
                    continue
                if self.selectors_overlap(left, right):
                    raise ValueError(
                        f"{self.operator_name} relations with different seeds overlap"
                    )

    @staticmethod
    def selectors_overlap(left, right):
        for left_item, right_item in zip(left, right):
            if isinstance(left_item, int) or isinstance(right_item, int):
                if left_item != right_item:
                    return False
                continue
            left_range = range(left_item.start, left_item.stop, left_item.step or 1)
            right_range = range(right_item.start, right_item.stop, right_item.step or 1)
            if not BatchRelationProtocol.ranges_overlap(
                (left_range.start, left_range.stop, left_range.step),
                (right_range.start, right_range.stop, right_range.step),
            ):
                return False
        return True

    def query_references(self, name):
        references = [self.data.get(name)]
        state = self.data.get("golden_state", {}).get("forward_inputs", {})
        references.append(state.get(name))
        return [value for value in references if value is not None]

    def fill_query_inputs(self, batch_index, sequence_slice, seed, relative_batch):
        selector, _count = self.query_selector(batch_index, sequence_slice)
        targets = (
            (self.query_references("query"), 0),
            (self.query_references("weights"), 1),
            (self.query_references("output_idx_offset"), 3),
        )
        if self.quant_mode != HIFLOAT8_QUANT_MODE:
            targets += ((self.query_references("query_dequant_scale"), 2),)
        for references, slot in targets:
            if not references:
                continue
            value = self.random_tensor(
                tuple(references[0][selector].shape),
                references[0],
                seed,
                relative_batch,
                slot,
            )
            for tensor in references:
                self.copy_selection(tensor, selector, value)

    def key_references(self, name):
        references = []
        if name == "key":
            references.append(self.data.get("cpu_key"))
        state = self.data.get("golden_state", {}).get("forward_inputs", {})
        references.append(state.get(name))
        return [value for value in references if value is not None]

    def input_references(self, name):
        state = self.data.get("golden_state", {}).get("forward_inputs", {})
        return [
            value
            for value in (self.data.get(name), state.get(name))
            if value is not None
        ]

    def fill_hifloat8_scales(self, seed):
        """Use stable global scales because mode 4 scale inputs have shape ``(1,)``."""
        for name, slot in (("query_dequant_scale", 2), ("key_dequant_scale", 11)):
            references = self.input_references(name)
            if not references:
                continue
            value = self.random_tensor((1,), references[0], seed, 0, slot).item()
            for tensor in references:
                if torch.is_tensor(tensor):
                    tensor.fill_(value)
                else:
                    np.asarray(tensor).fill(value)

    def scatter_paged(self, tensor, batch_index, value, seed, relative_batch):
        table = self.tensor_values(self.block_table[batch_index])
        block_size = int(tensor.shape[1])
        copied = 0
        owner = (seed, relative_batch)
        for block_id in table:
            if block_id < 0 or copied >= value.shape[0]:
                continue
            count = min(block_size, int(value.shape[0]) - copied)
            assignment = self.assigned_blocks.setdefault(block_id, owner)
            if assignment != owner:
                raise ValueError(
                    f"{self.operator_name} paged relations share block {block_id} "
                    "between different logical batches"
                )
            self.copy_selection(
                tensor,
                (block_id, slice(0, count, 1)),
                value[copied : copied + count],
            )
            copied += count
        if copied != value.shape[0]:
            raise ValueError(
                f"{self.operator_name} block table has insufficient capacity"
            )

    def fill_key_tensor(self, name, batch_index, seed, relative_batch, slot):
        tensor = self.data.get(name)
        if tensor is None:
            return
        key_length = self.k_lengths[batch_index]
        if self.layout_k == "BSND":
            selector = (batch_index, slice(0, key_length, 1))
            shape = tuple(tensor[selector].shape)
        elif self.layout_k == "TND":
            start, stop = self.k_prefix[batch_index : batch_index + 2]
            selector = (slice(start, stop, 1),)
            shape = tuple(tensor[selector].shape)
        elif self.layout_k == "PA_BBND":
            if self.block_table is None:
                raise ValueError(f"{self.operator_name} PA_BBND requires block_table")
            selector = None
            shape = (key_length, *tuple(tensor.shape[2:]))
        else:
            raise ValueError(
                f"{self.operator_name} unsupported key layout {self.layout_k!r}"
            )
        value = self.random_tensor(shape, tensor, seed, relative_batch, slot)
        if selector is None:
            self.scatter_paged(tensor, batch_index, value, seed, relative_batch)
        else:
            self.copy_selection(tensor, selector, value)

        for reference in self.key_references(name):
            if reference is tensor:
                continue
            if self.layout_k == "PA_BBND":
                permutation = (1, 0, *range(2, value.ndim))
                source = value
                if "float4" in str(source.dtype):
                    source = self.decode_mxfp4(source)
                reference[batch_index, :, :key_length].copy_(
                    source.permute(permutation)
                )
            else:
                self.copy_selection(reference, selector, value)

    def apply(self, relations):
        self.validate_relations(relations)
        if self.quantized and self.quant_mode not in SUPPORTED_QUANT_MODES:
            raise ValueError(
                f"{self.operator_name} batch consistency supports quant_mode 1 through 5"
            )
        for axes, slices, seed in relations:
            batch_start, batch_stop, batch_step = slices[0]
            sequence_slice = slices[1] if axes == (0, 1) else None
            for relative_batch, batch_index in enumerate(
                range(batch_start, batch_stop, batch_step)
            ):
                self.fill_query_inputs(
                    batch_index, sequence_slice, seed, relative_batch
                )
                self.fill_key_tensor("key", batch_index, seed, relative_batch, 10)
                if self.quant_mode != HIFLOAT8_QUANT_MODE:
                    self.fill_key_tensor(
                        "key_dequant_scale",
                        batch_index,
                        seed,
                        relative_batch,
                        11,
                    )
        if self.quant_mode == HIFLOAT8_QUANT_MODE:
            self.fill_hifloat8_scales(relations[0][2])


def normalize_indexer_inputs(
    data,
    attributes,
    operator_name,
    quantized=False,
    hifloat8_encoder=None,
):
    protocol = BatchRelationProtocol(operator_name)
    relations = protocol.parse(
        attributes.get("batch_axis"),
        attributes.get("batch_slice_info"),
        attributes.get("batch_seed"),
    )
    if relations is not None:
        IndexerBatchInputNormalizer(
            data,
            attributes,
            operator_name,
            quantized,
            hifloat8_encoder,
        ).apply(relations)
