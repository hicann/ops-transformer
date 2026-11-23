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

import numpy
import torch

__spec__ = {
    "moe_finalize_routing_v2": "MoeFinalizeRoutingV2TestSpec",
    "aclnnMoeFinalizeRoutingV2": "AclnnMoeFinalizeRoutingV2TestSpec",
    "aclnnMoeFinalizeRoutingV5": "AclnnMoeFinalizeRoutingV5TestSpec",
    "torch.ops.cann_ops_transformer.moe_finalize_routing": "TorchMoeFinalizeRoutingTestSpec",
}


def _to_numpy(value):
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu()
        try:
            return value.numpy()
        except TypeError:
            import ml_dtypes

            return value.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
    return value


def _restore_type(value, template):
    if (
        template is not None
        and hasattr(template, "cpu")
        and hasattr(template, "numpy")
        and value is not None
    ):
        try:
            return torch.from_numpy(value)
        except TypeError:
            return torch.from_numpy(value.astype(numpy.float32)).to(torch.bfloat16)
    return value


# Default to local comparison; --compare cross_check selects the L2 preset.
_TOLERANCE = {
    "float16": {"standard": "stat_rel_err", "level": "L2"},
    "bfloat16": {"standard": "stat_rel_err", "level": "L2"},
    "float32": {"standard": "stat_rel_err", "level": "L2"},
}


_E2M1_CODE_MASK = 0xF
_E2M1_MAGNITUDE_MASK = 0x7
_E2M1_SIGN_MASK = 0x8
_E2M1_MANTISSA_BITS = 1
_E2M1_MANTISSA_MASK = 0x1
_E2M1_EXPONENT_BIAS = 1
_E2M1_MANTISSA_UNIT = 0.5
_E8M0_EXPONENT_BIAS = 127
_E8M0_NAN_CODE = 0xFF
_MXFP4_SCALE_GROUP_SIZE = 32
_FIRST_ROW_ROUTING_MODE = 2
_ROUTING_MODE_COUNT = 4
_EXPERT_RANGE_SIZE = 2
_DISABLED_EXPERT_RANGE = (-1, -1)
_MXFP4_TEST_SEED = 20261112
_MXFP4_TEST_SCALE_EXPONENT_RADIUS = 4
_MXFP4_TEST_MIN_SCALE_STRIDE = 97
_MXFP4_TEST_NEXT_SCALE_STRIDE = 101


class _MoeFinalizeRoutingV2TorchImpl:
    """Independent Torch reference used by the remote GPU cross-check."""

    def __call__(
        self,
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        expanded_x_scale=None,
        *,
        drop_pad_mode=0,
        zero_expert_range=None,
        copy_expert_range=None,
        constant_expert_range=None,
        k=1,
        **kwargs,
    ):
        zero_expert_range = zero_expert_range or []
        copy_expert_range = copy_expert_range or []
        constant_expert_range = constant_expert_range or []
        if str(expanded_x.dtype) == "float4_e2m1":
            return self._mxfp4(
                expanded_x,
                expanded_row_idx,
                x1,
                x2,
                bias,
                scales,
                expanded_x_scale,
                expert_idx,
                drop_pad_mode,
                k,
                x,
                alpha1,
                alpha2,
                v,
                zero_expert_range,
                copy_expert_range,
                constant_expert_range,
            )
        output_dtype = expanded_x.dtype
        compute_dtype = (
            torch.float64 if output_dtype == torch.float32 else torch.float32
        )
        bsk = expanded_row_idx.shape[0]
        hidden_size = expanded_x.shape[-1]
        if hidden_size == 0:
            return [torch.empty((0,), dtype=output_dtype, device=expanded_x.device)]

        expanded_x_2d = expanded_x.reshape(-1, hidden_size)
        topk = scales.shape[1] if scales is not None else int(k)
        num_rows = bsk // topk
        out = torch.zeros(
            (num_rows, hidden_size), dtype=compute_dtype, device=expanded_x.device
        )
        if x1 is not None:
            out += x1.to(compute_dtype)
        if x2 is not None:
            out += x2.to(compute_dtype)

        for row in range(num_rows):
            for topk_idx in range(topk):
                if drop_pad_mode in (0, 1):
                    route_pos = topk_idx * num_rows + row
                else:
                    route_pos = row * topk + topk_idx
                expanded_idx = int(expanded_row_idx[route_pos].item())
                if drop_pad_mode in (1, 3):
                    if expanded_idx == -1:
                        continue
                elif expanded_idx >= expanded_x_2d.shape[0]:
                    continue

                if x is not None and expert_idx is not None:
                    zero_start, zero_end = zero_expert_range
                    copy_start, copy_end = copy_expert_range
                    const_start, const_end = constant_expert_range
                    expert_id = int(expert_idx[row, topk_idx].item())
                    if zero_start <= expert_id < zero_end:
                        continue
                    if copy_start <= expert_id < copy_end:
                        dst_row = x[row].to(compute_dtype)
                    elif (
                        v is not None
                        and alpha1 is not None
                        and alpha2 is not None
                        and const_start <= expert_id < const_end
                    ):
                        const_row = expert_id - const_start
                        dst_row = (
                            alpha1[const_row] * x[row]
                            + alpha2[const_row] * v[const_row]
                        ).to(compute_dtype)
                    else:
                        dst_row = expanded_x_2d[expanded_idx].to(compute_dtype)
                else:
                    dst_row = expanded_x_2d[expanded_idx].to(compute_dtype)

                if bias is not None and expert_idx is not None:
                    expert_id = int(expert_idx[row, topk_idx].item())
                    if expert_id < 0 or expert_id >= bias.shape[0]:
                        continue
                    dst_row = dst_row + bias[expert_id].to(compute_dtype)
                if scales is not None:
                    dst_row = dst_row * scales[row, topk_idx].to(compute_dtype)
                out[row] += dst_row

        return [out.to(output_dtype)]

    @staticmethod
    def _mxfp4(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expanded_x_scale,
        expert_idx,
        drop_pad_mode,
        k,
        x,
        alpha1,
        alpha2,
        v,
        zero_range,
        copy_range,
        constant_range,
    ):
        # The remote transport keeps custom FP4/E8M0 arrays as NumPy byte
        # storage. Decode and compute with independent Torch ops on the GPU.
        device = expanded_row_idx.device
        width = expanded_x.shape[-1]
        rows = expanded_x.size // width
        num_rows = expanded_row_idx.numel() // int(k)
        codes = (
            torch.from_numpy(numpy.ascontiguousarray(expanded_x).view(numpy.uint8))
            .to(device=device)
            .reshape(rows, width)
            .long()
            & _E2M1_CODE_MASK
        )
        # Decode sign/exponent/mantissa, independently of the CPU golden LUT.
        magnitude = codes & _E2M1_MAGNITUDE_MASK
        exponent = magnitude >> _E2M1_MANTISSA_BITS
        mantissa = (magnitude & _E2M1_MANTISSA_MASK).float()
        values = torch.where(
            exponent == 0,
            mantissa * _E2M1_MANTISSA_UNIT,
            torch.ldexp(
                1.0 + mantissa * _E2M1_MANTISSA_UNIT,
                exponent.int() - _E2M1_EXPONENT_BIAS,
            ),
        )
        values = torch.where((codes & _E2M1_SIGN_MASK) != 0, -values, values)
        scale_codes = (
            torch.from_numpy(
                numpy.ascontiguousarray(expanded_x_scale).view(numpy.uint8)
            )
            .to(device=device)
            .reshape(rows, -1)
        )
        factors = torch.ldexp(
            torch.ones_like(scale_codes, dtype=torch.float32),
            scale_codes.int() - _E8M0_EXPONENT_BIAS,
        )
        factors = torch.where(scale_codes == _E8M0_NAN_CODE, float("nan"), factors)
        columns = torch.arange(width, device=device) // _MXFP4_SCALE_GROUP_SIZE
        decoded = values * factors[:, columns]
        routes = (
            expanded_row_idx.reshape(int(k), num_rows).T
            if drop_pad_mode < _FIRST_ROW_ROUTING_MODE
            else expanded_row_idx.reshape(num_rows, int(k))
        )
        out = torch.zeros((num_rows, width), dtype=torch.float32, device=device)
        for residual in (x1, x2):
            if residual is not None:
                out = out + residual.float()
        for branch in range(int(k)):
            physical = routes[:, branch].long()
            valid = (physical >= 0) & (physical < rows)
            value = decoded[physical.clamp(0, rows - 1)]
            if expert_idx is not None:
                expert = expert_idx[:, branch].long()
                for bounds, kind in (
                    (zero_range, "zero"),
                    (copy_range, "copy"),
                    (constant_range, "constant"),
                ):
                    if len(bounds) != _EXPERT_RANGE_SIZE or bounds[0] < 0:
                        continue
                    selected = (expert >= bounds[0]) & (expert < bounds[1])
                    if kind == "zero":
                        valid = valid & ~selected
                    elif x is not None and kind == "copy":
                        value = torch.where(selected[:, None], x.float(), value)
                    elif x is not None and all(
                        t is not None for t in (alpha1, alpha2, v)
                    ):
                        ids = (expert - bounds[0]).clamp(0, alpha1.shape[0] - 1)
                        constant = (
                            alpha1[ids].float() * x.float()
                            + alpha2[ids].float() * v[ids].float()
                        )
                        value = torch.where(selected[:, None], constant, value)
            if bias is not None:
                expert = expert_idx[:, branch].long()
                valid = valid & (expert >= 0) & (expert < bias.shape[0])
                value = value + bias[expert.clamp(0, bias.shape[0] - 1)].float()
            if scales is not None:
                value = value * scales[:, branch, None].float()
            out = out + torch.where(valid[:, None], value, 0.0)
        return [out.to(torch.bfloat16)]


class _AclnnMoeFinalizeRoutingV2TorchImpl(_MoeFinalizeRoutingV2TorchImpl):
    def __call__(
        self,
        expandedX,
        expandedRowIdx,
        x1Optional=None,
        x2Optional=None,
        biasOptional=None,
        scalesOptional=None,
        expertIdxOptional=None,
        out=None,
        *,
        dropPadMode=0,
        k=1,
        **kwargs,
    ):
        return super().__call__(
            expandedX,
            expandedRowIdx,
            x1Optional,
            x2Optional,
            biasOptional,
            scalesOptional,
            expertIdxOptional,
            drop_pad_mode=dropPadMode,
            k=k,
        )


class MoeFinalizeRoutingV2TestSpec:
    third_party = {"torch": _MoeFinalizeRoutingV2TorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        expanded_x_scale=None,
        *,
        drop_pad_mode=0,
        zero_expert_range=None,
        copy_expert_range=None,
        constant_expert_range=None,
        k=1,
        **kwargs,
    ):
        if str(expanded_x.dtype) == "float4_e2m1":
            import ml_dtypes

            h = expanded_x.shape[-1]
            rows = expanded_x.size // h
            n = expanded_row_idx.size // k
            codes = expanded_x.view(numpy.uint8).reshape(rows, h) & _E2M1_CODE_MASK
            lut = numpy.array(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                numpy.float32,
            )
            # Each physical row has one E8M0 scale per 32 features.
            scale_bytes = expanded_x_scale.view(numpy.uint8).reshape(rows, -1)
            factors = numpy.exp2(
                scale_bytes.astype(numpy.float32) - numpy.float32(_E8M0_EXPONENT_BIAS)
            )
            factors[scale_bytes == _E8M0_NAN_CODE] = numpy.nan
            decoded = (
                lut[codes] * factors[:, numpy.arange(h) // _MXFP4_SCALE_GROUP_SIZE]
            )
            routes = (
                expanded_row_idx.reshape(k, n).T
                if drop_pad_mode < _FIRST_ROW_ROUTING_MODE
                else expanded_row_idx.reshape(n, k)
            )
            out = numpy.zeros((n, h), numpy.float32)
            for aux in (x1, x2):
                if aux is not None:
                    out += aux.astype(numpy.float32)
            for row in range(n):
                for branch in range(k):
                    physical = routes[row, branch]
                    if not 0 <= physical < rows:
                        continue
                    value = decoded[physical]
                    if expert_idx is not None:
                        expert = expert_idx[row, branch]
                        if (
                            zero_expert_range
                            and zero_expert_range[0] <= expert < zero_expert_range[1]
                        ):
                            continue
                        if (
                            x is not None
                            and copy_expert_range
                            and copy_expert_range[0] <= expert < copy_expert_range[1]
                        ):
                            value = x[row].astype(numpy.float32)
                        elif (
                            x is not None
                            and constant_expert_range
                            and constant_expert_range[0]
                            <= expert
                            < constant_expert_range[1]
                            and all(t is not None for t in (alpha1, alpha2, v))
                        ):
                            idx = expert - constant_expert_range[0]
                            value = alpha1[idx].astype(numpy.float32) * x[row].astype(
                                numpy.float32
                            ) + alpha2[idx].astype(numpy.float32) * v[idx].astype(
                                numpy.float32
                            )
                    if bias is not None:
                        expert = expert_idx[row, branch]
                        if not 0 <= expert < len(bias):
                            continue
                        value = value + bias[expert].astype(numpy.float32)
                    if scales is not None:
                        value = value * numpy.float32(scales[row, branch])
                    out[row] += value
            return [out.astype(ml_dtypes.bfloat16)]

        if zero_expert_range is None:
            zero_expert_range = []
        if copy_expert_range is None:
            copy_expert_range = []
        if constant_expert_range is None:
            constant_expert_range = []

        ori_dtype = expanded_x.dtype
        out_type = numpy.float32
        if expanded_x.dtype == numpy.float32:
            out_type = numpy.float64

        bsk = expanded_row_idx.shape[0]
        h = expanded_x.shape[-1]
        if h == 0:
            return [numpy.array([], dtype=out_type)]
        expanded_x = expanded_x.reshape(-1, h)
        K = k
        if scales is not None:
            K = scales.shape[1]
        num_rows = bsk // K

        out = numpy.zeros((num_rows, h), dtype=out_type)
        if x1 is not None:
            out = out + x1
        if x2 is not None:
            out = out + x2

        for i in range(num_rows):
            for kk in range(K):
                if drop_pad_mode == 0 or drop_pad_mode == 1:
                    expanded_row_idx_idx = kk * num_rows + i
                else:
                    expanded_row_idx_idx = i * K + kk
                expanded_row_idx_value = expanded_row_idx[expanded_row_idx_idx]
                if drop_pad_mode == 1 or drop_pad_mode == 3:
                    if expanded_row_idx_value == -1:
                        continue
                else:
                    if expanded_row_idx_value >= expanded_x.shape[0]:
                        continue

                if x is not None and expert_idx is not None:
                    zero_start, zero_end = zero_expert_range
                    copy_start, copy_end = copy_expert_range
                    const_start, const_end = constant_expert_range
                    expert_id = expert_idx[i, kk]
                    if zero_start <= expert_id < zero_end:
                        continue
                    elif copy_start <= expert_id < copy_end:
                        dst_row = x[i, :].astype(out_type)
                    elif (
                        x is not None
                        and v is not None
                        and alpha1 is not None
                        and alpha2 is not None
                        and const_start <= expert_id < const_end
                    ):
                        row_idx = expert_id - const_start
                        dst_row = (
                            alpha1[row_idx, :] * x[i, :]
                            + alpha2[row_idx, :] * v[row_idx, :]
                        ).astype(out_type)
                    else:
                        dst_row = expanded_x[
                            expanded_row_idx[expanded_row_idx_idx], :
                        ].astype(out_type)
                else:
                    dst_row = expanded_x[
                        expanded_row_idx[expanded_row_idx_idx], :
                    ].astype(out_type)
                if bias is not None and expert_idx is not None:
                    expert_id = expert_idx[i, kk]
                    if expert_id < 0 or expert_id >= bias.shape[0]:
                        continue
                    dst_row = dst_row + bias[expert_id, :].astype(out_type)
                if scales is not None:
                    dst_row = dst_row * scales[i, kk].astype(out_type)
                out[i, :] = out[i, :] + dst_row

        if ori_dtype == numpy.float32:
            return [out.astype(numpy.float32)]
        return [out.astype(ori_dtype)]

    @staticmethod
    def customize_inputs(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        expanded_x_scale=None,
        **kwargs,
    ):
        if str(expanded_x.dtype) == "float4_e2m1":
            # Reproduce the shared 16 scenarios without external data files.
            profiles = ("resident", "kh_full", "k_batch", "h_partial")
            name = kwargs.get("testcase_name", "")
            mode = kwargs.get("drop_pad_mode", 0)
            profile = next(
                (
                    i
                    for i, label in enumerate(profiles)
                    if name.startswith("mx4_" + label + "_")
                ),
                0,
            )
            rng = numpy.random.default_rng(
                _MXFP4_TEST_SEED + profile * _ROUTING_MODE_COUNT + mode
            )
            h = expanded_x.shape[-1]
            rows = expanded_x.size // h
            k = kwargs.get("k", 1)
            n = expanded_row_idx.size // k
            expanded_x.view(numpy.uint8)[...] = (
                (
                    numpy.arange(expanded_x.size, dtype=numpy.uint32)
                    % (_E2M1_CODE_MASK + 1)
                )
                .astype(numpy.uint8)
                .reshape(expanded_x.shape)
            )
            scale_bytes = rng.integers(
                _E8M0_EXPONENT_BIAS - _MXFP4_TEST_SCALE_EXPONENT_RADIUS,
                _E8M0_EXPONENT_BIAS + _MXFP4_TEST_SCALE_EXPONENT_RADIUS,
                expanded_x_scale.shape,
                dtype=numpy.uint8,
            )
            scale_bytes.flat[::_MXFP4_TEST_MIN_SCALE_STRIDE] = 0
            scale_bytes.flat[1::_MXFP4_TEST_NEXT_SCALE_STRIDE] = 1
            expanded_x_scale.view(numpy.uint8)[...] = scale_bytes
            routes = rng.integers(0, rows, (n, k), dtype=numpy.int32)
            if n > 0:
                routes[0, -1] = -1
                routes[-1] = -1
                # Keep a valid route even when the first and last tokens coincide.
                routes[0, 0] = rows - 1
            if n > 2:
                routes[1, 0] = rows
                if k > 1:
                    routes[1, 1] = 0
            expanded_row_idx[...] = (
                routes.T if mode < _FIRST_ROW_ROUTING_MODE else routes
            ).reshape(-1)
            for tensor in (x1, x2, bias, x, alpha1, alpha2, v):
                if tensor is not None:
                    tensor[...] = rng.normal(size=tensor.shape).astype(tensor.dtype)
            if expert_idx is not None and bias is not None:
                expert_idx[...] = rng.integers(
                    0, bias.shape[0], expert_idx.shape, dtype=numpy.int32
                )
                if n > 0 and k > 1:
                    expert_idx[0, 1] = -1
                    if k > 2:
                        expert_idx[0, 2] = bias.shape[0]
            if expert_idx is not None and (x is not None or bias is None):
                ranges = [
                    kwargs.get(name) or _DISABLED_EXPERT_RANGE
                    for name in (
                        "zero_expert_range",
                        "copy_expert_range",
                        "constant_expert_range",
                    )
                ]
                experts = max([bounds[1] for bounds in ranges] + [1])
                expert_idx[...] = (
                    numpy.arange(n * k, dtype=numpy.int32).reshape(n, k) % experts
                )
            return (
                expanded_x,
                expanded_row_idx,
                x1,
                x2,
                bias,
                scales,
                expert_idx,
                x,
                alpha1,
                alpha2,
                v,
                expanded_x_scale,
            )

        drop_pad_mode = kwargs.get("drop_pad_mode", 0)
        expand_n = expanded_row_idx.shape[0]
        expanded_row_idx = numpy.random.choice(
            expand_n, expand_n, replace=False
        ).astype(numpy.int32)
        if drop_pad_mode == 1 or drop_pad_mode == 3:
            e = expanded_x.shape[0]
            c = expanded_x.shape[1] if len(expanded_x.shape) > 2 else expand_n
            expanded_row_idx[expanded_row_idx >= e * c] = -1
        if bias is not None:
            e = bias.shape[0]
            expert_idx = (
                numpy.random.randint(0, e, size=expert_idx.size)
                .reshape(expert_idx.shape)
                .astype(numpy.int32)
            )
        return (
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
        )


class TorchMoeFinalizeRoutingTestSpec:
    """CPU reference for the Torch schema; preserve the CSV input ranges."""

    third_party = {"torch": _MoeFinalizeRoutingV2TorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        expanded_x,
        expanded_row_idx,
        x1=None,
        x2=None,
        bias=None,
        scales=None,
        expert_idx=None,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        drop_pad_mode=0,
        zero_expert_range=None,
        copy_expert_range=None,
        constant_expert_range=None,
        k=1,
        expanded_x_scale=None,
        **kwargs,
    ):
        return MoeFinalizeRoutingV2TestSpec.golden(
            *[
                _to_numpy(t)
                for t in (
                    expanded_x,
                    expanded_row_idx,
                    x1,
                    x2,
                    bias,
                    scales,
                    expert_idx,
                    x,
                    alpha1,
                    alpha2,
                    v,
                    expanded_x_scale,
                )
            ],
            drop_pad_mode=drop_pad_mode,
            k=k,
            zero_expert_range=zero_expert_range,
            copy_expert_range=copy_expert_range,
            constant_expert_range=constant_expert_range,
            **kwargs,
        )


class AclnnMoeFinalizeRoutingV2TestSpec:
    third_party = {"torch": _AclnnMoeFinalizeRoutingV2TorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        expandedX,
        expandedRowIdx,
        x1=None,
        x2=None,
        bias=None,
        scales=None,
        expertIdx=None,
        dropPadMode=0,
        out=None,
        *args,
        **kwargs,
    ):
        return MoeFinalizeRoutingV2TestSpec.golden(
            _to_numpy(expandedX),
            _to_numpy(expandedRowIdx),
            _to_numpy(x1),
            _to_numpy(x2),
            _to_numpy(bias),
            _to_numpy(scales),
            _to_numpy(expertIdx),
            x=_to_numpy(kwargs.get("x")),
            alpha1=_to_numpy(kwargs.get("alpha1")),
            alpha2=_to_numpy(kwargs.get("alpha2")),
            v=_to_numpy(kwargs.get("v")),
            drop_pad_mode=dropPadMode,
            zero_expert_range=kwargs.get("zeroExpertRange"),
            copy_expert_range=kwargs.get("copyExpertRange"),
            constant_expert_range=kwargs.get("constantExpertRange"),
            k=kwargs.get("k", 1),
        )

    @staticmethod
    def customize_inputs(
        expandedX,
        expandedRowIdx,
        x1=None,
        x2=None,
        bias=None,
        scales=None,
        expertIdx=None,
        dropPadMode=0,
        out=None,
        *args,
        **kwargs,
    ):
        templates = (expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx)
        results = MoeFinalizeRoutingV2TestSpec.customize_inputs(
            *[_to_numpy(t) for t in templates],
            x=None,
            alpha1=None,
            alpha2=None,
            v=None,
            **kwargs,
        )[:7]
        return tuple(
            _restore_type(value, template)
            for value, template in zip(results, templates)
        )


class AclnnMoeFinalizeRoutingV5TestSpec:
    """Bind the V5 header parameters to the shared operator reference."""

    @staticmethod
    def customize_inputs(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        expanded_x_scale,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
        out,
        **kwargs,
    ):
        return MoeFinalizeRoutingV2TestSpec.customize_inputs(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
            expanded_x_scale,
            drop_pad_mode=drop_pad_mode,
            k=k,
            zero_expert_range=zero_expert_range,
            copy_expert_range=copy_expert_range,
            constant_expert_range=constant_expert_range,
            **kwargs,
        )

    @staticmethod
    def golden(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x,
        alpha1,
        alpha2,
        v,
        expanded_x_scale,
        drop_pad_mode,
        zero_expert_range,
        copy_expert_range,
        constant_expert_range,
        k,
        out,
        **kwargs,
    ):
        return MoeFinalizeRoutingV2TestSpec.golden(
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
            expanded_x_scale,
            drop_pad_mode=drop_pad_mode,
            k=k,
            zero_expert_range=zero_expert_range,
            copy_expert_range=copy_expert_range,
            constant_expert_range=constant_expert_range,
            **kwargs,
        )

    tolerance = MoeFinalizeRoutingV2TestSpec.tolerance
