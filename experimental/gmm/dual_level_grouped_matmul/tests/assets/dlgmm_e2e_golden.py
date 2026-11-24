# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Torch packed-byte adapter and device-side NZ preparation for TTK eager ST."""

from pathlib import Path
from runpy import run_path

_kernel = run_path(str(Path(__file__).with_name("dlgmm_kernel_golden.py")))
_ref = run_path(str(Path(__file__).with_name("dlgmm_reference.py")))


def _physical_weight_inputs(weight, scale):
    # Public transposed views pack pairs along K, not along their last (N) axis.
    if hasattr(weight, "stride"):
        public_transpose = weight.stride(-2) == 1 and weight.stride(-1) > 1
    else:
        public_transpose = (
            weight.strides[-2] == weight.itemsize
            and weight.strides[-1] > weight.itemsize
        )
    if public_transpose:
        return weight.swapaxes(1, 2), scale.swapaxes(1, 2), True
    return weight, scale, False


def _output_dtype(value):
    import torch

    if value is None:
        value = torch.bfloat16
    if value not in (torch.float16, torch.bfloat16):
        raise ValueError("output_dtype must be torch.float16 or torch.bfloat16")
    return _kernel["_ge_output_dtype"](value)


class DualLevelGroupedMatmulTorchTestSpec:
    tolerance = _ref["TOLERANCE"]
    compare = staticmethod(_ref["isclose_compare"])

    @staticmethod
    def npu_preprocess(
        x,
        weight,
        x_level0_scale,
        x_level1_scale,
        weight_level0_scale,
        weight_level1_scale,
        level0_group_size=512,
        level1_group_size=32,
        *,
        group_list,
        **kwargs,
    ):
        import torch_npu

        # Restore the physical packed axis from the declared logical stride.
        w = _kernel["_single_tensor"](weight, "weight")
        b1 = _kernel["_single_tensor"](weight_level1_scale, "weight_level1_scale")
        physical_w, physical_b1, transposed = _physical_weight_inputs(w, b1)
        converted = torch_npu.npu_format_cast(physical_w, 29)
        weight[0] = converted.transpose(1, 2) if transposed else converted
        # Rank-four inputs created by the framework can use NCHW (format=0).
        # The public API requires actual ND storage, including for scale views.
        scale_nd = torch_npu.npu_format_cast(physical_b1.contiguous(), 2)
        weight_level1_scale[0] = scale_nd.transpose(1, 2) if transposed else scale_nd

    @staticmethod
    def golden(
        x,
        weight,
        x_level0_scale,
        x_level1_scale,
        weight_level0_scale,
        weight_level1_scale,
        level0_group_size=512,
        level1_group_size=32,
        *,
        group_list=None,
        bias=None,
        offset=None,
        antiquant_scale=None,
        antiquant_offset=None,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        split_item=3,
        group_type=0,
        group_list_type=0,
        act_type=0,
        tuning_config=None,
        output_dtype=None,
        x_dtype=None,
        weight_dtype=None,
        scale_dtype=None,
        per_token_scale_dtype=None,
        **kwargs,
    ):
        for name, value in (
            ("offset", offset),
            ("antiquant_scale", antiquant_scale),
            ("antiquant_offset", antiquant_offset),
            ("activation_input", activation_input),
            ("activation_quant_scale", activation_quant_scale),
            ("activation_quant_offset", activation_quant_offset),
        ):
            if value is not None:
                raise ValueError(
                    f"{name} is not supported by dual-level MXFP4 grouped matmul."
                )
        if "scale" in kwargs or "per_token_scale" in kwargs:
            raise ValueError(
                "Use weight_level1_scale and x_level1_scale; duplicate scale names are not exposed."
            )
        if split_item not in (2, 3) or group_type != 0:
            raise ValueError("split_item must be 2 or 3 and group_type must be 0.")
        if act_type not in (None, 0):
            raise ValueError(
                "act_type currently supports only no activation (0 or None)."
            )
        # The documented default None selects FP4/E8M0. Explicit hints use
        # TorchNPU's named semantic types rather than opaque integer codes.
        hints = (
            ("x_dtype", x_dtype, "float4_e2m1fn_x2"),
            ("weight_dtype", weight_dtype, "float4_e2m1fn_x2"),
            ("scale_dtype", scale_dtype, "float8_e8m0fnu"),
            ("per_token_scale_dtype", per_token_scale_dtype, "float8_e8m0fnu"),
        )
        for name, value, semantic_name in hints:
            if value is None:
                continue
            import torch_npu

            semantic = getattr(torch_npu, semantic_name)
            if value not in (None, semantic):
                raise ValueError(
                    f"{name} supports only torch_npu.{semantic_name} or None."
                )
        if group_list is None:
            raise ValueError("group_list is required.")
        group_list_type = 0 if group_list_type is None else group_list_type
        if (level0_group_size, level1_group_size) != (512, 32):
            raise ValueError("Only group sizes (512,32) are supported")
        for name, seq in (
            ("x", x),
            ("weight", weight),
            ("x_level0_scale", x_level0_scale),
            ("x_level1_scale", x_level1_scale),
            ("weight_level0_scale", weight_level0_scale),
            ("weight_level1_scale", weight_level1_scale),
        ):
            _kernel["_single_tensor"](seq, name)
        bias_tensor = _kernel["_single_tensor"](bias, "bias", optional=True)
        dtype = _output_dtype(output_dtype)
        physical_w, physical_b1, transposed = _physical_weight_inputs(
            weight[0], weight_level1_scale[0]
        )
        return _kernel["golden"](
            _ref["unpack_fp4"](x[0]),
            _ref["as_numpy"](x_level0_scale[0]),
            _ref["as_numpy"](x_level1_scale[0]),
            _ref["unpack_fp4"](physical_w),
            _ref["as_numpy"](weight_level0_scale[0]),
            _ref["as_numpy"](physical_b1),
            _ref["as_numpy"](group_list),
            None if bias_tensor is None else _ref["as_numpy"](bias_tensor),
            dtype=dtype,
            transpose_weight=transposed,
            group_list_type=group_list_type,
            tuning_config=tuning_config,
        )


__spec__ = {
    "torch.ops.cann_ops_transformer.dual_level_grouped_matmul": "DualLevelGroupedMatmulTorchTestSpec"
}
