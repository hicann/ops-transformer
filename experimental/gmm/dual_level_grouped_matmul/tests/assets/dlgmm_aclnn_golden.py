# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""ACLNN TensorList/NZ adapter; all arithmetic lives in the CPU reference."""

from pathlib import Path
from runpy import run_path

_kernel = run_path(str(Path(__file__).with_name("dlgmm_kernel_golden.py")))
_ref = run_path(str(Path(__file__).with_name("dlgmm_reference.py")))


def _transposed(value):
    return (
        value.stride(-2) == 1
        if hasattr(value, "stride")
        else value.strides[-2] == value.itemsize
    )


class AclnnDualLevelGroupedMatmulTestSpec:
    tolerance = _ref["TOLERANCE"]
    compare = staticmethod(_ref["isclose_compare"])

    @staticmethod
    def golden(
        x,
        weight,
        biasOptional,
        scaleOptional,
        perTokenScaleOptional,
        groupListOptional,
        xLevel0Scale,
        weightLevel0Scale,
        level0GroupSize,
        level1GroupSize,
        splitItem,
        groupType,
        groupListType,
        tuningConfigOptional,
        out,
        **kwargs,
    ):
        if (level0GroupSize, level1GroupSize, groupType) != (
            512,
            32,
            0,
        ) or splitItem not in (2, 3):
            raise ValueError(
                "Only group sizes (512,32), splitItem=2/3, groupType=0 are supported"
            )
        for name, seq in (
            ("x", x),
            ("weight", weight),
            ("scaleOptional", scaleOptional),
            ("perTokenScaleOptional", perTokenScaleOptional),
            ("xLevel0Scale", xLevel0Scale),
            ("weightLevel0Scale", weightLevel0Scale),
            ("out", out),
        ):
            _kernel["_single_tensor"](seq, name)
        bias = _kernel["_single_tensor"](biasOptional, "biasOptional", optional=True)
        dtype = _kernel["_ge_output_dtype"](out[0].dtype)
        if groupListOptional is None:
            raise ValueError("groupListOptional is required")
        a = _ref["as_numpy"](x[0])
        e, k, n = weight[0].shape
        if tuple(out[0].shape) != (a.shape[0], n):
            raise ValueError("out[0] must have shape (M,N)")
        trans = _transposed(weight[0])
        rows, cols = (n, k) if trans else (k, n)
        storage = _ref["whole_storage"](weight[0]).reshape(
            e, (cols + 63) // 64, (rows + 15) // 16, 16, 64
        )
        b = _ref["nz_to_nd"](storage, (e, rows, cols))
        b1 = _ref["as_numpy"](scaleOptional[0])
        if trans:
            b1 = b1.swapaxes(1, 2)
        return _kernel["golden"](
            a,
            _ref["as_numpy"](xLevel0Scale[0]),
            _ref["as_numpy"](perTokenScaleOptional[0]),
            b,
            _ref["as_numpy"](weightLevel0Scale[0]),
            b1,
            _ref["as_numpy"](groupListOptional),
            None if bias is None else _ref["as_numpy"](bias),
            dtype=dtype,
            transpose_weight=trans,
            group_list_type=groupListType,
            tuning_config=tuningConfigOptional,
        )


__spec__ = {
    "aclnnDualLevelGroupedMatmulWeightNz": "AclnnDualLevelGroupedMatmulTestSpec"
}
