# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""CPU numerical policy shared by Kernel, ACLNN and Torch test adapters.

Keep the DLQBMM order: FP32 matmul per 32 K, multiply A1 then B1,
accumulate within 512 K, multiply A0*B0 then partial, accumulate, add bias,
and round once to output dtype. FP4 arguments here are numerical values.
"""

import ast

import numpy as np

TOLERANCE = {name: {"standard": "isclose"} for name in ("float16", "bfloat16")}


def as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def cast_output_dtype(value, dtype):
    name = str(dtype).removeprefix("torch.")
    if name == "bfloat16":
        import ml_dtypes

        return value.astype(ml_dtypes.bfloat16)
    return value.astype(name)


def e8m0_to_float(value):
    value = as_numpy(value)
    if value.dtype in (np.dtype("uint8"), np.dtype("int8")):
        codes = value.view(np.uint8)
        # E8M0 code 0 represents 2**-127; 255 represents NaN.
        exponent = np.minimum(codes, 254).astype(np.float32) - np.float32(127)
        return np.where(codes == 255, np.nan, np.exp2(exponent)).astype(np.float32)
    return value.astype(np.float32)


def nz_to_nd(value, shape):
    value = as_numpy(value)
    d = value.ndim
    axes = list(range(d - 4)) + [d - 3, d - 2, d - 4, d - 1]
    dense = value.transpose(axes).reshape(
        value.shape[:-4]
        + (value.shape[-3] * value.shape[-2], value.shape[-4] * value.shape[-1])
    )
    return dense[tuple(slice(0, size) for size in shape)]


def whole_storage(value):
    """Read the allocation, not the logical strided view of an NZ tensor."""
    if hasattr(value, "untyped_storage"):
        count = value.untyped_storage().nbytes() // value.element_size()
        return as_numpy(value.as_strided((count,), (1,), 0))
    current = allocation = value
    while getattr(current, "base", None) is not None:
        current = current.base
        if isinstance(current, np.ndarray):
            allocation = current
    return np.asarray(allocation).reshape(-1)


def unpack_fp4(value):
    raw = as_numpy(value).view(np.uint8)
    codes = np.stack((raw & 15, raw >> 4), axis=-1).reshape(
        raw.shape[:-1] + (raw.shape[-1] * 2,)
    )
    table = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
    )
    return table[codes]


def isclose_compare(npu_out, golden_out, compare_context=None, **kwargs):
    """Retain the former DLQBMM isclose defaults and CSV overrides."""
    actual, expected = as_numpy(npu_out), as_numpy(golden_out)
    if actual is None or expected is None or actual.shape != expected.shape:
        return {
            "pass": False,
            "precision": 0.0,
            "error_info": "output shape/None mismatch",
        }
    if not actual.size:
        return {"pass": True, "precision": 100.0}
    fields = compare_context.csv_fields if compare_context else {}
    default = 0.001 if str(actual.dtype) in ("float16", "bfloat16") else 0.0001
    rtol, ptol = (default, default)
    if fields.get("precision_tolerances"):
        rtol, ptol = ast.literal_eval(fields["precision_tolerances"])[0]
    atol = (
        ast.literal_eval(fields["absolute_precision"])
        if fields.get("absolute_precision")
        else 1e-8
    )
    match = np.isclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    ratio = float(match.mean())
    result = {"pass": (1 - ratio) <= ptol, "precision": round(ratio * 100, 4)}
    if not result["pass"]:
        bad = np.flatnonzero(~match)
        index = int(bad[0])
        result.update(
            error_info=f"{bad.size}/{match.size} mismatched (rtol={rtol}, atol={atol}, ptol={ptol}); "
            f"first[{index}]: actual={actual.flat[index]} ({actual.dtype}), "
            f"expected={expected.flat[index]} ({expected.dtype})",
            diff_indices=bad[:20].tolist(),
        )
    return result


class DualLevelMatmulReference:
    """The original DLQBMM reference policy, without cross-repository imports."""

    compare = staticmethod(isclose_compare)

    @staticmethod
    def golden(
        x1,
        x2,
        x1_level0_scale,
        x1_level1_scale,
        x2_level0_scale,
        x2_level1_scale,
        bias=None,
        *,
        dtype=1,
        transpose_x1=False,
        transpose_x2=True,
        level0_group_size=512,
        level1_group_size=32,
        **kwargs,
    ):
        if transpose_x1 or (level0_group_size, level1_group_size) != (512, 32):
            raise ValueError(
                "Only non-transposed x and group sizes (512,32) are supported"
            )
        a, b = x1.astype(np.float32), x2.astype(np.float32)
        if transpose_x2:
            b = b.T
        m, k = a.shape
        n = b.shape[1]
        a0, b0 = x1_level0_scale.astype(np.float32), x2_level0_scale.astype(np.float32)
        a1 = e8m0_to_float(x1_level1_scale).reshape(m, -1)
        b1 = e8m0_to_float(x2_level1_scale).reshape(n, -1)
        acc = np.zeros((m, n), np.float32)
        for g0 in range(a0.shape[1]):
            partial0 = np.zeros_like(acc)
            for g1 in range(g0 * 16, min((g0 + 1) * 16, (k + 31) // 32)):
                section = slice(g1 * 32, min((g1 + 1) * 32, k))
                partial = a[:, section] @ b[section, :]
                partial0 += partial * a1[:, g1, None] * b1[None, :, g1]
            acc += a0[:, g0, None] * b0[None, g0, :] * partial0
        if bias is not None:
            acc += bias.astype(np.float32).reshape(1, -1)
        target = kwargs.get(
            "output_dtypes", ("float16" if dtype == 1 else "bfloat16",)
        )[0]
        return [cast_output_dtype(acc, target)]
