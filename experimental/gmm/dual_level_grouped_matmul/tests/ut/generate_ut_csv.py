#!/usr/bin/env python3
"""Generate CSV-driven UT case tables for DualLevelGroupedMatmul."""

import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_LOGICAL_DIMENSION = (1 << 31) - 2  # INT32_MAX - 1.

TILING_COLUMNS = [
    "case_name",
    "expectResult",
    "expect_tiling_key",
    "expect_core_num",
    "transpose_weight",
    "group_list_type",
    "dtype",
    "tuning_config",
    "soc_version",
    "short_tiling_buffer",
    "aic_num",
    "aiv_num",
    "x_shape",
    "x_dtype",
    "x_format",
    "x_level0_shape",
    "x_level0_dtype",
    "x_level0_format",
    "per_token_shape",
    "per_token_dtype",
    "per_token_format",
    "weight_shape",
    "weight_storage_shape",
    "weight_dtype",
    "weight_format",
    "weight_level0_shape",
    "weight_level0_dtype",
    "weight_level0_format",
    "scale_shape",
    "scale_storage_shape",
    "scale_dtype",
    "scale_format",
    "group_list_shape",
    "group_list_dtype",
    "group_list_format",
    "bias_shape",
    "bias_dtype",
    "bias_format",
    "y_shape",
    "y_dtype",
    "y_format",
    "level0_group_size",
    "level1_group_size",
]


def dims(*values):
    return " ".join(str(int(v)) for v in values)


def tiling_case(
    name,
    m,
    k,
    e,
    n,
    transpose,
    dtype,
    group_list_type,
    bias,
    expect="SUCCESS",
    **overrides,
):
    level0_groups = math.ceil(k / 512)
    mx_groups = math.ceil(k / 64)
    if transpose:
        weight_origin = dims(e, n, k)
        weight_storage = dims(e, math.ceil(k / 64), math.ceil(n / 16), 16, 64)
        # At the tiling level the transposed scale is a plain ND (E,N,q,2) tensor.
        scale_view = dims(e, n, math.ceil(k / 64), 2)
        scale_storage = ""
    else:
        weight_origin = dims(e, k, n)
        weight_storage = dims(e, math.ceil(n / 64), math.ceil(k / 16), 16, 64)
        scale_view = dims(e, mx_groups, n, 2)
        scale_storage = ""
    y_dtype = "BF16" if dtype == "BF16" else "FLOAT16"
    row = {
        "case_name": name,
        "expectResult": expect,
        "expect_tiling_key": str(
            (2 if dtype == "BF16" else 0) + (1 if transpose else 0)
        ),
        "expect_core_num": "32",
        "transpose_weight": "true" if transpose else "false",
        "group_list_type": str(group_list_type),
        "dtype": dtype,
        "tuning_config": "0",
        "level0_group_size": "",
        "level1_group_size": "",
        "soc_version": "Ascend950",
        "short_tiling_buffer": "false",
        "aic_num": "32",
        "aiv_num": "64",
        "x_shape": dims(m, k),
        "x_dtype": "FLOAT4_E2M1",
        "x_format": "ND",
        "x_level0_shape": dims(m, level0_groups),
        "x_level0_dtype": "FLOAT",
        "x_level0_format": "ND",
        "per_token_shape": dims(m, mx_groups, 2),
        "per_token_dtype": "FLOAT8_E8M0",
        "per_token_format": "ND",
        "weight_shape": weight_origin,
        "weight_storage_shape": weight_storage,
        "weight_dtype": "FLOAT4_E2M1",
        "weight_format": "FRACTAL_NZ",
        "weight_level0_shape": dims(e, level0_groups, n),
        "weight_level0_dtype": "FLOAT",
        "weight_level0_format": "ND",
        "scale_shape": scale_view,
        "scale_storage_shape": scale_storage,
        "scale_dtype": "FLOAT8_E8M0",
        "scale_format": "ND",
        "group_list_shape": dims(e, 2) if group_list_type == 2 else dims(e),
        "group_list_dtype": "INT64",
        "group_list_format": "ND",
        "bias_shape": dims(e, n) if bias else "",
        "bias_dtype": "FLOAT" if bias else "",
        "bias_format": "ND" if bias else "",
        "y_shape": dims(m, n),
        "y_dtype": y_dtype,
        "y_format": "ND",
    }
    row.update(overrides)
    return row


def build_tiling_rows():
    rows = []
    # Layout x dtype x group-list-type x bias matrix (base shape m33 k514 e3 n32/17).
    for transpose, tag in ((False, "nt"), (True, "wt")):
        for dtype in ("FLOAT16", "BF16"):
            for gl in (0, 1, 2):
                for bias in (False, True):
                    rows.append(
                        tiling_case(
                            f"tiling_{tag}_{dtype.lower()}_gl{gl}_{'bias' if bias else 'nobias'}",
                            33,
                            514,
                            3,
                            17 if transpose else 32,
                            transpose,
                            dtype,
                            gl,
                            bias,
                        )
                    )
    # Extra success shapes: aligned, K/N tails, boundary and logical NZ storage.
    rows.append(
        tiling_case(
            "tiling_shape_m17_k512_e3_n64_nt_fp16_gl0_nobias",
            17,
            512,
            3,
            64,
            False,
            "FLOAT16",
            0,
            False,
            weight_storage_shape="",
            level0_group_size="512",
            level1_group_size="32",
        )
    )
    rows.append(
        tiling_case(
            "tiling_shape_m129_k1024_e4_n130_wt_bf16_gl0_bias",
            129,
            1024,
            4,
            130,
            True,
            "BF16",
            0,
            True,
        )
    )
    rows.append(
        tiling_case(
            "tiling_shape_m1_k4_e1_n2_nt_fp16_gl1_nobias",
            1,
            4,
            1,
            2,
            False,
            "FLOAT16",
            1,
            False,
        )
    )
    rows.append(
        tiling_case(
            "tiling_shape_m16_k4098_e4_n16_wt_fp16_gl1_bias",
            16,
            4098,
            4,
            16,
            True,
            "FLOAT16",
            1,
            True,
        )
    )
    rows.append(
        tiling_case(
            "tiling_shape_m7_k64_e5_n66_nt_bf16_gl1_nobias",
            7,
            64,
            5,
            66,
            False,
            "BF16",
            1,
            False,
        )
    )

    # Failure cases.
    rows.append(
        tiling_case(
            "tiling_reject_weight_storage_short",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            weight_storage_shape=dims(3, 9, 1, 16, 64),
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_x_level0_shape",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            x_level0_shape=dims(33, 3),
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_level0_group_size",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            level0_group_size="256",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_level1_group_size",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            level0_group_size="512",
            level1_group_size="64",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_group_list_sparse_rank",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            2,
            False,
            expect="FAILED",
            group_list_shape=dims(3),
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_tuning_config",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            tuning_config="1",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_group_list_type",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            3,
            False,
            expect="FAILED",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_output_dtype",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            y_dtype="FLOAT",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_short_tiling_buffer",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            short_tiling_buffer="true",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_address_overflow",
            33,
            2000000000,
            2000000000,
            2000000000,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_soc_unsupported",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            soc_version="Ascend910B",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_missing_scale",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            scale_shape="",
            scale_storage_shape="",
            scale_dtype="",
            scale_format="",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_weight_dtype",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            weight_dtype="FLOAT16",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_odd_n_untransposed",
            33,
            514,
            3,
            17,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_odd_k",
            33,
            515,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_x_rank",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            x_shape=dims(33),
        )
    )
    rows.append(
        tiling_case(
            "tiling_reject_k_mismatch",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            x_shape=dims(33, 512),
        )
    )
    for name, k, result in (
        ("max_logical_k", MAX_LOGICAL_DIMENSION, "SUCCESS"),
        ("reject_k_above_max_logical_dim", MAX_LOGICAL_DIMENSION + 2, "FAILED"),
    ):
        rows.append(
            tiling_case(
                f"tiling_{name}", 1, k, 1, 32, False, "FLOAT16", 0, False, expect=result
            )
        )
    for aic, aiv, result in (
        (0, 0, "FAILED"),
        (32, 0, "FAILED"),
        (32, 32, "FAILED"),
        (16, 64, "SUCCESS"),
    ):
        rows.append(
            tiling_case(
                f"tiling_cores_{aic}_{aiv}",
                33,
                514,
                3,
                32,
                False,
                "FLOAT16",
                0,
                False,
                expect=result,
                aic_num=str(aic),
                aiv_num=str(aiv),
                expect_core_num=str(aic),
            )
        )
    return rows


INFER_SHAPE_COLUMNS = [
    "case_name",
    "case_type",
    "expectResult",
    "expect_y_shape",
    "transpose_weight",
    "group_list_type",
    "dtype",
    "tuning_config",
    "x_shape",
    "x_dtype",
    "x_format",
    "x_level0_shape",
    "x_level0_dtype",
    "x_level0_format",
    "per_token_shape",
    "per_token_dtype",
    "per_token_format",
    "weight_shape",
    "weight_dtype",
    "weight_format",
    "weight_level0_shape",
    "weight_level0_dtype",
    "weight_level0_format",
    "scale_shape",
    "scale_dtype",
    "scale_format",
    "group_list_shape",
    "group_list_dtype",
    "group_list_format",
    "bias_shape",
    "bias_dtype",
    "bias_format",
    "y_shape",
    "y_dtype",
    "y_format",
]


def infer_case(
    name,
    m,
    k,
    e,
    n,
    transpose,
    dtype,
    group_list_type,
    bias,
    expect="SUCCESS",
    case_type="shape",
    expect_y="",
    **overrides,
):
    level0_groups = max(math.ceil(k / 512), 0)
    mx_groups = max(math.ceil(k / 64), 0)
    weight_origin = dims(e, n, k) if transpose else dims(e, k, n)
    row = {
        "case_name": name,
        "case_type": case_type,
        "expectResult": expect,
        "expect_y_shape": expect_y if expect_y else dims(m, n),
        "transpose_weight": "true" if transpose else "false",
        "group_list_type": str(group_list_type),
        "dtype": dtype,
        "tuning_config": "0",
        "x_shape": dims(m, k),
        "x_dtype": "FLOAT4_E2M1",
        "x_format": "ND",
        "x_level0_shape": dims(m, level0_groups),
        "x_level0_dtype": "FLOAT",
        "x_level0_format": "ND",
        "per_token_shape": dims(m, mx_groups, 2),
        "per_token_dtype": "FLOAT8_E8M0",
        "per_token_format": "ND",
        "weight_shape": weight_origin,
        "weight_dtype": "FLOAT4_E2M1",
        "weight_format": "FRACTAL_NZ",
        "weight_level0_shape": dims(e, level0_groups, n),
        "weight_level0_dtype": "FLOAT",
        "weight_level0_format": "ND",
        "scale_shape": dims(e, mx_groups, n, 2),
        "scale_dtype": "FLOAT8_E8M0",
        "scale_format": "ND",
        "group_list_shape": dims(e, 2) if group_list_type == 2 else dims(e),
        "group_list_dtype": "INT64",
        "group_list_format": "ND",
        "bias_shape": dims(e, n) if bias else "",
        "bias_dtype": "FLOAT" if bias else "",
        "bias_format": "ND" if bias else "",
        "y_shape": dims(m, n),
        "y_dtype": "BF16" if dtype == "BF16" else "FLOAT16",
        "y_format": "ND",
    }
    row.update(overrides)
    return row


def build_infer_shape_rows():
    rows = []
    for transpose, tag in ((False, "nt"), (True, "wt")):
        for dtype in ("FLOAT16", "BF16"):
            for gl in (0, 1, 2):
                rows.append(
                    infer_case(
                        f"infershape_{tag}_{dtype.lower()}_gl{gl}_bias",
                        33,
                        514,
                        3,
                        17 if transpose else 32,
                        transpose,
                        dtype,
                        gl,
                        True,
                    )
                )
    rows.append(
        infer_case("infershape_dynamic_k_nt", 33, -1, 3, 32, False, "FLOAT16", 0, False)
    )
    rows.append(
        infer_case("infershape_dynamic_k_wt", 33, -1, 3, 17, True, "BF16", 1, False)
    )
    rows.append(
        infer_case(
            "infershape_reject_k_mismatch_nt",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            expect_y="",
            x_shape=dims(33, 512),
        )
    )
    rows.append(
        infer_case(
            "infershape_reject_k_mismatch_wt",
            33,
            514,
            3,
            17,
            True,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            expect_y="",
            x_shape=dims(33, 512),
        )
    )
    rows.append(
        infer_case(
            "infershape_reject_x_rank",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            expect_y="",
            x_shape=dims(33),
        )
    )
    rows.append(
        infer_case(
            "infershape_reject_weight_rank",
            33,
            514,
            3,
            32,
            False,
            "FLOAT16",
            0,
            False,
            expect="FAILED",
            expect_y="",
            weight_shape=dims(3, 514),
        )
    )
    for dtype, expect in (
        ("FLOAT16", "SUCCESS"),
        ("BF16", "SUCCESS"),
        ("FLOAT", "FAILED"),
    ):
        rows.append(
            infer_case(
                f"infershape_dtype_only_{dtype.lower()}",
                33,
                514,
                3,
                32,
                False,
                dtype,
                0,
                True,
                expect=expect,
                case_type="dtype_only",
                expect_y="",
            )
        )
    return rows


OPAPI_COLUMNS = [
    "caseName",
    "expectRet",
    "nullInput",
    "offsetInput",
    "xSpec",
    "xDtype",
    "xFormat",
    "weightSpec",
    "weightDtype",
    "weightFormat",
    "biasSpec",
    "biasDtype",
    "biasFormat",
    "scaleSpec",
    "scaleDtype",
    "scaleFormat",
    "perTokenSpec",
    "perTokenDtype",
    "perTokenFormat",
    "groupListSpec",
    "groupListDtype",
    "groupListFormat",
    "xLevel0Spec",
    "xLevel0Dtype",
    "xLevel0Format",
    "weightLevel0Spec",
    "weightLevel0Dtype",
    "weightLevel0Format",
    "outSpec",
    "outDtype",
    "outFormat",
    "level0GroupSize",
    "level1GroupSize",
    "splitItem",
    "groupType",
    "groupListType",
    "tuningConfig",
]


def acl_dims(*values):
    return ":".join(str(int(v)) for v in values)


def opapi_case(
    name,
    transpose,
    out_dtype,
    group_list_type,
    bias,
    m=33,
    k=514,
    e=3,
    expect="ACLNN_SUCCESS",
    n=None,
    **overrides,
):
    n = (17 if transpose else 32) if n is None else n
    level0_groups = math.ceil(k / 512)
    mx_groups = math.ceil(k / 64)
    if transpose:
        weight_spec = (
            f"{acl_dims(e, k, n)}#{acl_dims(k * n, 1, k)}#"
            f"{acl_dims(e, math.ceil(k / 64), math.ceil(n / 16), 16, 64)}"
        )
        scale_spec = (
            f"{acl_dims(e, mx_groups, n, 2)}#{acl_dims(n * mx_groups * 2, 2, mx_groups * 2, 1)}#"
            f"{acl_dims(e, n, mx_groups, 2)}"
        )
    else:
        weight_spec = (
            f"{acl_dims(e, k, n)}#{acl_dims(k * n, n, 1)}#"
            f"{acl_dims(e, math.ceil(n / 64), math.ceil(k / 16), 16, 64)}"
        )
        scale_spec = acl_dims(e, mx_groups, n, 2)
    row = {
        "caseName": name,
        "expectRet": expect,
        "nullInput": "NONE",
        "offsetInput": "NONE",
        "xSpec": acl_dims(m, k),
        "xDtype": "FLOAT4_E2M1",
        "xFormat": "ND",
        "weightSpec": weight_spec,
        "weightDtype": "FLOAT4_E2M1",
        "weightFormat": "FRACTAL_NZ",
        "biasSpec": acl_dims(e, n) if bias else "NONE",
        "biasDtype": "FLOAT",
        "biasFormat": "ND",
        "scaleSpec": scale_spec,
        "scaleDtype": "FLOAT8_E8M0",
        "scaleFormat": "ND",
        "perTokenSpec": acl_dims(m, mx_groups, 2),
        "perTokenDtype": "FLOAT8_E8M0",
        "perTokenFormat": "ND",
        "groupListSpec": acl_dims(e, 2) if group_list_type == 2 else acl_dims(e),
        "groupListDtype": "INT64",
        "groupListFormat": "ND",
        "xLevel0Spec": acl_dims(m, level0_groups),
        "xLevel0Dtype": "FLOAT",
        "xLevel0Format": "ND",
        "weightLevel0Spec": acl_dims(e, level0_groups, n),
        "weightLevel0Dtype": "FLOAT",
        "weightLevel0Format": "ND",
        "outSpec": acl_dims(m, n),
        "outDtype": out_dtype,
        "outFormat": "ND",
        "level0GroupSize": "512",
        "level1GroupSize": "32",
        "splitItem": "3",
        "groupType": "0",
        "groupListType": str(group_list_type),
        "tuningConfig": "NONE",
    }
    row.update(overrides)
    return row


def build_opapi_rows():
    rows = []
    for name, k, result in (
        ("max_logical_k", MAX_LOGICAL_DIMENSION, "ACLNN_SUCCESS"),
        (
            "reject_k_above_max_logical_dim",
            MAX_LOGICAL_DIMENSION + 2,
            "ACLNN_ERR_PARAM_INVALID",
        ),
    ):
        rows.append(
            opapi_case(
                f"checker_{name}",
                False,
                "FLOAT16",
                0,
                False,
                m=1,
                k=k,
                e=1,
                expect=result,
            )
        )
    for transpose, tag in ((False, "nt"), (True, "wt")):
        for dtype in ("FLOAT16", "BF16"):
            for gl in (0, 1, 2):
                for bias in (False, True):
                    rows.append(
                        opapi_case(
                            f"checker_{tag}_{dtype.lower()}_gl{gl}_{'bias' if bias else 'nobias'}",
                            transpose,
                            dtype,
                            gl,
                            bias,
                        )
                    )
    rows.append(
        opapi_case("checker_split_item_2", False, "FLOAT16", 0, False, splitItem="2")
    )
    rows.append(
        opapi_case(
            "checker_shape_m129_k1024_e4_n130_wt",
            True,
            "BF16",
            0,
            True,
            m=129,
            k=1024,
            e=4,
            n=130,
        )
    )
    rows.append(
        opapi_case(
            "checker_shape_m1_k4_e1_n2_nt",
            False,
            "FLOAT16",
            1,
            False,
            m=1,
            k=4,
            e=1,
            n=2,
        )
    )
    rows.append(
        opapi_case(
            "checker_shape_m16_k4098_e4_n16_wt",
            True,
            "FLOAT16",
            1,
            True,
            m=16,
            k=4098,
            e=4,
            n=16,
        )
    )
    # NZ padding increases the weight storage count to 2**63 while the view still fits in int64.
    rows.append(
        opapi_case(
            "checker_reject_nz_storage_product_overflow",
            False,
            "FLOAT16",
            0,
            False,
            m=1,
            k=4,
            e=2**30,
            n=2**29,
            expect="ACLNN_ERR_PARAM_INVALID",
        )
    )

    for null_input in (
        "x",
        "weight",
        "scale",
        "perToken",
        "groupList",
        "xLevel0",
        "weightLevel0",
        "out",
    ):
        rows.append(
            opapi_case(
                f"checker_null_{null_input}",
                True,
                "FLOAT16",
                0,
                False,
                expect="ACLNN_ERR_PARAM_NULLPTR",
                nullInput=null_input,
            )
        )
    for offset_input in (
        "x",
        "weight",
        "scale",
        "perToken",
        "groupList",
        "xLevel0",
        "weightLevel0",
        "out",
    ):
        rows.append(
            opapi_case(
                f"checker_offset_{offset_input}",
                True,
                "FLOAT16",
                0,
                False,
                expect="ACLNN_ERR_PARAM_INVALID",
                offsetInput=offset_input,
            )
        )

    m, n = 33, 32
    # The view fits the declared storage but is not a contiguous output.
    output_shape = acl_dims(m, n)
    rows.append(
        opapi_case(
            "checker_reject_output_stride",
            False,
            "FLOAT16",
            0,
            False,
            m=m,
            n=n,
            expect="ACLNN_ERR_PARAM_INVALID",
            outSpec=f"{output_shape}#{acl_dims(1, 1)}#{output_shape}",
        )
    )

    rows.append(
        opapi_case(
            "checker_reject_scale_stride",
            True,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            scaleSpec=(
                f"{acl_dims(3, 9, 17, 2)}#{acl_dims(17 * 9 * 2, 34, 2, 1)}#"
                f"{acl_dims(3, 17, 9, 2)}"
            ),
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_weight_storage_short",
            True,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            weightSpec=(
                f"{acl_dims(3, 514, 17)}#{acl_dims(514 * 17, 1, 514)}#"
                f"{acl_dims(3, 9, 1, 16, 64)}"
            ),
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_x_level0_shape",
            True,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            xLevel0Spec=acl_dims(33, 3),
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_level0_group_size",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            level0GroupSize="256",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_level1_group_size",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            level1GroupSize="64",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_group_list_type",
            False,
            "FLOAT16",
            3,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_split_item",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            splitItem="1",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_tuning_config",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            tuningConfig="1",
        )
    )
    limit = 9223372036854775807
    rows.append(
        opapi_case(
            "checker_reject_dense_product_overflow",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            weightLevel0Spec=f"{acl_dims(2, limit)}#{acl_dims(limit, 1)}#{acl_dims(2, limit)}",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_output_dtype",
            False,
            "FLOAT",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_x_dtype",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            xDtype="FLOAT16",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_scale_dtype",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            scaleDtype="FLOAT",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_group_list_dtype",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            groupListDtype="INT32",
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_odd_n_untransposed",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            weightSpec=(
                f"{acl_dims(3, 514, 17)}#{acl_dims(514 * 17, 17, 1)}#"
                f"{acl_dims(3, 1, 33, 16, 64)}"
            ),
            scaleSpec=acl_dims(3, 9, 17, 2),
            weightLevel0Spec=acl_dims(3, 2, 17),
            outSpec=acl_dims(33, 17),
            biasSpec=acl_dims(3, 17),
        )
    )
    rows.append(
        opapi_case(
            "checker_reject_odd_k",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            xSpec=acl_dims(33, 515),
            weightSpec=(
                f"{acl_dims(3, 515, 32)}#{acl_dims(515 * 32, 32, 1)}#"
                f"{acl_dims(3, 1, 33, 16, 64)}"
            ),
            xLevel0Spec=acl_dims(33, 2),
            perTokenSpec=acl_dims(33, 9, 2),
            weightLevel0Spec=acl_dims(3, 2, 32),
            scaleSpec=acl_dims(3, 9, 32, 2),
        )
    )
    for name in (
        "x",
        "weight",
        "bias",
        "scale",
        "perToken",
        "xLevel0",
        "weightLevel0",
        "out",
    ):
        for mode, status in (
            ("empty", "ACLNN_ERR_PARAM_INVALID"),
            ("element", "ACLNN_ERR_PARAM_NULLPTR"),
        ):
            rows.append(
                opapi_case(
                    f"checker_{mode}_{name}",
                    True,
                    "FLOAT16",
                    0,
                    True,
                    expect=status,
                    nullInput=f"{mode}_{name}",
                )
            )
    for name in ("weight", "bias", "perToken", "xLevel0", "weightLevel0"):
        rows.append(
            opapi_case(
                f"checker_dtype_{name}",
                True,
                "FLOAT16",
                0,
                True,
                expect="ACLNN_ERR_PARAM_INVALID",
                **{name + "Dtype": "FLOAT16"},
            )
        )
    rows.append(
        opapi_case(
            "checker_reject_group_type",
            False,
            "FLOAT16",
            0,
            False,
            expect="ACLNN_ERR_PARAM_INVALID",
            groupType="2",
        )
    )
    empty_cases = {
        "M": (0, 32, 64),
        "N": (4, 0, 64),
        "K": (4, 32, 0),
        "MN": (0, 0, 64),
        "MK": (0, 32, 0),
        "NK": (4, 0, 0),
        "MNK": (0, 0, 0),
    }
    for tag, (m, n, k) in empty_cases.items():
        for trans in (False, True):
            rows.append(
                opapi_case(
                    f"checker_empty_{tag}_{'wt' if trans else 'nt'}",
                    trans,
                    "FLOAT16",
                    0,
                    False,
                    m=m,
                    n=n,
                    k=k,
                    e=2,
                    expect="ACLNN_SUCCESS"
                    if m == 0 or n == 0
                    else "ACLNN_ERR_PARAM_INVALID",
                )
            )
    return rows


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            missing = set(columns) - set(row)
            extra = set(row) - set(columns)
            assert not missing and not extra, (path, row["case_name"], missing, extra)
            writer.writerow({c: row.get(c, "") for c in columns})
    print(f"{os.path.basename(path)}: {len(rows)} rows, {len(columns)} columns")


if __name__ == "__main__":
    write_csv(
        os.path.join(HERE, "op_host", "test_dual_level_grouped_matmul_tiling.csv"),
        TILING_COLUMNS,
        build_tiling_rows(),
    )
    write_csv(
        os.path.join(HERE, "op_host", "test_dual_level_grouped_matmul_infershape.csv"),
        INFER_SHAPE_COLUMNS,
        build_infer_shape_rows(),
    )
    write_csv(
        os.path.join(
            HERE,
            "op_host",
            "op_api",
            "test_aclnn_dual_level_grouped_matmul_weight_nz.csv",
        ),
        OPAPI_COLUMNS,
        build_opapi_rows(),
    )
