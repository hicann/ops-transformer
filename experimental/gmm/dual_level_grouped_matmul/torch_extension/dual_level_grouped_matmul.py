# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
from typing import List, Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


LEVEL0_GROUP_SIZE = 512
LEVEL1_GROUP_SIZE = 32
MATRIX_RANK = 2
WEIGHT_LEVEL1_SCALE_RANK = 4
WEIGHT_LEVEL1_SCALE_N_AXIS = 2
DEFAULT_SPLIT_ITEM = 3
BFLOAT16_SCHEMA_TYPE = 15


def _normalize_gmm_inputs(values):
    values = dict(values)
    for name in (
        "offset",
        "antiquant_scale",
        "antiquant_offset",
        "activation_input",
        "activation_quant_scale",
        "activation_quant_offset",
    ):
        if values[name] is not None:
            raise ValueError(
                f"{name} is not supported by dual-level MXFP4 grouped matmul."
            )
    if values["act_type"] not in (None, 0):
        raise ValueError("act_type currently supports only no activation (0 or None).")
    if values["group_list"] is None:
        raise ValueError("group_list must be provided.")
    if values["split_item"] is None or values["group_type"] is None:
        raise ValueError("split_item and group_type must be provided.")
    values["group_list_type"] = (
        0 if values["group_list_type"] is None else values["group_list_type"]
    )
    values["output_dtype"] = (
        torch.bfloat16 if values["output_dtype"] is None else values["output_dtype"]
    )
    return values


def _output_shape(x, weight_level1_scale):
    # Meta only needs the two tensors used to infer the output dimensions.
    if len(x) != 1 or len(weight_level1_scale) != 1:
        raise ValueError("x and weight_level1_scale must each contain one tensor.")
    if (
        x[0].ndim != MATRIX_RANK
        or weight_level1_scale[0].ndim != WEIGHT_LEVEL1_SCALE_RANK
    ):
        raise ValueError("x and weight_level1_scale must have rank 2 and 4.")
    return x[0].shape[0], weight_level1_scale[0].shape[WEIGHT_LEVEL1_SCALE_N_AXIS]


class DualLevelGroupedMatmulOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("dual_level_grouped_matmul", category="gmm")

    def sources(self):
        return ["csrc/gmm/dual_level_grouped_matmul.cpp"]

    def schema(self):
        return (
            "dual_level_grouped_matmul(Tensor[] x, Tensor[] weight, "
            "Tensor[] x_level0_scale, Tensor[] x_level1_scale, "
            "Tensor[] weight_level0_scale, Tensor[] weight_level1_scale, "
            f"int level0_group_size={LEVEL0_GROUP_SIZE}, int level1_group_size={LEVEL1_GROUP_SIZE}, *, "
            "Tensor? group_list=None, Tensor[]? bias=None, Tensor[]? offset=None, "
            "Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, "
            "Tensor[]? activation_input=None, Tensor[]? activation_quant_scale=None, "
            f"Tensor[]? activation_quant_offset=None, int? split_item={DEFAULT_SPLIT_ITEM}, int? group_type=0, "
            "int? group_list_type=0, int? act_type=0, int[]? tuning_config=None, "
            f"ScalarType? output_dtype={BFLOAT16_SCHEMA_TYPE}, int? x_dtype=None, int? weight_dtype=None, "
            "int? scale_dtype=None, int? per_token_scale_dtype=None) -> Tensor[]"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def dual_level_grouped_matmul_meta(
            x,
            weight,
            x_level0_scale,
            x_level1_scale,
            weight_level0_scale,
            weight_level1_scale,
            level0_group_size=LEVEL0_GROUP_SIZE,
            level1_group_size=LEVEL1_GROUP_SIZE,
            *,
            group_list=None,
            bias=None,
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=DEFAULT_SPLIT_ITEM,
            group_type=0,
            group_list_type=0,
            act_type=0,
            tuning_config=None,
            output_dtype=torch.bfloat16,
            x_dtype=None,
            weight_dtype=None,
            scale_dtype=None,
            per_token_scale_dtype=None,
        ):
            values = _normalize_gmm_inputs(locals())
            if values["output_dtype"] not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    "output_dtype must be torch.float16 or torch.bfloat16."
                )
            shape = _output_shape(x, weight_level1_scale)
            return [torch.empty(shape, device="meta", dtype=values["output_dtype"])]


_op_builder = DualLevelGroupedMatmulOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def _dual_level_grouped_matmul(
    x,
    weight,
    x_level0_scale,
    x_level1_scale,
    weight_level0_scale,
    weight_level1_scale,
    level0_group_size=LEVEL0_GROUP_SIZE,
    level1_group_size=LEVEL1_GROUP_SIZE,
    *,
    group_list=None,
    bias=None,
    offset=None,
    antiquant_scale=None,
    antiquant_offset=None,
    activation_input=None,
    activation_quant_scale=None,
    activation_quant_offset=None,
    split_item=DEFAULT_SPLIT_ITEM,
    group_type=0,
    group_list_type=0,
    act_type=0,
    tuning_config=None,
    output_dtype=torch.bfloat16,
    x_dtype=None,
    weight_dtype=None,
    scale_dtype=None,
    per_token_scale_dtype=None,
):
    values = _normalize_gmm_inputs(locals())
    return _op_builder.load().dual_level_grouped_matmul(
        values["x"],
        values["weight"],
        values["x_level0_scale"],
        values["x_level1_scale"],
        values["weight_level0_scale"],
        values["weight_level1_scale"],
        values["level0_group_size"],
        values["level1_group_size"],
        values["group_list"],
        values["bias"],
        values["split_item"],
        values["group_type"],
        values["group_list_type"],
        values["tuning_config"],
        values["output_dtype"],
        values["x_dtype"],
        values["weight_dtype"],
        values["scale_dtype"],
        values["per_token_scale_dtype"],
    )


def dual_level_grouped_matmul(
    x: List[torch.Tensor],
    weight: List[torch.Tensor],
    x_level0_scale: List[torch.Tensor],
    x_level1_scale: List[torch.Tensor],
    weight_level0_scale: List[torch.Tensor],
    weight_level1_scale: List[torch.Tensor],
    level0_group_size: int = LEVEL0_GROUP_SIZE,
    level1_group_size: int = LEVEL1_GROUP_SIZE,
    *,
    group_list: Optional[torch.Tensor] = None,
    bias: Optional[List[torch.Tensor]] = None,
    offset: Optional[List[torch.Tensor]] = None,
    antiquant_scale: Optional[List[torch.Tensor]] = None,
    antiquant_offset: Optional[List[torch.Tensor]] = None,
    activation_input: Optional[List[torch.Tensor]] = None,
    activation_quant_scale: Optional[List[torch.Tensor]] = None,
    activation_quant_offset: Optional[List[torch.Tensor]] = None,
    split_item: Optional[int] = DEFAULT_SPLIT_ITEM,
    group_type: Optional[int] = 0,
    group_list_type: Optional[int] = 0,
    act_type: Optional[int] = 0,
    tuning_config: Optional[List[int]] = None,
    output_dtype: Optional[torch.dtype] = torch.bfloat16,
    x_dtype: Optional[int] = None,
    weight_dtype: Optional[int] = None,
    scale_dtype: Optional[int] = None,
    per_token_scale_dtype: Optional[int] = None,
) -> List[torch.Tensor]:
    """GMM controls with level1 scale names; fixed K group sizes (512, 32)."""
    return torch.ops.cann_ops_transformer.dual_level_grouped_matmul(
        x,
        weight,
        x_level0_scale,
        x_level1_scale,
        weight_level0_scale,
        weight_level1_scale,
        level0_group_size,
        level1_group_size,
        bias=bias,
        offset=offset,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset,
        group_list=group_list,
        activation_input=activation_input,
        activation_quant_scale=activation_quant_scale,
        activation_quant_offset=activation_quant_offset,
        split_item=split_item,
        group_type=group_type,
        group_list_type=group_list_type,
        act_type=act_type,
        tuning_config=tuning_config,
        output_dtype=output_dtype,
        x_dtype=x_dtype,
        weight_dtype=weight_dtype,
        scale_dtype=scale_dtype,
        per_token_scale_dtype=per_token_scale_dtype,
    )
