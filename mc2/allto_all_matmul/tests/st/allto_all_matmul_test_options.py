# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Operator-local test-only options; index 0 preserves the native case description.

Boolean optional controls retain the described input/output when True and
disable that optional input/output using its native null/empty contract when
False. Layout controls apply only without a native field.
input_distribution selects unchanged native inputs (None), rank shards, or
the existing MoE uniform/balanced routing and token-marker layouts, or EPLB
fixed/random input generation.
This product is an encoding space, never an automatically scheduled case set.
"""

from itertools import product

OPTION_NAMES = (
    "x1_int4",
    "x2_int4",
    "x2_nz",
    "transpose_x2",
    "output_dtype",
    "use_bias",
    "use_x3",
    "use_x1_scale",
    "use_antiquant_offset",
    "use_comm_quant",
    "use_smooth",
    "comm_output",
    "matmul_output",
    "shared_mm",
    "packed_scale",
    "input_distribution",
    "deterministic",
)
OPTION_VALUES = (
    (False, True),
    (False, True),
    (False, True),
    (False, True),
    (None, "fp16", "bf16"),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (True, False),
    (False, True),
    (None, "rank_shards", "moe_uniform", "moe_balanced", "eplb_fixed", "eplb_random"),
    (None, True),
)
DEFAULT_OPTIONS = tuple(values[0] for values in OPTION_VALUES)
OPTIONS = list(product(*OPTION_VALUES))
