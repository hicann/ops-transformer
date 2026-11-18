# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""External TTK entry points; all DUT calls use the installed Torch operator.

TTK receives the enabled tensor outputs, in public out/final/chunk order.
The actual dispatcher API continues to return its original 3/16-slot tuple.
"""

import importlib.util
import json
import sys
from functools import lru_cache, wraps
from pathlib import Path
from typing import List, Optional

import torch


@lru_cache(None)
def load_asset(name):
    path = Path(__file__).with_name(f"{name}.py")
    if name == "spec":
        path = Path(__file__).resolve().parents[1] / "spec.py"
    spec = importlib.util.spec_from_file_location(f"_flash_gdn_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def call_kwargs(inputs, attributes):
    return load_asset("inputs").call_kwargs(inputs, attributes)


def invoke(inputs, attributes, *, entry="flash_gdn_fwd"):
    # Importing the built package registers the real torch.ops schemas/kernels.
    import cann_ops_transformer  # noqa: F401

    if not attributes.get("inference", True):
        raise ValueError("FlashGDN currently supports inference=True only")
    config = json.loads(attributes.get("test_config", "{}"))
    if config.get("kind", "accuracy") != "accuracy":
        raise ValueError("this adapter supports accuracy cases only")
    kwargs = call_kwargs(inputs, attributes)
    with torch.inference_mode():
        result = getattr(torch.ops.cann_ops_transformer, entry).default(**kwargs)
    load_asset("compare").check(result, inputs, attributes, entry=entry)
    return result


def flash_gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    metadata: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = None,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = True,
    layout: str = "TND",
    inference: bool = True,
    valid_lengths: Optional[List[int]] = None,
    initial_state_mode: str = "none",
    dt_bias_mode: str = "none",
    cu_seqlens_value: Optional[List[int]] = None,
    seqused_value: Optional[List[int]] = None,
    seed: int = 101,
    distribution: float = 1.0,
    omitted_attributes: str = "",
    test_config: str = "{}",
    testcase_name: str = "",
):
    attributes = locals().copy()
    inputs = [attributes.pop(name) for name in load_asset("inputs").INPUT_NAMES]
    return load_asset("compare").pack(invoke(inputs, attributes))


# Keep the same inspectable signature for the pre-existing redline CSV entry.
@wraps(flash_gdn_fwd)
def flash_gdn(*args, **kwargs):
    import inspect

    bound = inspect.signature(flash_gdn_fwd).bind(*args, **kwargs)
    bound.apply_defaults()
    attributes = dict(bound.arguments)
    inputs = [attributes.pop(name) for name in load_asset("inputs").INPUT_NAMES]
    return load_asset("compare").pack(invoke(inputs, attributes, entry="flash_gdn"))
