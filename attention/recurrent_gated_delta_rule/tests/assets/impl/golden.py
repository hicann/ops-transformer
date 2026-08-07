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

"""TTK golden adapter for RecurrentGatedDeltaRule (e2e + aclnn).

The CPU reference is loaded from tests/pytest/recurrent_gated_delta_rule_golden.py.
This adapter maps torch_npu / aclnn API-style parameters to the CPU golden
signature and aligns return values with NPU outputs (returned output + in-place
modified state).
"""

import gc
import importlib.util
import sys
from pathlib import Path


PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    """Load tests/pytest/recurrent_gated_delta_rule_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "recurrent_gated_delta_rule_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"rgdr_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


__golden__ = {
    "e2e": {
        "torch_npu.npu_recurrent_gated_delta_rule": "cpu_recurrent_gated_delta_rule"
    },
    "aclnn": {"aclnnRecurrentGatedDeltaRule": "aclnn_cpu_recurrent_gated_delta_rule"},
}


_PARAM_ORDER = [
    "beta",
    "scale",
    "actual_seq_lengths",
    "ssm_state_indices",
    "num_accepted_tokens",
    "g",
    "gk",
]


def cpu_recurrent_gated_delta_rule(query, key, value, state, *args, **kwargs):
    """Golden reference for torch_npu.npu_recurrent_gated_delta_rule.

    Receives the same parameters as the NPU API (positional or keyword).
    The CPU golden clones ``state`` internally, so the caller's tensor is not
    mutated. Returns ``(output, state)`` to align with NPU outputs: the returned
    ``npu_out`` and the in-place modified ``state`` (collected via
    ``inplace_input_indexes``).
    """
    mod = load_pytest_golden_module()
    p = {}
    for i, name in enumerate(_PARAM_ORDER):
        if name in kwargs:
            p[name] = kwargs[name]
        elif i < len(args):
            p[name] = args[i]
        else:
            p[name] = None
    output, state_out = mod.cpu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        p["beta"],
        p["scale"],
        p["actual_seq_lengths"],
        p["ssm_state_indices"],
        num_accepted_tokens=p["num_accepted_tokens"],
        g=p["g"],
        gk=p["gk"],
    )
    output = output.to(query.dtype)
    state_out = state_out.to(state.dtype)
    gc.collect()
    return output, state_out


def aclnn_cpu_recurrent_gated_delta_rule(
    query,
    key,
    value,
    beta,
    stateRef,
    actualSeqLengths,
    ssmStateIndices,
    g=None,
    gk=None,
    numAcceptedTokens=None,
    scaleValue=0.125,
    out=None,
    **kwargs,
):
    """Golden reference for aclnnRecurrentGatedDeltaRule.

    Parameters follow aclnnRecurrentGatedDeltaRuleGetWorkspaceSize (without
    workspaceSize and executor).  Maps aclnn parameter order to the CPU golden
    signature and returns a list [output, state_out] aligned with
    output_tensor_indexes (out first, stateRef second).
    """
    mod = load_pytest_golden_module()
    output, state_out = mod.cpu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        stateRef,
        beta,
        scaleValue,
        actualSeqLengths,
        ssmStateIndices,
        num_accepted_tokens=numAcceptedTokens,
        g=g,
        gk=gk,
    )
    output = output.to(query.dtype)
    state_out = state_out.to(stateRef.dtype)
    gc.collect()
    return [output, state_out]
