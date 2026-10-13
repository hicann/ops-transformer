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

"""Load the unified cann_ops_transformer extension for fused_gdn_gating pytest."""

import os
from pathlib import Path

import torch


_LOAD_ERRORS = []


def _prepend_env_path(env_name, paths):
    current = os.environ.get(env_name, "")
    current_parts = [part for part in current.split(":") if part]
    new_parts = [str(path) for path in paths if path and str(path) not in current_parts]
    if new_parts:
        os.environ[env_name] = ":".join(new_parts + current_parts)


def _custom_opp_roots():
    repo_root = Path(__file__).resolve().parents[5]
    build_candidates = [
        repo_root
        / "build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-x86_64.run"
        / "packages/vendors/custom_transformer",
    ]
    build_roots = [path for path in build_candidates if path.exists()]
    if build_roots:
        return build_roots
    candidates = [
        Path("/home/dyx/vendors/custom_transformer"),
        Path("/workspace/vendors/custom_transformer"),
    ]
    return [path for path in candidates if path.exists()]


def _configure_custom_opp_paths():
    custom_opp_roots = _custom_opp_roots()
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_roots)
    _prepend_env_path(
        "LD_LIBRARY_PATH", [root / "op_api" / "lib" for root in custom_opp_roots]
    )


def _load_custom_ops():
    _configure_custom_opp_paths()
    try:
        import cann_ops_transformer  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        _LOAD_ERRORS.append(str(error))


def has_npu_fused_gdn_gating_op():
    try:
        getattr(torch.ops.cann_ops_transformer, "fused_gdn_gating")
    except AttributeError:
        return False
    return True


def load_errors():
    return list(_LOAD_ERRORS)


_load_custom_ops()
