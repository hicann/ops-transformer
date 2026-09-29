# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import itertools

import pytest

from test_case import _npu_available, prepare_case, run_params, run_prepared
from test_case_paramset import ENABLED_PARAMS


def _flatten(spec):
    """{name: [values]} → 平坦参数字典列表（笛卡尔积；缺省项由 DEFAULTS 兜底）。"""
    names = list(spec.keys())
    return [
        dict(zip(names, combo))
        for combo in itertools.product(*(spec[name] for name in names))
    ]


def _fmt(value):
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    return str(value)


def _case_id(params, idx):
    name = params.get("Testcase_Name")
    if name is not None:
        return str(name)
    keys = (
        "template_run_mode",
        "B",
        "S1",
        "S2",
        "S2C",
        "K1",
        "K2",
        "cu_seqlens_q",
        "seqused_q",
        "ori_topk_length",
        "cmp_topk_length",
        "ori_kv_topk_mode",
        "cmp_kv_topk_mode",
        "block_size1",
        "block_size2",
        "seed",
        "dist",
        "kv_axis0_noncontiguous",
    )
    parts = [f"{k}{_fmt(params[k])}" for k in keys if k in params]
    return "_".join(parts) + f"_{idx:03d}"


def pytest_generate_tests(metafunc):
    if "params" not in metafunc.fixturenames:
        return
    mode = metafunc.config.getoption("data_mode")
    directory = metafunc.config.getoption("pt_dir")
    if mode != "run" and not directory:
        raise pytest.UsageError("--pt-dir is required for save/replay/save-run")
    if mode == "replay":
        from saved_cases import list_cases

        try:
            paths = list_cases(directory)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        metafunc.parametrize("params", paths, ids=[p.stem for p in paths])
        return
    excel = metafunc.config.getoption("mqsmla_excel")
    if excel:
        from excel_cases import load_excel_test_cases

        try:
            cases = load_excel_test_cases(
                excel, metafunc.config.getoption("mqsmla_sheet")
            )
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
    else:
        cases = [combo for spec in ENABLED_PARAMS for combo in _flatten(spec)]
    metafunc.parametrize(
        "params", cases, ids=[_case_id(p, i) for i, p in enumerate(cases)]
    )


def test_npu_against_cpu_golden(params, request):
    mode = request.config.getoption("data_mode")
    if mode != "save":
        if not _npu_available():
            pytest.skip("torch_npu or an NPU device is unavailable")
        import torch

        torch.npu.set_device(request.config.getoption("device_id"))
    if mode == "run":
        run_params(
            params,
            metadata_backend=request.config.getoption("metadata_backend"),
            verify_metadata_backends=request.config.getoption(
                "verify_metadata_backends"
            ),
        )
        return
    from saved_cases import case_path, load_case, save_case

    if mode == "replay":
        run_prepared(
            *load_case(params),
            metadata_backend=request.config.getoption("metadata_backend"),
            verify_metadata_backends=request.config.getoption(
                "verify_metadata_backends"
            ),
        )
        return
    index = request.node.callspec.indices["params"]
    path = case_path(
        request.config.getoption("pt_dir"), _case_id(params, index), index, params
    )
    prepared = prepare_case(params)
    save_case(path, prepared)
    del prepared
    if mode == "save-run":
        run_prepared(
            *load_case(path),
            metadata_backend=request.config.getoption("metadata_backend"),
            verify_metadata_backends=request.config.getoption(
                "verify_metadata_backends"
            ),
        )
