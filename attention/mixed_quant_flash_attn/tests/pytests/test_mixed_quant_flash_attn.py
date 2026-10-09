#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import gc

import torch
import pytest

from core.case_loader import (
    load_testcase_files,
    expand_cases,
    normalize_params,
    resolve_case_ids,
    get_data_dir,
)
from core.gen_data import generate_inputs
from core.postprocess import PostProcessor


_MARKER_TO_MODULE = {
    "func_redline": "functional_redline",
    "func_B250_redline": "functional_B250_redline",
    "func_stc": "functional_stc",
    "func_L0": "L0_0715",
    "perf_batch": "perf_batch_linearity",
    "perf_mtp": "perf_mtp_linearity",
    "perf_kvs": "perf_kvs_linearity",
    "perf_load_balance": "perf_load_balance",
    "perf_low_latency": "perf_low_latency",
}


def _load_cases_for_modules(module_names, case_id_arg):
    all_cases = expand_cases(module_names)
    run_cases = resolve_case_ids(case_id_arg, all_cases)
    valid_cases = []
    for name in run_cases:
        params = normalize_params(all_cases[name])
        n1 = params.get("N1", 1)
        n2 = params.get("N2", n1)
        if n1 >= n2 and n1 % n2 == 0:
            valid_cases.append(name)
    return valid_cases


def _get_modules_for_func(metafunc):
    markers = getattr(metafunc.function, "pytestmark", [])
    marker_names = set()
    for m in markers:
        marker_names.add(m.name if hasattr(m, "name") else m)
    if hasattr(metafunc.function, "mark"):
        pass
    func = metafunc.function
    for marker_name in marker_names:
        if marker_name in _MARKER_TO_MODULE:
            return [_MARKER_TO_MODULE[marker_name]]
    return None


def _build_parametrize_args(valid_cases, skip_set=None):
    if skip_set is None:
        skip_set = set()
    args = []
    for name in valid_cases:
        if name in skip_set:
            args.append(
                pytest.param(name, marks=pytest.mark.skip(reason="in SKIP_CASES"))
            )
        else:
            args.append(name)
    return args


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return

    case_id_arg = metafunc.config.getoption("--case_id")
    target_modules = _get_modules_for_func(metafunc)

    if target_modules is not None:
        valid_cases = _load_cases_for_modules(target_modules, case_id_arg)
    else:
        all_modules = load_testcase_files(testset=None)
        valid_cases = _load_cases_for_modules(all_modules, case_id_arg)
    metafunc.parametrize("case_name", _build_parametrize_args(valid_cases))


@pytest.fixture
def params_for_case(case_name):
    module_names = load_testcase_files(testset=None)
    all_cases = expand_cases(module_names)
    raw = all_cases[case_name]
    return normalize_params(raw)


@pytest.fixture
def data_dir(case_name, request):
    testset = request.config.getoption("--testset")
    return get_data_dir(case_name, testset)


@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")


@pytest.fixture(scope="session")
def postproc():
    return PostProcessor()


def _execute_step(case_name, params, case_dir, mode, dev_id, seed_val):
    case_safe = os.path.basename(case_dir)
    input_path = os.path.join(case_dir, f"{case_safe}_input.pt")
    cpu_path = os.path.join(case_dir, f"{case_safe}_cpu.pt")
    npu_path = os.path.join(case_dir, f"{case_safe}_npu.pt")
    for mode_ in mode:
        if mode_ == "gen":
            print(f"  [gen] Generating input data for {case_name}")
            tensors = generate_inputs(params, seed=seed_val)
            save_dict = {}
            for k, v in tensors.items():
                save_dict[k] = v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in params.items():
                if k not in save_dict:
                    save_dict[k] = v.cpu() if isinstance(v, torch.Tensor) else v
            torch.save(save_dict, input_path)
            print(f"  [gen] Saved to {input_path}")

        elif mode_ == "cpu":
            from core.cpu import run_cpu_golden

            if not os.path.exists(input_path):
                pytest.skip(f"Input file not found: {input_path}")
            print(f"  [cpu] Computing CPU golden for {case_name}")
            input_data = torch.load(input_path, map_location="cpu", weights_only=False)
            input_data["use_npu"] = False
            result = run_cpu_golden(input_data)
            torch.save(result, cpu_path)
            print(f"  [cpu] Saved to {cpu_path}")

        elif mode_ == "cpu_high":
            from core.cpu import run_cpu_golden

            if not os.path.exists(input_path):
                pytest.skip(f"Input file not found: {input_path}")
            print(
                f"  [cpu_high] Computing CPU golden (aclnnBatchMatMul) for {case_name}"
            )
            input_data = torch.load(input_path, map_location="cpu", weights_only=False)
            input_data["use_npu"] = True
            result = run_cpu_golden(input_data)
            torch.save(result, cpu_path)
            print(f"  [cpu_high] Saved to {cpu_path}")

        elif mode_ == "npu":
            from core.npu import run_npu

            if not os.path.exists(input_path):
                pytest.skip(f"Input file not found: {input_path}")
            print(f"  [npu] Computing NPU result for {case_name}")
            input_data = torch.load(input_path, map_location="cpu", weights_only=False)
            # input_data = {}
            # p = {}
            # for k, v in raw.items():
            #     if isinstance(v, torch.Tensor):
            #         input_data[k] = v
            #     else:
            #         p[k] = v
            # input_data["params"] = p

            # layout_kv = p.get("layout_kv", p.get("layout_q", "BNSD"))
            # if layout_kv.startswith("PA_") and "block_table" in p:
            #     input_data["block_table"] = p["block_table"]

            result = run_npu(input_data, device_id=dev_id)
            torch.save(result, npu_path)
            print(f"  [npu] Saved to {npu_path}")
            gc.collect()
            torch.npu.empty_cache()

        elif mode_ == "compare":
            from core.compare import compare_results

            if not os.path.exists(cpu_path):
                pytest.skip(f"CPU result not found: {cpu_path}")
            if not os.path.exists(npu_path):
                pytest.skip(f"NPU result not found: {npu_path}")
            print(f" cpu_path: {cpu_path}")
            print(f" npu_path: {npu_path}")
            cpu_result = torch.load(cpu_path, map_location="cpu", weights_only=False)
            npu_result = torch.load(npu_path, map_location="cpu", weights_only=False)
            cmp = compare_results(cpu_result, npu_result, case_name)
            return cmp

        else:
            pytest.fail(f"Unknown mode: {mode_}")

    return None


def _check_results(cmp):
    if cmp is None:
        return
    if not cmp["attn_out"]["passed"]:
        pytest.fail(
            f"Precision check failed: fail_ratio={cmp['attn_out']['fail_ratio'] * 100:.4f}%"
        )


@pytest.mark.func_rdv
@pytest.mark.ci
def test_func_rdv(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.func_redline
def test_func_redline(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.func_B250_redline
def test_func_B250_redline(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.func_stc
def test_func_stc(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.func_test
def test_func_test(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_rdv
def test_perf_rdv(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_redline
def test_perf_redline(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.debug
def test_debug(case_name, params_for_case, data_dir, mode, device_id, seed, postproc):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.func_L0
def test_func_L0(case_name, params_for_case, data_dir, mode, device_id, seed, postproc):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_batch
def test_perf_batch(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_mtp
def test_perf_mtp(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_kvs
def test_perf_kvs(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_load_balance
def perf_load_balance(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


@pytest.mark.perf_low_latency
def test_perf_low_latency(
    case_name, params_for_case, data_dir, mode, device_id, seed, postproc
):
    cmp = _execute_step(case_name, params_for_case, data_dir, mode, device_id, seed)
    if cmp is not None:
        postproc.record(case_name, cmp)
    _check_results(cmp)


def test_summary(postproc, mode):
    if mode == "compare":
        fail_cnt = postproc.print_summary()
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "result.csv"
        )
        postproc.save_csv(csv_path)


# if __name__ == "__main__":
#     main()
