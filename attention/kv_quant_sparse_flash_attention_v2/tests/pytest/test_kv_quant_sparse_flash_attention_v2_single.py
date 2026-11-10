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

import os
from pathlib import Path
import concurrent.futures
import pytest
import torch
import result_compare_method
from kv_quant_sparse_flash_attention_v2_paramset import get_model_params
import utils
import kv_quant_sparse_flash_attention_v2_golden

PT_SAVE_PATH = "./pt_files/"
DEVICE_ID = 0
RUN_NPU = True
SAVE_PT = False
RESULT_PATH = Path("result.xlsx")

PARAMSET_FILE = os.environ.get(
    "PARAMSET_FILE", "kv_quant_sparse_flash_attention_v2_paramset"
)
ENABLED_PARAMS = utils.load_paramset(PARAMSET_FILE)
PARAM_COMBINATION_SET = utils.combin_params(ENABLED_PARAMS)
case_id = 0


def execute_qsfa_v2(param_combination):
    global case_id
    params = utils.convert_param_combination_to_cs_format(param_combination)
    input_dict = kv_quant_sparse_flash_attention_v2_golden.generate_input_tensors(
        params
    )
    cpu_result, _, _ = kv_quant_sparse_flash_attention_v2_golden.compute_cpu(
        input_dict, params
    )
    test_data = {
        "Testcase_Name": params["case_name"],
        "params": params,
        "input": input_dict,
        "cpu_output": cpu_result,
    }
    if SAVE_PT:
        kv_quant_sparse_flash_attention_v2_golden._save_test_case(
            test_data, PT_SAVE_PATH
        )
    overall_result, compare_results = utils.qsfa_v2_run_npu(
        test_data, testcase_name=None, device_id=DEVICE_ID, result_path=RESULT_PATH
    )
    attn_info = compare_results.get("attn_out", {})
    fulfill_percent = attn_info.get("fulfill_percent", 0.0)
    case_id += 1
    return overall_result, fulfill_percent, test_data


@pytest.mark.ci
@pytest.mark.parametrize("param_combination", PARAM_COMBINATION_SET)
def test_kv_quant_sparse_flash_attention_v2(param_combination):
    test_data = None
    fulfill_percent = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(execute_qsfa_v2, param_combination)
        for future in concurrent.futures.as_completed([futures]):
            try:
                result, fulfill_percent, test_data = future.result()
                if result == "Failed":
                    case_name = test_data.get("Testcase_Name", "unknown")
                    pytest.fail(
                        f"用例名: {case_name}, 精度: {fulfill_percent:.4f}%",
                        pytrace=False,
                    )
            except Exception as e:
                params = test_data.get("params") if test_data else None
                case_name = (
                    test_data.get("Testcase_Name", "unknown")
                    if test_data
                    else "unknown"
                )
                if params:
                    utils.save_result(params, "Failed", "", RESULT_PATH)
                pytest.fail(f"用例名: {case_name}, 当前用例线程执行失败")


@pytest.mark.ci
def test_kv_quant_sparse_flash_attention_v2_model():
    from batch import kv_quant_sparse_flash_attention_v2_process as process

    arch = os.environ.get("QSFA_MODEL_ARCH", "A2").upper()
    combinations = utils.combin_params(get_model_params(arch))
    assert len(combinations) == 1
    params = utils.convert_param_combination_to_cs_format(combinations[0])

    expected_inputs = {
        "query_cache": (8, 128, 576),
        "key_cache": (7644, 128, 1, 656),
        "value_cache": (7644, 128, 1, 656),
        "sparse_indices": (8, 1, 2048),
        "block_table": (8, 256),
    }
    for name, shape in expected_inputs.items():
        assert tuple(params["shape_input"][name]) == shape, name
    assert len(params["actualseqlengths"]) == 8
    assert len(params["actualseqlengthskv"]) == 8

    inputs = kv_quant_sparse_flash_attention_v2_golden.generate_input_tensors(params)
    cpu_result, inputs, params = kv_quant_sparse_flash_attention_v2_golden.compute_cpu(
        inputs, params
    )
    assert cpu_result is not None
    for name, shape in expected_inputs.items():
        assert tuple(inputs[name].shape) == shape, name

    kv_dtype = torch.int8 if arch == "A2" else torch.float8_e4m3fn
    assert inputs["query_cache"].dtype == torch.bfloat16
    assert inputs["key_cache"].dtype == kv_dtype
    assert inputs["value_cache"].dtype == kv_dtype
    assert inputs["sparse_indices"].dtype == torch.int32
    assert inputs["block_table"].dtype == torch.int32

    npu_result = process.call_npu(inputs, params)
    expected_outputs = (
        ("attention", (8, 128, 512), torch.bfloat16),
        ("softmax_max", (1, 8, 128), torch.float32),
        ("softmax_sum", (1, 8, 128), torch.float32),
    )
    assert len(npu_result) == len(expected_outputs)
    for index, (name, shape, dtype) in enumerate(expected_outputs):
        assert tuple(cpu_result[index].shape) == shape, name
        assert tuple(npu_result[index].shape) == shape, name
        assert npu_result[index].dtype == dtype, name
        result, fulfill_percent = result_compare_method.check_result(
            cpu_result[index], npu_result[index]
        )
        assert result == "Pass", f"{name}: {fulfill_percent:.4f}%"

    assert torch.count_nonzero(npu_result[1].cpu()).item() > 0
    assert torch.count_nonzero(npu_result[2].cpu()).item() > 0


@pytest.mark.ci
@pytest.mark.parametrize("valid_count", [1, 20, 21, 22, 127, 128, 255, 256])
def test_rope_sparse_tail_zeroing(valid_count):
    device_name = torch.npu.get_device_name(0)
    if "910B" not in device_name and "910_93" not in device_name:
        pytest.skip("INT8 arch22 sparse-tail regression")

    from cann_ops_transformer.ops.attention.kv_quant_sparse_flash_attention_v2 import (
        kv_quant_sparse_flash_attention,
    )

    # Without explicit sinks, the default zero sink gives each valid token weight 1 / (valid_count + 1).
    # One nonzero value row exposes any contribution from the padded tail.
    latent = torch.zeros(256, 512, dtype=torch.int8)
    latent[0].fill_(1)
    rope = torch.zeros(256, 64, dtype=torch.bfloat16)
    scales = torch.ones(256, 4, dtype=torch.float32)
    packed = torch.cat(
        [latent.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)],
        dim=-1,
    ).view(torch.int8)
    key = packed.reshape(2, 128, 1, 656).npu()
    query = torch.zeros(1, 12, 576, dtype=torch.bfloat16)
    indices = torch.arange(256, dtype=torch.int32).reshape(1, 1, 256)
    indices[..., valid_count:] = -1

    attention, maximum, total = kv_quant_sparse_flash_attention(
        query.npu(),
        key,
        key,
        indices.npu(),
        576**-0.5,
        2,
        2,
        block_table=torch.arange(2, dtype=torch.int32).reshape(1, 2).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([256], dtype=torch.int32).npu(),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        return_softmax_lse=True,
    )
    torch.npu.synchronize()

    expected_attention = torch.full(
        (12, 512), 1 / (valid_count + 1), dtype=torch.bfloat16
    ).float()
    torch.testing.assert_close(
        attention.cpu().float().reshape(12, 512),
        expected_attention,
        rtol=0.01,
        atol=0.001,
    )
    torch.testing.assert_close(
        maximum.cpu().reshape(12), torch.zeros(12), rtol=0, atol=0
    )
    torch.testing.assert_close(
        total.cpu().reshape(12),
        torch.full((12,), valid_count + 1, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
