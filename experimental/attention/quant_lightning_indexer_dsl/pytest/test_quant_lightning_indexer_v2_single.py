# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import importlib
import itertools
import os
from pathlib import Path

import pytest
import torch
import torch_npu

import quant_lightning_indexer_v2_golden
import result_compare_method
from qliv2_test_utils import PARAM_NAMES, QliV2ResultWriter, ensure_comparison_passed


paramset = os.environ.get("QLIV2_PARAMSET", "default")
if paramset != "default":
    raise ValueError(f"unsupported paramset: {paramset}")
ENABLED_PARAMSETS = importlib.import_module(
    "test_quant_lightning_indexer_v2_paramset"
).ENABLED_PARAMSETS
requested_names = {
    name.strip()
    for name in os.environ.get("QLIV2_CASE_NAMES", "").split(",")
    if name.strip()
}
matched_names = set()
SAVE_PT_DIR = os.environ.get("QLIV2_SINGLE_SAVE_PT_DIR", "").strip()
RESULT_PATH = os.environ.get("QLIV2_SINGLE_RESULT_PATH", "").strip()


param_names = list(PARAM_NAMES) + ["run_mode"]
param_combinations = []
for paramset_name, params in ENABLED_PARAMSETS:
    missing = [name for name in param_names if name not in params]
    if missing:
        raise ValueError(f"{paramset_name} is missing parameters: {missing}")
    combinations = itertools.product(*(params[name] for name in param_names))
    combinations = list(combinations)
    for combo_index, combo in enumerate(combinations, start=1):
        case_name = (
            paramset_name
            if len(combinations) == 1
            else f"{paramset_name}_{combo_index:03d}"
        )
        if requested_names and not (
            paramset_name in requested_names or case_name in requested_names
        ):
            continue
        matched_names.update((paramset_name, case_name))
        values = dict(zip(param_names, combo))
        values["case_name"] = case_name
        param_combinations.append(pytest.param(values, id=case_name))

unknown_names = sorted(requested_names - matched_names)
if unknown_names:
    raise ValueError(f"unknown case(s): {unknown_names}")


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_qliv2(param_combinations):
    torch_npu.npu.set_device(int(os.environ.get("QLIV2_DEVICE_ID", "0")))
    run_mode = (
        os.environ.get("QLIV2_RUN_MODE", param_combinations["run_mode"]).strip().lower()
    )
    if run_mode not in ("eager", "acl_graph"):
        raise ValueError("run mode must be eager or acl_graph")
    test_data = tuple(param_combinations[name] for name in PARAM_NAMES)
    case_name = QliV2ResultWriter.case_name(
        test_data, explicit_name=param_combinations["case_name"]
    )

    if run_mode == "acl_graph":
        from quant_lightning_indexer_v2_acl_graph import qliv2_output_acl_graph

        cpu_result, npu_result, topk_value, cpu_value, npu_value = (
            qliv2_output_acl_graph(test_data)
        )
    elif SAVE_PT_DIR:
        case_data = quant_lightning_indexer_v2_golden.generate_qliv2_test_data(
            test_data
        )
        case_path = Path(SAVE_PT_DIR) / f"{case_name}.pt"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case_data, case_path)
        from batch.quant_lightning_indexer_v2_pt_loadprocess import test_qliv2_process

        outputs = test_qliv2_process(case_path, device_id=0)
        cpu_result, npu_result, topk_value, cpu_value, npu_value, _, _ = outputs
    else:
        cpu_result, npu_result, topk_value, cpu_value, npu_value = (
            quant_lightning_indexer_v2_golden.qliv2_output_single(test_data)
        )

    result, fulfill_percent = result_compare_method.check_result(
        cpu_result,
        npu_result,
        topk_value,
        param_combinations["output_idx_offset"],
        test_data,
        cpu_value,
        npu_value,
    )
    value_result, value_percent = "N/A", 0
    if bool(param_combinations["return_value"]):
        value_result, value_percent = result_compare_method.check_result_return_value(
            cpu_value,
            npu_value,
            test_data,
            cpu_result,
            npu_result,
            topk_value,
            param_combinations["output_idx_offset"],
        )
    if RESULT_PATH:
        QliV2ResultWriter.append(
            RESULT_PATH,
            QliV2ResultWriter.row(
                case_name,
                test_data,
                result,
                fulfill_percent,
                value_result,
                value_percent,
            ),
        )
    ensure_comparison_passed(
        case_name, result, fulfill_percent, value_result, value_percent
    )


@pytest.mark.npu
@pytest.mark.parametrize("backend", ["acl_graph"])
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("automatic", [False])
def test_qli_graph(backend, layout, return_value, automatic):
    import json
    import cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl
    from indexer_test_runtime import check_graph_sequence
    from qli_test_utils.golden import quant_lightning_indexer_golden
    from qli_test_utils.result_compare import compare_qli_outputs

    torch.npu.set_device(int(os.environ.get("QLIV2_DEVICE_ID", "1")))
    output = Path(os.environ.get("INDEXER_GRAPH_OUTPUT", "tmp/stage5_6"))
    output = (
        output
        / f"qli_{backend}_{layout}_values{int(return_value)}_auto{int(automatic)}"
    )
    output.mkdir(parents=True, exist_ok=True)
    phases = {}
    for phase, seed in (("initial", 5201), ("updated", 5202)):
        generator = torch.Generator().manual_seed(seed)
        key = torch.randint(
            0, 256, (18, 128, 1, 64), dtype=torch.uint8, generator=generator
        )
        scale = torch.randint(
            125, 130, (18, 128, 1, 2, 2), dtype=torch.uint8, generator=generator
        )
        if layout == "TND":
            key = (
                key.reshape(2, 1152, 1, 64)[:, :1025].reshape(2050, 1, 64).contiguous()
            )
            scale = (
                scale.reshape(2, 1152, 1, 2, 2)[:, :1025]
                .reshape(2050, 1, 2, 2)
                .contiguous()
            )
        phases[phase] = {
            "q": torch.randint(
                0, 256, (4, 32, 64), dtype=torch.uint8, generator=generator
            ),
            "k": key,
            "w": torch.rand((4, 32), generator=generator),
            "descale_q": torch.randint(
                125, 130, (4, 32, 2, 2), dtype=torch.uint8, generator=generator
            ),
            "descale_k": scale,
            "cu_seqlens_q": torch.tensor([0, 2, 4], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, 1025, 2050], dtype=torch.int32),
            "seqused_q": torch.tensor([2, 2], dtype=torch.int32),
            "seqused_k": torch.tensor(
                [1025, 1025] if phase == "initial" else [503, 519], dtype=torch.int32
            ),
            "cmp_residual_k": torch.tensor([0, 1], dtype=torch.int32),
            "block_table": torch.arange(18, dtype=torch.int32).reshape(2, 9)
            if layout == "PA_BBND"
            else None,
        }
    namespace = torch.ops.cann_ops_transformer.ds41
    options = dict(
        topk=512,
        quant_mode=1,
        max_seqlen_q=2,
        mask_mode=3,
        cmp_ratio=2,
        layout_k=layout,
        return_value=return_value,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )

    def metadata(inputs):
        return namespace.quant_lightning_indexer_metadata(
            inputs["cu_seqlens_q"],
            inputs["cu_seqlens_k"],
            inputs["seqused_q"],
            inputs["seqused_k"],
            inputs["cmp_residual_k"],
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=512,
            max_seqlen_q=2,
            mask_mode=3,
            cmp_ratio=2,
            layout_k=layout,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
        )

    def function(inputs):
        schedule = None if automatic else metadata(inputs)
        outputs = namespace.quant_lightning_indexer(
            **inputs, metadata=schedule, **options
        )
        return outputs if automatic else (*outputs, schedule)

    names = (
        "sparse_indices",
        "sparse_values",
        "candidate_block_indices",
        "candidate_block_length",
    )
    expected, schedules = {}, {}
    for phase, inputs in phases.items():
        golden_args = dict(inputs)
        golden_args["q_descale"] = golden_args.pop("descale_q")
        golden_args["k_descale"] = golden_args.pop("descale_k")
        golden = quant_lightning_indexer_golden(**golden_args, **options)
        device_inputs = {
            name: value.npu() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        schedules[phase] = metadata(device_inputs).cpu()
        expected[phase] = tuple(value.cpu() for value in function(device_inputs))
        report = compare_qli_outputs(golden, dict(zip(names, expected[phase][:4])))
        torch.save(
            {
                "inputs": inputs,
                "metadata": schedules[phase],
                "golden": {name: golden[name] for name in names},
                "eager": expected[phase],
            },
            output / f"baseline_{phase}.pt",
        )
        (output / f"baseline_{phase}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        assert report["pass"], f"{phase}: {report!r}"
    assert not torch.equal(schedules["initial"], schedules["updated"])
    assert not torch.equal(expected["initial"][0], expected["updated"][0])

    def check_output(phase, values):
        assert len(values) == len(expected[phase])
        for actual, reference in zip(values, expected[phase]):
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            assert torch.equal(actual, reference)

    torch.save(
        {"metadata": schedules, "outputs": expected}, output / "eager_reference.pt"
    )
    check_graph_sequence(
        function, phases["initial"], phases["updated"], backend, check_output, output
    )
