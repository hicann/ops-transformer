# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import itertools
import os
from pathlib import Path

import pytest
import torch

from qsli_test_utils import compare_qsli_case, generate_qsli_test_data, run_qsli_case
from test_quant_sparse_lightning_indexer_paramset import ENABLED_PARAMSETS

PARAM_NAMES = (
    "batch_size",
    "query_length",
    "sequence_length",
    "cmp_ratio",
    "storage_mode",
    "mask_mode",
    "topk",
    "candidate_block_length",
    "output_idx_offset",
    "seed",
)
REQUESTED = {
    x.strip() for x in os.environ.get("QSLI_CASE_NAMES", "").split(",") if x.strip()
}
SAVE_PT_DIR = os.environ.get("QSLI_SINGLE_SAVE_PT_DIR", "").strip()

PARAMS = []
matched = set()
for paramset_name, values in ENABLED_PARAMSETS:
    names = PARAM_NAMES + tuple(name for name in values if name not in PARAM_NAMES)
    combinations = itertools.product(*(values[name] for name in names))
    for index, combination in enumerate(combinations, 1):
        case_name = paramset_name if index == 1 else f"{paramset_name}_{index:03d}"
        if REQUESTED and paramset_name not in REQUESTED and case_name not in REQUESTED:
            continue
        matched.update((paramset_name, case_name))
        PARAMS.append(
            pytest.param(dict(zip(names, combination)), case_name, id=case_name)
        )
if REQUESTED - matched:
    raise ValueError(f"unknown case(s): {sorted(REQUESTED - matched)}")


@pytest.mark.ci
@pytest.mark.npu
@pytest.mark.parametrize("params,case_name", PARAMS)
def test_qsli_single(params, case_name):
    case_data = generate_qsli_test_data(params)
    if SAVE_PT_DIR:
        target = Path(SAVE_PT_DIR) / f"{case_name}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case_data, target)
        print(f"saved: {target}")
    actual = run_qsli_case(case_data, int(os.environ.get("QSLI_DEVICE_ID", "0")))
    result, fulfill = compare_qsli_case(case_data, actual)
    print(f"result={result}, fulfill_percent={fulfill}")


@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("return_value", [False, True])
def test_qsli_torch_extension_metadata(layout, return_value, monkeypatch):
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl
    from qsli_test_utils import _to_npu_tensors
    from ops.quant_sparse_lightning_indexer_dsl import (
        quant_sparse_lightning_indexer as direct,
    )
    from ops.quant_sparse_lightning_indexer_metadata_dsl import (
        quant_sparse_lightning_indexer_metadata as direct_metadata,
    )

    torch.npu.set_device(int(os.environ.get("QSLI_DEVICE_ID", "0")))
    data = generate_qsli_test_data(
        dict(
            batch_size=2,
            query_length=2,
            sequence_length=512,
            layout_k=layout,
            return_value=return_value,
            cmp_ratio=2,
            cmp_residual_k=[0, 1],
            candidate_block_length=64,
            output_idx_offset=True,
            seed=4202,
        )
    )
    tensors = _to_npu_tensors(data["tensors"])
    params = data["params"]
    namespace = torch.ops.cann_ops_transformer.ds41
    meta_args = (
        tensors["candidate_length"],
        tensors["cu"],
        tensors["cu_k"],
        tensors["used_q"],
        tensors["used_k"],
        tensors["residual"],
    )
    meta_options = dict(
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        quant_mode=1,
        candidate_block_size=8,
        max_seqlen_q=2,
        mask_mode=params["mask_mode"],
        cmp_ratio=2,
    )
    metadata = namespace.quant_sparse_lightning_indexer_metadata(
        *meta_args, layout_k=layout, **meta_options
    )
    expected_metadata = direct_metadata(*meta_args, layout_k=layout, **meta_options)
    assert torch.equal(metadata.cpu(), expected_metadata.cpu())
    positional = (
        tensors["q"],
        tensors["k"],
        tensors["weights"],
        tensors["descale_q"],
        tensors["candidate"],
        tensors["candidate_length"],
    )
    options = dict(
        descale_k=tensors.get("descale_k"),
        topk=512,
        quant_mode=1,
        candidate_block_size=8,
        max_seqlen_q=2,
        mask_mode=params["mask_mode"],
        cmp_ratio=2,
        cu_seqlens_q=tensors["cu"],
        cu_seqlens_k=tensors["cu_k"],
        seqused_q=tensors["used_q"],
        seqused_k=tensors["used_k"],
        cmp_residual_k=tensors["residual"],
        block_table=tensors["block_table"],
        output_idx_offset=tensors["offset"],
        return_value=return_value,
    )
    explicit = namespace.quant_sparse_lightning_indexer(
        *positional, metadata=metadata, layout_k=layout, **options
    )
    expected = direct(*positional, metadata=metadata, layout_k=layout, **options)
    for actual, reference in zip(explicit, expected):
        assert torch.equal(actual.cpu(), reference.cpu())
    result = dict(
        zip(("sparse_indices", "sparse_values"), (value.cpu() for value in explicit))
    )
    compare_qsli_case(data, result)
    from unittest.mock import Mock

    metadata_call = Mock(
        side_effect=AssertionError("unexpected automatic metadata call")
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            namespace, "quant_sparse_lightning_indexer_metadata", metadata_call
        )
        with pytest.raises(ValueError, match="metadata is required"):
            namespace.quant_sparse_lightning_indexer(
                *positional, layout_k=layout, **options
            )
    metadata_call.assert_not_called()


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize(
    "candidate_count,sequence_length,mask_mode",
    [
        (0, 512, 0),
        (1, 5, 0),
        (63, 520, 3),
        (64, 520, 3),
        (65, 520, 3),
    ],
)
def test_qsli_tile_smoke(layout, candidate_count, sequence_length, mask_mode):
    data = generate_qsli_test_data(
        dict(
            batch_size=2,
            query_length=2,
            sequence_length=sequence_length,
            layout_k=layout,
            candidate_block_length=candidate_count,
            topk=64,
            cmp_ratio=1,
            mask_mode=mask_mode,
            return_value=True,
            seed=4301,
        )
    )
    actual = run_qsli_case(data, int(os.environ.get("QSLI_DEVICE_ID", "0")))
    compare_qsli_case(data, actual)


@pytest.mark.npu
@pytest.mark.parametrize(
    "storage_mode", ["key_dim0_noncontiguous", "scale_dim0_noncontiguous"]
)
def test_qsli_tnd_stride_smoke(storage_mode):
    data = generate_qsli_test_data(
        dict(
            batch_size=2,
            query_length=2,
            sequence_length=513,
            layout_k="TND",
            candidate_block_length=65,
            topk=64,
            cmp_ratio=1,
            mask_mode=0,
            return_value=True,
            storage_mode=storage_mode,
            seed=4302,
        )
    )
    actual = run_qsli_case(data, int(os.environ.get("QSLI_DEVICE_ID", "0")))
    compare_qsli_case(data, actual)


@pytest.mark.npu
@pytest.mark.parametrize("backend", ["acl_graph"])
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize("automatic", [False])
def test_qsli_graph(backend, layout, return_value, automatic):
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl
    from indexer_test_runtime import check_graph_sequence
    from qsli_test_utils import convert_qsli_case_to_tnd, compare_qsli_all_rows

    torch.npu.set_device(int(os.environ.get("QSLI_DEVICE_ID", "1")))
    initial = generate_qsli_test_data(
        dict(
            batch_size=2,
            query_length=2,
            sequence_length=1025,
            layout_k="PA_BBND",
            return_value=return_value,
            cmp_ratio=2,
            mask_mode=3,
            cu_seqlens_q=[0, 2, 4],
            cu_seqlens_k=[0, 1025, 2050],
            seqused_q=[2, 2],
            seqused_k=[1025, 1025],
            cmp_residual_k=[0, 1],
            candidate_block_length=129,
            output_idx_offset=False,
            seed=5101,
        ),
        defer_golden=True,
    )
    updated = generate_qsli_test_data(
        {
            **initial["params"],
            "seed": 5102,
            "q_scale_datarange": [0.5, 2],
            "k_scale_datarange": [0.5, 2],
        },
        defer_golden=True,
    )
    updated["tensors"]["used_q"][:] = torch.tensor([2, 2], dtype=torch.int32)
    updated["tensors"]["used_k"][:] = torch.tensor([503, 519], dtype=torch.int32)
    updated["tensors"]["candidate_length"][:, 0] = torch.tensor(
        [63, 63, 65, 65], dtype=torch.int32
    )
    for row in range(4):
        count = 63 if row < 2 else 65
        updated["tensors"]["candidate"][row, 0, :count] = torch.arange(
            count, dtype=torch.int32
        )
    if layout == "TND":
        initial, updated = (
            convert_qsli_case_to_tnd(case) for case in (initial, updated)
        )
    namespace = torch.ops.cann_ops_transformer.ds41

    def metadata(tensors):
        return namespace.quant_sparse_lightning_indexer_metadata(
            tensors["candidate_length"],
            tensors["cu"],
            tensors["cu_k"],
            tensors["used_q"],
            tensors["used_k"],
            tensors["residual"],
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=512,
            quant_mode=1,
            candidate_block_size=8,
            max_seqlen_q=2,
            mask_mode=3,
            cmp_ratio=2,
            layout_k=layout,
        )

    def function(tensors):
        schedule = None if automatic else metadata(tensors)
        result = namespace.quant_sparse_lightning_indexer(
            tensors["q"],
            tensors["k"],
            tensors["weights"],
            tensors["descale_q"],
            tensors["candidate"],
            tensors["candidate_length"],
            descale_k=tensors["descale_k"],
            cu_seqlens_q=tensors["cu"],
            cu_seqlens_k=tensors["cu_k"],
            seqused_q=tensors["used_q"],
            seqused_k=tensors["used_k"],
            cmp_residual_k=tensors["residual"],
            block_table=tensors["block_table"],
            metadata=schedule,
            topk=512,
            quant_mode=1,
            candidate_block_size=8,
            max_seqlen_q=2,
            mask_mode=3,
            cmp_ratio=2,
            layout_k=layout,
            return_value=return_value,
        )
        return result if automatic else (*result, schedule)

    cases = {"initial": initial, "updated": updated}
    expected, schedules = {}, {}
    names = (
        "q",
        "k",
        "weights",
        "descale_q",
        "descale_k",
        "candidate",
        "candidate_length",
        "cu",
        "cu_k",
        "used_q",
        "used_k",
        "residual",
        "block_table",
    )
    inputs = {
        phase: {name: case["tensors"].get(name) for name in names}
        for phase, case in cases.items()
    }
    for phase, tensors in inputs.items():
        device_inputs = {
            name: value.npu() if isinstance(value, torch.Tensor) else value
            for name, value in tensors.items()
        }
        schedules[phase] = metadata(device_inputs).cpu()
        expected[phase] = tuple(value.cpu() for value in function(device_inputs))
        compare_qsli_all_rows(
            cases[phase],
            dict(zip(("sparse_indices", "sparse_values"), expected[phase][:2])),
        )
    assert not torch.equal(schedules["initial"], schedules["updated"])
    assert not torch.equal(expected["initial"][0], expected["updated"][0])

    def check_output(phase, values):
        assert len(values) == len(expected[phase])
        for actual, reference in zip(values, expected[phase]):
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            assert torch.equal(actual, reference)

    output = Path(os.environ.get("INDEXER_GRAPH_OUTPUT", "tmp/stage5_6"))
    output = (
        output
        / f"qsli_{backend}_{layout}_values{int(return_value)}_auto{int(automatic)}"
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"metadata": schedules, "outputs": expected}, output / "eager_reference.pt"
    )
    check_graph_sequence(
        function, inputs["initial"], inputs["updated"], backend, check_output, output
    )
