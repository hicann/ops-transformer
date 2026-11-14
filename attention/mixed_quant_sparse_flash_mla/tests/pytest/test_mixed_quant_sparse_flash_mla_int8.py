# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""INT8 CSA CPU checks and device regression using the required ATK metrics.

ATK is required only when device tests are enabled. This small device regression
does not replace full L0 acceptance coverage or its repeated executions.
"""

import math
import os
import random
import struct

import numpy as np
import pytest
import torch

import check_valid_param
import mixed_quant_sparse_flash_mla_golden as golden


LAYOUTS = [("BSND", "BSND"), ("BSND", "PA_BBND"), ("TND", "TND"), ("TND", "PA_BBND")]
DTYPES = [torch.float16, torch.bfloat16]


def case_params(layout_q, layout_kv, dtype):
    """Use the existing gen_data schema, with padding queries and unequal KV lengths."""
    return dict(
        Testcase_Name=f"int8_csa_{layout_q}_{layout_kv}_{dtype}",
        layout_q=layout_q,
        layout_kv=layout_kv,
        q_type=dtype,
        ori_kv_type=torch.int8,
        cmp_kv_type=torch.int8,
        B=2,
        S1=3,
        S2=68,
        T1=6 if layout_q == "TND" else None,
        N1=4,
        N2=1,
        D=512,
        K=16,
        block_num1=7,
        block_num2=3,
        block_size1=16,
        block_size2=16,
        cu_seqlens_q=[0, 3, 6],
        seqused_q=[3, 2],
        seqused_ori_kv=[32, 68],
        seqused_cmp_kv=[8, 17],
        cu_seqlens_ori_kv=[0, 32, 100],
        cu_seqlens_cmp_kv=[0, 8, 25],
        cmp_residual_kv=[1, 0],
        softmax_scale=512**-0.5,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=7,
        ori_win_right=0,
        quant_mode=1,
        tile_size=64,
        rope_head_dim=64,
        template_run_mode="CSA",
        topk_value_mode=1,
        return_softmax_lse=True,
        batch_consistency=False,
        q_datarange=[-0.25, 0.25],
        ori_kv_datarange=[-4, 4],
        cmp_kv_datarange=[-4, 4],
    )


@pytest.fixture(autouse=True)
def seeded_inputs():
    # Isolate deterministic input generation from other pytest modules.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        np.random.seed(1701)
        random.seed(1701)
        yield
    np.random.set_state(numpy_state)
    random.setstate(python_state)


def scalar_decode(record):
    """Independent signed-byte/BF16 decoding using struct, without Torch views."""
    raw = bytes(int(value) & 255 for value in record.flatten().tolist())
    fields = struct.unpack("<64H448b7H18x", raw)

    def bf16(bits):
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]

    rope = [bf16(value) for value in fields[:64]]
    scales = [bf16(value) for value in fields[512:]]
    nope = [value * scales[index // 64] for index, value in enumerate(fields[64:512])]
    return np.array(nope + rope, dtype=np.float64)


def physical_record(op, stream, batch, token):
    records, layout = op[f"{stream}_kv"], op["layout_kv"]
    if layout == "PA_BBND":
        block = int(op[f"{stream}_block_table"][batch, token // records.shape[1]])
        return records[block, token % records.shape[1], 0]
    if layout == "TND":
        return records[int(op[f"cu_seqlens_{stream}_kv"][batch]) + token, 0]
    return records[batch, token, 0]


def dense_oracle(data, round_kv=True):
    """Independent dense FP64 softmax, optionally after 16-bit KV rounding.

    CPU integration checks retain their existing rounded-KV behavior by default.
    Device comparisons pass round_kv=False to preserve raw decoded INT8/BF16
    values in the high-precision Golden, without a Q-dtype intermediate cast.
    """
    op = data["op_input"]
    q = op["q"]
    output = torch.zeros(q.shape, dtype=torch.float64)
    heads = q.shape[-2]
    lse_shape = (
        (q.shape[0], 1, q.shape[1], heads)
        if op["layout_q"] == "BSND"
        else (1, q.shape[0], heads)
    )
    lse = torch.zeros(lse_shape, dtype=torch.float64)
    for batch in range(len(op["seqused_q"])):
        oq, oc = int(op["seqused_ori_kv"][batch]), int(op["seqused_cmp_kv"][batch])
        qlen = int(op["seqused_q"][batch])
        caches = {}
        for stream, length in (("ori", oq), ("cmp", oc)):
            rows = [
                scalar_decode(physical_record(op, stream, batch, t))
                for t in range(length)
            ]
            decoded = np.array(rows, dtype=np.float64).reshape(length, 512)
            caches[stream] = (
                torch.from_numpy(decoded).to(q.dtype).double().numpy()
                if round_kv
                else decoded
            )
        for s in range(qlen):
            query_index = (
                (batch, s)
                if op["layout_q"] == "BSND"
                else (int(op["cu_seqlens_q"][batch]) + s,)
            )
            anchor = oq - qlen + s
            positions = list(range(oq))
            if op["ori_mask_mode"] == 3:
                positions = [t for t in positions if t <= anchor]
            if op["ori_mask_mode"] == 4:
                positions = [
                    t
                    for t in positions
                    if (op["ori_win_left"] == -1 or t >= anchor - op["ori_win_left"])
                    and (op["ori_win_right"] == -1 or t <= anchor + op["ori_win_right"])
                ]
            visible = oc
            if op["cmp_mask_mode"] == 3:
                residual = (
                    int(op["cmp_residual_kv"][batch])
                    if op["cmp_residual_kv"] is not None
                    else 0
                )
                visible = max(
                    0,
                    min(
                        oc,
                        (oc * op["cmp_ratio"] + residual - qlen + s + 1)
                        // op["cmp_ratio"],
                    ),
                )
            indices = op["cmp_sparse_indices"][query_index][0].tolist()
            count = min(len(indices), visible)
            if op["cmp_topk_length"] is not None:
                count = min(count, int(op["cmp_topk_length"][query_index][0]))
            selected = []
            for value in indices[:count]:
                if value == -1:
                    break
                if 0 <= value < visible:
                    selected.append(value)
            keys = np.concatenate(
                (caches["ori"][positions], caches["cmp"][selected]), axis=0
            )
            if len(keys) == 0:
                continue
            for head in range(heads):
                scores = (
                    keys @ q[query_index][head].double().numpy() * op["softmax_scale"]
                )
                sink = float(op["sinks"][head])
                maximum = max(float(scores.max()), sink)
                weights = np.exp(scores - maximum)
                denominator = math.fsum(weights.tolist()) + math.exp(sink - maximum)
                output[query_index][head] = torch.from_numpy(
                    weights @ keys / denominator
                )
                lse_index = (
                    (batch, 0, s, head)
                    if op["layout_q"] == "BSND"
                    else (0, query_index[0], head)
                )
                lse[lse_index] = maximum + math.log(denominator)
    return output, lse


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize(
    "ori_mask,cmp_mask", [(o, c) for o in (0, 3, 4) for c in (0, 3)]
)
def test_generated_int8_inputs_match_independent_math(
    dtype, layout_q, layout_kv, ori_mask, cmp_mask
):
    params = case_params(layout_q, layout_kv, dtype)
    params.update(ori_mask_mode=ori_mask, cmp_mask_mode=cmp_mask)
    if ori_mask != 4:
        params.update(ori_win_left=-1, ori_win_right=-1)
    if cmp_mask == 0:
        params.update(cmp_residual_kv=None, cmp_kv_topk_mode="fullK")
    check_valid_param.check_valid_param(params)
    data = golden.gen_data(params)
    assert data["op_input"]["ori_kv"].dtype == torch.int8
    assert data["op_input"]["cmp_kv"].dtype == torch.int8
    assert data["cpu_output"].dtype == dtype
    assert_independent_math(data)


def assert_independent_math(data):
    dtype = data["op_input"]["q"].dtype
    expected, expected_lse = dense_oracle(data)
    # CPU self-check tolerance includes the existing Golden's rounded exp/PV.
    # This is deliberately separate from device acceptance and formal L0 rules.
    atol = 0.004 if dtype == torch.bfloat16 else 0.0006
    torch.testing.assert_close(
        data["cpu_output"].double(), expected, rtol=0.008, atol=atol
    )
    torch.testing.assert_close(
        data["cpu_lse"].double(), expected_lse, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize("ratio", [1, 128])
def test_compression_ratio_limits(dtype, layout_q, layout_kv, ratio):
    params = case_params(layout_q, layout_kv, dtype)
    params.update(cmp_ratio=ratio, cmp_residual_kv=None)
    if ratio == 128:
        params.update(
            S2=513,
            seqused_ori_kv=[259, 513],
            seqused_cmp_kv=[2, 4],
            cu_seqlens_ori_kv=[0, 259, 772],
            cu_seqlens_cmp_kv=[0, 2, 6],
            block_num1=50,
            block_num2=2,
            cmp_residual_kv=[3, 1],
        )
    assert_independent_math(golden.gen_data(params))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero_topk", [False, True])
def test_sparse_prefix_duplicates_and_sentinel(dtype, zero_topk):
    data = golden.gen_data(case_params("BSND", "BSND", dtype), generate_golden=False)
    # Repeated indices contribute twice; -1 terminates the prefix, including
    # any otherwise valid entries after it. Explicit topk=0 excludes CMP KV.
    data["op_input"]["cmp_sparse_indices"][..., :4] = torch.tensor(
        [1, 1, -1, 0], dtype=torch.int32
    )
    if zero_topk:
        lengths = torch.zeros((2, 3, 1), dtype=torch.int32)
        data["op_input"]["cmp_topk_length"] = lengths
    golden.generate_cpu_golden(data)
    assert_independent_math(data)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_no_valid_kv_has_zero_output_and_lse_even_with_sinks(
    dtype, layout_q, layout_kv
):
    data = golden.gen_data(
        case_params(layout_q, layout_kv, dtype), generate_golden=False
    )
    for name in ("seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
        data["op_input"][name] = torch.zeros(2, dtype=torch.int32)
    golden.generate_cpu_golden(data)
    assert torch.count_nonzero(data["cpu_output"]) == 0
    assert torch.count_nonzero(data["cpu_lse"]) == 0


def test_decode_all_signed_values_and_seven_groups():
    payload = np.resize(np.arange(-128, 128, dtype=np.int8), 448)
    rope = np.resize(np.array([0x3F80, 0xC000], dtype="<u2"), 64)  # BF16 +1/-2.
    scales = np.array(
        [0x3F80, 0x4000, 0x3F00, 0x4080, 0x3E80, 0x4100, 0x3E00], dtype="<u2"
    )
    raw = rope.tobytes() + payload.tobytes() + scales.tobytes() + bytes([255] * 18)
    record = torch.from_numpy(np.frombuffer(raw, dtype=np.int8).copy()).reshape(
        1, 1, 1, 608
    )
    actual = golden.decode_int8_cache(record, "BSND", [1])[0, 0, 0].double().numpy()
    np.testing.assert_array_equal(actual, scalar_decode(record[0, 0, 0]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_golden_reads_operator_bytes_and_ignores_padding(dtype):
    params = case_params("BSND", "BSND", dtype)
    params.update(ori_mask_mode=0, ori_win_left=-1, ori_win_right=-1)
    data = golden.gen_data(params)
    original = data["cpu_output"].clone()
    data["op_input"]["ori_kv"][..., 590:] = 127
    golden.generate_cpu_golden(data)
    torch.testing.assert_close(data["cpu_output"], original, rtol=0, atol=0)
    data["op_input"]["ori_kv"][..., 128:576] = 100
    golden.generate_cpu_golden(data)
    assert not torch.equal(data["cpu_output"], original)


@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_input_only_and_lse_disabled(layout_q, layout_kv):
    params = case_params(layout_q, layout_kv, torch.float16)
    params["return_softmax_lse"] = False
    data = golden.gen_data(params, generate_golden=False)
    assert data["cpu_output"] is None and data["cpu_lse"] is None
    golden.generate_cpu_golden(data)
    assert data["cpu_output"].shape == data["op_input"]["q"].shape
    assert data["cpu_lse"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"q_type": torch.float32},
        {"cmp_kv_type": torch.float8_e4m3fn},
        {"ori_kv_type": torch.float8_e4m3fn},
        {"quant_mode": 0},
        {"quant_mode": 2},
        {"quant_mode": 3},
        {"quant_mode": 4},
        {"tile_size": 32},
        {"rope_head_dim": 32},
        {"template_run_mode": "HCA"},
        {"topk_value_mode": 2},
        {"D": 576},
        {"N2": 2},
        {"layout_q": "BNSD"},
        {"layout_kv": "BNBD"},
        {"batch_consistency": True},
        {"ori_mask_mode": 2},
        {"cmp_mask_mode": 4},
        {"cmp_ratio": 0},
        {"N1": 129},
    ],
)
def test_invalid_int8_combinations_fail(change):
    params = case_params("BSND", "BSND", torch.float16)
    params.update(change)
    with pytest.raises(ValueError):
        golden.gen_data(params, generate_golden=False)


def assert_l0_dual_benchmark(actual, benchmark, high_precision, output_name):
    """Apply the installed ATK comparator; missing ATK is an enabled-test failure.

    These finite-data device regressions use the requirement's L0 error ratios.
    They do not claim 5000-case/50-run coverage or full commercial acceptance.
    """
    from atk.configs.case_config import CaseConfig
    from atk.tasks.post_process.cv_fused_double_benchmark_compare import (
        CvFusedDoubleBenchmarkAccuracyCompare,
    )

    actual, benchmark, high_precision = (
        value.detach().cpu() for value in (actual, benchmark, high_precision)
    )
    assert actual.shape == benchmark.shape == high_precision.shape, output_name
    assert actual.dtype == benchmark.dtype, output_name
    assert high_precision.dtype == torch.float64, output_name
    for name, value in (
        ("actual", actual),
        ("same-precision benchmark", benchmark),
        ("FP64 Golden", high_precision),
    ):
        assert torch.isfinite(value).all().item(), (
            f"{output_name}: {name} must be finite for these cases"
        )
    comparator = CvFusedDoubleBenchmarkAccuracyCompare(
        CaseConfig(
            standard={
                "acc": {
                    "cv_fused_double_benchmark": {
                        "max_re_ratio": 10.0,
                        "avg_re_ratio": 2.0,
                        "root_mean_squared_ratio": 2.0,
                        "number_count_radio": 2.0,
                    }
                }
            }
        )
    )
    comparator.benchmark_standard.init_by_dtype(actual.dtype)
    actual_result = comparator.compute_with_golden(actual, high_precision)
    benchmark_result = comparator.compute_with_golden(benchmark, high_precision)
    actual_result.do_summary(benchmark_result)
    passed, message = actual_result.get_result_msg()
    assert passed, (
        f"{output_name}: ATK L0 double-benchmark failed: {message}; {actual_result.get_compare_data()}"
    )


@pytest.mark.int8_npu
@pytest.mark.skipif(
    os.environ.get("MQSMLA_INT8_NPU") != "1",
    reason="A2/A3 NPU execution requires MQSMLA_INT8_NPU=1",
)
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_int8_csa_device(dtype, layout_q, layout_kv, return_lse, tmp_path, monkeypatch):
    # Keep the real metadata + Torch runner and the existing low-precision Golden.
    # Compare actual/benchmark errors against independent raw FP64 mathematics.
    # When enabled, import/dispatch errors fail the test; never treat them as skips.
    from batch.mixed_quant_sparse_flash_mla_process import test_mqsmla_quant_process_ci

    monkeypatch.setenv("ATK_TASK_OUTPUT_PATH", str(tmp_path))
    params = case_params(layout_q, layout_kv, dtype)
    params["return_softmax_lse"] = return_lse
    data = golden.gen_data(params)
    high_precision, high_precision_lse = dense_oracle(data, round_kv=False)
    actual, expected, expected_lse, actual_lse = test_mqsmla_quant_process_ci(
        data,
        device_id=int(os.environ.get("DEVICE_ID", "0")),
    )
    assert actual.shape == expected.shape and actual.dtype == dtype
    assert_l0_dual_benchmark(actual, expected, high_precision, "attention")
    if return_lse:
        assert actual_lse.shape == expected_lse.shape
        assert_l0_dual_benchmark(
            actual_lse, expected_lse, high_precision_lse, "softmax LSE"
        )


@pytest.mark.int8_npu
@pytest.mark.skipif(
    os.environ.get("MQSMLA_INT8_NPU") != "1",
    reason="A2/A3 NPU execution requires MQSMLA_INT8_NPU=1",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
@pytest.mark.parametrize(
    "heads,ori_mask,cmp_mask", [(17, 0, 0), (1, 3, 3), (128, 4, 3)]
)
def test_int8_csa_device_multiple_tiles(
    dtype, layout_q, layout_kv, heads, ori_mask, cmp_mask, tmp_path, monkeypatch
):
    """Wrap the three-slot KV buffer and exercise Cube N/M tails and query transitions."""
    from batch.mixed_quant_sparse_flash_mla_process import test_mqsmla_quant_process_ci

    monkeypatch.setenv("ATK_TASK_OUTPUT_PATH", str(tmp_path))
    params = case_params(layout_q, layout_kv, dtype)
    params.update(
        N1=heads,
        S2=1025,
        K=640,
        cmp_ratio=1,
        cmp_residual_kv=None,
        ori_mask_mode=ori_mask,
        cmp_mask_mode=cmp_mask,
        ori_win_left=516 if ori_mask == 4 else -1,
        ori_win_right=0 if ori_mask == 4 else -1,
        cmp_kv_topk_mode="fullK" if cmp_mask == 0 else "no",
        seqused_ori_kv=[513, 1025],
        seqused_cmp_kv=[517, 1027],
        cu_seqlens_ori_kv=[0, 513, 1538],
        cu_seqlens_cmp_kv=[0, 517, 1544],
        block_num1=98,
        block_num2=98,
    )
    data = golden.gen_data(params)
    high_precision, high_precision_lse = dense_oracle(data, round_kv=False)
    actual, expected, expected_lse, actual_lse = test_mqsmla_quant_process_ci(
        data,
        device_id=int(os.environ.get("DEVICE_ID", "0")),
    )
    assert_l0_dual_benchmark(actual, expected, high_precision, "multi-tile attention")
    assert_l0_dual_benchmark(
        actual_lse, expected_lse, high_precision_lse, "multi-tile softmax LSE"
    )


@pytest.mark.int8_npu
@pytest.mark.skipif(
    os.environ.get("MQSMLA_INT8_NPU") != "1",
    reason="A2/A3 NPU execution requires MQSMLA_INT8_NPU=1",
)
@pytest.mark.parametrize("empty_case", ["zero_lengths", "sentinel_indices"])
@pytest.mark.parametrize(
    "sink",
    [2.0, -2.0, -math.inf],
    ids=["positive_sink", "negative_sink", "minus_inf_sink"],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_int8_csa_device_no_valid_kv_returns_zero(
    dtype, layout_q, layout_kv, sink, empty_case
):
    """An empty attention row stays zero even when a sink supplies normalization."""
    from batch.mixed_quant_sparse_flash_mla_process import test_mqsmla_quant_process_ci

    params = case_params(layout_q, layout_kv, dtype)
    data = golden.gen_data(params, generate_golden=False)
    op = data["op_input"]
    op["seqused_ori_kv"] = torch.zeros(params["B"], dtype=torch.int32)
    op["cmp_residual_kv"] = torch.zeros(params["B"], dtype=torch.int32)
    op["sinks"] = torch.full((params["N1"],), sink, dtype=torch.float32)
    if empty_case == "zero_lengths":
        op["seqused_cmp_kv"] = torch.zeros(params["B"], dtype=torch.int32)
    else:
        # CMP storage and used lengths remain nonempty, but no index contributes.
        op["cmp_sparse_indices"].fill_(-1)
    golden.generate_cpu_golden(data)

    actual, _, _, actual_lse = test_mqsmla_quant_process_ci(
        data,
        device_id=int(os.environ.get("DEVICE_ID", "0")),
    )
    assert actual.shape == op["q"].shape and actual.dtype == dtype
    expected_lse_shape = (
        (params["B"], 1, params["S1"], params["N1"])
        if layout_q == "BSND"
        else (1, params["T1"], params["N1"])
    )
    assert actual_lse.shape == expected_lse_shape and actual_lse.dtype == torch.float32
    # These exact expectations do not depend on the CPU attention implementation.
    assert torch.count_nonzero(actual.cpu()).item() == 0
    assert torch.count_nonzero(actual_lse.cpu()).item() == 0


@pytest.mark.parametrize("quant_mode", [2, 3])
@pytest.mark.int8_npu
@pytest.mark.skipif(
    os.environ.get("MQSMLA_INT8_NPU") != "1",
    reason="A2/A3 NPU execution requires MQSMLA_INT8_NPU=1",
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_int8_csa_device_rejects_other_quant_modes(dtype, quant_mode):
    from batch.mixed_quant_sparse_flash_mla_process import test_mqsmla_quant_process_ci

    # Prepare valid mode-1 records, then change only the public API mode.
    params = case_params("BSND", "BSND", dtype)
    data = golden.gen_data(params)
    data["op_input"]["quant_mode"] = quant_mode
    with pytest.raises(RuntimeError, match="INT8 CSA requires quant_mode=1"):
        test_mqsmla_quant_process_ci(
            data, device_id=int(os.environ.get("DEVICE_ID", "0"))
        )


@pytest.mark.parametrize("quant_mode", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout_q,layout_kv", LAYOUTS)
def test_int8_generation_rejects_other_quant_modes(
    dtype, layout_q, layout_kv, quant_mode
):
    params = case_params(layout_q, layout_kv, dtype)
    params["quant_mode"] = quant_mode
    with pytest.raises(ValueError, match="INT8 CSA requires quant_mode=1"):
        golden.gen_data(params)
