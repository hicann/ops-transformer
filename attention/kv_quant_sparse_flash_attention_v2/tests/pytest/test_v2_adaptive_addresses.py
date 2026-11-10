# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""V2 address oracle, LSE checks and runtime adaptive-key coverage."""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

ORACLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "kv_quant_sparse_flash_attention/tests/pytest/test_arch22_adaptive_addresses.py"
)
spec = importlib.util.spec_from_file_location("qsfa_address_oracle", ORACLE_PATH)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
CASES = oracle.CASES + (
    ("trim_1024", 257, 4096, 1024, None),
    ("trim_short_64", 257, 4096, 64, None),
    ("trim_all_empty", 257, 4096, 0, None),
    ("segment_cache", 257, 4096, 4096, None),
    ("capacity_3071", 257, 4096, 3071, None),
    ("capacity_3072", 257, 4096, 3072, None),
    ("capacity_3073", 257, 4096, 3073, None),
)


def worker(index, lse, sink, output):
    from cann_ops_transformer.ops.attention.kv_quant_sparse_flash_attention_v2 import (
        kv_quant_sparse_flash_attention,
    )

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    case = CASES[index]
    dtype = torch.bfloat16 if index % 2 == 0 else torch.float16
    query, cache, indices, table, _, addresses = oracle._inputs_and_reference(
        case, 15349 + index, dtype
    )
    valid = case[3]
    expected = torch.zeros(512, dtype=torch.float64)
    for logical in indices.reshape(-1)[:valid].tolist():
        page, offset = divmod(logical, 128)
        expected += cache[int(table[0, page]), offset, 0, :512].double() * 0.5
    # The public Torch V2 wrapper supplies a zero sink when no sink is given.
    sink_value = 1.25 if sink else 0.0
    expected = (
        (expected / (valid + math.exp(sink_value)))
        .to(dtype)
        .reshape(1, 1, 512)
        .repeat(1, 4, 1)
    )
    q, kv, sparse, pages = (value.npu() for value in (query, cache, indices, table))
    explicit_sink = (
        torch.full((4,), sink_value, dtype=torch.float32).npu() if sink else None
    )
    result = kv_quant_sparse_flash_attention(
        q,
        kv,
        kv,
        sparse,
        scale_value=512**-0.5,
        key_quant_mode=2,
        value_quant_mode=2,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=0,
        block_table=pages,
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([case[1] * 128], dtype=torch.int32).npu(),
        sinks=explicit_sink,
        return_softmax_lse=lse,
    )
    torch.npu.synchronize()
    actual = tuple(value.cpu() for value in result)
    torch.testing.assert_close(actual[0], expected, rtol=0.02, atol=2.0e-4)
    if lse:
        maximum = sink_value
        total = valid * math.exp(-maximum) + 1
        torch.testing.assert_close(
            actual[1], torch.full_like(actual[1], maximum), rtol=0.0, atol=1.0e-6
        )
        torch.testing.assert_close(
            actual[2], torch.full_like(actual[2], total), rtol=1.0e-4, atol=1.0e-5
        )
    else:
        assert actual[1].numel() == actual[2].numel() == 0
    output.write_text(
        json.dumps(
            {
                "case": case[0],
                "valid": valid,
                "lse": lse,
                "explicit_sink": sink,
                "address_count": len(addresses),
                "max_abs_error": (actual[0].float() - expected.float())
                .abs()
                .max()
                .item(),
            }
        )
    )


@pytest.mark.ci
@pytest.mark.parametrize("index", range(len(CASES)), ids=[case[0] for case in CASES])
@pytest.mark.parametrize("lse", [False, True])
@pytest.mark.parametrize("sink", [False, True])
def test_v2_adaptive_addresses(index, lse, sink, tmp_path):
    import torch_npu  # noqa: F401

    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name(0):
        pytest.skip("Ascend 910B is required")
    profiler = shutil.which("msprof")
    if profiler is None:
        pytest.skip("msprof is required to verify the runtime tiling key")
    worker_command = [sys.executable]
    bootstrap = os.environ.get("QSFA_TEST_BOOTSTRAP")
    if bootstrap:
        worker_command += [
            bootstrap,
            str(Path(__file__).resolve().parents[4]),
            "--script",
        ]
    result = tmp_path / "result.json"
    worker_command += [
        str(Path(__file__).resolve()),
        "--worker",
        str(index),
        "--lse",
        str(int(lse)),
        "--sink",
        str(int(sink)),
        "--output",
        str(result),
    ]
    command = [
        profiler,
        "--output=" + str(tmp_path / "profile"),
        "--task-time=on",
        "--ai-core=off",
        "--python-path=" + sys.executable,
    ] + worker_command
    environment = os.environ.copy()
    environment.update(ASCEND_GLOBAL_LOG_LEVEL="1", ASCEND_SLOG_PRINT_TO_STDOUT="1")
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1200,
    )
    (tmp_path / "msprof.log").write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout[-12000:]
    record = json.loads(result.read_text())
    summaries = list((tmp_path / "profile").rglob("op_summary*.csv"))
    assert len(summaries) == 1, summaries
    with summaries[0].open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        key_column = next(
            (
                key
                for key in reader.fieldnames
                if key.lower().replace(" ", "").replace("_", "") == "tilingkey"
            ),
            None,
        )
        rows = [
            row for row in reader if row["OP Type"] == "KvQuantSparseFlashAttentionV2"
        ]
    assert len(rows) == 1, rows
    if key_column:
        key = int(rows[0][key_column], 0)
    else:
        keys = set(map(int, re.findall(r"QSFA tilingKey_: (\d+)", completed.stdout)))
        assert len(keys) == 1, keys
        key = next(iter(keys))
    assert bool(key & oracle.ADAPTIVE_KEY_MASK) == (index != 0), (record, key)
    record["executed_tiling_key"] = key
    result.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--lse", type=int, required=True)
    parser.add_argument("--sink", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(args.worker, bool(args.lse), bool(args.sink), args.output)
