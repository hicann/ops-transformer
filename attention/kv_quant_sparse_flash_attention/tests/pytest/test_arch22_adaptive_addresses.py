# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""910B INT8/no-RoPE address regression; requires the built custom OPP and msprof.

Run with QSFA_REGRESSION_DEVICE=<idle device> pytest -q <this file>.
The profiler tiling key must select the adaptive specialization, except for
the explicit legacy control. A scalar page-table oracle computes every
referenced byte address and the uniform-attention output independently.
Only selected KV rows contain nonzero values, making wrong row reads visible.
These cases use aligned or >=512 sparse sizes, without a local tiling patch.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


# name, table width, K, valid prefix, optional restricted logical page window
CASES = (
    ("legacy_256_2560", 256, 2560, 2560, None),
    ("adaptive_257_2560", 257, 2560, 2560, None),
    ("adaptive_256_2561", 256, 2561, 2561, None),
    ("adaptive_257_2561", 257, 2561, 2561, None),
    ("two_blocks_odd_tail", 257, 1025, 1025, None),
    ("invalid_suffix", 257, 2561, 2049, None),
    ("window_cross_1024", 1062, 2560, 2560, (1018, 1040)),
    ("segmented_table", 2050, 2560, 2560, None),
    ("discrete_page_gather", 8193, 1025, 769, None),
)
ADAPTIVE_KEY_MASK = 1 << 15


def _inputs_and_reference(case, seed, dtype):
    _, width, k, valid, window = case
    generator = torch.Generator().manual_seed(seed)
    table = torch.randperm(width + 3, generator=generator)[:width].to(torch.int32)
    first, last = window if window is not None else (0, width - 1)
    lo, hi = first * 128, (last + 1) * 128 - 1
    middle = torch.randperm(hi - lo - 1, generator=generator)[: k - 2] + lo + 1
    indices = torch.cat((torch.tensor([lo, hi]), middle)).to(torch.int32)
    indices[valid:] = -1
    cache = torch.zeros((width + 3, 128, 1, 528), dtype=torch.int8)
    cache[..., 512:].view(torch.float32).fill_(0.5)
    features = torch.randint(512, (valid,), generator=generator)
    pages = table[indices[:valid].long() // 128].long()
    offsets = indices[:valid].long() % 128
    cache[pages, offsets, 0, features] = 64

    addresses = []
    expected = torch.zeros(512, dtype=torch.float64)
    for logical_index in indices[:valid].tolist():
        logical_page, offset = divmod(logical_index, 128)
        physical_page = int(table[logical_page])
        addresses.append((physical_page * 128 + offset) * 528)
        expected += cache[physical_page, offset, 0, :512].to(torch.float64) * 0.5
    assert len(set(addresses)) == valid
    expected = (expected / valid).to(dtype).reshape(1, 1, 512).repeat(1, 4, 1)
    query = torch.zeros((1, 4, 512), dtype=dtype)
    return (
        query,
        cache,
        indices.reshape(1, 1, k),
        table.reshape(1, width),
        expected,
        addresses,
    )


def _worker(output, device):
    torch.set_num_threads(4)
    torch.npu.set_device(device)
    results = []
    for i, case in enumerate(CASES):
        # Distinct shape/dtype pairs keep Host tiling logs observable for each case.
        dtype = torch.bfloat16 if i % 2 == 0 or i == 5 else torch.float16
        query, cache, indices, table, expected, addresses = _inputs_and_reference(
            case, 15349 + i, dtype
        )
        q, kv, sparse, pages = (x.npu() for x in (query, cache, indices, table))
        print("QSFA_REGRESSION_CASE=" + case[0], flush=True)
        result = torch_npu.npu_kv_quant_sparse_flash_attention(
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
            actual_seq_lengths_kv=torch.tensor(
                [case[1] * 128], dtype=torch.int32
            ).npu(),
        )
        torch.npu.synchronize()
        actual = result.cpu()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-4, msg=case[0])
        results.append(
            {
                "case": case[0],
                "expected_adaptive": i != 0,
                "valid_indices": case[3],
                "reference_address_sha256": hashlib.sha256(
                    json.dumps(addresses).encode()
                ).hexdigest(),
                "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
            }
        )
        del result, actual, q, kv, sparse, pages, query, cache, indices, table
        torch.npu.empty_cache()
    output.write_text(json.dumps(results, indent=2))


@pytest.mark.ci
def test_arch22_adaptive_addresses(tmp_path):
    if torch_npu is None or not torch.npu.is_available():
        pytest.skip("Ascend NPU is required")
    device = int(os.environ.get("QSFA_REGRESSION_DEVICE", "0"))
    torch.npu.set_device(device)
    if "910B" not in torch.npu.get_device_name(device):
        pytest.skip("This regression targets arch22/910B")
    msprof = shutil.which("msprof")
    if msprof is None:
        pytest.skip("msprof is required to assert the executed tiling specialization")
    result_path = tmp_path / "address_results.json"
    command = [
        msprof,
        "--output=" + str(tmp_path / "profile"),
        "--task-time=on",
        "--ai-core=off",
        "--python-path=" + sys.executable,
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(result_path),
        "--device",
        str(device),
    ]
    profile_env = os.environ.copy()
    profile_env.update(ASCEND_GLOBAL_LOG_LEVEL="1", ASCEND_SLOG_PRINT_TO_STDOUT="1")
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1200,
        env=profile_env,
    )
    (tmp_path / "msprof.log").write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout[-12000:]
    records = json.loads(result_path.read_text())
    summaries = list((tmp_path / "profile").rglob("op_summary*.csv"))
    assert len(summaries) == 1, summaries
    with summaries[0].open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        key_column = next(
            (
                k
                for k in reader.fieldnames
                if k.lower().replace(" ", "").replace("_", "") == "tilingkey"
            ),
            None,
        )
        rows = [
            row for row in reader if row["OP Type"] == "KvQuantSparseFlashAttention"
        ]
    rows.sort(key=lambda row: float(row["Task Start Time(us)"]))
    assert len(rows) == len(records) == len(CASES)
    logged_keys = {}
    current_case = None
    for line in completed.stdout.splitlines():
        marker = re.search(r"QSFA_REGRESSION_CASE=(\w+)", line)
        if marker:
            current_case = marker[1]
        key_match = re.search(r"QSFA tilingKey_: (\d+)", line)
        if key_match and current_case is not None:
            logged_keys.setdefault(current_case, set()).add(int(key_match[1]))
    for record, row in zip(records, rows):
        if key_column is not None:
            key = int(row[key_column], 0)
        else:
            keys = logged_keys.get(record["case"], set())
            assert len(keys) == 1, (
                "Missing or ambiguous runtime specialization",
                record,
                keys,
            )
            key = next(iter(keys))
        assert bool(key & ADAPTIVE_KEY_MASK) == record["expected_adaptive"], (
            record,
            key,
        )
        record["executed_tiling_key"] = key
    result_path.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    _worker(args.output, args.device)
