#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent arch22 (A2/A3) ordinary-SFA graph replay benchmark (full, unpadded rows)."""

import argparse
import json
import math
import statistics

import torch
import torch_npu  # noqa: F401
from test_sfa_split_kv_arch22 import sfa_operator


def measure(op, heads, batch, query_per_batch, dtype, rope, calls, samples):
    total = batch * query_per_batch
    generator = torch.Generator().manual_seed(20261008 + total + heads)
    kv_length, page, topk = 4096, 128, 2048
    q = torch.randn(total, heads, 512, generator=generator).to(dtype).npu()
    k = torch.randn(batch * kv_length, 512, generator=generator).to(dtype).npu()
    qr = (
        torch.randn(total, heads, 64, generator=generator).to(dtype).npu()
        if rope
        else None
    )
    kr = (
        torch.randn(batch * kv_length, 64, generator=generator).to(dtype).npu()
        if rope
        else None
    )
    indices = (
        torch.stack(
            [
                torch.randperm(kv_length, generator=generator)[:topk]
                for _ in range(total)
            ]
        )
        .int()
        .unsqueeze(1)
        .npu()
    )
    pages = k.reshape(-1, page, 1, 512)
    rope_pages = kr.reshape(-1, page, 1, 64) if rope else None
    block_table = (
        torch.arange(batch * kv_length // page, dtype=torch.int32)
        .reshape(batch, -1)
        .npu()
    )
    lengths_q = (
        torch.arange(1, batch + 1, dtype=torch.int32).mul_(query_per_batch).npu()
    )
    lengths_kv = torch.full((batch,), kv_length, dtype=torch.int32).npu()

    def run():
        return op(
            q,
            pages,
            pages,
            indices,
            1 / math.sqrt(576 if rope else 512),
            block_table=block_table,
            actual_seq_lengths_query=lengths_q,
            actual_seq_lengths_kv=lengths_kv,
            query_rope=qr,
            key_rope=rope_pages,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )

    for _ in range(6):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            output = run()
    graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(output[0]).all()
    references = []
    for row in range(total):
        positions = (row // query_per_batch) * kv_length + indices[row, 0].long()
        q_ref, k_ref = q[row].float(), k[positions].float()
        if rope:
            q_ref = torch.cat((q_ref, qr[row].float()), dim=-1)
            k_ref = torch.cat((k_ref, kr[positions].float()), dim=-1)
        scores = (q_ref @ k_ref.T) / math.sqrt(576 if rope else 512)
        references.append(scores.softmax(dim=-1) @ k[positions].float())
    reference = torch.stack(references)
    errors = (output[0].float() - reference).flatten(1).norm(
        dim=-1
    ) / reference.flatten(1).norm(dim=-1)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.005
    assert errors.max().item() < tolerance
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        begin, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000 / calls)
    return dict(
        g=heads,
        b=batch,
        q=query_per_batch,
        total_q=total,
        dtype=str(dtype),
        rope=rope,
        lse=False,
        median_us=statistics.median(values),
        samples_us=values,
        worst_row_rel_l2=errors.max().item(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--samples", type=int, default=101)
    args = parser.parse_args()
    op = sfa_operator.__wrapped__()
    print(
        json.dumps(
            dict(
                variant=args.variant,
                device=torch.npu.get_device_name(0),
                kv_length=4096,
                topk=2048,
                calls=args.calls,
                samples=args.samples,
            )
        ),
        flush=True,
    )
    cases = [
        (g, b, q, torch.bfloat16, True)
        for g in (1, 8, 32, 64, 128)
        for b, q in ((1, 1), (1, 4), (2, 2), (4, 4), (8, 8))
    ]
    cases += [
        (g, 1, 1, dtype, rope)
        for g in (8, 128)
        for dtype in (torch.bfloat16, torch.float16)
        for rope in (False, True)
    ]
    for case in dict.fromkeys(cases):
        record = measure(op, *case, args.calls, args.samples)
        record["variant"] = args.variant
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
