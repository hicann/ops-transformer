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

import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

import result_compare_method
import stem_indexer_golden
import custom_ops  # noqa: F401


DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
torch_npu.npu.set_device(int(DEVICE_ID))


class TestCustomStemIndexer(TestCase):
    def test_stem_indexer(self):
        b = 1
        q_heads = 32
        kv_heads = 8
        q_len = 512
        kv_len = 32768
        head_dim = 128
        stem_block_size = 128
        stem_stride = 16
        flatten_dim = stem_stride * head_dim
        max_qb = (q_len + stem_block_size - 1) // stem_block_size
        max_kb = (kv_len + stem_block_size - 1) // stem_block_size
        causal = False
        alpha = 1.0
        initial_blocks = 4
        window_size = 4

        np.random.seed(0)
        qflat = torch.tensor(
            np.random.uniform(-1, 1, (b, q_heads, max_qb, flatten_dim))
        ).to(torch.bfloat16)
        kflat = torch.tensor(
            np.random.uniform(-1, 1, (b, kv_heads, max_kb, flatten_dim))
        ).to(torch.bfloat16)
        vbias = torch.arange(max_kb, dtype=torch.float32).reshape(1, 1, max_kb)
        vbias = vbias.repeat(b, kv_heads, 1) * 1000.0
        q_seq_lens = torch.tensor([q_len], dtype=torch.int32)
        kv_seq_lens = torch.tensor([kv_len], dtype=torch.int32)
        num_prompt_tokens = torch.tensor([kv_len], dtype=torch.int32)

        case = {
            "case_id": "SI_MANUAL_001",
            "testcase_name": "manual_long_s2_non_causal",
            "expected_result": "PASS",
            "batch_size": b,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "q_seq_lens": [q_len],
            "kv_seq_lens": [kv_len],
            "num_prompt_tokens": [kv_len],
            "causal": causal,
            "alpha": alpha,
            "stem_block_size": stem_block_size,
            "stem_stride": stem_stride,
            "initial_blocks": initial_blocks,
            "window_size": window_size,
            "qflat_dtype": "BF16",
            "kflat_dtype": "BF16",
            "vbias_dtype": "FP32",
            "special_setting": "",
        }
        cpu_inputs = {"qflat": qflat, "kflat": kflat, "vbias": vbias}
        cpu_indices, cpu_seq_len = stem_indexer_golden.stem_indexer_golden(
            case, cpu_inputs
        )

        qflat = qflat.npu()
        kflat = kflat.npu()
        vbias = vbias.npu()
        q_seq_lens = q_seq_lens.npu()
        kv_seq_lens = kv_seq_lens.npu()
        num_prompt_tokens = num_prompt_tokens.npu()
        metadata = torch.ops.custom.npu_stem_indexer_metadata(
            q_seq_lens,
            kv_seq_lens,
            q_heads,
            kv_heads,
            causal=causal,
            stem_block_size=stem_block_size,
            dim_qkflat=head_dim,
            window_size=window_size,
        )

        expected_g_size = q_heads // kv_heads
        expected_score_scale = 1.0 / ((stem_block_size // stem_stride) ** 2)
        expected_score_workspace_bytes = 3 * 32 * 1024
        print(
            "[StemIndexer Expected Init] "
            f"bSize={b} qHeadNum={q_heads} kvHeadNum={kv_heads} gSize={expected_g_size} "
            f"maxQb={max_qb} maxKb={max_kb} headDim={head_dim} flattenDim={flatten_dim} causal={int(causal)} "
            f"stemBlockSize={stem_block_size} stemStride={stem_stride} "
            f"initialBlocks={initial_blocks} windowSize={window_size} "
            f"mBaseSize=64 s2BaseSize=256 scoreScale={expected_score_scale:.6f} alpha={alpha:.6f} "
            "kBlockNumRateMedium=0.200000 kBlockNumBiasMedium=30 "
            "kBlockNumRateLarge=0.100000 kBlockNumBiasLarge=30 "
            f"scoreWorkspaceBytes={expected_score_workspace_bytes}",
            flush=True,
        )

        npu_indices, npu_seq_len = torch.ops.custom.npu_stem_indexer(
            qflat,
            kflat,
            vbias,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens=num_prompt_tokens,
            metadata=metadata,
            **stem_indexer_golden.get_call_attrs(case),
        )
        torch_npu.npu.synchronize()

        npu_indices = npu_indices.cpu().to(torch.int32)
        npu_seq_len = npu_seq_len.cpu().to(torch.int32)

        print("compare result")
        torch.set_printoptions(profile="full")
        print("npu_seq_len", npu_seq_len)
        print("cpu_seq_len", cpu_seq_len)
        print("npu_indices", npu_indices[0, 0, 0, : int(cpu_seq_len[0, 0, 0])])
        print("cpu_indices", cpu_indices[0, 0, 0, : int(cpu_seq_len[0, 0, 0])])

        self.assertEqual(tuple(npu_seq_len.shape), tuple(cpu_seq_len.shape))
        self.assertEqual(tuple(npu_indices.shape), tuple(cpu_indices.shape))
        self.assertTrue(torch.equal(npu_seq_len, cpu_seq_len))
        result_compare_method.assert_stem_indexer_result(
            cpu_indices, cpu_seq_len, (npu_indices, npu_seq_len), case, cpu_inputs
        )


if __name__ == "__main__":
    run_tests()
