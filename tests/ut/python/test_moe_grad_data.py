#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run with: python -m unittest discover -s tests/ut/python -p test_moe_grad_data.py."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OPS = ("moe_token_permute_with_ep_grad", "moe_token_unpermute_grad")


def script_path(op):
    return (
        ROOT
        / "moe"
        / op
        / "tests"
        / "ut"
        / "op_kernel"
        / (op + "_data")
        / "gen_data.py"
    )


def read_tensor(directory, name, dtype, shape):
    numpy_dtype = {
        torch.bfloat16: np.int16,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int32: np.int32,
    }[dtype]
    result = torch.from_numpy(np.fromfile(directory / name, dtype=numpy_dtype))
    if dtype == torch.bfloat16:
        result = result.view(torch.bfloat16)
    return result.reshape(shape)


class MoeGradDataTest(unittest.TestCase):
    def generate(self, op, directory, dtype, flag, topk, extra=()):
        args = [sys.executable, str(script_path(op)), "5", str(topk), "7", dtype]
        if op == OPS[1]:
            args.append("float32")
        args += [str(flag), "--output-dir", str(directory), *extra]
        result = subprocess.run(args, cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_permute_matches_forward_autograd(self):
        for dtype_name, dtype in (
            ("bfloat16_t", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
        ):
            for flag in (True, False):
                with (
                    self.subTest(dtype=dtype_name, probs=flag),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    directory = Path(temporary)
                    self.generate(
                        OPS[0],
                        directory,
                        dtype_name,
                        flag,
                        3,
                        ("--start", "2", "--end", "13"),
                    )
                    indices = read_tensor(
                        directory, "input_sorted_indices.bin", torch.int32, (15,)
                    ).long()
                    self.assertEqual(
                        torch.sort(indices).values.tolist(), list(range(15))
                    )
                    self.assertNotEqual(indices.tolist(), list(range(15)))
                    permutation = torch.argsort(indices)[2:13]
                    original = torch.zeros(5, 7, requires_grad=True)
                    forward = original.index_select(0, permutation // 3)
                    upstream = read_tensor(
                        directory, "input_permuted_tokens.bin", dtype, (11, 7)
                    )
                    reference = torch.autograd.grad(
                        forward, original, upstream.float()
                    )[0].to(dtype)
                    golden = read_tensor(
                        directory, "output_unpermuted_tokens.bin", dtype, (5, 7)
                    )
                    torch.testing.assert_close(golden, reference, rtol=0, atol=0)
                    if flag:
                        original_probs = torch.zeros(15, requires_grad=True)
                        forward_probs = original_probs.index_select(0, permutation)
                        upstream_probs = read_tensor(
                            directory, "input_probs.bin", dtype, (11,)
                        )
                        reference_probs = torch.autograd.grad(
                            forward_probs, original_probs, upstream_probs.float()
                        )[0]
                        golden_probs = read_tensor(
                            directory, "output_probs_grad.bin", dtype, (15,)
                        )
                        torch.testing.assert_close(
                            golden_probs, reference_probs.to(dtype), rtol=0, atol=0
                        )
                    else:
                        self.assertFalse((directory / "input_probs.bin").exists())
                        self.assertFalse((directory / "output_probs_grad.bin").exists())

    def test_unpermute_matches_forward_autograd(self):
        for dtype_name, dtype in (
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
        ):
            for flag in (True, False):
                with (
                    self.subTest(dtype=dtype_name, probs=flag),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    directory = Path(temporary)
                    topk = 3 if flag else 1
                    self.generate(OPS[1], directory, dtype_name, flag, topk)
                    indices = read_tensor(
                        directory, "input_sorted_indices.bin", torch.int32, (5 * topk,)
                    ).long()
                    self.assertEqual(
                        torch.sort(indices).values.tolist(), list(range(5 * topk))
                    )
                    tokens = read_tensor(
                        directory, "input_permuted_tokens.bin", dtype, (5 * topk, 7)
                    )
                    tokens = tokens.float().requires_grad_()
                    gathered = tokens.index_select(0, indices).reshape(5, topk, 7)
                    inputs = [tokens]
                    if flag:
                        probs = read_tensor(
                            directory, "input_probs.bin", torch.float32, (5, topk)
                        ).requires_grad_()
                        gathered = gathered * probs.unsqueeze(-1)
                        inputs.append(probs)
                    upstream = read_tensor(
                        directory, "input_unpermuted_output_d.bin", dtype, (5, 7)
                    ).float()
                    reference = torch.autograd.grad(gathered.sum(1), inputs, upstream)
                    golden = read_tensor(
                        directory,
                        "output_permuted_tokens_grad.bin",
                        dtype,
                        (5 * topk, 7),
                    )
                    torch.testing.assert_close(
                        golden, reference[0].to(dtype), rtol=0, atol=0
                    )
                    if flag:
                        golden_probs = read_tensor(
                            directory, "output_probs_grad.bin", torch.float32, (5, topk)
                        )
                        torch.testing.assert_close(
                            golden_probs, reference[1], rtol=0, atol=0
                        )
                    else:
                        self.assertFalse((directory / "input_probs.bin").exists())
                        self.assertFalse((directory / "output_probs_grad.bin").exists())

    def test_false_removes_only_optional_outputs(self):
        for op in OPS:
            with self.subTest(op=op), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sentinel = directory / "unrelated.bin"
                sentinel.write_bytes(b"keep")
                self.generate(op, directory, "bfloat16", True, 1)
                self.assertTrue((directory / "output_probs_grad.bin").exists())
                self.generate(op, directory, "bfloat16", False, 1)
                self.assertFalse((directory / "input_probs.bin").exists())
                self.assertFalse((directory / "output_probs_grad.bin").exists())
                self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_invalid_flags_and_missing_probs_topk(self):
        for op in OPS:
            spec = importlib.util.spec_from_file_location(op, script_path(op))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.subTest(op=op):
                self.assertTrue(module.parse_bool("True"))
                self.assertFalse(module.parse_bool("False"))
                with self.assertRaises(argparse.ArgumentTypeError):
                    module.parse_bool("invalid")
        with tempfile.TemporaryDirectory() as temporary:
            args = [
                sys.executable,
                str(script_path(OPS[1])),
                "5",
                "3",
                "7",
                "bfloat16",
                "float32",
                "False",
            ]
            result = subprocess.run(args, cwd=temporary, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("topk must be 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
