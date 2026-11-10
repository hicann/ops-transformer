# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import importlib
import os
import unittest

import torch

PACKAGE = os.environ.get("TGR_TORCH_PACKAGE", "cann_ops_transformer")
module = importlib.import_module(f"{PACKAGE}.ops.mhc.text_gated_residual")
forward = module.text_gated_residual
backward = importlib.import_module(
    f"{PACKAGE}.ops.mhc.text_gated_residual.text_gated_residual"
)._text_gated_residual_backward


def reference(x, wd, wu, wg, gamma, channels):
    v = x.float().reshape(*x.shape[:-1], channels, -1)
    norm = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)).flatten(-2)
    if gamma is not None:
        norm = norm * (1 + gamma)
    down = norm @ wd.float().T
    up = torch.nn.functional.silu(down / channels) @ wu.float().T
    mixed = (norm * torch.sigmoid(up)).reshape(*v.shape).mean(-2)
    inject = (
        2 * torch.sigmoid((norm @ wg.float().T) / channels) if wg is not None else None
    )
    return mixed, inject


class TestTextGatedResidual(unittest.TestCase):
    def test_public_exports(self):
        self.assertEqual(module.__all__, ["text_gated_residual"])
        self.assertFalse(hasattr(module, "text_gated_residual_grad"))
        self.assertFalse(
            hasattr(torch.ops.cann_ops_transformer, "text_gated_residual_grad")
        )

    def test_meta_contract(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.empty((1, 2, 1024), dtype=dtype, device="meta")
            wd = x.new_empty((64, 1024))
            wu = x.new_empty((1024, 64))
            wg = x.new_empty((4, 1024))
            gamma = x.new_empty((1024,), dtype=torch.float32)
            for gate in (None, wg):
                for scale in (None, gamma):
                    outputs = torch.ops.cann_ops_transformer.text_gated_residual(
                        x,
                        wd,
                        wu,
                        w_gate=gate,
                        gamma=scale,
                        channels=4,
                        need_backward=True,
                    )
                    self.assertEqual(outputs[0].shape, (1, 2, 256))
                    self.assertEqual(outputs[2].dtype, torch.float32)
                    result = backward(
                        outputs[0],
                        outputs[1] if gate is not None else None,
                        x,
                        wd,
                        wu,
                        gate,
                        scale,
                        outputs[2],
                        outputs[1] if gate is not None else None,
                        outputs[3],
                        outputs[4],
                        4,
                    )
                    for grad, value in zip(result, (x, wd, wu, gate, scale)):
                        if value is None:
                            self.assertIsNone(grad)
                        else:
                            self.assertEqual(grad.shape, value.shape)
                            self.assertEqual(grad.dtype, value.dtype)
        for dim, channels in ((10, 4), (1024, 0)):
            with self.assertRaises(RuntimeError):
                torch.ops.cann_ops_transformer.text_gated_residual(
                    torch.empty((1, 2, dim), dtype=torch.float16, device="meta"),
                    wd,
                    wu,
                    channels=channels,
                )

    @unittest.skipIf(os.environ.get("TGR_META_ONLY") == "1", "Meta-only run")
    def test_autograd_npu(self):
        torch.npu.set_device(int(os.environ.get("TGR_NPU_DEVICE", "0")))
        torch.manual_seed(42)
        for dtype in (torch.float16, torch.bfloat16):
            for gate in (False, True):
                for gamma in (False, True):
                    for selected in ("all", "x", "w_down", "w_up", "w_gate", "gamma"):
                        if (
                            selected == "w_gate"
                            and not gate
                            or selected == "gamma"
                            and not gamma
                        ):
                            continue
                        for loss_kind in (
                            ("both", "mixed", "inject") if gate else ("mixed",)
                        ):
                            with self.subTest(
                                dtype=dtype,
                                gate=gate,
                                gamma=gamma,
                                selected=selected,
                                loss=loss_kind,
                            ):
                                names = ("x", "w_down", "w_up", "w_gate", "gamma")
                                values = [
                                    torch.randn(
                                        1,
                                        int(os.environ.get("TGR_TEST_M", "32")),
                                        1024,
                                        dtype=dtype,
                                    ),
                                    torch.randn(64, 1024, dtype=dtype) * 0.01,
                                    torch.randn(1024, 64, dtype=dtype) * 0.01,
                                    torch.randn(4, 1024, dtype=dtype) * 0.01
                                    if gate
                                    else None,
                                    torch.randn(1024) * 0.01 if gamma else None,
                                ]
                                cpu = [
                                    v.float()
                                    .detach()
                                    .requires_grad_(selected in ("all", n))
                                    if v is not None
                                    else None
                                    for n, v in zip(names, values)
                                ]
                                npu = [
                                    v.npu()
                                    .detach()
                                    .requires_grad_(selected in ("all", n))
                                    if v is not None
                                    else None
                                    for n, v in zip(names, values)
                                ]
                                expected = reference(*cpu, 4)
                                actual = forward(
                                    npu[0], npu[1], npu[2], npu[3], 4, npu[4]
                                )
                                actual = actual if gate else (actual, None)
                                grads = [
                                    torch.randn_like(y).to(dtype).float()
                                    if y is not None
                                    else None
                                    for y in expected
                                ]
                                indexes = (
                                    (0, 1)
                                    if loss_kind == "both"
                                    else ((0,) if loss_kind == "mixed" else (1,))
                                )
                                cpu_loss = sum(
                                    (expected[i] * grads[i]).sum() for i in indexes
                                )
                                if cpu_loss.requires_grad:
                                    cpu_loss.backward()
                                sum(
                                    (actual[i] * grads[i].to(dtype).npu()).sum()
                                    for i in indexes
                                ).backward()
                                for i in range(2 if gate else 1):
                                    torch.testing.assert_close(
                                        actual[i].float().cpu(),
                                        expected[i],
                                        rtol=0.02,
                                        atol=0.01,
                                    )
                                for n, a, b in zip(names, npu, cpu):
                                    if b is None or not b.requires_grad:
                                        if a is not None:
                                            self.assertIsNone(a.grad)
                                        continue
                                    if b.grad is None:
                                        self.assertEqual(
                                            torch.count_nonzero(a.grad).item(), 0
                                        )
                                        continue
                                    self.assertEqual(a.grad.dtype, a.dtype)
                                    error = (
                                        (a.grad.float().cpu() - b.grad)
                                        .square()
                                        .mean()
                                        .sqrt()
                                    )
                                    scale = (
                                        b.grad.square().mean().sqrt().clamp_min(1e-6)
                                    )
                                    self.assertLess(
                                        (error / scale).item(),
                                        0.08 if dtype == torch.bfloat16 else 0.02,
                                        n,
                                    )
                                with torch.no_grad():
                                    y = forward(
                                        npu[0], npu[1], npu[2], npu[3], 4, npu[4]
                                    )
                                    self.assertFalse(
                                        (y[0] if gate else y).requires_grad
                                    )


if __name__ == "__main__":
    unittest.main()
