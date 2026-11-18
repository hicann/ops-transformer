# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small NPU smoke test for all three registered GDN entry points.

Run after building the Torch package from this checkout:
    python attention/flash_gdn/examples/smoke_flash_gdn.py --device npu:0
"""

import argparse

import torch
import torch_npu  # noqa: F401

from cann_ops_transformer import flash_gdn, flash_gdn_fwd, flash_gdn_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    # One active key/query coordinate gives an analytic two-token recurrence.
    # State: s1 = .5 * 2 = 1; s2 = .5 * s1 + .5 * (4 - .5 * s1) = 2.25.
    q = torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16, device=args.device)
    q[..., 0] = 1
    k = q.clone()
    v = torch.zeros_like(q)
    v[0, 0, :, 0] = torch.tensor([2.0, 4.0], device=args.device)
    g = torch.full((1, 1, 2), -0.6931471805599453, device=args.device)
    beta = torch.full_like(g, 0.5)
    md = flash_gdn_metadata(
        batch_size=1, nheads_qk=1, nheads_v=1, seqlens=2, layout="BNSD"
    )
    assert md.dtype == torch.int32 and md.shape == (13,) and md.device.type == "npu"
    with torch.inference_mode():
        route = flash_gdn(
            q, k, v, g, beta, md, layout="BNSD", scale=1.0, output_final_state=True
        )
        fwd = flash_gdn_fwd(
            q, k, v, g, beta, md, layout="BNSD", scale=1.0, output_final_state=True
        )
    torch.npu.synchronize()
    assert len(route) == 3 and len(fwd) == 16
    assert route[2] is None and all(x is None for x in fwd[2:])
    expected = torch.zeros_like(v, device="cpu").float()
    expected[0, 0, :, 0] = torch.tensor([1.0, 2.25])
    expected_state = torch.zeros(1, 1, 128, 128)
    expected_state[0, 0, 0, 0] = 2.25
    for result in (route, fwd):
        torch.testing.assert_close(
            result[0].cpu().float(), expected, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            result[1].cpu(), expected_state, atol=1e-5, rtol=1e-5
        )
    print("PASS: flash_gdn_metadata, flash_gdn, flash_gdn_fwd")
    print("out[0,0,:,0] =", route[0][0, 0, :, 0].cpu().float().tolist())
    print("final_state[0,0,0,0] =", route[1][0, 0, 0, 0].cpu().item())


if __name__ == "__main__":
    main()
