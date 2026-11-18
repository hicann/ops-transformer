# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run after installing the Torch extension with the flash_gdn operator."""

import torch
import torch_npu  # noqa: F401

from cann_ops_transformer import flash_gdn, flash_gdn_metadata

B, N, S, D = 1, 1, 8, 128
q = torch.nn.functional.normalize(torch.randn(B, N, S, D, device="npu"), dim=-1).to(
    torch.bfloat16
)
k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
v = torch.randn_like(q)
g = -torch.rand(B, N, S, device="npu") * 0.1
beta = torch.rand_like(g)
metadata = flash_gdn_metadata(
    batch_size=B, nheads_qk=N, nheads_v=N, seqlens=S, layout="BNSD"
)
with torch.inference_mode():
    out, final_state, chunk_states = flash_gdn(
        q, k, v, g, beta, metadata, layout="BNSD", output_final_state=True
    )
assert out.shape == v.shape and final_state.shape == (B, N, D, D)
assert chunk_states is None
