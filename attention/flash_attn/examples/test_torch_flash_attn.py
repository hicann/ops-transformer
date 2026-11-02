# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run metadata and FlashAttn on Ascend950 with a known uniform-attention result.

Install the FlashAttn custom package and its matching cann_ops_transformer wheel,
source the CANN and custom-package environments, then run this file with Python.
The Q length of 256 exercises DN unmerged routing when its switch is enabled.
"""

import math

import torch
import torch_npu  # noqa: F401
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata


def main():
    torch.npu.set_device(0)
    batch, query_length, kv_length = 1, 256, 256
    query_heads, kv_heads, head_dim = 8, 2, 64
    query = torch.zeros(
        batch, query_length, query_heads, head_dim, dtype=torch.float16, device="npu:0"
    )
    key = torch.zeros(
        batch, kv_length, kv_heads, head_dim, dtype=torch.float16, device="npu:0"
    )
    value = torch.ones_like(key)
    metadata = flash_attn_metadata(
        num_heads_q=query_heads,
        num_heads_kv=kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim,
        batch_size=batch,
        max_seqlen_q=query_length,
        max_seqlen_kv=kv_length,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )
    output, lse = flash_attn(
        query,
        key,
        value,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(head_dim),
        mask_mode=0,
        max_seqlen_q=query_length,
        max_seqlen_kv=kv_length,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=True,
    )
    torch.npu.synchronize()
    # Zero logits give uniform attention: output is one and LSE is log(KV length).
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=1e-3)
    torch.testing.assert_close(
        lse, torch.full_like(lse, math.log(kv_length)), rtol=0, atol=1e-4
    )
    print(
        f"FlashAttn example passed: output={tuple(output.shape)}, lse={tuple(lse.shape)}"
    )


if __name__ == "__main__":
    main()
