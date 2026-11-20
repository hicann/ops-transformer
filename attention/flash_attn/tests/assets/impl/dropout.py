# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU Philox4x32-7 masks in the operator's Query row order."""

import numpy as np
import torch


def philox_bytes(counter, seed, carry=None):
    """Return 16 random bytes per counter, including carry into counter word 2."""
    counter = np.asarray(counter, dtype=np.uint64)
    c0, c1 = counter & 0xFFFFFFFF, counter >> 32
    c2 = np.zeros_like(counter) if carry is None else np.asarray(carry, dtype=np.uint64)
    c3 = np.zeros_like(counter)
    key0, key1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    for _ in range(7):
        product0 = c0 * np.uint64(0xD2511F53)
        product1 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (product1 >> 32) ^ c1 ^ np.uint64(key0),
            product1 & 0xFFFFFFFF,
            (product0 >> 32) ^ c3 ^ np.uint64(key1),
            product0 & 0xFFFFFFFF,
        )
        key0 = (key0 + 0x9E3779B9) & 0xFFFFFFFF
        key1 = (key1 + 0xBB67AE85) & 0xFFFFFFFF
    words = np.stack((c0, c1, c2, c3), axis=-1)
    return np.stack(
        [(words >> shift) & 255 for shift in (0, 8, 16, 24)], axis=-1
    ).reshape(*counter.shape, 16)


class DropoutMask:
    def __init__(
        self, dropout_p, seed, offset, heads, q_lens, kv_lens, layout_q, kv_heads
    ):
        if not 0 <= dropout_p < 1:
            raise ValueError("dropout_p must be finite and in [0, 1)")
        self.dropout_p = np.float32(dropout_p)
        if self.dropout_p >= 1:
            raise ValueError("dropout_p rounds to 1.0")
        self.keep_prob = float(np.float32(1) - self.dropout_p)
        for name, tensor in (("seed", seed), ("offset", offset)):
            if tensor is not None and (
                tensor.dtype != torch.uint64 or tensor.shape != (1,)
            ):
                raise ValueError(f"{name} must be a uint64 Tensor of shape (1,)")
        if self.dropout_p == 0 and (seed is not None or offset is not None):
            raise ValueError("seed and offset must be absent when dropout_p is 0")
        self.seed = int(seed.item()) if seed is not None else 0
        self.base_counter = (int(offset.item()) + 15) // 16 if offset is not None else 0
        self.threshold = int(np.float32(self.keep_prob) * np.float32(255))
        self.q_lens, self.kv_lens = q_lens, kv_lens
        self.layout_q = layout_q
        self.groups = heads // kv_heads
        self.prefix = [0]
        for q, kv in zip(q_lens, kv_lens):
            self.prefix.append(self.prefix[-1] + heads * q * ((kv + 15) // 16))

    def for_head(self, batch, head):
        def apply(weights, qs, qe, ks, ke):
            mask = self.mask(batch, head, qs, qe, ks, ke)
            return (
                weights.masked_fill(~torch.from_numpy(mask).to(weights.device), 0.0)
                / self.keep_prob
            )

        return apply

    def mask(self, batch, head, qs, qe, ks, ke):
        q_len, kv_len = self.q_lens[batch], self.kv_lens[batch]
        stride = (kv_len + 15) // 16
        row = np.arange(qs, qe, dtype=np.uint64)[:, None]
        if self.layout_q in ("BSND", "TND"):
            group_start = head // self.groups * self.groups * q_len
            row = np.uint64(group_start + head % self.groups) + row * np.uint64(
                self.groups
            )
        else:
            row += np.uint64(head * q_len)
        column = np.arange(ks // 16, (ke + 15) // 16, dtype=np.uint64)[None, :]
        relative = np.uint64(self.prefix[batch]) + row * np.uint64(stride) + column
        counter = np.uint64(self.base_counter) + relative
        carry = counter < np.uint64(self.base_counter)
        values = philox_bytes(counter, self.seed, carry)
        bits = (values <= self.threshold).reshape(qe - qs, -1)
        begin = ks % 16
        return bits[:, begin : begin + ke - ks]
