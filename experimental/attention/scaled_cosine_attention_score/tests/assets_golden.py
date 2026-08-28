# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NumPy golden for ScaledCosineAttentionScore."""

import numpy as np


def scaled_cosine_attention_score_golden(
    query, key, scale, clamp_max=4.6052, eps=1.0e-12
):
    query_fp32 = query.astype(np.float32)
    key_fp32 = key.astype(np.float32)
    query_inv = 1.0 / np.sqrt(
        np.sum(query_fp32 * query_fp32, axis=-1, keepdims=True) + eps
    )
    key_inv = 1.0 / np.sqrt(np.sum(key_fp32 * key_fp32, axis=-1, keepdims=True) + eps)
    cosine = np.matmul(query_fp32 * query_inv, np.swapaxes(key_fp32 * key_inv, -1, -2))
    per_head = np.exp(np.minimum(np.asarray(scale, np.float32).reshape(-1), clamp_max))
    return cosine * per_head.reshape(1, query.shape[1], 1, 1)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    q = rng.normal(size=(1, 2, 17, 20)).astype(np.float16)
    k = rng.normal(size=q.shape).astype(np.float16)
    s = rng.normal(size=(2, 1, 1)).astype(np.float32)
    y = scaled_cosine_attention_score_golden(q, k, s)
    assert y.shape == (1, 2, 17, 17)
    assert np.isfinite(y).all()
    print("golden self-check OK", y.shape)
