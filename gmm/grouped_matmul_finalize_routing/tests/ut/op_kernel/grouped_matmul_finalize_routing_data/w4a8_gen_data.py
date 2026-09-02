# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch_npu
import sys
import tensorflow as tf
from scipy.special import softmax

bfloat16 = tf.bfloat16.as_numpy_dtype


def generate_data(groupNum, batch, topK, m, k, n):
    quantGroupSize = 256
    x = torch.randint(-5, 5, (m, k)).to(torch.int8)
    x_copy = x.clone()
    x2 = x.clone()
    weight = torch.randint(-5, 5, (topK, k, n), dtype=torch.int32)
    bias = torch.zeros((topK, n), dtype=torch.float32)
    scale_np = np.random.normal(0, 0.01, (topK, 1, n)).astype(np.float32)
    perGroupScale = np.ones([topK, k // quantGroupSize, n]).astype(np.float32)
    scaleUint32 = (scale_np * perGroupScale).astype(np.float16).astype(np.float32)

    scaleUint32.dtype = np.uint32
    scaleUint64 = np.zeros((topK, k // quantGroupSize, n * 2), dtype=np.uint32)
    scaleUint64[..., ::2] = scaleUint32
    scaleUint64.dtype = np.int64
    scale = torch.from_numpy(scaleUint64)
    offset = torch.randint(-5, 5, (topK, k // quantGroupSize, n)).to(torch.float32)
    groupList = torch.zeros((topK,), dtype=torch.int64).fill_(batch)
    perTokenScale = torch.zeros((m, 1), dtype=torch.float32).uniform_()
    perTokenScale_new = perTokenScale.reshape(m)

    shareInputOffset = batch // 2
    logits_ori = np.random.normal(0, 0.1, [batch, topK]).astype(np.float32)
    routing = np.argsort(logits_ori, axis=1, kind="stable")[:, -topK:].astype(np.int32)
    logits = softmax(
        logits_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing],
        axis=1,
    ).astype(np.float32)
    logits = logits.reshape(m)
    logits = np.ones(m).astype(np.float32)
    logits = torch.from_numpy(logits)

    sourceRow = (np.argsort(routing.reshape(-1), kind="stable") // topK).astype(
        np.int64
    )
    sourceRow = torch.from_numpy(sourceRow)
    residSacle = 0
    residual_ori = np.random.normal(0, 0.1, (max(batch // 4, 1), n)).astype(np.float32)
    residual = torch.from_numpy(residual_ori).to(torch.bfloat16)

    torch.save(x, "x.bin")
    torch.save(weight, "weight.bin")
    torch.save(groupList, "groupList.bin")
    torch.save(scale, "scale.bin")
    torch.save(bias, "bias.bin")
    torch.save(offset, "offset.bin")
    torch.save(perTokenScale_new, "perTokenScale.bin")
    torch.save(residual, "residual.bin")
    torch.save(logits, "logits.bin")
    torch.save(sourceRow, "sourceRow_int64.bin")

    y = np.zeros((m, n)).astype(np.float32)
    y = torch.from_numpy(y)
    torch.save(y, "y.bin")


if __name__ == "__main__":
    generate_data(8, 72, 8, 576, 2048, 7168)
