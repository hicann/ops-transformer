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
import tensorflow as tf
import sys
from scipy.special import softmax

bfloat16 = tf.bfloat16.as_numpy_dtype


def generate_data(groupNum, batch, topK, m, k, n):
    groupList = np.array([batch] * groupNum, dtype=np.int64)
    x = np.random.randint(-10, 10, [m, k]).astype(np.int8)
    weight = np.random.randint(-10, 10, [groupNum, k, n]).astype(np.int8)
    scale = np.random.normal(0, 0.01, [groupNum, n]).astype(np.int8)
    perTokenScale = np.random.normal(0, 0.01, [m, 1]).astype(np.int8)

    offset = batch // 2
    logits_ori = np.random.normal(0, 0.1, [batch, groupNum]).astype(np.float32)
    routing = np.argsort(logits_ori, axis=1)[:, -topK:].astype(np.int32)
    logits = softmax(
        logits_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing],
        axis=1,
    ).astype(np.float32)
    logits = logits.reshape(m)
    sourceRow = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    sourceRow_int32 = (np.argsort(routing.reshape(-1)) // topK).astype(np.int32)
    residSacle = 1
    residual_ori = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    residual = residual_ori.astype(bfloat16)

    x.tofile("./x.bin")
    weight.tofile("./weight.bin")
    groupList.tofile("./groupList.bin")
    scale.tofile("./scale.bin")
    perTokenScale.tofile("./perTokenScale.bin")
    logits.tofile("./logits.bin")
    residual.tofile("./residual.bin")
    sourceRow.tofile("./sourceRow.bin")
    sourceRow_int32.tofile("./sourceRow_int32.bin")
    if n == 8192:
        y = np.zeros((m, n)).astype(np.float32)
    elif n == 7168:
        y = np.zeros((batch, n)).astype(np.float32)
    y.tofile("./y.bin")


if __name__ == "__main__":
    datalist = [[1, 1024, 1, 1024, 1024, 8192], [8, 768, 8, 6144, 2048, 7168]]
    arg = sys.argv[1]
    groupNum_in, batch_in, topK_in, m_in, k_in, n_in = datalist[int(arg)]
    generate_data(groupNum_in, batch_in, topK_in, m_in, k_in, n_in)
