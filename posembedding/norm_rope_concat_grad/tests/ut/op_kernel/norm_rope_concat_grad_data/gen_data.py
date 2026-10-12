#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import os
import numpy as np
import torch


def norm(x, mean, rstd, modeType, weight=None, bias=None, eps=1e-5):
    headdim = x.shape[-1]

    x = x.float()
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, correction=0, keepdim=True)
    rstd = 1 / torch.sqrt(var + eps)
    out = (x - mean) * rstd
    if modeType == 2:
        out = weight.float() * out + bias.float()
    return out


def image_rotary_emb(hidden_states, ropeSin, ropeCos, mode=1):
    out = torch.empty_like(hidden_states)
    if mode == 1:
        x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        rotated_x = torch.stack([-x2, x1], dim=-1).flatten(3)

        out = hidden_states.float() * ropeCos + rotated_x.float() * ropeSin
        return out.type_as(hidden_states)
    else:
        x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)

        rotated_x = torch.cat([-x2, x1], dim=-1)

        out = hidden_states.float() * ropeCos + rotated_x.float() * ropeSin
        return out.type_as(hidden_states)


def gen_data_and_golden(
    batchSize,
    headNum,
    headDim,
    totalQuerySeq,
    totalKeySeq,
    querySeq,
    keySeq,
    ropeSeq,
    origin_dtype,
    tilingkey,
):
    torch.manual_seed(42)
    batchSize = int(batchSize)
    headNum = int(headNum)
    headDim = int(headDim)
    totalQuerySeq = int(totalQuerySeq)
    totalKeySeq = int(totalKeySeq)
    querySeq = int(querySeq)
    keySeq = int(keySeq)
    ropeSeq = int(ropeSeq)
    tilingkey = int(tilingkey)

    print("batchSize=", batchSize)
    print("headNum=", headNum)
    print("headDim=", headDim)
    print("totalQuerySeq=", totalQuerySeq)
    print("totalKeySeq=", totalKeySeq)
    print("querySeq=", querySeq)
    print("keySeq=", keySeq)
    print("ropeSeq=", ropeSeq)
    print("origin_dtype=", origin_dtype)
    print("tilingkey=", tilingkey)

    # 生成入参数据
    if origin_dtype == "float32":
        origin_dtype = torch.float32
    elif origin_dtype == "float16":
        origin_dtype = torch.float16
    elif origin_dtype == "bfloat16":
        origin_dtype = torch.bfloat16
    else:
        raise ValueError(f"不存在该数据类型origin_dtype：{origin_dtype}")

    # 获取shape
    encoderQuerySeq = totalQuerySeq - querySeq
    encoderKeySeq = totalKeySeq - keySeq
    normType = tilingkey // 100000000
    normAddedType = (tilingkey % 100000000) // 100000
    ropeType = (tilingkey % 100000) // 100
    concatOrder = (tilingkey % 100) // 10
    eps = 1e-5

    gradQueryOutput = (
        torch.randn([batchSize, headNum, totalQuerySeq, headDim], dtype=origin_dtype)
        * 10
        - 5
    )
    gradKeyOutput = (
        torch.randn([batchSize, headNum, totalKeySeq, headDim], dtype=origin_dtype) * 10
        - 5
    )
    gradValueOutput = (
        torch.randn([batchSize, headNum, totalKeySeq, headDim], dtype=origin_dtype) * 10
        - 5
    )
    query = (
        torch.randn([batchSize, querySeq, headNum, headDim], dtype=origin_dtype) * 10
        - 5
    ).requires_grad_(True)
    key = (
        torch.randn([batchSize, keySeq, headNum, headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    value = (
        torch.randn([batchSize, keySeq, headNum, headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    encoderQuery = (
        torch.randn([batchSize, encoderQuerySeq, headNum, headDim], dtype=origin_dtype)
        * 10
        - 5
    ).requires_grad_(True)
    encoderKey = (
        torch.randn([batchSize, encoderKeySeq, headNum, headDim], dtype=origin_dtype)
        * 10
        - 5
    ).requires_grad_(True)
    encoderValue = (
        torch.randn([batchSize, encoderKeySeq, headNum, headDim], dtype=origin_dtype)
        * 10
        - 5
    ).requires_grad_(True)
    normQueryWeight = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    normQueryBias = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    normKeyWeight = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    normKeyBias = (torch.randn([headDim], dtype=origin_dtype) * 10 - 5).requires_grad_(
        True
    )
    normAddedQueryWeight = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    normAddedQueryBias = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    normAddedKeyWeight = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)
    normAddedKeyBias = (
        torch.randn([headDim], dtype=origin_dtype) * 10 - 5
    ).requires_grad_(True)

    normQueryMean = torch.mean(query.float(), dim=-1, keepdim=True)
    normQueryVar = torch.var(query.float(), correction=0, dim=-1, keepdim=True)
    normQueryRstd = torch.rsqrt(normQueryVar + eps).to(torch.float32)  # 倒数标准差

    normKeyMean = torch.mean(key.float(), dim=-1, keepdim=True)
    normKeyVar = torch.var(key.float(), correction=0, dim=-1, keepdim=True)
    normKeyRstd = torch.rsqrt(normKeyVar + eps).to(torch.float32)  # 倒数标准差

    normAddedQueryMean = torch.mean(encoderQuery.float(), dim=-1, keepdim=True)
    normAddedQueryVar = torch.var(
        encoderQuery.float(), correction=0, dim=-1, keepdim=True
    )
    normAddedQueryRstd = torch.rsqrt(normAddedQueryVar + eps).to(
        torch.float32
    )  # 倒数标准差

    normAddedKeyMean = torch.mean(encoderKey.float(), dim=-1, keepdim=True)
    normAddedKeyVar = torch.var(encoderKey.float(), correction=0, dim=-1, keepdim=True)
    normAddedKeyRstd = torch.rsqrt(normAddedKeyVar + eps).to(
        torch.float32
    )  # 倒数标准差

    positionIds = torch.arange(ropeSeq)
    invFreq = 1.0 / (10000 ** (torch.arange(0, headDim).float() / headDim))
    freqs = positionIds.unsqueeze(1) * invFreq.unsqueeze(0)
    ropeSin = torch.sin(freqs).to(origin_dtype)
    ropeCos = torch.cos(freqs).to(origin_dtype)

    # 保存入参到本地
    gradQueryOutput.detach().numpy().tofile("./gradQueryOutput.bin")
    gradKeyOutput.detach().numpy().tofile("./gradKeyOutput.bin")
    gradValueOutput.detach().numpy().tofile("./gradValueOutput.bin")
    query.detach().numpy().tofile("./query.bin")
    key.detach().numpy().tofile("./key.bin")
    encoderQuery.detach().numpy().tofile("./encoderQuery.bin")
    encoderKey.detach().numpy().tofile("./encoderKey.bin")
    normQueryWeight.detach().numpy().tofile("./normQueryWeight.bin")
    normQueryMean.detach().numpy().tofile("./normQueryMean.bin")
    normQueryRstd.detach().numpy().tofile("./normQueryRstd.bin")
    normKeyWeight.detach().numpy().tofile("./normKeyWeight.bin")
    normKeyMean.detach().numpy().tofile("./normKeyMean.bin")
    normKeyRstd.detach().numpy().tofile("./normKeyRstd.bin")
    normAddedQueryWeight.detach().numpy().tofile("./normAddedQueryWeight.bin")
    normAddedQueryMean.detach().numpy().tofile("./normAddedQueryMean.bin")
    normAddedQueryRstd.detach().numpy().tofile("./normAddedQueryRstd.bin")
    normAddedKeyWeight.detach().numpy().tofile("./normAddedKeyWeight.bin")
    normAddedKeyMean.detach().numpy().tofile("./normAddedKeyMean.bin")
    normAddedKeyRstd.detach().numpy().tofile("./normAddedKeyRstd.bin")
    ropeSin.detach().numpy().tofile("./ropeSin.bin")
    ropeCos.detach().numpy().tofile("./ropeCos.bin")

    # 开始计算
    if normType != 0:
        queryNorm = norm(
            query,
            normQueryMean,
            normQueryRstd,
            normType,
            normQueryWeight,
            normQueryBias,
            eps,
        )
        keyNorm = norm(
            key, normKeyMean, normKeyRstd, normType, normKeyWeight, normKeyBias, eps
        )
    else:
        queryNorm = query
        keyNorm = key

    if normAddedType != 0:
        encoderQueryNorm = norm(
            encoderQuery,
            normAddedQueryMean,
            normAddedQueryRstd,
            normAddedType,
            normAddedQueryWeight,
            normAddedQueryBias,
            eps,
        )
        encoderKeyNorm = norm(
            encoderKey,
            normAddedKeyMean,
            normAddedKeyRstd,
            normAddedType,
            normAddedKeyWeight,
            normAddedKeyBias,
            eps,
        )
    else:
        encoderQueryNorm = encoderQuery
        encoderKeyNorm = encoderKey

    if concatOrder == 0:
        queryConcat = torch.cat([queryNorm, encoderQueryNorm], dim=1)
        keyConcat = torch.cat([keyNorm, encoderKeyNorm], dim=1)
        valueConcat = torch.cat([value, encoderValue], dim=1)
    else:
        queryConcat = torch.cat([encoderQueryNorm, queryNorm], dim=1)
        keyConcat = torch.cat([encoderKeyNorm, keyNorm], dim=1)
        valueConcat = torch.cat([encoderValue, value], dim=1)

    queryConcat = torch.permute(queryConcat, (0, 2, 1, 3))
    keyConcat = torch.permute(keyConcat, (0, 2, 1, 3))
    valueConcat = torch.permute(valueConcat, (0, 2, 1, 3))

    if ropeType != 0:
        queryConcat[:, :, :ropeSeq] = image_rotary_emb(
            queryConcat[:, :, :ropeSeq], ropeSin, ropeCos, mode=ropeType
        )
        keyConcat[:, :, :ropeSeq] = image_rotary_emb(
            keyConcat[:, :, :ropeSeq], ropeSin, ropeCos, mode=ropeType
        )

    # 反向求导
    queryConcat.backward(gradQueryOutput, retain_graph=True)
    keyConcat.backward(gradKeyOutput, retain_graph=True)
    valueConcat.backward(gradValueOutput, retain_graph=True)

    # 获取梯度值
    gradQuery = query.grad.clone()
    gradKey = key.grad.clone()
    gradValue = value.grad.clone()
    gradencoderQuery = encoderQuery.grad.clone()
    gradEncoderKey = encoderKey.grad.clone()
    gradEncoderValue = encoderValue.grad.clone()

    if normType == 2:
        gradNormQueryWeight = normQueryWeight.grad.clone()
        gradNormQueryBias = normQueryBias.grad.clone()
        gradNormKeyWeight = normKeyWeight.grad.clone()
        gradNormKeyBias = normKeyBias.grad.clone()
    else:
        gradNormQueryWeight = torch.empty([]).zero_()
        gradNormQueryBias = torch.empty([]).zero_()
        gradNormKeyWeight = torch.empty([]).zero_()
        gradNormKeyBias = torch.empty([]).zero_()

    if normAddedType == 2:
        gradNormAddedQueryWeight = normAddedQueryWeight.grad.clone()
        gradNormAddedQueryBias = normAddedQueryBias.grad.clone()
        gradNormAddedKeyWeight = normAddedKeyWeight.grad.clone()
        gradNormAddedKeyBias = normAddedKeyBias.grad.clone()
    else:
        gradNormAddedQueryWeight = torch.empty([]).zero_()
        gradNormAddedQueryBias = torch.empty([]).zero_()
        gradNormAddedKeyWeight = torch.empty([]).zero_()
        gradNormAddedKeyBias = torch.empty([]).zero_()

    # 保存结果到本地
    gradQuery.to(origin_dtype).numpy().tofile("./golden_gradQuery.bin")
    gradKey.to(origin_dtype).numpy().tofile("./golden_gradKey.bin")
    gradValue.to(origin_dtype).numpy().tofile("./golden_gradValue.bin")
    gradencoderQuery.to(origin_dtype).numpy().tofile("./golden_gradEncoderQuery.bin")
    gradEncoderKey.to(origin_dtype).numpy().tofile("./golden_gradEncoderKey.bin")
    gradEncoderValue.to(origin_dtype).numpy().tofile("./golden_gradEncoderValue.bin")
    gradNormQueryWeight.to(origin_dtype).numpy().tofile(
        "./golden_gradNormQueryWeight.bin"
    )
    gradNormQueryBias.to(origin_dtype).numpy().tofile("./golden_gradNormQueryBias.bin")
    gradNormKeyWeight.to(origin_dtype).numpy().tofile("./golden_gradNormKeyWeight.bin")
    gradNormKeyBias.to(origin_dtype).numpy().tofile("./golden_gradNormKeyBias.bin")
    gradNormAddedQueryWeight.to(origin_dtype).numpy().tofile(
        "./golden_gradNormAddedQueryWeight.bin"
    )
    gradNormAddedQueryBias.to(origin_dtype).numpy().tofile(
        "./golden_gradNormAddedQueryBias.bin"
    )
    gradNormAddedKeyWeight.to(origin_dtype).numpy().tofile(
        "./golden_gradNormAddedKeyWeight.bin"
    )
    gradNormAddedKeyBias.to(origin_dtype).numpy().tofile(
        "./golden_gradNormAddedKeyBias.bin"
    )


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data_and_golden(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
        sys.argv[8],
        sys.argv[9],
        sys.argv[10],
    )
    exit(0)
