#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
import random


def test_compressor_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu", weights_only=False)

    params = test_data["params"]
    cpu_result = test_data["cpu_result"]
    kv_mask_result = test_data["kv_mask_result"]
    update_kv = test_data["update_kv"]
    update_score = test_data["update_score"]
    cpu_kv_state = test_data["cpu_kv_state"]
    cpu_score_state = test_data["cpu_score_state"]

    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    x = test_data["x"]
    wkv = test_data["wkv"]
    wgate = test_data["wgate"]
    x_descale = test_data["x_descale"]
    wkv_descale = test_data["wkv_descale"]
    wgate_descale = test_data["wgate_descale"]
    kv_state = test_data["kv_state"]
    score_state = test_data["score_state"]
    ape = test_data["ape"]
    block_table = test_data["block_table"]
    cu_seqlens = test_data["cu_seqlens"]
    seqused = test_data["seqused"]
    start_pos = test_data["start_pos"]
    cmp_ratio = test_data["cmp_ratio"]
    coff = test_data["coff"]
    cache_mode = test_data["cache_mode"]
    quant_mode = test_data["quant_mode"]

    is_contiguous_flag = True
    if cache_mode == 1:  # 连续buffer(支持在第二维非连续)
        if is_contiguous_flag:
            state_cache = torch.zeros(
                (kv_state.shape[0], kv_state.shape[1], 2 * kv_state.shape[2])
            )
            state_cache[:, :, : state_cache.shape[2] // 2] = kv_state.clone()
            state_cache[:, :, state_cache.shape[2] // 2 :] = score_state.clone()
        else:
            layer_stride = random.randint(1, 10)
            print(f"layer_stride: {layer_stride}")
            state_cache_pad = torch.zeros(
                (
                    kv_state.shape[0],
                    (kv_state.shape[1] + layer_stride) * kv_state.shape[2] * 2,
                )
            )
            print(f"state_cache_pad: shape {state_cache_pad.shape}")
            # 使用 as_strided 创建非连续视图
            state_cache = torch.as_strided(
                state_cache_pad,
                size=(kv_state.shape[0], kv_state.shape[1], kv_state.shape[2] * 2),
                stride=(
                    (kv_state.shape[1] + layer_stride) * kv_state.shape[2] * 2,
                    kv_state.shape[2] * 2,
                    1,
                ),
            )
            # 填充数据
            state_cache[:, :, : kv_state.shape[2]] = kv_state.clone()
            state_cache[:, :, kv_state.shape[2] :] = score_state.clone()
            print(
                f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, "
                f"is_contiguous: {state_cache.is_contiguous()}, stride: {state_cache.stride()}"
            )
    else:
        layer_num = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_num - 1)
        print(f"layer_num: {layer_num}")
        print(f"layer_start_idx: {layer_start_idx}")
        state_cache_pad = torch.zeros(
            (kv_state.shape[0], layer_num * kv_state.shape[1] * kv_state.shape[2] * 2)
        )
        print(f"state_cache_pad: shape {state_cache_pad.shape}")
        state_cache = state_cache_pad[
            :,
            layer_start_idx * kv_state.shape[1] * kv_state.shape[2] * 2 : (
                layer_start_idx + 1
            )
            * kv_state.shape[1]
            * kv_state.shape[2]
            * 2,
        ].view(-1, kv_state.shape[1], kv_state.shape[2] * 2)
        state_cache[:, :, : state_cache.shape[2] // 2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2] // 2 :] = score_state.clone()
        print(
            f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride0: {state_cache.stride(0)}"
        )

    state_cache = state_cache.npu()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.npu()
    if seqused is not None:
        seqused = seqused.npu()
    if start_pos is not None:
        start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if block_table is not None:
        kv_block_table = block_table.npu()

    # 调用compressor算子
    npu_result = torch.ops.cann_ops_transformer.quant_compressor(
        x.npu(),
        wkv.npu(),
        wgate.npu(),
        state_cache,
        ape.npu(),
        x_descale=x_descale.npu(),
        wkv_descale=wkv_descale.npu(),
        wgate_descale=wgate_descale.npu(),
        state_block_table=kv_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        quant_mode=quant_mode,
        cmp_ratio=cmp_ratio,
        coff=coff,
        cache_mode=cache_mode,
    )
    return {
        "cpu_result": cpu_result,
        "kv_mask_result": kv_mask_result,
        "npu_result": npu_result,
        "cpu_kv_state": cpu_kv_state,
        "mask_cpu_kv_state": update_kv,
        "cpu_score_state": cpu_score_state,
        "mask_cpu_score_state": update_score,
        "state_cache": state_cache,
        "params": params,
    }
