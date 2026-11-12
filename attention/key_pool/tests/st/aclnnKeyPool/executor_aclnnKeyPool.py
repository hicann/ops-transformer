#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
# KeyPool ATK ST 用例

`aclnnKeyPool/atk_aclnnKeyPool.json` 提供 24 条门槛用例，配套执行器为
`aclnnKeyPool/executor_aclnnKeyPool.py`。用例参考已验证的 KeyPool ATK 资产，
用于算子修改后的运行与精度回归。

## 覆盖范围

- FP16、BF16；BSH、TH 布局；单 Batch 与多 Batch。
- `D=128/512`，`H=1024/1536`。
- `cmpRatio=2/4/8/16/32/64/128`。
- LayerNorm 开启和关闭，非零 `startPos`、历史状态参与池化、跨 cache block。
- 完整组、尾部不足一组、单 token 和 BSH 空序列。

TH 前缀和与 cache block 映射由执行器根据用例 shape 确定，物理 block 0
保留为哨兵，每个 Batch 使用独立的物理 block。可选参数的零元素占位只在
这些用例中转换为空指针；LayerNorm 参数成对提供。

## 参考计算与比较

执行器按输入 dtype 对 K/Gate 投影和 softmax 权重做舍入，LayerNorm 的
中间计算保持 FP32，FP64 数学标杆保持 FP64。CPU 参考计算不修改初始 cache。

沿用参考资产的 `cv_fused_double_benchmark` 标准：最大相对误差比 5，
平均相对误差比和均方根误差比均为 1.5，需要使用支持该标准的 ATK 版本。
`pooledKeyOut` 仅比较接口定义的有效压缩结果，未初始化的容量行只在比较副本
中置零。执行器另外读取实际传入的 `stateCacheRef` 设备分配，检查尾组写回及
其余位置的保持情况；该检查不会向 ACLNN 接口追加输出参数。

## 运行

先加载 CANN 和 ATK 环境。节点配置需包含 `pyaclnn` accuracy 节点和 CPU
accuracy 标杆节点，并为前者指定空闲 NPU。测试自定义算子包时再加载该包的
环境；测试 CANN 自带算子时使用 CANN 的算子库。

在独立结果目录运行，避免覆盖已有报告：

```bash
atk task -n /path/to/nodes.yaml \
    -c /path/to/key_pool/tests/st/aclnnKeyPool/atk_aclnnKeyPool.json \
    -p /path/to/key_pool/tests/st/aclnnKeyPool/executor_aclnnKeyPool.py \
    -s 0 -e 24 -mt 1
```

检查逐例执行结果、精度结果及汇总中的通过数量；命令正常结束不代表精度通过。
"""

import ctypes

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


SIGNATURE = (
    "aclnnStatus aclnnKeyPoolGetWorkspaceSize("
    "const aclTensor *hiddenStates, const aclTensor *wk, const aclTensor *gateWeight, "
    "const aclTensor *ape, aclTensor *stateCacheRef, const aclTensor *cacheBlockTable, "
    "const aclTensor *startPos, const aclTensor *normWeightOptional, "
    "const aclTensor *normBiasOptional, const aclTensor *cosOptional, "
    "const aclTensor *sinOptional, const aclTensor *cuSeqlensOptional, "
    "const aclTensor *sequsedOptional, int64_t cmpRatio, double normEps, "
    "int64_t rotaryMode, int64_t stateCacheStrideDim0, aclTensor *pooledKeyOut, "
    "uint64_t *workspaceSize, aclOpExecutor **executor)"
)


def prepare_inputs(args):
    for name in (
        "normWeightOptional",
        "normBiasOptional",
        "cosOptional",
        "sinOptional",
        "cuSeqlensOptional",
        "sequsedOptional",
    ):
        if args[name] is not None and args[name].numel() == 0:
            args[name] = None
    table = args["cacheBlockTable"]
    # Physical block zero is a sentinel. Give every batch its own cache blocks.
    args["cacheBlockTable"] = torch.arange(
        1, table.numel() + 1, dtype=torch.int32, device=table.device
    ).reshape(table.shape)
    args["stateCacheStrideDim0"] = args["stateCacheRef"].stride(0)
    if args["hiddenStates"].ndim == 2:
        total = args["hiddenStates"].shape[0]
        batch = table.shape[0]
        args["cuSeqlensOptional"] = torch.tensor(
            [i * total // batch for i in range(batch + 1)],
            dtype=torch.int32,
            device=table.device,
        )


def golden(args, return_cache=False):
    """Project, optionally normalize, pool complete groups and update tail state."""
    hidden = args["hiddenStates"]
    compute_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    cache = args["stateCacheRef"].clone()
    batch = args["cacheBlockTable"].shape[0]
    ratio, d = args["cmpRatio"], args["wk"].shape[0]
    if hidden.ndim == 3:
        chunks = list(hidden.unbind(0))
        capacity = (hidden.shape[1] + ratio - 1) // ratio
        shape = (batch, capacity, d)
    else:
        ends = args["cuSeqlensOptional"].tolist()
        chunks = [hidden[ends[b] : ends[b + 1]] for b in range(batch)]
        capacity = min(hidden.shape[0], hidden.shape[0] // ratio + batch)
        shape = (capacity, d)
    pooled = torch.zeros(shape, dtype=hidden.dtype, device=hidden.device)
    output_offset = 0
    for b, chunk in enumerate(chunks):
        key = (
            (chunk.to(compute_dtype) @ args["wk"].to(compute_dtype).T)
            .to(hidden.dtype)
            .to(compute_dtype)
        )
        gate = (
            (chunk.to(compute_dtype) @ args["gateWeight"].to(compute_dtype).T)
            .to(hidden.dtype)
            .to(compute_dtype)
        )
        if args["normWeightOptional"] is not None:
            mean = key.mean(dim=-1, keepdim=True)
            variance = (key - mean).square().mean(dim=-1, keepdim=True)
            key = (key - mean) / torch.sqrt(variance + args["normEps"])
            key = key * args["normWeightOptional"] + args["normBiasOptional"]
        start = int(args["startPos"][b])
        end = start + chunk.shape[0]
        first = start // ratio
        count = end // ratio - first
        for group in range(first, end // ratio):
            keys, gates = [], []
            for r in range(ratio):
                position = group * ratio + r
                if position >= start:
                    keys.append(key[position - start])
                    gates.append(gate[position - start])
                else:
                    logical, offset = divmod(position, cache.shape[1])
                    physical = int(args["cacheBlockTable"][b, logical])
                    keys.append(cache[physical, offset, :d])
                    gates.append(cache[physical, offset, d:])
            probability = (torch.stack(gates) + args["ape"]).softmax(dim=0)
            probability = probability.to(hidden.dtype).to(compute_dtype)
            index = (
                (b, group - first)
                if hidden.ndim == 3
                else output_offset + group - first
            )
            pooled[index] = (
                (probability * torch.stack(keys)).sum(dim=0).to(hidden.dtype)
            )
        # Only the unfinished compression group is written back to stateCacheRef.
        for position in range(max(start, end - end % ratio), end):
            logical, offset = divmod(position, cache.shape[1])
            physical = int(args["cacheBlockTable"][b, logical])
            cache[physical, offset, :d] = key[position - start]
            cache[physical, offset, d:] = gate[position - start]
        output_offset += count
    return (pooled, cache) if return_cache else pooled


@register("key_pool_st")
class KeyPoolApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prepare_inputs(input_data.kwargs)

    def __call__(self, input_data: InputDataset, with_output=False):
        return golden(input_data.kwargs)

    def get_cpp_func_signature_type(self):
        return SIGNATURE


@register("key_pool_st_npu")
class KeyPoolAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prepare_inputs(input_data.kwargs)
        _, self.expected_cache = golden(input_data.kwargs, return_cache=True)
        self.expected_cache = self.expected_cache.cpu()
        self.input_dtype = input_data.kwargs["hiddenStates"].dtype
        args = input_data.kwargs
        hidden, ratio = args["hiddenStates"], args["cmpRatio"]
        if hidden.ndim == 3:
            lengths = [hidden.shape[1]] * hidden.shape[0]
            self.valid_mask = torch.zeros(
                (hidden.shape[0], (hidden.shape[1] + ratio - 1) // ratio),
                dtype=torch.bool,
            )
        else:
            lengths = torch.diff(args["cuSeqlensOptional"]).tolist()
            self.valid_mask = torch.zeros(
                min(hidden.shape[0], hidden.shape[0] // ratio + len(lengths)),
                dtype=torch.bool,
            )
        offset = 0
        for batch, length in enumerate(lengths):
            start = int(args["startPos"][batch])
            count = (start + length) // ratio - start // ratio
            if hidden.ndim == 3:
                self.valid_mask[batch, :count] = True
            else:
                self.valid_mask[offset : offset + count] = True
                offset += count
        input_args, output_packages = super().init_by_input_data(input_data)
        for index, name in enumerate(
            (
                "normWeightOptional",
                "normBiasOptional",
                "cosOptional",
                "sinOptional",
                "cuSeqlensOptional",
                "sequsedOptional",
            ),
            start=7,
        ):
            if input_data.kwargs[name] is None:
                input_args[index] = ctypes.cast(
                    ctypes.c_void_p(None), ctypes.POINTER(AclTensor)
                )
        # Keep the actual in-place device allocation alive for readback. It is an
        # input to GetWorkspaceSize, never an additional ACLNN output argument.
        self.device_cache = input_args[4]
        return input_args, output_packages

    def after_call(self, output_packages):
        actual_cache = self.acl_tensor_to_torch(self.device_cache).cpu()
        tolerance = 2e-3 if self.input_dtype == torch.float16 else 2e-2
        torch.testing.assert_close(
            actual_cache, self.expected_cache, rtol=tolerance, atol=tolerance
        )
        outputs = super().after_call(output_packages)
        # The interface leaves unused output capacity undefined. Mask only those
        # rows in a comparison copy; every completed group remains observable.
        pooled = outputs[0].cpu().clone()
        pooled.masked_fill_(~self.valid_mask[..., None], 0)
        outputs[0] = pooled
        return outputs

    def get_cpp_func_signature_type(self):
        return SIGNATURE
