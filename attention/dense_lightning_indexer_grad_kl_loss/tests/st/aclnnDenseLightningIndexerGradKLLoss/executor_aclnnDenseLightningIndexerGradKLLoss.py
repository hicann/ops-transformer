#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
# DenseLightningIndexerGradKLLoss ATK ST 用例

`aclnnDenseLightningIndexerGradKLLoss/` 包含 16 条门槛用例和配套执行器。
参考既有 DenseLightningIndexerKLLossTrain 测试资产；其实际 ACLNN 接口为
`aclnnDenseLightningIndexerGradKLLoss`。

## 覆盖范围

- FP16、BF16；BSND、TND 布局；单 Batch 和多 Batch。
- Attention 头数 32/64/128，Indexer 头数 8/16/32/64。
- RoPE 成对提供或为空，`D=128`、`Dr=64`。
- 单 token、非对齐长度、不等长 TND Batch、跨 128/256 长度边界。
- `sparseMode=3` 的右下角因果掩码，Query 长度不超过对应 Key 长度。

这些用例保留参考资产的张量取值范围和 `single_bm/high_performance` 比较标准，
将原有 16K 至 1M 的性能规模缩小为门槛回归规模。最大 Query/Key 总长度为
257/513，不作为原有长序列性能用例的替代。

## 输入和参考计算

TND 的有效序列数组使用明确的累积终点，覆盖完整输入，不在执行时随机改写。
BSND 的序列数组在 ACLNN 调用时传类型匹配的空指针。RoPE 的零元素占位
只在这些用例中解释为空指针；有 RoPE 的用例必须成对提供。

执行器先从 Query、Key、Indexer 和 weights 计算满足因果掩码的 FP32
softmax max/sum，再将同一组统计量提供给 CPU 参考和 NPU 接口。CPU 参考
消费传入的统计量，按已验证参考的计算顺序计算 KL Loss 和三个梯度；最后
两个矩阵乘法的梯度输入先转换为输入 dtype，累加保持 FP32。

四个输出均参与精度比较，不能仅检查运行成功或 Loss。

## 运行

加载 CANN 和 ATK 环境，节点配置包含 `pyaclnn` accuracy 节点和 CPU
accuracy 标杆节点，为前者指定空闲 NPU。使用独立结果目录运行：

```bash
atk task -n /path/to/nodes.yaml \
    -c /path/to/dense_lightning_indexer_grad_kl_loss/tests/st/aclnnDenseLightningIndexerGradKLLoss/atk_aclnnDenseLightningIndexerGradKLLoss.json \
    -p /path/to/dense_lightning_indexer_grad_kl_loss/tests/st/aclnnDenseLightningIndexerGradKLLoss/executor_aclnnDenseLightningIndexerGradKLLoss.py \
    -s 0 -e 16 -mt 1
```

测试自定义算子包时需加载对应包的环境。检查逐例执行结果、四个输出的精度
详情和最终通过数量；命令正常结束不代表精度通过。
"""

import ctypes

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray, AclTensor


SIGNATURE = (
    "aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize("
    "const aclTensor *query, const aclTensor *key, const aclTensor *queryIndex, "
    "const aclTensor *keyIndex, const aclTensor *weights, const aclTensor *softmaxMax, "
    "const aclTensor *softmaxSum, const aclTensor *softmaxMaxIndex, "
    "const aclTensor *softmaxSumIndex, const aclTensor *queryRope, const aclTensor *keyRope, "
    "const aclIntArray *actualSeqLengthsQuery, const aclIntArray *actualSeqLengthsKey, "
    "double scaleValue, char *layout, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, "
    "const aclTensor *dQueryIndex, const aclTensor *dKeyIndex, const aclTensor *dWeights, "
    "const aclTensor *loss, uint64_t *workspaceSize, aclOpExecutor **executor)"
)


def batch_slices(args):
    """Yield batch views; TND arrays contain cumulative sequence ends."""
    if args["layout"] == "BSND":
        for batch in range(args["query"].shape[0]):
            yield batch, batch
    else:
        q_start = k_start = 0
        for q_end, k_end in zip(
            args["actualSeqLengthsQuery"], args["actualSeqLengthsKey"]
        ):
            yield slice(q_start, q_end), slice(k_start, k_end)
            q_start, k_start = q_end, k_end


def distributions(args, q_slice, k_slice):
    """Compute right-down causal attention target and indexer logits in FP32."""
    q, k = args["query"][q_slice].float(), args["key"][k_slice].float()
    if args["queryRope"] is not None:
        q = torch.cat((q, args["queryRope"][q_slice].float()), dim=-1)
        k = torch.cat((k, args["keyRope"][k_slice].float()), dim=-1)
    qi = args["queryIndex"][q_slice].float()
    ki = args["keyIndex"][k_slice][:, 0].float()
    w = args["weights"][q_slice].float()
    valid = torch.arange(k.shape[0], device=q.device)[None, :] <= (
        k.shape[0] - q.shape[0] + torch.arange(q.shape[0], device=q.device)[:, None]
    )
    attention = torch.einsum("snd,tnd->nst", q, k) * args["scaleValue"]
    attention.masked_fill_(~valid[None, :, :], -torch.inf)
    score = torch.einsum("snd,td->nst", qi, ki).relu()
    logits = torch.einsum("nst,sn->st", score, w)
    logits.masked_fill_(~valid, -torch.inf)
    return attention, logits, score


def prepare_inputs(args):
    # Zero-element placeholders in the corpus mean absent optional RoPE tensors.
    for name in ("queryRope", "keyRope"):
        if args[name] is not None and args[name].numel() == 0:
            args[name] = None
    maxima, sums, index_maxima, index_sums = [], [], [], []
    for q_slice, k_slice in batch_slices(args):
        attention, logits, _ = distributions(args, q_slice, k_slice)
        maximum = attention.amax(dim=-1)
        index_maximum = logits.amax(dim=-1)
        maxima.append(maximum[..., None])
        sums.append((attention - maximum[..., None]).exp().sum(dim=-1)[..., None])
        index_maxima.append(index_maximum[None, :])
        index_sums.append((logits - index_maximum[:, None]).exp().sum(dim=-1)[None, :])
    combine = (
        torch.stack if args["layout"] == "BSND" else lambda xs: torch.cat(xs, dim=1)
    )
    args["softmaxMax"] = combine(maxima).contiguous()
    args["softmaxSum"] = combine(sums).contiguous()
    args["softmaxMaxIndex"] = combine(index_maxima).contiguous()
    args["softmaxSumIndex"] = combine(index_sums).contiguous()


def golden(args):
    dq = torch.zeros_like(args["queryIndex"])
    dk = torch.zeros_like(args["keyIndex"])
    dw = torch.zeros_like(args["weights"])
    loss = torch.zeros(1, dtype=torch.float32, device=dq.device)
    for q_slice, k_slice in batch_slices(args):
        attention, logits, score = distributions(args, q_slice, k_slice)
        if args["layout"] == "BSND":
            maximum = args["softmaxMax"][q_slice, ..., 0]
            total = args["softmaxSum"][q_slice, ..., 0]
            index_maximum = args["softmaxMaxIndex"][q_slice, 0]
            index_total = args["softmaxSumIndex"][q_slice, 0]
        else:
            maximum = args["softmaxMax"][:, q_slice, 0]
            total = args["softmaxSum"][:, q_slice, 0]
            index_maximum = args["softmaxMaxIndex"][0, q_slice]
            index_total = args["softmaxSumIndex"][0, q_slice]
        target = ((attention - maximum[..., None]).exp() / total[..., None]).sum(dim=0)
        target *= 1.0 / attention.shape[0]
        index_probability = (logits - index_maximum[:, None]).exp() / index_total[
            :, None
        ]
        loss += (
            target
            * (target.clamp_min(1e-8).log() - index_probability.clamp_min(1e-8).log())
        ).sum()
        di = index_probability - target
        dw[q_slice] = torch.einsum("st,nst->sn", di, score).to(dw.dtype)
        # The Cube inputs for the final gradient matmuls use the input dtype.
        # Match the established reference's FP32 multiply order before the Cast.
        ds = (
            (score > 0).float()
            * args["weights"][q_slice].float().T[:, :, None]
            * di[None, :, :]
        ).to(dq.dtype)
        dq[q_slice] = torch.einsum(
            "nst,td->snd", ds.float(), args["keyIndex"][k_slice][:, 0].float()
        ).to(dq.dtype)
        dk[k_slice] = torch.einsum(
            "nst,snd->td", ds.float(), args["queryIndex"][q_slice].float()
        )[:, None, :].to(dk.dtype)
    return dq, dk, dw, loss


@register("dense_lightning_indexer_grad_kl_loss_st")
class DenseLightningIndexerGradKLLossApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prepare_inputs(input_data.kwargs)

    def __call__(self, input_data: InputDataset, with_output=False):
        return golden(input_data.kwargs)

    def get_cpp_func_signature_type(self):
        return SIGNATURE


@register("dense_lightning_indexer_grad_kl_loss_st_npu")
class DenseLightningIndexerGradKLLossAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        prepare_inputs(input_data.kwargs)
        input_args, output_packages = super().init_by_input_data(input_data)
        if input_data.kwargs["queryRope"] is None:
            for index in (9, 10):
                input_args[index] = ctypes.POINTER(AclTensor)()
        if input_data.kwargs["layout"] == "BSND":
            for index in (11, 12):
                input_args[index] = ctypes.POINTER(AclIntArray)()
        return input_args, output_packages

    def get_cpp_func_signature_type(self):
        return SIGNATURE
