# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class LightningIndexerKLLossOpBuilder(OpBuilder):
    def __init__(self):
        super(LightningIndexerKLLossOpBuilder, self).__init__(
            "lightning_indexer_kl_loss", category="attention"
        )

    def sources(self):
        return ["csrc/attention/lightning_indexer_kl_loss.cpp"]

    def schema(self) -> str:
        return "lightning_indexer_kl_loss(Tensor target_score, Tensor index_probs, float eps=1e-9, str weight_type='logits') -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def lightning_indexer_kl_loss_meta(
            target_score, index_probs, eps=1e-9, weight_type="logits"
        ):
            return torch.empty((), dtype=target_score.dtype, device="meta")


lightning_indexer_kl_loss_op_builder = LightningIndexerKLLossOpBuilder()
lightning_indexer_kl_loss_op_builder._ensure_initialized()


@impl(get_as_library(), lightning_indexer_kl_loss_op_builder.name, "PrivateUse1")
def lightning_indexer_kl_loss(
    target_score, index_probs, eps=1e-9, weight_type="logits"
):
    """Lightning Indexer KL 散度损失函数。

    计算 teacher 侧原始未归一化主注意力分数（target_score）与 student 侧 indexer softmax
    后概率分布（index_probs）之间的 KL 散度。

    数值处理：
    - teacher 侧 y：clamp_min(norm_target, eps) 防止 y=0 处 log(0) 导致 0 * (-inf) = NaN
    - student 侧 Y：log(Y + eps) 保住 Y→0 处的梯度

    Args:
        target_score (Tensor): teacher 侧压缩段原始未归一化主注意力分数，sum != 1。
            支持 float16、bfloat16、float32。
        index_probs (Tensor): student 侧 indexer softmax 后的概率分布，shape 与 target_score 一致。
            支持 float16、bfloat16、float32。
        eps (float): 数值稳定常数，默认 1e-9。
        weight_type (str): 外层权重选择，'logits' 用原始 y（默认），'probs' 用归一化概率 p。

    Returns:
        Tensor: KL 散度标量损失，dtype 与输入一致。

    计算公式:
        P = y / sum(y, dim=-1, keepdim=True)
        log_P = log(clamp_min(P, eps))
        log_Y = log(Y + eps)
        if weight_type == 'logits':
            loss = sum((log_P - log_Y) * y)
        else:  # 'probs'
            loss = sum((log_P - log_Y) * P)

    调用示例:
        >>> import torch
        >>> import torch_npu
        >>> from cann_ops_transformer import lightning_indexer_kl_loss
        >>>
        >>> target_score = torch.randn(4, 10, 128, dtype=torch.float32).npu()
        >>> index_probs = torch.softmax(torch.randn(4, 10, 128), dim=-1).npu()
        >>> loss = lightning_indexer_kl_loss(target_score, index_probs)
    """
    op_module = lightning_indexer_kl_loss_op_builder.load()
    return op_module.lightning_indexer_kl_loss(
        target_score, index_probs, eps, weight_type
    )
