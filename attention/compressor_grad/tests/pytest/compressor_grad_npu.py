#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""NPU 后端 — compressor 反向 (compressor_grad) 自动调用测试。

通过 torch.autograd 自动调用 compressor_grad:
1. 前向调用 compressor op，启用 requires_grad
2. 用上游梯度 d_cpm_kv 构造 loss
3. loss.backward() 自动触发 _compressor_backward → compressor_grad 算子
4. 收集 x / wkv / wgate / ape 的梯度输出

"""

import torch
from typing import Any, Dict, Optional

try:
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

from cann_ops_transformer.ops.compressor import _compressor_forward


def _valid_block_mask(cmp_kv, cu_seqlens, seqused, start_pos, cmp_ratio, seq_size=None):
    """有效压缩块 mask（与 totalValid 计算一致；padding 无效块置 False）。

    返回与 cmp_kv 同形状的 bool 张量：BSH (B, blocks, D) 每 batch 前 vb 块有效；
    TH (rows, D) 前 totalValid 行有效（块索引全局连续）。
    """
    d3 = cmp_kv.dim() == 3
    B = (
        cmp_kv.shape[0]
        if d3
        else (int(cu_seqlens.shape[0] - 1) if cu_seqlens is not None else 0)
    )
    valid = torch.zeros_like(cmp_kv, dtype=torch.bool)
    if d3:  # BSH: (B, blocks, D)
        for i in range(B):
            sp = int(start_pos[i]) if start_pos is not None else 0
            sq = int(seqused[i]) if seqused is not None else seq_size
            cmp_limit = (sp + sq) // cmp_ratio * cmp_ratio
            vb = (cmp_limit - sp + cmp_ratio - 1) // cmp_ratio if cmp_limit > sp else 0
            valid[i, :vb, :] = True
    else:  # TH: (rows, D)
        rows = cmp_kv.shape[0]
        total = 0
        for i in range(B):
            sp = int(start_pos[i]) if start_pos is not None else 0
            sq = (
                int(seqused[i])
                if seqused is not None
                else (
                    int(cu_seqlens[i + 1] - cu_seqlens[i])
                    if cu_seqlens is not None
                    else 0
                )
            )
            cmp_limit = (sp + sq) // cmp_ratio * cmp_ratio
            if cmp_limit > sp:
                total += (cmp_limit - sp + cmp_ratio - 1) // cmp_ratio
        valid[: min(total, rows), :] = True
    return valid


class NPUBackend:
    """compressor_grad NPU 后端 — 通过 autograd 自动调用反向算子。

    用法:

        backend = NPUBackend(device_id=0)
        grads = backend.compute(inputs)
        # grads["d_x"], grads["d_wkv"], grads["d_wgate"], grads["d_ape"]
    """

    name = "npu"

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._device = torch.device(f"npu:{device_id}")
        self._compressor_fn = None
        if _HAS_TORCH_NPU:
            torch.npu.set_device(device_id)

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_TORCH_NPU

    def clear_cache(self):
        if _HAS_TORCH_NPU:
            torch.npu.empty_cache()

    def compute(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """通过 compressor 前向 + autograd backward 调用 compressor_grad。

        通路说明:
        - torch.ops.cann_ops_transformer._compressor_forward 返回
          (cmp_kv, softmax_score, kv) 并注册了 autograd SetupContext。
        - backward 中调用 _compressor_backward → csrc CompressorBwd
          即 compressor_grad 算子，自动计算 d_x/d_wkv/d_wgate/d_ape。

        Args:
            inputs:
                x:             输入张量   [T, H] 或 [B, S, H]
                wkv:          KV 投影权重 (coff * D, H)
                wgate:          Gate 投影权重 (coff * D, H)
                state_cache:   state cache (block_num, block_size, 2*coff*D)
                ape:           位置编码   (cmp_ratio, coff * D)
                block_table:   cache 模式 1 的 block table
                cmp_ratio:     压缩率 (默认 4)
                coff:          重叠因子 (默认 1)
                cache_mode:    cache 模式 (默认 1)
                input_layout:  "BSH" 或 "TH"
                cu_seqlens:    TH 布局下的累积序列长度
                seqused:       每个 batch 的序列已用长度
                start_pos:     起始位置
                d_cpm_kv:     上游梯度 ∂Loss/∂cmp_kv — 用于 backward
        Returns:
            d_x:        x 的梯度 ∂Loss/∂x
            d_wkv:     wkv 的梯度 ∂Loss/∂wkv
            d_wgate:     wgate 的梯度 ∂Loss/∂wgate
            d_ape:      ape 的梯度 ∂Loss/∂ape
        """
        x = inputs["x"]
        wkv = inputs["wkv"]
        wgate = inputs["wgate"]
        state_cache = inputs["state_cache"]
        ape = inputs["ape"]
        d_cpm_kv = inputs["d_cpm_kv"]

        cmp_ratio = inputs.get("cmp_ratio", 4)
        coff = inputs.get("coff", 1)
        cache_mode = inputs.get("cache_mode", 1)
        layout = inputs.get("input_layout", "BSH")

        cu_seqlens = inputs.get("cu_seqlens")
        seqused = inputs.get("seqused")
        start_pos = inputs.get("start_pos")
        block_table = inputs.get("block_table")

        x = x.to(self._device).detach()
        wkv = wkv.to(self._device).detach()
        wgate = wgate.to(self._device).detach()
        state_cache = state_cache.to(self._device)
        ape = ape.to(self._device)
        d_cpm_kv = d_cpm_kv.to(self._device)

        cu_seqlens = cu_seqlens.to(self._device) if cu_seqlens is not None else None
        seqused = seqused.to(self._device) if seqused is not None else None
        start_pos = start_pos.to(self._device) if start_pos is not None else None
        block_table = block_table.to(self._device) if block_table is not None else None

        # 设置 requires_grad 以启用 autograd 反向传播
        x.requires_grad_(True)
        wkv.requires_grad_(True)
        wgate.requires_grad_(True)
        ape.requires_grad_(True)

        cmp_kv, softmax_score, kv, _ = _compressor_forward(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            state_block_table=block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            cmp_ratio=cmp_ratio,
            coff=coff,
            cache_mode=cache_mode,
            grad_enabled=True,
        )
        # backward 会释放 autograd 保存的中间量（sm/kv）→ 提前 clone 脱离图
        softmax_score_saved = softmax_score.detach().clone()
        kv_saved = kv.detach().clone()
        torch.npu.synchronize()

        # 用上游梯度 d_cpm_kv 构造 loss → backward 自动调用 compressor_grad
        # ⚠️ 只对有效压缩块计算：padding 无效块在 cmp_kv 中未写（可能垃圾/NaN），
        #    必须 mask 掉，否则其值污染 loss 与梯度（参考正向 cmp_mask 语义）
        loss = (
            cmp_kv
            * d_cpm_kv
            * _valid_block_mask(
                cmp_kv,
                cu_seqlens,
                seqused,
                start_pos,
                cmp_ratio,
                seq_size=x.shape[1] if x.dim() == 3 else None,
            )
        ).sum()
        loss.backward(retain_graph=True)
        torch.npu.synchronize()

        d_x = x.grad
        d_wkv = wkv.grad
        d_wgate = wgate.grad
        d_ape = ape.grad

        if d_x is None:
            d_x = torch.zeros_like(x)
        if d_wkv is None:
            d_wkv = torch.zeros_like(wkv)
        if d_wgate is None:
            d_wgate = torch.zeros_like(wgate)
        if d_ape is None:
            d_ape = torch.zeros(
                cmp_ratio, wkv.size(0), device=self._device, dtype=torch.float32
            )

        return {
            "d_x": d_x,
            "d_wkv": d_wkv,
            "d_wgate": d_wgate,
            "d_ape": d_ape,
            "cmp_kv": cmp_kv.detach(),
            "softmax_score": softmax_score_saved,
            "kv": kv_saved,
        }
