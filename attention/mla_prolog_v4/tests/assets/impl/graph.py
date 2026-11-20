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

"""使用公开算子入口的 TTK 图执行适配，静态缓冲区在捕获前准备。"""

import importlib
import os

import torch


def _adapter():
    """返回即时执行适配模块，按需加载被测算子。"""
    return importlib.import_module("mla_prolog_v4_ttk_ops")


class MlaPrologV4AclGraph(torch.nn.Module):
    """ACLGraph 编译目标，forward 仅调用可原地修改 Cache 的 MLA 算子。

    TTK 按参数名称分离静态权重、归一化权重和控制属性，将其传给 __init__；
    x、kv_cache、cache_index 及可选 scale/RoPE 张量传给 forward。
    必须保留全部控制属性，尤其不能将 PA_NZ 固定为 PA_BSND。
    """

    def __init__(
        self,
        wqa,
        wqb,
        wkva,
        wkb,
        norm_weight_qa,
        norm_weight_kva,
        *,
        norm_eps=1e-5,
        qkv_quant_mode=0,
        cache_mode="PA_BSND",
        qr_scale=1.0,
        kvr_scale=1.0,
        rope_mode=1,
        return_qr=False,
        weight_format=None,
        **kwargs,
    ):
        super().__init__()
        adapter = _adapter()
        weight_format = adapter._resolve_weight_format(
            (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
        )
        # 在 forward 解析算子前完成一次模块加载及 torch.library schema 注册。

        sample = adapter._load_under_test()
        # TTK 在设备绑定时依据 tensor_formats 完成 NZ 转换，捕获前仅保留已准备的权重缓冲区。

        self.register_buffer("wqa", wqa)
        self.register_buffer("wqb", wqb)
        self.register_buffer("wkva", wkva)
        self.register_buffer("wkb", wkb)
        self.register_buffer("norm_weight_qa", norm_weight_qa)
        self.register_buffer("norm_weight_kva", norm_weight_kva)
        # 在捕获前拒绝布尔模式和非有限标量，不执行静默强制转换。

        adapter._validate_unknown_attributes(kwargs)
        adapter._validate_weight_format(weight_format)
        (norm_eps, qr_scale, kvr_scale, qkv_mode, cache_mode, rope_mode) = (
            adapter._validate_attributes(
                norm_eps,
                qr_scale,
                kvr_scale,
                qkv_quant_mode,
                cache_mode,
                rope_mode,
                return_qr,
            )
        )
        self.norm_eps = norm_eps
        self.qkv_quant_mode = qkv_mode
        self.cache_mode = cache_mode
        self.qr_scale = qr_scale
        self.kvr_scale = kvr_scale
        self.rope_mode = rope_mode
        self.return_qr = return_qr
        self._mla_op = sample.mla_prolog_v2
        self.qkv_quant_mode = sample.mla_prolog_v2.QkvQuantMode(qkv_mode)
        self.cache_mode = sample.mla_prolog_v2.CacheMode(cache_mode)
        self.rope_mode = sample.mla_prolog_v2.RopeMode(rope_mode)

    def forward(
        self,
        x,
        kv_cache,
        cache_index,
        quant_scale_kv=None,
        rope_sin=None,
        rope_cos=None,
    ):
        q, descale_q, qr = self._mla_op(
            x,
            self.wqa,
            self.wqb,
            self.wkva,
            self.wkb,
            self.norm_weight_qa,
            self.norm_weight_kva,
            kv_cache,
            cache_index,
            quant_scale_kv=quant_scale_kv,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
            norm_eps=self.norm_eps,
            qkv_quant_mode=self.qkv_quant_mode,
            cache_mode=self.cache_mode,
            qr_scale=self.qr_scale,
            kvr_scale=self.kvr_scale,
            rope_mode=self.rope_mode,
            return_qr=self.return_qr,
        )
        # 返回与即时执行相同的四个槽位，Cache 按选定布局原地更新。

        return q, descale_q, qr, kv_cache
