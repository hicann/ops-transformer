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

import torch


def _adapter():
    """返回即时执行适配模块，按需加载被测算子。"""
    return importlib.import_module("quant_mla_ttk_ops")


class QuantMlaPrologAclGraph(torch.nn.Module):
    """ACLGraph 编译目标，forward 仅调用可原地修改 Cache 的 Quant MLA 算子。

    TTK 按参数名称将静态权重、MXFP8 权重 scale、归一化权重和控制属性传给 __init__；
    x、descale_x、kv_cache、cache_index 及可选 scale/RoPE 张量传给 forward。
    必须保留全部控制属性，尤其不能将 PA_NZ 固定为 PA_BSND。
    """

    def __init__(
        self,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        *,
        norm_eps=1e-5,
        w_qr_quant_mode=1,
        qkv_quant_mode=0,
        cache_mode=0,
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
        adapter._validate_unknown_attributes(kwargs)
        # 在构造图对象前按严格 bool 契约检查输入。
        adapter._validate_modes(
            qkv_quant_mode, w_qr_quant_mode, cache_mode, rope_mode, return_qr
        )
        # 在 forward 解析算子前完成一次模块加载及 torch.library schema 注册。

        sample = adapter._load_under_test()
        self.register_buffer("wqa", wqa)
        self.register_buffer("wqb", wqb)
        self.register_buffer("wkva", wkva)
        self.register_buffer("wkb", wkb)
        # CSV 已声明 NZ descale 布局，TTK 在捕获前完成 format-29 转换；此处只规范 dtype 视图，不重新打包。

        self.register_buffer("descale_wqa", adapter._as_e8m0(descale_wqa))
        self.register_buffer("descale_wqb", adapter._as_e8m0(descale_wqb))
        self.register_buffer("descale_wkva", adapter._as_e8m0(descale_wkva))
        self.register_buffer("norm_weight_qa", norm_weight_qa)
        self.register_buffer("norm_weight_kva", norm_weight_kva)
        self.norm_eps = norm_eps
        self.w_qr_quant_mode = adapter._mode_value(
            sample, "w_qr_quant_mode", w_qr_quant_mode
        )
        self.qkv_quant_mode = adapter._mode_value(
            sample, "qkv_quant_mode", qkv_quant_mode
        )
        self.cache_mode = adapter._mode_value(sample, "cache_mode", cache_mode)
        self.qr_scale = qr_scale
        self.kvr_scale = kvr_scale
        self.rope_mode = adapter._mode_value(sample, "rope_mode", rope_mode)
        self.return_qr = return_qr
        # descale 已具有最终格式，forward 内不再转换。
        self.weight_format = weight_format
        # 实际格式查询属于主机调用，只在捕获前校验一次，不进入被跟踪的 forward。

        adapter._validate_device_formats(
            (wqa, wqb, wkva),
            (descale_wqa, descale_wqb, descale_wkva),
            None,
            adapter._mode_int("w_qr_quant_mode", w_qr_quant_mode),
            weight_format,
        )
        self._sample = sample

    def forward(
        self,
        x,
        descale_x,
        kv_cache,
        cache_index,
        quant_scale_kv=None,
        rope_sin=None,
        rope_cos=None,
    ):
        q, descale_q, qr, descale_qr = _adapter()._invoke(
            self._sample,
            x,
            self.wqa,
            self.wqb,
            self.wkva,
            self.wkb,
            descale_x,
            self.descale_wqa,
            self.descale_wqb,
            self.descale_wkva,
            self.norm_weight_qa,
            self.norm_weight_kva,
            kv_cache,
            cache_index,
            quant_scale_kv,
            rope_sin,
            rope_cos,
            norm_eps=self.norm_eps,
            w_qr_quant_mode=self.w_qr_quant_mode,
            qkv_quant_mode=self.qkv_quant_mode,
            cache_mode=self.cache_mode,
            qr_scale=self.qr_scale,
            kvr_scale=self.kvr_scale,
            rope_mode=self.rope_mode,
            return_qr=self.return_qr,
            weight_format=self.weight_format,
            sync=False,
            check_formats=False,
        )
        # 返回与即时执行相同的五个槽位，Cache 按选定布局原地更新。

        return q, descale_q, qr, descale_qr, kv_cache
