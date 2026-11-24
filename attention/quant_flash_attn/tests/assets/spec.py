#!/usr/bin/python3
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

import importlib.util
import logging
import sys
from pathlib import Path

try:
    from cann_ops_transformer.ops import quant_flash_attn_metadata, quant_flash_attn
except Exception as e:  # noqa: BLE001 - XPU CPU 主机可能未装算子包
    logging.warning("Failed to import cann_ops_transformer.ops: %s", e)

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def load_impl_module(stem):
    """懒加载 impl 模块。

    与 flash_attn 资产一致: 不在 import spec.py 时 (主进程) 级联 import
    golden/graph (其中会 import torch_npu), 避免 fork 子进程 re-init NPU 报错。
    改为在 golden/customize_inputs/compare/npu_preprocess 首次被 ttk 调用时
    (fork 之后的 worker) 才加载。
    """
    # 模块名必须稳定(hash(path)每进程随机) — torch.compile/
    # npugraph_ex 的 dynamo 追踪需按名 import 回模块, 未注册时报
    # ModuleNotFoundError: qfa_assets_impl_graph_xxx
    mod_name = f"qfa_assets_impl_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _impl_cache[stem] = module
    return _impl_cache[stem]


# ==============================================================================
# quant_mode dispatcher
# TTK 的 Spec 类是静态绑定 (golden/customize_inputs 在类定义时即固定),
# 不能在运行时按 case 的 quant_mode 切换。因此用 dispatcher 函数包装:
#   - quant_mode=1 → MXFP8 路径 (cpu_qfa_mxfp8 / generate_qfa_mxfp8_inputs)
#   - quant_mode=6 → GQA FP8 路径 (cpu_qfa_gqa_fp8 / generate_qfa_gqa_fp8_inputs)
# Spec 类的 golden/customize_inputs 绑定 dispatcher, 运行时按 kwargs 派发。
# ==============================================================================


def _resolve_quant_mode(args, kwargs):
    """直调 op 后 quant_mode 是位置参数 (args[6])，不再出现在 kwargs。"""
    qm = args[6] if len(args) > 6 else kwargs.get("quant_mode", 1)
    return int(qm) if qm is not None else 1


def _golden_dispatch(*args, **kwargs):
    qm = _resolve_quant_mode(args, kwargs)
    if qm == 12:
        return load_impl_module("int8").cpu_golden(*args, **kwargs)
    if "int8" in _impl_cache:
        _impl_cache["int8"].clear_reference()
    golden_module = load_impl_module("golden")
    if qm == 6:
        return golden_module.cpu_qfa_gqa_fp8(*args, **kwargs)
    if qm == 0:
        return golden_module.cpu_qfa_hif8(*args, **kwargs)
    if qm == 10:
        return golden_module.cpu_qfa_hifp4(*args, **kwargs)
    return golden_module.cpu_qfa_mxfp8(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    qm = _resolve_quant_mode(args, kwargs)
    if qm == 12:
        return load_impl_module("int8").customize_inputs(*args, **kwargs)
    inputs_module = load_impl_module("inputs")
    if qm == 6:
        return inputs_module.generate_qfa_gqa_fp8_inputs(*args, **kwargs)
    if qm == 0:
        return inputs_module.generate_qfa_hif8_inputs(*args, **kwargs)
    if qm == 10:
        return inputs_module.generate_qfa_hifp4_inputs(*args, **kwargs)
    return inputs_module.generate_qfa_mxfp8_inputs(*args, **kwargs)


def _xpu_reference(*args, **kwargs):
    """XPU 三方参考: 固定分段量化 golden (原生 bf16 输出, 不升档).

    由远端 xpu_server 调用, 产出 cross_check 的 third_party. 必须与 NPU 的
    P 量化噪声同量级: 全局精确参考会让 |third_party - bench| 趋零, 比值恒爆.
    """
    return load_impl_module("golden").qfa_sectioned_reference(*args, **kwargs)


def _cross_check_active():
    """CLI 是否启用三方比对 (--compare cross_check)."""
    try:
        from ttk.switches import get_global_storage
    except Exception:  # pylint: disable=broad-except
        return False
    try:
        method = getattr(get_global_storage(), "compare_method", None)
    except Exception:  # pylint: disable=broad-except
        return False
    return method is not None and str(method).lower() == "cross_check"


class _CompareHook:
    """compare 钩子的动态解析.

    TTK 的自定义 compare 只收到 outputs/goldens, 拿不到 third_parties.
    三方比对时把 compare 解析成 None, 让框架退回标准 compare, 由
    CrossCheckComparison 用 golden(升档真值) + third_party(XPU) 判定;
    两方比对仍走 impl/compare.py.
    """

    def __init__(self, func):
        self._func = func

    def __get__(self, instance, owner):
        return None if _cross_check_active() else self._func


def _compare_dispatch(*outputs, compare_context=None, **kwargs):
    testcase_name = getattr(compare_context, "testcase_name", None)
    int8_impl = _impl_cache.get("int8")
    if int8_impl is not None and int8_impl.has_reference(testcase_name):
        return int8_impl.compare(*outputs, testcase_name=testcase_name)
    return load_impl_module("compare").compare(*outputs, **kwargs)


class QuantFlashAttnSpec:
    """quant_flash_attn 测试规范，按 quant_mode 派发。

    quant_mode=1: MXFP8 (Q/K per-token-group, V per-channel-group, descale=e8m0)
    quant_mode=6: GQA FP8 全量化 (Q/K per-token-head, V per-head, descale=FP32)
    quant_mode=0: HIF8 per-tensor 量化
    quant_mode=12: A2/A3 INT8 全量化，支持 BNSD/BSND/TND、SWA、LSE，metadata 为 None

    A5 模式的 metadata 由 npu_preprocess 在主算子调用前生成并回填,
    主算子直接由 ttk 调用 torch.ops.cann_ops_transformer.quant_flash_attn。

    golden/customize_inputs/compare/npu_preprocess 均懒加载 impl 模块,
    避免主进程 fork 前 import torch_npu (与 flash_attn 资产一致)。
    """

    golden = staticmethod(_golden_dispatch)
    customize_inputs = staticmethod(_inputs_dispatch)

    # 两方→impl/compare.py; 三方(--compare cross_check)→解析成 None, 走框架标准 compare.
    compare = _CompareHook(_compare_dispatch)

    # XPU 三方参考: 复用 CPU 分段量化实现, 输出原生精度(不升档).
    third_party = {"torch": _xpu_reference}

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return load_impl_module("npu_preprocess").run(*args, **kwargs)

    torch_graph = load_impl_module("graph").QuantFlashAttnAclGraph

    # 两方走 impl/compare.py 的 CSV 容差; 三方由框架原生比对, 档位取 level.
    # L2 = mare/mere/rmse 2.0/1.2/1.2 (与 quant_flash_mla 一致).
    tolerance = {
        "float16": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
            "level": "L2",
        },
        "bfloat16": {
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
            "level": "L2",
        },
        "float32": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
            "level": "L2",
        },
    }


class QuantFlashAttnMetadataSpec:
    """quant_flash_attn_metadata 生成器的 TestSpec。

    与 flash_attn 资产的 FlashAttnMetadataSpec 一致: 只提供 customized inputs
    (把 cu_seqlens/seqused 描述向量回填进 metadata API 输入), 无独立测试套件。
    """

    customize_inputs = load_impl_module(
        "metadata_inputs"
    ).generate_quant_flash_attn_metadata_inputs


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_attn": "QuantFlashAttnSpec",
    "torch.ops.cann_ops_transformer.quant_flash_attn_metadata": (
        "QuantFlashAttnMetadataSpec"
    ),
}
